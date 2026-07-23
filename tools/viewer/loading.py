"""Data loading (numpy only — no viser). Cached per clip by the viewer.

Every loader here reads one clip's ``out/<stem>/`` tree into plain numpy: the
fused scene + human geometry, the per-frame cameras, the world-frame gravity, and
each SMPL-X body source (raw / kinematic / dynamics). :func:`load_clip_data`
orchestrates them into a :class:`ClipData`, which :mod:`tools.viewer.scene` then
turns into viser nodes. ``torch`` / ``viser.transforms`` are imported lazily
inside the loaders that need them, so importing this module stays light.
"""
from __future__ import annotations

import hashlib
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tools.geometry.surface import point_cloud_to_mesh

from .helpers import _camera_world_pose, _top4_skin_weights

_REPO_ROOT = Path(__file__).resolve().parents[2]

# SMPL-X hand joints — fingers, excluded from the body skeleton (wrists kept) and
# OR-folded into the wrist for contact. Shared with :mod:`tools.viewer.scene`.
_FINGER_TOKENS = ("index", "middle", "pinky", "ring", "thumb")
# Confidence quantile dropped from the displayed scene cloud, so what you see is the
# cloud the optimizer / estimate_contacts actually run their SDF against (mirrors
# ``[penetration]/[contacts].conf_drop_q`` in configs/human_optimization.toml).
_SCENE_CONF_DROP_Q = 0.1


def _load_gravity(pred: Path) -> dict | None:
    """World-frame gravity from ``geocalib/gravity.npz`` (unit down + source flag)."""
    f = pred / "geocalib" / "gravity.npz"
    if not f.exists():
        return None
    gd = np.load(f)
    g = np.asarray(gd["gravity_world"], np.float32)
    g = g / (float(np.linalg.norm(g)) + 1e-9)
    source = str(gd["source"]) if "source" in gd.files else "geocalib"
    spread = (float(gd["inlier_spread_median_deg"])
              if "inlier_spread_median_deg" in gd.files else float("nan"))
    return {"g": g, "is_fallback": source == "fallback_down", "spread": spread}


def _load_dynamics(pred: Path) -> dict | None:
    """Numpy arrays for the force-only dynamics skeleton (``human_optim/dynamics.npz``)."""
    f = pred / "human_optim" / "dynamics.npz"
    if not f.exists():
        return None
    dd = np.load(f, allow_pickle=True)
    tm = dd["torque_mag"].astype(np.float32)                    # (P, N, J)
    valid = dd["valid_mask"].astype(bool)                       # (P, N)
    tmv = tm[valid]
    # Torque → sphere-scale reference: 95th-pct magnitude over valid frames
    # (robust to spikes), shared across people for comparability.
    ref = float(np.percentile(tmv, 95)) if tmv.size else 1.0
    if ref <= 1e-6:
        ref = float(tmv.max()) if tmv.size and tmv.max() > 0 else 1.0
    names = [str(n) for n in dd["joint_names"]] if "joint_names" in dd.files else None
    jc = dd["joint_contact"].astype(bool) if "joint_contact" in dd.files else None
    if jc is not None and names is not None:
        # Fold each hand's finger contacts onto its wrist so the wrist sphere (the
        # hand's single force point) turns red whenever the hand grips — the contact
        # sits on the fingers/palm, so the wrist's own label alone misses most grips.
        for side in ("left", "right"):
            wrist = f"{side}_wrist"
            if wrist in names:
                w = names.index(wrist)
                fingers = [i for i, n in enumerate(names)
                           if n.startswith(f"{side}_") and any(t in n for t in _FINGER_TOKENS)]
                if fingers:
                    jc[:, :, w] |= jc[:, :, fingers].any(axis=2)
    return {
        "jw": dd["joints_world"].astype(np.float32),            # (P, N, J, 3)
        "tm": tm,
        "fw": dd["contact_forces_world"].astype(np.float32),    # (P, N, J, 3)
        "bw": (dd["base_wrench"].astype(np.float32)             # (P, N, 6): root force | torque
               if "base_wrench" in dd.files else None),
        "jc": jc,                                               # (P, N, J): per-joint in-contact (fingers folded to wrist)
        "joint_names": names,                                   # (J,) native names (finger toggle)
        "valid": valid,
        "parents": np.asarray(dd["parents"], np.int32),         # (J,)
        "oids": (np.asarray(dd["object_ids"], np.int32) if "object_ids" in dd.files
                 else np.arange(dd["joints_world"].shape[0], dtype=np.int32)),
        "ref": ref,
        "P": dd["joints_world"].shape[0],
    }


def _load_skeleton(pred: Path) -> dict | None:
    """Body-skeleton topology + per-joint contact, confidence, and depth fit.

    The joint tree (``parents`` / ``joint_names``, 52-joint ``SMPLXMid`` order) comes
    from ``human_optim/kinematic_1.npz`` so the skeleton can be drawn even without a
    contact pass; the per-joint contact flags — and the schema-5 per-joint
    ``label_confidence`` ∈ [0, 1] (both classes) that shades the sphere — come from
    ``human_optim/contacts_1.npz`` when present (same joint order). ``None`` when no
    kinematic body exists.
    """
    kin = pred / "human_optim" / "kinematic_1.npz"
    if not kin.exists():
        return None
    kd = np.load(kin, allow_pickle=True)
    out: dict = {
        "parents": np.asarray(kd["parents"], np.int32),          # (J,)
        "joint_names": [str(x) for x in kd["joint_names"]],      # (J,)
        "contact": None,                                         # (P, N, J) bool or None
        "confidence": None,                                      # (P, N, J) float or None
        "joint_depth_gap_cm": None,                              # (P, N, J) float or None
        "joint_depth_trust": None,                               # (P, N, J) float or None
        "contact_oids": None,                                    # (P,) or None
    }
    cf = pred / "human_optim" / "contacts_1.npz"
    if cf.exists():
        cd = np.load(cf, allow_pickle=True)
        expected = (
            len(np.asarray(kd["object_ids"])), int(np.asarray(kd["num_frames"]).item()),
            len(out["joint_names"]),
        )
        contact = np.asarray(cd["joint_contact"], bool)
        contact_names = ([str(x) for x in np.asarray(cd["joint_names"]).tolist()]
                         if "joint_names" in cd.files else [])
        contact_oids = (np.asarray(cd["object_ids"], np.int32)
                        if "object_ids" in cd.files else np.empty(0, np.int32))
        body_oids = np.asarray(kd["object_ids"], np.int32)
        if (contact.shape != expected or contact_names != out["joint_names"]
                or contact_oids.shape != (expected[0],)
                or not np.array_equal(contact_oids, body_oids)):
            warnings.warn(
                f"ignoring incompatible contact artifact {cf}; regenerate estimate_contacts",
                RuntimeWarning, stacklevel=2)
            return out
        contact = contact.copy()                              # folding below must not mutate the npz
        depth = {}
        for key in ("joint_depth_gap_cm", "joint_depth_trust"):
            if key not in cd.files:
                continue
            candidate = np.asarray(cd[key], np.float32)
            if candidate.shape == expected:
                depth[key] = candidate.copy()
            else:
                warnings.warn(
                    f"ignoring incompatible {key} in {cf}; regenerate estimate_contacts",
                    RuntimeWarning, stacklevel=2)
        conf = None
        try:
            schema = int(np.asarray(cd["confidence_schema"]).item())
            method = str(np.asarray(cd["confidence_method"]).item())
            candidate = np.asarray(cd["label_confidence"], np.float32)
            saved_body_hash = str(np.asarray(cd["source_body_sha256"]).item())
            with kin.open("rb") as body_file:
                current_body_hash = hashlib.file_digest(body_file, "sha256").hexdigest()
            valid_confidence = (
                schema == 5 and method == "depth-trust-motion-v1"
                and candidate.shape == expected and np.isfinite(candidate).all()
                and not bool(((candidate < 0.0) | (candidate > 1.0)).any())
                and saved_body_hash == current_body_hash
            )
        except (KeyError, TypeError, ValueError):
            valid_confidence = False
        if valid_confidence:
            conf = candidate.copy()
        else:
            warnings.warn(
                f"ignoring stale/invalid joint confidence in {cf}; regenerate estimate_contacts",
                RuntimeWarning, stacklevel=2)
        # OR-fold each hand's finger contact into its wrist: the skeleton draws only
        # the wrist (fingers are excluded), and a hand on the wall touches through the
        # fingers — the same "resolve hand contact at the wrist" convention the dynamics
        # stage uses. Fold the confidence to match the OR: when the folded label is
        # contact, take the strongest touching vote (max over in-contact members); when
        # free, take the weakest member because the free label is an AND across all
        # members. Depth fit takes the worst member: largest absolute gap and lowest
        # trust. Mutates the copies; the npz is untouched.
        names = out["joint_names"]
        for side in ("left", "right"):
            wrist = names.index(f"{side}_wrist")
            fingers = [j for j, nm in enumerate(names)
                       if nm.startswith(f"{side}_") and any(t in nm for t in _FINGER_TOKENS)]
            if not fingers:
                continue
            members = [wrist] + fingers                       # incl. the wrist's own state
            mcon = contact[..., members]                      # (P, N, M) — read before the wrist write
            folded = mcon.any(-1)                             # (P, N)
            if conf is not None:
                mconf = conf[..., members]                    # (P, N, M)
                c_hit = np.where(mcon, mconf, -np.inf).max(-1)  # strongest touching vote
                conf[..., wrist] = np.where(folded, c_hit, mconf.min(-1))
            gap = depth.get("joint_depth_gap_cm")
            if gap is not None:
                # worst member by magnitude, but keep its SIGNED value (sign
                # distinguishes cloud-beyond-mesh stretch from cloud-in-front).
                mgap = gap[..., members]
                mag = np.where(np.isfinite(mgap), np.abs(mgap), -np.inf)
                signed = np.take_along_axis(mgap, mag.argmax(-1)[..., None], -1)[..., 0]
                gap[..., wrist] = np.where(np.isfinite(mag.max(-1)), signed, np.nan)
            trust = depth.get("joint_depth_trust")
            if trust is not None:
                trust[..., wrist] = trust[..., members].min(-1)
            contact[..., wrist] = folded
        out["contact"] = contact
        out["confidence"] = conf
        out.update(depth)
        out["contact_oids"] = contact_oids
    return out


_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _load_video_frames(pred: Path, n_frames: int, *, video_dir: Path | None,
                       max_dim: int = 400) -> tuple[np.ndarray | None, float | None]:
    """Load the clip's original footage + its playback fps, aligned to the frame index.

    Finds ``<video_dir>/<clip-stem>.<ext>`` (``video_dir`` defaults to the repo's
    ``data/``) and decodes it the **same way the pipeline did** — cv2 sequential —
    so the RGB lines up with the geometry frame-for-frame. VGGT records the
    original index of each processed frame in ``vggt/camera.npz``
    (``frame_indices``); those pick the frames matching the ``n_frames`` extrinsics
    (identity unless a long clip was uniformly subsampled). Decoding with a
    different backend (e.g. decord) can report a different frame count on the same
    file, so cv2 is used here to stay byte-aligned with the pipeline. Frames are
    downscaled so the longer side is ``max_dim`` for a light sidebar pane.

    :param pred: the clip's ``out/<stem>/`` directory.
    :param n_frames: number of pipeline frames (extrinsics) to align to.
    :param video_dir: directory holding ``<stem>.<ext>``; ``None`` → repo ``data/``.
    :param max_dim: longer-side pixel cap for the returned frames.
    :returns: ``(frames, fps)`` — ``(n_frames, h, w, 3)`` ``uint8`` RGB indexed by the
        frame slider (black where a frame is missing) and the real-time playback fps
        (native container fps, scaled for any frame subsample); ``(None, None)`` if no
        source video is found.
    """
    import cv2

    vdir = video_dir if video_dir is not None else _REPO_ROOT / "data"
    video = next((vdir / f"{pred.name}{ext}" for ext in _VIDEO_EXTS
                  if (vdir / f"{pred.name}{ext}").exists()), None)
    if video is None:
        return None, None

    fi = None
    cam = pred / "vggt" / "camera.npz"
    if cam.exists():
        cd = np.load(cam)
        if "frame_indices" in cd.files:
            fi = np.asarray(cd["frame_indices"], np.int64)
    if fi is None:                                  # older clip → assume 1:1
        fi = np.arange(n_frames, dtype=np.int64)
    fi = fi[:n_frames]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None, None
    # Real-time playback default from the native container fps. When the pipeline
    # uniformly subsampled a long clip (non-identity frame_indices), the kept frames
    # span more source time than they number, so scale by that span to stay real-time
    # (identity → unchanged); None when cv2 can't report the fps.
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    span = float(fi[-1] - fi[0]) if len(fi) > 1 else 0.0
    play_fps = (raw_fps * (len(fi) - 1) / span if raw_fps > 0.0 and span > 0.0
                else (raw_fps or None))
    want = {int(i) for i in fi}
    kept: dict[int, np.ndarray] = {}
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in want:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            s = max_dim / max(h, w)
            if s < 1.0:
                rgb = cv2.resize(rgb, (round(w * s), round(h * s)),
                                 interpolation=cv2.INTER_AREA)
            kept[i] = rgb
        i += 1
    cap.release()
    if not kept:
        return None, None

    h, w = next(iter(kept.values())).shape[:2]
    out = np.zeros((n_frames, h, w, 3), np.uint8)
    for f, orig in enumerate(fi):
        frame = kept.get(int(orig))
        if frame is not None:
            out[f] = frame
    return out, play_fps


def _load_smplx_mid(device: str):
    """Load the 52-joint SMPL-X body + resolve the torch device (``mid, dev``).

    The body comes from the rewritten BetterHuman via :func:`tools.body.load_body`,
    which reproduces the old ``SMPLXMid`` surface the viewer uses (``forward_shape``,
    ``faces``, ``lbs_weights``, ``joint_names``, ``from_classic`` …).
    """
    import torch

    from tools.body import load_body

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    return load_body(device=dev), dev


def _smplx_kinematic_skinned(npz_path: Path, *, device: str = "cuda"):
    """Skinning data from a world-frame kinematic SMPL-X body (``human_optim/kinematic_*.npz``).

    The params are already world-frame (``frame='world'``) on the 52-joint
    ``SMPLXMid`` body, so there is no camera→world fold. Each person has a single
    ``betas`` identity, so one constant rest mesh + skin weights is exact and only
    the per-frame skeleton moves — viser does LBS in the browser.

    Returns ``(faces (F, 3), object_ids (P,), people)`` where ``people[p]`` is
    ``None`` (no valid frames) or a dict with ``v_shaped (V, 3)``, ``j_rest
    (J, 3)``, ``weights (V, J)``, per-frame ``bone_wxyz (N, J, 4)`` /
    ``bone_pos (N, J, 3)`` (NaN where invalid) and ``valid (N,)``.
    """
    import torch

    mid, dev = _load_smplx_mid(device)
    d = np.load(npz_path, allow_pickle=True)
    q_all = d["q"]                          # (P, N, 211)
    betas_all = d["betas"]                  # (P, 10)
    valid = d["valid_mask"].astype(bool)    # (P, N)
    P, N = q_all.shape[:2]
    scales = d["scale"] if "scale" in d.files else np.ones(P, np.float32)  # ≡1 (compat)
    object_ids = (np.asarray(d["object_ids"], np.int32)
                  if "object_ids" in d.files else np.arange(P, dtype=np.int32))
    faces = mid.faces.cpu().numpy().astype(np.int32)
    weights = _top4_skin_weights(mid.lbs_weights.cpu().numpy())
    J = weights.shape[1]

    people: list[dict | None] = []
    with torch.no_grad():
        for p in range(P):
            idx = np.flatnonzero(valid[p])
            if idx.size == 0:
                people.append(None)
                continue
            betas = torch.as_tensor(betas_all[p], dtype=torch.float32, device=dev)[None]
            v_shaped, j_rest = mid.forward_shape(betas)
            B = idx.size
            q = torch.as_tensor(q_all[p][idx], dtype=torch.float32, device=dev)
            jw = mid.forward(betas.expand(B, -1), q).joints_world.tensor()  # (B, 52, 7) world
            # Global scale about the pelvis (rest geometry about rest pelvis, world
            # bones about world pelvis) — no-op at s=1, kept for consumer compat.
            s = float(scales[p])
            if s != 1.0:
                root_rest = j_rest[:, :1, :].clone()
                v_shaped = root_rest + s * (v_shaped - root_rest)
                j_rest = root_rest + s * (j_rest - root_rest)
                root_w = jw[..., :1, :3]
                jw = torch.cat([root_w + s * (jw[..., :3] - root_w), jw[..., 3:]], dim=-1)
            quat = jw[..., 3:7].cpu().numpy()                              # xyzw
            bone_wxyz = np.full((N, J, 4), np.nan, np.float32)
            bone_pos = np.full((N, J, 3), np.nan, np.float32)
            bone_wxyz[idx] = quat[..., [3, 0, 1, 2]].astype(np.float32)    # → wxyz
            bone_pos[idx] = jw[..., :3].cpu().numpy().astype(np.float32)
            people.append({
                "v_shaped": v_shaped[0].cpu().numpy().astype(np.float32),
                "j_rest": j_rest[0].cpu().numpy().astype(np.float32),
                "weights": weights,
                "bone_wxyz": bone_wxyz,
                "bone_pos": bone_pos,
                "valid": valid[p],
            })
    return faces, object_ids, people


def _smplx_params_skinned(path: Path, extr: np.ndarray, *, device: str = "cuda"):
    """Skinning data for a camera-frame classic-SMPL-X param file.

    Used for both the raw ``sam3d/smplx_params.npz`` and the pre-fit
    ``geometry/prefit.npz`` — both store classic smplx-PyPI params in the VGGT
    camera frame with the same keys.

    Poses the **same 52-joint ``SMPLXMid`` model + FK** as the kinematic sources —
    no hand-rolled kinematics. Each person's classic smplx-PyPI params (camera
    frame) are converted per frame with :meth:`SMPLXMid.from_classic` (which adds
    the ``flat_hand_mean=False`` hand mean, so fingers are correct by construction)
    and posed with ``SMPLXMid.forward``. ``betas`` are averaged over each person's
    valid frames into **one identity**, so the rest mesh + skin weights are
    constant and only the per-frame skeleton moves (LBS in the browser, one
    upload). The classic root is camera-frame, so ``world_from_cam`` (the inverse
    VGGT extrinsic) is folded into each posed joint transform to reach the metric
    world frame.

    Same ``(faces, object_ids, people)`` contract as
    :func:`_smplx_kinematic_skinned`; ``None`` if no params exist.
    """
    if not path.exists():
        return None

    import torch
    import viser.transforms as vt

    mid, dev = _load_smplx_mid(device)
    sx = np.load(path)
    keys = ("betas", "transl", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose")
    params = {k: sx[k] for k in keys}
    if params["betas"].ndim == 2:  # promote legacy single-person (N, …) → (1, N, …)
        params = {k: v[None] for k, v in params.items()}
    valid = sx["valid_mask"]
    valid = (valid[None] if valid.ndim == 1 else valid).astype(bool)
    P, N = params["betas"].shape[:2]
    object_ids = (np.asarray(sx["object_ids"], np.int32)
                  if "object_ids" in sx.files else np.arange(P, dtype=np.int32))
    faces = mid.faces.cpu().numpy().astype(np.int32)
    weights = _top4_skin_weights(mid.lbs_weights.cpu().numpy())
    J = weights.shape[1]

    # world_from_cam per frame: T_wc = [[Rᵀ, −Rᵀt], [0, 1]].
    M = min(N, len(extr))
    Rcw = extr[:M, :3, :3].astype(np.float32)
    Twc = np.zeros((M, 4, 4), np.float32)
    Twc[:, :3, :3] = Rcw.transpose(0, 2, 1)
    Twc[:, :3, 3] = -np.einsum("nji,nj->ni", Rcw, extr[:M, :3, 3].astype(np.float32))
    Twc[:, 3, 3] = 1.0
    Twc_t = torch.from_numpy(Twc).to(dev)

    def to_t(a: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(dev)

    people: list[dict | None] = []
    with torch.no_grad():
        for p in range(P):
            idx = np.flatnonzero(valid[p, :M])
            if idx.size == 0:
                people.append(None)
                continue
            B = idx.size
            # One averaged identity; from_classic packs classic params → q (adds
            # the hand mean); forward gives camera-frame joint SE3s.
            betas = to_t(params["betas"][p][idx].mean(0, keepdims=True))       # (1, 10)
            v_shaped, j_rest = mid.forward_shape(betas)
            q = mid.from_classic(
                betas.expand(B, -1), to_t(params["body_pose"][p][idx]),
                to_t(params["transl"][p][idx]), to_t(params["global_orient"][p][idx]),
                to_t(params["left_hand_pose"][p][idx]), to_t(params["right_hand_pose"][p][idx]))
            Gw = Twc_t[idx][:, None] @ mid.forward(betas.expand(B, -1), q).joints_world.matrix()

            wxyz = vt.SO3.from_matrix(
                Gw[:, :, :3, :3].reshape(-1, 3, 3).cpu().numpy()).wxyz.reshape(B, J, 4)
            bone_wxyz = np.full((N, J, 4), np.nan, np.float32)
            bone_pos = np.full((N, J, 3), np.nan, np.float32)
            bone_wxyz[idx] = wxyz.astype(np.float32)
            bone_pos[idx] = Gw[:, :, :3, 3].cpu().numpy().astype(np.float32)
            vmask = np.zeros(N, bool)
            vmask[idx] = True
            people.append({
                "v_shaped": v_shaped[0].cpu().numpy().astype(np.float32),
                "j_rest": j_rest[0].cpu().numpy().astype(np.float32),
                "weights": weights,
                "bone_wxyz": bone_wxyz,
                "bone_pos": bone_pos,
                "valid": vmask,
            })
    return faces, object_ids, people


@dataclass
class ClipData:
    """Cached numpy for one clip (no viser handles)."""

    path: Path
    name: str
    scene_points: np.ndarray
    scene_colors: np.ndarray
    human_points: np.ndarray
    human_colors: np.ndarray
    human_fid: np.ndarray
    human_obj: np.ndarray
    obj_ids: list[int]
    extr: np.ndarray
    aspect: float
    fov_y: np.ndarray
    gravity: dict | None
    skinned: dict          # name -> (faces, object_ids, people); "original" first
    dynamics: dict | None
    skeleton: dict | None  # kinematic body-skeleton topology + per-joint contact
    n_frames: int
    # scene-surface meshes, built lazily and cached per method ("bpa"/"poisson");
    # a present key maps to (verts, faces, vertex_colors, vertex_normals) or None.
    scene_meshes: dict = field(default_factory=dict)
    scene_normals: np.ndarray | None = None   # camera-oriented normals (from fuse, else lazy)
    scene_voxel: float = 0.0                  # merge voxel size (m) — sizes the normal arrows
    video_frames: np.ndarray | None = None    # original footage, frame-slider aligned (sidebar pane)
    fps: float | None = None                  # real-time playback fps (from the source video)


def _subsample_human_mask(fid: np.ndarray, obj: np.ndarray, max_per_frame: int) -> np.ndarray:
    """Boolean keep-mask capping human points to ``max_per_frame`` per (frame, climber).

    The human depth cloud is dense (~27k pts/frame on some clips) and preloaded
    for every frame, so uploading it in full dominates clip-load time. A random
    per-(frame, climber) cap keeps the visible shape at a fraction of the bytes;
    seeded for a stable draw across reloads.
    """
    rng = np.random.default_rng(0)
    keep = np.ones(len(fid), bool)
    key = fid.astype(np.int64) * (int(obj.max()) + 1) + obj.astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    for s, e in zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [len(ks)]))):
        if e - s > max_per_frame:
            drop = rng.choice(order[s:e], (e - s) - max_per_frame, replace=False)
            keep[drop] = False
    return keep


def load_clip_data(pred: Path, *, device: str, show_smplx: bool,
                   human_max_points: int = 0, video_dir: Path | None = None) -> ClipData:
    """Load one clip's geometry + all SMPL-X body sources into numpy."""
    geo = pred / "geometry"
    scene = np.load(geo / "scene.npz")
    human = np.load(geo / "human.npz")
    transform = np.load(geo / "transform.npz")
    if "metric" in transform.files and not bool(transform["metric"]):
        print("[viewer] WARNING: geometry is UP-TO-SCALE (run estimate_scale) — "
              "metric bodies will not align with the scene")
    extr = transform["extrinsics"]

    # Show the same scene cloud the optimizer / estimate_contacts run against: drop the
    # lowest-confidence quantile of voxels, so a point you see is a point that exists
    # for contact (raw VGGT confidence has no fixed scale, hence a quantile gate).
    sp, sco = scene["points"], scene["colors"]
    sn = scene["normals"] if "normals" in scene.files else None   # per-point, from fuse
    if "confidence" in scene.files and _SCENE_CONF_DROP_Q > 0.0:
        conf = scene["confidence"]
        if len(conf) == len(sp) and len(conf):
            keep = conf >= np.quantile(conf, _SCENE_CONF_DROP_Q)
            sp, sco = sp[keep], sco[keep]
            sn = sn[keep] if sn is not None else None

    hp, hc = human["points"], human["colors"]
    hf, ho = human["frame_id"], human["obj_id"]
    if human_max_points and len(hp):
        keep = _subsample_human_mask(hf, ho, human_max_points)
        hp, hc, hf, ho = hp[keep], hc[keep], hf[keep], ho[keep]

    # Body sources, in display order — "original" (raw params, averaged identity)
    # first, then the refined kinematic fits; all skinned meshes (LBS in browser).
    skinned: dict = {}
    if show_smplx:
        if (pred / "sam3d" / "smplx_params.npz").exists():
            orig = _smplx_params_skinned(pred / "sam3d" / "smplx_params.npz", extr, device=device)
            if orig is not None:
                skinned["original"] = orig
        for name, fn in (("kinematics", "kinematic_1.npz"), ("kinematics_2", "kinematic_2.npz")):
            if (pred / "human_optim" / fn).exists():
                skinned[name] = _smplx_kinematic_skinned(pred / "human_optim" / fn, device=device)
        # prefit: the scale stage's dual scale+body alignment (camera frame, like original).
        if (pred / "geometry" / "prefit.npz").exists():
            pf = _smplx_params_skinned(pred / "geometry" / "prefit.npz", extr, device=device)
            if pf is not None:
                skinned["prefit"] = pf

    video_frames, play_fps = _load_video_frames(pred, len(extr), video_dir=video_dir)
    return ClipData(
        path=pred, name=pred.name,
        scene_points=sp, scene_colors=sco,
        human_points=hp, human_colors=hc, human_fid=hf, human_obj=ho,
        obj_ids=[int(o) for o in np.unique(ho)],
        extr=extr, aspect=float(transform["image_width"]) / float(transform["image_height"]),
        fov_y=np.radians(transform["fov_y_deg"].astype(np.float64)),
        gravity=_load_gravity(pred),
        skinned=skinned,
        dynamics=_load_dynamics(pred),
        skeleton=_load_skeleton(pred) if show_smplx else None,
        n_frames=len(extr),
        scene_normals=sn,
        scene_voxel=float(scene["voxel_size"]) if "voxel_size" in scene.files else 0.0,
        video_frames=video_frames, fps=play_fps)


def ensure_scene_mesh(data: ClipData, method: str) -> tuple | None:
    """Build (once per ``method``, cached) a scene-cloud surface — ``bpa`` or ``poisson``.

    Deferred until a surface is first shown so a points-only session pays nothing.
    Feeds :func:`point_cloud_to_mesh` the camera-oriented normals from
    :func:`tools.geometry.normals.estimate_point_normals` (the cameras are the
    clip's own VGGT extrinsics), computed once and reused across methods.
    """
    if data.scene_normals is None:
        import torch

        from tools.geometry.normals import estimate_point_normals
        centers = np.stack([_camera_world_pose(e)[0] for e in data.extr]).astype(np.float64)
        data.scene_normals = estimate_point_normals(
            torch.from_numpy(data.scene_points.astype(np.float64)),
            torch.from_numpy(centers)).cpu().numpy()
    if method not in data.scene_meshes:
        data.scene_meshes[method] = point_cloud_to_mesh(
            data.scene_points, data.scene_colors, method=method, normals=data.scene_normals)
    return data.scene_meshes[method]
