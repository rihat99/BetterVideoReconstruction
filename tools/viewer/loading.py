"""Data loading (numpy only — no viser). Cached per clip by the viewer.

Every loader here reads one clip's ``out/<stem>/`` tree into plain numpy: the
fused scene + human geometry, the per-frame cameras, the world-frame gravity, and
each SMPL-X body source (raw / kinematic / kindyn). :func:`load_clip_data`
orchestrates them into a :class:`ClipData`, which :mod:`tools.viewer.scene` then
turns into viser nodes. ``torch`` / ``viser.transforms`` are imported lazily
inside the loaders that need them, so importing this module stays light.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tools.geometry.surface import point_cloud_to_mesh

from .helpers import _camera_world_pose, _top4_skin_weights

_REPO_ROOT = Path(__file__).resolve().parents[2]

# SMPL-X hand joints — the fingers. The body skeleton draws them small (see
# ``_SKEL_FINGER_SCALE`` in :mod:`tools.viewer.scene`) and the force sources sum
# their per-joint forces into the wrist. Shared with :mod:`tools.viewer.scene`.
_FINGER_TOKENS = ("index", "middle", "pinky", "ring", "thumb")
# The contact-label schema this viewer reads: contact lives on the 12 body-mounted
# contact frames (``frame_contact``), not on joints. An older artifact is ignored.
_CONTACT_SCHEMA = 2
# Confidence quantile dropped from the displayed scene cloud, so what you see is the
# cloud the optimizer / estimate_contacts actually run their SDF against (mirrors
# ``[penetration]/[contacts].conf_drop_q`` in configs/human_optimization.toml).
_SCENE_CONF_DROP_Q = 0.1
# The MHR body the ``mhr_1_sup`` source poses: BetterHuman's converted LOD1 archive
# (18 439 vertices, 127 bones) — the LOD the corpus MHR fit was made on. Given
# explicitly, like the SMPL-X path in ``tools/body.py``, so no env var is needed.
_MHR_MODEL, _MHR_LOD = Path(
    "/home/rikhat.akizhanov/better/BetterHuman/models/MHR/converted/mhr_lod1.npz"), 1
# Loaded MHR bodies by device — reading the archive costs ~4 s, and every clip with
# an ``mhr_1.npz`` needs the same one.
_MHR_CACHE: dict = {}


def _load_gravity(pred: Path) -> dict | None:
    """World-frame gravity from ``geocalib/gravity.npz`` (unit down + source flag).

    When the gravity stage fitted and accepted a ground plane, ``ground`` carries
    it (centroid / up-normal / draw extent / angle to the prior down) so the
    viewer can draw the grid the pipeline actually used; a rejected fit leaves
    ``ground`` ``None`` and puts the reason in ``ground_reject``.
    """
    f = pred / "geocalib" / "gravity.npz"
    if not f.exists():
        return None
    gd = np.load(f)
    g = np.asarray(gd["gravity_world"], np.float32)
    g = g / (float(np.linalg.norm(g)) + 1e-9)
    source = str(gd["source"]) if "source" in gd.files else "geocalib"
    spread = (float(gd["inlier_spread_median_deg"])
              if "inlier_spread_median_deg" in gd.files else float("nan"))
    ground = None
    if "ground_accepted" in gd.files and bool(gd["ground_accepted"]):
        ground = {"centroid": np.asarray(gd["ground_centroid"], np.float64),
                  "normal": np.asarray(gd["ground_normal"], np.float64),
                  "extent": float(gd["ground_extent"]),
                  "angle_deg": float(gd["ground_angle_deg"])}
    reject = str(gd["ground_reject"]) if "ground_reject" in gd.files else ""
    return {"g": g, "is_fallback": source == "fallback_down", "source": source,
            "spread": spread, "ground": ground, "ground_reject": reject}


def _load_force_data(pred: Path, filename: str) -> dict | None:
    """Numpy arrays for a force/torque body source (``human_optim/<filename>``).

    Shared by the kinematics+dynamics bodies ``kindyn_1.npz`` / ``kindyn_manual.npz``
    (the manual-annotation reference chain). Both store the same force fields, so the
    viewer draws them the same way. Hand forces are folded to the wrists here, once,
    so the per-joint display never draws arrows on individual finger joints.
    """
    f = pred / "human_optim" / filename
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
    fw = dd["contact_forces_world"].astype(np.float32)          # (P, N, J, 3)
    if names is not None:
        # Sum each hand's finger forces onto its wrist (the joint we call "the hand")
        # and zero the finger columns, so the per-joint display draws ONE arrow per
        # gripping hand instead of one per loaded finger joint.
        for side in ("left", "right"):
            wrist = f"{side}_wrist"
            if wrist in names:
                w = names.index(wrist)
                fingers = [i for i, n in enumerate(names)
                           if n.startswith(f"{side}_") and any(t in n for t in _FINGER_TOKENS)]
                if fingers:
                    fw[:, :, w] += fw[:, :, fingers].sum(axis=2)
                    fw[:, :, fingers] = 0.0
    return {
        "jw": dd["joints_world"].astype(np.float32),            # (P, N, J, 3)
        "tm": tm,
        "fw": fw,                                               # (P, N, J, 3), hands at wrists
        "ff": (dd["frame_forces"].astype(np.float32)            # (P, N, F, 3): the same forces
               if "frame_forces" in dd.files else None),        # per contact FRAME, unfolded
        "ff_names": ([str(x) for x in dd["contact_frame_names"]]  # (F,) which frames those are
                     if "contact_frame_names" in dd.files else None),
        "bw": (dd["base_wrench"].astype(np.float32)             # (P, N, 6): root force | torque
               if "base_wrench" in dd.files else None),         #   in the ROOT-LOCAL frame
        "rootq": (dd["q"][:, :, 3:7].astype(np.float32)         # (P, N, 4) scalar-last root quat
                  if "q" in dd.files else None),                #   (rotates bw to world for drawing)
        "joint_names": names,                                   # (J,) native names (finger toggle)
        "valid": valid,
        "parents": np.asarray(dd["parents"], np.int32),         # (J,)
        "oids": (np.asarray(dd["object_ids"], np.int32) if "object_ids" in dd.files
                 else np.arange(dd["joints_world"].shape[0], dtype=np.int32)),
        "ref": ref,
        "P": dd["joints_world"].shape[0],
    }


def _load_sensor_forces(pred: Path) -> dict | None:
    """Measured wall-sensor forces (``sensor_forces.npz``), if present.

    Written by ``scripts/peter_sensor_forces.py`` for the climb_wall clips: the
    rig's four load-cell channels rotated into the world frame, in the reaction
    sense (force on the climber, directly comparable to the predicted contact
    forces). The cells support ONE shared structure, so a single channel is that
    corner's reaction — limb-dominated but mixed across limbs (only the channel
    sum is a clean physical force; the npz stores it as ``total_world``). Drawn
    by the ``sensor forces`` toggle in the dynamics folder.

    The climb_wall_3 ground-truth clips (``scripts/peter_gt``) keep the file at the
    clip root instead, where it is a per-limb board reading rather than a corner
    reaction; both live in the same world frame, so either location is read here.

    Force plates (``scripts/opencap_sensor_forces.py``) also store where each force
    acts — the centre of pressure on the floor, ``cop_world`` — so their arrows are
    drawn from that point instead of the limb joint (NaN where the plate saw nothing).
    """
    f = next((c for c in (pred / "human_optim" / "sensor_forces.npz",
                          pred / "sensor_forces.npz") if c.exists()), None)
    if f is None:
        return None
    dd = np.load(f, allow_pickle=True)
    return {
        "rw": dd["reaction_world"].astype(np.float32),   # (N, 4, 3) per channel
        "limbs": [str(x) for x in dd["limbs"]],          # channel -> limb name
        "cop": (dd["cop_world"].astype(np.float32)       # (N, 4, 3) centre of pressure,
                if "cop_world" in dd.files else None),   # force plates only
    }


def _load_plate_estimate(pred: Path) -> dict | None:
    """Our solved foot forces in the force plates' own format (``estimated_forces.npz``).

    Written by ``scripts/export_opencap_mot.py`` next to the plates' ``sensor_forces.npz``:
    each foot's solved contact forces collapsed to one force acting at one centre of
    pressure on the floor — the same representation the plates report — in the
    dataset's OpenSim ground frame. Rotated back into our world here with the ``R``/``t``
    the file carries, so the viewer draws them beside the measured plate arrows.
    """
    f = pred / "human_optim" / "estimated_forces.npz"
    if not f.exists():
        return None
    dd = np.load(f, allow_pickle=True)
    R = np.asarray(dd["R_world_from_mocap"], np.float64)
    t = np.asarray(dd["t_world_from_mocap"], np.float64)
    return {
        "rw": (dd["force"].astype(np.float64) @ R.T).astype(np.float32),        # (N, 2, 3)
        "cop": (dd["cop"].astype(np.float64) @ R.T + t).astype(np.float32),     # (N, 2, 3)
        "limbs": [str(x) for x in dd["limbs"]],
    }


def _load_view_cameras(pred: Path) -> dict | None:
    """The calibrated static cameras of a multi-view clip (``cameras.npz``).

    Written by the climb_wall_3 ground-truth chain (``scripts/peter_gt``): three
    fixed, calibrated cameras watching one climber, already in the metric world the
    geometry uses. ``offset`` is how far each view lags ``cam_left``, whose clock the
    frame slider follows — view ``k``'s video frame ``t + offset[k]`` is the same
    instant as frame ``t``. ``None`` for an ordinary single-camera clip.
    """
    f = pred / "cameras.npz"
    if not f.exists():
        return None
    cd = np.load(f, allow_pickle=True)
    K = np.asarray(cd["K"], np.float64)                     # (V, 3, 3) pixel intrinsics
    wh = np.asarray(cd["image_size"], np.int64)             # (V, 2) width, height
    return {
        "names": [str(x) for x in cd["view_names"]],
        "extr": np.asarray(cd["extrinsics"], np.float64),   # (V, 4, 4) cam-from-world
        "fov_y": 2.0 * np.arctan(wh[:, 1] / (2.0 * K[:, 1, 1])),
        "aspect": wh[:, 0] / wh[:, 1],
        "offset": (np.asarray(cd["frame_offset"], np.int64) if "frame_offset" in cd.files
                   else np.zeros(len(wh), np.int64)),
    }


def _load_view_videos(pred: Path, n_frames: int, views: dict,
                      max_dim: int = 400) -> dict[str, np.ndarray] | None:
    """Each calibrated view's footage (``views/<name>/video.mp4``), on the frame clock.

    Decoded with cv2 like the single-camera pane (decoder backends disagree on frame
    counts, and cv2 is what the pipeline used) and shifted by that view's
    ``frame_offset``, so index ``t`` is the same instant in every view and matches the
    frame slider. Frames are downscaled to a longer side of ``max_dim`` for a light
    sidebar pane; ``None`` when the clip has no per-view videos.
    """
    import cv2

    out: dict[str, np.ndarray] = {}
    for k, name in enumerate(views["names"]):
        path = pred / "views" / name / "video.mp4"
        if not path.exists():
            continue
        off = int(views["offset"][k])
        cap = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        for i in range(off + n_frames):
            ok, bgr = cap.read()
            if not ok:
                break
            if i < off:                          # this view's lag: skip into its clock
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            s = max_dim / max(h, w)
            if s < 1.0:
                rgb = cv2.resize(rgb, (round(w * s), round(h * s)),
                                 interpolation=cv2.INTER_AREA)
            frames.append(rgb)
        cap.release()
        if frames:
            buf = np.zeros((n_frames, *frames[0].shape), np.uint8)   # black past the end
            buf[:len(frames)] = frames
            out[name] = buf
    return out or None


def _expand_frame_columns(arr: np.ndarray, names: list[str],
                          body_names: list[str]) -> np.ndarray | None:
    """Scatter per-contact-frame columns ``arr (P, N, F_sub, …)`` into body order.

    Stage configs may restrict detection to a subset of the body's contact
    frames (``[contacts].frames`` — the peter rigs run hands+feet only); the
    viewer poses the FULL set, so a subset artifact is expanded here by name,
    uncovered frames zero/False. ``None`` if a name is unknown to the body.
    """
    pos = {n: i for i, n in enumerate(body_names)}
    if any(n not in pos for n in names):
        return None
    full = np.zeros(arr.shape[:2] + (len(body_names),) + arr.shape[3:], arr.dtype)
    full[:, :, [pos[n] for n in names]] = arr
    return full


def _load_skeleton(pred: Path, body_fn: str = "kinematic_1.npz",
                   contacts_fn: str | None = "contacts_1.npz",
                   frame_names: list[str] | None = None,
                   frame_parents: np.ndarray | None = None) -> dict | None:
    """Body-skeleton topology + per-contact-frame contact labels.

    The joint tree (``parents`` / ``joint_names``, 52 native joints) comes from
    ``human_optim/<body_fn>``, so the skeleton draws even without a contact pass.
    Contact itself lives on the **contact frames** — named points on the skin,
    each rigidly attached to a joint — and is read from ``human_optim/<contacts_fn>``
    when that file carries the current label schema
    (``contact_label_schema == _CONTACT_SCHEMA``). Each refined body pairs with the
    contact artifact measured ON it (kinematic_1 ↔ contacts_1, kinematic_2 ↔
    contacts_2). ``contacts_fn=None`` means the body npz is **self-labeled** — the
    kindyn stages write ``frame_contact`` next to ``q`` themselves — so the labels
    come straight from ``body_fn`` with no schema check.

    A missing, older, or mismatched contact file just leaves the contact fields
    ``None``: the body and its skeleton still load, they only lose the contact
    layer. ``None`` when no such body exists.

    :param pred: the clip's ``out/<stem>/`` directory.
    :param body_fn: refined-body npz under ``human_optim/``.
    :param contacts_fn: contact npz measured on that body, or ``None`` (self-labeled).
    :param frame_names: the contact-frame names the posed body actually reports
        (from :func:`_smplx_kinematic_skinned`); the labels are dropped when the
        file names a different frame set.
    """
    kin = pred / "human_optim" / body_fn
    if not kin.exists():
        return None
    kd = np.load(kin, allow_pickle=True)
    out: dict = {
        "parents": np.asarray(kd["parents"], np.int32),          # (J,)
        "joint_names": [str(x) for x in kd["joint_names"]],      # (J,)
        "frame_contact": None,                                   # (P, N, F) bool or None
        "frame_names": None,                                     # (F,) contact-frame names
        "frame_parent": None,                                    # (F,) parent joint per frame
        "contact_oids": None,                                    # (P,) or None
    }
    if contacts_fn is None:
        if "frame_contact" not in kd.files:
            return out
        names = [str(x) for x in np.asarray(kd["contact_frame_names"]).tolist()]
        contact = np.asarray(kd["frame_contact"], bool)
        parent = np.asarray(kd["contact_frame_parents"], np.int32)
        if frame_names is not None and names != frame_names:
            contact = _expand_frame_columns(contact, names, frame_names)
            if contact is None or frame_parents is None:
                return out
            names, parent = list(frame_names), frame_parents
        out["frame_contact"] = contact
        out["frame_names"] = names
        out["frame_parent"] = parent
        out["contact_oids"] = np.asarray(kd["object_ids"], np.int32)
        return out
    cf = pred / "human_optim" / contacts_fn
    if not cf.exists():
        return out
    cd = np.load(cf, allow_pickle=True)
    schema = (int(np.asarray(cd["contact_label_schema"]).item())
              if "contact_label_schema" in cd.files else -1)
    if schema != _CONTACT_SCHEMA:
        warnings.warn(
            f"ignoring contact artifact {cf} written for label schema {schema} "
            f"(viewer reads {_CONTACT_SCHEMA}); rerun estimate_contacts",
            RuntimeWarning, stacklevel=2)
        return out
    names = [str(x) for x in np.asarray(cd["contact_frame_names"]).tolist()]
    contact = np.asarray(cd["frame_contact"], bool)
    parent = np.asarray(cd["frame_parent_joint"], np.int32)
    contact_oids = np.asarray(cd["object_ids"], np.int32)
    body_oids = np.asarray(kd["object_ids"], np.int32)
    expected = (len(body_oids), int(np.asarray(kd["num_frames"]).item()), len(names))
    mismatch = frame_names is not None and names != frame_names
    if contact.shape == expected and mismatch:
        expanded = _expand_frame_columns(contact, names, frame_names)
        if expanded is not None and frame_parents is not None:
            contact, names, parent = expanded, list(frame_names), frame_parents
            expected = expected[:2] + (len(names),)
            mismatch = False
    if contact.shape != expected or mismatch or not np.array_equal(contact_oids, body_oids):
        warnings.warn(
            f"ignoring incompatible contact artifact {cf}; regenerate estimate_contacts",
            RuntimeWarning, stacklevel=2)
        return out
    out["frame_contact"] = contact
    out["frame_names"] = names
    out["frame_parent"] = parent
    out["contact_oids"] = contact_oids
    return out


_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _manifest_video(pred: Path) -> Path | None:
    """Source video the batch pipeline recorded for this clip, if it still exists.

    ``scripts/pipeline.py`` appends one row per processed clip to
    ``_batch_manifest.jsonl`` next to the clip's output tree, including the
    absolute ``source`` path — so a batch stored away from the repo still knows
    where its footage lives. The last row wins (reruns append).
    """
    manifest = pred.parent / "_batch_manifest.jsonl"
    if not manifest.exists():
        return None
    found = None
    for line in manifest.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        named = (row.get("stem") == pred.name
                 or Path(row.get("out_dir", "")).name == pred.name)
        if named and row.get("source"):
            found = Path(row["source"])
    if found is not None and found.is_file() and found.suffix.lower() in _VIDEO_EXTS:
        return found
    return None


def _load_video_frames(pred: Path, n_frames: int, *, video_dir: Path | None,
                       max_dim: int = 400) -> tuple[np.ndarray | None, float | None]:
    """Load the clip's original footage + its playback fps, aligned to the frame index.

    Finds ``<video_dir>/<clip-stem>.<ext>`` (``video_dir`` defaults to the repo's
    ``data/``), falling back to the ``source`` path in the batch manifest next to
    the clip (see :func:`_manifest_video`), and decodes it the **same way the
    pipeline did** — cv2 sequential —
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
        video = _manifest_video(pred)
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

    The same forward pass also reports the body's 12 **contact frames** — the skin
    points contact is detected on — so the viewer draws them exactly where the
    stages measured them: ``frame_pos``/``frame_normal`` per frame, in
    ``frame_names`` order.

    Returns ``(faces (F, 3), object_ids (P,), people)`` where ``people[p]`` is
    ``None`` (no valid frames) or a dict with ``v_shaped (V, 3)``, ``j_rest
    (J, 3)``, ``weights (V, J)``, per-frame ``bone_wxyz (N, J, 4)`` /
    ``bone_pos (N, J, 3)`` / ``frame_pos (N, F, 3)`` / ``frame_normal (N, F, 3)``
    (NaN where invalid), ``frame_names`` and ``valid (N,)``.
    """
    import torch

    from tools.body import frame_normals

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
    frame_names = list(mid.contact_frame_names)
    frame_parent = np.asarray([int(x) for x in mid.contact_frame_parents], np.int32)
    F = len(frame_names)

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
            posed = mid.forward(betas.expand(B, -1), q)
            jw = posed.joints_world.tensor()                               # (B, 52, 7) world
            fw = posed.frames_world                                        # (B, F, 7) contact frames
            # Global scale about the pelvis (rest geometry about rest pelvis, world
            # bones and contact frames about the world pelvis) — no-op at s=1, kept
            # for consumer compat. A uniform scale moves points but not directions,
            # so the frame rotations (and their normals) need no correction.
            s = float(scales[p])
            if s != 1.0:
                root_rest = j_rest[:, :1, :].clone()
                v_shaped = root_rest + s * (v_shaped - root_rest)
                j_rest = root_rest + s * (j_rest - root_rest)
                root_w = jw[..., :1, :3]
                fw = torch.cat([root_w + s * (fw[..., :3] - root_w), fw[..., 3:]], dim=-1)
                jw = torch.cat([root_w + s * (jw[..., :3] - root_w), jw[..., 3:]], dim=-1)
            quat = jw[..., 3:7].cpu().numpy()                              # xyzw
            bone_wxyz = np.full((N, J, 4), np.nan, np.float32)
            bone_pos = np.full((N, J, 3), np.nan, np.float32)
            bone_wxyz[idx] = quat[..., [3, 0, 1, 2]].astype(np.float32)    # → wxyz
            bone_pos[idx] = jw[..., :3].cpu().numpy().astype(np.float32)
            frame_pos = np.full((N, F, 3), np.nan, np.float32)
            frame_normal = np.full((N, F, 3), np.nan, np.float32)
            frame_pos[idx] = fw[..., :3].cpu().numpy().astype(np.float32)
            frame_normal[idx] = frame_normals(fw).cpu().numpy().astype(np.float32)
            people.append({
                "v_shaped": v_shaped[0].cpu().numpy().astype(np.float32),
                "j_rest": j_rest[0].cpu().numpy().astype(np.float32),
                "weights": weights,
                "bone_wxyz": bone_wxyz,
                "bone_pos": bone_pos,
                "frame_pos": frame_pos,
                "frame_normal": frame_normal,
                "frame_names": frame_names,
                "frame_parent": frame_parent,
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


def _load_mhr(device: str):
    """Load the MHR LOD1 body once per process (``body, torch device``)."""
    import torch

    import better_human as bh

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    key = str(dev)
    if key not in _MHR_CACHE:
        _MHR_CACHE[key] = bh.MHR(_MHR_MODEL, lod=_MHR_LOD, use_expression=False,
                                 use_correctives=False, compute_mass=False, device=dev)
    return _MHR_CACHE[key], dev


def _mhr_skinned(npz_path: Path, *, device: str = "cuda"):
    """Skinning data for the MHR fit of the refined body (``human_optim/mhr_1.npz``).

    ``mhr_1.npz`` is the kindyn body re-expressed on Meta's MHR rig: one identity
    per person, the full native ``lbs_params (204)`` per frame, and a world-frame
    ``q (132)`` per frame. Posing it here shows that fit as a moving mesh — the same
    fit ``mhr_sup_1.npz`` replays into keypoint / vertex training targets, which
    carry no topology of their own. Body only: no contact labels, no forces.

    **Every parameter has to be read per frame.** MHR has two kinds of size knob:
    fixed ``scale_*`` ones, and six "flexible" ones — spine, neck and arm and leg
    length, shoulder and hip width — that the MHR fitter may re-estimate on every
    frame, and does. Over one clip the fitted rest skeleton's legs swing by 10 cm
    and its spine by 9 cm, all hanging off a pelvis that never moves; identity, the
    other 67 size knobs and the rest mesh stay put. Shaping once and posing every
    frame off that one skeleton is wrong by 2.4 cm (median) to 8.6 cm, and wrong
    differently each frame — which reads as a mis-shaped body that moves oddly. So
    the per-frame skinning transform
    ``world_pose ∘ inverse_bind(frame)`` is folded into the bone pose here, and the
    mesh is uploaded with ONE rest skeleton and identity bind rotations (the same
    convention as the SMPL-X sources). The browser's LBS then reproduces
    BetterHuman's own posing to ~0.3 cm — all that is dropped are the per-joint
    scales (≥ 0.95, applied about each joint) and the pose correctives, which are
    per-vertex per-frame and cannot ride on a fixed rest mesh.

    Same ``(faces, object_ids, people)`` contract as :func:`_smplx_kinematic_skinned`.
    """
    import torch
    from better_robot.lie import se3, so3

    from better_human.bodies.mhr.body import MHRClassic

    mhr, dev = _load_mhr(device)
    d = np.load(npz_path)
    q_all = d["q_world"]                       # (P, N, 132), world frame
    valid = d["valid_mask"].astype(bool)       # (P, N)
    P, N = q_all.shape[:2]
    object_ids = np.asarray(d["object_ids"], np.int32)
    faces = mhr.structure.faces.cpu().numpy().astype(np.int32)

    people: list[dict | None] = []
    with torch.no_grad():
        for p in range(P):
            idx = np.flatnonzero(valid[p])
            if idx.size == 0:
                people.append(None)
                continue
            lbs = torch.as_tensor(d["lbs_params"][p][idx], device=dev)         # (B, 204)
            shaped, _q = mhr.from_classic(MHRClassic(
                identity_coeffs=torch.as_tensor(d["identity"][p], device=dev)[None]
                .expand(len(lbs), -1),
                model_parameters=lbs))                       # a body per frame
            q = torch.as_tensor(q_all[p][idx], dtype=torch.float32, device=dev)
            poses = shaped.native_joint_poses(shaped.fk(q))                    # (B, J, 7) xyzw
            skin = se3.compose(poses, shaped.values.inverse_bind_poses)        # (B, J, 7)
            # One rest skeleton for the upload (the identity fixes the rest mesh, so
            # only the skeleton re-proportions); the bone pose carries the rest.
            j_rest = shaped.values.rest_joints[0]                              # (J, 3)
            rot = so3.to_matrix(skin[..., 3:7])                                # (B, J, 3, 3)
            pos = (rot @ j_rest.unsqueeze(-1)).squeeze(-1) + skin[..., :3]
            J = j_rest.shape[0]
            bone_wxyz = np.full((N, J, 4), np.nan, np.float32)
            bone_pos = np.full((N, J, 3), np.nan, np.float32)
            bone_wxyz[idx] = skin[..., [6, 3, 4, 5]].cpu().numpy()             # → wxyz
            bone_pos[idx] = pos.cpu().numpy()
            vmask = np.zeros(N, bool)
            vmask[idx] = True
            people.append({
                "v_shaped": shaped.values.v_shaped[0].cpu().numpy().astype(np.float32),
                "j_rest": j_rest.cpu().numpy().astype(np.float32),
                "weights": _top4_skin_weights(
                    shaped.values.skinning_weight_matrix.cpu().numpy()),
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
    forces: dict           # force source name ("kindyn_1"/"kindyn_manual") -> force dict
    sensors: dict | None   # measured wall-sensor forces (climb_wall), see _load_sensor_forces
    plates: dict | None    # our forces in the plates' force-at-COP format, see _load_plate_estimate
    skeletons: dict        # body source name -> joint tree + contact-frame labels (_load_skeleton)
    n_frames: int
    # scene-surface meshes, built lazily and cached per method ("bpa"/"poisson");
    # a present key maps to (verts, faces, vertex_colors, vertex_normals) or None.
    scene_meshes: dict = field(default_factory=dict)
    scene_normals: np.ndarray | None = None   # camera-oriented normals (from fuse, else lazy)
    scene_voxel: float = 0.0                  # merge voxel size (m) — sizes the normal arrows
    video_frames: np.ndarray | None = None    # original footage, frame-slider aligned (sidebar pane)
    fps: float | None = None                  # real-time playback fps (from the source video)
    views: dict | None = None                 # calibrated static cameras (_load_view_cameras)
    video_views: dict | None = None           # view name -> its footage (_load_view_videos)


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


def _fov_y_rad(transform) -> np.ndarray:
    """Per-frame vertical field of view (radians) from ``geometry/transform.npz``.

    VGGT records it directly (``fov_y_deg``); a calibrated rig saves only the pixel
    intrinsics, so there it is derived from the focal length and the image height.
    """
    if "fov_y_deg" in transform.files:
        return np.radians(transform["fov_y_deg"].astype(np.float64))
    fy = transform["intrinsics_px_orig"][:, 1, 1].astype(np.float64)
    return 2.0 * np.arctan(float(transform["image_height"]) / (2.0 * fy))


def load_clip_data(pred: Path, *, device: str, show_smplx: bool,
                   human_max_points: int = 0, video: bool = True,
                   video_dir: Path | None = None) -> ClipData:
    """Load one clip's geometry + all SMPL-X body sources into numpy."""
    geo = pred / "geometry"
    scene = np.load(geo / "scene.npz")
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

    # Per-frame human depth points — absent on clips fused from calibrated cameras
    # rather than VGGT depth (the climb_wall_3 ground truth), which just leaves the
    # human-points layer empty.
    if (geo / "human.npz").exists():
        human = np.load(geo / "human.npz")
        hp, hc = human["points"], human["colors"]
        hf, ho = human["frame_id"], human["obj_id"]
    else:
        hp, hc = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
        hf, ho = np.zeros(0, np.int32), np.zeros(0, np.int32)
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
        for name, fn in (("kinematics", "kinematic_1.npz"), ("kinematics_2", "kinematic_2.npz"),
                         ("kindyn_1", "kindyn_1.npz"), ("kindyn_manual", "kindyn_manual.npz")):
            if (pred / "human_optim" / fn).exists():
                skinned[name] = _smplx_kinematic_skinned(pred / "human_optim" / fn, device=device)
        # mhr_1_sup: the kindyn body re-fitted onto the MHR rig — a mesh and nothing
        # else (its contacts and forces live on the SMPL-X sources it came from).
        if (pred / "human_optim" / "mhr_1.npz").exists():
            skinned["mhr_1_sup"] = _mhr_skinned(pred / "human_optim" / "mhr_1.npz",
                                                device=device)
        # prefit: the scale stage's dual scale+body alignment (camera frame, like original).
        if (pred / "geometry" / "prefit.npz").exists():
            pf = _smplx_params_skinned(pred / "geometry" / "prefit.npz", extr, device=device)
            if pf is not None:
                skinned["prefit"] = pf

    # Force/torque sources, in display order — the kinematics+dynamics bodies,
    # whichever exist.
    forces: dict = {}
    for name, fn in (("kindyn_1", "kindyn_1.npz"),
                     ("kindyn_manual", "kindyn_manual.npz")):
        fd = _load_force_data(pred, fn)
        if fd is not None:
            forces[name] = fd

    # Body skeletons + their contact-frame labels. The posed body names the contact
    # frames it reports; an artifact for a SUBSET of them (a [contacts].frames
    # restriction, e.g. the peter rigs) is expanded into body order, anything else dropped.
    # The kindyn bodies are self-labeled (contacts_fn=None): their npz carries
    # frame_contact next to q, which feeds the contact-frames layer; their skeleton
    # stays the force-source torque skeleton (ClipScene skips their 22-joint overlay).
    skeletons: dict = {}
    if show_smplx:
        for name, body_fn, contacts_fn in (("kinematics", "kinematic_1.npz", "contacts_1.npz"),
                                           ("kinematics_2", "kinematic_2.npz", "contacts_2.npz"),
                                           ("kindyn_1", "kindyn_1.npz", None),
                                           ("kindyn_manual", "kindyn_manual.npz", None)):
            body = skinned.get(name)
            person = next((p for p in body[2] if p is not None), None) if body else None
            sk = _load_skeleton(pred, body_fn, contacts_fn,
                                frame_names=None if person is None else person["frame_names"],
                                frame_parents=None if person is None else person["frame_parent"])
            if sk is not None:
                skeletons[name] = sk

    # Video pane: a multi-view clip plays whichever calibrated view the sidebar's
    # ``view`` dropdown picks (all on the frame slider's clock); everything else has
    # the one source clip.
    views = _load_view_cameras(pred)
    video_views = (_load_view_videos(pred, len(extr), views)
                   if video and views is not None else None)
    if video_views:
        video_frames, play_fps = video_views[next(iter(video_views))], None
    else:
        video_frames, play_fps = (_load_video_frames(pred, len(extr), video_dir=video_dir)
                                  if video else (None, None))
    if play_fps is None and "fps" in transform.files:
        # No decodable source video (video pane off, or --scenes farm without the
        # clip) — fall back to the container fps the pipeline recorded, scaled for
        # any uniform frame subsample exactly like the video path above.
        fi = transform["frame_indices"] if "frame_indices" in transform.files else None
        span = float(fi[-1] - fi[0]) if fi is not None and len(fi) > 1 else 0.0
        raw = float(transform["fps"])
        play_fps = (raw * (len(fi) - 1) / span if span > 0.0 else raw) or None
    return ClipData(
        path=pred, name=pred.name,
        scene_points=sp, scene_colors=sco,
        human_points=hp, human_colors=hc, human_fid=hf, human_obj=ho,
        obj_ids=[int(o) for o in np.unique(ho)],
        extr=extr, aspect=float(transform["image_width"]) / float(transform["image_height"]),
        fov_y=_fov_y_rad(transform),
        gravity=_load_gravity(pred),
        skinned=skinned,
        forces=forces,
        sensors=_load_sensor_forces(pred),
        plates=_load_plate_estimate(pred),
        skeletons=skeletons,
        n_frames=len(extr),
        scene_normals=sn,
        scene_voxel=float(scene["voxel_size"]) if "voxel_size" in scene.files else 0.0,
        video_frames=video_frames, fps=play_fps,
        views=views, video_views=video_views)


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
