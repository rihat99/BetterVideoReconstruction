"""The pipeline's human chain on one climb_wall_3 clip, seen by three calibrated cameras.

Same stages and artifacts as the monocular pipeline (``human_optim/kinematic_1.npz`` →
``contacts_1.npz`` → ``kinematic_2.npz`` → ``kindyn_1.npz``), same optimizer, same weight
columns (``human_gt.toml``). What is different: the image evidence is summed over the
three cameras, and the contact labels come from the force boards.

``fit`` (kinematic_1)
    Each camera contributes the pipeline's two 2D terms — Sapiens keypoints
    (``reproj_sapiens``) and the anchor to that camera's own SAM-3D-Body estimate
    (``reproj_init``) — through its calibrated intrinsics and pose. Three cameras
    triangulate what one cannot (depth, limb lengths). The shape (``betas``) is fitted in the
    refine + L-BFGS phases against the participant's real height and mass, unless a frozen
    shape file ``_shapes/<participant>.npz`` exists (then it is used as is).
``shape``
    One shape per participant: the mean of the fitted betas over their clips, frozen into
    ``_shapes/<participant>.npz``. The clips are then refitted with it.
``contacts`` (contacts_1)
    A limb is in contact where the board force exceeds a threshold (with hysteresis).
    The label lands on the limb's contact frames within ``near_scene_m`` of the wall; when
    none is, on its nearest frame over the span.
``kin2`` (kinematic_2)
    The pipeline's contact-patch repair: each labelled frame is pulled onto the wall patch
    seen at its own pixel.
``kindyn`` (kindyn_1)
    Pose + forces solved together against the labels; per-limb forces compared with the
    boards. Feet keep the friction cones; hands get a free force (a grip can pull).
``sync``
    Each camera's frame lag behind cam_left (run once, before ``fit``).
``cam3``
    cam_3's pose re-solved per clip: Sapiens keypoints triangulated from the calibrated
    front pair, then PnP into cam_3 (its intrinsics are calibrated; its pose was knocked
    between recordings). Applied only where it lowers cam_3's reprojection error.

Two things every stage shares, both from the boards and the calibration rather than
from the images: the WALL PLANE (``geometry/wall.npz``; nothing of the climber can be
behind it, so a hinge on every mesh vertex behind the plane replaces a heavy point-cloud
penetration weight), and BOARD STILLNESS (a limb the boards call loaded has all its
contact frames velocity-damped — the pipeline's ``ann_velocity`` fed with limb labels
already in ``kinematic_1``). ``kindyn`` adds ``board_force``: each limb's summed solved
force is pulled to the board's measurement, so the boards shape the pose through the
dynamics.

    uv run --no-sync python scripts/peter_gt/fit_multiview.py --clip a_00_01_part3 --stage fit
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scipy.spatial import cKDTree  # noqa: E402

from tools.human_optim.camera import fold_root_to_world, world_from_cam  # noqa: E402
from tools.human_optim.boards import (  # noqa: E402
    LIMBS, frame_contact as board_frame_contact, limb_contact, limb_of_frame,
)
from tools.human_optim.config import load_config  # noqa: E402
from tools.human_optim.force_confidence import force_confidence  # noqa: E402
from tools.human_optim.keypoints import smooth_tracks  # noqa: E402
from tools.human_optim.kindyn import _GRAVITY_MAG, build_dynamics_spec  # noqa: E402
from tools.human_optim.losses import init_reprojection_gm, sapiens_reprojection_gm  # noqa: E402
from tools.human_optim.optimizer import PhaseSpec, optimize_person  # noqa: E402
from tools.human_optim.prefit import smooth_finger_quats  # noqa: E402
from tools.human_optim.qmath import normalize_q  # noqa: E402
from tools.human_optim.stages import (  # noqa: E402
    _ALL_DOF, _ROOT_DOF, _append_classic, _body_context, _fill_invalid_frames, _load_scene,
    _new_out, _save_body, load_frame_contacts,
)
from tools.npz_io import atomic_savez  # noqa: E402
from tools.paths import PETER  # noqa: E402

HUMAN_ROOT = PETER / "climb_wall_3_gt/human"
SHAPES = HUMAN_ROOT / "_shapes"
CONFIG = Path(__file__).resolve().parent / "human_gt.toml"
VIEW_NAMES = ["cam_left", "cam_right", "cam_3"]
FINGER = ("index", "middle", "pinky", "ring", "thumb")
#: The two climbers (clip name ``a_*`` / ``b_*``), as told by the user.
PARTICIPANTS = {"a": {"name": "A", "height_m": 1.80, "mass_kg": 72.5},
                "b": {"name": "B", "height_m": 1.60, "mass_kg": 48.0}}
INIT_SHIFT_SMOOTH_S = 0.5     # the init pelvis shift to the multi-view mean is smoothed over this


def participant(clip: str) -> dict:
    return PARTICIPANTS[clip[0]]


def gt_section(name: str) -> dict:
    """A section of ``human_gt.toml`` the pipeline's config loader does not expose ([multiview], [boards])."""
    with CONFIG.open("rb") as f:
        return tomllib.load(f)[name]


# ───────────────────────────── inputs ─────────────────────────────

def load_clip(clip: str, device: str, frame_offset: np.ndarray | None = None) -> dict:
    """Cameras, frame count and per-view SMPL-X estimates + Sapiens keypoints of one clip."""
    clip_dir = HUMAN_ROOT / clip
    cams = np.load(clip_dir / "cameras.npz", allow_pickle=True)
    # A camera can lag cam_left by a frame or two (cameras.npz "frame_offset", 0 when unknown):
    # view k's video frame t + offset_k shows the same instant as cam_left's frame t. The
    # fit runs on cam_left's clock, so every view is shifted by its offset and the clip is
    # shortened by the largest one. ``frame_offset`` overrides the stored one (the sync stage).
    offsets = (np.asarray(frame_offset, int) if frame_offset is not None
               else np.asarray(cams["frame_offset"], int) if "frame_offset" in cams.files else np.zeros(3, int))
    n = int(cams["num_frames"]) - int(offsets.max())
    views = []
    for k, view in enumerate(VIEW_NAMES):
        vdir = clip_dir / "views" / view
        sp = dict(np.load(vdir / "sam3d" / "smplx_params.npz", allow_pickle=True))
        if int(sp["num_people"]) != 1:
            print(f"  [{view}] {int(sp['num_people'])} people tracked — using the first")
        off = int(offsets[k])
        assert int(sp["num_frames"]) >= off + n, (view, int(sp["num_frames"]), off, n)
        params = {key: np.asarray(sp[key][0][off:off + n]) for key in
                  ("betas", "body_pose", "global_orient", "transl", "left_hand_pose", "right_hand_pose")}
        # A tracked frame whose converter output is not finite is no observation either.
        finite = np.all([np.isfinite(a).reshape(n, -1).all(1) for a in params.values()], axis=0)
        valid = np.asarray(sp["valid_mask"][0][off:off + n], bool) & finite
        kp = np.zeros((n, 308, 2), np.float32)
        sc = np.zeros((n, 308), np.float32)
        pose_path = vdir / "sapiens" / "pose.npz"
        if pose_path.exists():
            pose = np.load(pose_path, allow_pickle=True)
            kp = np.asarray(pose["keypoints"][0][off:off + n], np.float32)
            sc = np.asarray(pose["keypoint_scores"][0][off:off + n], np.float32)
            seen = np.isfinite(kp).all(-1) & np.isfinite(sc)      # a non-finite keypoint is no observation
            kp, sc = np.where(seen[..., None], kp, 0.0), np.where(seen, sc, 0.0)
            sc[~valid] = 0.0
        else:
            print(f"  [{view}] no sapiens/pose.npz — no keypoint witness in this view")
        views.append({"name": view, "params": params, "frame_offset": off, "valid": valid,
                      "kp": kp, "kp_scores": sc,
                      "K": torch.as_tensor(cams["K"][k], dtype=torch.float32, device=device),
                      "extr": torch.as_tensor(cams["extrinsics"][k], dtype=torch.float32, device=device),
                      "size": (int(cams["image_size"][k][0]), int(cams["image_size"][k][1]))})
    return {"dir": clip_dir, "n": n, "fps": float(cams["fps"]), "views": views,
            "camera_set": str(cams["camera_set"]), "cameras": cams, "frame_offset": offsets}


def view_bodies(mid, views: list[dict], n: int, device: str) -> None:
    """Pose each view's converter estimate in the WORLD (its camera folded in); adds
    ``q_world``, ``betas`` (mean over valid frames), ``joints`` (N, 52, 3) per view."""
    for v in views:
        p, valid = v["params"], v["valid"]
        betas = torch.as_tensor(p["betas"][valid].mean(0) if valid.any() else np.nan_to_num(p["betas"]).mean(0),
                                dtype=torch.float32, device=device)
        t = lambda k: torch.as_tensor(_fill_invalid_frames(p[k], valid), dtype=torch.float32, device=device)
        betas_T = betas[None].expand(n, -1)
        q_cam = mid.from_classic(betas_T, t("body_pose"), t("transl"), t("global_orient"),
                                 t("left_hand_pose"), t("right_hand_pose"))
        wfc = world_from_cam(v["extr"][None].expand(n, -1, -1))
        v["q_world"] = fold_root_to_world(q_cam, wfc)
        v["betas"] = betas
        with torch.no_grad():
            v["joints"] = mid.forward(betas_T, normalize_q(v["q_world"], mid.num_joints)).joints_world.tensor()[..., :3]


def project_view(points: torch.Tensor, extr: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pinhole projection of world points ``(..., 3)`` through one static camera → ``(..., 2)``."""
    cam = points @ extr[:3, :3].T + extr[:3, 3]
    uv = cam[..., :2] / cam[..., 2:3].clamp_min(1e-6)
    return torch.stack([K[0, 0] * uv[..., 0] + K[0, 2], K[1, 1] * uv[..., 1] + K[1, 2]], -1)


def body_joint_idx(mid, device: str) -> torch.Tensor:
    """The 22 body joints (no fingers) the SAM-3D-Body anchor holds."""
    names = list(mid.joint_names)
    return torch.tensor([i for i in range(mid.num_joints) if not any(f in names[i].lower() for f in FINGER)],
                        dtype=torch.long, device=device)


# ───────────────────────── image evidence per camera ─────────────────────────

def image_terms(views: list[dict], mid, corr, fvids: torch.Tensor, knobs: dict, fps: float,
                device: str, margin_px: float) -> tuple[dict, dict]:
    """The pipeline's two 2D terms, once per camera.

    ``reproj_sapiens/<view>``: :func:`sapiens_reprojection_gm` to that view's Goliath-308
    keypoints. ``reproj_init/<view>``: :func:`init_reprojection_gm` to the projection of that
    view's OWN SAM-3D-Body estimate (its 22 body joints, frozen targets — the converter's
    good 2D alignment in that camera, not its depth). Targets are SG-smoothed like the
    pipeline does; the readout (:func:`pixel_readout`) measures against the raw ones.
    Returns ``(terms, targets)`` — ``targets[view]`` keeps the raw arrays for the readout.
    """
    kp_cfg, ri_cfg = knobs["keypoints"], knobs["reproj_init"]
    thresh, sap_gm, init_gm = float(kp_cfg["score_thresh"]), float(kp_cfg["gm_scale_px"]), float(ri_cfg["gm_scale_px"])
    body_idx = body_joint_idx(mid, device)
    terms, targets = {}, {}
    for v in views:
        n = len(v["valid"])
        extr, K = v["extr"][None].expand(n, -1, -1), v["K"][None].expand(n, -1, -1)
        valid = torch.as_tensor(v["valid"], device=device)
        kp = torch.as_tensor(v["kp"], dtype=torch.float32, device=device)
        sc = torch.as_tensor(v["kp_scores"], dtype=torch.float32, device=device)
        kp_loss = kp
        if float(kp_cfg.get("smooth_window_s", 0.0)) > 0.0:
            sm = smooth_tracks(v["kp"], v["kp_scores"] >= thresh, window_s=float(kp_cfg["smooth_window_s"]), fps=fps)
            kp_loss = torch.as_tensor(sm, dtype=torch.float32, device=device)
        # That view's own estimate projected into that view: the 2D anchor targets.
        with torch.no_grad():
            init_px = project_view(v["joints"][:, body_idx], v["extr"], v["K"])          # (n, Jb, 2)
        w, h = v["size"]
        inside = ((init_px[..., 0] > -margin_px) & (init_px[..., 0] < w + margin_px)
                  & (init_px[..., 1] > -margin_px) & (init_px[..., 1] < h + margin_px)
                  & torch.isfinite(init_px).all(-1) & valid[:, None])
        init_px = torch.nan_to_num(init_px)
        init_loss = init_px
        if float(ri_cfg.get("smooth_window_s", 0.0)) > 0.0:
            sm = smooth_tracks(init_px.cpu().numpy(), inside.cpu().numpy(),
                               window_s=float(ri_cfg["smooth_window_s"]), fps=fps)
            init_loss = torch.as_tensor(sm, dtype=torch.float32, device=device)
        joint_w = inside.float()

        def sap_term(st, extr=extr, K=K, kp_loss=kp_loss, sc=sc):
            return sapiens_reprojection_gm(st["jw"], st["vw"], extr, K, kp_loss, sc, corr, fvids,
                                           score_thresh=thresh, gm_scale_px=sap_gm)

        def init_term(st, extr=extr, K=K, init_loss=init_loss, joint_w=joint_w, valid=valid):
            return init_reprojection_gm(st["jw"], body_idx, extr, K, init_loss, gm_scale_px=init_gm,
                                        frame_mask=valid, joint_weight=joint_w)
        terms[f"reproj_sapiens/{v['name']}"] = sap_term
        terms[f"reproj_init/{v['name']}"] = init_term
        targets[v["name"]] = {"kp": kp, "scores": sc, "init": init_px, "init_w": joint_w, "body_idx": body_idx}
    return terms, targets


def view_weights(column: dict, views: list[dict], sigma_px: np.ndarray | None = None) -> dict:
    """The stage's weight column with the two image terms spread over the cameras.

    The cameras share the column's weight (so the image evidence as a whole keeps the
    balance against the priors the monocular column was tuned with): equally, or in
    proportion to ``1/σ²`` when ``sigma_px`` holds each camera's measured keypoint
    consistency (``cameras.npz view_sigma_px``, written by the ``cam3`` stage — the front
    pair agrees to 2-5 px, the side camera to ~20 px). The built-in single-camera terms
    are switched off (their targets are only kept for the readout).
    """
    if sigma_px is None:
        share = np.full(len(views), 1.0 / len(views))
    else:
        inv = 1.0 / np.square(np.asarray(sigma_px, float).clip(min=1.0))
        share = inv / inv.sum()
    w = dict(column)
    for base in ("reproj_sapiens", "reproj_init"):
        for v, f in zip(views, share):
            w[f"{base}/{v['name']}"] = float(column.get(base, 0.0)) * float(f)
        w[base] = 0.0
    return w


def view_sigma(cams) -> np.ndarray | None:
    return np.asarray(cams["view_sigma_px"], float) if "view_sigma_px" in cams.files else None


def pixel_readout(mid, views: list[dict], targets: dict, corr, fvids: torch.Tensor, betas: torch.Tensor,
                  q: torch.Tensor, thresh: float) -> dict[str, tuple[float, float]]:
    """Per camera ``(sapiens px, anchor px)``: mean pixel error of the posed body to the raw
    Sapiens keypoints (scores ≥ thresh) and to the SAM-3D-Body anchor targets."""
    out = {}
    with torch.no_grad():
        o = mid.forward(betas[None].expand(q.shape[0], -1), normalize_q(q, mid.num_joints))
        jw, vw = o.joints_world.tensor()[..., :3], o.vertices
        for v in views:
            t = targets[v["name"]]
            pred = torch.cat([jw[:, corr.joint_model], vw[:, fvids[corr.lm_frame]]], 1)
            px = project_view(pred, v["extr"], v["K"])
            w = (t["scores"][:, corr.sap] >= thresh).float()
            err = (px - t["kp"][:, corr.sap]).norm(dim=-1)
            sap = float((err * w).sum() / w.sum().clamp_min(1.0))
            pxi = project_view(jw[:, t["body_idx"]], v["extr"], v["K"])
            erri = (pxi - t["init"]).norm(dim=-1)
            init = float((erri * t["init_w"]).sum() / t["init_w"].sum().clamp_min(1.0))
            out[v["name"]] = (sap, init)
    return out


def fmt_readout(r: dict[str, tuple[float, float]]) -> str:
    return " ".join(f"{k}: sap {s:.1f} / anchor {i:.1f}" for k, (s, i) in r.items())


def root_jitter_cm(q: torch.Tensor, fps: float) -> str:
    """Root speed statistics, cm per frame — the teleport / jitter check."""
    step = np.linalg.norm(np.diff(q[:, :3].detach().cpu().numpy(), axis=0), axis=1) * 100
    return (f"root step cm/frame med {np.median(step):.2f} p99 {np.percentile(step, 99):.1f} "
            f"max {step.max():.1f} ({fps:.0f} fps)")


# ───────────────────────────── shape ─────────────────────────────

def shape_of(mid, betas: torch.Tensor, *, fresh: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """``(height m, mass kg)`` of the neutral-pose body at ``betas (10,)``; both differentiable.

    Height is the vertical extent of the shaped mesh; mass is the Dempster-density inertia
    bake BetterHuman does for the dynamics, so the kindyn body weighs exactly this.
    """
    body = mid.shaped(betas, fresh=fresh)
    v = body.values.v_shaped
    height = v[..., 1].max() - v[..., 1].min()
    mass = body.values.body_inertias[..., 1:, 0].sum()
    return height, mass


def shape_terms(mid, betas0: torch.Tensor, who: dict) -> dict:
    """``shape_height`` / ``shape_mass`` (squared error to the participant's real values, m² / kg²)
    and ``betas_prior`` (squared distance to the converter mean)."""
    h_target, m_target = float(who["height_m"]), float(who["mass_kg"])

    def height_term(st):
        h, _ = shape_of(mid, st["betas"], fresh=True)
        return (h - h_target).square()

    def mass_term(st):
        _, m = shape_of(mid, st["betas"], fresh=True)
        return (m - m_target).square()
    return {"shape_height": height_term, "shape_mass": mass_term,
            "betas_prior": lambda st: (st["betas"] - betas0).square().sum()}


def initial_pose(views: list[dict], valid_any: np.ndarray, fps: float, device: str) -> torch.Tensor:
    """cam_left's estimate, its pelvis moved to the mean of the views' pelvises.

    The per-frame shift is smoothed over ``INIT_SHIFT_SMOOTH_S`` so the three estimates'
    frame-to-frame disagreement does not become jitter in the start pose; the root phase
    then places the body from the 2D evidence of all cameras.
    """
    q = views[0]["q_world"].clone()
    pelvis = torch.stack([v["joints"][:, 0] for v in views])                      # (V, N, 3)
    vw = torch.stack([torch.as_tensor(v["valid"], device=device) for v in views]).float()[..., None]
    mean = (pelvis * vw).sum(0) / vw.sum(0).clamp_min(1.0)
    shift = (mean - pelvis[0]).cpu().numpy()
    shift[~valid_any] = 0.0
    med_cm = float(np.median(np.linalg.norm(shift[valid_any], axis=1))) * 100 if valid_any.any() else 0.0
    if med_cm > 100.0:     # three monocular estimates of ONE person disagree by decimetres, not metres
        raise SystemExit(f"the views' pelvises disagree by {med_cm:.0f} cm — the per-view SAM3 tracks "
                         "are probably different people; check sam3/ per view")
    shift = smooth_tracks(shift[:, None, :], valid_any[:, None], window_s=INIT_SHIFT_SMOOTH_S, fps=fps)[:, 0]
    q[:, :3] += torch.as_tensor(shift, dtype=torch.float32, device=device)
    print(f"  init: pelvis shift to the multi-view mean, median {med_cm:.1f} cm")
    return q


def load_wall(clip_dir: Path, device: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(normal, point)`` of the wall plane (``geometry/wall.npz``), normal out of the wall."""
    path = clip_dir / "geometry" / "wall.npz"
    if not path.exists():
        print(f"  no {path.name} — wall-plane term off (run prepare_human.py --scene-only)")
        return None
    z = np.load(path)
    return (torch.as_tensor(z["normal"], dtype=torch.float32, device=device),
            torch.as_tensor(z["point"], dtype=torch.float32, device=device))


def wall_penetration_term(wall: tuple[torch.Tensor, torch.Tensor], sub: torch.Tensor):
    """Squared hinge on every subsampled mesh vertex and contact frame BEHIND the wall plane."""
    normal, point = wall

    def term(st):
        pts = torch.cat([st["vw"][:, sub], st["fw"]], 1)              # (T, Ns + F, 3)
        behind = torch.relu(-((pts - point) * normal).sum(-1))         # metres behind the plane
        with torch.no_grad():
            info = {"cm": behind.max() * 100.0, "n": (behind > 0.01).sum()}
        # Summed over points, averaged over frames: every buried point pays its own depth, so a
        # single hand 3 cm inside the wall is not diluted by the thousand points that are fine.
        return behind.square().sum() / behind.shape[0], info
    return term


def board_limb_on(clip_dir: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(limb_on (N, 4) bool, force_world (N, 4, 3))`` from the boards with the [boards] hysteresis."""
    boards = gt_section("boards")
    return limb_contact(clip_dir / "sensor_forces.npz", n,
                        float(boards["force_enter_n"]), float(boards["force_exit_n"]))


def board_frame_labels(limb_on: np.ndarray, frame_names: list[str], valid: np.ndarray) -> np.ndarray:
    """``(N, F)`` int8: +1 on every contact frame of a limb the boards call loaded, else -1."""
    lab = np.full((len(limb_on), len(frame_names)), -1, np.int8)
    for c, name in enumerate(frame_names):
        limb = limb_of_frame(name)
        if limb is not None:
            lab[limb_on[:, LIMBS.index(limb)], c] = 1
    lab[~valid] = -1
    return lab


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def view_extras(data: dict, views: list[dict]) -> dict:
    """Per-view bookkeeping every body file carries (viewer + overlay read it)."""
    return dict(view_names=np.array(VIEW_NAMES), view_valid=np.stack([v["valid"] for v in views]),
                view_q_world=np.stack([v["q_world"].cpu().numpy() for v in views]).astype(np.float32),
                view_betas=np.stack([v["betas"].cpu().numpy() for v in views]).astype(np.float32),
                frame_offset=np.asarray(data["frame_offset"], np.int32),
                camera_set=np.str_(data["camera_set"]), config=np.str_(str(CONFIG)))


def readout_extras(before: dict, after: dict) -> dict:
    return dict(reproj_sapiens_px_init=np.array([before[v][0] for v in VIEW_NAMES], np.float32),
                reproj_sapiens_px_after=np.array([after[v][0] for v in VIEW_NAMES], np.float32),
                reproj_anchor_px_init=np.array([before[v][1] for v in VIEW_NAMES], np.float32),
                reproj_anchor_px_after=np.array([after[v][1] for v in VIEW_NAMES], np.float32))


# ───────────────────────────── stage: fit ─────────────────────────────

def run_fit(clip: str, device: str) -> None:
    cfg = load_config(CONFIG)
    knobs = cfg.knobs
    data = load_clip(clip, device)
    views, n, fps = data["views"], data["n"], data["fps"]
    tag = f"fit[{clip}]"
    mid, fvids, corr, sub, chamfer_w = _body_context(knobs, device)
    torch.manual_seed(0)
    view_bodies(mid, views, n, device)
    valid_any = np.any([v["valid"] for v in views], axis=0)
    if not valid_any.any():
        raise SystemExit(f"[{tag}] no view tracked any frame — nothing to fit")
    counts = " ".join(f"{v['name']}={int(v['valid'].sum())}" for v in views)
    kp_counts = " ".join(f"{v['name']}={float((v['kp_scores'] >= knobs['keypoints']['score_thresh']).mean()):.0%}"
                         for v in views)
    print(f"[{tag}] {n} frames, camera set {data['camera_set']}; tracked per view: {counts}; "
          f"keypoints ≥ thresh: {kp_counts}")

    q_init = initial_pose(views, valid_any, fps, device)
    smooth_finger_quats(q_init, valid_any, float(cfg.stage_opts["kinematics_1"].get("finger_smooth_s", 0.0)) * fps)
    q_init[:, 3:] += 1e-6 * torch.randn_like(q_init[:, 3:])   # break bit-exact static-frame ties

    who = participant(clip)
    shape_file = SHAPES / f"{who['name']}.npz"
    fit_betas = bool(cfg.stage_opts["kinematics_1"].get("fit_betas", True)) and not shape_file.exists()
    if shape_file.exists():
        betas0 = torch.as_tensor(np.load(shape_file)["betas"], dtype=torch.float32, device=device)
        shape_source = f"frozen {shape_file.name}"
    elif fit_betas:
        wv = torch.as_tensor([float(v["valid"].sum()) for v in views], device=device)
        betas0 = (torch.stack([v["betas"] for v in views]) * wv[:, None]).sum(0) / wv.sum()
        shape_source = "fitted (height + mass + images)"
    else:
        wv = torch.as_tensor([float(v["valid"].sum()) for v in views], device=device)
        betas0 = (torch.stack([v["betas"] for v in views]) * wv[:, None]).sum(0) / wv.sum()
        shape_source = "converter mean (fit_betas off)"
    with torch.no_grad():
        h0, m0 = (float(x) for x in shape_of(mid, betas0))
    print(f"[{tag}] participant {who['name']} ({who['height_m']:.2f} m, {who['mass_kg']:.1f} kg); "
          f"shape {shape_source}; start height {h0:.3f} m, mass {m0:.1f} kg")

    terms, targets = image_terms(views, mid, corr, fvids, knobs, fps, device,
                                 float(gt_section("multiview")["image_margin_px"]))
    if fit_betas:
        terms.update(shape_terms(mid, betas0, who))
    wall = load_wall(data["dir"], device)
    if wall is not None:
        terms["wall_penetration"] = wall_penetration_term(wall, sub)
    # Board stillness: every contact frame of a limb the boards call loaded is velocity-damped
    # (the optimizer's ann_velocity on these labels); no attraction/acquisition here.
    limb_on, _ = board_limb_on(data["dir"], n)
    labels = torch.as_tensor(board_frame_labels(limb_on, list(mid.contact_frame_names), valid_any),
                             dtype=torch.int8, device=device)
    print(f"[{tag}] board stillness on " + " ".join(f"{l}={int(limb_on[:, i].sum())}f" for i, l in enumerate(LIMBS)))
    it_root, it = cfg.iters["root"], cfg.iters["kinematics_1"]
    sig = view_sigma(data["cameras"])
    w_root, w = view_weights(cfg.weights_for("root"), views, sig), view_weights(cfg.weights_for("kinematics_1"), views, sig)
    print(f"[{tag}] camera weights " + " ".join(f"{v['name']}={w['reproj_sapiens/' + v['name']] / max(float(cfg.weights_for('kinematics_1')['reproj_sapiens']), 1e-9):.2f}" for v in views)
          + (f" (σ px {sig.round(1)})" if sig is not None else " (equal)"))
    shape_dof = (*_ALL_DOF, "betas") if fit_betas else _ALL_DOF
    phases = [PhaseSpec("root", w_root, it_root.get("root", 0), _ROOT_DOF, False, "adam"),
              PhaseSpec("full", w, it.get("full", 0), _ALL_DOF, False, "adam"),
              PhaseSpec("refine", w, it.get("refine", 0), shape_dof, True, "adam"),
              PhaseSpec("lbfgs", w, it.get("lbfgs", 0), shape_dof, True, "lbfgs")]

    wfc = world_from_cam(torch.stack([v["extr"] for v in views]))
    scene_pts, scene_normals, scene_conf = _load_scene(data["dir"], knobs, wfc, device, tag)
    left = views[0]
    extr_T, K_T = left["extr"][None].expand(n, -1, -1), left["K"][None].expand(n, -1, -1)
    frame_valid = torch.as_tensor(valid_any, device=device)
    thresh = float(knobs["keypoints"]["score_thresh"])
    before_px = pixel_readout(mid, views, targets, corr, fvids, betas0, q_init, thresh)
    print(f"[{tag}] px at init   {fmt_readout(before_px)}")

    q_opt, before, after, aux = optimize_person(
        mid, betas0, q_init, extrinsics=extr_T, K=K_T,
        depth_points=[torch.zeros((0, 3), device=device) for _ in range(n)],
        target_kp=targets["cam_left"]["kp"], scores=targets["cam_left"]["scores"],   # readout only (weight 0)
        corr=corr, frame_vertex_ids=fvids, sub=sub, knobs=knobs, phases=phases,
        scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, fps=fps,
        contact_labels=labels, contact_axis="frames",
        chamfer_weights=chamfer_w, frame_valid=frame_valid, cam_centers=None,
        freeze_fingers=bool(cfg.stage_opts["kinematics_1"].get("freeze_fingers", True)),
        extra_terms=terms, fit_betas=fit_betas)
    print(f"    before: {before}\n    after:  {after}")
    betas_opt = aux["betas"] if fit_betas else betas0
    after_px = pixel_readout(mid, views, targets, corr, fvids, betas_opt, q_opt, thresh)
    with torch.no_grad():
        h1, m1 = (float(x) for x in shape_of(mid, betas_opt))
    print(f"[{tag}] px after     {fmt_readout(after_px)}")
    print(f"[{tag}] height {h0:.3f} -> {h1:.3f} m (target {who['height_m']:.2f}), mass {m0:.1f} -> {m1:.1f} kg "
          f"(target {who['mass_kg']:.1f}); |Δβ| max {float((betas_opt - betas0).abs().max()):.2f}")
    print(f"[{tag}] {root_jitter_cm(q_opt, fps)}")

    out = _new_out()
    _append_classic(out, mid, betas_opt[None].expand(n, -1), q_opt)
    num_cf = len(mid.contact_frame_names)
    extra = dict(
        drift_pre=np.full((1, n, num_cf), np.nan, np.float32),
        auto_labels=np.full((1, n, num_cf), -1, np.int8),
        contact_frame_names=np.array(list(mid.contact_frame_names)),
        betas_init=betas0.cpu().numpy()[None].astype(np.float32), betas_fitted=np.bool_(fit_betas),
        participant=np.str_(who["name"]), shape_source=np.str_(shape_source),
        height_m=np.float32(h1), mass_kg=np.float32(m1),
        method=np.str_("multiview-v2: sapiens + sam3d anchor per calibrated camera"),
        **view_extras(data, views), **readout_extras(before_px, after_px))
    out_path = data["dir"] / "human_optim" / "kinematic_1.npz"
    out_path.parent.mkdir(exist_ok=True)
    _save_body(out_path, mid, out, [betas_opt.cpu().numpy().astype(np.float32)], np.array([0], np.int32),
               valid_any[None], n, 1, left["size"][1], left["size"][0], fps, extra=extra)
    print(f"[{tag}] wrote {out_path}")


# ───────────────────────────── stage: shape ─────────────────────────────

def run_shape(clip: str, device: str) -> None:
    """Freeze one shape for the clip's participant: the mean of the betas fitted so far."""
    who = participant(clip)
    SHAPES.mkdir(exist_ok=True)
    out = SHAPES / f"{who['name']}.npz"
    if out.exists():          # resume: the clips were already refitted with this shape
        print(f"[shape] participant {who['name']}: {out} exists — kept")
        return
    fitted, names = [], []
    for d in sorted(HUMAN_ROOT.iterdir()):
        if d.name[0] not in PARTICIPANTS or participant(d.name) is not who:
            continue
        p = d / "human_optim" / "kinematic_1.npz"
        if p.exists():
            z = np.load(p, allow_pickle=True)
            if bool(z["betas_fitted"]):
                fitted.append(np.asarray(z["betas"][0], np.float32))
                names.append(d.name)
    if not fitted:
        raise SystemExit(f"[shape] no clip of participant {who['name']} has a fitted shape yet")
    betas = np.mean(fitted, 0)
    cfg = load_config(CONFIG)
    mid, *_ = _body_context(cfg.knobs, device)
    with torch.no_grad():
        h, m = (float(x) for x in shape_of(mid, torch.as_tensor(betas, device=device)))
    spread = np.std(fitted, 0).max() if len(fitted) > 1 else 0.0
    np.savez(out, betas=betas, clips=np.array(names), height_m=np.float32(h), mass_kg=np.float32(m),
             target_height_m=np.float32(who["height_m"]), target_mass_kg=np.float32(who["mass_kg"]))
    print(f"[shape] participant {who['name']}: {len(fitted)} clip(s) {names}; betas spread (max std) {spread:.2f}; "
          f"height {h:.3f} m (target {who['height_m']:.2f}), mass {m:.1f} kg (target {who['mass_kg']:.1f}) -> {out}")


# ─────────────────────────── stage: contacts ───────────────────────────

def run_contacts(clip: str, device: str) -> None:
    cfg = load_config(CONFIG)
    boards = gt_section("boards")
    enter_n, exit_n, near_m = float(boards["force_enter_n"]), float(boards["force_exit_n"]), float(boards["near_scene_m"])
    clip_dir = HUMAN_ROOT / clip
    tag = f"contacts[{clip}]"
    body_path = clip_dir / "human_optim" / "kinematic_1.npz"
    kin = dict(np.load(body_path, allow_pickle=True))
    n = int(kin["num_frames"])
    valid = np.asarray(kin["valid_mask"][0], bool)
    limb_on, force_world = board_limb_on(clip_dir, n)
    force_n = np.linalg.norm(force_world, axis=-1)                              # (N, 4)

    mid, *_ = _body_context(cfg.knobs, device)
    frame_names = list(mid.contact_frame_names)
    assert frame_names == [str(x) for x in kin["contact_frame_names"]]
    with torch.no_grad():
        betas = torch.as_tensor(kin["betas"][0], device=device)[None].expand(n, -1)
        q = torch.as_tensor(kin["q"][0], device=device)
        frames = mid.forward(betas, normalize_q(q, mid.num_joints)).frames_world[..., :3].cpu().numpy()
    scene = np.load(clip_dir / "geometry" / "scene.npz")
    dist = cKDTree(np.asarray(scene["points"], np.float64)).query(
        frames.reshape(-1, 3), k=1, workers=-1)[0].reshape(n, -1).astype(np.float32)   # (N, F)

    frame_limb = [limb_of_frame(name) for name in frame_names]
    contact, rescued = board_frame_contact(limb_on, valid, dist, frame_names, near_m)
    per_limb = []
    for i, limb in enumerate(LIMBS):
        cols = [c for c, l in enumerate(frame_limb) if l == limb]
        gaps = dist[:, cols][rescued[:, cols]]
        per_limb.append(f"{limb}={int(limb_on[:, i].sum())}f (far {int(rescued[:, cols].any(1).sum())}f"
                        + (f", gap med {np.median(gaps) * 100:.0f} max {gaps.max() * 100:.0f} cm)" if gaps.size else ")"))
    lab_d = dist[contact] * 100
    print(f"[{tag}] board contact: {' '.join(per_limb)}")
    print(f"[{tag}] labelled frames' distance to the wall: med {np.median(lab_d):.1f} p90 {np.percentile(lab_d, 90):.1f} cm"
          if lab_d.size else f"[{tag}] no contact at all")
    print(f"[{tag}] on frames: " + " ".join(f"{nm}={int(contact[:, c].sum())}" for c, nm in enumerate(frame_names)))

    out_path = clip_dir / "human_optim" / "contacts_1.npz"
    atomic_savez(
        out_path,
        contact_label_schema=np.int32(2),
        contact_label_method=np.str_("boards-v3: measured |F| with hysteresis; per loaded span, on the limb's "
                                     "frames within near_scene_m of the wall for >= 1/3 of the span, else its nearest"),
        frame_contact=contact[None], contact_frame_names=np.array(frame_names),
        frame_parent_joint=np.array([int(x) for x in mid.contact_frame_parents], np.int64),
        frame_vertex=np.array([int(x) for x in mid.contact_frame_vertices], np.int64),
        frame_limb=np.array([l or "" for l in frame_limb]), frame_scene_dist_m=dist[None],
        frame_rescued=rescued[None],
        limb_names=np.array(LIMBS), limb_contact=limb_on[None], limb_force_world=force_world[None],
        limb_force_n=force_n[None], force_enter_n=np.float32(enter_n), force_exit_n=np.float32(exit_n),
        near_scene_m=np.float32(near_m),
        object_ids=np.array([0], np.int32), valid_mask=valid[None], num_people=np.int32(1),
        num_frames=np.int32(n), fps=np.float32(kin["fps"]),
        source_body=np.str_(body_path.name), source_body_sha256=np.str_(sha256(body_path)),
        source_scene_sha256=np.str_(sha256(clip_dir / "geometry" / "scene.npz")),
        source_transform_sha256=np.str_(sha256(clip_dir / "geometry" / "transform.npz")),
        source_sensor_sha256=np.str_(sha256(clip_dir / "sensor_forces.npz")))
    print(f"[{tag}] wrote {out_path}")


# ─────────────────────────── stage: kin2 ───────────────────────────

def run_kin2(clip: str, device: str) -> None:
    """The pipeline's kinematics_2 (contact-patch repair) on the board labels."""
    cfg = load_config(CONFIG)
    knobs = cfg.knobs
    data = load_clip(clip, device)
    views, n, fps = data["views"], data["n"], data["fps"]
    clip_dir = data["dir"]
    tag = f"kin2[{clip}]"
    kin_path = clip_dir / "human_optim" / "kinematic_1.npz"
    kin = dict(np.load(kin_path, allow_pickle=True))
    mid, fvids, corr, sub, chamfer_w = _body_context(knobs, device)
    frame_names = list(mid.contact_frame_names)
    _, frame_contact = load_frame_contacts(clip_dir, "1", kin_path, n, frame_names, tag)
    torch.manual_seed(0)
    view_bodies(mid, views, n, device)
    valid = np.asarray(kin["valid_mask"][0], bool)
    betas = torch.as_tensor(kin["betas"][0], dtype=torch.float32, device=device)
    q_world = torch.as_tensor(kin["q"][0], dtype=torch.float32, device=device).clone()
    q_world[:, 3:] += 1e-6 * torch.randn_like(q_world[:, 3:])
    lab = np.where(frame_contact[0], np.int8(1), np.int8(-1))      # +1 / -1, never 0 (repulsion inert)
    lab[~valid] = -1
    contact_labels = torch.as_tensor(lab, dtype=torch.int8, device=device)
    print(f"[{tag}] {n} frames, labels on {int((lab == 1).any(1).sum())} frames")

    terms, targets = image_terms(views, mid, corr, fvids, knobs, fps, device,
                                 float(gt_section("multiview")["image_margin_px"]))
    wall = load_wall(clip_dir, device)
    if wall is not None:
        terms["wall_penetration"] = wall_penetration_term(wall, sub)
    it, w = cfg.iters["kinematics_2"], view_weights(cfg.weights_for("kinematics_2"), views, view_sigma(data["cameras"]))
    phases = [PhaseSpec("root", w, it.get("root", 0), _ROOT_DOF, True, "adam"),
              PhaseSpec("full", w, it.get("full", 0), _ALL_DOF, True, "adam"),
              PhaseSpec("refine", w, it.get("refine", 0), _ALL_DOF, True, "adam"),
              PhaseSpec("lbfgs", w, it.get("lbfgs", 0), _ALL_DOF, True, "lbfgs")]
    phases = [p for p in phases if p.iters > 0]
    wfc = world_from_cam(torch.stack([v["extr"] for v in views]))
    scene_pts, scene_normals, scene_conf = _load_scene(clip_dir, knobs, wfc, device, tag)
    left = views[0]
    extr_T, K_T = left["extr"][None].expand(n, -1, -1), left["K"][None].expand(n, -1, -1)
    thresh = float(knobs["keypoints"]["score_thresh"])
    before_px = pixel_readout(mid, views, targets, corr, fvids, betas, q_world, thresh)
    q_opt, before, after, _ = optimize_person(
        mid, betas, q_world, extrinsics=extr_T, K=K_T,
        depth_points=[torch.zeros((0, 3), device=device) for _ in range(n)],
        target_kp=targets["cam_left"]["kp"], scores=targets["cam_left"]["scores"],
        corr=corr, frame_vertex_ids=fvids, sub=sub, knobs=knobs, phases=phases,
        scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, fps=fps,
        contact_labels=contact_labels, contact_axis="frames", image_hw=(left["size"][1], left["size"][0]),
        chamfer_weights=chamfer_w, frame_valid=torch.as_tensor(valid, device=device), cam_centers=None,
        extra_terms=terms)
    print(f"    before: {before}\n    after:  {after}")
    after_px = pixel_readout(mid, views, targets, corr, fvids, betas, q_opt, thresh)
    print(f"[{tag}] px before   {fmt_readout(before_px)}\n[{tag}] px after    {fmt_readout(after_px)}")
    print(f"[{tag}] {root_jitter_cm(q_opt, fps)}")

    out = _new_out()
    _append_classic(out, mid, betas[None].expand(n, -1), q_opt)
    extra = {k: kin[k] for k in ("drift_pre", "auto_labels", "contact_frame_names", "participant",
                                 "shape_source", "height_m", "mass_kg")}
    extra.update(source_body=np.str_(kin_path.name), source_body_sha256=np.str_(sha256(kin_path)),
                 source_scene_sha256=np.str_(sha256(clip_dir / "geometry" / "scene.npz")),
                 source_transform_sha256=np.str_(sha256(clip_dir / "geometry" / "transform.npz")),
                 **view_extras(data, views), **readout_extras(before_px, after_px))
    out_path = clip_dir / "human_optim" / "kinematic_2.npz"
    _save_body(out_path, mid, out, [betas.cpu().numpy().astype(np.float32)], np.array([0], np.int32),
               valid[None], n, 1, left["size"][1], left["size"][0], fps, extra=extra)
    print(f"[{tag}] wrote {out_path}")


# ──────────────────────────── stage: kindyn ────────────────────────────

def board_force_term(spec, frame_limb: list, measured: np.ndarray, valid: np.ndarray, device: str):
    """Each limb's summed solved force pulled to the board's measurement (world frame).

    Same normalisation as the base-wrench residual (``0.5 * res_scale * Σ‖·‖²`` over the
    counted rows), so its weight column reads like ``base_wrench``'s. Frames without a
    finite measurement drop out.
    """
    cols = [[c for c, l in enumerate(frame_limb) if l == limb] for limb in LIMBS]
    target = torch.as_tensor(np.nan_to_num(measured), dtype=torch.float32, device=device)     # (T, 4, 3)
    rows = spec.dyn_row * torch.as_tensor(np.isfinite(measured).all(-1).all(-1) & valid, device=device).float()

    def term(st):
        f = st["f_world"]                                                     # (T, C, 3)
        limb = torch.stack([f[:, c].sum(1) for c in cols], 1)                # (T, 4, 3)
        r2 = (limb - target).square().sum(-1).sum(-1)                        # (T,)
        with torch.no_grad():
            info = {"N": ((limb - target).norm(dim=-1) * rows[:, None]).sum() / rows.sum().clamp_min(1.0) / 4.0}
        return 0.5 * spec.res_scale * (r2 * rows).sum(), info
    return term


def run_kindyn(clip: str, device: str) -> None:
    cfg = load_config(CONFIG)
    knobs = cfg.knobs
    kd = cfg.stage_opts["kindyn_1"]
    data = load_clip(clip, device)
    views, n, fps = data["views"], data["n"], data["fps"]
    clip_dir = data["dir"]
    tag = f"kindyn[{clip}]"
    kin1_path = clip_dir / "human_optim" / "kinematic_1.npz"
    body_name = str(kd.get("bodies", "kinematic_2"))
    body_path = clip_dir / "human_optim" / f"{body_name}.npz"
    kin = dict(np.load(body_path, allow_pickle=True))
    if body_name != "kinematic_1":
        for key, src in (("source_body_sha256", kin1_path), ("source_scene_sha256", clip_dir / "geometry" / "scene.npz"),
                         ("source_transform_sha256", clip_dir / "geometry" / "transform.npz")):
            if str(kin[key]) != sha256(src):
                raise SystemExit(f"[{tag}] {body_name} is stale for {src.name} — rerun --stage kin2")
    mid, fvids, corr, sub, chamfer_w = _body_context(knobs, device)
    frame_names = list(mid.contact_frame_names)
    # The labels were read on kinematic_1 (the boards decide, the body only picks the frame).
    _, frame_contact = load_frame_contacts(clip_dir, "1", kin1_path, n, frame_names, tag)
    cnt = dict(np.load(clip_dir / "human_optim" / "contacts_1.npz", allow_pickle=True))
    if str(cnt["source_sensor_sha256"]) != sha256(clip_dir / "sensor_forces.npz"):
        raise SystemExit(f"[{tag}] contacts_1 was made from another sensor_forces.npz — rerun --stage contacts")
    grav = np.load(clip_dir / "geocalib" / "gravity.npz")
    gravity_world = np.asarray(grav["gravity_world"], np.float32)

    torch.manual_seed(0)
    view_bodies(mid, views, n, device)
    valid = np.asarray(kin["valid_mask"][0], bool)
    betas = torch.as_tensor(kin["betas"][0], dtype=torch.float32, device=device)
    q_world = torch.as_tensor(kin["q"][0], dtype=torch.float32, device=device).clone()
    q_world[:, 3:] += 1e-6 * torch.randn_like(q_world[:, 3:])
    lab = np.where(frame_contact[0], np.int8(1), np.int8(-1))
    lab[~valid] = -1
    contact_labels = torch.as_tensor(lab, dtype=torch.int8, device=device)
    print(f"[{tag}] {n} frames from {body_name}, contacts on {int((lab == 1).any(1).sum())} frames, "
          f"gravity {gravity_world.round(3)} ({str(grav['source'])})")

    terms, targets = image_terms(views, mid, corr, fvids, knobs, fps, device,
                                 float(gt_section("multiview")["image_margin_px"]))
    weights = view_weights(cfg.weights_for("kindyn_1"), views, view_sigma(data["cameras"]))
    cones = str(kd.get("cones", "feet"))
    half_angle = float(knobs.get("friction", {}).get("half_angle_deg", 45.0))
    spec = build_dynamics_spec(
        mid, betas, contact_labels, valid, gravity_world, weights=weights,
        dyn_opt=knobs["dynamics_optimizer"], fps=fps, device=device, q_world=q_world,
        cones=cones != "off", half_angle_deg=half_angle, torque_table=knobs["torque"]["multipliers"])
    frame_limb = [limb_of_frame(name) for name in frame_names]
    wall = load_wall(clip_dir, device)
    if wall is not None:
        terms["wall_penetration"] = wall_penetration_term(wall, sub)
    measured_np = np.asarray(cnt["limb_force_world"][0], np.float32)
    terms["board_force"] = board_force_term(spec, frame_limb, measured_np, valid, device)
    if cones == "feet":
        # A hand on a hold can pull in any direction: its four pyramid edges become the
        # vertex directions of a regular tetrahedron, which span every force with
        # non-negative activations — the same parametrization, no cone.
        hand_cols = [c for c, l in enumerate(frame_limb) if l is not None and l.endswith("hand")]
        tetra = torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                             dtype=spec.cone_gen.dtype, device=spec.cone_gen.device) / math.sqrt(3.0)
        spec.cone_gen[:, hand_cols] = tetra
        print(f"[{tag}] cones on the feet only; {len(hand_cols)} hand frames carry a free force "
              "(the printed cone violation counts the hands as if coned — ignore it for them)")
    phases = [PhaseSpec("lbfgs", weights, cfg.iters["kindyn_1"].get("lbfgs", 0), _ALL_DOF, True, "lbfgs")]

    wfc = world_from_cam(torch.stack([v["extr"] for v in views]))
    scene_pts, scene_normals, scene_conf = _load_scene(clip_dir, knobs, wfc, device, tag)
    left = views[0]
    extr_T, K_T = left["extr"][None].expand(n, -1, -1), left["K"][None].expand(n, -1, -1)
    thresh = float(knobs["keypoints"]["score_thresh"])
    q_opt, before, after, aux = optimize_person(
        mid, betas, q_world, extrinsics=extr_T, K=K_T,
        depth_points=[torch.zeros((0, 3), device=device) for _ in range(n)],
        target_kp=targets["cam_left"]["kp"], scores=targets["cam_left"]["scores"],
        corr=corr, frame_vertex_ids=fvids, sub=sub, knobs=knobs, phases=phases,
        scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, fps=fps,
        contact_labels=contact_labels, contact_axis="frames", image_hw=(left["size"][1], left["size"][0]),
        chamfer_weights=chamfer_w, frame_valid=torch.as_tensor(valid, device=device),
        cam_centers=None, dynamics=spec, extra_terms=terms)
    print(f"    before: {before}\n    after:  {after}")
    dyn = aux["dynamics"]
    after_px = pixel_readout(mid, views, targets, corr, fvids, betas, q_opt, thresh)
    print(f"[{tag}] px after    {fmt_readout(after_px)}\n[{tag}] {root_jitter_cm(q_opt, fps)}")

    # Solved force per limb (its frames summed) against the boards.
    limb_force = np.stack([dyn["frame_forces"][:, [c for c, l in enumerate(frame_limb) if l == limb]].sum(1)
                           for limb in LIMBS], 1)                                     # (N, 4, 3)
    measured = np.asarray(cnt["limb_force_world"][0], np.float32)
    enter_n = float(cnt["force_enter_n"])
    size_ours, size_meas = np.linalg.norm(limb_force, axis=-1), np.linalg.norm(measured, axis=-1)
    valid = valid & np.isfinite(size_meas).all(1)                     # compare only where the boards spoke
    both = (size_ours > enter_n) & (size_meas > enter_n) & valid[:, None]
    lines = []
    for i, limb in enumerate(LIMBS):
        m = both[:, i]
        cos = (limb_force[m, i] * measured[m, i]).sum(-1) / (size_ours[m, i] * size_meas[m, i])
        angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()) if m.any() else float("nan")
        mae = float(np.abs(size_ours[valid, i] - size_meas[valid, i]).mean()) if valid.any() else float("nan")
        lines.append(f"{limb}: MAE {mae:.0f} N, angle {angle:.0f} deg on {int(m.sum())} frames, "
                     f"mean ours/measured {size_ours[valid, i].mean():.0f}/{size_meas[valid, i].mean():.0f} N")
    tot = np.abs(np.linalg.norm(limb_force.sum(1), axis=-1) - np.linalg.norm(measured.sum(1), axis=-1))[valid].mean()
    print(f"[{tag}] solved vs measured per limb:\n    " + "\n    ".join(lines) + f"\n    total |F| MAE {tot:.0f} N")
    dyn_rows = spec.dyn_row.bool().cpu().numpy()
    fc = force_confidence(dyn["base_wrench"], dyn["frame_forces"], dyn_rows, float(dyn["total_mass"]) * _GRAVITY_MAG)

    out = _new_out()
    _append_classic(out, mid, betas[None].expand(n, -1), q_opt)
    jc = np.zeros((n, mid.num_joints), bool)
    for c, jb in enumerate(spec.parent_body):
        jc[:, jb] |= lab[:, c] == 1
    extra = dict(
        contact_forces_world=dyn["force_world"][None], torque_mag=dyn["torque_mag"][None],
        base_wrench=dyn["base_wrench"][None], joints_world=dyn["joints_world"][None],
        joint_torques=dyn["joint_torques"][None], frame_forces=dyn["frame_forces"][None],
        frame_contact=(lab == 1)[None], contact_frame_names=np.array(frame_names),
        contact_frame_parents=np.array([int(x) for x in mid.contact_frame_parents], np.int64),
        joint_contact=jc[None], total_mass=np.array([dyn["total_mass"]], np.float32),
        gravity_world=gravity_world,
        base_wrench_diagnostics=np.array([[dyn[k] for k in ("base_wrench_before", "base_wrench_warm",
                                                             "base_wrench_after", "cone_violation_max_n")]], np.float32),
        force_confidence=fc["confidence"][None], force_confidence_clip=np.array([fc["clip"]], np.float32),
        cones=np.str_(cones), cone_half_angle_deg=np.float32(half_angle),
        source_bodies=np.array(body_name, dtype="U16"), source_labels=np.array("boards", dtype="U8"),
        limb_names=np.array(LIMBS), limb_force_world=limb_force[None].astype(np.float32),
        limb_force_measured=measured[None], limb_contact_measured=np.asarray(cnt["limb_contact"]),
        drift_pre=kin["drift_pre"], auto_labels=kin["auto_labels"],
        participant=kin["participant"], height_m=kin["height_m"], mass_kg=kin["mass_kg"],
        **view_extras(data, views),
        reproj_sapiens_px_after=np.array([after_px[v][0] for v in VIEW_NAMES], np.float32),
        reproj_anchor_px_after=np.array([after_px[v][1] for v in VIEW_NAMES], np.float32))
    out_path = clip_dir / "human_optim" / "kindyn_1.npz"
    _save_body(out_path, mid, out, [betas.cpu().numpy().astype(np.float32)], np.array([0], np.int32),
               valid[None], n, 1, left["size"][1], left["size"][0], fps, extra=extra)
    print(f"[{tag}] wrote {out_path}")


# ───────────────────────────── stage: sync ─────────────────────────────

SYNC_OFFSETS = range(-4, 5)
#: Joints whose image motion makes a one-frame lag visible: wrists, ankles, head, elbows, knees.
SYNC_JOINTS = ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "head",
               "left_elbow", "right_elbow", "left_knee", "right_knee")


def fundamental(extr_a: np.ndarray, K_a: np.ndarray, extr_b: np.ndarray, K_b: np.ndarray) -> np.ndarray:
    """Fundamental matrix ``F`` with ``x_b^T F x_a = 0`` for pixels of cameras a and b."""
    rel = extr_b @ np.linalg.inv(extr_a)                       # camera b from camera a
    R, t = rel[:3, :3], rel[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return np.linalg.inv(K_b).T @ tx @ R @ np.linalg.inv(K_a)


def epipolar_px(px_a: np.ndarray, px_b: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Symmetric epipolar distance (px) per point pair, ``(..., 2)`` pixels each."""
    ha = np.concatenate([px_a, np.ones_like(px_a[..., :1])], -1)
    hb = np.concatenate([px_b, np.ones_like(px_b[..., :1])], -1)
    lb = ha @ F.T                                                # line in b for a point of a
    la = hb @ F                                                  # line in a for a point of b
    val = np.abs((hb * lb).sum(-1))
    return 0.5 * val * (1.0 / np.linalg.norm(lb[..., :2], axis=-1) + 1.0 / np.linalg.norm(la[..., :2], axis=-1))


def run_sync(clip: str, device: str, apply: bool) -> None:
    """Measure each view's lag behind cam_left from the SAM-3D-Body 2D joints.

    For a lagging view, the epipolar residual between cam_left's joint pixels at frame ``t``
    and the view's at ``t + d`` is smallest at the true lag ``d`` — a depth-free test, so the
    monocular depth errors of the estimates do not enter. The curves are shallow on real
    clips (the monocular joint noise of 3-12 px dwarfs a one-frame move), so what gets
    APPLIED is the chessboard calibration's own offset for the clip's camera set (measured
    to sub-pixel on the board: 00 [+2, +2], 16/32 [+1, +2] for cam_right/cam_3); the clip's
    curves are printed next to it as a consistency check and stored in ``cameras.npz``.
    Run BEFORE the fit.
    """
    from prepare_human import CALIB_REF, GT_SCENE

    cfg = load_config(CONFIG)
    clip_dir = HUMAN_ROOT / clip
    cams = dict(np.load(clip_dir / "cameras.npz", allow_pickle=True))
    data = load_clip(clip, device, frame_offset=np.zeros(3, int))    # measure on the raw clock
    views, n = data["views"], data["n"]
    mid, *_ = _body_context(cfg.knobs, device)
    view_bodies(mid, views, n, device)
    jidx = torch.tensor([list(mid.joint_names).index(j) for j in SYNC_JOINTS], device=device)
    px = [project_view(v["joints"][:, jidx], v["extr"], v["K"]).cpu().numpy() for v in views]
    valid = [v["valid"] for v in views]
    K = [v["K"].cpu().numpy().astype(np.float64) for v in views]
    E = [v["extr"].cpu().numpy().astype(np.float64) for v in views]
    offsets = np.zeros(3, np.int32)
    curves = {}
    for k in (1, 2):
        F = fundamental(E[0], K[0], E[k], K[k])
        curve = {}
        for d in SYNC_OFFSETS:
            a = slice(max(0, -d), n - max(0, d))                 # cam_left frames t
            b = slice(max(0, d), n - max(0, -d))                 # view k frames t + d
            ok = valid[0][a] & valid[k][b]
            curve[d] = float(np.median(epipolar_px(px[0][a][ok], px[k][b][ok], F)))
        best = min(curve, key=curve.get)
        offsets[k] = best
        curves[views[k]["name"]] = curve
        print(f"[sync[{clip}]] {views[k]['name']} vs cam_left: median epipolar px per lag "
              + " ".join(f"{d:+d}:{curve[d]:.1f}" for d in SYNC_OFFSETS) + f"  -> measured lag {best:+d}")
    set_name = str(cams["camera_set"])
    calib = np.load(GT_SCENE / CALIB_REF.get(set_name, set_name) / "calibration.npz")
    calib_off = np.array([0, *np.asarray(calib["frame_offset"], int)], np.int32)   # [left, right, cam_3]
    print(f"[sync[{clip}]] calibration lag {calib_off.tolist()} (set {set_name}); measured {offsets.tolist()}")
    if apply:
        cams["frame_offset"] = calib_off
        cams["frame_offset_measured"] = offsets
        cams["frame_offset_curves_px"] = np.array([[curves[v][d] for d in SYNC_OFFSETS] for v in ("cam_right", "cam_3")], np.float32)
        cams["frame_offset_lags"] = np.array(list(SYNC_OFFSETS), np.int32)
        np.savez(clip_dir / "cameras.npz", **cams)
        print(f"[sync[{clip}]] wrote frame_offset {calib_off.tolist()} into cameras.npz")


# ───────────────────────────── stage: cam3 ─────────────────────────────

CAM3_MIN_SCORE = 0.6       # a Sapiens keypoint votes only when confident in every camera involved
CAM3_MAX_EPI_PX = 4.0      # ... and consistent between the calibrated front pair (epipolar residual)


def run_cam3(clip: str, device: str) -> None:
    """Re-solve cam_3's pose for this clip from the climber (PnP on triangulated keypoints).

    The front pair is chessboard-calibrated and never moved; cam_3 was knocked between
    recordings (sets 16b/16c have no chessboard pose at all). Its intrinsics are the
    calibrated ones, so its pose follows from the climber: Sapiens keypoints confident and
    epipolar-consistent in cam_left + cam_right are triangulated (metric), and cam_3's
    confident keypoints of the same (frame, keypoint) give a PnP problem. RANSAC + LM
    refinement, started from the stored pose. The new pose is written into
    ``cameras.npz`` only when it lowers cam_3's median reprojection error; the previous
    pose is kept as ``extrinsics_before_pnp``. Run after ``sync``, before ``fit``.
    """
    import cv2

    clip_dir = HUMAN_ROOT / clip
    tag = f"cam3[{clip}]"
    data = load_clip(clip, device)
    views, n = data["views"], data["n"]
    cams = dict(np.load(clip_dir / "cameras.npz", allow_pickle=True))
    if "extrinsics_before_pnp" not in cams:          # idempotent: always solve from the stored calibration pose
        cams["extrinsics_before_pnp"] = cams["extrinsics"].copy()
        cams["cam3_pose_source_before_pnp"] = cams["cam3_pose_source"]
    K = [v["K"].cpu().numpy().astype(np.float64) for v in views]
    E = [np.asarray(cams["extrinsics_before_pnp"][k], np.float64) for k in range(3)]
    kp = [v["kp"].astype(np.float64) for v in views]
    sc = [v["kp_scores"] for v in views]
    F_lr = fundamental(E[0], K[0], E[1], K[1])
    conf = (sc[0] >= CAM3_MIN_SCORE) & (sc[1] >= CAM3_MIN_SCORE) & (sc[2] >= CAM3_MIN_SCORE)
    t_idx, k_idx = np.nonzero(conf)
    epi = epipolar_px(kp[0][t_idx, k_idx], kp[1][t_idx, k_idx], F_lr)
    keep = epi <= CAM3_MAX_EPI_PX
    t_idx, k_idx = t_idx[keep], k_idx[keep]
    if len(t_idx) < 200:
        raise SystemExit(f"[{tag}] only {len(t_idx)} usable keypoints — cannot solve cam_3")
    P0, P1 = K[0] @ E[0][:3], K[1] @ E[1][:3]
    X = cv2.triangulatePoints(P0, P1, kp[0][t_idx, k_idx].T, kp[1][t_idx, k_idx].T)
    X = (X[:3] / X[3]).T                                                     # (M, 3) world, metres
    x3 = kp[2][t_idx, k_idx]

    def reproj_px(extr: np.ndarray) -> np.ndarray:
        cam = X @ extr[:3, :3].T + extr[:3, 3]
        uv = cam[:, :2] / cam[:, 2:3]
        return np.linalg.norm(uv * [K[2][0, 0], K[2][1, 1]] + [K[2][0, 2], K[2][1, 2]] - x3, axis=1)

    before = reproj_px(E[2])
    rvec0, _ = cv2.Rodrigues(E[2][:3, :3])
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        X.astype(np.float64), x3.astype(np.float64), K[2], None, rvec=rvec0.copy(), tvec=E[2][:3, 3].copy(),
        useExtrinsicGuess=True, iterationsCount=3000, reprojectionError=8.0, confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 100:
        raise SystemExit(f"[{tag}] PnP failed ({0 if inl is None else len(inl)} inliers)")
    inl = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(X[inl], x3[inl], K[2], None, np.asarray(rvec, np.float64).reshape(3, 1), np.asarray(tvec, np.float64).reshape(3, 1))
    new = np.eye(4)
    new[:3, :3] = cv2.Rodrigues(rvec)[0]
    new[:3, 3] = np.asarray(tvec).reshape(3)
    after = reproj_px(new)
    d_rot = np.degrees(np.arccos(np.clip((np.trace(new[:3, :3].T @ E[2][:3, :3]) - 1) / 2, -1, 1)))
    c_old, c_new = -E[2][:3, :3].T @ E[2][:3, 3], -new[:3, :3].T @ new[:3, 3]
    print(f"[{tag}] {len(t_idx)} keypoints over {len(np.unique(t_idx))} frames, {len(inl)} PnP inliers; "
          f"cam_3 reprojection px median {np.median(before):.1f} -> {np.median(after):.1f} "
          f"(p90 {np.percentile(before, 90):.1f} -> {np.percentile(after, 90):.1f}); "
          f"pose moved {d_rot:.2f} deg, {np.linalg.norm(c_new - c_old) * 1000:.0f} mm")
    # Each camera's keypoint consistency, for the fit's camera weights: the front pair's
    # epipolar residual over all confident keypoints (both get it), cam_3's reprojection
    # residual of the triangulated points.
    front = float(np.median(epi))
    cams["view_sigma_px"] = np.array([front, front, float(min(np.median(before), np.median(after)))], np.float32)
    ext = cams["extrinsics_before_pnp"].copy()
    if np.median(after) < np.median(before):
        ext[2] = new.astype(np.float32)
        cams["cam3_pose_source"] = np.str_(f"pnp-sapiens (from {str(cams['cam3_pose_source_before_pnp'])})")
        cams["cam3_pnp_px"] = np.array([np.median(before), np.median(after)], np.float32)
        print(f"[{tag}] writing the refined cam_3 pose into cameras.npz")
    else:
        cams["cam3_pose_source"] = cams["cam3_pose_source_before_pnp"]
        cams.pop("cam3_pnp_px", None)
        print(f"[{tag}] no improvement — the calibration pose stays")
    cams["extrinsics"] = ext
    np.savez(clip_dir / "cameras.npz", **cams)


STAGES = {"sync": None, "cam3": run_cam3, "fit": run_fit, "shape": run_shape, "contacts": run_contacts,
          "kin2": run_kin2, "kindyn": run_kindyn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-apply", action="store_true", help="sync: only print the lag curves")
    args = ap.parse_args()
    if args.stage == "sync":
        run_sync(args.clip, args.device, apply=not args.no_apply)
    else:
        STAGES[args.stage](args.clip, args.device)


if __name__ == "__main__":
    main()
