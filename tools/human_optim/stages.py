"""The kinematic stage runners: root / kinematics_1 / kinematics_2.

One place for the per-stage orchestration the ``human_optimization`` dispatcher
drives. The three *kinematic* stages share the same machinery — load the inputs,
per person build the world-frame ``q``, run :func:`~tools.human_optim.optimizer.optimize_person`
over a phase list, write the refined-body npz — differing only in where ``q`` starts
(the scale stage's pre-fit — converter when absent — vs the previous refinement),
the weight columns their phases use, and (for
``kinematics_2``) the manual contact annotation. The force-only ``dynamics`` stage
lives in :mod:`tools.human_optim.dynamics`.

The caller (``scripts/stages/human_optimization.py``) has already put
``better_human`` on ``sys.path``; this module assumes it. Weights + knobs come from a
:class:`~tools.human_optim.config.HumanOptimConfig`; outputs are placed by
:mod:`tools.human_optim.paths`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from tools.geometry.normals import load_scene_cloud
from tools.smplx_robot import load_smplx_mid
from tools.smplx_robot.vertex_weights import geodesic_endeffector_weights

from . import paths
from .camera import fold_root_to_world, project, transform_points, world_from_cam
from .config import HumanOptimConfig
from .keypoints import build_correspondence, visibility_weights
from .optimizer import PhaseSpec, optimize_person
from .prefit import smooth_finger_quats
from .qmath import normalize_q
from .visibility import subsample_indices

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SMPLX_MODEL = _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
_FINGER = ("index", "middle", "pinky", "ring", "thumb")
_ALL_DOF = ("transl", "orient", "joints")
_ROOT_DOF = ("transl", "orient")


# ─────────────────────────────── io helpers ──────────────────────────────

# E1 limb-quota sampling: the four extremity groups whose depth evidence the uniform
# stride starves (torso pixels vastly outnumber hand/foot pixels). Group id 0 = rest.
_LIMB_GROUPS = ("left_hand", "right_hand", "left_foot", "right_foot")


def _limb_group_ids(joint_names: list[str]) -> np.ndarray:
    """``(J,)`` int group id per SMPLXMid joint: 0 = rest, 1.. = :data:`_LIMB_GROUPS`
    (hand = wrist + fingers, foot = ankle + foot — the ``_ANN_TO_SMPLX`` groups)."""
    gid = np.zeros(len(joint_names), np.int64)
    idx = {n: i for i, n in enumerate(joint_names)}
    for g, gname in enumerate(_LIMB_GROUPS, start=1):
        for jn in _ANN_TO_SMPLX[gname]:
            if jn in idx:
                gid[idx[jn]] = g
    return gid


def _point_group_ids(points: np.ndarray, extr_t: np.ndarray, K_t: np.ndarray,
                     joint_px_t: np.ndarray, joint_group: np.ndarray) -> np.ndarray:
    """``(n,)`` group id per world point: nearest projected init joint in PIXEL space.

    The human-cloud points come from this frame's own depth map, so they all project
    in front of this frame's camera; assignment in pixels (not 3D) is deliberate — a
    depth streak displaces a point along the ray, which barely moves its pixel, so
    the streaked point still lands on its true limb group.
    """
    cam = points @ extr_t[:3, :3].T + extr_t[:3, 3]                 # (n, 3) camera frame
    uvh = cam @ K_t.T
    uv = uvh[:, :2] / np.clip(uvh[:, 2:3], 1e-6, None)              # (n, 2) pixels
    d2 = ((uv[:, None, :] - joint_px_t[None, :, :]) ** 2).sum(-1)   # (n, J)
    return joint_group[np.argmin(d2, axis=1)]


def _limb_quota_select(p: np.ndarray, c: np.ndarray, g: np.ndarray, conf_drop_q: float,
                       max_points: int, limb_quota: int) -> np.ndarray:
    """One frame's ``sampling='limb_quota'`` selection: group-aware gate + limb quotas.

    The ``conf_drop_q`` quantile gate runs **within each group** (a global per-frame
    gate eats the depth-edge limbs first — limb evidence would be destroyed before any
    quota could act). Then each limb group keeps up to ``limb_quota`` points (uniform
    stride; everything if smaller) and the rest group fills the remaining
    ``max_points`` budget by uniform stride. Total never exceeds ``max_points``.
    Deterministic; kept points stay in original order.
    """
    n_groups = len(_LIMB_GROUPS) + 1
    kept = np.ones(len(p), bool)
    if conf_drop_q > 0.0:
        for gi in range(n_groups):
            m = g == gi
            if m.any():
                kept &= ~m | (c >= np.quantile(c[m], conf_drop_q))
    idx = np.flatnonzero(kept)
    sel, limb_total = [], 0
    for gi in range(1, n_groups):
        gidx = idx[g[idx] == gi]
        if len(gidx) > limb_quota:
            gidx = gidx[np.linspace(0, len(gidx) - 1, limb_quota).astype(int)]
        sel.append(gidx)
        limb_total += len(gidx)
    rest = idx[g[idx] == 0]
    budget = max(0, max_points - limb_total)
    if len(rest) > budget:
        rest = rest[np.linspace(0, len(rest) - 1, budget).astype(int)] if budget else rest[:0]
    sel.append(rest)
    out = np.sort(np.concatenate(sel))
    if len(out) > max_points:            # only reachable when 4·limb_quota > max_points
        out = out[np.linspace(0, len(out) - 1, max_points).astype(int)]
    return p[out]


def _person_depth_points(human: dict, oid: int, N: int, conf_drop_q: float,
                         max_points: int, device, *,
                         sampling: str = "stride", limb_quota: int = 200,
                         joint_px: np.ndarray | None = None,
                         joint_group: np.ndarray | None = None,
                         extrinsics: np.ndarray | None = None,
                         intrinsics: np.ndarray | None = None) -> list[torch.Tensor]:
    """Per-frame world-frame depth points for one person (conf-gated, capped).

    The confidence gate is a **per-frame quantile**: within each frame, drop that
    frame's lowest ``conf_drop_q`` fraction of points by VGGT depth-confidence. VGGT
    confidence has no fixed scale (≥~1 by construction) so an absolute floor never
    fires. A *per-clip* quantile would instead pile its whole drop onto the frames
    whose confidence sits systematically lower (VGGT's per-frame median drifts up to
    ~4× within a clip — measured on sport_climb — so a clip-wide q=0.3 empties >90 %
    of some frames while barely touching others, and the emptied frames keep only
    reprojection to hold the body). Gating per frame keeps ``1 − conf_drop_q`` of
    *every* frame by construction, matching ``clean_depth`` / ``estimate_scale``.
    The human cloud gets no confidence gate at fuse, so this is its only quality
    filter before chamfer.

    ``sampling='stride'`` (default) caps each frame by a uniform stride after the
    global per-frame gate — the legacy path, bit-exact. ``'limb_quota'`` assigns each
    point to a body-part group first (:func:`_point_group_ids`, nearest projected
    init joint in pixel space via ``joint_px (N, J, 2)`` / ``joint_group (J,)`` /
    the per-frame cameras), then gates + caps group-aware
    (:func:`_limb_quota_select`) so each hand/foot keeps up to ``limb_quota`` points.
    """
    if sampling not in ("stride", "limb_quota"):
        raise ValueError(f"[chamfer] sampling must be 'stride' or 'limb_quota', got {sampling!r}")
    if sampling == "limb_quota" and 4 * limb_quota > max_points:
        raise ValueError(
            f"[chamfer] 4*limb_quota ({4 * limb_quota}) exceeds max_points ({max_points}) — "
            f"the limb quotas would be globally re-strided (limbs can vanish); lower limb_quota "
            f"or raise max_points")
    use_quota = (sampling == "limb_quota" and joint_px is not None
                 and joint_group is not None and extrinsics is not None and intrinsics is not None)
    person = human["obj_id"] == oid
    pts = human["points"][person]
    conf = human["confidence"][person]
    fr = human["frame_id"][person]
    order = np.argsort(fr, kind="stable")
    pts, conf, fr = pts[order], conf[order], fr[order]
    bnd = np.searchsorted(fr, np.arange(N + 1))
    out = []
    for t in range(N):
        p, c = pts[bnd[t]:bnd[t + 1]], conf[bnd[t]:bnd[t + 1]]
        if use_quota and len(p):
            g = _point_group_ids(p, extrinsics[t], intrinsics[t], joint_px[t], joint_group)
            p = _limb_quota_select(p, c, g, conf_drop_q, max_points, limb_quota)
        else:
            if conf_drop_q > 0.0 and c.size:
                p = p[c >= np.quantile(c, conf_drop_q)]   # per-frame: keep the top 1−q of THIS frame
            if len(p) > max_points:
                p = p[np.linspace(0, len(p) - 1, max_points).astype(int)]
        out.append(torch.as_tensor(p, dtype=torch.float32, device=device))
    return out


def _sapiens_for(pose: dict, oid: int, N: int):
    """Return ``(keypoints (N,308,2), scores (N,308))`` for the person with id ``oid``."""
    ids = [int(x) for x in pose["object_ids"]]
    p = ids.index(oid)
    return pose["keypoints"][p], pose["keypoint_scores"][p]


def _fill_invalid_frames(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace each invalid-frame row of ``arr`` (T, …) with its nearest valid frame.

    The converter writes NaN/Inf params for untracked (``valid_mask=False``) frames;
    the init forward pass can't pose those (``mat2SE3`` rejects the non-finite
    rotation). Holding the nearest valid pose across a gap keeps the trajectory finite
    and flat there (no smoothness spike into valid neighbours). A no-op when every
    frame is valid, so fully-tracked clips are untouched.
    """
    if valid.all():
        return arr
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.nan_to_num(arr)
    nearest = idx[np.abs(np.arange(len(arr))[:, None] - idx[None, :]).argmin(1)]
    return arr[nearest]


def _init_params(pred: Path, sp: dict, object_ids: np.ndarray, tag: str) -> tuple[dict, str]:
    """The parameter source ``(params, name)`` the kinematic fit starts from.

    The scale stage's pre-fit bodies (``geometry/prefit.npz``) when present —
    depth-aligned and reprojection-refined, so the priors and the init-projection
    anchor pin to a better pose than the raw converter's. The file is row-aligned
    with ``smplx_params.npz`` (converter values on frames the pre-fit did not
    touch) and stores the same classic params, so it swaps in directly; a clip
    whose scale ran ruler-only (no pre-fit) keeps the converter params.
    """
    prefit_path = pred / "geometry" / "prefit.npz"
    if not prefit_path.exists():
        return sp, "converter"
    pf = dict(np.load(prefit_path, allow_pickle=True))
    if int(pf["num_frames"]) != int(sp["num_frames"]) or not np.array_equal(
            np.asarray(pf["object_ids"], dtype=np.int32),
            np.asarray(object_ids, dtype=np.int32)):
        raise SystemExit(f"[{tag}] {prefit_path} does not match sam3d/smplx_params.npz — "
                         "re-run estimate_scale")
    return pf, "prefit"


def _load_scene(pred: Path, knobs: dict, wfc: torch.Tensor, device, tag: str):
    """Scene cloud + camera-oriented PCA normals for the penetration/attraction terms.

    Returns ``(points, normals, confidence)`` or ``(None, None, None)`` when the
    clip has no ``geometry/scene.npz`` (the stage then runs without scene terms)."""
    scene_path = pred / "geometry" / "scene.npz"
    if not scene_path.exists():
        print(f"[{tag}] no geometry/scene.npz; scene terms OFF")
        return None, None, None
    sc = dict(np.load(scene_path, allow_pickle=True))
    # Quantile confidence gate + fuse-stored normals (PCA fallback for old
    # artifacts) — shared with estimate_contacts / score_contacts.
    scene_pts, scene_normals, scene_conf = load_scene_cloud(
        sc, wfc[:, :3, 3], conf_drop_q=float(knobs["penetration"].get("conf_drop_q", 0.0)),
        k=int(knobs["penetration"]["normal_k"]), device=device)
    src = "fuse" if "normals" in sc else "PCA fallback"
    print(f"[{tag}] scene terms ON ({scene_pts.shape[0]} pts, normals: {src})")
    return scene_pts, scene_normals, scene_conf


def _body_context(knobs: dict, device: str):
    """The per-clip body machinery both kinematic stages build identically.

    Returns ``(mid, fvids, corr, sub, chamfer_w)``: the SMPLXMid model, its frame
    landmark vertex ids, the Goliath-308 correspondence, the chamfer vertex
    subsample, and the per-vertex extremity-emphasis chamfer weights (geodesic
    linear, log-mapped to ``[lo, hi]`` over ``radius`` m — defaults reproduce the
    base field).
    """
    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    fvids = torch.as_tensor([int(x) for x in mid.frame_vertex_ids], dtype=torch.long, device=device)
    corr = build_correspondence(list(mid.frame_names),
                                include_hands=bool(knobs["keypoints"]["include_hands"]), device=device)
    sub = subsample_indices(mid.num_vertices, int(knobs["chamfer"]["vertex_subsample"]), device=device)
    chamfer_w = torch.as_tensor(
        geodesic_endeffector_weights(
            mid.vertices_template, mid.faces, mid.lbs_weights, mid.joint_names,
            lo=float(knobs["chamfer"].get("emphasis_lo", 0.1)),
            hi=float(knobs["chamfer"].get("emphasis_hi", 10.0)),
            radius=float(knobs["chamfer"].get("emphasis_radius", 1.0))),
        dtype=torch.float32, device=device)[sub]                                   # (Ns,)
    return mid, fvids, corr, sub, chamfer_w


def _person_visual_inputs(pose: dict, human: dict, oid: int, N: int, vmask: np.ndarray,
                          knobs: dict, device: str, *, mid, betas_T: torch.Tensor,
                          q_world: torch.Tensor, extr: torch.Tensor, K: torch.Tensor):
    """One person's 2D + depth evidence, invalid frames blanked.

    Returns ``(target_kp, scores, depth, frame_valid)``: Sapiens keypoints +
    scores, the per-frame chamfer depth clouds (conf-gated, capped — with the
    ``limb_quota`` grouping when configured, referenced to this ``q_world``'s
    projected joints), and the validity mask. Untracked frames get zero scores
    and an empty cloud so no image/depth term can read a held pose as evidence.
    """
    kp_np, sc_np = _sapiens_for(pose, oid, N)
    target_kp = torch.as_tensor(kp_np, dtype=torch.float32, device=device)
    scores = torch.as_tensor(sc_np, dtype=torch.float32, device=device)
    sampling = str(knobs["chamfer"].get("sampling", "stride"))
    jpx = jgrp = extr_np = K_np = None
    if sampling == "limb_quota":   # project the init body's joints once — the grouping reference
        with torch.no_grad():
            jw0 = mid.forward(betas_T, normalize_q(q_world, mid.num_joints)
                              ).joints_world.tensor()[..., :3]
            jpx = project(transform_points(jw0, extr), K).cpu().numpy()
        jgrp = _limb_group_ids(list(mid.joint_names))
        extr_np, K_np = extr.cpu().numpy(), K.cpu().numpy()
    depth = _person_depth_points(human, oid, N, float(knobs["chamfer"]["conf_drop_q"]),
                                 int(knobs["chamfer"]["max_points"]), device,
                                 sampling=sampling,
                                 limb_quota=int(knobs["chamfer"].get("limb_quota", 200)),
                                 joint_px=jpx, joint_group=jgrp,
                                 extrinsics=extr_np, intrinsics=K_np)
    for i in range(N):
        if not vmask[i]:
            scores[i] = 0.0
            depth[i] = depth[i][:0]
    frame_valid = torch.as_tensor(vmask, dtype=torch.bool, device=device)
    return target_kp, scores, depth, frame_valid


def _new_out() -> dict[str, list]:
    return {k: [] for k in ("global_orient", "transl", "body_pose",
                            "left_hand_pose", "right_hand_pose", "q")}


def _append_classic(out: dict, mid, betas_T: torch.Tensor, q_opt: torch.Tensor) -> None:
    """Convert an optimized ``q`` back to classic SMPL-X params and append to ``out``."""
    cls = mid.to_classic(betas_T, q_opt)
    key_map = {"global_orient": "global_orient", "transl": "translation",
               "body_pose": "body_pose", "left_hand_pose": "left_hand_pose",
               "right_hand_pose": "right_hand_pose"}
    for out_k, cls_k in key_map.items():
        out[out_k].append(cls[cls_k].cpu().numpy().astype(np.float32))
    out["q"].append(q_opt.cpu().numpy().astype(np.float32))


def _save_body(out_path: Path, mid, out: dict, betas_out: list, object_ids: np.ndarray,
               valid_mask: np.ndarray, N: int, P: int, H: int, W: int, fps: float,
               extra: dict | None = None) -> dict:
    """Assemble + write the refined-body npz (the shared kinematic_1 / kinematic_2 schema).

    Body optimized at native (converter) size — no per-body scale now that the metric
    geometry is human-anchored upstream. ``scale`` is kept as ones so the consumers that
    re-apply it about the pelvis stay a no-op. ``extra`` (optional) is merged in as extra
    arrays — ``kinematics_1`` uses it to persist ``drift_pre`` / ``auto_labels``.
    """
    nb = mid.num_joints
    parents = [-1] + [int(mid.parent_tree[0, i]) for i in range(1, nb)]
    save = {k: np.stack(v, 0) for k, v in out.items()}
    save.update(
        betas=np.stack(betas_out, 0),
        scale=np.ones(P, np.float32),
        jaw_pose=np.zeros((P, N, 3), np.float32),
        leye_pose=np.zeros((P, N, 3), np.float32),
        reye_pose=np.zeros((P, N, 3), np.float32),
        expression=np.zeros((P, N, 10), np.float32),
        joint_names=np.array(list(mid.joint_names), dtype="U16"),
        parents=np.array(parents, dtype=np.int32),
        object_ids=object_ids,
        valid_mask=valid_mask,
        num_people=np.int32(P), num_frames=np.int32(N),
        video_height=np.int32(H), video_width=np.int32(W),
        fps=np.float32(fps), frame=np.array("world", dtype="U8"),
        model_type=np.array("smplx_mid", dtype="U16"), num_betas=np.int32(10),
    )
    if extra:
        save.update(extra)
    np.savez(out_path, **save)
    return save


# ─────────────────────────────── phase plans ─────────────────────────────

def _kinematic_phases(cfg: HumanOptimConfig, stages_to_run) -> list[PhaseSpec]:
    """Phases for the init-anchored ``root`` / ``kinematics_1`` pass.

    ``root`` (root SE3 only, no scene) uses the root column; ``kinematics_1``'s
    ``full`` → ``refine`` → ``lbfgs`` (all DOFs; scene on from refine) use the
    kinematics_1 column — all in one call so the priors stay anchored to the
    starting pose (:func:`_init_params`: the scale pre-fit, else the converter).
    """
    phases: list[PhaseSpec] = []
    if "root" in stages_to_run:
        it = cfg.iters["root"]
        phases.append(PhaseSpec("root", cfg.weights_for("root"), it.get("root", 0), _ROOT_DOF, False, "adam"))
    if "kinematics_1" in stages_to_run:
        it = cfg.iters["kinematics_1"]
        w = cfg.weights_for("kinematics_1")
        phases.append(PhaseSpec("full", w, it.get("full", 0), _ALL_DOF, False, "adam"))
        phases += [
            PhaseSpec("refine", w, it.get("refine", 0), _ALL_DOF, True, "adam"),
            PhaseSpec("lbfgs", w, it.get("lbfgs", 0), _ALL_DOF, True, "lbfgs"),
        ]
    return phases


def _kinematics_2_phases(cfg: HumanOptimConfig) -> list[PhaseSpec]:
    """Phases for ``kinematics_2`` (all use the kinematics_2 column).

    Scene is on from the ``root`` phase so the contact-patch repair (ann_acquisition +
    ann_seating) acts from the start — the rigid ``root`` phase does the bulk push toward
    the wall, then ``full`` → ``refine`` re-settle the pose. The scene candidates and the
    per-instance target patches both re-prune at every phase start (from the current pose).
    L-BFGS is off for kinematics_2 (its iters default to 0): the per-phase contact patches
    make an active set whose churn invalidates strong-Wolfe curvature. 0-iter phases are
    dropped so their (discarded) on_start candidate + patch build never runs."""
    it = cfg.iters["kinematics_2"]
    w = cfg.weights_for("kinematics_2")
    specs = [
        PhaseSpec("root", w, it.get("root", 0), _ROOT_DOF, True, "adam"),
        PhaseSpec("full", w, it.get("full", 0), _ALL_DOF, True, "adam"),
        PhaseSpec("refine", w, it.get("refine", 0), _ALL_DOF, True, "adam"),
        PhaseSpec("lbfgs", w, it.get("lbfgs", 0), _ALL_DOF, True, "lbfgs"),
    ]
    return [s for s in specs if s.iters > 0]


# ─────────────────────── kinematic stages (root / kin1) ───────────────────

def run_kinematic(source, pred: Path, cfg: HumanOptimConfig, stages_to_run,
                  device: str, *, overlay: bool = False, labels_mode: str = "auto",
                  freeze_fingers: bool = True) -> Path:
    """Run ``root`` and/or ``kinematics_1`` for every tracked person → ``human_optim/``.

    ``q`` starts from the scale stage's pre-fit bodies when ``geometry/prefit.npz``
    exists (else the converter params) — see :func:`_init_params`; the priors and
    the init-projection anchor pin to that starting pose. Writes ``kinematic_1.npz``
    when ``kinematics_1`` is in ``stages_to_run``, else the intermediate ``root.npz``
    (root-only placement).

    ``labels_mode`` must be ``'auto'``: the frozen contact labels are detected
    geometrically on the 12 contact frames inside the optimization (the
    ``[auto_contact]`` operating point).

    ``freeze_fingers`` (default on) pins the 30 finger quaternions at their
    converter values: the finger joints carry the two fingertip contact frames but
    are not fitted. Turning it off lets the hand pose be *observed* — pair it with
    ``[keypoints].include_hands`` so the fingertips have a 2D witness. Because the
    frozen pose is whatever the converter produced frame by frame, its jitter is
    smoothed out of the init trajectory first (``[stages.kinematics_1].finger_smooth_s``).
    """
    knobs = cfg.knobs
    pred = Path(pred)
    tag = "human_optim/kinematic"

    sp = dict(np.load(pred / "sam3d" / "smplx_params.npz", allow_pickle=True))
    pose = dict(np.load(pred / "sapiens" / "pose.npz", allow_pickle=True))
    human = dict(np.load(pred / "geometry" / "human.npz", allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    if not bool(tr.get("metric", True)):  # absent flag = old already-metric artifact
        raise SystemExit(f"[{tag}] geometry under {pred} is up-to-scale — run estimate_scale after fuse")

    N, P = int(sp["num_frames"]), int(sp["num_people"])
    H, W = int(sp["video_height"]), int(sp["video_width"])
    object_ids = np.asarray(sp["object_ids"], dtype=np.int32)
    fps = float(tr.get("fps", 30.0))
    # Frozen fingers inherit the converter's per-frame hand pose, jitter included. Smooth
    # that trajectory once at init (sigma in seconds → frames); 0 leaves it untouched.
    finger_sigma_frames = float(cfg.stage_opts["kinematics_1"].get("finger_smooth_s", 0.0)) * fps
    params, init_src = _init_params(pred, sp, object_ids, tag)
    print(f"[{tag}] init pose: {init_src}")

    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)       # (N,4,4)
    K = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)  # (N,3,3)
    wfc = world_from_cam(extr)
    cam_centers = wfc[:, :3, 3]                     # (N,3) world camera centers (radial-velocity damper)
    scene_pts, scene_normals, scene_conf = _load_scene(pred, knobs, wfc, device, tag)

    mid, fvids, corr, sub, chamfer_w = _body_context(knobs, device)
    torch.manual_seed(0)
    phases = _kinematic_phases(cfg, stages_to_run)

    # Occlusion-aware down-weighting of the visual terms (kinematics_1 only). 0 is off.
    vis_strength = float(knobs.get("occlusion", {}).get("strength", 0.0))
    parents = [-1] + [int(mid.parent_tree[0, j]) for j in range(1, mid.num_joints)]

    # kinematics_1 auto-contact operating point. optimize_person detects labels once at the
    # refine phase, guarded on ann_attraction[kinematics_1] > 0, so this is inert at weight 0.
    auto_cfg = None
    if "auto_contact" in knobs and "kinematics_1" in stages_to_run:
        ac = dict(knobs["auto_contact"])
        # Stillness is ONE notion shared with the readout: both inherit [contact_gate]. A
        # [auto_contact] override is still honoured, but the shipped config does not use one —
        # a detector that is stricter than the readout only hides labels from the attraction.
        ac.setdefault("window_s", float(knobs["contact_gate"]["window_s"]))
        ac.setdefault("still_m", float(knobs["contact_gate"]["still_m"]))
        auto_cfg = ac

    if labels_mode != "auto":
        raise SystemExit(
            f"[{tag}] [stages.kinematics_1].labels must be 'auto', got {labels_mode!r} — "
            "contact lives on the 12 contact frames and no external model predicts those")

    out = _new_out()
    betas_out = []
    drift_pre_p, auto_labels_p = [], []   # refine-start drift + frozen labels, per person
    for p in range(P):
        oid = int(object_ids[p])
        vmask = np.asarray(sp["valid_mask"][p], dtype=bool)
        betas_mean = torch.as_tensor(params["betas"][p][vmask].mean(0), dtype=torch.float32, device=device)

        def t(k):  # per-frame (N, ...) param tensor; invalid frames held from nearest valid
            return torch.as_tensor(_fill_invalid_frames(params[k][p], vmask),
                                   dtype=torch.float32, device=device)
        betas_T = betas_mean[None].expand(N, -1)
        q_cam = mid.from_classic(betas_T, t("body_pose"), t("transl"), t("global_orient"),
                                 t("left_hand_pose"), t("right_hand_pose"))
        q_world = fold_root_to_world(q_cam, wfc)
        smooth_finger_quats(q_world, vmask, finger_sigma_frames)
        q_world[:, 3:] += 1e-6 * torch.randn_like(q_world[:, 3:])   # break bit-exact static-frame ties

        target_kp, scores, depth, frame_valid = _person_visual_inputs(
            pose, human, oid, N, vmask, knobs, device,
            mid=mid, betas_T=betas_T, q_world=q_world, extr=extr, K=K)

        keypoint_vis = joint_vis = None
        if vis_strength > 0.0:
            keypoint_vis, joint_vis = visibility_weights(
                scores, corr, parents, strength=vis_strength,
                window_s=float(knobs["keypoints"].get("smooth_window_s", 0.0)),
                fps=fps, frame_valid=frame_valid)
            seen = joint_vis[frame_valid][:, corr.joint_model]
            print(f"[{tag}] visibility strength={vis_strength:g}: mean={float(seen.mean()):.3f}, "
                  f"{float((seen < 0.5).float().mean()):.1%} of corresponded joint-frames < 0.5")

        print(f"[{tag}] person obj_id={oid} ({int(vmask.sum())}/{N} valid) …")
        q_opt, before, after, aux = optimize_person(
            mid, betas_mean, q_world, extrinsics=extr, K=K, depth_points=depth,
            target_kp=target_kp, scores=scores, corr=corr, frame_vertex_ids=fvids,
            sub=sub, knobs=knobs, phases=phases,
            scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, fps=fps,
            chamfer_weights=chamfer_w, frame_valid=frame_valid,
            cam_centers=cam_centers, auto_contact=auto_cfg, freeze_fingers=freeze_fingers,
            keypoint_vis=keypoint_vis, joint_vis=joint_vis)
        print(f"    before: {before}\n    after:  {after}")
        # Persist drift_pre + auto_labels: drift_pre is the honest pre-attraction gate source
        # for the readout; auto_labels track the body produced by this optimization.
        dp, al = aux["drift_pre"], aux["auto_labels"]

        _append_classic(out, mid, betas_T, q_opt)
        betas_out.append(betas_mean.cpu().numpy().astype(np.float32))
        # None ⇒ detection did not run for this person (no scene / weight 0): write NaN drift +
        # -1 labels; consumers then fall back to computing drift from the body.
        num_cf = len(mid.contact_frame_names)
        drift_pre_p.append(dp.cpu().numpy().astype(np.float32) if dp is not None
                           else np.full((N, num_cf), np.nan, np.float32))
        auto_labels_p.append(al.cpu().numpy().astype(np.int8) if al is not None
                             else np.full((N, num_cf), -1, np.int8))

    body = "kinematic_1" if "kinematics_1" in stages_to_run else "root"
    out_path = paths.body_path(pred, body)
    extra = dict(drift_pre=np.stack(drift_pre_p, 0), auto_labels=np.stack(auto_labels_p, 0),
                 contact_frame_names=np.array(list(mid.contact_frame_names)))
    save = _save_body(out_path, mid, out, betas_out, object_ids, sp["valid_mask"], N, P, H, W, fps, extra=extra)
    print(f"[{tag}] wrote {out_path}")

    if overlay and body == "kinematic_1":
        from .overlay import render_keypoint_overlay
        ov = render_keypoint_overlay(
            source, paths.overlay_path(pred, "kinematic_1"), mid, save, corr, fvids,
            extr, K, target_scheme=pose, fps=fps, device=device)
        print(f"[{tag}] wrote {ov}")
    return out_path


# ─────────────────── kinematics_2 (annotation-guided) ─────────────────────

def _hand_smplx_joints(side: str) -> list[str]:
    """The SMPLXMid joints one ``<side>_hand`` annotation covers: wrist + 15 fingers."""
    return [f"{side}_wrist"] + [f"{side}_{f}{k}" for f in _FINGER for k in (1, 2, 3)]


_ANN_TO_SMPLX: dict[str, list[str]] = {
    "left_hand": _hand_smplx_joints("left"),
    "right_hand": _hand_smplx_joints("right"),
    "left_foot": ["left_ankle", "left_foot"],
    "right_foot": ["right_ankle", "right_foot"],
    "left_knee": ["left_knee"],
    "right_knee": ["right_knee"],
    "left_hip": ["left_hip"],
    "right_hip": ["right_hip"],
    "left_elbow": ["left_elbow"],
    "right_elbow": ["right_elbow"],
}

# The 14-joint schema (its own l/r ankle + shoulder labels, and its 'foot' means the TOE,
# not the whole ankle+foot region). Hands stay wrist+fingers; foot maps to the toe joint
# only (ankle is its own label); the rest map to their same-named SMPLXMid joints.
_ANN_TO_SMPLX_14: dict[str, list[str]] = {
    "left_hand": _hand_smplx_joints("left"),
    "right_hand": _hand_smplx_joints("right"),
    "left_foot": ["left_foot"],
    "right_foot": ["right_foot"],
    "left_ankle": ["left_ankle"],
    "right_ankle": ["right_ankle"],
    "left_knee": ["left_knee"],
    "right_knee": ["right_knee"],
    "left_elbow": ["left_elbow"],
    "right_elbow": ["right_elbow"],
    "left_shoulder": ["left_shoulder"],
    "right_shoulder": ["right_shoulder"],
    "left_hip": ["left_hip"],
    "right_hip": ["right_hip"],
}


def _ann_mapping(ann_names: list[str]) -> dict[str, list[str]]:
    """The annotation→SMPLXMid group mapping for this schema, by EXACT joint-name set.

    The ``ann_names`` set must match the legacy 10-joint schema (:data:`_ANN_TO_SMPLX`) or the
    14-joint schema (:data:`_ANN_TO_SMPLX_14`) exactly — a partial or misspelled annotation
    raises (with the observed names) rather than silently dropping the unmatched joints."""
    names = set(ann_names)
    if names == set(_ANN_TO_SMPLX):
        return _ANN_TO_SMPLX
    if names == set(_ANN_TO_SMPLX_14):
        return _ANN_TO_SMPLX_14
    raise ValueError(
        f"unrecognized contact annotation schema {sorted(ann_names)}; expected the 10-joint "
        f"{sorted(_ANN_TO_SMPLX)} or the 14-joint {sorted(_ANN_TO_SMPLX_14)} joint set")


def _contact_labels(contacts_p: np.ndarray, ann_names: list[str],
                    mid_names: list[str], N: int) -> np.ndarray:
    """Expand one person's ``(J_ann, N_ann)`` annotation into ``(N, 52)`` SMPLXMid labels.

    The schema (10- or 14-joint) is resolved by :func:`_ann_mapping` (exact-name match).
    """
    mapping = _ann_mapping(ann_names)
    Na = contacts_p.shape[1]
    if Na != N:
        raise ValueError(f"annotation has {Na} frames but the body trajectory has {N}")
    mid_idx = {n: i for i, n in enumerate(mid_names)}
    ann_idx = {n: i for i, n in enumerate(ann_names)}
    lab = np.full((N, len(mid_names)), -1, dtype=np.int8)
    for ann_name, smplx_joints in mapping.items():
        if ann_name not in ann_idx:
            continue
        series = contacts_p[ann_idx[ann_name]]
        for jn in smplx_joints:
            if jn in mid_idx:
                lab[:, mid_idx[jn]] = series
    return lab


def load_frame_contacts(pred: Path, tag: str, body_path: Path, N: int,
                        frame_names: list[str], stage_tag: str):
    """Load + validate ``contacts_<tag>.npz`` as an AUTO label source (the contact frames).

    ``body_path`` is the body the contacts were measured ON (and that the caller is
    about to refine) — sha256 provenance ties the artifact to it, plus to the scene
    and transform, so a stale readout can never quietly drive a refinement. Returns
    ``(cnt_oids, frame_contact)`` with ``frame_contact (Pc, N, F)`` bool.
    """
    cnt_path = paths.contacts_path(pred, tag)
    if not cnt_path.exists():
        raise SystemExit(f"missing {cnt_path} — run estimate_contacts "
                         f"(--bodies {body_path.stem}) first")
    cnt = dict(np.load(cnt_path, allow_pickle=True))
    cnt_oids = [int(x) for x in cnt["object_ids"]]
    try:
        frame_contact = np.asarray(cnt["frame_contact"], bool)
        cnt_names = [str(x) for x in np.asarray(cnt["contact_frame_names"]).tolist()]
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"{cnt_path} carries no frame_contact / contact_frame_names — it predates the "
            "contact-frame detector; regenerate it with estimate_contacts") from exc
    if cnt_names != list(frame_names):
        raise SystemExit(f"{cnt_path} contact-frame order does not match the body's")
    expected = (len(cnt_oids), N, len(frame_names))
    if frame_contact.shape != expected:
        raise SystemExit(f"{cnt_path} frame_contact must be {expected}, got {frame_contact.shape}")
    provenance = (
        (body_path, "source_body_sha256"),
        (pred / "geometry" / "scene.npz", "source_scene_sha256"),
        (pred / "geometry" / "transform.npz", "source_transform_sha256"),
    )
    for source_path, hash_key in provenance:
        with source_path.open("rb") as source_file:
            expected_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
        if hash_key not in cnt or str(np.asarray(cnt[hash_key]).item()) != expected_hash:
            raise SystemExit(
                f"{cnt_path} is stale for {source_path}; regenerate estimate_contacts")
    print(f"[{stage_tag}] AUTO labels from {cnt_path.name}, ids {cnt_oids}")
    return cnt_oids, frame_contact


#: Which contact frames one coarse annotation joint covers — the inverse of the fold
#: ``scripts/score_frame_contacts.py`` scores with. Used to put MANUAL per-joint
#: annotations on the frame axis for the physics stages, whose contact points ARE the
#: frames. Names cover BOTH frame sets (legacy ``toe_under`` and the full-body set's
#: ``ball_* / big_toe`` / elbow / forearm / shoulder / hip_side frames);
#: :func:`annotation_frame_labels` keeps only the names the body actually carries.
#: ``hip`` maps to the greater-trochanter frame only — the sit bones, torso and head
#: frames have no annotation joint and stay unlabeled.
_ANN_TO_FRAMES: dict[str, tuple[str, ...]] = {
    "left_hand": ("palm_left", "finger_left", "palm_distal_left", "thumb_left"),
    "right_hand": ("palm_right", "finger_right", "palm_distal_right", "thumb_right"),
    "left_knee": ("knee_left",),
    "right_knee": ("knee_right",),
    # 14-joint schema: "foot" means the forefoot, "ankle" the heel. In the 10-joint
    # schema "foot" is the whole foot, so the heel joins it (see :func:`annotation_frame_labels`).
    "left_foot": ("toe_under_left", "toe_above_left", "ball_inner_left", "ball_outer_left",
                  "big_toe_left"),
    "right_foot": ("toe_under_right", "toe_above_right", "ball_inner_right",
                   "ball_outer_right", "big_toe_right"),
    "left_ankle": ("heel_left",),
    "right_ankle": ("heel_right",),
    "left_elbow": ("elbow_left", "forearm_left"),
    "right_elbow": ("elbow_right", "forearm_right"),
    "left_shoulder": ("shoulder_left",),
    "right_shoulder": ("shoulder_right",),
    "left_hip": ("hip_side_left",),
    "right_hip": ("hip_side_right",),
}


def annotation_frame_labels(contacts_p: np.ndarray, ann_names: list[str],
                            frame_names: list[str], N: int) -> np.ndarray:
    """Expand one person's ``(J_ann, N)`` annotation onto the contact frames ``(N, F)``.

    Tri-state in, tri-state out: an annotated joint's value is copied onto every contact
    frame it covers (:data:`_ANN_TO_FRAMES`); a frame no annotated joint covers stays
    ``-1`` (unlabeled). The schema is resolved by :func:`_ann_mapping`, and in the
    10-joint schema — where ``foot`` means the whole foot — the heel inherits the foot's
    label too.
    """
    mapping = _ann_mapping(ann_names)          # validates the schema (raises on anything else)
    if contacts_p.shape[1] != N:
        raise ValueError(f"annotation has {contacts_p.shape[1]} frames but the body has {N}")
    ten_joint = mapping is _ANN_TO_SMPLX
    frame_idx = {n: i for i, n in enumerate(frame_names)}
    ann_idx = {n: i for i, n in enumerate(ann_names)}
    labels = np.full((N, len(frame_names)), -1, dtype=np.int8)
    for ann_name, frames in _ANN_TO_FRAMES.items():
        if ann_name not in ann_idx:
            continue
        if ten_joint and ann_name.endswith("_ankle"):
            continue                                    # no ankle label in the 10-joint schema
        covered = list(frames)
        if ten_joint and ann_name.endswith("_foot"):
            covered.append(f"heel_{ann_name.split('_')[0]}")
        series = contacts_p[ann_idx[ann_name]]
        for name in covered:
            if name in frame_idx:
                labels[:, frame_idx[name]] = series
    return labels


def resolve_annotation_path(source, annotations) -> Path:
    """The annotation npz: explicit ``--annotations``, else ``<source>.npz`` / ``data/<stem>.npz`` / ``data_2/<stem>.npz``."""
    if annotations is not None:
        return Path(annotations)
    src = Path(source)
    cand = src.with_suffix(".npz")
    if cand.exists():
        return cand
    legacy = _REPO_ROOT / "data" / f"{src.stem}.npz"
    if legacy.exists():
        return legacy
    return _REPO_ROOT / "data_2" / f"{src.stem}.npz"


def run_kinematics_2(source, pred: Path, cfg: HumanOptimConfig, device: str, *,
                     annotations=None, overlay: bool = False, labels_mode: str = "manual") -> Path:
    """Contact-guided refinement of ``kinematic_1.npz`` → ``human_optim/kinematic_2.npz``.

    ``labels_mode='auto'`` (the pipeline default) reads our own ``contacts_1.npz``: one
    label per **contact frame** per video frame, positives-only (touching → ``+1``, else
    ``-1``, **never** ``0``, so the clearance repulsion stays inert by construction).
    Each of the 12 contact frames is then its own repair instance — the acquisition pulls
    that exact point toward the patch of wall it appears on, because the contact frames are
    where contact is detected, forced and measured. No gate: a witness reports ~0 confidence
    on a floating body, which is exactly the body this stage exists to repair.

    ``labels_mode='manual'`` uses the human annotation ``data/<stem>.npz`` (a metrics-only
    comparison path) and keeps the older per-joint machinery: labels reach the mesh through
    each vertex's nearest joint, and the repair instances are the 14 annotation limb groups.
    """
    knobs = cfg.knobs
    pred = Path(pred)
    tag = "human_optim/kinematic_2"

    kin_path = paths.body_path(pred, "kinematic_1")
    kin = dict(np.load(kin_path, allow_pickle=True))
    pose = dict(np.load(pred / "sapiens" / "pose.npz", allow_pickle=True))
    human = dict(np.load(pred / "geometry" / "human.npz", allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    if not bool(tr.get("metric", True)):  # absent flag = old already-metric artifact
        raise SystemExit(f"[{tag}] geometry under {pred} is up-to-scale — run estimate_scale after fuse")

    N, P = int(kin["num_frames"]), int(kin["num_people"])
    H, W = int(kin["video_height"]), int(kin["video_width"])
    object_ids = np.asarray(kin["object_ids"], dtype=np.int32)
    fps = float(tr.get("fps", kin.get("fps", 30.0)))

    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)
    K = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)
    wfc = world_from_cam(extr)
    cam_centers = wfc[:, :3, 3]                     # (N,3) world camera centers (radial-velocity damper)
    scene_pts, scene_normals, scene_conf = _load_scene(pred, knobs, wfc, device, tag)

    mid, fvids, corr, sub, chamfer_w = _body_context(knobs, device)
    mid_names = list(mid.joint_names)
    frame_names = list(mid.contact_frame_names)

    # Label source: our own contacts_1 on the 12 contact frames (auto) or the human
    # per-joint annotation (manual).
    ann_contacts = ann_names = ann_oids = cnt_oids = ann_ignored = None
    cnt_contact = None
    if labels_mode == "auto":
        cnt_oids, cnt_contact = load_frame_contacts(pred, "1", kin_path, N, frame_names, tag)
    else:
        ann_path = resolve_annotation_path(source, annotations)
        ann = dict(np.load(ann_path, allow_pickle=True))
        ann_contacts = ann["contacts"]
        ann_names = [str(x) for x in ann["joint_names"]]
        ann_oids = [int(x) for x in ann["object_ids"]]
        ann_ignored = np.asarray(ann["ignored"], bool) if "ignored" in ann else np.zeros(len(ann_oids), bool)
        print(f"[{tag}] annotation {ann_path} — {ann_contacts.shape[0]} people, joints {ann_names}, ids {ann_oids}")

    torch.manual_seed(0)
    phases = _kinematics_2_phases(cfg)

    # Contact-patch repair groups. Auto mode needs none — each contact frame is its own
    # one-point group. Manual mode maps every annotation group to its member SMPLXMid joint
    # indices (the annotation's own schema, the same mapping :func:`_contact_labels` uses).
    contact_axis = "frames" if labels_mode == "auto" else "joints"
    contact_groups = None
    if labels_mode != "auto":
        jidx = {n: i for i, n in enumerate(mid_names)}
        contact_groups = [g for g in ([jidx[j] for j in members if j in jidx]
                                      for members in _ann_mapping(ann_names).values()) if g]

    out = _new_out()
    betas_out = []
    for p in range(P):
        oid = int(object_ids[p])
        # Manual mode: a person whose annotation object is ignored (or absent) gets NO labels and
        # its kinematic_1 arrays are copied through byte-for-byte — no label-less re-refine drift,
        # no normalize/to_classic round-trip.
        if labels_mode != "auto" and (oid not in ann_oids or bool(ann_ignored[ann_oids.index(oid)])):
            why = "absent" if oid not in ann_oids else "ignored"
            print(f"[{tag}] person obj_id={oid}: annotation {why} — copying kinematic_1 through unchanged")
            for key in out:
                out[key].append(np.asarray(kin[key][p], np.float32))
            betas_out.append(np.asarray(kin["betas"][p], np.float32))
            continue

        vmask = np.asarray(kin["valid_mask"][p], dtype=bool)
        betas_mean = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        q_world = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device).clone()
        q_world[:, 3:] += 1e-6 * torch.randn_like(q_world[:, 3:])
        betas_T = betas_mean[None].expand(N, -1)

        target_kp, scores, depth, frame_valid = _person_visual_inputs(
            pose, human, oid, N, vmask, knobs, device,
            mid=mid, betas_T=betas_T, q_world=q_world, extr=extr, K=K)

        contact_labels = None
        if labels_mode == "auto":
            if cnt_oids is not None and oid in cnt_oids:
                fc = cnt_contact[cnt_oids.index(oid)]                          # (N, 12)
                lab = np.where(fc, np.int8(1), np.int8(-1)).astype(np.int8)    # +1 / -1, NEVER 0 (repulsion inert)
                lab[~vmask] = -1
                contact_labels = torch.as_tensor(lab, dtype=torch.int8, device=device)
                per_frame = " ".join(f"{n}={int((lab[:, i] == 1).sum())}"
                                     for i, n in enumerate(frame_names))
                print(f"[{tag}] obj_id={oid}: AUTO contacts on "
                      f"{int((lab == 1).any(1).sum())} frames  ({per_frame})")
            else:
                print(f"[{tag}] obj_id={oid}: no auto contacts (re-refine only)")
        else:   # manual mode reaches here only for a labelled, non-ignored person
            lab = _contact_labels(ann_contacts[ann_oids.index(oid)], ann_names, mid_names, N)
            lab[~vmask] = -1
            contact_labels = torch.as_tensor(lab, dtype=torch.int8, device=device)
            nc = int((lab == 1).any(1).sum()); nn = int((lab == 0).any(1).sum())
            print(f"[{tag}] person obj_id={oid}: contacts on {nc} frames, no-contact on {nn} frames")

        print(f"[{tag}] person obj_id={oid} ({int(vmask.sum())}/{N} valid) …")
        q_opt, before, after, _ = optimize_person(
            mid, betas_mean, q_world, extrinsics=extr, K=K, depth_points=depth,
            target_kp=target_kp, scores=scores, corr=corr, frame_vertex_ids=fvids,
            sub=sub, knobs=knobs, phases=phases,
            scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, fps=fps,
            contact_labels=contact_labels, contact_axis=contact_axis, contact_groups=contact_groups,
            image_hw=(H, W), chamfer_weights=chamfer_w, frame_valid=frame_valid, cam_centers=cam_centers)
        print(f"    before: {before}\n    after:  {after}")
        _append_classic(out, mid, betas_T, q_opt)
        betas_out.append(betas_mean.cpu().numpy().astype(np.float32))

    # Forward the honest pre-enforcement drift + frozen labels from kinematic_1. The contact
    # readout on THIS body (contacts_2) prices stillness by drift, and kin2's own velocity
    # damper holds exactly the frames contacts_1 called touching still — so re-measuring drift
    # here would let contacts_1 confirm itself through contacts_2. drift_pre is the motion
    # measured before ANY contact term acted, in kinematics_1, which is the one measurement no
    # contact enforcement anywhere in the chain can manufacture. It stays the source.
    extra = {k: kin[k] for k in ("drift_pre", "auto_labels", "contact_frame_names") if k in kin}
    out_path = paths.body_path(pred, "kinematic_2")
    save = _save_body(out_path, mid, out, betas_out, object_ids, kin["valid_mask"], N, P, H, W, fps,
                      extra=extra or None)
    print(f"[{tag}] wrote {out_path}")

    if overlay:
        from .overlay import render_keypoint_overlay
        ov = render_keypoint_overlay(
            source, paths.overlay_path(pred, "kinematic_2"), mid, save, corr, fvids,
            extr, K, target_scheme=pose, fps=fps, device=device)
        print(f"[{tag}] wrote {ov}")
    return out_path
