"""The human-optimization stage runners: root / kinematics_1 / kinematics_2 / dynamics.

One place for the per-stage orchestration the ``human_optimization`` dispatcher
drives. The three *kinematic* stages share the same machinery — load the inputs,
per person build the world-frame ``q``, run :func:`~tools.human_optim.optimizer.optimize_person`
over a phase list, write the refined-body npz — differing only in where ``q`` starts
(converter vs the previous refinement), the weight columns their phases use, and (for
``kinematics_2``) the manual contact annotation. ``dynamics`` is a separate force-only
RNEA solve on the frozen ``q``.

The caller (``scripts/stages/human_optimization.py``) has already put
``better_human`` on ``sys.path``; this module assumes it. Weights + knobs come from a
:class:`~tools.human_optim.config.HumanOptimConfig`; outputs are placed by
:mod:`tools.human_optim.paths`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from tools.geometry.normals import estimate_point_normals
from tools.smplx_robot import (
    load_smplx_mid, make_smplx_inertial_model, remap_q_bh_to_model,
)
from tools.smplx_robot.dynamics import velocity_acceleration_from_trajectory
from tools.smplx_robot.vertex_weights import geodesic_endeffector_weights

from . import paths
from .camera import fold_root_to_world, project, transform_points, world_from_cam
from .config import HumanOptimConfig
from .keypoints import build_correspondence
from .optimizer import PhaseSpec, optimize_person
from .qmath import normalize_q
from .visibility import subsample_indices

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SMPLX_MODEL = _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
_GRAVITY_MAG = 9.81  # m/s²
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


def _load_scene(pred: Path, knobs: dict, wfc: torch.Tensor, device, tag: str):
    """Scene cloud + camera-oriented PCA normals for the penetration/attraction terms.

    Returns ``(points, normals, confidence)`` or ``(None, None, None)`` when the
    clip has no ``geometry/scene.npz`` (the stage then runs without scene terms)."""
    scene_path = pred / "geometry" / "scene.npz"
    if not scene_path.exists():
        print(f"[{tag}] no geometry/scene.npz; scene terms OFF")
        return None, None, None
    sc = dict(np.load(scene_path, allow_pickle=True))
    scene_pts = torch.as_tensor(sc["points"], dtype=torch.float32, device=device)
    scene_conf = torch.as_tensor(sc["confidence"], dtype=torch.float32, device=device)
    # Quantile confidence gate: drop the lowest conf_drop_q fraction of scene voxels
    # (the raw VGGT confidence scale is arbitrary, so the old absolute floor never
    # fired). The scene is already per-frame conf-gated at fuse; this trims the
    # least-reliable merged voxels before the penetration/attraction NN search.
    q = float(knobs["penetration"].get("conf_drop_q", 0.0))
    if q > 0.0 and scene_conf.numel():
        keep = scene_conf >= torch.quantile(scene_conf, q)
        scene_pts, scene_conf = scene_pts[keep], scene_conf[keep]
    nk = int(knobs["penetration"]["normal_k"])
    scene_normals = estimate_point_normals(scene_pts, wfc[:, :3, 3], k=nk)
    print(f"[{tag}] scene terms ON ({scene_pts.shape[0]} pts, PCA normals k={nk})")
    return scene_pts, scene_normals, scene_conf


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
    """Phases for the converter-anchored ``root`` / ``kinematics_1`` pass.

    ``root`` (root SE3 only, no scene) uses the root column; ``kinematics_1``'s
    ``full`` → ``refine`` → ``lbfgs`` (all DOFs; scene on from refine) use the
    kinematics_1 column — all in one call so the priors stay anchored to the
    converter pose.
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
    """Phases for ``kinematics_2`` (all use the kinematics_2 column; gentle re-settle
    from the kinematics_1 fit, then the annotation-driven scene refinement)."""
    it = cfg.iters["kinematics_2"]
    w = cfg.weights_for("kinematics_2")
    return [
        PhaseSpec("root", w, it.get("root", 0), _ROOT_DOF, False, "adam"),
        PhaseSpec("full", w, it.get("full", 0), _ALL_DOF, False, "adam"),
        PhaseSpec("refine", w, it.get("refine", 0), _ALL_DOF, True, "adam"),
        PhaseSpec("lbfgs", w, it.get("lbfgs", 0), _ALL_DOF, True, "lbfgs"),
    ]


# ─────────────────────── kinematic stages (root / kin1) ───────────────────

def run_kinematic(source, pred: Path, cfg: HumanOptimConfig, stages_to_run,
                  device: str, *, overlay: bool = False) -> Path:
    """Run ``root`` and/or ``kinematics_1`` for every tracked person → ``human_optim/``.

    Writes ``kinematic_1.npz`` when ``kinematics_1`` is in ``stages_to_run``, else the
    intermediate ``root.npz`` (root-only placement).
    """
    knobs = cfg.knobs
    pred = Path(pred)
    tag = "human_optim/kinematic"

    sp = dict(np.load(pred / "sam3d" / "smplx_params.npz", allow_pickle=True))
    pose = dict(np.load(pred / "sapiens" / "pose.npz", allow_pickle=True))
    human = dict(np.load(pred / "geometry" / "human.npz", allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))

    N, P = int(sp["num_frames"]), int(sp["num_people"])
    H, W = int(sp["video_height"]), int(sp["video_width"])
    object_ids = np.asarray(sp["object_ids"], dtype=np.int32)
    fps = float(tr.get("fps", 30.0))

    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)       # (N,4,4)
    K = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)  # (N,3,3)
    wfc = world_from_cam(extr)
    cam_centers = wfc[:, :3, 3]                     # (N,3) world camera centers (radial-velocity damper)
    scene_pts, scene_normals, scene_conf = _load_scene(pred, knobs, wfc, device, tag)

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    fvids = torch.as_tensor([int(x) for x in mid.frame_vertex_ids], dtype=torch.long, device=device)
    corr = build_correspondence(list(mid.frame_names), include_hands=bool(knobs["keypoints"]["include_hands"]), device=device)
    sub = subsample_indices(mid.num_vertices, int(knobs["chamfer"]["vertex_subsample"]), device=device)
    # Per-vertex body-part emphasis for chamfer (geodesic linear, log-mapped to [lo, hi] over
    # radius m): extremities ×hi, torso ×lo. Defaults (0.1 / 10.0 / 1.0) reproduce the base field.
    chamfer_w = torch.as_tensor(
        geodesic_endeffector_weights(
            mid.vertices_template, mid.faces, mid.lbs_weights, mid.joint_names,
            lo=float(knobs["chamfer"].get("emphasis_lo", 0.1)),
            hi=float(knobs["chamfer"].get("emphasis_hi", 10.0)),
            radius=float(knobs["chamfer"].get("emphasis_radius", 1.0))),
        dtype=torch.float32, device=device)[sub]                                   # (Ns,)
    torch.manual_seed(0)
    phases = _kinematic_phases(cfg, stages_to_run)

    # kinematics_1 auto-contact: resolve the [auto_contact] groups → SMPLXMid joint indices and
    # fold in the shared [contact_gate] stillness. optimize_person detects labels once at the
    # refine phase, guarded on ann_attraction[kinematics_1] > 0, so this is inert at weight 0.
    auto_cfg = None
    if "auto_contact" in knobs and "kinematics_1" in stages_to_run:
        ac = dict(knobs["auto_contact"])
        # Detector stillness: [auto_contact] may pin its OWN window_s / still_m — decoupled from the
        # readout's [contact_gate] so the readout recall knob can move WITHOUT changing the frozen
        # labels / bodies; omit to inherit [contact_gate] (one shared notion of "still").
        ac.setdefault("window_s", float(knobs["contact_gate"]["window_s"]))
        ac.setdefault("still_m", float(knobs["contact_gate"]["still_m"]))
        idx = {n: i for i, n in enumerate(mid.joint_names)}
        members = {g: [idx[j] for j in _ANN_TO_SMPLX.get(g, []) if j in idx]
                   for g in ac.get("groups", [])}
        ac["group_members"] = members
        ac["group_joints"] = sorted({i for v in members.values() for i in v})
        auto_cfg = ac

    out = _new_out()
    betas_out = []
    drift_pre_p, auto_labels_p = [], []   # refine-start drift + frozen labels, per person
    for p in range(P):
        oid = int(object_ids[p])
        vmask = np.asarray(sp["valid_mask"][p], dtype=bool)
        betas_mean = torch.as_tensor(sp["betas"][p][vmask].mean(0), dtype=torch.float32, device=device)

        def t(k):  # per-frame (N, ...) param tensor; invalid frames held from nearest valid
            return torch.as_tensor(_fill_invalid_frames(sp[k][p], vmask),
                                   dtype=torch.float32, device=device)
        betas_T = betas_mean[None].expand(N, -1)
        q_cam = mid.from_classic(betas_T, t("body_pose"), t("transl"), t("global_orient"),
                                 t("left_hand_pose"), t("right_hand_pose"))
        q_world = fold_root_to_world(q_cam, wfc)
        q_world[:, 3:] += 1e-6 * torch.randn_like(q_world[:, 3:])   # break bit-exact static-frame ties

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

        print(f"[{tag}] person obj_id={oid} ({int(vmask.sum())}/{N} valid) …")
        q_opt, before, after, aux = optimize_person(
            mid, betas_mean, q_world, extrinsics=extr, K=K, depth_points=depth,
            target_kp=target_kp, scores=scores, corr=corr, frame_vertex_ids=fvids,
            sub=sub, knobs=knobs, phases=phases,
            scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, fps=fps,
            chamfer_weights=chamfer_w, frame_valid=frame_valid,
            cam_centers=cam_centers, auto_contact=auto_cfg)
        print(f"    before: {before}\n    after:  {after}")
        # Persist drift_pre + auto_labels: drift_pre is the honest pre-attraction gate source
        # for the readout; auto_labels track the body produced by this optimization.
        dp, al = aux["drift_pre"], aux["auto_labels"]

        _append_classic(out, mid, betas_T, q_opt)
        betas_out.append(betas_mean.cpu().numpy().astype(np.float32))
        # None ⇒ detection did not run for this person (no scene / weight 0): write NaN drift +
        # -1 labels; consumers then fall back to computing drift from the body.
        drift_pre_p.append(dp.cpu().numpy().astype(np.float32) if dp is not None
                           else np.full((N, mid.num_joints), np.nan, np.float32))
        auto_labels_p.append(al.cpu().numpy().astype(np.int8) if al is not None
                             else np.full((N, mid.num_joints), -1, np.int8))

    body = "kinematic_1" if "kinematics_1" in stages_to_run else "root"
    out_path = paths.body_path(pred, body)
    extra = dict(drift_pre=np.stack(drift_pre_p, 0), auto_labels=np.stack(auto_labels_p, 0))
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


def _contact_labels(contacts_p: np.ndarray, ann_names: list[str],
                    mid_names: list[str], N: int) -> np.ndarray:
    """Expand one person's ``(J_ann, N_ann)`` annotation into ``(N, 52)`` SMPLXMid labels."""
    if "left_ankle" in ann_names:   # the 14-joint schema marker (data_2 annotations)
        raise ValueError(
            "kinematics_2 only supports the legacy 10-joint annotation schema; this annotation "
            "has its own ankle labels (14-joint schema, e.g. data_2/) whose 'foot' means the "
            "toe — the 10-joint _ANN_TO_SMPLX mapping (foot = ankle+foot) would mislabel it")
    Na = contacts_p.shape[1]
    if Na != N:
        raise ValueError(f"annotation has {Na} frames but the body trajectory has {N}")
    mid_idx = {n: i for i, n in enumerate(mid_names)}
    ann_idx = {n: i for i, n in enumerate(ann_names)}
    lab = np.full((N, len(mid_names)), -1, dtype=np.int8)
    for ann_name, smplx_joints in _ANN_TO_SMPLX.items():
        if ann_name not in ann_idx:
            continue
        series = contacts_p[ann_idx[ann_name]]
        for jn in smplx_joints:
            if jn in mid_idx:
                lab[:, mid_idx[jn]] = series
    return lab


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

    ``labels_mode='manual'`` (default) uses the human annotation ``data/<stem>.npz``;
    ``'auto'`` uses our own ``contacts_1.npz`` labels instead —
    positives-only (in-contact → ``+1``, else ``-1``, **never** ``0``, so the
    clearance repulsion stays inert by construction), each weighted by the T3
    ``label_confidence``.
    """
    knobs = cfg.knobs
    pred = Path(pred)
    tag = "human_optim/kinematic_2"

    kin = dict(np.load(paths.body_path(pred, "kinematic_1"), allow_pickle=True))
    pose = dict(np.load(pred / "sapiens" / "pose.npz", allow_pickle=True))
    human = dict(np.load(pred / "geometry" / "human.npz", allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))

    N, P = int(kin["num_frames"]), int(kin["num_people"])
    H, W = int(kin["video_height"]), int(kin["video_width"])
    object_ids = np.asarray(kin["object_ids"], dtype=np.int32)
    fps = float(tr.get("fps", kin.get("fps", 30.0)))

    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)
    K = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)
    wfc = world_from_cam(extr)
    cam_centers = wfc[:, :3, 3]                     # (N,3) world camera centers (radial-velocity damper)
    scene_pts, scene_normals, scene_conf = _load_scene(pred, knobs, wfc, device, tag)

    # Label source: our own contacts_1 (auto) or the human annotation (manual).
    cnt = ann_contacts = ann_names = ann_oids = cnt_oids = None
    if labels_mode == "auto":
        cnt_path = paths.contacts_path(pred, "1")
        if not cnt_path.exists():
            raise SystemExit(f"missing {cnt_path} — run estimate_contacts (--bodies kinematic_1) first")
        cnt = dict(np.load(cnt_path, allow_pickle=True))
        cnt_oids = [int(x) for x in cnt["object_ids"]]
        print(f"[{tag}] AUTO labels from {cnt_path.name} (contacts_1), ids {cnt_oids}")
    else:
        ann_path = resolve_annotation_path(source, annotations)
        ann = dict(np.load(ann_path, allow_pickle=True))
        ann_contacts = ann["contacts"]
        ann_names = [str(x) for x in ann["joint_names"]]
        ann_oids = [int(x) for x in ann["object_ids"]]
        print(f"[{tag}] annotation {ann_path} — {ann_contacts.shape[0]} people, joints {ann_names}, ids {ann_oids}")

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    mid_names = list(mid.joint_names)
    fvids = torch.as_tensor([int(x) for x in mid.frame_vertex_ids], dtype=torch.long, device=device)
    corr = build_correspondence(list(mid.frame_names), include_hands=bool(knobs["keypoints"]["include_hands"]), device=device)
    sub = subsample_indices(mid.num_vertices, int(knobs["chamfer"]["vertex_subsample"]), device=device)
    # Per-vertex body-part emphasis for chamfer (geodesic linear, log-mapped to [lo, hi] over
    # radius m): extremities ×hi, torso ×lo. Defaults (0.1 / 10.0 / 1.0) reproduce the base field.
    chamfer_w = torch.as_tensor(
        geodesic_endeffector_weights(
            mid.vertices_template, mid.faces, mid.lbs_weights, mid.joint_names,
            lo=float(knobs["chamfer"].get("emphasis_lo", 0.1)),
            hi=float(knobs["chamfer"].get("emphasis_hi", 10.0)),
            radius=float(knobs["chamfer"].get("emphasis_radius", 1.0))),
        dtype=torch.float32, device=device)[sub]                                   # (Ns,)
    torch.manual_seed(0)
    phases = _kinematics_2_phases(cfg)

    out = _new_out()
    betas_out = []
    for p in range(P):
        oid = int(object_ids[p])
        vmask = np.asarray(kin["valid_mask"][p], dtype=bool)
        betas_mean = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        q_world = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device).clone()
        q_world[:, 3:] += 1e-6 * torch.randn_like(q_world[:, 3:])
        betas_T = betas_mean[None].expand(N, -1)

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

        contact_labels = contact_conf = None
        if labels_mode == "auto":
            if cnt_oids is not None and oid in cnt_oids:
                cp = cnt_oids.index(oid)
                jc = np.asarray(cnt["joint_contact"][cp], bool)                # (N, 52)
                lab = np.where(jc, np.int8(1), np.int8(-1)).astype(np.int8)    # +1 / -1, NEVER 0 (repulsion inert)
                lab[~vmask] = -1
                contact_labels = torch.as_tensor(lab, dtype=torch.int8, device=device)
                if "label_confidence" in cnt:
                    cf = np.asarray(cnt["label_confidence"][cp], np.float32)
                    contact_conf = torch.as_tensor(cf, dtype=torch.float32, device=device)
                nc = int((lab == 1).any(1).sum())
                print(f"[{tag}] obj_id={oid}: AUTO contacts on {nc} frames"
                      f"{' (confidence-weighted)' if contact_conf is not None else ''}")
            else:
                print(f"[{tag}] obj_id={oid}: no auto contacts (re-refine only)")
        elif oid in ann_oids:
            lab = _contact_labels(ann_contacts[ann_oids.index(oid)], ann_names, mid_names, N)
            lab[~vmask] = -1
            contact_labels = torch.as_tensor(lab, dtype=torch.int8, device=device)
            nc = int((lab == 1).any(1).sum()); nn = int((lab == 0).any(1).sum())
            print(f"[{tag}] person obj_id={oid}: contacts on {nc} frames, no-contact on {nn} frames")
        else:
            print(f"[{tag}] person obj_id={oid}: NO annotation (re-refine only)")

        print(f"[{tag}] person obj_id={oid} ({int(vmask.sum())}/{N} valid) …")
        q_opt, before, after, _ = optimize_person(
            mid, betas_mean, q_world, extrinsics=extr, K=K, depth_points=depth,
            target_kp=target_kp, scores=scores, corr=corr, frame_vertex_ids=fvids,
            sub=sub, knobs=knobs, phases=phases,
            scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, fps=fps,
            contact_labels=contact_labels, contact_conf=contact_conf,
            chamfer_weights=chamfer_w, frame_valid=frame_valid, cam_centers=cam_centers)
        print(f"    before: {before}\n    after:  {after}")
        _append_classic(out, mid, betas_T, q_opt)
        betas_out.append(betas_mean.cpu().numpy().astype(np.float32))

    # Forward the honest pre-enforcement drift + frozen labels from kinematic_1: kin2's
    # own velocity damper must not be able to game the contact readout, so estimate_contacts on this
    # body reads the SAME drift_pre kinematic_1 measured before ANY enforcement. Fixes a latent gap
    # that until now dropped drift_pre for the manual kinematics_2 too.
    extra = {k: kin[k] for k in ("drift_pre", "auto_labels") if k in kin}
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


# ─────────────────────────── dynamics (force-only RNEA) ───────────────────

def _contact_catalog(joint_names: list[str]) -> dict[str, tuple[str, list[str]]]:
    """``group_name -> (representative_joint, member_joints)`` for SMPLXMid."""
    have = set(joint_names)
    cat: dict[str, tuple[str, list[str]]] = {}
    for side in ("left", "right"):
        hand = [f"{side}_wrist"] + [
            n for n in joint_names
            if n.startswith(f"{side}_") and any(f in n for f in _FINGER)
        ]
        cat[f"{side}_hand"] = (f"{side}_wrist", [n for n in hand if n in have])
        foot = [n for n in (f"{side}_ankle", f"{side}_foot") if n in have]
        rep = f"{side}_foot" if f"{side}_foot" in have else f"{side}_ankle"
        cat[f"{side}_foot"] = (rep, foot)
        for jn in ("knee", "elbow", "hip"):
            n = f"{side}_{jn}"
            if n in have:
                cat[f"{side}_{jn}"] = (n, [n])
    return cat


def _world_to_local(oMi: torch.Tensor, contact_ids: torch.Tensor) -> torch.Tensor:
    """``R_world_to_local (T, C, 3, 3)`` at the contact joints (q frozen ⇒ static)."""
    from better_robot.lie import so3
    R_lw = so3.to_matrix(oMi[:, contact_ids, 3:7])     # local→world
    return R_lw.transpose(-1, -2).contiguous()


def _build_fext(f_world, mask, R_w2l, contact_ids, njoints):
    """``(fext (T, njoints, 6), f_eff (T, C, 3))`` — masked world force → body wrench."""
    T, C, _ = f_world.shape
    f_eff = f_world * mask[..., None]
    f_local = torch.einsum("tcij,tcj->tci", R_w2l, f_eff)
    fext_lin = torch.zeros((T, njoints, 3), dtype=f_world.dtype, device=f_world.device)
    index = contact_ids.view(1, C, 1).expand(T, C, 3)
    fext_lin = fext_lin.scatter(dim=1, index=index, src=f_local)
    fext = torch.cat([fext_lin, torch.zeros_like(fext_lin)], dim=-1)
    return fext, f_eff


def _solve_person(mid, betas, scale, q_bh, contact_bool, valid, gravity_world, *,
                  weights, inertia, groups, dyn_opt, fps, device, label=""):
    """Solve world-frame contact forces for one person. Returns a result dict."""
    import better_robot as br
    from better_robot.kinematics import forward_kinematics

    dev = torch.device(device)
    dtype = torch.float32
    J = mid.num_joints
    N = q_bh.shape[0]
    dt = 1.0 / float(fps)

    model = make_smplx_inertial_model(
        mid, betas, float(scale), total_mass_kg=float(inertia["total_mass_kg"]),
        density=float(inertia["density"]), device=dev, dtype=dtype,
    )
    g = torch.as_tensor(gravity_world, dtype=dtype, device=dev)
    g = g / (g.norm() + 1e-9) * _GRAVITY_MAG
    g6 = torch.cat([g, torch.zeros(3, dtype=dtype, device=dev)])
    model = dataclasses.replace(model, gravity=g6)
    total_mass = float(model.body_inertias[1:, 0].sum())

    q_model = remap_q_bh_to_model(normalize_q(q_bh.to(dev), J), mid, model)
    with torch.no_grad():
        v, a = velocity_acceleration_from_trajectory(model, q_model, dt)
        oMi = forward_kinematics(model, q_model).joint_pose_world          # (N, njoints, 7)

    catalog = _contact_catalog(list(mid.joint_names))
    name_to_bh = {n: i for i, n in enumerate(mid.joint_names)}
    group_names = [g for g in groups if g in catalog]
    rep_bh, rep_model, masks = [], [], []
    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=dev)
    for gname in group_names:
        rep, members = catalog[gname]
        member_idx = [name_to_bh[m] for m in members if m in name_to_bh]
        active = contact_bool[:, member_idx].any(dim=1) & valid_t              # (N,)
        rep_bh.append(name_to_bh[rep])
        rep_model.append(int(model.body_name_to_id[rep]))
        masks.append(active)
    C = len(group_names)
    mask = torch.stack(masks, dim=1).to(dtype) if C else torch.zeros((N, 0), dtype=dtype, device=dev)
    contact_ids = torch.tensor(rep_model, dtype=torch.long, device=dev)
    R_w2l = _world_to_local(oMi, contact_ids).detach()

    valid_pair = (valid_t[1:] & valid_t[:-1]).to(dtype)                    # (N-1,)
    fpair = (mask[1:] * mask[:-1] * valid_pair[:, None]) if C else None    # (N-1, C)

    f_world = torch.zeros((N, C, 3), dtype=dtype, device=dev, requires_grad=True)
    data = model.create_data(batch_shape=(N,), device=dev, dtype=dtype)
    w_base = float(weights["base_wrench"])
    w_mag = float(weights["force_magnitude"])
    w_fs = float(weights["force_smooth"])
    w_ts = float(weights["torque_smooth"])

    def closure():
        opt.zero_grad()
        fext, f_eff = _build_fext(f_world, mask, R_w2l, contact_ids, model.njoints)
        tau = br.rnea(model, data, q_model, v, a, fext=fext)
        loss = 0.5 * ((w_base * tau[..., :6]).pow(2).sum() + (w_mag * f_eff).pow(2).sum())
        if N > 1:
            df = f_world[1:] - f_world[:-1]                                # (N-1, C, 3)
            loss = loss + 0.5 * w_fs * (fpair[..., None] * df.pow(2)).sum()
            dtau = tau[1:, 6:] - tau[:-1, 6:]                             # (N-1, nv-6)
            loss = loss + 0.5 * w_ts * (valid_pair[:, None] * dtau.pow(2)).sum()
        loss.backward()
        return loss

    if C > 0 and bool(mask.any()):
        opt = torch.optim.LBFGS(
            [f_world], lr=float(dyn_opt["lr"]), max_iter=int(dyn_opt["max_iter"]),
            history_size=int(dyn_opt["history_size"]), tolerance_grad=float(dyn_opt["tolerance_grad"]),
            tolerance_change=float(dyn_opt["tolerance_change"]), line_search_fn="strong_wolfe")
        with torch.no_grad():
            base0 = float((br.rnea(model, data, q_model, v, a).clone()[..., :6]).pow(2).sum().sqrt())
        opt.step(closure)
    else:
        base0 = float("nan")

    with torch.no_grad():
        f_final = (f_world * mask[..., None]).detach()                        # (N, C, 3)
        fext, _ = _build_fext(f_final, mask, R_w2l, contact_ids, model.njoints)
        tau = br.rnea(model, data, q_model, v, a, fext=fext)                  # (N, nv)
        base1 = float(tau[..., :6].pow(2).sum().sqrt())

        idx_vs = [int(x) for x in model.idx_vs]
        nvs = [int(x) for x in model.nvs]
        torque_mag = np.zeros((N, J), np.float32)
        joints_world = np.zeros((N, J, 3), np.float32)
        force_world = np.zeros((N, J, 3), np.float32)
        for jb in range(J):
            mb = int(model.body_name_to_id[mid.joint_names[jb]])
            n = nvs[mb]
            if n > 0:
                torque_mag[:, jb] = tau[:, idx_vs[mb]:idx_vs[mb] + n].norm(dim=-1).cpu().numpy()
            joints_world[:, jb] = oMi[:, mb, :3].cpu().numpy()
        for c, jb in enumerate(rep_bh):
            force_world[:, jb] = f_final[:, c].cpu().numpy()

    inval = ~np.asarray(valid, bool)
    torque_mag[inval] = 0.0
    force_world[inval] = 0.0
    fnorm = f_final.norm(dim=-1)                                              # (N, C)
    act = mask.bool().any(dim=1).cpu().numpy()
    mean_f = float(fnorm.sum(dim=-1)[mask.bool().any(dim=1)].mean()) if act.any() else 0.0

    fjit = tjit = 0.0
    if N > 1:
        with torch.no_grad():
            df = (f_final[1:] - f_final[:-1]).norm(dim=-1)                    # (N-1, C)
            fp = (mask[1:] * mask[:-1]).bool()
            fjit = float(df[fp].mean()) if bool(fp.any()) else 0.0
            dtq = (tau[1:, 6:] - tau[:-1, 6:]).norm(dim=-1)                   # (N-1,)
            vp = valid_pair.bool()
            tjit = float(dtq[vp].mean()) if bool(vp.any()) else 0.0
    print(f"  [{label}] mass={total_mass:.1f}kg  contact-frames={int(act.sum())}/{N}  "
          f"base wrench {base0:.1f}→{base1:.2f}  mean|f|={mean_f:.0f} N "
          f"(weight≈{total_mass * _GRAVITY_MAG:.0f} N)  jitter |Δf|={fjit:.1f} N |Δτ|={tjit:.2f}")

    return dict(
        joints_world=joints_world, torque_mag=torque_mag, force_world=force_world,
        contact_forces=f_final.cpu().numpy().astype(np.float32),
        joint_torques=tau.cpu().numpy().astype(np.float32),
        total_mass=np.float32(total_mass),
        rep_joint_names=[mid.joint_names[j] for j in rep_bh],
        base_wrench_before=np.float32(base0), base_wrench_after=np.float32(base1),
    )


def run_dynamics(pred: Path, cfg: HumanOptimConfig, device: str, *, bodies="kinematic_2") -> Path:
    """Force-only RNEA on a refined body → ``human_optim/dynamics.npz``."""
    knobs = cfg.knobs
    weights = cfg.weights_for("dynamics")
    pred = Path(pred)

    body = "kinematic_2" if bodies in ("kinematics_2", "kinematic_2") else "kinematic_1"
    kin_path = paths.body_path(pred, body)
    cnt_path = paths.contacts_path(pred, "2" if body == "kinematic_2" else "1")
    grav_path = pred / "geocalib" / "gravity.npz"
    for pth in (kin_path, cnt_path):
        if not pth.exists():
            raise SystemExit(f"missing {pth} — run human_optimization {body} + estimate_contacts first")

    kin = dict(np.load(kin_path, allow_pickle=True))
    cnt = dict(np.load(cnt_path, allow_pickle=True))
    fps = float(kin["fps"])

    N, P = int(kin["num_frames"]), int(kin["num_people"])
    object_ids = np.asarray(kin["object_ids"], np.int32)
    cnt_oids = list(np.asarray(cnt["object_ids"], np.int32))
    joint_contact = np.asarray(cnt["joint_contact"], bool)                    # (Pc, N, 52)

    if grav_path.exists():
        gd = np.load(grav_path)
        gravity_world = np.asarray(gd["gravity_world"], np.float32)
        gsrc = str(gd["source"]) if "source" in gd.files else "geocalib"
    else:
        gravity_world = np.array([0.0, 1.0, 0.0], np.float32)                 # VGGT world-down fallback
        gsrc = "fallback_down(no geocalib)"
    print(f"[human_optim/dynamics] {pred.name}: P={P} N={N} fps={fps:.1f}  bodies={bodies}  "
          f"gravity={gravity_world.round(2)} ({gsrc})")

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    J = mid.num_joints
    joints_world = np.zeros((P, N, J, 3), np.float32)
    torque_mag = np.zeros((P, N, J), np.float32)
    contact_forces_world = np.zeros((P, N, J, 3), np.float32)
    joint_torques = np.zeros((P, N, 6 + 3 * (J - 1)), np.float32)
    total_mass = np.zeros(P, np.float32)
    rep_names_p: list = []
    raw_forces_p: list = []

    for p in range(P):
        oid = int(object_ids[p])
        cp = cnt_oids.index(oid) if oid in cnt_oids else p
        res = _solve_person(
            mid, torch.as_tensor(kin["betas"][p], dtype=torch.float32),
            float(kin["scale"][p]) if "scale" in kin else 1.0,
            torch.as_tensor(kin["q"][p], dtype=torch.float32),
            torch.as_tensor(joint_contact[cp], dtype=torch.bool, device=device),
            np.asarray(kin["valid_mask"][p], bool),
            gravity_world, weights=weights, inertia=knobs["inertia"],
            groups=knobs["contact"]["groups"], dyn_opt=knobs["dynamics_optimizer"],
            fps=fps, device=device, label=f"obj {oid}",
        )
        joints_world[p] = res["joints_world"]
        torque_mag[p] = res["torque_mag"]
        contact_forces_world[p] = res["force_world"]
        joint_torques[p] = res["joint_torques"]
        total_mass[p] = res["total_mass"]
        rep_names_p.append(res["rep_joint_names"])
        raw_forces_p.append(res["contact_forces"])

    out_path = paths.dynamics_path(pred)
    np.savez(
        out_path,
        joints_world=joints_world, torque_mag=torque_mag,
        contact_forces_world=contact_forces_world, joint_torques=joint_torques,
        contact_forces=np.stack(raw_forces_p, 0),
        contact_force_joints=np.array(rep_names_p, dtype="<U16"),
        total_mass=total_mass, gravity_world=gravity_world.astype(np.float32),
        joint_contact=joint_contact[:P] if joint_contact.shape[0] >= P else joint_contact,
        object_ids=object_ids, joint_names=kin["joint_names"],
        parents=np.asarray(kin["parents"], np.int32), valid_mask=kin["valid_mask"],
        num_people=np.int32(P), num_frames=np.int32(N),
        video_height=np.int32(kin["video_height"]), video_width=np.int32(kin["video_width"]),
        fps=np.float32(fps), frame="world", source_bodies=str(bodies),
    )
    print(f"[human_optim/dynamics] wrote {out_path}")
    return out_path
