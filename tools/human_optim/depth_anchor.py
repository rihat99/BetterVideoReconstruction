"""Body-anchored world rescaling for gross human/scene depth disagreement.

VGGT supplies one coherent but arbitrary-scale world for the camera and scene.
The fused human cloud also sets that scale, so a bad human depth can move the wall.
SAM-3D supplies an independent metric body depth from the person's apparent size.
Small disagreements are ambiguous because the body initializer is only approximate.
Gross disagreement instead refutes the human-cloud ruler, not the static wall shape.
The gate therefore compares init extremities with the full fused scene surface.
Signed body/scene correspondences then predict a uniform world rescale exactly.
The chosen scale lets some frames touch without burying the low-gap tail too deeply.
Fired human points retain their old camera-frame depth while cameras follow the world.
One clip still has one world, so contradictory people conservatively veto the change.

MIN_FRAMES = 20: shorter tracks do not establish a clip-wide depth failure.
MIN_FINITE_FRAC = 0.6: most valid frames must support both gate signals.
PRUNE_MARGIN_WIDE_M = 0.8: signed correspondences must remain available at floater range.
GRAZING_MIN_NTP_M = 0.2: nearly tangent normals make the exact scale root unstable.
TOUCH_Q = 0.25: only a quarter of frames must reach the wall under noisy init depth.
TOUCH_EPS_M = 0.02: two centimetres is the prospective near-surface target.
BURIAL_Q = 0.05: protect the most-buried five percent without chasing single outliers.
BURIAL_CAP_M = 0.08: stay inside the ten-centimetre penetration loss authority.
BURIAL_GATE_M = 0.10: gross median burial independently refutes the depth ruler.
R_AGREE_TOL = 0.05: people disagreeing by over five percent falsify one-scale error.
HEALTHY_BAND_M = (-0.05, 0.20): an unfired person's median must remain plausibly near scene.
VERIFY_MAX_ITERS = 6: exact re-check of the chosen r; NN correspondences switch under
rescale (measured up to 23 cm on single frames), so the linear prediction only proposes —
the burial budget is enforced on ground-truth gaps, backing r off toward 1 if violated.
MIN_CORR_FRAMES / MIN_CORR_FRAC = 10 / 0.3: a clip-wide scale needs clip-wide evidence;
after grazing rejection a fired person must keep at least this many correspondences.
MIN_EFFECT_R = 0.02: a rescale within 2% of 1 rescues nothing — veto instead of
labeling the clip refuted while applying a no-op.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import brentq

from tools.geometry.normals import estimate_point_normals
from tools.smplx_robot import load_smplx_mid

from .camera import fold_root_to_world, world_from_cam
from .contacts import lift_joint_contact, scene_sdf, vertex_to_joint
from .qmath import normalize_q
from .stages import _DEFAULT_SMPLX_MODEL, _fill_invalid_frames


MIN_FRAMES = 20
MIN_FINITE_FRAC = 0.6
PRUNE_MARGIN_WIDE_M = 0.8
GRAZING_MIN_NTP_M = 0.2
TOUCH_Q = 0.25
TOUCH_EPS_M = 0.02
BURIAL_Q = 0.05
BURIAL_CAP_M = 0.08
BURIAL_GATE_M = 0.10
R_AGREE_TOL = 0.05
HEALTHY_BAND_M = (-0.05, 0.20)
VERIFY_MAX_ITERS = 6
MIN_CORR_FRAMES = 10
MIN_CORR_FRAC = 0.3
MIN_EFFECT_R = 0.02


def predicted_gaps(sdf, ntp, r):
    """Fixed-correspondence post-rescale gaps ``g(r)`` in metres."""
    return np.asarray(sdf) - (float(r) - 1.0) * np.asarray(ntp)


def predicted_quantiles(sdf, ntp, r):
    """Predicted p05/p25/p50 signed gaps for one correspondence set."""
    if len(sdf) == 0:
        return np.full(3, np.nan, np.float32)
    return np.quantile(predicted_gaps(sdf, ntp, r), [0.05, 0.25, 0.50]).astype(np.float32)


def _constraint_roots(sdf, ntp, q, threshold):
    """All sampled roots of a continuous predicted-gap quantile near its exact line roots."""
    line_roots = 1.0 + (np.asarray(sdf) - threshold) / np.asarray(ntp)
    line_roots = line_roots[np.isfinite(line_roots) & (line_roots > 0.0)]
    if line_roots.size:
        lo = max(np.finfo(np.float64).eps, float(line_roots.min()) * 0.8)
        hi = max(1.0, float(line_roots.max()) * 1.2)
    else:
        lo, hi = 0.5, 2.0
    lo, hi = min(lo, 1.0), max(hi, 1.0)
    grid = np.unique(np.r_[np.linspace(lo, hi, 4097), line_roots, 1.0])

    def f(r):
        return float(np.quantile(predicted_gaps(sdf, ntp, r), q) - threshold)

    vals = np.asarray([f(r) for r in grid])
    roots = list(grid[np.isclose(vals, 0.0, atol=1e-10)])
    crossings = np.flatnonzero(vals[:-1] * vals[1:] < 0.0)
    roots.extend(brentq(f, float(grid[i]), float(grid[i + 1])) for i in crossings)
    return np.unique(np.asarray(roots, dtype=np.float64)), lo, hi


def select_rescale(sdf, ntp):
    """Choose the closest-to-one scale satisfying touch and burial; burial wins conflicts."""
    sdf = np.asarray(sdf, dtype=np.float64)
    ntp = np.asarray(ntp, dtype=np.float64)
    touch_roots, lo_t, hi_t = _constraint_roots(sdf, ntp, TOUCH_Q, TOUCH_EPS_M)
    burial_roots, lo_b, hi_b = _constraint_roots(sdf, ntp, BURIAL_Q, -BURIAL_CAP_M)
    candidates = np.unique(np.r_[1.0, touch_roots, burial_roots,
                                 min(lo_t, lo_b), max(hi_t, hi_b)])

    def values(r):
        g = predicted_gaps(sdf, ntp, r)
        return float(np.quantile(g, TOUCH_Q)), float(np.quantile(g, BURIAL_Q))

    feasible = []
    burial_ok = []
    for r in candidates:
        touch, burial = values(r)
        if burial >= -BURIAL_CAP_M - 1e-9:
            burial_ok.append((max(touch - TOUCH_EPS_M, 0.0), abs(r - 1.0), r))
            if touch <= TOUCH_EPS_M + 1e-9:
                feasible.append((abs(r - 1.0), r))
    if feasible:
        r = min(feasible)[1]
        conflict = False
    else:
        r = min(burial_ok)[2] if burial_ok else 1.0
        conflict = True
    return float(r), predicted_quantiles(sdf, ntp, r), conflict


def _pose_init(mid, sp, person, frame_indices, wfc, device):
    """Pose one converter body with the exact initialization used by ``run_kinematic``."""
    valid_all = np.asarray(sp["valid_mask"][person], dtype=bool)
    valid = valid_all[frame_indices]
    betas_mean = torch.as_tensor(sp["betas"][person][valid_all].mean(0),
                                 dtype=torch.float32, device=device)

    def param(key):
        filled = _fill_invalid_frames(sp[key][person], valid_all)
        return torch.as_tensor(filled[frame_indices], dtype=torch.float32, device=device)

    betas_t = betas_mean[None].expand(len(frame_indices), -1)
    q_cam = mid.from_classic(betas_t, param("body_pose"), param("transl"),
                             param("global_orient"), param("left_hand_pose"),
                             param("right_hand_pose"))
    q_world = fold_root_to_world(q_cam, wfc)
    joints, vertices = [], []
    for start in range(0, len(frame_indices), 64):
        end = min(start + 64, len(frame_indices))
        out = mid.forward(betas_t[start:end], normalize_q(q_world[start:end], mid.num_joints))
        joints.append(out.joints_world.tensor()[..., :3])
        vertices.append(out.vertices)
    return betas_mean, valid, torch.cat(joints), torch.cat(vertices)


def _person_signals(mid, sp, person, frame_indices, extr, scene_pts, scene_normals,
                    scene_conf, device, float_gate_m):
    """Gate one person and retain the exact winning SDF correspondence per valid frame."""
    valid_all = np.asarray(sp["valid_mask"][person], dtype=bool)
    valid = valid_all[frame_indices]
    if not valid_all.any() or not valid.any():
        return dict(object_id=int(sp["object_ids"][person]), eligible=False, fired=False,
                    gap_m=float("nan"), burial_m=float("nan"), support=0.0,
                    n_valid=int(valid.sum()), n_finite=0,
                    frame=np.empty(0, np.int32), sdf=np.empty(0), ntp=np.empty(0),
                    r_star=np.empty(0))
    wfc = world_from_cam(extr)
    betas, _, joints, vertices = _pose_init(
        mid, sp, person, frame_indices, wfc, device)
    names = list(mid.joint_names)
    ee = torch.as_tensor([names.index(n) for n in
                          ("left_wrist", "right_wrist", "left_foot", "right_foot")],
                         dtype=torch.long, device=device)
    valid_idx = torch.nonzero(torch.as_tensor(valid, device=device), as_tuple=True)[0]

    gap = np.full(len(frame_indices), np.nan, np.float32)
    burial = np.full(len(frame_indices), np.nan, np.float32)
    if valid_idx.numel():
        for start in range(0, len(valid_idx), 32):
            idx = valid_idx[start:start + 32]
            best = None
            for s0 in range(0, scene_pts.shape[0], 100_000):   # scene-chunked: bounded memory
                d = torch.cdist(joints[idx][:, ee], scene_pts[s0:s0 + 100_000]).amin(dim=(1, 2))
                best = d if best is None else torch.minimum(best, d)
            gap[idx.cpu().numpy()] = best.cpu().numpy()

    with torch.no_grad():
        v_rest, j_rest = mid.forward_shape(betas[None])
    v2j = vertex_to_joint(v_rest[0], j_rest[0])
    sdf_v, _, _ = scene_sdf(vertices[valid_idx], scene_pts, scene_normals, scene_conf,
                            prune_margin_m=PRUNE_MARGIN_WIDE_M)
    usable = sdf_v < torch.finfo(sdf_v.dtype).max * 0.5
    _, jsdf_cm = lift_joint_contact(usable, sdf_v, v2j, mid.num_joints)
    burial[valid_idx.cpu().numpy()] = torch.nan_to_num(
        jsdf_cm[:, ee].amin(1) / 100.0, nan=float("nan")).cpu().numpy()

    ee_vertex = torch.isin(v2j, ee)
    ee_vid = torch.nonzero(ee_vertex, as_tuple=True)[0]
    win_sdf, win_local = sdf_v[:, ee_vertex].min(1)
    win_vid = ee_vid[win_local]
    corr_frame, corr_sdf, corr_ntp, corr_r = [], [], [], []
    for row, frame in enumerate(valid_idx.tolist()):
        if not bool(win_sdf[row] < torch.finfo(win_sdf.dtype).max * 0.5):
            continue
        verts_t = vertices[frame]
        lo = verts_t.amin(0) - PRUNE_MARGIN_WIDE_M
        hi = verts_t.amax(0) + PRUNE_MARGIN_WIDE_M
        mask = ((scene_pts >= lo) & (scene_pts <= hi)).all(-1)
        if not bool(mask.any()):
            continue
        scene_idx = torch.nonzero(mask, as_tuple=True)[0]
        vertex = verts_t[win_vid[row]]
        nearest = torch.cdist(vertex[None], scene_pts[mask]).argmin()
        point_idx = scene_idx[nearest]
        point, normal = scene_pts[point_idx], scene_normals[point_idx]
        R, t = extr[frame, :3, :3], extr[frame, :3, 3]
        vertex_cam = R @ vertex + t
        point_cam = R @ point + t
        normal_cam = R @ normal
        ntp = torch.dot(normal_cam, point_cam)
        # camera-oriented normals give nᵀp < 0; a positive value means a wrong-facing
        # normal, whose root is as unstable as a grazing one — reject both.
        if not bool(torch.isfinite(ntp)) or float(ntp) > -GRAZING_MIN_NTP_M:
            continue
        ntv = torch.dot(normal_cam, vertex_cam)
        corr_frame.append(frame)
        corr_sdf.append(float(win_sdf[row]))
        corr_ntp.append(float(ntp))
        corr_r.append(float(ntv / ntp))

    finite = np.isfinite(gap) & np.isfinite(burial) & valid
    n_valid = int(valid.sum())
    n_finite = int(finite.sum())
    support = n_finite / n_valid if n_valid else 0.0
    gap_med = float(np.nanmedian(gap[valid])) if np.isfinite(gap[valid]).any() else float("nan")
    burial_med = (float(np.nanmedian(burial[valid]))
                  if np.isfinite(burial[valid]).any() else float("nan"))
    eligible = n_valid >= MIN_FRAMES and support >= MIN_FINITE_FRAC
    fired = eligible and (gap_med > float_gate_m or burial_med < -BURIAL_GATE_M)
    return dict(object_id=int(sp["object_ids"][person]), eligible=eligible, fired=fired,
                gap_m=gap_med, burial_m=burial_med, support=support,
                n_valid=n_valid, n_finite=n_finite,
                frame=np.asarray(corr_frame, dtype=np.int32),
                sdf=np.asarray(corr_sdf), ntp=np.asarray(corr_ntp),
                r_star=np.asarray(corr_r))


@torch.no_grad()
def true_gap_quantiles(mid, sp, persons, frame_indices, extr, scene_pts, scene_normals,
                       scene_conf, device, r):
    """Ground-truth post-rescale gap quantiles by re-posing in the candidate world.

    Under ``s → s·r`` scene world points scale by ``r`` about the origin and camera
    translations follow (``t' = r·t``), while the init body keeps its camera-frame
    position. Normal DIRECTIONS are invariant under that uniform scaling, so the
    pre-computed normals are reused. Unlike the linear predictor this re-runs the
    full nearest-neighbour SDF, so correspondence switches are captured exactly.
    Returns pooled ``(touch_q, burial_q)`` over ``persons``: touch = per-frame min
    over the four EE joints, burial = per-frame min over ALL joints (the deepest
    body part after rescale need not be an extremity).
    """
    extr_r = extr.clone()
    extr_r[:, :3, 3] *= r
    pts_r = scene_pts * r
    wfc_r = world_from_cam(extr_r)
    names = list(mid.joint_names)
    ee = [names.index(n) for n in ("left_wrist", "right_wrist", "left_foot", "right_foot")]
    touch_all, burial_all = [], []
    for person in persons:
        valid_all = np.asarray(sp["valid_mask"][person], dtype=bool)
        valid = valid_all[frame_indices]
        if not valid_all.any() or not valid.any():
            continue
        betas, _, _, vertices = _pose_init(mid, sp, person, frame_indices, wfc_r, device)
        valid_idx = torch.nonzero(torch.as_tensor(valid, device=device), as_tuple=True)[0]
        v_rest, j_rest = mid.forward_shape(betas[None])
        v2j = vertex_to_joint(v_rest[0], j_rest[0])
        sdf_v, _, _ = scene_sdf(vertices[valid_idx], pts_r, scene_normals, scene_conf,
                                prune_margin_m=PRUNE_MARGIN_WIDE_M)
        usable = sdf_v < torch.finfo(sdf_v.dtype).max * 0.5
        _, jsdf_cm = lift_joint_contact(usable, sdf_v, v2j, mid.num_joints)
        j = (jsdf_cm / 100.0).cpu().numpy()
        with np.errstate(all="ignore"):
            touch_all.append(np.nanmin(j[:, ee], axis=1))
            burial_all.append(np.nanmin(j, axis=1))
    if not touch_all:
        return float("nan"), float("nan")
    touch = np.concatenate(touch_all)
    burial = np.concatenate(burial_all)
    touch, burial = touch[np.isfinite(touch)], burial[np.isfinite(burial)]
    if touch.size == 0 or burial.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(touch, TOUCH_Q)), float(np.quantile(burial, BURIAL_Q))


@torch.no_grad()
def verify_built_world(smplx_path, scene_points, scene_confidence, extrinsics,
                       frame_indices, fired_object_ids, *, device):
    """Final acceptance check on the candidate scene that will actually be written.

    The in-selection verifier scales the pass-1 voxel centroids, but the applied pass
    re-voxelizes raw points at the new metric — surfaces can move by up to a voxel,
    which is material against the 8 cm burial budget. Re-pose the fired init bodies
    in the BUILT candidate world (already at ``s_eff``; ``r=1`` here) and return the
    pooled ``(touch_q, burial_q)`` ground-truth gaps.
    """
    sp = dict(np.load(smplx_path, allow_pickle=True))
    sp_ids = [int(x) for x in sp["object_ids"]]
    persons = [sp_ids.index(int(o)) for o in fired_object_ids if int(o) in sp_ids]
    if not persons:
        return float("nan"), float("nan")
    pts = torch.as_tensor(scene_points, dtype=torch.float32, device=device)
    conf = torch.as_tensor(scene_confidence, dtype=torch.float32, device=device)
    extr = torch.as_tensor(extrinsics, dtype=torch.float32, device=device)
    if pts.shape[0] == 0:
        return float("nan"), float("nan")
    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    cameras = world_from_cam(extr)[:, :3, 3]
    normals = estimate_point_normals(pts, cameras, k=16)
    return true_gap_quantiles(mid, sp, persons, np.asarray(frame_indices, dtype=int),
                              extr, pts, normals, conf, device, 1.0)


@torch.no_grad()
def evaluate_body_anchor(smplx_path, scene_points, scene_confidence, extrinsics,
                         frame_indices, object_ids, *, device, float_gate_m):
    """Evaluate all tracked bodies and return the consistent clip rescale plus diagnostics."""
    sp = dict(np.load(smplx_path, allow_pickle=True))
    sp_ids = [int(x) for x in sp["object_ids"]]
    selected = [sp_ids.index(int(oid)) for oid in object_ids if int(oid) in sp_ids]
    scene_pts = torch.as_tensor(scene_points, dtype=torch.float32, device=device)
    scene_conf = torch.as_tensor(scene_confidence, dtype=torch.float32, device=device)
    extr = torch.as_tensor(extrinsics, dtype=torch.float32, device=device)
    frames = np.asarray(frame_indices, dtype=int)
    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    cameras = world_from_cam(extr)[:, :3, 3]
    normals = estimate_point_normals(scene_pts, cameras, k=16)

    people = [_person_signals(mid, sp, p, frames, extr, scene_pts, normals,
                              scene_conf, device, float_gate_m) for p in selected]
    if not people:
        return dict(rescale_r=1.0, people=[], depth_refuted=np.zeros(0, bool),
                    anchor_gap_m=np.empty(0, np.float32),
                    anchor_burial_m=np.empty(0, np.float32),
                    anchor_support=np.empty(0, np.float32),
                    anchor_pred_gap_q=np.empty((0, 3), np.float32), consistent=True,
                    reason="", burial_conflict=False,
                    anchor_true_touch=float("nan"), anchor_true_burial=float("nan"))
    fired = [p for p in people if p["fired"]]
    consistent = True
    reason = ""
    burial_conflict = False
    r = 1.0

    def _healthy_band_ok(r_check):
        for p in people:
            if not p["fired"] and len(p["sdf"]):
                med = float(np.quantile(predicted_gaps(p["sdf"], p["ntp"], r_check), 0.50))
                if not HEALTHY_BAND_M[0] <= med <= HEALTHY_BAND_M[1]:
                    return False, (f"healthy obj_id={p['object_id']} would move to median gap "
                                   f"{med:+.3f} m")
        return True, ""

    if fired:
        min_corr = [max(MIN_CORR_FRAMES, int(MIN_CORR_FRAC * p["n_valid"])) for p in fired]
        if any(len(p["sdf"]) < m for p, m in zip(fired, min_corr)):
            consistent, reason = False, (
                "a fired person keeps too few non-grazing correspondences "
                f"({[len(p['sdf']) for p in fired]} vs minima {min_corr})")
        else:
            pooled_sdf = np.concatenate([p["sdf"] for p in fired])
            pooled_ntp = np.concatenate([p["ntp"] for p in fired])
            r, _, burial_conflict = select_rescale(pooled_sdf, pooled_ntp)
            individual = [select_rescale(p["sdf"], p["ntp"])[0] for p in fired]
            if max(individual) - min(individual) > R_AGREE_TOL:
                consistent = False
                reason = f"fired persons' individual r values disagree: {individual}"
            if consistent:
                consistent, reason = _healthy_band_ok(r)
    if not consistent:
        r = 1.0
    # Exact verification: the linear prediction assumed fixed NN correspondences, which
    # switch under rescale (measured up to 23 cm on single frames). Re-check the burial
    # budget on ground-truth gaps in the candidate world; back r off toward 1 until it
    # holds. Touch is reported, not enforced (missing touch = partial rescue, safe).
    # Losing SDF support at the candidate scale (NaN), backing off into a no-op
    # (|r−1| < MIN_EFFECT_R), or leaving the healthy band at the final r all VETO the
    # rescale — an unverified or ineffective world change must never ship as "refuted".
    true_touch, true_burial = float("nan"), float("nan")
    if consistent and fired:
        if abs(r - 1.0) < MIN_EFFECT_R:
            consistent, reason = False, (
                f"selected r={r:.4f} is within {MIN_EFFECT_R:.0%} of 1 — no effective rescale"
                + (" (burial-infeasible selection)" if burial_conflict else ""))
            r = 1.0
    if consistent and fired:
        fired_persons = [sel for sel, p in zip(selected, people) if p["fired"]]
        for _ in range(VERIFY_MAX_ITERS):
            true_touch, true_burial = true_gap_quantiles(
                mid, sp, fired_persons, frames, extr, scene_pts, normals, scene_conf,
                device, r)
            if not np.isfinite(true_burial) or true_burial >= -BURIAL_CAP_M - 1e-6:
                break
            r = r + 0.5 * (1.0 - r)
        if not np.isfinite(true_burial):
            consistent, reason = False, "verification lost SDF support (no finite gaps at candidate scale)"
        elif true_burial < -BURIAL_CAP_M - 1e-6:
            consistent, reason = False, (
                f"verified burial p05 {true_burial:+.3f} m still violates the cap at r={r:.4f}")
        elif abs(r - 1.0) < MIN_EFFECT_R:
            consistent, reason = False, (
                f"verification backed r off to {r:.4f} — no effective rescale remains")
        else:
            consistent, reason = _healthy_band_ok(r)
            if consistent:
                reason = ""
        if not consistent:
            r = 1.0
    pred_q = np.stack([predicted_quantiles(p["sdf"], p["ntp"], r) for p in people])
    applied = np.asarray([p["fired"] and consistent for p in people], dtype=bool)
    return dict(rescale_r=float(r), people=people, depth_refuted=applied,
                anchor_gap_m=np.asarray([p["gap_m"] for p in people], np.float32),
                anchor_burial_m=np.asarray([p["burial_m"] for p in people], np.float32),
                anchor_support=np.asarray([p["support"] for p in people], np.float32),
                anchor_pred_gap_q=pred_q.astype(np.float32), consistent=consistent,
                reason=reason, burial_conflict=burial_conflict,
                anchor_true_touch=float(true_touch), anchor_true_burial=float(true_burial))
