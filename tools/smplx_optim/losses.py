"""The four kinematic-stage losses: reprojection, chamfer, pose prior, smoothness.

All operate on a single person's world-frame trajectory. ``q`` tensors are
expected pre-normalized (see :func:`tools.smplx_optim.qmath.normalize_q`).
Posed joints / vertices come from ``better_human``; this module only assembles
the costs.
"""

from __future__ import annotations

import torch

from .camera import project, transform_points
from .keypoints import Correspondence


def reprojection_loss(
    joints_world: torch.Tensor,    # (T, J, 3)
    vertices_world: torch.Tensor,  # (T, V, 3)
    extrinsics: torch.Tensor,      # (T, 4, 4) cam_from_world
    K: torch.Tensor,               # (T, 3, 3) pixel intrinsics
    target_kp: torch.Tensor,       # (T, 308, 2) pixel
    scores: torch.Tensor,          # (T, 308)
    corr: Correspondence,
    frame_vertex_ids: torch.Tensor,  # (num_frames,) long — model landmark verts
    *,
    score_thresh: float,
    robust_px: float = 0.0,
) -> torch.Tensor:
    """Score-weighted reprojection error of matched keypoints, in **pixels**.

    Robust L2 distance per keypoint (not squared) so a few bad detections don't
    dominate. Keypoints below ``score_thresh`` are dropped; survivors are
    weighted by their detection score. Units are pixels, so the config weight is
    "cost per pixel of error" — directly comparable to the metre-unit chamfer.

    ``robust_px`` (when > 0) truncates the per-keypoint error: a keypoint off by
    more than ``robust_px`` contributes a constant cost and **zero gradient**.
    With the keypoint correspondence correct (Goliath-308, see
    :mod:`tools.smplx_optim.keypoints`) well-tracked joints reproject at <~12px,
    so this acts as an **outlier guard** — it stops a grossly mis-detected or
    occluded keypoint on a bad frame from yanking the pose, without affecting the
    reliable correspondences.
    """
    body = joints_world[:, corr.joint_model, :]                 # (T, Kb, 3)
    lm = vertices_world[:, frame_vertex_ids[corr.lm_frame], :]  # (T, Kl, 3)
    pred_world = torch.cat([body, lm], dim=1)                   # (T, K, 3)

    pred_px = project(transform_points(pred_world, extrinsics), K)  # (T, K, 2)
    tgt = target_kp[:, corr.sap, :]                            # (T, K, 2)
    w = scores[:, corr.sap]                                    # (T, K)
    w = torch.where(w >= score_thresh, w, torch.zeros_like(w))

    dist = (pred_px - tgt).norm(dim=-1)                        # (T, K) pixels
    if robust_px > 0.0:
        dist = dist.clamp(max=robust_px)
    return (w * dist).sum() / w.sum().clamp_min(1.0)


def chamfer_loss(
    vertices_sub: torch.Tensor,          # (T, Ns, 3) subsampled posed vertices
    vismask: torch.Tensor,               # (T, Ns) bool — front-facing
    depth_points: list[torch.Tensor],    # per-frame (Nd, 3) world points
) -> torch.Tensor:
    """Bidirectional chamfer between visible vertices and depth points.

    Averaged over frames that have both visible vertices and depth points.
    Gradients flow through ``vertices_sub``; ``vismask`` and ``depth_points``
    are constants for a given closure call.
    """
    terms = []
    for t, D in enumerate(depth_points):
        if D.shape[0] == 0:
            continue
        Vv = vertices_sub[t][vismask[t]]          # (Nvis, 3)
        if Vv.shape[0] == 0:
            continue
        d = torch.cdist(Vv, D)                    # (Nvis, Nd)
        terms.append(d.min(dim=1).values.mean() + d.min(dim=0).values.mean())
    if not terms:
        return vertices_sub.new_zeros(())
    return torch.stack(terms).mean()


def pose_prior_loss(
    q: torch.Tensor,        # (T, nq) normalized, bh order
    q_init: torch.Tensor,   # (T, nq) normalized, bh order
    num_joints: int,
) -> torch.Tensor:
    """Smooth quaternion distance ``1 − (q·q_init)²`` over joints (not root).

    Excludes the root SE3 (``q[:7]``): the body / hand pose stays near the
    converter's per-frame estimate while root translation + orientation move
    freely.
    """
    shape = q.shape[:-1]
    qj = q[..., 7:].reshape(*shape, num_joints - 1, 4)
    qij = q_init[..., 7:].reshape(*shape, num_joints - 1, 4)
    cos = (qj * qij).sum(dim=-1)
    return (1.0 - cos.square()).mean()


def root_orient_prior_loss(
    q: torch.Tensor,        # (T, nq) normalized, bh order
    q_init: torch.Tensor,   # (T, nq) normalized, bh order
) -> torch.Tensor:
    """Smooth quaternion distance ``1 − (q·q_init)²`` on the ROOT orientation only.

    The root SE3 (``q[:7]``) is excluded from :func:`pose_prior_loss` so the body
    can be freely placed + oriented by reprojection + chamfer. But that leaves the
    root **orientation** (``q[3:7]``) undamped at the clip boundary — the
    finite-difference smoothness stencil constrains the end frames the least — so
    it can run away (~90°) and break the image fit. A *gentle* pull toward the
    converter's per-frame orientation (which already reprojects at ~5px) stops the
    runaway without preventing legitimate few-degree refinement. Root translation
    (``q[:3]``) stays free, so depth placement is unaffected.
    """
    cos = (q[..., 3:7] * q_init[..., 3:7]).sum(dim=-1)
    return (1.0 - cos.square()).mean()


def smoothness_loss(
    q_model: torch.Tensor,   # (T, nq) normalized, MODEL (DFS) order
    model,
    *,
    accel_root: float,
    accel_joint: float,
    jerk_root: float,
    jerk_joint: float,
) -> torch.Tensor:
    """Acceleration + jerk penalty via exact Lie ``model.difference``.

    Penalizes raw frame-to-frame tangent finite-differences (the constant
    ``dt`` factors are folded into the weights — equivalent for a fixed-fps
    clip and far better conditioned than ``/dt³`` jerk on a jittery init):
    ``d1 = q_{t+1} ⊖ q_t`` (velocity), ``accel = Δd1`` (jerk-free 2nd diff),
    ``jerk = Δaccel``. Root tangent is the first 6 components (se3), joints the
    remaining 153 — weighted separately since the root carries the global
    trajectory and the joints the articulation.
    """
    d1 = model.difference(q_model[:-1], q_model[1:])   # (T-1, nv) — velocity
    accel = d1[1:] - d1[:-1]                            # (T-2, nv) — acceleration
    jerk = accel[1:] - accel[:-1]                       # (T-3, nv) — jerk

    ar, aj = accel[:, :6], accel[:, 6:]
    jr, jj = jerk[:, :6], jerk[:, 6:]
    return (
        accel_root * ar.square().mean()
        + accel_joint * aj.square().mean()
        + jerk_root * jr.square().mean()
        + jerk_joint * jj.square().mean()
    )
