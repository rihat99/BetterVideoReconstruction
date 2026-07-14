"""Per-frame SMPL-X body overfit to the metric human depth cloud.

The scale stage turns VGGT's single up-to-scale factor into a metric one, ``s``.
This module then **overfits** a fresh SMPL-X body to that scaled depth — one frame
at a time, with **free shape** (per-frame betas) and no temporal coupling. It is
the opposite of the kinematic stage, which smooths the trajectory and freezes body
size. The question it answers: how well can a body match the human depth when we
only ask it to sit on the depth and reproject to the 2D keypoints, with nothing
else pulling on it?

Everything happens in the **VGGT camera frame** (origin at the eye, +z forward),
so no world transform is needed: the SAM-3D body is already camera-frame, and the
depth cloud is the person's ``s · vggt_depth`` back-projected with an identity
extrinsic. Four costs, per frame, summed over all of a person's valid frames:

* **chamfer** — strong, robust (Geman-McClure): front-facing body vertices ↔ the
  metric depth points. This is what seats the body on the depth.
* **reproj** — Sapiens Goliath-308 keypoints, robust + score-weighted: holds the
  body on the image so chamfer can't slide it off to chase bad depth.
* **pose prior** — a *small* pull of the joint rotations toward the SAM-3D init
  pose: just enough to keep joints with no evidence from drifting, and far weaker
  than the kinematic stage's prior.
* **betas prior** — a tiny pull of the shape toward the SMPL-X mean (0): a gauge
  so the free shape cannot run away to a degenerate body.

No penetration, smoothing, contact, or extremity weighting — by design.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.geometry.backproject import backproject

from .keypoints import Correspondence
from .losses import geman_mcclure, pose_prior_loss, sapiens_reprojection_gm
from .qmath import normalize_q
from .visibility import front_facing_mask, vertex_normals


@dataclass
class OverfitWeights:
    """Loss weights + robust scales for :func:`overfit_person`."""

    chamfer: float = 1.0        # the strong depth-seating term
    reproj: float = 0.5         # 2D anchor holding the body on the image
    pose: float = 0.02          # small pull toward the init pose (relaxed)
    betas: float = 1e-3         # tiny shape gauge toward the SMPL-X mean
    gm_chamfer_m: float = 0.3   # chamfer robust scale (m): larger ⇒ more L2-like
    gm_reproj_px: float = 10.0  # reproj robust scale (px)
    score_thresh: float = 0.3   # ignore Sapiens keypoints below this score


def camera_depth_points(
    depth: torch.Tensor,      # (Hv, Wv) VGGT up-to-scale depth
    conf: torch.Tensor,       # (Hv, Wv) VGGT depth confidence
    mask: torch.Tensor,       # (Hv, Wv) bool — this person's silhouette
    K_vggt: torch.Tensor,     # (3, 3) pixel intrinsics on the VGGT grid
    scale: float,
    *,
    conf_drop_q: float,
    max_points: int,
) -> torch.Tensor:
    """Back-project ``scale · depth`` over the person mask into camera-frame points.

    Returns ``(M, 3)`` metric points (identity extrinsic ⇒ camera frame). Drops the
    lowest ``conf_drop_q`` fraction of in-mask pixels by confidence, then caps the
    count at ``max_points`` by uniform stride so the per-frame ``cdist`` stays small.
    """
    valid = mask & torch.isfinite(depth) & (depth > 0)
    if conf_drop_q > 0.0 and bool(valid.any()):
        thresh = torch.quantile(conf[valid].float(), conf_drop_q)
        valid = valid & (conf >= thresh)
    if not bool(valid.any()):
        return depth.new_zeros((0, 3))
    eye = torch.eye(4, dtype=depth.dtype, device=depth.device)
    pts = backproject(scale * depth, K_vggt, eye, mask=valid)  # (M, 3)
    if pts.shape[0] > max_points:
        stride = pts.shape[0] / float(max_points)
        idx = (torch.arange(max_points, device=pts.device) * stride).long().clamp_max(pts.shape[0] - 1)
        pts = pts[idx]
    return pts


def robust_chamfer(
    vertices_sub: torch.Tensor,   # (T, Ns, 3) subsampled posed vertices (grad)
    vismask: torch.Tensor,        # (T, Ns) bool — front-facing
    cloud: torch.Tensor,          # (T, Md, 3) per-frame depth points, padded
    cloud_valid: torch.Tensor,    # (T, Md) bool — real (vs padding) points
    *,
    gm_scale_m: float,
    chunk: int = 16,
) -> torch.Tensor:
    """Bidirectional Geman-McClure chamfer between visible vertices and depth points.

    Residuals below ``gm_scale_m`` pull ~quadratically; gross outliers (spaghetti
    depth, occluded vertices) saturate with vanishing gradient, so they cannot yank
    the body. Averaged over frames that have both a visible vertex and a depth point.
    Computed in frame-chunks (one batched ``cdist`` each) to bound memory.
    """
    num_frames = vertices_sub.shape[0]
    if cloud.shape[1] == 0:
        return vertices_sub.new_zeros(())
    big = vertices_sub.new_tensor(1.0e6)          # finite fill ⇒ no inf/nan in masked slots
    s2 = vertices_sub.new_tensor(gm_scale_m ** 2)
    n_vis = vismask.sum(1)                         # (T,) visible verts / frame
    n_pts = cloud_valid.sum(1)                     # (T,) depth points / frame
    frame_ok = (n_vis > 0) & (n_pts > 0)
    parts = []
    for start in range(0, num_frames, chunk):
        end = min(start + chunk, num_frames)
        dist = torch.cdist(vertices_sub[start:end], cloud[start:end])          # (b, Ns, Md)
        v2d = dist.masked_fill(~cloud_valid[start:end][:, None, :], big).amin(2)  # (b, Ns)
        d2v = dist.masked_fill(~vismask[start:end][:, :, None], big).amin(1)      # (b, Md)
        gm_v = geman_mcclure(v2d.square(), s2)
        gm_d = geman_mcclure(d2v.square(), s2)
        v_mean = (gm_v * vismask[start:end]).sum(1) / n_vis[start:end].clamp_min(1)
        d_mean = (gm_d * cloud_valid[start:end]).sum(1) / n_pts[start:end].clamp_min(1)
        parts.append(v_mean + d_mean)                                          # (b,)
    terms = torch.where(frame_ok, torch.cat(parts), vertices_sub.new_zeros(()))
    return terms.sum() / frame_ok.sum().clamp_min(1)


def _front_facing_sub(
    vertices: torch.Tensor, faces: torch.Tensor, sub: torch.Tensor, chunk: int,
) -> torch.Tensor:
    """Front-facing mask on the subsampled vertices ``(T, Ns)``, computed detached.

    Visibility only *gates* which vertices chamfer against the depth (the depth sees
    the front surface), so the mask carries no gradient. Normals need face
    connectivity, hence the full mesh; done in frame-chunks under ``no_grad`` to keep
    the transient normal buffers small.
    """
    masks = []
    with torch.no_grad():
        for start in range(0, vertices.shape[0], chunk):
            end = min(start + chunk, vertices.shape[0])
            normals = vertex_normals(vertices[start:end], faces)               # (b, V, 3)
            masks.append(front_facing_mask(vertices[start:end], normals)[:, sub])
    return torch.cat(masks, 0)


@dataclass
class OverfitResult:
    """Output of :func:`overfit_person`."""

    betas: torch.Tensor   # (T, 10) per-frame shape
    q: torch.Tensor       # (T, 211) per-frame normalized config (camera frame)
    before: dict          # loss terms at init
    after: dict           # loss terms at the end


def overfit_person(
    mid,
    betas_init: torch.Tensor,     # (T, 10)
    q_init: torch.Tensor,         # (T, 211) camera-frame config from SAM-3D
    *,
    K_orig: torch.Tensor,         # (T, 3, 3) pixel intrinsics at original resolution
    cloud: torch.Tensor,          # (T, Md, 3) camera-frame metric depth points, padded
    cloud_valid: torch.Tensor,    # (T, Md) bool
    target_kp: torch.Tensor,      # (T, 308, 2) Sapiens keypoints (original-res px)
    scores: torch.Tensor,         # (T, 308) Sapiens detection scores
    corr: Correspondence,
    frame_vertex_ids: torch.Tensor,
    sub: torch.Tensor,            # (Ns,) chamfer vertex subsample
    weights: OverfitWeights,
    iters: int,
    lr: float,
    chunk: int = 16,
) -> OverfitResult:
    """Overfit one person's per-frame SMPL-X body to the depth cloud + 2D keypoints.

    Optimizes ``betas (T, 10)`` and ``q (T, 211)`` as independent leaves with Adam.
    Frames are batched together but never coupled — the loss has no cross-frame term.
    """
    num_joints = mid.num_joints
    num_frames = q_init.shape[0]
    device = q_init.device
    identity = torch.eye(4, device=device).expand(num_frames, 4, 4)
    q0 = normalize_q(q_init, num_joints).detach()

    betas = betas_init.clone().requires_grad_(True)
    q = q_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([betas, q], lr=lr)

    def losses(record: bool) -> tuple[torch.Tensor, dict]:
        q_norm = normalize_q(q, num_joints)
        out = mid.forward(betas, q_norm)
        vertices = out.vertices                                   # (T, V, 3)
        joints = out.joints_world.tensor()[..., :3]               # (T, J, 3)
        front = _front_facing_sub(vertices, mid.faces, sub, chunk)
        chamfer = robust_chamfer(vertices[:, sub], front, cloud, cloud_valid,
                                 gm_scale_m=weights.gm_chamfer_m, chunk=chunk)
        reproj = sapiens_reprojection_gm(
            joints, vertices, identity, K_orig, target_kp, scores, corr, frame_vertex_ids,
            score_thresh=weights.score_thresh, gm_scale_px=weights.gm_reproj_px)
        pose = pose_prior_loss(q_norm, q0, num_joints)
        betas_reg = betas.square().mean()
        total = (weights.chamfer * chamfer + weights.reproj * reproj
                 + weights.pose * pose + weights.betas * betas_reg)
        terms = {}
        if record:
            terms = {"chamfer": float(chamfer), "reproj": float(reproj),
                     "pose": float(pose), "betas": float(betas_reg)}
        return total, terms

    with torch.no_grad():
        _, before = losses(record=True)
    for _ in range(iters):
        opt.zero_grad()
        total, _ = losses(record=False)
        total.backward()
        opt.step()
    with torch.no_grad():
        _, after = losses(record=True)

    return OverfitResult(
        betas=betas.detach(),
        q=normalize_q(q, num_joints).detach(),
        before=before,
        after=after,
    )
