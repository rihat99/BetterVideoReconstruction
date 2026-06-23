"""Per-person world-frame kinematic optimization (autograd L-BFGS).

The leaf variable is ``q (T, 211)`` in ``better_human`` order; quaternions are
normalized inside every closure (so there is no exp-retraction singularity).
Each closure: ``better_human`` forward → world joints / vertices → the four
losses (reprojection, chamfer, pose prior, smoothness). Visibility for chamfer
is recomputed detached each call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .camera import project, transform_points
from .losses import (
    chamfer_loss, pose_prior_loss, reprojection_loss, root_orient_prior_loss,
    smoothness_loss,
)
from .qmath import normalize_q
from .visibility import front_facing_mask, vertex_normals


@dataclass
class Metrics:
    reproj_px: float
    chamfer_m: float
    accel: float
    jerk: float
    scale: float = 1.0

    def __str__(self) -> str:
        return (f"reproj={self.reproj_px:.2f}px  chamfer={self.chamfer_m * 100:.2f}cm  "
                f"‖accel‖={self.accel:.4f}  ‖jerk‖={self.jerk:.4f}  scale={self.scale:.3f}")


def _dof_mask(active: tuple[str, ...], nq: int, device) -> torch.Tensor:
    """``(nq,)`` 0/1 mask selecting the ``q`` DOFs trainable this stage.

    ``q`` packs the root translation (``[:3]``), the root quaternion (``[3:7]``),
    then the parent-local joint quaternions (``[7:]``). Freezing a DOF group is
    done by zeroing its gradient after backward, so only the named groups move.
    """
    m = torch.zeros(nq, device=device)
    if "transl" in active:
        m[0:3] = 1.0
    if "orient" in active:
        m[3:7] = 1.0
    if "joints" in active:
        m[7:] = 1.0
    return m


def _scale_about_root(jw: torch.Tensor, vw: torch.Tensor, scale: torch.Tensor):
    """Uniformly scale joints + vertices about the pelvis (joint 0).

    ``root + s·(X − root)`` leaves the pelvis fixed and grows / shrinks the whole
    body (skeleton *and* mesh) by ``s`` — exactly equivalent to scaling every bone
    length and the posed mesh uniformly. The pose (joint rotations) is untouched,
    so a global metric size mismatch between body and depth cloud is absorbed by
    one scalar instead of by limb-stretching pose deformation.
    """
    root = jw[:, :1, :]
    return root + scale * (jw - root), root + scale * (vw - root)


def _visibility(vertices_sub_det: torch.Tensor, normals_sub: torch.Tensor,
                extrinsics: torch.Tensor) -> torch.Tensor:
    """Front-facing mask ``(T, Ns)`` for already-subsampled vertices."""
    vcam = transform_points(vertices_sub_det, extrinsics)
    ncam = torch.einsum("tij,tvj->tvi", extrinsics[:, :3, :3], normals_sub)
    return front_facing_mask(vcam, ncam)


@torch.no_grad()
def _metrics(mid, model, betas, q_world, scale, *, extrinsics, K, depth_points, target_kp,
            scores, corr, frame_vertex_ids, sub, score_thresh, remap) -> Metrics:
    qn = normalize_q(q_world, mid.num_joints)
    out = mid.forward(betas, qn)
    jw = out.joints_world.tensor()[..., :3]
    vw = out.vertices
    jw, vw = _scale_about_root(jw, vw, scale)
    # reprojection in raw pixels (un-normalized) over scored keypoints
    body = jw[:, corr.joint_model, :]
    lm = vw[:, frame_vertex_ids[corr.lm_frame], :]
    pred_px = project(transform_points(torch.cat([body, lm], 1), extrinsics), K)
    tgt = target_kp[:, corr.sap, :]
    w = (scores[:, corr.sap] >= score_thresh).float()
    err = (pred_px - tgt).norm(dim=-1)
    reproj = float((w * err).sum() / w.sum().clamp_min(1.0))
    # chamfer
    n = vertex_normals(vw, mid.faces)
    vismask = _visibility(vw[:, sub], n[:, sub], extrinsics)
    cham = float(chamfer_loss(vw[:, sub], vismask, depth_points))
    # accel / jerk norms (raw tangent diffs)
    qm = remap(qn)
    d1 = model.difference(qm[:-1], qm[1:])
    accel = d1[1:] - d1[:-1]
    jerk = accel[1:] - accel[:-1]
    return Metrics(reproj, cham, float(accel.norm()), float(jerk.norm()), float(scale))


def optimize_person(
    mid, model, betas_mean: torch.Tensor, q_world_init: torch.Tensor, *,
    extrinsics: torch.Tensor, K: torch.Tensor, depth_points: list[torch.Tensor],
    target_kp: torch.Tensor, scores: torch.Tensor, corr, frame_vertex_ids: torch.Tensor,
    sub: torch.Tensor, cfg: dict, remap,
) -> tuple[torch.Tensor, float, Metrics, Metrics]:
    """Optimize one person. Returns ``(q_world_opt, scale_opt, before, after)``."""
    T = q_world_init.shape[0]
    betas = betas_mean[None].expand(T, -1).contiguous()
    w, sm, oc = cfg["weights"], cfg["smoothness"], cfg["optimizer"]
    score_thresh = float(cfg["keypoints"]["score_thresh"])
    robust_px = float(cfg["keypoints"].get("robust_px", 0.0))
    w_scale = float(w.get("scale_prior", 5.0))
    w_orient = float(w.get("orient_prior", 50.0))

    mk = lambda q, s: _metrics(
        mid, model, betas, q, s, extrinsics=extrinsics, K=K, depth_points=depth_points,
        target_kp=target_kp, scores=scores, corr=corr, frame_vertex_ids=frame_vertex_ids,
        sub=sub, score_thresh=score_thresh, remap=remap)
    one = q_world_init.new_tensor(1.0)
    before = mk(q_world_init, one)

    q_init = q_world_init.detach().clone()
    q = q_init.clone().requires_grad_(True)
    # One global scale per person (init 1.0), optimized jointly. It absorbs the
    # metric size mismatch between the SMPL-X body and the depth cloud so chamfer
    # need not deform the pose; a gentle prior keeps it near 1 when depth is scarce.
    scale = q_init.new_tensor(float(oc.get("scale_init", 1.0))).requires_grad_(True)

    # Coarse-to-fine to dodge the local minimum single-stage Adam falls into:
    # when all DOFs move at once, the joints deform to chase the (sometimes
    # wholesale-wrong-depth) chamfer cloud and the body ends up off-image AND
    # contorted. Stage 1 places the body *rigidly* — root translation + global
    # scale only, pose frozen at the converter's good per-frame estimate — so it
    # finds the right depth/size without any articulation. Stage 2 unfreezes
    # orientation + joints and refines locally from that good placement. DOFs are
    # frozen by zeroing their q-gradient; each stage gets a fresh Adam so no
    # stale momentum leaks across the unfreeze.
    nq = q.shape[1]
    iters2 = int(oc["iters"])
    stages = (
        ("placement", _dof_mask(("transl",), nq, q.device), int(oc.get("iters_stage1", 150))),
        ("full", _dof_mask(("transl", "orient", "joints"), nq, q.device), iters2),
    )

    # Adam (not L-BFGS): the per-coordinate adaptive step is robust to the very
    # different gradient scales here (root translation in metres vs joint
    # quaternions, pixel reprojection vs metre chamfer) that stall L-BFGS.
    for name, mask, n_iter in stages:
        opt = torch.optim.Adam([
            {"params": [q], "lr": float(oc["lr"])},
            {"params": [scale], "lr": float(oc.get("lr_scale", oc["lr"]))},
        ])
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[n_iter // 2, 3 * n_iter // 4], gamma=float(oc["lr_decay"]))
        for it in range(n_iter):
            opt.zero_grad()
            qn = normalize_q(q, mid.num_joints)
            out = mid.forward(betas, qn)
            jw = out.joints_world.tensor()[..., :3]
            vw = out.vertices
            jw, vw = _scale_about_root(jw, vw, scale)
            with torch.no_grad():
                normals = vertex_normals(vw.detach(), mid.faces)
                vismask = _visibility(vw.detach()[:, sub], normals[:, sub], extrinsics)
            L_rep = reprojection_loss(jw, vw, extrinsics, K, target_kp, scores, corr,
                                      frame_vertex_ids, score_thresh=score_thresh, robust_px=robust_px)
            L_ch = chamfer_loss(vw[:, sub], vismask, depth_points)
            L_pp = pose_prior_loss(qn, q_init, mid.num_joints)
            L_ro = root_orient_prior_loss(qn, q_init)
            L_sm = smoothness_loss(remap(qn), model, accel_root=float(sm["accel_root"]),
                                   accel_joint=float(sm["accel_joint"]), jerk_root=float(sm["jerk_root"]),
                                   jerk_joint=float(sm["jerk_joint"]))
            L_sc = (scale - 1.0).square()
            loss = (float(w["reprojection"]) * L_rep + float(w["chamfer"]) * L_ch
                    + float(w["pose_prior"]) * L_pp + w_orient * L_ro + L_sm + w_scale * L_sc)
            loss.backward()
            q.grad.mul_(mask)  # freeze the DOF groups not active this stage
            opt.step()
            sched.step()
            if it % max(1, n_iter // 4) == 0 or it == n_iter - 1:
                print(f"    [{name:9s}] it {it:4d}  loss={loss.item():.4f}  "
                      f"(rep={L_rep.item():.2f}px ch={L_ch.item() * 100:.1f}cm "
                      f"pose={L_pp.item():.4f} smooth={L_sm.item():.3f} scale={scale.item():.3f})")

    with torch.no_grad():
        q_opt = normalize_q(q.detach(), mid.num_joints)
        s_opt = scale.detach()
    return q_opt, float(s_opt), before, mk(q_opt, s_opt)
