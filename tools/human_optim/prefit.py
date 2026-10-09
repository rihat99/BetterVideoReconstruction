"""Pre-fit: align the SMPL-X body to its fused depth points at a fixed scale.

One half of the scale stage's **dual** optimization (see docs/scale.md). The
converter's body can sit a little off the person's depth points — translated
along the ray, root slightly rotated, limbs not quite on the surface — and that
misalignment leaks straight into the measured world scale. The scale stage
therefore alternates two moves until they agree:

* a **body round** (this module): at a fixed scale ``s``, refine each person's
  per-frame pose so the body sits on its depth points;
* a **scale re-measurement** (the stage's human ruler): with the bodies
  realigned, re-read the metric factor from the body-vs-cloud depth ratio.

The body round optimizes ``q (T, 211)`` — root translation and orientation free,
joint rotations allowed but pulled back toward the converter pose (bigger
changes cost more), fingers frozen (no depth evidence at cloud resolution).
Shape is **frozen to one canonical betas per person**: the body's size is the
ruler, so letting it stretch inside a scale optimization would be circular.
Four terms:

* **chamfer** — one-sided, robust: every depth point must be explained by a
  nearby *visible* body vertex. The reverse direction is deliberately absent —
  the cloud only sees the person's front, and pulling occluded vertices into
  the observed patch is how bodies collapse.
* **reproj** — Sapiens Goliath-308 keypoints hold the body on the image, so the
  chamfer resolves along the viewing ray instead of sliding sideways.
* **pose** — quadratic pull of the joint rotations toward the converter pose.
* **smooth** — a tiny velocity-matching tie on the posed joints: consecutive
  frames should move like the converter's did (suppresses per-frame jitter
  without fighting real motion).

The refined bodies double as the kinematic stage's starting pose
(``geometry/prefit.npz``). Everything runs in the VGGT camera frame, like the
converter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from better_robot.optim import Phase, Problem, run

from tools.geometry.human_scale import weighted_median

from .keypoints import Correspondence
from .losses import geman_mcclure, pose_prior_loss, root_orient_prior_loss, sapiens_reprojection_gm
from .qmath import normalize_q
from .solve import State, adam, add_term, phase_weights, term_logger
from .visibility import front_facing_mask, vertex_normals

# Joints 22..51 are the fingers (mid.joint_names); their quaternion blocks are
# the contiguous tail of q, starting at 7 + 4 * (22 - 1).
_FINGER_Q_START = 7 + 4 * 21


@dataclass
class PrefitWeights:
    """Loss weights + robust scales for :func:`prefit_person_round`."""

    chamfer: float = 1.0        # depth points must be explained by visible verts
    reproj: float = 0.5         # 2D anchor holding the body on the image
    pose: float = 2.0           # quadratic pull toward the converter pose
    orient: float = 1.0         # gentle root-orientation pull (stops ~180° flips)
    transl: float = 0.5         # gentle root-translation pull (m²) — see the term
    smooth: float = 0.1         # tiny converter-velocity tie on posed joints
    gm_chamfer_m: float = 0.3   # chamfer robust scale (m): larger ⇒ more L2-like
    gm_reproj_px: float = 10.0  # reproj robust scale (px)
    score_thresh: float = 0.3   # ignore Sapiens keypoints below this score


def cloud_chamfer(
    vertices_sub: torch.Tensor,   # (T, Ns, 3) subsampled posed vertices (grad)
    vismask: torch.Tensor,        # (T, Ns) bool — front-facing
    cloud: torch.Tensor,          # (T, Md, 3) per-frame depth points, padded
    cloud_valid: torch.Tensor,    # (T, Md) bool — real (vs padding) points
    *,
    gm_scale_m: float,
    chunk: int = 16,
    frame_weights: torch.Tensor | None = None,   # (T,) per-frame depth-quality weight
) -> torch.Tensor:
    """One-sided Geman-McClure chamfer: each depth point → nearest visible vertex.

    Residuals below ``gm_scale_m`` pull ~quadratically; gross outliers saturate
    with vanishing gradient. Averaged over frames that have both a visible
    vertex and a depth point; frame-chunked ``cdist`` bounds memory.
    ``frame_weights`` makes that average a weighted one (the depth-smear hook,
    :mod:`tools.geometry.depth_smear`); ``None`` is the plain mean.
    """
    num_frames = vertices_sub.shape[0]
    if cloud.shape[1] == 0:
        return vertices_sub.new_zeros(())
    big = vertices_sub.new_tensor(1.0e6)          # finite fill ⇒ no inf/nan in masked slots
    s2 = vertices_sub.new_tensor(gm_scale_m ** 2)
    n_pts = cloud_valid.sum(1)                     # (T,) depth points / frame
    frame_ok = (vismask.sum(1) > 0) & (n_pts > 0)
    parts = []
    for start in range(0, num_frames, chunk):
        end = min(start + chunk, num_frames)
        dist = torch.cdist(vertices_sub[start:end], cloud[start:end])            # (b, Ns, Md)
        d2v = dist.masked_fill(~vismask[start:end][:, :, None], big).amin(1)     # (b, Md)
        gm = geman_mcclure(d2v.square(), s2)
        parts.append((gm * cloud_valid[start:end]).sum(1) / n_pts[start:end].clamp_min(1))
    terms = torch.where(frame_ok, torch.cat(parts), vertices_sub.new_zeros(()))
    if frame_weights is None:
        return terms.sum() / frame_ok.sum().clamp_min(1)
    fw = frame_weights * frame_ok
    return (terms * fw).sum() / fw.sum().clamp_min(1e-6)


def joint_velocity_loss(
    joints: torch.Tensor,    # (T, J, 3) posed joints (grad)
    joints0: torch.Tensor,   # (T, J, 3) converter-pose joints (detached)
    pairs: torch.Tensor,     # (P,) long — indices i where frames i, i+1 are consecutive
) -> torch.Tensor:
    """Match the converter's per-joint velocities on truly consecutive frames.

    Position-space (no quaternion sign issues), and a *velocity* target rather
    than zero motion, so real movement is untouched — only per-frame jitter
    relative to the init trajectory is damped.
    """
    if pairs.numel() == 0:
        return joints.new_zeros(())
    v = joints[pairs + 1] - joints[pairs]
    v0 = joints0[pairs + 1] - joints0[pairs]
    return (v - v0).square().sum(-1).mean()


def finger_freeze_mask(nq: int, device) -> torch.Tensor:
    """Gradient mask over ``q``: ones everywhere, zeros on the finger tail."""
    mask = torch.ones(nq, device=device)
    mask[_FINGER_Q_START:] = 0.0
    return mask


def _valid_spans(valid: np.ndarray) -> list[tuple[int, int]]:
    """The contiguous ``[start, end)`` runs of True in a ``(T,)`` bool mask."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], valid.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def smooth_finger_quats(q: torch.Tensor, valid: np.ndarray, sigma_frames: float) -> torch.Tensor:
    """Gaussian-smooth the finger quaternions of ``q (T, 211)`` along time, in place.

    The converter's per-frame hand pose is reasonable but jittery, and the fingers are
    frozen during the fit, so that jitter survives into the result unless it is removed
    here. Quaternions are hemisphere-aligned first (``q`` and ``−q`` are the same
    rotation, so a raw sign flip would average to nonsense), then each component is
    Gaussian-filtered over time and renormalized — the standard cheap approximation,
    accurate at the per-frame angles a hand actually moves through.

    Each contiguous run of ``valid`` frames is filtered on its own: untracked frames
    hold a copy of their nearest valid pose (:func:`~tools.human_optim.stages._fill_invalid_frames`),
    and averaging that plateau into real frames would drag them. The span edges use a
    replicated border, so the first and last frames keep their value instead of decaying.

    :param q: pose ``(T, 211)``; the finger block ``q[:, 91:]`` is overwritten.
    :param valid: ``(T,)`` bool — the frames the tracker actually saw.
    :param sigma_frames: Gaussian sigma in frames; ``<= 0`` is a no-op.
    """
    if sigma_frames <= 0:
        return q
    for start, end in _valid_spans(valid):
        num = end - start
        if num < 3:
            continue
        radius = min(int(round(3.0 * sigma_frames)), num - 1)
        offsets = torch.arange(-radius, radius + 1, device=q.device, dtype=q.dtype)
        kernel = torch.exp(-0.5 * (offsets / sigma_frames).square())
        track = q[start:end, _FINGER_Q_START:].reshape(num, -1, 4)
        dots = (track[1:] * track[:-1]).sum(-1)                            # (num-1, J)
        signs = torch.where(dots < 0, -1.0, 1.0).cumprod(0)                # sequential flip
        aligned = torch.cat([track[:1], track[1:] * signs[..., None]], 0)
        channels = aligned.reshape(num, -1).T[None]                        # (1, 4J, num)
        padded = torch.nn.functional.pad(channels, (radius, radius), mode="replicate")
        weights = (kernel / kernel.sum()).view(1, 1, -1).expand(channels.shape[1], 1, -1)
        smoothed = torch.nn.functional.conv1d(padded, weights, groups=channels.shape[1])
        out = smoothed[0].T.reshape(num, -1, 4)
        out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        q[start:end, _FINGER_Q_START:] = out.reshape(num, -1)
    return q


def _front_facing_sub(
    vertices: torch.Tensor, faces: torch.Tensor, sub: torch.Tensor, chunk: int,
) -> torch.Tensor:
    """Front-facing mask on the subsampled vertices ``(T, Ns)``, computed detached.

    Visibility only *gates* which vertices the depth points may match (the depth
    sees the front surface); it carries no gradient. Recomputed at the start of
    every body round from the *current* pose, so a root rotation in an earlier
    round cannot leave the mask stale.
    """
    masks = []
    with torch.no_grad():
        for start in range(0, vertices.shape[0], chunk):
            end = min(start + chunk, vertices.shape[0])
            normals = vertex_normals(vertices[start:end], faces)               # (b, V, 3)
            masks.append(front_facing_mask(vertices[start:end], normals)[:, sub])
    return torch.cat(masks, 0)


def prefit_person_round(
    mid,
    betas: torch.Tensor,          # (T, 10) canonical shape, repeated (frozen)
    q_init: torch.Tensor,         # (T, 211) current pose (this round's start)
    q0n: torch.Tensor,            # (T, 211) converter pose, normalized (the prior)
    *,
    s0: float,                    # this round's starting scale
    cloud: torch.Tensor,          # (T, Md, 3) camera-frame depth points, UNSCALED, padded
    cloud_valid: torch.Tensor,    # (T, Md) bool
    K_orig: torch.Tensor,         # (T, 3, 3) pixel intrinsics at original resolution
    target_kp: torch.Tensor,      # (T, 308, 2) Sapiens keypoints (original-res px)
    scores: torch.Tensor,         # (T, 308) Sapiens detection scores
    corr: Correspondence,
    frame_vertex_ids: torch.Tensor,
    sub: torch.Tensor,            # (Ns,) chamfer vertex subsample
    pairs: torch.Tensor,          # (P,) long — consecutive-frame pair starts
    weights: PrefitWeights,
    iters: int,
    lr: float,
    chunk: int = 16,
    freeze_scale: bool = False,
    cloud_frame_weights: torch.Tensor | None = None,   # (T,) depth-smear weights
) -> tuple[torch.Tensor, float]:
    """One body-alignment round; returns ``(refined q (T, 211), fitted scale)``.

    The scale is a leaf **inside** the round (``log s``, so it is always
    positive), not a constant: body depth and world scale are one soft
    direction — the chamfer cannot tell "body nearer" from "world smaller" —
    and if the round pinned the cloud at a fixed scale, the body could move
    toward the image evidence only by the small chamfer-vs-reproj equilibrium
    offset per round (measured: a constant ~0.4%/round crawl that never
    terminates). Optimized jointly, the tug-of-war resolves within the round:
    reprojection decides the body's absolute depth, and the scale follows the
    body.

    ``freeze_scale`` pins ``log s`` at ``s0`` (used from round 2 on): the world
    has ONE scale, so after the free first round finds each person's depth, the
    later rounds align every body to the driver's pooled consensus instead of
    letting per-person scales drift apart.
    """
    num_joints = mid.num_joints
    num_frames = q_init.shape[0]
    device = q_init.device
    identity = torch.eye(4, device=device).expand(num_frames, 4, 4)

    def _forward(q, log_s) -> State:
        st = State()
        q_norm = normalize_q(q[0], num_joints)
        out = mid.forward(betas, q_norm)
        st.put("qn", q_norm)
        st.put("vertices", out.vertices)                          # (T, V, 3)
        st.put("joints", out.joints_world.tensor()[..., :3])      # (T, J, 3)
        st.put("log_s", log_s[0])
        return st

    problem = Problem({"q": q_init[None], "log_s": torch.tensor([[float(np.log(s0))]], device=device)},
                      time=("q",))
    problem.cached(_forward, name="st", reads=("q", "log_s"))

    with torch.no_grad():
        out0 = mid.forward(betas, normalize_q(q_init, num_joints))
        front = _front_facing_sub(out0.vertices, mid.faces, sub, chunk)
        joints0 = mid.forward(betas, q0n).joints_world.tensor()[..., :3].detach()

    def _chamfer(st: State) -> torch.Tensor:
        # Evaluated in CLOUD units (body / s vs the raw cloud), with the metric
        # robust kernel converted at the current scale. In metric units every
        # cloud-noise residual is proportional to s, so a fixed metric kernel
        # rewards shrinking the world regardless of evidence — s slides down a
        # few % on every clip (measured). In cloud units the noise cost is
        # s-invariant and only the real size/depth signals move the scale.
        s = torch.exp(st["log_s"])
        return cloud_chamfer(
            st["vertices"][:, sub] / s, front, cloud, cloud_valid,
            gm_scale_m=weights.gm_chamfer_m / float(s.detach()), chunk=chunk,
            frame_weights=cloud_frame_weights)

    add_term(problem, "chamfer", _chamfer)
    add_term(problem, "reproj", lambda st: sapiens_reprojection_gm(
        st["joints"], st["vertices"], identity, K_orig, target_kp, scores, corr,
        frame_vertex_ids, score_thresh=weights.score_thresh, gm_scale_px=weights.gm_reproj_px))
    add_term(problem, "pose", lambda st: pose_prior_loss(st["qn"], q0n, num_joints))
    # The root gets GENTLE pulls toward the converter placement, not hard locks.
    # Orientation: on weak frames a ~180° flip is otherwise nearly free once the
    # robust data terms saturate. Translation: depth along the ray is the poorly
    # observed direction — on a far, small person the reprojection barely changes
    # over ±5% depth, and with nothing anchoring it the body (and s with it)
    # wanders that valley. A quadratic pull in metres is negligible for the
    # cm-scale corrections the pre-fit exists for and firm against metre wander.
    add_term(problem, "orient", lambda st: root_orient_prior_loss(st["qn"], q0n))
    add_term(problem, "transl", lambda st: (st["qn"][..., :3] - q0n[..., :3])
             .square().sum(-1).mean())
    add_term(problem, "smooth", lambda st: joint_velocity_loss(st["joints"], joints0, pairs))

    wcol = {"chamfer": weights.chamfer, "reproj": weights.reproj,
            "pose": weights.pose, "orient": weights.orient,
            "transl": weights.transl, "smooth": weights.smooth}
    free = {"q": finger_freeze_mask(q_init.shape[-1], device).bool()}
    if freeze_scale:
        free["log_s"] = torch.zeros(1, dtype=torch.bool, device=device)
    fit = run(problem, [Phase(adam(lr, iters), iters, name="prefit", weights=phase_weights(problem, wcol),
                              free=free, on_step=term_logger("prefit", iters), stop_every=0)]).x
    return normalize_q(fit["q"][0], num_joints), float(torch.exp(fit["log_s"]))


def measure_person_scale(
    mid,
    betas: torch.Tensor,          # (T, 10) canonical shape
    q: torch.Tensor,              # (T, 211) the refined pose
    *,
    cloud: torch.Tensor,          # (T, Md, 3) camera-frame depth points, UNSCALED, padded
    cloud_valid: torch.Tensor,    # (T, Md) bool
    sub: torch.Tensor,            # (Ns,) chamfer vertex subsample
    s: float,                     # the scale the body round just fitted against
    gm_scale_m: float,
    chunk: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame ``(log_scale, weight)`` from the SAME correspondences the body fit.

    The scale move of the dual optimization. Each depth point matches its nearest
    *visible* body vertex at the current scale — exactly the pairing the chamfer
    minimized — and votes ``log(z_vertex) − log(z_cloud)`` with its Geman-McClure
    IRLS weight, so a point the chamfer had written off as an outlier gets almost
    no vote here either. Per frame the votes reduce to a weighted median. Because
    the readout uses the fit's own correspondences, a converged body round
    measures back the very ``s`` it was fitted at — the alternation has a true
    fixed point (a projection-based readout does not: it also sees grazing
    silhouette vertices the fit never matched, a small systematic bias that
    compounds across rounds).
    """
    num_frames = q.shape[0]
    logs = np.full(num_frames, np.nan, dtype=np.float64)
    ws = np.zeros(num_frames, dtype=np.float64)
    if cloud.shape[1] == 0:
        return logs, ws
    with torch.no_grad():
        out = mid.forward(betas, normalize_q(q, mid.num_joints))
        front = _front_facing_sub(out.vertices, mid.faces, sub, chunk)
        verts = out.vertices[:, sub]                                  # (T, Ns, 3)
        big = verts.new_tensor(1.0e6)
        for start in range(0, num_frames, chunk):
            end = min(start + chunk, num_frames)
            dist = torch.cdist(verts[start:end], cloud[start:end] * s)          # (b, Ns, Md)
            dist = dist.masked_fill(~front[start:end][:, :, None], big)
            d, nn = dist.min(1)                                                 # (b, Md)
            zv = verts[start:end][..., 2].gather(1, nn)                         # nearest-vert depth
            zc = cloud[start:end][..., 2]                                       # unscaled cloud depth
            w = (1.0 + (d / gm_scale_m).square()).pow(-2)                       # GM IRLS weight
            ok = cloud_valid[start:end] & (zc > 0) & (zv > 0)
            w = torch.where(ok, w, torch.zeros_like(w))
            ratio = torch.where(ok, torch.log(zv.clamp_min(1e-9)) - torch.log(zc.clamp_min(1e-9)),
                                torch.zeros_like(zv))
            w_np, r_np = w.cpu().numpy(), ratio.cpu().numpy()
            for k in range(end - start):
                total = float(w_np[k].sum())
                if total > 1e-6:   # a frame of only saturated matches has no say
                    logs[start + k] = weighted_median(r_np[k], w_np[k])
                    ws[start + k] = total
    return logs, ws
