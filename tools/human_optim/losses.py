"""The kinematic-stage losses: reprojection, chamfer, depth push, pose prior, smoothness.

Reprojection is two **robust Geman-McClure** terms against complementary 2D
evidence: :func:`init_reprojection_gm` (the body joints anchored to their OWN
initial converter-pose projection — a dense, self-consistent, detector-free 2D
anchor with identity correspondence) and :func:`sapiens_reprojection_gm` (Sapiens
Goliath-308 keypoints, weighted by detection score). Both use :func:`geman_mcclure`
so a few wrong observations saturate instead of yanking the pose; which terms are
active (and their relative weight) is set in the TOML.

All operate on a single person's world-frame trajectory. ``q`` tensors are
expected pre-normalized (see :func:`tools.human_optim.qmath.normalize_q`).
Posed joints / vertices come from ``better_human``; this module only assembles
the costs.
"""

from __future__ import annotations

import torch

from .camera import project, transform_points
from .keypoints import Correspondence
from .motion import tangent_difference


def geman_mcclure(r2: torch.Tensor, scale2: torch.Tensor) -> torch.Tensor:
    """Geman-McClure robustifier ``ρ(r) = r² / (s² + r²)`` ∈ [0, 1).

    ``r2`` is the squared residual (px²), ``scale2`` the squared robust scale
    (px², broadcastable). Quadratic in ``r`` for ``r ≪ s`` (so well-localized
    points pull normally), saturating toward 1 for ``r ≫ s`` (so a gross outlier
    contributes a bounded cost with **vanishing gradient** — it can't yank the
    pose). ``scale2`` sets the residual at which a point transitions from pulling
    to saturating.
    """
    return r2 / (scale2 + r2)


def sapiens_reprojection_gm(
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
    gm_scale_px: float,
) -> torch.Tensor:
    """Robust (Geman-McClure) reprojection to **Sapiens Goliath-308** keypoints.

    Sapiens gives a per-keypoint detection **score** but no visibility / no
    per-point uncertainty: a *uniform* GM scale ``gm_scale_px`` (no per-point σ)
    and the **score** as the per-point weight (gated below ``score_thresh``).
    Returns the score-weighted mean GM cost ∈ [0, 1) — an independent cross-check
    on the init-projection anchor.
    """
    body = joints_world[:, corr.joint_model, :]                 # (T, Kb, 3)
    lm = vertices_world[:, frame_vertex_ids[corr.lm_frame], :]  # (T, Kl, 3)
    pred_world = torch.cat([body, lm], dim=1)                   # (T, K, 3)

    pred_px = project(transform_points(pred_world, extrinsics), K)  # (T, K, 2)
    tgt = target_kp[:, corr.sap, :]                            # (T, K, 2)
    w = scores[:, corr.sap]                                    # (T, K)
    w = torch.where(w >= score_thresh, w, torch.zeros_like(w))

    r2 = ((pred_px - tgt) ** 2).sum(dim=-1)                    # (T, K) px²
    gm = geman_mcclure(r2, pred_px.new_tensor(gm_scale_px) ** 2)
    return (w * gm).sum() / w.sum().clamp_min(1.0)


def init_reprojection_gm(
    joints_world: torch.Tensor,    # (T, J, 3) current posed (scaled) joints
    body_idx: torch.Tensor,        # (Jb,) long — body-joint indices to anchor
    extrinsics: torch.Tensor,      # (T, 4, 4) cam_from_world
    K: torch.Tensor,               # (T, 3, 3) pixel intrinsics
    target_px: torch.Tensor,       # (T, Jb, 2) frozen — each body joint's INITIAL projection
    *,
    gm_scale_px: float,
) -> torch.Tensor:
    """Robust (Geman-McClure) reprojection of the body joints to their own INITIAL
    (converter-pose) projection — a dense, self-consistent 2D anchor.

    The converter's per-frame SMPL-X pose already reprojects well (~5px) and the
    correspondence here is the **identity** (a model joint to its OWN projected
    pixel), so this sidesteps the Sapiens Goliath-308 mapping + per-frame detector
    noise entirely and anchors *every* body joint (denser than Sapiens' sparse
    set). It is a **regularizer toward the converter's trusted 2D**: it holds the
    body image-locked, so chamfer seats DEPTH without sliding the body off-image and
    the body keeps its converter size relative to the scene. The cost is that it is
    **not an independent observation** — it cannot *correct* a converter 2D error,
    only preserve the (good) converter pose against the 3D terms' pull. Returns the
    mean GM cost ∈ [0, 1) (0 at init — the body starts at the pose that defined the
    target).
    """
    pred_px = project(transform_points(joints_world[:, body_idx, :], extrinsics), K)  # (T, Jb, 2)
    r2 = ((pred_px - target_px) ** 2).sum(dim=-1)                                      # (T, Jb) px²
    gm = geman_mcclure(r2, pred_px.new_tensor(gm_scale_px) ** 2)                       # (T, Jb)
    return gm.mean()


def pad_frames(
    tensors: list[torch.Tensor], *, channels: int | None, device, dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a per-frame list of ragged ``(n_t, channels)`` tensors into a dense
    ``(T, M, channels)`` buffer + a ``(T, M)`` bool validity mask (``M = max n_t``).

    The chamfer / penetration clouds vary in size per frame, which is what forced
    the old per-frame Python loop. Padding them to a common width once (outside the
    optimization loop — the depth cloud is static and the scene candidates are
    built once per stage) lets the per-iteration cost be a single chunked-batched
    ``cdist`` instead of ``T`` tiny ones. ``channels=None`` pads per-frame scalars
    ``(n_t,) → (T, M)``.
    """
    T = len(tensors)
    M = max((int(x.shape[0]) for x in tensors), default=0)
    shape = (T, M, channels) if channels is not None else (T, M)
    buf = torch.zeros(shape, device=device, dtype=dtype)
    valid = torch.zeros((T, M), dtype=torch.bool, device=device)
    for i, x in enumerate(tensors):
        n = int(x.shape[0])
        if n:
            buf[i, :n] = x
            valid[i, :n] = True
    return buf, valid


def chamfer_loss(
    vertices_sub: torch.Tensor,   # (T, Ns, 3) subsampled posed vertices
    vismask: torch.Tensor,        # (T, Ns) bool — front-facing
    depth_padded: torch.Tensor,   # (T, Md, 3) per-frame depth points, padded
    depth_valid: torch.Tensor,    # (T, Md) bool — real (vs padding) depth points
    *,
    chunk: int = 16,
) -> torch.Tensor:
    """Bidirectional chamfer between visible vertices and depth points.

    Averaged over frames that have both visible vertices and depth points.
    Gradients flow through ``vertices_sub``; ``vismask`` and the (padded) depth
    cloud are constants for a given call.

    Computed in frame-chunks: each chunk is a single batched
    ``cdist`` over ``(chunk, Ns, Md)`` rather than ``chunk`` separate per-frame
    ``cdist`` calls — collapsing the launch-bound per-frame loop (the real cost on
    a long clip) into a handful of batched kernels, while keeping the distance
    tensor small (bounded by ``chunk``). Padding columns / occluded vertices are
    masked to ``+inf`` so they never win a min; frames with no visible vertex or no
    depth point drop out of the average. Numerically identical to the per-frame
    form (validated to fp tolerance).
    """
    T = vertices_sub.shape[0]
    INF = torch.finfo(vertices_sub.dtype).max
    nv = vismask.sum(1)                       # (T,) visible verts / frame
    nd = depth_valid.sum(1)                   # (T,) depth points / frame
    fvalid = (nv > 0) & (nd > 0)
    parts = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        d = torch.cdist(vertices_sub[s:e], depth_padded[s:e])              # (b, Ns, Md)
        v2d = d.masked_fill(~depth_valid[s:e][:, None, :], INF).min(2).values  # (b, Ns)
        d2v = d.masked_fill(~vismask[s:e][:, :, None], INF).min(1).values      # (b, Md)
        v2d_mean = (v2d * vismask[s:e]).sum(1) / nv[s:e].clamp_min(1)
        d2v_mean = (d2v * depth_valid[s:e]).sum(1) / nd[s:e].clamp_min(1)
        parts.append(v2d_mean + d2v_mean)                                 # (b,)
    terms = torch.where(fvalid, torch.cat(parts), vertices_sub.new_zeros(()))
    return terms.sum() / fvalid.sum().clamp_min(1)


def scene_penetration(
    vertices_sub: torch.Tensor,    # (T, Ns, 3) subsampled posed vertices (all — not vis-gated)
    scene_padded: torch.Tensor,    # (T, Mc, 3) per-frame candidate scene points, padded
    normal_padded: torch.Tensor,   # (T, Mc, 3) outward unit normals, padded
    conf_padded: torch.Tensor,     # (T, Mc) point confidence, padded
    scene_valid: torch.Tensor,     # (T, Mc) bool — real (vs padding) candidates
    *,
    conf_thresh: float,
    margin_m: float,
    max_dist_m: float,
    max_pen_m: float,
    huber_m: float,
    active_mask: torch.Tensor | None = None,  # (T, Ns) bool — restrict to these verts
    chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft repulsion pushing the body mesh out of the static scene cloud.

    A point-cloud SDF: for each subsampled body vertex, the nearest *candidate*
    scene point gives a signed distance ``s = (v − p)·n`` (``n`` the scene
    normal). ``s < 0`` ⇒ the vertex is behind the surface (penetrating). The
    nearest-neighbour search is detached (an argmin is not differentiable);
    gradients flow through ``v`` in the signed distance, so the repulsion pushes
    the vertex out **along the surface normal** (the SDF gradient).

    Penalty = dead-zone + robust kernel: no cost until penetration exceeds
    ``margin_m`` (tolerates soft tissue + depth/keypoint error), then a Huber
    ramp on the excess ``e = relu(−s − margin_m)`` — quadratic for ``e ≤
    huber_m`` (small penetration → small force) and linear beyond (one
    deeply-buried vertex can't dominate). A vertex is gated out when its matched
    scene point is low confidence (``conf_thresh``), when it is farther than
    ``max_dist_m`` from any scene point (not near a surface / cloud hole), or
    when the "penetration" is deeper than ``max_pen_m`` (almost always a wrong
    correspondence behind an occluder, not a real collision). ``active_mask``
    (optional ``(T, Ns)`` bool) further restricts which vertices repel — used by
    the kinematics_2 stage to apply a *stronger* repulsion only to the vertices of
    joints a human annotated as NOT in contact.

    The candidate cloud is padded (:func:`pad_frames`) and processed in
    frame-chunks — one batched ``cdist`` + gather per chunk rather than ``T``
    per-frame searches. Returns ``(loss, mean_depth_cm, n_penetrating)``; the
    latter two are detached diagnostics. ``loss`` is the mean over active
    vertices, averaged over frames that have any — matching
    :func:`chamfer_loss`'s per-frame reduction. Numerically identical to the
    per-frame form (validated to fp tolerance).
    """
    T = vertices_sub.shape[0]
    if scene_padded.shape[1] == 0:        # no scene candidates near the body in any frame
        z = vertices_sub.new_zeros(())
        return z, z, vertices_sub.new_zeros((), dtype=torch.long)
    INF = torch.finfo(vertices_sub.dtype).max
    tot_depth = vertices_sub.new_zeros(())
    tot_count = vertices_sub.new_zeros((), dtype=torch.long)
    parts, fparts = [], []
    for s0 in range(0, T, chunk):
        e0 = min(s0 + chunk, T)
        V = vertices_sub[s0:e0]                                       # (b, Ns, 3)
        with torch.no_grad():
            d = torch.cdist(V, scene_padded[s0:e0])
            d = d.masked_fill(~scene_valid[s0:e0][:, None, :], INF)
            dmin, idx = d.min(dim=2)                                  # (b, Ns) — detached NN
        gi = idx[..., None].expand(-1, -1, 3)
        p = torch.gather(scene_padded[s0:e0], 1, gi)                  # (b, Ns, 3)
        n = torch.gather(normal_padded[s0:e0], 1, gi)
        c = torch.gather(conf_padded[s0:e0], 1, idx)                  # (b, Ns)
        sgn = ((V - p) * n).sum(dim=-1)                               # (b, Ns) signed, diff through V
        has = scene_valid[s0:e0].any(dim=1, keepdim=True)             # frame has any candidate
        active = (c > conf_thresh) & (dmin < max_dist_m) & (sgn.detach() > -max_pen_m) & has
        if active_mask is not None:
            active = active & active_mask[s0:e0]
        excess = torch.relu(-sgn - margin_m)                         # penetration beyond the dead-zone
        rho = torch.where(excess <= huber_m, 0.5 * excess * excess,
                          huber_m * (excess - 0.5 * huber_m))
        a = active.to(rho.dtype)
        na = a.sum(1)                                                 # (b,) active verts / frame
        parts.append((rho * a).sum(1) / na.clamp_min(1))
        fparts.append(na > 0)
        with torch.no_grad():
            pen = active & (sgn < 0)                                  # actually-penetrating verts
            tot_depth = tot_depth + (-sgn[pen]).sum()
            tot_count = tot_count + pen.sum()
    fvalid = torch.cat(fparts)
    if not bool(fvalid.any()):
        z = vertices_sub.new_zeros(())
        return z, z, tot_count
    terms = torch.where(fvalid, torch.cat(parts), vertices_sub.new_zeros(()))
    loss = terms.sum() / fvalid.sum().clamp_min(1)
    depth_cm = (tot_depth / tot_count.clamp_min(1)) * 100.0
    return loss, depth_cm.detach(), tot_count


def clearance_repulsion(
    vertices_sub: torch.Tensor,    # (T, Ns, 3) subsampled posed vertices (grad through q)
    scene_padded: torch.Tensor,    # (T, Mc, 3) per-frame candidate scene points, padded
    normal_padded: torch.Tensor,   # (T, Mc, 3) outward unit normals, padded
    conf_padded: torch.Tensor,     # (T, Mc) point confidence, padded
    scene_valid: torch.Tensor,     # (T, Mc) bool — real (vs padding) candidates
    *,
    conf_thresh: float,
    clearance_m: float,
    max_dist_m: float,
    max_pen_m: float,
    huber_m: float,
    active_mask: torch.Tensor | None = None,  # (T, Ns) bool — restrict to these verts
    chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Push vertices to a CLEARANCE outside the scene (for annotated no-contact parts).

    Where :func:`scene_penetration` only acts once a vertex is *behind* the surface
    (signed distance ``s = (v−p)·n < 0``), this pushes any active vertex whose ``s``
    is below ``clearance_m`` out to that clearance — so a body part a human marked
    NOT in contact ends up more than the contact threshold clear of the wall and
    stops registering as contact (the gap :func:`scene_penetration` left: a part
    hovering 0–``close_thresh`` cm outside the wall is "no contact" to this loss but
    "in contact" to the contact readout). Penalty = Huber of the shortfall
    ``e = relu(clearance_m − s)`` (quadratic ``≤ huber_m`` then linear); gradient
    w.r.t. ``v`` is the outward normal, so descending pushes the vertex out. Same
    confidence / ``max_dist_m`` / ``max_pen_m`` gates as :func:`scene_penetration`,
    plus ``active_mask`` (the no-contact verts). Chunked-batched ``cdist`` (its own
    NN pass). Returns ``(loss, mean_shortfall_cm, n_pushed)`` — the latter two
    detached diagnostics."""
    T = vertices_sub.shape[0]
    if scene_padded.shape[1] == 0:
        z = vertices_sub.new_zeros(())
        return z, z, vertices_sub.new_zeros((), dtype=torch.long)
    INF = torch.finfo(vertices_sub.dtype).max
    tot_gap = vertices_sub.new_zeros(())
    tot_count = vertices_sub.new_zeros((), dtype=torch.long)
    parts, fparts = [], []
    for s0 in range(0, T, chunk):
        e0 = min(s0 + chunk, T)
        V = vertices_sub[s0:e0]                                       # (b, Ns, 3)
        with torch.no_grad():
            d = torch.cdist(V, scene_padded[s0:e0])
            d = d.masked_fill(~scene_valid[s0:e0][:, None, :], INF)
            dmin, idx = d.min(dim=2)                                  # (b, Ns) — detached NN
        gi = idx[..., None].expand(-1, -1, 3)
        p = torch.gather(scene_padded[s0:e0], 1, gi)                  # (b, Ns, 3)
        n = torch.gather(normal_padded[s0:e0], 1, gi)
        c = torch.gather(conf_padded[s0:e0], 1, idx)                  # (b, Ns)
        sgn = ((V - p) * n).sum(dim=-1)                               # (b, Ns) signed, diff through V
        has = scene_valid[s0:e0].any(dim=1, keepdim=True)
        active = (c > conf_thresh) & (dmin < max_dist_m) & (sgn.detach() > -max_pen_m) & has
        if active_mask is not None:
            active = active & active_mask[s0:e0]
        excess = torch.relu(clearance_m - sgn)                       # shortfall below the clearance
        rho = torch.where(excess <= huber_m, 0.5 * excess * excess,
                          huber_m * (excess - 0.5 * huber_m))
        a = active.to(rho.dtype)
        na = a.sum(1)
        parts.append((rho * a).sum(1) / na.clamp_min(1))
        fparts.append(na > 0)
        with torch.no_grad():
            pushed = active & (excess > 0)
            tot_gap = tot_gap + (sgn.detach() * pushed).sum()
            tot_count = tot_count + pushed.sum()
    fvalid = torch.cat(fparts)
    if not bool(fvalid.any()):
        z = vertices_sub.new_zeros(())
        return z, z, tot_count
    terms = torch.where(fvalid, torch.cat(parts), vertices_sub.new_zeros(()))
    loss = terms.sum() / fvalid.sum().clamp_min(1)
    gap_cm = (tot_gap / tot_count.clamp_min(1)) * 100.0
    return loss, gap_cm.detach(), tot_count


def scene_attraction(
    vertices_sub: torch.Tensor,    # (T, Ns, 3) subsampled posed vertices (grad through q)
    scene_padded: torch.Tensor,    # (T, Mc, 3) per-frame candidate scene points, padded
    normal_padded: torch.Tensor,   # (T, Mc, 3) outward unit normals, padded
    conf_padded: torch.Tensor,     # (T, Mc) point confidence, padded
    scene_valid: torch.Tensor,     # (T, Mc) bool — real (vs padding) candidates
    *,
    conf_thresh: float,
    band_m: float,
    max_dist_m: float,
    active_mask: torch.Tensor | None = None,  # (T, Ns) bool — restrict to these verts
    chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft attraction pulling near-wall body vertices ONTO the static scene.

    The inverse of :func:`scene_penetration`: where penetration pushes vertices
    that are *behind* the surface back out, this draws vertices that hover *just
    outside* the surface onto it — so a body part a few cm off the wall (e.g. a
    climbing hand whose monocular human-depth lands a little in front of the merged
    scene wall, the common "close but not touching" case) is pulled into contact.

    Same point-cloud SDF as the penetration loss: the nearest *candidate* scene
    point gives a signed distance ``s = (v − p)·n`` (``n`` outward), the
    nearest-neighbour search detached, gradients flowing through ``v``. A vertex is
    **attracted** when it sits just outside the surface within the band —
    ``0 < s < band_m`` — and is confident / near a surface (``conf_thresh`` /
    ``max_dist_m``). The loss is the mean of ``s`` over those vertices, so its
    gradient w.r.t. ``v`` is the outward normal and **descending it moves the
    vertex inward, onto the wall** — until ``s`` reaches 0, where the band's lower
    bound switches the term off and the penetration dead-zone takes over (so the
    two never fight: attraction owns ``s∈(0, band)``, penetration ``s<−margin``).
    Vertices already inside (``s ≤ 0``) or farther than ``band_m`` are left to the
    penetration / chamfer terms; band-gating alone selects the genuinely near-wall
    parts (hands/feet) since the bulk torso usually sits beyond the band.
    ``active_mask`` (optional ``(T, Ns)`` bool) further restricts which vertices are
    attracted — used by the kinematics_2 stage to do a *real* attraction (with a
    wide ``band_m``) only on the vertices of joints a human annotated as in contact.

    Chunked-batched ``cdist`` (its own NN pass, mirroring the penetration loop).
    Returns ``(loss, mean_gap_cm, n_attracted)`` — the latter two detached
    diagnostics (the mean outside-gap of the attracted verts, shrinking as they are
    drawn in, and how many are in-band).
    """
    T = vertices_sub.shape[0]
    if scene_padded.shape[1] == 0:        # no scene candidates near the body in any frame
        z = vertices_sub.new_zeros(())
        return z, z, vertices_sub.new_zeros((), dtype=torch.long)
    INF = torch.finfo(vertices_sub.dtype).max
    tot_gap = vertices_sub.new_zeros(())
    tot_count = vertices_sub.new_zeros((), dtype=torch.long)
    parts, fparts = [], []
    for s0 in range(0, T, chunk):
        e0 = min(s0 + chunk, T)
        V = vertices_sub[s0:e0]                                   # (b, Ns, 3)
        with torch.no_grad():
            d = torch.cdist(V, scene_padded[s0:e0])
            d = d.masked_fill(~scene_valid[s0:e0][:, None, :], INF)
            dmin, idx = d.min(dim=2)                              # (b, Ns) — detached NN
        gi = idx[..., None].expand(-1, -1, 3)
        p = torch.gather(scene_padded[s0:e0], 1, gi)              # (b, Ns, 3)
        n = torch.gather(normal_padded[s0:e0], 1, gi)
        c = torch.gather(conf_padded[s0:e0], 1, idx)              # (b, Ns)
        sgn = ((V - p) * n).sum(dim=-1)                           # (b, Ns) signed, diff through V
        has = scene_valid[s0:e0].any(dim=1, keepdim=True)
        sd = sgn.detach()
        active = (c > conf_thresh) & (dmin < max_dist_m) & (sd > 0.0) & (sd < band_m) & has
        if active_mask is not None:
            active = active & active_mask[s0:e0]
        a = active.to(sgn.dtype)
        na = a.sum(1)                                             # (b,) attracted verts / frame
        parts.append((sgn * a).sum(1) / na.clamp_min(1))         # mean gap; grad pulls v inward (−n)
        fparts.append(na > 0)
        with torch.no_grad():
            tot_gap = tot_gap + (sd * a).sum()
            tot_count = tot_count + active.sum()
    fvalid = torch.cat(fparts)
    if not bool(fvalid.any()):
        z = vertices_sub.new_zeros(())
        return z, z, tot_count
    terms = torch.where(fvalid, torch.cat(parts), vertices_sub.new_zeros(()))
    loss = terms.sum() / fvalid.sum().clamp_min(1)
    gap_cm = (tot_gap / tot_count.clamp_min(1)) * 100.0
    return loss, gap_cm.detach(), tot_count


def contact_velocity_loss(
    vertices_sub: torch.Tensor,    # (T, Ns, 3) subsampled posed vertices (grad through q)
    vel_vert: torch.Tensor,        # (T, Ns) per-vertex joint speed (m/s), DETACHED gate input
    scene_padded: torch.Tensor,    # (T, Mc, 3) per-frame candidate scene points, padded
    normal_padded: torch.Tensor,   # (T, Mc, 3) outward unit normals, padded
    conf_padded: torch.Tensor,     # (T, Mc) point confidence, padded
    scene_valid: torch.Tensor,     # (T, Mc) bool — real (vs padding) candidates
    *,
    conf_thresh: float,
    max_dist_m: float,
    max_pen_m: float,
    contact_band_m: float,
    vel_ref: float,
    vel_exp: float,
    chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Velocity-gated contact repulsion: push *fast-moving* contacts off the scene.

    A real contact is (near-)stationary; a body vertex that is in contact with the
    scene **while its joint moves fast** is unphysical (a limb clipping through the
    wall mid-swing). This loss finds contacts *inside* the optimization loop —
    differentiably, via the same point-cloud SDF as :func:`scene_penetration` — and
    repels them with a force that **scales with the joint's speed**:

      * **contact term** ``relu(contact_band_m − s)`` where ``s = (v−p)·n`` is the
        signed distance to the nearest scene point (negative = inside). It is > 0
        when a vertex is inside the scene or just outside it (within
        ``contact_band_m``); its gradient w.r.t. ``v`` is the outward normal, so
        minimizing it pushes the vertex **out along the surface** (the repulsion).
        The NN is detached; gradients flow through ``v`` only.
      * **velocity gate** ``g = (vel/vel_ref)^vel_exp`` — a smooth, monotone,
        threshold-free function of the (detached) per-vertex joint speed
        ``vel_vert``: ≈ 0 for a slow planted contact, large for a fast one. The
        joint speed is the LOCAL-frame linear velocity magnitude from the
        BetterRobot Jacobian (:func:`tools.human_optim.velocity.joint_local_linear_speed`),
        mapped to each vertex by its nearest joint.

    ``L = mean_active( g · relu(contact_band_m − s) )``. The same gates as the
    penetration loss keep strays out: ``conf_thresh`` (confident scene points),
    ``dmin < max_dist_m`` (near a surface), ``s > −max_pen_m`` (no behind-occluder
    correspondences). Reduced per active vertex then averaged over frames that have
    any — matching :func:`scene_penetration`. Chunked-batched ``cdist`` (its own
    nearest-neighbour pass, mirroring the penetration loop). Returns
    ``(loss, mean_inside_depth_cm, n_contact)`` — the latter two detached diagnostics.
    """
    T = vertices_sub.shape[0]
    if scene_padded.shape[1] == 0:        # no scene candidates near the body in any frame
        z = vertices_sub.new_zeros(())
        return z, z, vertices_sub.new_zeros((), dtype=torch.long)
    INF = torch.finfo(vertices_sub.dtype).max
    gate = (vel_vert / vel_ref).clamp_min(0.0) ** vel_exp        # (T, Ns) detached, smart increasing fn
    tot_depth = vertices_sub.new_zeros(())
    tot_count = vertices_sub.new_zeros((), dtype=torch.long)
    parts, fparts = [], []
    for s0 in range(0, T, chunk):
        e0 = min(s0 + chunk, T)
        V = vertices_sub[s0:e0]                                   # (b, Ns, 3)
        with torch.no_grad():
            d = torch.cdist(V, scene_padded[s0:e0])
            d = d.masked_fill(~scene_valid[s0:e0][:, None, :], INF)
            dmin, idx = d.min(dim=2)                              # (b, Ns) — detached NN
        gi = idx[..., None].expand(-1, -1, 3)
        p = torch.gather(scene_padded[s0:e0], 1, gi)              # (b, Ns, 3)
        n = torch.gather(normal_padded[s0:e0], 1, gi)
        c = torch.gather(conf_padded[s0:e0], 1, idx)              # (b, Ns)
        sgn = ((V - p) * n).sum(dim=-1)                           # (b, Ns) signed, diff through V
        has = scene_valid[s0:e0].any(dim=1, keepdim=True)
        active = (c > conf_thresh) & (dmin < max_dist_m) & (sgn.detach() > -max_pen_m) & has
        contact = torch.relu(contact_band_m - sgn)               # >0 inside / within band; grad = outward n
        term = gate[s0:e0] * contact                             # velocity-gated repulsion
        a = (active & (contact > 0)).to(term.dtype)              # only gated contacting verts count
        na = a.sum(1)                                            # (b,)
        parts.append((term * a).sum(1) / na.clamp_min(1))
        fparts.append(na > 0)
        with torch.no_grad():
            inside = active & (sgn < 0)
            tot_depth = tot_depth + (-sgn[inside]).sum()
            tot_count = tot_count + (active & (contact > 0)).sum()
    fvalid = torch.cat(fparts)
    if not bool(fvalid.any()):
        z = vertices_sub.new_zeros(())
        return z, z, tot_count
    terms = torch.where(fvalid, torch.cat(parts), vertices_sub.new_zeros(()))
    loss = terms.sum() / fvalid.sum().clamp_min(1)
    depth_cm = (tot_depth / tot_count.clamp_min(1)) * 100.0
    return loss, depth_cm.detach(), tot_count


def joint_velocity_loss(
    joints_world: torch.Tensor,    # (T, J, 3) posed (scaled) world joints — grad through q
    contact_labels: torch.Tensor,  # (T, J) int8 tri-state: 1 = annotated in-contact this frame
    *,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Damp the world velocity of joints a human annotated as in contact.

    A joint touching the wall should be (near-)stationary while the contact lasts,
    so for each frame-to-frame step we penalize the squared world displacement of
    every joint flagged in contact at the outgoing frame::

        L = mean_{t, j : contact[t,j]==1} ‖p_{t+1,j} − p_{t,j}‖²

    (the constant ``fps²`` folds into the weight, as in :func:`smoothness_loss`).
    That displacement magnitude equals the joint's LOCAL-frame linear-velocity
    magnitude — the BetterRobot-Jacobian quantity in
    :mod:`tools.human_optim.velocity` — because the body-frame rotation preserves
    norm; but the finite difference of the posed world joints is directly
    differentiable through ``q`` (that Jacobian path is detached), so we use it as
    the loss. Gradients flow through ``joints_world`` only. Returns
    ``(loss, mean_speed_cm_s, n_contact_steps)`` — the latter two detached
    diagnostics (mean in-contact joint speed cm/s, and how many (frame, joint)
    steps were penalized)."""
    if joints_world.shape[0] < 2:
        z = joints_world.new_zeros(())
        return z, z, joints_world.new_zeros((), dtype=torch.long)
    dp = joints_world[1:] - joints_world[:-1]            # (T-1, J, 3) world displacement / step
    sp2 = (dp * dp).sum(dim=-1)                          # (T-1, J) squared per-step speed (×fps²→weight)
    gate = (contact_labels[:-1] == 1).to(sp2.dtype)      # in-contact at the outgoing frame
    n = gate.sum()
    loss = (sp2 * gate).sum() / n.clamp_min(1.0)
    with torch.no_grad():
        speed_cm_s = ((sp2.sqrt() * gate).sum() / n.clamp_min(1.0)) * float(fps) * 100.0
        n_steps = (contact_labels[:-1] == 1).sum()
    return loss, speed_cm_s.detach(), n_steps


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
    return (1.0 - (qj * qij).sum(dim=-1).square()).mean()


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


def depth_push_loss(
    joints_world: torch.Tensor,    # (T, J, 3) posed (scaled) world joints
    extrinsics: torch.Tensor,      # (T, 4, 4) cam_from_world
    frame_mask: torch.Tensor,      # (T,) bool — frames where the push is active (have depth)
) -> torch.Tensor:
    """Gentle push of the body AWAY from the camera (along +z_cam).

    Returns the mean over active frames of the NEGATIVE camera-frame depth of the
    pelvis (joint 0), so minimizing it slides the body deeper along the viewing
    direction. The reprojection terms hold the body on the camera rays, so — with
    the global scale free — the body GROWS to keep filling the silhouette as it
    recedes: it slides out along the rays toward the metric depth cloud and
    balances the scene-penetration push-out (which shoves it the other way, toward
    the camera). The pelvis is **scale-invariant** (scale is applied about it), so
    this term moves only the root translation; scale responds indirectly through
    reprojection. Gated to frames that have a depth cloud (``frame_mask``) so the
    recession is always met by a chamfer counter-pull — on a depth-less frame
    nothing pushes back, so the term stays off there to avoid runaway.
    """
    z_cam = transform_points(joints_world[:, :1, :], extrinsics)[:, 0, 2]  # (T,) pelvis depth
    m = frame_mask.to(z_cam.dtype)
    return -(z_cam * m).sum() / m.sum().clamp_min(1.0)


def smoothness_loss(
    q: torch.Tensor,   # (T, nq) normalized, better_human order
    num_joints: int,
    *,
    accel_root: float,
    accel_joint: float,
    jerk_root: float,
    jerk_joint: float,
) -> torch.Tensor:
    """Acceleration + jerk penalty via the exact batched Lie tangent.

    Penalizes raw frame-to-frame tangent finite-differences (the constant
    ``dt`` factors are folded into the weights — equivalent for a fixed-fps
    clip and far better conditioned than ``/dt³`` jerk on a jittery init):
    ``d1 = q_{t+1} ⊖ q_t`` (velocity, :func:`tangent_difference`),
    ``accel = Δd1`` (2nd diff), ``jerk = Δaccel``. Root tangent is the first 6
    components (se3), joints the remaining ``3·(num_joints-1)`` — weighted
    separately since the root carries the global trajectory and the joints the
    articulation.
    """
    d1 = tangent_difference(q, num_joints)             # (T-1, nv) — velocity
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
