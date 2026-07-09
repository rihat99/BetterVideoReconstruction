"""The kinematic-stage losses: reprojection, chamfer, pose prior, smoothness.

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
    frame_mask: torch.Tensor | None = None,  # (T,) bool — average over only these (valid) frames
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
    target). ``frame_mask`` restricts the mean to the given (valid) frames, so
    untracked frames — whose held converter pose follows the camera — do not anchor.
    """
    pred_px = project(transform_points(joints_world[:, body_idx, :], extrinsics), K)  # (T, Jb, 2)
    r2 = ((pred_px - target_px) ** 2).sum(dim=-1)                                      # (T, Jb) px²
    gm = geman_mcclure(r2, pred_px.new_tensor(gm_scale_px) ** 2)                       # (T, Jb)
    if frame_mask is None:
        return gm.mean()
    m = frame_mask.to(gm.dtype)                                                        # (T,) 1 = valid
    return (gm * m[:, None]).sum() / (m.sum() * gm.shape[1]).clamp_min(1.0)


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
    vertex_weights: torch.Tensor | None = None,   # (Ns,) per-vertex body-part emphasis
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

    ``vertex_weights`` ``(Ns,)`` reweights the **vertex→depth** term into a *weighted*
    mean over the visible vertices (the reverse depth→vertex term stays uniform — a
    raw depth point carries no body-part label). It emphasizes some body regions
    (e.g. the extremities, whose metric depth VGGT most distorts) over the torso; a
    weight of ``None`` reproduces the plain uniform mean.
    """
    T = vertices_sub.shape[0]
    if depth_padded.shape[1] == 0:            # no depth in any frame (else the cdist→min(2) below crashes)
        return vertices_sub.new_zeros(())
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
        if vertex_weights is None:
            v2d_mean = (v2d * vismask[s:e]).sum(1) / nv[s:e].clamp_min(1)
        else:
            vis_w = vismask[s:e] * vertex_weights                         # (b, Ns)
            v2d_mean = (v2d * vis_w).sum(1) / vis_w.sum(1).clamp_min(1e-6)
        d2v_mean = (d2v * depth_valid[s:e]).sum(1) / nd[s:e].clamp_min(1)
        parts.append(v2d_mean + d2v_mean)                                 # (b,)
    terms = torch.where(fvalid, torch.cat(parts), vertices_sub.new_zeros(()))
    return terms.sum() / fvalid.sum().clamp_min(1)


def wall_contact_loss(
    ee_world: torch.Tensor,       # (T, Nee, 3) end-effector joint centres (world)
    wall_padded: torch.Tensor,    # (T, Mc, 3) per-frame scene candidate points, padded
    wall_valid: torch.Tensor,     # (T, Mc) bool — real (vs padding) candidates
    *,
    margin_m: float,
    temp_m: float,
    gm_scale_m: float,
    chunk: int = 16,
    frame_mask: torch.Tensor | None = None,   # (T,) bool — valid frames
) -> tuple[torch.Tensor, float, int]:
    """Re-anchor prior (T3-B): the body's closest hand/foot should touch the wall.

    Per valid frame, each EE joint's distance to its nearest scene point is soft-combined
    across the ``Nee`` joints (``s_min = −τ·logsumexp(−d/τ)``, a smooth min over joints) and
    penalised past ``margin_m`` with a saturating Geman-McClure kernel of scale ``gm_scale_m``,
    so an airborne / hopeless frame costs a bounded amount instead of dominating. Gradient
    pulls the closest EE toward its nearest scene point. Returns ``(loss, mean s_min cm, n
    valid frames)``.
    """
    T = ee_world.shape[0]
    if wall_padded.shape[1] == 0:
        return ee_world.new_zeros(()), float("nan"), 0
    INF = torch.finfo(ee_world.dtype).max
    smins = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        d = torch.cdist(ee_world[s:e], wall_padded[s:e])                   # (b, Nee, Mc)
        d = d.masked_fill(~wall_valid[s:e][:, None, :], INF)
        dist_j = d.min(2).values                                          # (b, Nee) nearest wall / joint
        smins.append(-temp_m * torch.logsumexp(-dist_j / temp_m, dim=1))  # (b,) soft-min over joints
    s_min = torch.cat(smins)                                              # (T,)
    fvalid = wall_valid.any(1)
    if frame_mask is not None:
        fvalid = fvalid & frame_mask
    s_min = torch.where(fvalid, s_min, torch.zeros_like(s_min))           # kill INF on wall-less frames
    gap = (s_min - margin_m).clamp_min(0.0)
    gm = geman_mcclure(gap ** 2, ee_world.new_tensor(gm_scale_m) ** 2)    # (T,) ∈ [0,1)
    n = int(fvalid.sum())
    mean_cm = float((s_min[fvalid].mean() * 100)) if n > 0 else float("nan")
    return torch.where(fvalid, gm, torch.zeros_like(gm)).sum() / fvalid.sum().clamp_min(1), mean_cm, n


def scene_signed_distance(
    vertices_sub: torch.Tensor,    # (T, Ns, 3) subsampled posed vertices (grad through q)
    scene_padded: torch.Tensor,    # (T, Mc, 3) per-frame candidate scene points, padded
    normal_padded: torch.Tensor,   # (T, Mc, 3) outward unit normals, padded
    conf_padded: torch.Tensor,     # (T, Mc) point confidence, padded
    scene_valid: torch.Tensor,     # (T, Mc) bool — real (vs padding) candidates
    *,
    chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One detached nearest-neighbour pass, shared by the three scene losses.

    For each subsampled body vertex, find its nearest *candidate* scene point ``p``
    (with outward normal ``n``) and return the point-cloud SDF signed distance
    ``s = (v − p)·n`` — negative behind the surface — plus that point's Euclidean
    distance and confidence. The nearest-neighbour search is detached (an argmin is
    not differentiable); gradients flow through ``v`` in ``s`` only. Penetration,
    contact attraction and clearance repulsion differ *only* in the penalty they put
    on ``s`` (:func:`penetration_penalty` / :func:`attraction_penalty` /
    :func:`clearance_penalty`), so running several on the same vertices (penetration +
    contact attraction) shares this one ``cdist`` instead of
    repeating it per loss.

    Chunked over frames — one batched ``cdist`` + gather per chunk — so the distance
    tensor stays bounded by ``chunk`` while the returned signals span the clip.
    Returns ``(sgn, dmin, conf, has)``: ``sgn (T, Ns)`` signed distance (grad through
    ``vertices_sub``), detached ``dmin (T, Ns)`` Euclidean NN distance + ``conf
    (T, Ns)`` NN-point confidence, and ``has (T, 1)`` bool = the frame has any
    candidate. An empty candidate cloud (``Mc == 0``) returns all-far / no-candidate
    signals, so every penalty head reduces to zero.
    """
    T, Ns = vertices_sub.shape[0], vertices_sub.shape[1]
    dev, dt = vertices_sub.device, vertices_sub.dtype
    INF = torch.finfo(dt).max
    if scene_padded.shape[1] == 0:            # no candidates in any frame
        return (vertices_sub.new_zeros((T, Ns)),
                torch.full((T, Ns), INF, device=dev, dtype=dt),
                torch.zeros((T, Ns), device=dev, dtype=conf_padded.dtype),
                torch.zeros((T, 1), device=dev, dtype=torch.bool))
    sgns, dmins, confs = [], [], []
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
        sgns.append(((V - p) * n).sum(dim=-1))                        # (b, Ns) signed, diff through V
        dmins.append(dmin)
        confs.append(c)
    return (torch.cat(sgns, 0), torch.cat(dmins, 0), torch.cat(confs, 0),
            scene_valid.any(dim=1, keepdim=True))


def _huber(excess: torch.Tensor, huber_m: float) -> torch.Tensor:
    """Huber ramp on a non-negative ``excess``: quadratic ``≤ huber_m``, linear beyond."""
    return torch.where(excess <= huber_m, 0.5 * excess * excess,
                       huber_m * (excess - 0.5 * huber_m))


def penetration_penalty(
    sgn: torch.Tensor,     # (T, Ns) signed distance (grad through the posed vertex)
    dmin: torch.Tensor,    # (T, Ns) detached Euclidean NN distance
    conf: torch.Tensor,    # (T, Ns) detached NN-point confidence
    has: torch.Tensor,     # (T, 1) bool — frame has a candidate
    *,
    conf_thresh: float,
    margin_m: float,
    max_dist_m: float,
    max_pen_m: float,
    huber_m: float,
    active_mask: torch.Tensor | None = None,  # (T, Ns) bool — restrict to these verts
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft repulsion pushing the body mesh out of the scene (penalty head on the SDF).

    Penalty = dead-zone + robust kernel on the penetration excess ``e =
    relu(−s − margin_m)``: no cost until penetration exceeds ``margin_m`` (tolerates
    soft tissue + depth/keypoint error), then a Huber ramp (quadratic ``≤ huber_m``,
    linear beyond — one deeply-buried vertex can't dominate). A vertex is gated out
    when its matched scene point is low confidence (``conf_thresh``), farther than
    ``max_dist_m`` (not near a surface / cloud hole), or "penetrating" deeper than
    ``max_pen_m`` (almost always a wrong correspondence behind an occluder).
    ``active_mask`` further restricts which vertices repel (kinematics_2 uses it for a
    stronger repulsion on no-contact joints). The gradient through ``s`` pushes the
    vertex out along the surface normal. Returns ``(loss, mean_depth_cm,
    n_penetrating)`` — ``loss`` the mean robust penalty over all active
    vertex-instances jointly (Σ ÷ total active count), the latter two detached."""
    active = (conf > conf_thresh) & (dmin < max_dist_m) & (sgn.detach() > -max_pen_m) & has
    if active_mask is not None:
        active = active & active_mask
    rho = _huber(torch.relu(-sgn - margin_m), huber_m)               # dead-zone + Huber
    a = active.to(rho.dtype)
    n_active = a.sum()
    if not bool(n_active > 0):
        z = sgn.new_zeros(())
        return z, z, sgn.new_zeros((), dtype=torch.long)
    loss = (rho * a).sum() / n_active.clamp_min(1)                   # joint mean over all active verts
    with torch.no_grad():
        pen = active & (sgn < 0)                                     # actually-penetrating verts
        depth_cm = ((-sgn[pen]).sum() / pen.sum().clamp_min(1)) * 100.0
        n_pen = pen.sum()
    return loss, depth_cm.detach(), n_pen


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
    """Penetration loss = :func:`scene_signed_distance` + :func:`penetration_penalty`.

    Kept as a standalone entry point (the diagnostics pass + external callers); the
    per-iteration optimizer shares one :func:`scene_signed_distance` across the scene
    penalties instead of calling this per loss."""
    sgn, dmin, conf, has = scene_signed_distance(
        vertices_sub, scene_padded, normal_padded, conf_padded, scene_valid, chunk=chunk)
    return penetration_penalty(sgn, dmin, conf, has, conf_thresh=conf_thresh, margin_m=margin_m,
                               max_dist_m=max_dist_m, max_pen_m=max_pen_m, huber_m=huber_m,
                               active_mask=active_mask)


def clearance_penalty(
    sgn: torch.Tensor,     # (T, Ns) signed distance (grad through the posed vertex)
    dmin: torch.Tensor,    # (T, Ns) detached Euclidean NN distance
    conf: torch.Tensor,    # (T, Ns) detached NN-point confidence
    has: torch.Tensor,     # (T, 1) bool — frame has a candidate
    *,
    conf_thresh: float,
    clearance_m: float,
    max_dist_m: float,
    max_pen_m: float,
    huber_m: float,
    active_mask: torch.Tensor | None = None,  # (T, Ns) bool — restrict to these verts
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Push active vertices out to a CLEARANCE outside the scene (penalty head).

    Where :func:`penetration_penalty` only acts behind the surface, this pushes any
    active vertex whose signed distance ``s`` is below ``clearance_m`` out to that
    clearance — so a part a human marked NOT in contact ends up more than the contact
    threshold clear of the wall and stops registering as contact. Penalty = Huber of
    the shortfall ``e = relu(clearance_m − s)``; the gradient w.r.t. ``v`` is the
    outward normal, so descending pushes the vertex out. Same confidence /
    ``max_dist_m`` / ``max_pen_m`` gates as :func:`penetration_penalty`, plus
    ``active_mask`` (the no-contact verts). Returns ``(loss, mean_shortfall_cm,
    n_pushed)`` — the latter two detached diagnostics."""
    active = (conf > conf_thresh) & (dmin < max_dist_m) & (sgn.detach() > -max_pen_m) & has
    if active_mask is not None:
        active = active & active_mask
    excess = torch.relu(clearance_m - sgn)                           # shortfall below the clearance
    rho = _huber(excess, huber_m)
    a = active.to(rho.dtype)
    n_active = a.sum()
    if not bool(n_active > 0):
        z = sgn.new_zeros(())
        return z, z, sgn.new_zeros((), dtype=torch.long)
    loss = (rho * a).sum() / n_active.clamp_min(1)                   # joint mean over all active verts
    with torch.no_grad():
        pushed = active & (excess > 0)
        gap_cm = ((sgn.detach() * pushed).sum() / pushed.sum().clamp_min(1)) * 100.0
        n_pushed = pushed.sum()
    return loss, gap_cm.detach(), n_pushed


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
    """Clearance repulsion = :func:`scene_signed_distance` + :func:`clearance_penalty`
    (standalone entry point; the optimizer shares the signed-distance pass)."""
    sgn, dmin, conf, has = scene_signed_distance(
        vertices_sub, scene_padded, normal_padded, conf_padded, scene_valid, chunk=chunk)
    return clearance_penalty(sgn, dmin, conf, has, conf_thresh=conf_thresh, clearance_m=clearance_m,
                             max_dist_m=max_dist_m, max_pen_m=max_pen_m, huber_m=huber_m,
                             active_mask=active_mask)


def attraction_penalty(
    sgn: torch.Tensor,     # (T, Ns) signed distance (grad through the posed vertex)
    dmin: torch.Tensor,    # (T, Ns) detached Euclidean NN distance
    conf: torch.Tensor,    # (T, Ns) detached NN-point confidence
    has: torch.Tensor,     # (T, 1) bool — frame has a candidate
    *,
    conf_thresh: float,
    band_m: float,
    max_dist_m: float,
    active_mask: torch.Tensor | None = None,    # (T, Ns) bool — restrict to these verts
    active_weight: torch.Tensor | None = None,  # (T, Ns) float in [0,1] — per-vertex weight
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft attraction pulling near-wall vertices ONTO the scene (penalty head).

    The inverse of :func:`penetration_penalty`: a vertex hovering *just outside* the
    surface within the band — ``0 < s < band_m`` — and near a confident surface
    (``conf_thresh`` / ``max_dist_m``) is pulled in. The loss is the mean of ``s`` over
    those vertices, so its gradient w.r.t. ``v`` is the outward normal and descending
    moves the vertex inward, onto the wall — until ``s`` reaches 0, where the band's
    lower bound switches it off and the penetration dead-zone takes over (so the two
    never fight: attraction owns ``s∈(0, band)``, penetration ``s<−margin``).
    ``active_mask`` restricts which vertices are attracted — the kinematics_1
    auto-detected contact joints, or the kinematics_2 manually-annotated ones.
    ``active_weight`` scales each attracted vertex's contribution by its
    joint's label confidence, so an uncertain auto-contact pulls proportionally less
    (``None`` ⇒ every active vertex weighted 1, the original behaviour). Returns
    ``(loss, mean_gap_cm, n_attracted)`` — the latter two detached."""
    sd = sgn.detach()
    active = (conf > conf_thresh) & (dmin < max_dist_m) & (sd > 0.0) & (sd < band_m) & has
    if active_mask is not None:
        active = active & active_mask
    if not bool(active.any()):
        z = sgn.new_zeros(())
        return z, z, sgn.new_zeros((), dtype=torch.long)
    a = active.to(sgn.dtype)
    if active_weight is not None:
        a = a * active_weight
    loss = (sgn * a).sum() / a.sum().clamp_min(1e-6)             # grad pulls each attracted v inward (−n)
    with torch.no_grad():
        gap_cm = ((sd * active.to(sd.dtype)).sum() / active.sum().clamp_min(1)) * 100.0
        n_attr = active.sum()
    return loss, gap_cm.detach(), n_attr


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
    """Scene attraction = :func:`scene_signed_distance` + :func:`attraction_penalty`
    (standalone entry point; the optimizer shares the signed-distance pass)."""
    sgn, dmin, conf, has = scene_signed_distance(
        vertices_sub, scene_padded, normal_padded, conf_padded, scene_valid, chunk=chunk)
    return attraction_penalty(sgn, dmin, conf, has, conf_thresh=conf_thresh, band_m=band_m,
                              max_dist_m=max_dist_m, active_mask=active_mask)


def joint_velocity_loss(
    joints_world: torch.Tensor,    # (T, J, 3) posed (scaled) world joints — grad through q
    contact_labels: torch.Tensor,  # (T, J) int8 tri-state: 1 = annotated in-contact this frame
    *,
    fps: float,
    joint_weight: torch.Tensor | None = None,  # (T, J) float in [0,1] — per-joint confidence weight
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Damp the world velocity of joints a human annotated as in contact.

    A joint touching the wall should be (near-)stationary while the contact lasts,
    so for each frame-to-frame step we penalize the squared world displacement of
    every joint flagged in contact at **both** ends of the step::

        L = mean_{t, j : contact[t,j]==1 ∧ contact[t+1,j]==1} ‖p_{t+1,j} − p_{t,j}‖²

    (the constant ``fps²`` folds into the weight, as in :func:`smoothness_loss`).
    Gating on both frames matches the dynamics stage's ``fpair`` convention: the
    release step (contact→free) is not damped, and any step touching an invalid
    frame drops out for free (held frames carry label ``-1`` ≠ 1).
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
    pair = (contact_labels[:-1] == 1) & (contact_labels[1:] == 1)  # in contact at BOTH ends of the step
    gate = pair.to(sp2.dtype)
    if joint_weight is not None:                         # down-weight uncertain contacts (min over the step)
        gate = gate * torch.minimum(joint_weight[:-1], joint_weight[1:])
    n = gate.sum()
    loss = (sp2 * gate).sum() / n.clamp_min(1.0)
    with torch.no_grad():
        speed_cm_s = ((sp2.sqrt() * gate).sum() / n.clamp_min(1.0)) * float(fps) * 100.0
        n_steps = pair.sum()
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


def radial_velocity_loss(
    joints_world: torch.Tensor,   # (T, J, 3) posed world joints — grad through q
    cam_centers: torch.Tensor,    # (T, 3) world-frame camera centers
    *,
    frame_valid: torch.Tensor | None = None,   # (T,) bool
) -> torch.Tensor:
    """Damp joint velocity **along the per-frame viewing ray** — the depth jitter.

    Depth along the camera ray is the direction monocular evidence constrains worst,
    so that is where trajectory jitter concentrates. For each step ``t → t+1`` the
    world velocity ``v = p_{t+1} − p_t`` is projected onto the unit viewing ray
    ``r = normalize(p_t − C_t)`` (camera center ``C_t`` to the joint) and only that
    radial component is penalized — image-plane motion is left to reprojection. The
    ray is **detached**: the loss damps motion *along* the ray, it must never rotate
    the ray to cut the cost; using the camera *position* (never a camera derivative)
    makes the moving camera a non-issue. The constant ``fps²`` folds into the weight,
    as in :func:`joint_velocity_loss` / :func:`smoothness_loss`. A step counts only if
    ``frame_valid`` at BOTH ends (the fpair convention). ``T<2`` ⇒ 0.
    """
    if joints_world.shape[0] < 2:
        return joints_world.new_zeros(())
    v = joints_world[1:] - joints_world[:-1]                    # (T-1, J, 3) world velocity / step
    r = joints_world[:-1] - cam_centers[:-1, None, :]           # (T-1, J, 3) camera → joint
    r = (r / r.norm(dim=-1, keepdim=True).clamp_min(1e-6)).detach()   # unit ray, detached
    sq = (v * r).sum(dim=-1).square()                          # (T-1, J) squared radial speed (×fps²→weight)
    if frame_valid is None:
        return sq.mean()
    gate = (frame_valid[1:] & frame_valid[:-1]).to(sq.dtype)[:, None]  # (T-1, 1) step valid at both ends
    return (sq * gate).sum() / gate.expand_as(sq).sum().clamp_min(1.0)


def smoothness_loss(
    q: torch.Tensor,   # (T, nq) normalized, better_human order
    num_joints: int,
    *,
    accel_root: float,
    accel_joint: float,
    jerk_root: float,
    jerk_joint: float,
    snap_root: float = 0.0,
    snap_joint: float = 0.0,
) -> torch.Tensor:
    """Acceleration + jerk (+ optional snap) penalty via the exact batched Lie tangent.

    Penalizes raw frame-to-frame tangent finite-differences (the constant
    ``dt`` factors are folded into the weights — equivalent for a fixed-fps
    clip and far better conditioned than ``/dt³`` jerk on a jittery init):
    ``d1 = q_{t+1} ⊖ q_t`` (velocity, :func:`tangent_difference`),
    ``accel = Δd1`` (2nd diff), ``jerk = Δaccel`` (3rd), ``snap = Δjerk`` (4th). Root
    tangent is the first 6 components (se3), joints the remaining ``3·(num_joints-1)``
    — weighted separately since the root carries the global trajectory and the joints
    the articulation. Snap adds a heavier penalty on the highest-frequency jitter
    (defaults 0 = the accel/jerk-only field). A clip too short for a difference (``T<3``
    accel, ``T<4`` jerk, ``T<5`` snap) contributes 0 for that term, not NaN.
    """
    d1 = tangent_difference(q, num_joints)             # (T-1, nv) — velocity
    accel = d1[1:] - d1[:-1]                            # (T-2, nv) — acceleration
    jerk = accel[1:] - accel[:-1]                       # (T-3, nv) — jerk
    snap = jerk[1:] - jerk[:-1]                         # (T-4, nv) — snap

    ar, aj = accel[:, :6], accel[:, 6:]
    jr, jj = jerk[:, :6], jerk[:, 6:]
    sr, sj = snap[:, :6], snap[:, 6:]
    z = q.new_zeros(())
    msq = lambda x: x.square().mean() if x.numel() > 0 else z   # empty diff (short clip) → 0, not NaN
    return (
        accel_root * msq(ar)
        + accel_joint * msq(aj)
        + jerk_root * msq(jr)
        + jerk_joint * msq(jj)
        + snap_root * msq(sr)
        + snap_joint * msq(sj)
    )
