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
    visibility: torch.Tensor | None = None,  # (T, 308) per-keypoint weight in [0,1]
) -> torch.Tensor:
    """Robust (Geman-McClure) reprojection to **Sapiens Goliath-308** keypoints.

    Sapiens gives a per-keypoint detection **score** but no visibility / no
    per-point uncertainty: a *uniform* GM scale ``gm_scale_px`` (no per-point σ)
    and the **score** as the per-point weight (gated below ``score_thresh``).
    Returns the score-weighted mean GM cost ∈ [0, 1) — an independent cross-check
    on the init-projection anchor.

    ``visibility`` (:func:`~tools.human_optim.keypoints.visibility_weights`) scales each
    keypoint's contribution again, so a keypoint the body occludes pulls less. The
    denominator stays the *unscaled* score sum, so down-weighting a subset genuinely
    lowers the term instead of redistributing it onto the keypoints that stayed.
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
    num = w * gm if visibility is None else w * visibility[:, corr.sap] * gm
    return num.sum() / w.sum().clamp_min(1.0)


def init_reprojection_gm(
    joints_world: torch.Tensor,    # (T, J, 3) current posed (scaled) joints
    body_idx: torch.Tensor,        # (Jb,) long — body-joint indices to anchor
    extrinsics: torch.Tensor,      # (T, 4, 4) cam_from_world
    K: torch.Tensor,               # (T, 3, 3) pixel intrinsics
    target_px: torch.Tensor,       # (T, Jb, 2) frozen — each body joint's INITIAL projection
    *,
    gm_scale_px: float,
    frame_mask: torch.Tensor | None = None,  # (T,) bool — average over only these (valid) frames
    joint_weight: torch.Tensor | None = None,  # (T, Jb) per-joint weight in [0,1]
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
    ``joint_weight`` (:func:`~tools.human_optim.keypoints.visibility_weights`) scales each
    joint's contribution, so an occluded joint — whose starting pose came from a converter
    estimate the camera could not see either — is held less tightly. The denominator is
    untouched, so the anchor as a whole weakens rather than shifting its pull onto the
    joints that stayed visible.
    """
    pred_px = project(transform_points(joints_world[:, body_idx, :], extrinsics), K)  # (T, Jb, 2)
    r2 = ((pred_px - target_px) ** 2).sum(dim=-1)                                      # (T, Jb) px²
    gm = geman_mcclure(r2, pred_px.new_tensor(gm_scale_px) ** 2)                       # (T, Jb)
    if joint_weight is not None:
        gm = gm * joint_weight
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
    frame_weights: torch.Tensor | None = None,    # (T,) per-frame depth-quality weight
    sat_m: float = 0.0,                           # saturate per-point distances (m); 0 = off
) -> torch.Tensor:
    """Bidirectional chamfer between visible vertices and depth points.

    Averaged over frames that have both visible vertices and depth points.
    Gradients flow through ``vertices_sub``; ``vismask`` and the (padded) depth
    cloud are constants for a given call.

    Computed in frame-chunks: each chunk is a single batched
    ``cdist`` over ``(chunk, Ns, Md)`` rather than ``chunk`` separate per-frame
    ``cdist`` calls — collapsing the launch-bound per-frame loop (the real cost on
    a long clip) into a handful of batched kernels, while keeping the distance
    tensor small (bounded by ``chunk``). The distance matrix is built under
    ``no_grad`` and used only to pick nearest neighbours; the differentiable
    distance is recomputed on the gathered NN *pairs* alone, so backward never
    touches the ``(b, Ns, Md)`` matrix — same loss and gradient (the min is
    achieved at the gathered pair), ~2.5× faster and ~10× less peak memory than
    carrying the matrix through autograd. Padding columns / occluded vertices are
    masked to ``+inf`` so they never win the argmin; frames with no visible vertex
    or no depth point drop out of the average.

    ``vertex_weights`` ``(Ns,)`` reweights the **vertex→depth** term into a *weighted*
    mean over the visible vertices (the reverse depth→vertex term stays uniform — a
    raw depth point carries no body-part label). It emphasizes some body regions
    (e.g. the extremities, whose metric depth VGGT most distorts) over the torso; a
    weight of ``None`` reproduces the plain uniform mean.

    ``frame_weights`` ``(T,)`` turns the frame average into a weighted one — the
    depth-quality hook (:mod:`tools.geometry.depth_smear`): a frame whose human
    depth is smeared along the view ray should pull the body far less than a clean
    one. ``None``, or all-equal weights, reproduce the plain mean exactly.

    ``sat_m`` caps each per-point distance at ``sat_m * tanh(d / sat_m)`` before the
    means. Near the body (``d ≪ sat_m``) this is the identity, so the shipped
    calibration is untouched; far outliers — a depth tail smeared along the view
    ray, a mismatched patch behind an occluder — saturate and their *gradient*
    fades to zero instead of pulling with constant force forever. The plain mean
    is L1-like (bounded pull per point, but it never lets go); this is its
    outlier-proof version, the same lesson the prefit's Geman-McClure chamfer and
    both reprojection terms already carry. ``0`` = off.
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
        v, p = vertices_sub[s:e], depth_padded[s:e]
        with torch.no_grad():
            d = torch.cdist(v, p)                                              # (b, Ns, Md)
            jv = d.masked_fill(~depth_valid[s:e][:, None, :], INF).argmin(2)   # (b, Ns)
            iv = d.masked_fill(~vismask[s:e][:, :, None], INF).argmin(1)       # (b, Md)
        v2d = (v - p.gather(1, jv[..., None].expand(-1, -1, 3))).norm(dim=-1)  # (b, Ns)
        d2v = (p - v.gather(1, iv[..., None].expand(-1, -1, 3))).norm(dim=-1)  # (b, Md)
        if sat_m > 0.0:
            v2d = sat_m * torch.tanh(v2d / sat_m)
            d2v = sat_m * torch.tanh(d2v / sat_m)
        if vertex_weights is None:
            v2d_mean = (v2d * vismask[s:e]).sum(1) / nv[s:e].clamp_min(1)
        else:
            vis_w = vismask[s:e] * vertex_weights                         # (b, Ns)
            v2d_mean = (v2d * vis_w).sum(1) / vis_w.sum(1).clamp_min(1e-6)
        d2v_mean = (d2v * depth_valid[s:e]).sum(1) / nd[s:e].clamp_min(1)
        parts.append(v2d_mean + d2v_mean)                                 # (b,)
    terms = torch.where(fvalid, torch.cat(parts), vertices_sub.new_zeros(()))
    if frame_weights is None:
        return terms.sum() / fvalid.sum().clamp_min(1)
    fw = frame_weights * fvalid
    return (terms * fw).sum() / fw.sum().clamp_min(1e-6)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft repulsion pushing the body mesh out of the scene (penalty head on the SDF).

    Penalty = dead-zone + robust kernel on the penetration excess ``e =
    relu(−s − margin_m)``: no cost until penetration exceeds ``margin_m`` (tolerates
    soft tissue + depth/keypoint error), then a Huber ramp (quadratic ``≤ huber_m``,
    linear beyond — one deeply-buried vertex can't dominate). A vertex is gated out
    when its matched scene point is low confidence (``conf_thresh``), farther than
    ``max_dist_m`` (not near a surface / cloud hole), or "penetrating" deeper than
    ``max_pen_m`` (almost always a wrong correspondence behind an occluder).
    The gradient through ``s`` pushes the vertex out along the surface normal.
    Returns ``(loss, mean_depth_cm, n_penetrating)`` — ``loss`` the mean robust
    penalty over all active vertex-instances jointly (Σ ÷ total active count),
    the latter two detached. Zero active vertices reduce to zeros without a branch
    (no per-iteration host sync)."""
    active = (conf > conf_thresh) & (dmin < max_dist_m) & (sgn.detach() > -max_pen_m) & has
    rho = _huber(torch.relu(-sgn - margin_m), huber_m)               # dead-zone + Huber
    a = active.to(rho.dtype)
    n_active = a.sum()
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
    chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penetration loss = :func:`scene_signed_distance` + :func:`penetration_penalty`.

    Kept as a standalone entry point for the optimizer's before/after metrics pass;
    the per-iteration optimizer shares one :func:`scene_signed_distance` across the
    scene penalties instead of calling this per loss."""
    sgn, dmin, conf, has = scene_signed_distance(
        vertices_sub, scene_padded, normal_padded, conf_padded, scene_valid, chunk=chunk)
    return penetration_penalty(sgn, dmin, conf, has, conf_thresh=conf_thresh, margin_m=margin_m,
                               max_dist_m=max_dist_m, max_pen_m=max_pen_m, huber_m=huber_m)


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
    loss = (rho * a).sum() / n_active.clamp_min(1)                   # joint mean over all active verts
    with torch.no_grad():
        pushed = active & (excess > 0)
        gap_cm = ((sgn.detach() * pushed).sum() / pushed.sum().clamp_min(1)) * 100.0
        n_pushed = pushed.sum()
    return loss, gap_cm.detach(), n_pushed


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
    Returns ``(loss, mean_gap_cm, n_attracted)`` — the latter two detached. Zero
    attracted vertices reduce to zeros without a branch (no per-iteration host
    sync)."""
    sd = sgn.detach()
    active = (conf > conf_thresh) & (dmin < max_dist_m) & (sd > 0.0) & (sd < band_m) & has
    if active_mask is not None:
        active = active & active_mask
    a = active.to(sgn.dtype)
    loss = (sgn * a).sum() / a.sum().clamp_min(1e-6)             # grad pulls each attracted v inward (−n)
    with torch.no_grad():
        gap_cm = ((sd * active.to(sd.dtype)).sum() / active.sum().clamp_min(1)) * 100.0
        n_attr = active.sum()
    return loss, gap_cm.detach(), n_attr


def acquisition_penalty(
    v_star: torch.Tensor,   # (I, 3) posed member vertex (grad through q)
    p_star: torch.Tensor,   # (I, 3) its nearest target-patch point (detached)
    *,
    delta_m: float,
    resolved_count: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Long-range pull of ONE vertex per positive contact instance toward its target patch.

    A contact label means SOME surface point of the group touches the wall, so only the
    group's single closest posed vertex ``v*`` is pulled toward its nearest patch point ``p*``
    (both chosen detached, upstream) — not the whole Voronoi patch. The penalty is a
    pseudo-Huber of the Euclidean gap ``d = ‖v* − p*‖``: ``ρ = δ²·(√(1 + (d/δ)²) − 1)`` —
    quadratic within ``δ`` (fades to 0 at contact) and ~linear far away (bounded, ~constant
    gradient), so a distant instance keeps a steady pull without a runaway force. Mean over the
    FIXED per-phase resolved-instance count (patches built once), so instances do not dilute
    each other as the fit moves. Gradient flows through ``v*`` only. Returns
    ``(loss, mean_gap_cm, n_instances)`` — the latter two detached diagnostics."""
    d = (v_star - p_star).norm(dim=-1)                              # (I,)
    rho = (delta_m * delta_m) * (torch.sqrt(1.0 + (d / delta_m).square()) - 1.0)
    loss = rho.sum() / max(int(resolved_count), 1)
    with torch.no_grad():
        gap_cm = d.mean() * 100.0 if d.numel() else d.new_zeros(())
    return loss, gap_cm.detach(), int(d.shape[0])


def seating_penalty(
    v_star: torch.Tensor,   # (I, 3) posed member vertex (grad through q)
    p_star: torch.Tensor,   # (I, 3) its nearest target-patch point (detached)
    n_star: torch.Tensor,   # (I, 3) that point's outward unit normal (detached)
    *,
    target_m: float,
    band_m: float,
    huber_m: float,
    resolved_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Near-field SYMMETRIC seat of the acquisition vertex at slight penetration.

    Where :func:`acquisition_penalty` only closes the gap, this fixes WHERE the vertex sits on
    the patch plane: the signed distance ``s = (v* − p*)·n*`` is driven to ``target_m`` (a few
    mm inside) by a Huber on ``|s − target_m|``, active only within ``±band_m`` of the surface
    (detached gate — outside the band acquisition owns the pull). Being symmetric it also
    pushes the vertex back OUT once it overshoots inward — the anti-burial term the penetration
    guards' dropout cannot provide. Mean over the SAME fixed resolved-instance count as
    acquisition, so an instance entering/leaving the band does not rescale the others'
    gradient. Gradient flows through ``v*``. Returns ``(loss, mean_seat_cm, n_active)`` — the
    latter two detached diagnostics."""
    s = ((v_star - p_star) * n_star).sum(dim=-1)                    # (I,) patch-plane signed distance
    active = s.detach().abs() <= band_m
    rho = _huber((s - target_m).abs(), huber_m)                     # symmetric Huber toward target_m
    a = active.to(rho.dtype)
    loss = (rho * a).sum() / max(int(resolved_count), 1)
    with torch.no_grad():
        seat_cm = ((s * a).sum() / a.sum().clamp_min(1)) * 100.0
        n_active = active.sum()
    return loss, seat_cm.detach(), n_active


def joint_velocity_loss(
    joints_world: torch.Tensor,    # (T, J, 3) posed (scaled) world joints — grad through q
    contact_labels: torch.Tensor,  # (T, J) int8 tri-state: 1 = annotated in-contact this frame
    *,
    fps: float,
    world_quats: torch.Tensor | None = None,   # (T, J, 4) scalar-last world joint orientation
    angular_weight: float = 0.0,               # angular term weight relative to the linear one
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Damp the world velocity of joints a human annotated as in contact.

    A joint touching the wall should be (near-)stationary while the contact lasts,
    so for each frame-to-frame step we penalize the squared world displacement of
    every joint flagged in contact at **both** ends of the step::

        L_lin = mean_{t, j : contact[t,j]==1 ∧ contact[t+1,j]==1} ‖p_{t+1,j} − p_{t,j}‖²

    (the constant ``fps²`` folds into the weight). Gating on both frames matches
    kindyn's ``fpair`` convention: the release step (contact→free) is not
    damped, and any step touching an invalid frame drops out for free (held frames
    carry label ``-1`` ≠ 1). The finite difference of the posed world joints is
    directly differentiable through ``q``, so gradients flow through
    ``joints_world`` only.

    A joint that grips also should not *twist*: when ``world_quats`` is given and
    ``angular_weight > 0`` the same in-contact gate additionally damps the
    joint's **angular** velocity — the frame-to-frame change of its world
    orientation, measured as ``1 − (q_t · q_{t+1})²`` (= ``sin²(θ/2)`` for a
    relative rotation ``θ``, the same quaternion distance the pose priors use).
    The combined loss is ``L_lin + angular_weight · L_ang``. Holding position is
    the harder constraint (linear jitter in metres vs the dimensionless angular
    distance), so ``angular_weight`` is meant to be small. Returns
    ``(loss, mean_speed_cm_s, mean_ang_deg_s, n_contact_steps)`` — the middle two
    detached diagnostics (mean in-contact joint linear speed cm/s and angular
    speed deg/s), and how many (frame, joint) steps were penalized."""
    if joints_world.shape[0] < 2:
        z = joints_world.new_zeros(())
        return z, z, z, joints_world.new_zeros((), dtype=torch.long)
    dp = joints_world[1:] - joints_world[:-1]            # (T-1, J, 3) world displacement / step
    sp2 = (dp * dp).sum(dim=-1)                          # (T-1, J) squared per-step speed (×fps²→weight)
    pair = (contact_labels[:-1] == 1) & (contact_labels[1:] == 1)  # in contact at BOTH ends of the step
    gate = pair.to(sp2.dtype)
    n = gate.sum()
    loss = (sp2 * gate).sum() / n.clamp_min(1.0)
    ang_deg_s = joints_world.new_zeros(())
    if world_quats is not None and angular_weight > 0.0:
        dot = (world_quats[1:] * world_quats[:-1]).sum(dim=-1)      # (T-1, J) cos(θ/2), sign-ambiguous
        loss = loss + angular_weight * ((1.0 - dot.square()) * gate).sum() / n.clamp_min(1.0)
        with torch.no_grad():
            theta = 2.0 * torch.acos(dot.abs().clamp(max=1.0))     # relative rotation angle (rad)
            ang_deg_s = (torch.rad2deg(theta) * gate).sum() / n.clamp_min(1.0) * float(fps)
    with torch.no_grad():
        speed_cm_s = ((sp2.sqrt() * gate).sum() / n.clamp_min(1.0)) * float(fps) * 100.0
        n_steps = pair.sum()
    return loss, speed_cm_s.detach(), ang_deg_s.detach(), n_steps


def joint_acceleration_loss(
    joints_world: torch.Tensor,    # (T, J, 3) posed (scaled) world joints — grad through q
    contact_labels: torch.Tensor,  # (T, J) int8 tri-state: 1 = annotated in-contact this frame
    *,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Damp the world (Cartesian) ACCELERATION of joints a human annotated as in contact.

    The sibling of :func:`joint_velocity_loss` one time-derivative up: a gripping
    joint should move not just slowly but *steadily* — no jerk into or out of the
    hold. For every 3-frame stencil we penalize the squared world second difference
    of each joint flagged in contact across **all three** frames::

        L = mean_{t, j : contact[t−1,j]=contact[t,j]=contact[t+1,j]=1}
                ‖p_{t+1,j} − 2 p_{t,j} + p_{t−1,j}‖²

    That second difference is the joint origin's LOCAL_WORLD_ALIGNED linear
    acceleration (the world-axis 2nd time derivative of its position), with the
    constant ``fps²`` folded into the weight. Gating on all three frames drops the
    enter/exit steps and any stencil that touches an
    invalid frame (held frames carry label ``-1`` ≠ 1). Gradients flow through
    ``joints_world`` only. Returns ``(loss, mean_accel_cm_s2, n_contact_stencils)`` —
    the latter two detached diagnostics."""
    if joints_world.shape[0] < 3:
        z = joints_world.new_zeros(())
        return z, z, joints_world.new_zeros((), dtype=torch.long)
    d2 = joints_world[2:] - 2.0 * joints_world[1:-1] + joints_world[:-2]   # (T-2, J, 3) world 2nd diff
    a2 = (d2 * d2).sum(dim=-1)                                             # (T-2, J) squared per-stencil accel
    trip = ((contact_labels[:-2] == 1) & (contact_labels[1:-1] == 1)      # in contact at all 3 stencil frames
            & (contact_labels[2:] == 1))
    gate = trip.to(a2.dtype)
    n = gate.sum()
    loss = (a2 * gate).sum() / n.clamp_min(1.0)
    with torch.no_grad():
        accel_cm_s2 = ((a2.sqrt() * gate).sum() / n.clamp_min(1.0)) * float(fps) ** 2 * 100.0
        n_stencils = trip.sum()
    return loss, accel_cm_s2.detach(), n_stencils


def pose_prior_loss(
    q: torch.Tensor,        # (T, nq) normalized, bh order
    q_init: torch.Tensor,   # (T, nq) normalized, bh order
    num_joints: int,
    *,
    num_prior_joints: int | None = None,
    joint_weight: torch.Tensor | None = None,  # (T, num_joints-1) per-joint weight in [0,1]
) -> torch.Tensor:
    """Smooth quaternion distance ``1 − (q·q_init)²`` over joints (not root).

    Excludes the root SE3 (``q[:7]``): the body / hand pose stays near the
    converter's per-frame estimate while root translation + orientation move
    freely.

    ``num_prior_joints`` covers only the first N parent-local quaternions — the
    body block, dropping the fingers. Used when the finger pose is being *fitted*
    (``freeze_fingers`` off): the converter's hand pose is then the thing we
    stopped trusting, so pulling the fingers back to it is incoherent. The
    denominator stays the full joint count, so dropping the fingers does not
    silently stiffen the prior on the joints it still covers.

    ``joint_weight`` (:func:`~tools.human_optim.keypoints.visibility_weights`) scales each
    joint's contribution, so a joint the camera could not see is pulled less toward a
    starting pose the camera could not see either. Like ``num_prior_joints`` it holds the
    denominator, so the prior *loosens* on those joints instead of stiffening on the rest.
    """
    shape = q.shape[:-1]
    qj = q[..., 7:].reshape(*shape, num_joints - 1, 4)
    qij = q_init[..., 7:].reshape(*shape, num_joints - 1, 4)
    vals = 1.0 - (qj * qij).sum(dim=-1).square()
    if joint_weight is not None:
        vals = vals * joint_weight
    if num_prior_joints is None:
        return vals.mean()
    return vals[..., :num_prior_joints].sum() / vals.numel()


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
    as in :func:`joint_velocity_loss`. A step counts only if ``frame_valid`` at
    BOTH ends (the fpair convention). ``T<2`` ⇒ 0.
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
