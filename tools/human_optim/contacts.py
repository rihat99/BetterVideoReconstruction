"""Scene-contact estimation for the optimized SMPL-X bodies.

Given a posed (world-frame, scaled) SMPL-X mesh and the static scene point cloud,
mark which body vertices touch the scene with a point-cloud SDF, then lift the
per-vertex contacts to **per-joint** contacts by nearest-joint assignment (each
vertex belongs to the joint center it is spatially closest to in the rest pose).

The SDF is the same signed distance the kinematic stage's scene-penetration loss
uses (:func:`tools.human_optim.losses.scene_penetration`): for each body vertex,
the nearest scene point ``p`` with its outward PCA normal ``n`` gives a signed
distance ``(v − p)·n`` — negative behind the surface (penetrating), positive in
front — using the camera-oriented normals from
:func:`tools.geometry.normals.estimate_point_normals`. A vertex is *in contact*
when its signed distance falls below a small threshold (inside the scene, or just
outside it).

This is a one-shot analysis (no gradients, run once per clip), so the scene
nearest-neighbour is a plain per-frame pruned ``cdist`` rather than the
chunked-batched machinery the per-iteration losses need.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def scene_sdf(
    vertices: torch.Tensor,        # (T, V, 3) posed world-frame (scaled) vertices
    scene_pts: torch.Tensor,       # (M, 3) world scene points
    scene_normals: torch.Tensor,   # (M, 3) outward unit normals
    scene_conf: torch.Tensor,      # (M,) per-point confidence
    *,
    prune_margin_m: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-vertex signed distance to the scene surface, frame by frame.

    Returns ``(sdf, dmin, conf)``, each ``(T, V)``:

      * ``sdf``  — **perpendicular** signed distance ``(v−p)·n`` (metres); negative
        = behind the surface (penetrating). This is the *same* convention the
        kinematic penetration loss uses, so contact and repulsion agree: a vertex
        far to the *side* of the surface reads ~0 (grazing) rather than as a deep
        penetration equal to its Euclidean distance — the far ``dmin`` guard is
        what rejects those lateral strays, not a spurious large negative ``sdf``.
      * ``dmin`` — Euclidean distance to that nearest scene point (metres); the
        far-guard (a vertex with no scene point within ``max_dist_m`` is not a
        contact, whatever its perpendicular sign).
      * ``conf`` — confidence of that nearest scene point.

    The scene cloud is large (~10⁵ pts) but only points near the body matter, so
    each frame prunes to the posed-vertex AABB + ``prune_margin_m`` before the
    ``cdist`` (the body is ~2 m; the pruned set is a few thousand points). Frames
    whose body box contains no scene point return ``+inf`` distance / zero
    confidence (⇒ no contact).
    """
    T, V = vertices.shape[0], vertices.shape[1]
    dev, dt = vertices.device, vertices.dtype
    INF = torch.finfo(dt).max
    sdf = torch.full((T, V), INF, device=dev, dtype=dt)
    dmin = torch.full((T, V), INF, device=dev, dtype=dt)
    conf = torch.zeros((T, V), device=dev, dtype=scene_conf.dtype)
    for t in range(T):
        v = vertices[t]                                          # (V, 3)
        lo = v.amin(0) - prune_margin_m
        hi = v.amax(0) + prune_margin_m
        m = ((scene_pts >= lo) & (scene_pts <= hi)).all(dim=-1)  # (M,) bool
        if not bool(m.any()):
            continue
        cp, cn, cc = scene_pts[m], scene_normals[m], scene_conf[m]
        d = torch.cdist(v[None], cp[None])[0]                    # (V, Mc)
        dm, idx = d.min(dim=1)                                   # (V,)
        p, n = cp[idx], cn[idx]
        sdf[t] = ((v - p) * n).sum(dim=-1)                       # (V,) perpendicular signed distance
        dmin[t] = dm
        conf[t] = cc[idx]
    return sdf, dmin, conf


def contact_from_sdf(
    sdf: torch.Tensor,      # (T, V) signed Euclidean SDF (m)
    dmin: torch.Tensor,     # (T, V) Euclidean NN distance (m)
    conf: torch.Tensor,     # (T, V) NN-point confidence
    *,
    close_thresh_m: float,
    max_dist_m: float,
    conf_thresh: float,
) -> torch.Tensor:
    """Boolean per-vertex contact ``(T, V)``.

    A vertex is in contact when it is **inside** the scene (``sdf < 0``) or
    **just outside** it (``0 ≤ sdf < close_thresh_m``) — i.e. ``sdf <
    close_thresh_m`` — provided it is genuinely near a confident surface. Two
    guards (mirroring the penetration loss) keep strays from false-flagging:
    ``dmin < max_dist_m`` rejects a vertex whose only "penetration" is a deep,
    almost-always-spurious correspondence behind an occluder, and ``conf >
    conf_thresh`` trusts only well-observed scene points.
    """
    return (sdf < close_thresh_m) & (dmin < max_dist_m) & (conf > conf_thresh)


@torch.no_grad()
def joint_occlusion_counts(
    vertices: torch.Tensor,        # (T, V, 3) posed world-frame vertices
    joints: torch.Tensor,          # (T, J, 3) posed world-frame joint centres
    cam_centers: torch.Tensor,     # (T, 3) world camera centres
    v2j: torch.Tensor,             # (V,) long vertex→joint
    coarse_of_joint: torch.Tensor, # (J,) int coarse-group id per joint (-1 = none)
    *,
    perp_m: float = 0.04,
    front_m: float = 0.05,
    subsample: int = 2500,
    chunk: int = 16,
) -> torch.Tensor:
    """Per-(frame, joint) count of body vertices that occlude the joint from the camera ``(T, J)``.

    A vertex occludes joint ``j`` when it lies within ``perp_m`` of the camera→joint ray and
    at least ``front_m`` closer to the camera than the joint (in front of it), and its nearest
    joint is **not** in ``j``'s own coarse group — so a hand's own fingers do not count as
    occluding the wrist, but the torso / other limbs do. The count is a monotone occlusion
    margin: 0 = fully visible, large = deeply hidden behind other body parts. Vertices are
    uniformly subsampled to ``subsample`` for speed (a count over a dense region is robust).
    """
    T, V, _ = vertices.shape
    J = joints.shape[1]
    if V > subsample:
        vidx = torch.linspace(0, V - 1, subsample, device=vertices.device).long()
        vertices, v2j = vertices[:, vidx], v2j[vidx]
    coarse_of_v = coarse_of_joint[v2j]                                       # (Vs,)
    valid_occ = (coarse_of_v[:, None] != coarse_of_joint[None, :])          # (Vs,J) may this vertex occlude j?
    n_occ = torch.zeros((T, J), device=vertices.device)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        C = cam_centers[s:e]                                                 # (F,3)
        u = joints[s:e] - C[:, None, :]                                      # (F,J,3)
        L = u.norm(dim=-1)                                                   # (F,J) joint depth along ray
        uhat = u / L[..., None].clamp_min(1e-9)
        w = vertices[s:e] - C[:, None, :]                                    # (F,Vs,3)
        depth = torch.einsum("fvc,fjc->fvj", w, uhat)                        # (F,Vs,J)
        perp = (w[:, :, None, :] - depth[..., None] * uhat[:, None, :, :]).norm(dim=-1)
        occ = (perp < perp_m) & (depth > 0.0) & (depth < (L[:, None, :] - front_m)) & valid_occ[None]
        n_occ[s:e] = occ.sum(dim=1).float()
    return n_occ


def vertex_to_joint(vertices: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    """Map each vertex to its **nearest joint center** ``(V,)`` (rest-pose 3D).

    ``vertices (V, 3)`` and ``joints (J, 3)`` are the shaped rest-pose mesh and
    joint locations (``SMPLBase.forward_shape``); each vertex is assigned to the
    spatially closest joint. This is the "vertex belongs to joint" relation used
    to lift per-vertex contacts to per-joint contacts.

    Nearest-joint, rather than LBS-``argmax`` (the joint that *deforms* the
    vertex), keeps the contact label where the surface physically is. Skinning
    dominance pulls a whole limb segment up to its *proximal* joint — the thigh,
    including flesh sitting right at the knee, all reads "hip" — displacing the
    contact 10–30 cm from the touch. Nearest-joint roughly halves that
    displacement and orphans nothing (every joint still owns ≥25 verts, fingers
    included).
    """
    return torch.cdist(vertices[None], joints[None])[0].argmin(dim=1)


def lift_joint_contact(
    vcontact: torch.Tensor,         # (T, V) bool per-vertex contact
    sdf: torch.Tensor,              # (T, V) signed SDF (m) — for the diagnostic
    v2j: torch.Tensor,              # (V,) long vertex→joint
    num_joints: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift per-vertex contacts to per-joint ``(T, J)``.

    A joint is in contact when **any** vertex assigned to it is in contact (a
    ``scatter`` max over the joint's vertices). Also returns ``joint_sdf_cm
    (T, J)`` — the *minimum* signed distance (cm) over each joint's vertices, the
    continuous quantity behind the boolean (handy for thresholding / diagnosis).
    A joint with no near-surface vertex this frame gets ``nan``.
    """
    T = vcontact.shape[0]
    dev = vcontact.device
    INF = torch.finfo(sdf.dtype).max
    idx = v2j[None].expand(T, -1)                               # (T, V)
    jc = torch.zeros((T, num_joints), device=dev)
    jc.scatter_reduce_(1, idx, vcontact.to(jc.dtype), reduce="amax", include_self=True)
    jsdf = torch.full((T, num_joints), INF, device=dev, dtype=sdf.dtype)
    jsdf.scatter_reduce_(1, idx, sdf, reduce="amin", include_self=True)
    jsdf_cm = torch.where(jsdf < INF * 0.5, jsdf * 100.0, torch.full_like(jsdf, float("nan")))
    return jc > 0.5, jsdf_cm


def joint_guarded_min_sdf(
    sdf: torch.Tensor,      # (T, V) signed SDF (m)
    dmin: torch.Tensor,     # (T, V) Euclidean NN distance (m)
    conf: torch.Tensor,     # (T, V) NN-point confidence
    v2j: torch.Tensor,      # (V,) long vertex→joint
    num_joints: int,
    *,
    max_dist_m: float,
    conf_thresh: float,
) -> torch.Tensor:
    """Per-joint minimum signed distance over each joint's GUARDED vertices ``(T, J)``.

    A vertex contributes only if it passes the same far / confidence guards as
    :func:`contact_from_sdf` (``dmin < max_dist_m`` and ``conf > conf_thresh``);
    guarded-out vertices are set to ``+inf`` so they never win the per-joint min, and
    a joint with no guarded vertex this frame is ``+inf``. This is the continuous
    per-joint signal the temporal hysteresis (:func:`hysteresis_spans`) thresholds —
    the guarded counterpart of :func:`lift_joint_contact`'s raw ``joint_sdf`` (which
    stays unguarded as the diagnostic channel). Feeding hysteresis the guarded min
    keeps its ``enter`` threshold exactly equivalent to the old per-vertex boolean
    (``∃`` guarded vertex with ``sdf < enter``) while letting a looser ``exit`` hold
    the span.
    """
    guard = (dmin < max_dist_m) & (conf > conf_thresh)
    INF = torch.finfo(sdf.dtype).max
    gsdf = torch.where(guard, sdf, torch.full_like(sdf, INF))
    T = sdf.shape[0]
    idx = v2j[None].expand(T, -1)
    out = torch.full((T, num_joints), INF, device=sdf.device, dtype=sdf.dtype)
    out.scatter_reduce_(1, idx, gsdf, reduce="amin", include_self=True)
    return out


def hysteresis_spans(
    series: torch.Tensor,   # (T, J) per-joint continuous signal (smaller = more in contact)
    *,
    enter_below: float,
    exit_below: float,
    min_frames: int,
) -> torch.Tensor:
    """Temporal hysteresis + min-duration on a per-joint scalar series → ``(T, J)`` bool.

    ``series`` is a per-frame continuous signal where *smaller* means more in contact
    (here the guarded per-joint min signed distance). Per joint, independently:

      * **enter** a contact span at the first frame with ``series < enter_below``;
      * **stay** in it while ``series < exit_below`` — a looser threshold, so labels
        do not blink on and off at the boundary (``exit_below ≥ enter_below``);
      * close the span otherwise. A ``+inf`` / NaN frame (no guarded vertex) fails
        both tests and breaks the span.

    Then any span shorter than ``min_frames`` is dropped (1–2-frame flicker). Returns
    the boolean contact mask. Pure temporal post-process, no gradients.
    """
    T, J = series.shape
    enter = series < enter_below                       # (T, J) — start condition (⊆ stay)
    stay = series < exit_below                         # (T, J) — continue condition
    active = torch.zeros(J, dtype=torch.bool, device=series.device)
    out = torch.zeros((T, J), dtype=torch.bool, device=series.device)
    for t in range(T):
        active = (active & stay[t]) | enter[t]         # hysteresis: continue-if-staying, or (re)enter
        out[t] = active
    if min_frames > 1 and T > 0:
        oc = out.cpu().numpy()
        for j in range(J):
            col = oc[:, j]
            t = 0
            while t < T:
                if col[t]:
                    s = t
                    while t < T and col[t]:
                        t += 1
                    if t - s < min_frames:
                        col[s:t] = False
                else:
                    t += 1
        out = torch.as_tensor(oc, device=series.device)
    return out


# ── drift / stillness (for the estimate_contacts contact readout) ──
# Instantaneous joint speed cannot tell a gripping hand from a slow near-wall pass:
# a grip's few-mm per-frame fit jitter × ~30 fps is 0.1–0.3 m/s, the same range as a
# slow pass. The discriminative signal is the NET displacement over a short window
# (~1/3 s): jitter cancels so a grip nets < ~2 cm while a slow pass accumulates 3–7 cm.

def joint_drift(
    joints_world: torch.Tensor,   # (T, J, 3) posed world-frame joints
    *,
    fps: float,
    window_s: float,
) -> torch.Tensor:
    """Net joint displacement across a centered time window ``(T, J)`` metres.

    For every frame ``t`` and joint ``j``, the straight-line distance between the
    joint's position ``window_s/2`` seconds before and after::

        drift[t, j] = ‖p[t + h, j] − p[t − h, j]‖,   h = round(window_s · fps / 2)

    The window is clamped to the clip at the two ends, and the raw displacement is
    rescaled by ``2h / (hi − lo)`` so a half-length end window reports the same drift
    a full interior window would for the same motion — otherwise the first/last ``h``
    frames read as artificially still (a fast joint there would false-contact). Two
    gathers, no loop, no gradients — a signal to READ, never to optimize. ``T < 2`` ⇒
    all-zero.
    """
    T = joints_world.shape[0]
    if T < 2:
        return joints_world.new_zeros(joints_world.shape[:2])
    h = max(1, int(round(window_s * float(fps) / 2.0)))
    idx = torch.arange(T, device=joints_world.device)
    lo = (idx - h).clamp_min(0)
    hi = (idx + h).clamp_max(T - 1)
    disp = (joints_world[hi] - joints_world[lo]).norm(dim=-1)          # (T, J)
    span = (hi - lo).clamp_min(1).to(disp.dtype)                      # actual window frames (2h interior)
    return disp * (2 * h / span)[:, None]                            # scale short end-windows to full width


def stillness_weight(drift: torch.Tensor, *, still_m: float) -> torch.Tensor:
    """Per-joint stillness weight ``w ∈ [0, 1]`` from :func:`joint_drift` ``(T, J)``.

    A linear ramp ``w = clamp(1 − drift / (2·still_m), 0, 1)``: ``1`` at zero drift
    (fully still), ``0.5`` at ``still_m``, ``0`` at ``2·still_m`` and beyond. The one
    knob ``still_m`` is the net drift at which a joint is "half in contact"; it is the
    contact readout's recall knob — :func:`drift_gated_contact` kills a contact where
    ``w == 0`` (drift ≥ ``2·still_m``).
    """
    return (1.0 - drift / (2.0 * float(still_m))).clamp(0.0, 1.0)


def drift_gated_contact(
    guarded_min_sdf: torch.Tensor,   # (T, J) guarded per-joint min signed distance (m)
    drift: torch.Tensor,             # (T, J) per-joint net window drift (m)
    *,
    close_thresh_m: float,
    exit_thresh_m: float,
    min_frames: int,
    still_m: float,
) -> torch.Tensor:
    """The contact readout ``(T, J)`` bool: drift-gate, then temporal hysteresis.

    A joint-frame can only be in contact when it is both **near the wall** (its
    guarded min signed distance below the hysteresis thresholds) and **still** (net
    window drift under ``2·still_m``, i.e. :func:`stillness_weight` ``> 0``). Frames
    that fail the stillness gate have their distance forced to ``+inf`` so
    :func:`hysteresis_spans` breaks the span there — exactly how the old hard speed
    veto plugged in, but with the windowed-drift signal that actually separates a
    grip from a slow pass. Shared by ``estimate_contacts`` and the offline scorer so
    both score the identical readout.
    """
    w = stillness_weight(drift, still_m=still_m)
    INF = torch.finfo(guarded_min_sdf.dtype).max
    gated = torch.where(w > 0.0, guarded_min_sdf, torch.full_like(guarded_min_sdf, INF))
    return hysteresis_spans(gated, enter_below=close_thresh_m,
                            exit_below=exit_thresh_m, min_frames=min_frames)


def _bool_spans(col) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of a 1-D numpy bool array → list of ``(start, end)`` half-open."""
    out, t, T = [], 0, len(col)
    while t < T:
        if col[t]:
            s = t
            while t < T and col[t]:
                t += 1
            out.append((s, t))
        else:
            t += 1
    return out


def merge_contact_spans(
    jc: torch.Tensor,               # (T, J) bool contact readout
    joints_world: torch.Tensor,     # (T, J, 3) posed world-frame joints
    guarded_min_sdf: torch.Tensor,  # (T, J) guarded per-joint min signed distance (m)
    *,
    merge_dist_m: float,
    max_gap_s: float,
    gap_sdf_m: float,
    fps: float,
) -> torch.Tensor:
    """Bridge short same-joint contact gaps where the joint stayed put **and** near the wall.

    Real climbing has few contact switches: when a joint's two contact spans are separated
    by a brief gap that the joint barely moved across *and* stayed near the wall throughout,
    it never left the hold and the gap is detection flicker. Per joint, for consecutive
    spans (last contact frame ``t1``, next span's first contact frame ``t2``) with
    ``(t2 − t1) / fps ≤ max_gap_s``, the gap ``[t1+1, t2)`` is filled iff:

      * ``‖p[t2] − p[t1]‖ ≤ merge_dist_m`` — the joint returns to where it left, **and**
      * ``max(guarded_min_sdf[t1 … t2]) < gap_sdf_m`` — the **re-grip guard**: it stayed
        near the wall the whole gap. A hand that leaves a hold, wanders, and re-grips the
        *same* hold also ends where it started, but was genuinely off the wall in between
        (its mid-gap ``gsdf`` spikes past ``gap_sdf_m`` / to ``+inf``) — so it is not filled.

    Pure per-joint temporal post-process, no gradients. ``merge_dist_m ≤ 0`` disables it.
    """
    if merge_dist_m <= 0.0:
        return jc
    T, J = jc.shape
    max_gap = int(round(float(max_gap_s) * float(fps)))
    out = jc.clone()
    jc_np = jc.detach().cpu().numpy()
    jw = joints_world.detach().cpu().numpy()
    gs = guarded_min_sdf.detach().cpu().numpy()
    for j in range(J):
        spans = _bool_spans(jc_np[:, j])
        for a in range(len(spans) - 1):
            t1 = spans[a][1] - 1            # last contact frame of span a
            t2 = spans[a + 1][0]           # first contact frame of span b
            if (t2 - t1) > max_gap:
                continue                   # gap too long to be flicker
            if float(np.linalg.norm(jw[t2, j] - jw[t1, j])) > merge_dist_m:
                continue                   # the joint really moved between grips
            if float(gs[t1:t2 + 1, j].max()) >= gap_sdf_m:
                continue                   # re-grip guard: it left the wall in between
            out[t1 + 1:t2, j] = True
    return out
