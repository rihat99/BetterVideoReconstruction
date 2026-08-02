"""Scene-contact estimation for the optimized SMPL-X bodies.

**Contact lives on the 12 contact frames** — named mesh points (toes, heels,
knees, palms, fingertips) baked onto the body, each carrying its own outward
surface normal (:mod:`tools.body`). :func:`detect_frame_contacts` is the one
detector both the kinematic stage's frozen auto-labels and the standalone
``estimate_contacts`` readout call, so the two can never drift apart. The rest of
this module is the generic span/drift machinery it is built from
(:func:`joint_drift`, :func:`stillness_weight`, :func:`hysteresis_spans`,
:func:`drift_gated_contact`, :func:`merge_contact_spans` — all agnostic to what
the ``K`` axis means) plus the older per-joint pieces the out-of-scope stages
still import.

The older per-joint model, kept for those stages:
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

from collections.abc import Sequence
from typing import NamedTuple

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


_MT_EPS = 1e-8


def _moller_trumbore(
    origin: torch.Tensor, dirs: torch.Tensor,
    v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ray-triangle intersection for ``K`` rays vs ``F`` triangles (one shared origin).

    :param origin: ``(3,)`` common ray origin (the camera centre).
    :param dirs: ``(K, 3)`` unit ray directions.
    :param v0: ``(F, 3)`` first triangle vertex; ``v1`` / ``v2`` the other two.
    :returns: ``(t_hit (K, F), hit (K, F) bool)`` — ``t_hit`` is the along-ray distance
        to the intersection. A triangle with a non-finite (NaN) vertex never registers
        a hit, so invalid frames and missing other-person meshes drop out for free.
    """
    num_rays, num_faces = dirs.shape[0], v0.shape[0]
    edge1 = v1 - v0                                            # (F,3)
    edge2 = v2 - v0                                            # (F,3)
    pvec = torch.linalg.cross(
        dirs[:, None, :].expand(-1, num_faces, -1),
        edge2[None, :, :].expand(num_rays, -1, -1))            # (K,F,3)
    det = (edge1[None] * pvec).sum(-1)                         # (K,F)
    inv_det = 1.0 / torch.where(det.abs() < _MT_EPS, torch.full_like(det, _MT_EPS), det)
    tvec = origin - v0                                         # (F,3)
    bary_u = (tvec[None] * pvec).sum(-1) * inv_det             # (K,F)
    qvec = torch.linalg.cross(tvec, edge1)                    # (F,3)
    bary_v = (dirs[:, None, :] * qvec[None]).sum(-1) * inv_det  # (K,F)
    t_hit = (edge2[None] * qvec[None]).sum(-1) * inv_det       # (K,F)
    hit = ((det.abs() >= _MT_EPS) & (bary_u >= 0.0) & (bary_v >= 0.0)
           & (bary_u + bary_v <= 1.0) & (t_hit > _MT_EPS))
    return t_hit, hit


@torch.no_grad()
def joint_occlusion_gap(
    vertices: torch.Tensor,        # (T, V, 3) posed world-frame vertices
    joints: torch.Tensor,          # (T, J, 3) posed world-frame joint centres
    cam_centers: torch.Tensor,     # (T, 3) world camera centres
    faces: torch.Tensor,           # (F, 3) long SMPL-X triangles (shared topology)
    v2j: torch.Tensor,             # (V,) long vertex→joint
    occlusion_group: torch.Tensor, # (J,) nonnegative local-surface group id per joint
    *,
    other_vertices: torch.Tensor | None = None,  # (T,V,3) or (T,Pother,V,3)
) -> torch.Tensor:
    """Per-(frame, joint) in-front body-occlusion gap along the camera→joint ray ``(T, J)``.

    Casts one ray from the camera centre through each joint centre and intersects it
    (Möller–Trumbore) with the **full-resolution** triangles of the target person and
    of every other tracked person. The gap, in metres, is::

        gap = joint_depth − nearest_hit_depth   (hits strictly in front of the joint)

    the thickness of body surface standing in front of the joint. Only intersections
    nearer than the joint centre count, so a positive gap always means a surface sits
    between the camera and the joint (occluded). A surface at or behind the joint is not
    an occluder: with nothing in front the gap is ``NaN`` (clear) — the same value the
    joint gets when no triangle is hit at all.

    A triangle is the joint's **own** skin (never an occluder) when any of its three
    vertices is owned — via ``v2j`` — by a joint in the target joint's
    ``occlusion_group``. Other people's triangles are never own, so another tracked
    person can occlude the joint. ``other_vertices`` shares the target's ``faces``
    topology (the same ``V`` vertices per person); a triangle with a non-finite vertex
    (an invalid frame, or a missing other-person mesh) never registers a hit.
    """
    T, V, _ = vertices.shape
    J = joints.shape[1]
    if occlusion_group.shape != (J,) or bool((occlusion_group < 0).any()):
        raise ValueError("occlusion_group must contain one nonnegative group ID per joint")
    device = vertices.device
    faces = faces.long()
    num_faces = faces.shape[0]

    verts_all, faces_all = vertices, faces
    if other_vertices is not None:
        if (other_vertices.ndim not in (3, 4) or other_vertices.shape[0] != T
                or other_vertices.shape[-1] != 3 or other_vertices.shape[-2] != V):
            raise ValueError(
                "other_vertices must have shape (T,V,3) or (T,Pother,V,3) with the target's V")
        people = other_vertices[:, None] if other_vertices.ndim == 3 else other_vertices
        verts_all = torch.cat((vertices, *people.unbind(dim=1)), dim=1)   # (T,(1+Po)*V,3)
        faces_all = torch.cat(
            [faces + k * V for k in range(1 + people.shape[1])], dim=0)   # ((1+Po)*F,3)

    # Per-face own-group test: a target triangle is own for joint j when ANY of its 3
    # vertices belongs to j's occlusion group (conservative — avoids seam holes). Other
    # people's faces carry a sentinel group that matches no joint, so are never own.
    face_group = torch.full((faces_all.shape[0], 3), -1, dtype=torch.long, device=device)
    face_group[:num_faces] = occlusion_group[v2j][faces]                  # (F,3)
    own_mask = (occlusion_group[:, None, None] == face_group[None]).any(dim=-1)  # (J,Ftot)

    inf = torch.tensor(float("inf"), device=device, dtype=vertices.dtype)
    gap = torch.empty((T, J), device=device, dtype=vertices.dtype)
    for t in range(T):                                                    # one frame bounds memory
        origin = cam_centers[t]
        direction = joints[t] - origin                                   # (J,3)
        length = direction.norm(dim=-1)                                  # (J,)
        dirs = direction / length.clamp_min(1e-9)[:, None]
        v0, v1, v2 = (verts_all[t][faces_all[:, k]] for k in range(3))   # (Ftot,3) each
        t_hit, hit = _moller_trumbore(origin, dirs, v0, v1, v2)          # (J,Ftot)
        # An occluder must sit strictly in FRONT of the joint: eps < t_hit < ray length − eps.
        # The lower bound is already in `hit`; a hit at or behind the joint centre is not an
        # occluder, so when none survives the joint is clear (NaN gap), like no hit at all.
        in_front = hit & ~own_mask & (t_hit < length[:, None] - _MT_EPS)
        nearest = torch.where(in_front, t_hit, inf).amin(dim=1)          # (J,)
        gap[t] = torch.where(torch.isfinite(nearest), length - nearest,
                             torch.full_like(length, float("nan")))
    return gap


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


def lift_joint_min_distance(
    distance_m: torch.Tensor,       # (T, V) nonnegative per-vertex distance (m)
    v2j: torch.Tensor,              # (V,) long vertex→joint
    num_joints: int,
) -> torch.Tensor:
    """Minimum Euclidean scene distance for each joint's owned surface ``(T,J)``.

    Unlike signed wall-normal SDF, this is a true clearance magnitude and cannot
    read near zero merely because a far-away correspondence lies tangent to a bad
    scene normal. A joint with no finite scene candidate is returned as ``NaN``.
    """
    T = distance_m.shape[0]
    INF = torch.finfo(distance_m.dtype).max
    idx = v2j[None].expand(T, -1)
    finite_distance = torch.where(
        torch.isfinite(distance_m), distance_m, torch.full_like(distance_m, INF))
    out = torch.full(
        (T, num_joints), INF, device=distance_m.device, dtype=distance_m.dtype)
    out.scatter_reduce_(1, idx, finite_distance, reduce="amin", include_self=True)
    return torch.where(out < INF * 0.5, out, torch.full_like(out, float("nan")))


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
    valid_mask: torch.Tensor | None = None,
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
    all-zero. When ``valid_mask (T,)`` is supplied, a sample whose current frame
    or either window endpoint is invalid is returned as ``NaN``. This prevents a
    camera-glued tracking gap from manufacturing motion at an adjacent valid
    frame; downstream confidence treats the missing motion witness as zero.
    """
    T = joints_world.shape[0]
    if T < 2:
        drift = joints_world.new_zeros(joints_world.shape[:2])
        if valid_mask is not None:
            valid_mask = torch.as_tensor(
                valid_mask, dtype=torch.bool, device=joints_world.device)
            if valid_mask.shape != (T,):
                raise ValueError(
                    f"valid_mask must have shape ({T},), got {tuple(valid_mask.shape)}")
            drift = torch.where(
                valid_mask[:, None], drift, torch.full_like(drift, float("nan")))
        return drift
    h = max(1, int(round(window_s * float(fps) / 2.0)))
    idx = torch.arange(T, device=joints_world.device)
    lo = (idx - h).clamp_min(0)
    hi = (idx + h).clamp_max(T - 1)
    disp = (joints_world[hi] - joints_world[lo]).norm(dim=-1)          # (T, J)
    span = (hi - lo).clamp_min(1).to(disp.dtype)                      # actual window frames (2h interior)
    drift = disp * (2 * h / span)[:, None]                           # scale short end-windows to full width
    if valid_mask is not None:
        valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=joints_world.device)
        if valid_mask.shape != (T,):
            raise ValueError(f"valid_mask must have shape ({T},), got {tuple(valid_mask.shape)}")
        window_valid = valid_mask & valid_mask[lo] & valid_mask[hi]
        drift = torch.where(window_valid[:, None], drift, torch.full_like(drift, float("nan")))
    return drift


def stillness_weight(drift: torch.Tensor, *, still_m: float) -> torch.Tensor:
    """Per-joint stillness weight ``w ∈ [0, 1]`` from :func:`joint_drift` ``(T, J)``.

    A linear ramp ``w = clamp(1 − drift / (2·still_m), 0, 1)``: ``1`` at zero drift
    (fully still), ``0.5`` at ``still_m``, ``0`` at ``2·still_m`` and beyond. The one
    knob ``still_m`` is the net drift at which a joint is "half in contact"; it is the
    contact readout's recall knob — :func:`drift_gated_contact` kills a contact where
    ``w == 0`` (drift ≥ ``2·still_m``).
    """
    return (1.0 - drift / (2.0 * float(still_m))).clamp(0.0, 1.0)


def drift_gate(guarded_min_sdf: torch.Tensor, drift: torch.Tensor, *,
               still_m: float) -> torch.Tensor:
    """``guarded_min_sdf (T, K)`` with every step that failed the stillness gate at ``+inf``.

    A step counts as still while :func:`stillness_weight` is above zero (drift below
    ``2·still_m``). Forcing the rest to the sentinel — rather than masking — is what lets
    :func:`hysteresis_spans` run unchanged: a moving step simply looks infinitely far from
    the wall, so it can neither start nor hold a span.
    """
    weight = stillness_weight(drift, still_m=still_m)
    infinity = torch.finfo(guarded_min_sdf.dtype).max
    return torch.where(weight > 0.0, guarded_min_sdf,
                       torch.full_like(guarded_min_sdf, infinity))


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
    gated = drift_gate(guarded_min_sdf, drift, still_m=still_m)
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


# ── contact frames: the shared detector ───────────────────────────────────────
# A contact frame is one mesh point that carries its own outward surface normal
# (tools/body.py). TWO things must agree before it counts as touching:
#   surface  — a confident scene point is close, and the frame sits at/just
#              inside it (signed distance below the hysteresis thresholds);
#   stillness — the frame barely drifted over a short window.
# The frame's own normal is deliberately NOT a gate. Both inputs it would need are
# too noisy to veto on: the scene normals come from a reconstructed point cloud, and
# the frame's normal comes from the fitted pose. Gating on "does the skin face the
# wall?" threw away real contacts — a fingertip 3 cm from a hold whose baked pad
# normal pointed away because the hand pose was wrong. The normal survives in exactly
# ONE role, telling the two toe frames apart: see :func:`suppress_toe_pair`.


@torch.no_grad()
def frame_scene_signals(
    frame_points: torch.Tensor,    # (T, F, 3) world contact-frame positions
    frame_normals: torch.Tensor,   # (T, F, 3) world outward unit normals
    scene_pts: torch.Tensor,       # (M, 3) world scene points
    scene_normals: torch.Tensor,   # (M, 3) outward unit normals
    scene_conf: torch.Tensor,      # (M,) per-point confidence
    *,
    prune_margin_m: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-(frame index, contact frame) scene signals ``(sdf, dmin, conf, normal_dot)``.

    Each output is ``(T, F)``:

      * ``sdf`` — **perpendicular** signed distance ``(p_f − p_nn)·n_scene`` to the
        nearest scene point (metres); negative = behind the surface. Same
        convention as :func:`scene_sdf`, so contact and the penetration loss agree.
      * ``dmin`` — Euclidean distance to that nearest point (the far guard).
      * ``conf`` — that point's confidence (the cloud-hole guard).
      * ``normal_dot`` — ``dot(n_frame, −n_scene)`` ∈ [−1, 1]: ``+1`` when the
        skin looks straight into the wall, ``−1`` when it faces away.

    Only 12 points per time step, so the nearest neighbour is a plain per-frame
    ``cdist`` after pruning the scene to the points' AABB + ``prune_margin_m``
    (the margin must be at least the caller's ``max_dist_m``, or a true neighbour
    just outside the box is lost). A time step with no scene point in its box
    returns ``+inf`` distance, zero confidence and ``normal_dot = −1`` — no contact.
    """
    num_steps, num_frames = frame_points.shape[0], frame_points.shape[1]
    device, dtype = frame_points.device, frame_points.dtype
    infinity = torch.finfo(dtype).max
    sdf = torch.full((num_steps, num_frames), infinity, device=device, dtype=dtype)
    dmin = torch.full((num_steps, num_frames), infinity, device=device, dtype=dtype)
    conf = torch.zeros((num_steps, num_frames), device=device, dtype=scene_conf.dtype)
    normal_dot = torch.full((num_steps, num_frames), -1.0, device=device, dtype=dtype)
    for t in range(num_steps):
        points = frame_points[t]                                          # (F, 3)
        if not bool(torch.isfinite(points).all()):
            continue
        low = points.amin(0) - prune_margin_m
        high = points.amax(0) + prune_margin_m
        near_box = ((scene_pts >= low) & (scene_pts <= high)).all(dim=-1)  # (M,) bool
        if not bool(near_box.any()):
            continue
        candidate_pts = scene_pts[near_box]
        candidate_normals = scene_normals[near_box]
        distance = torch.cdist(points[None], candidate_pts[None])[0]       # (F, Mc)
        nearest_distance, nearest = distance.min(dim=1)                    # (F,)
        point, normal = candidate_pts[nearest], candidate_normals[nearest]
        sdf[t] = ((points - point) * normal).sum(dim=-1)
        dmin[t] = nearest_distance
        conf[t] = scene_conf[near_box][nearest]
        normal_dot[t] = -(frame_normals[t] * normal).sum(dim=-1)
    return sdf, dmin, conf, normal_dot


def frame_guarded_sdf(
    sdf: torch.Tensor,          # (T, F) perpendicular signed distance (m)
    dmin: torch.Tensor,         # (T, F) Euclidean NN distance (m)
    conf: torch.Tensor,         # (T, F) NN-point confidence
    *,
    max_dist_m: float,
    conf_thresh: float,
) -> torch.Tensor:
    """The continuous contact signal ``(T, F)``: ``sdf`` where both guards pass, ``+inf`` else.

    Two guards, both cheap and both "no evidence ⇒ no contact":

      * ``dmin < max_dist_m`` — the nearest scene point must genuinely be near
        (a match behind an occluder is not a surface);
      * ``conf > conf_thresh`` — a hole in the cloud must read as *unknown*, not
        as *no wall*.

    Forcing the failures to ``+inf`` (rather than masking) is what lets the plain
    hysteresis / minimum-duration / span-merge machinery run unchanged — a guarded
    step simply looks infinitely far from the wall, so it can never start or hold
    a contact span.
    """
    guard = (dmin < max_dist_m) & (conf > conf_thresh)
    infinity = torch.finfo(sdf.dtype).max
    return torch.where(guard, sdf, torch.full_like(sdf, infinity))


#: The two contact frames on each toe. They sit millimetres apart on the same toe, so no
#: distance / confidence / stillness gate can ever separate them — only their normals can.
TOE_PAIRS: tuple[tuple[str, str], ...] = (
    ("toe_under_left", "toe_above_left"),
    ("toe_under_right", "toe_above_right"),
)


def toe_pair_slots(frame_names: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Resolve :data:`TOE_PAIRS` to ``(under, above)`` column indices into ``frame_names``.

    Always by NAME, never by arithmetic on positions: the stored frame order is not the
    obvious one (slots 10/11 are finger_RIGHT then finger_LEFT). A pair whose two names are
    not both present is skipped, so a caller with a partial frame set still works.
    """
    slot = {str(name): i for i, name in enumerate(frame_names)}
    return tuple((slot[under], slot[above]) for under, above in TOE_PAIRS
                 if under in slot and above in slot)


def suppress_toe_pair(
    gated_sdf: torch.Tensor,    # (T, F) drift-gated signed distance (m); +inf = cannot contact
    normal_dot: torch.Tensor,   # (T, F) dot(frame normal, −scene normal)
    pairs: Sequence[tuple[int, int]],
    *,
    exit_thresh_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep at most ONE toe frame per foot — the only place a normal decides anything.

    ``toe_under`` (a toe hooked *under* a hold) and ``toe_above`` (a toe standing *on* it)
    are the same place on the toe with opposite normals, so every other gate accepts or
    rejects them together and both fire at once. We assume at most one of a pair really
    touches at any step — both-at-once is physically possible but rare, and deliberately
    ignored — and let the normal break the tie.

    **Candidacy**: at step ``t`` a toe is a candidate when its gated signal is below
    ``exit_thresh_m``, the loosest level at which :func:`hysteresis_spans` could still hold
    it in a span. Where both toes of a foot are candidates, the one with the larger
    ``normal_dot`` (looking more into the wall) keeps its signal and the other is forced to
    the ``+inf`` sentinel — before hysteresis, so the loser can neither enter nor hold a
    span at that step. Exact ties keep ``toe_under``, so the result never depends on
    evaluation order. Steps where only one toe is a candidate pass through untouched.

    Returns ``(gated_sdf, suppressed)``, both ``(T, F)`` — ``suppressed`` marks the steps
    this rule forced out and is a pure diagnostic.
    """
    out = gated_sdf.clone()
    suppressed = torch.zeros_like(gated_sdf, dtype=torch.bool)
    infinity = torch.finfo(gated_sdf.dtype).max
    for under, above in pairs:
        both = (out[:, under] < exit_thresh_m) & (out[:, above] < exit_thresh_m)
        drop_above = both & (normal_dot[:, under] >= normal_dot[:, above])
        drop_under = both & ~drop_above
        out[:, above] = out[:, above].masked_fill(drop_above, infinity)
        out[:, under] = out[:, under].masked_fill(drop_under, infinity)
        suppressed[:, above] |= drop_above
        suppressed[:, under] |= drop_under
    return out, suppressed


class FrameContacts(NamedTuple):
    """One clip's contact-frame readout — the boolean plus every signal behind it."""

    #: ``(T, F)`` bool — the label.
    contact: torch.Tensor
    #: ``(T, F)`` the guarded signed distance the spans were cut from (``+inf`` = guarded out).
    guarded_sdf: torch.Tensor
    #: ``(T, F)`` raw perpendicular signed distance (m).
    sdf: torch.Tensor
    #: ``(T, F)`` Euclidean distance to the nearest scene point (m).
    dmin: torch.Tensor
    #: ``(T, F)`` confidence of that nearest scene point.
    conf: torch.Tensor
    #: ``(T, F)`` ``dot(frame normal, −scene normal)`` — diagnostic, and the toe tiebreaker.
    normal_dot: torch.Tensor
    #: ``(T, F)`` bool — steps :func:`suppress_toe_pair` forced out (diagnostic).
    toe_suppressed: torch.Tensor


def detect_frame_contacts(
    frame_points: torch.Tensor,    # (T, F, 3) world contact-frame positions
    frame_normals: torch.Tensor,   # (T, F, 3) world outward unit normals
    drift: torch.Tensor,           # (T, F) net window drift (m) — the caller's measurement
    scene_pts: torch.Tensor,       # (M, 3) world scene points
    scene_normals: torch.Tensor,   # (M, 3) outward unit normals
    scene_conf: torch.Tensor,      # (M,) per-point confidence
    *,
    frame_names: Sequence[str],
    close_thresh_m: float,
    exit_thresh_m: float,
    min_frames: int,
    still_m: float,
    max_dist_m: float,
    conf_thresh: float,
    prune_margin_m: float,
) -> FrameContacts:
    """Detect contact on the 12 contact frames — the single detector for the whole pipeline.

    Contact is **near the scene AND still**, in four steps: surface signals
    (:func:`frame_scene_signals`), the distance/confidence guards
    (:func:`frame_guarded_sdf`), the stillness gate (:func:`drift_gate`) plus the toe-pair
    tiebreak (:func:`suppress_toe_pair`), then temporal hysteresis
    (:func:`hysteresis_spans`) — a frame enters a contact span below ``close_thresh_m``,
    holds it below the looser ``exit_thresh_m``, and spans shorter than ``min_frames`` are
    dropped. ``frame_names`` names the ``F`` columns and is only used to find the toe pairs
    by name.

    ``drift`` is supplied by the CALLER, never measured here, because *which*
    motion measurement is honest depends on the stage: the kinematic fit passes
    the drift of its starting pose (before any contact attraction acted), and the
    readout stage passes that same persisted ``drift_pre``. Re-measuring drift on
    an enforced body would be manufactured evidence. A ``NaN`` drift (an untracked
    step) fails the stillness gate, so it can never hold a span.
    """
    sdf, dmin, conf, normal_dot = frame_scene_signals(
        frame_points, frame_normals, scene_pts, scene_normals, scene_conf,
        prune_margin_m=prune_margin_m)
    guarded_sdf = frame_guarded_sdf(
        sdf, dmin, conf, max_dist_m=max_dist_m, conf_thresh=conf_thresh)
    gated, toe_suppressed = suppress_toe_pair(
        drift_gate(guarded_sdf, drift, still_m=still_m), normal_dot,
        toe_pair_slots(frame_names), exit_thresh_m=exit_thresh_m)
    contact = hysteresis_spans(gated, enter_below=close_thresh_m,
                               exit_below=exit_thresh_m, min_frames=min_frames)
    return FrameContacts(contact, guarded_sdf, sdf, dmin, conf, normal_dot, toe_suppressed)
