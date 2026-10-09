"""Scene-contact estimation for the optimized SMPL-X bodies.

**Contact lives on the contact frames** — named mesh points (toes, heels,
knees, palms, fingertips) baked onto the body, each carrying its own outward
surface normal (:mod:`bvr.body`). :func:`detect_frame_contacts` is the one
detector both the kinematic stage's frozen auto-labels and the standalone
``estimate_contacts`` readout call, so the two can never drift apart. The rest of
this module is the generic span/drift machinery it is built from
(:func:`joint_drift`, :func:`stillness_weight`, :func:`hysteresis_spans`,
:func:`drift_gated_contact`, :func:`merge_contact_spans` — all agnostic to what
the ``K`` axis means) plus the older per-joint pieces the frozen offline scorer
(``scripts/score_contacts.py``) still imports.

The older per-joint model, kept for those stages:
Given a posed (world-frame, scaled) SMPL-X mesh and the static scene point cloud,
mark which body vertices touch the scene with a point-cloud SDF, then lift the
per-vertex contacts to **per-joint** contacts by nearest-joint assignment (each
vertex belongs to the joint center it is spatially closest to in the rest pose).

The SDF is the same signed distance the kinematic stage's scene-penetration loss
uses (:func:`bvr.human_optim.losses.scene_penetration`): for each body vertex,
the nearest scene point ``p`` with its outward PCA normal ``n`` gives a signed
distance ``(v − p)·n`` — negative behind the surface (penetrating), positive in
front — using the camera-oriented normals from
:func:`bvr.geometry.normals.estimate_point_normals`. A vertex is *in contact*
when its signed distance falls below a small threshold (inside the scene, or just
outside it).

This is a one-shot analysis (no gradients, run once per clip), so the scene
nearest-neighbour is a plain per-frame pruned ``cdist`` rather than the
chunked-batched machinery the per-iteration losses need.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch


#: Raw-detection blip killer, deliberately a FIXED frame count at every fps: at 60 fps a
#: true contact often surfaces as dense 3–5-frame flicker, and a longer gate would delete
#: it before the stage's gap-welding can fuse it into a span (measured: 2nwG contact
#: recall .67 → .42 with a 0.1 s gate inside detection). The duration rule that IS
#: fps-aware (:func:`min_span_frames`) runs in ``estimate_contacts`` after the welding.
DETECT_MIN_FRAMES = 3

#: Contact-frame name stems (``<stem>_<side>``) that belong to a hand / to a foot.
HAND_FRAMES = ("palm", "palm_distal", "finger", "thumb")
FOOT_FRAMES = ("heel", "ball_inner", "ball_outer", "big_toe", "toe_above")


def min_span_frames(min_span_s: float, fps: float) -> int:
    """``[contacts].min_span_s`` (seconds) → the frame count the POST-WELD span rule uses.

    ``ceil`` so the duration gate never weakens with frame rate: 0.1 s is 3 frames
    at every shipped 24–30 fps rate (the old ``min_frames = 3``) and 5–6 at 50–60
    fps. Applied to the fused labels after gap-welding — never inside raw detection,
    where :data:`DETECT_MIN_FRAMES` rules (see its note).
    """
    return max(1, math.ceil(float(min_span_s) * float(fps)))


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

    A vertex contributes only if it passes the far / confidence guards
    (``dmin < max_dist_m`` and ``conf > conf_thresh``); guarded-out vertices are set
    to ``+inf`` so they never win the per-joint min, and a joint with no guarded
    vertex this frame is ``+inf``. This is the continuous per-joint signal the
    temporal hysteresis (:func:`hysteresis_spans`) thresholds: its ``enter``
    threshold is exactly "∃ guarded vertex with ``sdf < enter``" while a looser
    ``exit`` holds the span.
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
    keep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Temporal hysteresis + min-duration on a per-joint scalar series → ``(T, J)`` bool.

    ``series`` is a per-frame continuous signal where *smaller* means more in contact
    (here the guarded per-joint min signed distance). Per joint, independently:

      * **enter** a contact span at the first frame with ``series < enter_below``;
      * **stay** in it while ``series < exit_below`` — a looser threshold, so labels
        do not blink on and off at the boundary (``exit_below ≥ enter_below``);
      * close the span otherwise. A ``+inf`` / NaN frame (no guarded vertex) fails
        both tests and breaks the span.

    Then any span shorter than ``min_frames`` is dropped (1–2-frame flicker) — unless
    it contains a ``keep`` step (``(T, J)`` bool): a step that *presses into* the
    surface is evidence, not flicker, however brief (see :func:`drift_gate` — the
    penetration exception). Returns the boolean contact mask. Pure temporal
    post-process, no gradients.
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
        kc = keep.cpu().numpy() if keep is not None else None
        for j in range(J):
            col = oc[:, j]
            t = 0
            while t < T:
                if col[t]:
                    s = t
                    while t < T and col[t]:
                        t += 1
                    if t - s < min_frames and (kc is None or not kc[s:t, j].any()):
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
               still_m: float, still_override_m: float = -1.0) -> torch.Tensor:
    """``guarded_min_sdf (T, K)`` with every step that failed the stillness gate at ``+inf``.

    A step counts as still while :func:`stillness_weight` is above zero (drift below
    ``2·still_m``). Forcing the rest to the sentinel — rather than masking — is what lets
    :func:`hysteresis_spans` run unchanged: a moving step simply looks infinitely far from
    the wall, so it can neither start nor hold a span.

    ``still_override_m`` makes strong closeness outrank stillness: a step whose guarded
    signed distance is below it keeps its signal even while moving fast. A sprinting foot
    plant or a cartwheel hand is a real contact that no drift window can ever call still —
    the window straddles the swing — but it *presses into the floor*, and that penetration
    is evidence stillness cannot veto. The default (−1 m, below any reachable signed
    distance) disables the override; the drift must still be a real measurement (an
    untracked NaN step never passes).
    """
    weight = stillness_weight(drift, still_m=still_m)
    pressed = (guarded_min_sdf < still_override_m) & torch.isfinite(drift)
    infinity = torch.finfo(guarded_min_sdf.dtype).max
    return torch.where((weight > 0.0) | pressed, guarded_min_sdf,
                       torch.full_like(guarded_min_sdf, infinity))


def load_gate(guarded_min_sdf: torch.Tensor, drift: torch.Tensor,
              load_bw: torch.Tensor, *, still_m: float, still_override_m: float,
              min_load_bw: float,
              mute: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Penetration outranks stillness — but load outranks penetration.

    The stillness gate has one exception (``still_override_m``, :func:`drift_gate`): a
    step pressing within a centimetre of the surface counts however fast it moves. A
    sprint plant or a cartwheel hand is a real contact no drift window can call still.
    The trouble is that a hand sweeping ALONG the wall presses just as close — the
    exception cannot tell a plant from a sweep, and on the annotated corpus it alone
    costs the geometric arm four points of precision.

    The force solve can tell them apart: a plant carries weight, a sweep carries none.
    So wherever the solve was heard, a pressing step keeps its exception only if the
    solve routes at least ``min_load_bw`` of body weight through it. Where the solve
    was **muted** — ``mute (T,)`` from :func:`bvr.human_optim.forces.solve_mute`, a
    rope or a belay carrying the body, a trajectory too broken to explain — physics
    said nothing and the exception stands as it always did: missing evidence reads as
    *unknown*, never as a negative. Still steps are never touched; stillness needs no
    second witness. (Making load REPLACE stillness outright was measured 2026-08-30
    and refuted: −56% false positives but −6 points of recall — a heel sharing a hold
    with its own forefoot carries no solvable load of its own.)

    Returns ``(gated, pressed)``: the gated signal for the hysteresis, and the steps
    whose surviving press exempts their span from the duration rules downstream.
    ``min_load_bw <= 0`` keeps every press — the plain ``still_override_m`` rule.
    """
    still_only = drift_gate(guarded_min_sdf, drift, still_m=still_m,
                            still_override_m=-1.0)
    if still_override_m <= 0.0:
        return still_only, torch.zeros_like(guarded_min_sdf, dtype=torch.bool)
    pressed = (guarded_min_sdf < still_override_m) & torch.isfinite(drift)
    if min_load_bw > 0.0:
        confirmed = load_bw >= min_load_bw
        if mute is not None:
            confirmed = confirmed | mute[:, None]
        pressed = pressed & confirmed
    return torch.where(pressed, guarded_min_sdf, still_only), pressed


def drift_gated_contact(
    guarded_min_sdf: torch.Tensor,   # (T, J) guarded per-joint min signed distance (m)
    drift: torch.Tensor,             # (T, J) per-joint net window drift (m)
    *,
    close_thresh_m: float,
    exit_thresh_m: float,
    min_frames: int,
    still_m: float,
    still_override_m: float = -1.0,
) -> torch.Tensor:
    """The contact readout ``(T, J)`` bool: drift-gate, then temporal hysteresis.

    A joint-frame can only be in contact when it is both **near the wall** (its
    guarded min signed distance below the hysteresis thresholds) and **still** (net
    window drift under ``2·still_m``, i.e. :func:`stillness_weight` ``> 0``). Frames
    that fail the stillness gate have their distance forced to ``+inf`` so
    :func:`hysteresis_spans` breaks the span there — exactly how the old hard speed
    veto plugged in, but with the windowed-drift signal that actually separates a
    grip from a slow pass. Shared by ``estimate_contacts`` and the offline scorer so
    both score the identical readout. ``still_override_m``: see :func:`drift_gate` —
    a step pressing below it also exempts its span from the ``min_frames`` rule
    (the same evidence outranks stillness *and* duration).
    """
    gated = drift_gate(guarded_min_sdf, drift, still_m=still_m,
                       still_override_m=still_override_m)
    return hysteresis_spans(gated, enter_below=close_thresh_m,
                            exit_below=exit_thresh_m, min_frames=min_frames,
                            keep=gated < still_override_m)


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
# (bvr/body.py). TWO things must agree before it counts as touching:
#   surface  — a confident scene point is close, and the frame sits at/just
#              inside it (signed distance below the hysteresis thresholds);
#   stillness — the frame barely drifted over a short window (with one exception,
#              the penetration override still_override_m: a step pressing into the
#              surface counts however fast it moves).
# The readout stage, which has solved the forces by then, holds the EXCEPTION to a
# second witness: a pressing-but-moving step keeps it only where the solve routes
# real load through the frame, or was muted (see load_gate). Still steps never need
# the solve's blessing.
# The frame's own normal is deliberately NOT a gate. Both inputs it would need are
# too noisy to veto on: the scene normals come from a reconstructed point cloud, and
# the frame's normal comes from the fitted pose. Gating on "does the skin face the
# wall?" threw away real contacts — a fingertip 3 cm from a hold whose baked pad
# normal pointed away because the hand pose was wrong. The normal survives in exactly
# ONE role, telling the two toe sides apart: see :func:`suppress_toe_sides`.


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

    Only a few dozen points per time step, so the nearest neighbour is a plain per-frame
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


#: The foot's contact frames, underside vs top. A foot stands on its underside frames
#: (heel included) or presses the top of its toe against a hold (toe hook) — never both
#: at once. The two sides sit centimetres apart through the foot, so no distance /
#: confidence / stillness gate can ever separate them — only their normals can. Names
#: cover both frame sets: the legacy 12-frame set's underside is ``toe_under`` + ``heel``,
#: the full-body set's (``better_contacts.json``) is ``ball_inner`` / ``ball_outer`` /
#: ``big_toe`` + ``heel``; :func:`toe_side_slots` keeps only the names that exist.
TOE_SIDES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("toe_under_left", "ball_inner_left", "ball_outer_left", "big_toe_left", "heel_left"),
     ("toe_above_left",)),
    (("toe_under_right", "ball_inner_right", "ball_outer_right", "big_toe_right",
      "heel_right"),
     ("toe_above_right",)),
)


def toe_side_slots(
    frame_names: Sequence[str],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Resolve :data:`TOE_SIDES` to per-foot ``(sole, top)`` column-index tuples.

    Always by NAME, never by arithmetic on positions: the stored frame order is not the
    obvious one (slots 10/11 are finger_RIGHT then finger_LEFT). Absent names are
    dropped; a foot whose sole or top side comes out empty is skipped entirely, so a
    caller with a partial frame set still works.
    """
    slot = {str(name): i for i, name in enumerate(frame_names)}
    feet = []
    for sole_names, top_names in TOE_SIDES:
        sole = tuple(slot[n] for n in sole_names if n in slot)
        top = tuple(slot[n] for n in top_names if n in slot)
        if sole and top:
            feet.append((sole, top))
    return tuple(feet)


def suppress_toe_sides(
    gated_sdf: torch.Tensor,    # (T, F) drift-gated signed distance (m); +inf = cannot contact
    normal_dot: torch.Tensor,   # (T, F) dot(frame normal, −scene normal)
    sides: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    exit_thresh_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep at most ONE side of each foot's toe — the only place a normal decides anything.

    The underside frames (a foot standing *on* a hold) and ``toe_above`` (a toe hooked *under*
    one) face opposite ways through the foot, so every other gate accepts or rejects them
    together and both sides fire at once. We assume at most one side really touches at any
    step — both-at-once is physically possible but rare, and deliberately ignored — and
    let the normals break the tie.

    **Candidacy**: at step ``t`` a frame is a candidate when its gated signal is below
    ``exit_thresh_m``, the loosest level at which :func:`hysteresis_spans` could still hold
    it in a span. Where both sides of a foot have candidates, each side is scored by its
    best-facing candidate (the largest ``normal_dot``); the losing side's candidate frames
    are forced to the ``+inf`` sentinel — before hysteresis, so they can neither enter nor
    hold a span at that step. Exact ties keep the underside, so the result never depends
    on evaluation order. Steps where only one side has candidates pass through untouched.

    Returns ``(gated_sdf, suppressed)``, both ``(T, F)`` — ``suppressed`` marks the steps
    this rule forced out and is a pure diagnostic.
    """
    out = gated_sdf.clone()
    suppressed = torch.zeros_like(gated_sdf, dtype=torch.bool)
    infinity = torch.finfo(gated_sdf.dtype).max
    lowest = torch.finfo(normal_dot.dtype).min
    for sole, top in sides:
        sole_cand = out[:, sole] < exit_thresh_m                     # (T, S)
        top_cand = out[:, top] < exit_thresh_m                       # (T, K)
        both = sole_cand.any(-1) & top_cand.any(-1)                  # (T,)
        sole_score = normal_dot[:, sole].masked_fill(~sole_cand, lowest).amax(-1)
        top_score = normal_dot[:, top].masked_fill(~top_cand, lowest).amax(-1)
        drop_top = both & (sole_score >= top_score)
        drop_sole = both & ~drop_top
        for columns, cand, drop in ((top, top_cand, drop_top), (sole, sole_cand, drop_sole)):
            forced = cand & drop[:, None]                            # (T, len(columns))
            cols = list(columns)
            out[:, cols] = out[:, cols].masked_fill(forced, infinity)
            suppressed[:, cols] |= forced
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
    #: ``(T, F)`` bool — steps :func:`suppress_toe_sides` forced out (diagnostic).
    toe_suppressed: torch.Tensor
    #: ``(T, F)`` bool — the surviving penetration exception (see :func:`load_gate`);
    #: the readout exempts these steps' spans from its duration rules.
    pressed: torch.Tensor


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
    still_override_m: float = -1.0,
    max_dist_m: float,
    conf_thresh: float,
    prune_margin_m: float,
    load_bw: torch.Tensor | None = None,
    min_load_bw: float = 0.0,
    load_mute: torch.Tensor | None = None,
) -> FrameContacts:
    """Detect contact on the contact frames — the single detector for the whole pipeline.

    Contact is **near the scene AND still**, in four steps: surface signals
    (:func:`frame_scene_signals`), the distance/confidence guards
    (:func:`frame_guarded_sdf`), the stillness gate (:func:`drift_gate`) plus the toe-side
    tiebreak (:func:`suppress_toe_sides`), then temporal hysteresis
    (:func:`hysteresis_spans`) — a frame enters a contact span below ``close_thresh_m``,
    holds it below the looser ``exit_thresh_m``, and spans shorter than ``min_frames`` are
    dropped. ``frame_names`` names the ``F`` columns and is only used to find the toe sides
    by name. ``still_override_m`` lets strong closeness outrank the stillness gate — a
    fast foot plant pressing into the floor is a contact no drift window can call still
    (see :func:`drift_gate`) — and the ``min_frames`` rule: a span pressing below it
    survives however brief (a one-frame bounce is 1–2 frames at 25 fps, and penetration
    is not flicker). The default disables both.

    ``load_bw (T, F)`` is the solved contact load in body weights. When it is given with
    ``min_load_bw > 0``, the penetration exception needs the solve's blessing
    (:func:`load_gate`): a pressing-but-moving step counts only where the solve routes
    at least ``min_load_bw`` through it, or was muted (``load_mute``). Still steps are
    untouched. Only the readout stage can supply it — the kinematic stage detects its
    auto-labels before any force is solved — so the default leaves the two-gate
    behaviour untouched.

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
    if load_bw is not None:
        gated, pressed = load_gate(guarded_sdf, drift, load_bw, still_m=still_m,
                                   still_override_m=still_override_m,
                                   min_load_bw=min_load_bw, mute=load_mute)
    else:
        gated = drift_gate(guarded_sdf, drift, still_m=still_m,
                           still_override_m=still_override_m)
        pressed = gated < still_override_m
    gated, toe_suppressed = suppress_toe_sides(
        gated, normal_dot, toe_side_slots(frame_names), exit_thresh_m=exit_thresh_m)
    contact = hysteresis_spans(gated, enter_below=close_thresh_m,
                               exit_below=exit_thresh_m, min_frames=min_frames,
                               keep=pressed)
    return FrameContacts(contact, guarded_sdf, sdf, dmin, conf, normal_dot,
                         toe_suppressed, pressed)
