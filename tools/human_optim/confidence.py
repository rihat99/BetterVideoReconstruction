"""Continuous per-joint confidence for the final contact/no-contact label.

The contact stage has already produced a binary label.  This module answers one
smaller question: *how much should a learner trust that label in this frame?*
The binary label itself is never changed; the result is a heuristic training
weight in ``[0, 1]``, not a calibrated probability.

Confidence is ``visibility * evidence``.  Visibility (in frame, not hidden by a
body part along the camera ray) multiplies every label, so an occluded or
out-of-frame joint is zero.  The *evidence* is where v6 differs from earlier
versions: it is **source-conditional**, because how a label was decided caps how
much any witness can trust it.

Why condition on the source.  The completed contact label carries a
``contact_source`` bitmask (1=detector, 2=gate, 4=force).  Gate labels are
*created* from wall distance and stillness, so distance/stillness "witnesses"
would endorse them tautologically and precision plateaus at the known
near-and-still false-positive floor.  Conditioning on the source breaks that
circularity: each source gets a cap that only a *semi-independent* witness may
lift.

Contact evidence, by source (only the six limb group joints — wrists, feet,
ankles — carry gate/force bits; every other joint is detector-only):

``detector`` (bit 1)
    Surface evidence — the guarded signed distance in a plausible contact band,
    reduced under implausibly deep penetration.  Top tier: the corpus detector
    is precise (held-out precision 0.983).

``gate-only`` (bit 2, no detector)
    Surface evidence hard-capped below the top tier — this is where the
    near-and-still hovering false positives live.  Solved contact force is
    semi-independent of the distance/stillness that created the gate, so force
    corroboration continuously lifts the cap toward a mid-high value.

``force-only`` (bit 4, no detector, no gate)
    The misregistration rescues: honest contacts a limb 10-30 cm off the wall
    made, where surface evidence is ~0.  Their evidence is the solved force
    share ``|f| / body-weight`` (already suppressed on a fast-moving limb by the
    solve's work penalty), capped mid.

No-contact evidence is depth-trusted ``motion`` (net joint speed over the readout
window, scaled by how well the local body fit agrees with the human depth cloud).
Two honesty fixes over earlier versions:

* Missing depth data is an *absent* witness, not agreement.  Where too few cloud
  points matched the joint, motion could otherwise ride the placeholder trust of
  one; v6 caps such no-contact confidence below the top tier.
* The static support solve allocating ~no force to a limb is a semi-independent
  no-contact witness; it lifts the missing-depth cap for that limb.

Guarded distance only adjusts how much speed is needed to trust motion; it is
never no-contact evidence by itself.  A stationary no-contact joint stays
untrusted even when the reconstruction places it far from the wall, because a
whole-body floater can have a good local depth fit and a large apparent wall
distance while its no-contact label is still wrong.
"""
from __future__ import annotations

import numpy as np
import torch


# Artifact version of the confidence written to contacts_<tag>.npz. Bump both
# together. Light-weight consumers keep literal copies pinned to these values:
# scripts/corpus.py, scripts/export_contact_dataset.py, tools/viewer/loading.py.
# 8: labels the predicted-contact fusion changed carry ITS confidence (the fused
#    two-witness agreement, tools/human_optim/fusion.py), not the terms below.
CONTACT_CONFIDENCE_SCHEMA = 8
CONTACT_CONFIDENCE_METHOD = "source-conditional-v1"
# The contact stage's scene-free mode (use_scene = false): confidence is the model's
# contact probability × stillness (no scene witnesses). A legitimate current method,
# not a stale artifact — consumers accept either.
SCENE_FREE_CONFIDENCE_METHOD = "scene_free_pred_still_v1"

CONTACT_CONFIDENCE_DEFAULTS: dict[str, float] = {
    # Plausible signed-distance band for a contact.  Soft sigmoids avoid hard
    # thresholds while making deep burial lose, rather than gain, confidence.
    "surface_inner_m": -0.02,
    "surface_outer_m": 0.05,
    "surface_soft_m": 0.01,
    "penetration_floor": 0.40,
    # Motion scale and response power interpolate smoothly from the original far
    # curve to a cubic near-surface curve using the contact band's sigmoid.
    "motion_scale_mps": 0.10,
    "motion_near_scale_mps": 0.15,
    "motion_near_power": 3.0,
    # Visibility.  A body-occlusion gap (metres of surface in front of the joint
    # along its camera ray) past occ_margin_m fades the joint out over occ_soft_m.
    "image_margin_px": 10.0,
    "occ_margin_m": 0.05,
    "occ_soft_m": 0.01,
    # Source-conditional caps (v6).  Each caps how far a label of a given source
    # can be trusted; only a semi-independent witness lifts it.
    "gate_only_cap": 0.50,      # gate-added contact, no force corroboration (the FP floor)
    "gate_force_cap": 0.85,     # gate contact fully corroborated by solved force
    "force_only_cap": 0.70,     # force-added contact (misregistration rescue, positionally uncertain)
    "force_share_scale": 0.25,  # |f|/body-weight scale of the force ramp (evidence and corroboration)
    # No-contact missing-depth caps default OFF (1.0 = no cap): the premise that a
    # missing depth witness marks an unreliable no-contact was refuted on the ground
    # truth (see docs/contact_confidence.md). The mechanism stays; lower a knob to study it.
    "nodata_cap": 1.00,         # no-contact with an absent depth witness
    "nodata_force_cap": 1.00,   # no-contact, absent depth, but the solve allocates ~no force here
}


def confidence_config(overrides: dict | None = None) -> dict[str, float]:
    """Return validated confidence knobs with ``overrides`` applied."""
    unknown = set(overrides or {}) - set(CONTACT_CONFIDENCE_DEFAULTS)
    if unknown:
        raise ValueError(
            "unknown contact_confidence option(s): " + ", ".join(sorted(unknown)))
    cfg = {**CONTACT_CONFIDENCE_DEFAULTS, **(overrides or {})}
    numeric = {k: float(v) for k, v in cfg.items()}
    for key, value in numeric.items():
        if not np.isfinite(value):
            raise ValueError(f"contact_confidence.{key} must be finite")
    if numeric["surface_inner_m"] >= numeric["surface_outer_m"]:
        raise ValueError("contact_confidence.surface_inner_m must be < surface_outer_m")
    for key in ("surface_soft_m", "motion_scale_mps", "motion_near_scale_mps",
                "motion_near_power",
                "image_margin_px", "occ_margin_m", "occ_soft_m", "force_share_scale"):
        if numeric[key] <= 0.0:
            raise ValueError(f"contact_confidence.{key} must be > 0")
    for key in ("penetration_floor", "gate_only_cap", "gate_force_cap",
                "force_only_cap", "nodata_cap", "nodata_force_cap"):
        if not 0.0 <= numeric[key] <= 1.0:
            raise ValueError(f"contact_confidence.{key} must be in [0, 1]")
    if numeric["motion_near_scale_mps"] < numeric["motion_scale_mps"]:
        raise ValueError(
            "contact_confidence.motion_near_scale_mps must be >= motion_scale_mps")
    if numeric["motion_near_power"] < 1.0:
        raise ValueError("contact_confidence.motion_near_power must be >= 1")
    if numeric["gate_force_cap"] < numeric["gate_only_cap"]:
        raise ValueError("contact_confidence.gate_force_cap must be >= gate_only_cap")
    if numeric["nodata_force_cap"] < numeric["nodata_cap"]:
        raise ValueError("contact_confidence.nodata_force_cap must be >= nodata_cap")
    return numeric


# Joint groups used only to exclude a joint's own skin from body occlusion. A ray
# from the camera through a joint centre must pass through that joint's own surface,
# so any triangle owned by a group member is not counted as an occluder. Every joint
# not listed here receives its own unique group; unlike a shared -1 sentinel, unrelated
# ungrouped joints can occlude one another.
_FINGERS = ("index", "middle", "pinky", "ring", "thumb")
# The torso is one deep block of skin owned by many neighbouring joints, so each
# axial joint centre sits behind ~13-20 cm of its neighbours' surface. Sharing one
# group across the whole torso means only a limb (or another person) crossing in
# front can occlude an axial joint — its own belly/back/chest skin never does.
_AXIAL_OCCLUSION_GROUP = (
    "pelvis", "spine1", "spine2", "spine3", "left_hip", "right_hip",
    "left_collar", "right_collar", "neck", "left_shoulder", "right_shoulder",
)
_LOCAL_OCCLUSION_GROUPS: tuple[tuple[str, ...], ...] = (
    tuple(["left_wrist"] + [f"left_{finger}{k}" for finger in _FINGERS for k in (1, 2, 3)]),
    tuple(["right_wrist"] + [f"right_{finger}{k}" for finger in _FINGERS for k in (1, 2, 3)]),
    ("left_ankle", "left_foot"),
    ("right_ankle", "right_foot"),
    _AXIAL_OCCLUSION_GROUP,
)


def occlusion_group_ids(joint_names: list[str]) -> np.ndarray:
    """Return a nonnegative local-surface group ID for every model joint.

    Each complete hand and each ankle/foot pair shares a group so its own local
    skin does not hide its joint centre. A hand is one coarse contact state, so
    fingers hiding one another must not erase the whole-hand free label. The whole
    torso shares one axial group so a deep axial joint (pelvis, spines, hips,
    collars, neck, shoulders) never self-occludes on its neighbours' torso skin —
    only a limb or another person in front hides it. The head and every other joint
    stay unique. No sentinel value is shared by unrelated joints.
    """
    names = {name: i for i, name in enumerate(joint_names)}
    out = np.arange(len(joint_names), dtype=np.int32)
    next_group = len(joint_names)
    for members in _LOCAL_OCCLUSION_GROUPS:
        present = [names[name] for name in members if name in names]
        if present:
            out[present] = next_group
            next_group += 1
    return out


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    """Cubic smoothstep after clipping to ``[0,1]``."""
    x = torch.clamp(x, 0.0, 1.0)
    return x.square() * (3.0 - 2.0 * x)


def source_conditional_evidence(
    labels: torch.Tensor,
    surface: torch.Tensor,
    motion: torch.Tensor,
    contact_source: torch.Tensor,
    force_share: torch.Tensor,
    depth_present: torch.Tensor,
    c: dict,
) -> torch.Tensor:
    """Source-conditional label evidence in ``[0, 1]`` for one person's ``(N,J)`` labels.

    The single place the v6 confidence differs from a plain
    ``where(label, surface, motion)``.  Kept as a pure tensor function so the
    stage and any offline audit combine identical algebra.

    :param labels: ``(N,J)`` bool — the final completed contact label.
    :param surface: ``(N,J)`` raw contact (surface) evidence in ``[0, 1]``.
    :param motion: ``(N,J)`` raw depth-trusted no-contact (motion) evidence in ``[0, 1]``.
    :param contact_source: ``(N,J)`` int bitmask 1=detector, 2=gate, 4=force.
    :param force_share: ``(N,J)`` solved ``|f| / body-weight`` at the six limb group
        joints, ``NaN`` at every joint with no force witness (all others).
    :param depth_present: ``(N,J)`` bool — a local human-depth witness matched the joint.
    :param c: validated confidence knobs (:func:`confidence_config`).
    """
    src = contact_source.to(torch.int64)
    has_det = (src & 1) != 0
    has_gate = (src & 2) != 0
    has_force = (src & 4) != 0

    witness = torch.isfinite(force_share)
    share = torch.nan_to_num(force_share, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    scale = c["force_share_scale"]
    force_ramp = torch.where(witness, -torch.expm1(-share / scale), torch.zeros_like(share))
    force_absent = torch.where(witness, torch.exp(-share / scale), torch.zeros_like(share))

    # Contact.  Detector (and every detector-only non-limb joint) keeps the top
    # surface tier.  A gate-only add is capped, its cap lifted continuously toward
    # gate_force_cap by force corroboration.  A force-only add reads the force
    # share as its evidence (surface may be ~0 for a misregistered limb), capped.
    gate_cap = c["gate_only_cap"] + (c["gate_force_cap"] - c["gate_only_cap"]) * force_ramp
    force_only_ev = torch.minimum(
        torch.full_like(surface, c["force_only_cap"]),
        torch.maximum(surface, c["force_only_cap"] * force_ramp))
    contact_ev = surface.clone()
    contact_ev = torch.where(has_gate & ~has_det, torch.minimum(surface, gate_cap), contact_ev)
    contact_ev = torch.where(has_force & ~has_gate & ~has_det, force_only_ev, contact_ev)

    # No-contact.  Depth-trusted motion, but an absent depth witness caps it; the
    # solve allocating ~no force to a limb lifts that cap (a semi-independent
    # no-contact witness).  Non-limb joints have no force witness, so their cap
    # stays at nodata_cap.
    nodata_cap = c["nodata_cap"] + (c["nodata_force_cap"] - c["nodata_cap"]) * force_absent
    nc_ev = torch.where(depth_present, motion, torch.minimum(motion, nodata_cap))

    return torch.where(labels, contact_ev, nc_ev)


def joint_label_confidence(
    labels: torch.Tensor,
    guarded_sdf_m: torch.Tensor,
    drift_m: torch.Tensor,
    depth_trust: torch.Tensor,
    depth_points: torch.Tensor,
    contact_source: torch.Tensor,
    force_share: torch.Tensor,
    projected_px: torch.Tensor,
    camera_depth_m: torch.Tensor,
    occlusion_gap_m: torch.Tensor,
    *,
    height: int,
    width: int,
    window_s: float,
    min_depth_points: int,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Compute continuous confidence terms for one person's ``(N,J)`` labels.

    Returns the image, 3-D occlusion, and combined visibility terms; the raw
    occlusion gap; the raw surface and motion evidence terms; and the final
    source-conditional confidence (:func:`source_conditional_evidence`).
    ``occlusion_gap_m`` is the in-front body-occlusion gap in metres
    (:func:`tools.human_optim.contacts.joint_occlusion_gap`); ``NaN`` = no surface in
    front = fully visible.  ``depth_points`` (matched cloud count) and
    ``min_depth_points`` decide whether a depth witness is present;
    ``contact_source`` is the provenance bitmask and ``force_share`` the solved
    ``|f|/body-weight`` (``NaN`` off the six limb group joints).
    """
    c = confidence_config(cfg)
    labels = labels.bool()
    shape = labels.shape
    if labels.ndim != 2:
        raise ValueError(f"expected per-joint labels (N,J), got {tuple(shape)}")
    for name, value in (
        ("guarded_sdf_m", guarded_sdf_m), ("drift_m", drift_m),
        ("depth_trust", depth_trust), ("depth_points", depth_points),
        ("contact_source", contact_source), ("force_share", force_share),
        ("camera_depth_m", camera_depth_m),
        ("occlusion_gap_m", occlusion_gap_m),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}")
    if projected_px.shape != (*shape, 2):
        raise ValueError(
            f"projected_px must have shape {(*shape, 2)}, got {tuple(projected_px.shape)}")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not np.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("window_s must be finite and positive")
    if min_depth_points <= 0:
        raise ValueError("min_depth_points must be positive")

    # Near-surface contact support.  Deep penetration lowers confidence to a
    # floor instead of either proving contact or erasing it entirely: signed SDF
    # is noisy when the reconstructed surface normal is wrong.  Nonfinite/no
    # guarded scene evidence is zero.
    finite_guarded = torch.isfinite(guarded_sdf_m)
    d = torch.nan_to_num(guarded_sdf_m, nan=0.0, posinf=0.0, neginf=0.0)
    near = torch.sigmoid((c["surface_outer_m"] - d) / c["surface_soft_m"])
    not_deep = torch.sigmoid((d - c["surface_inner_m"]) / c["surface_soft_m"])
    penetration_reliability = c["penetration_floor"] + (1.0 - c["penetration_floor"]) * not_deep
    surface = near * penetration_reliability
    surface = torch.where(finite_guarded, surface, torch.zeros_like(surface))

    # Guarded distance adjusts the speed scale but never contributes evidence:
    # zero speed therefore stays zero at every distance. Near the contact band,
    # the inflated scale suppresses slow reconstruction drift while the cubic
    # response preserves a clearly moving approach/leave witness.
    speed = torch.nan_to_num(drift_m, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    speed = speed / float(window_s)
    proximity = torch.sigmoid((c["surface_outer_m"] - d) / c["surface_soft_m"])
    proximity = torch.where(finite_guarded, proximity, torch.zeros_like(proximity))
    motion_scale = c["motion_scale_mps"] + proximity * (
        c["motion_near_scale_mps"] - c["motion_scale_mps"])
    motion_power = 1.0 + proximity * (c["motion_near_power"] - 1.0)
    motion = -torch.expm1(-torch.pow(speed / motion_scale, motion_power))
    trust = torch.nan_to_num(depth_trust, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    motion = motion * trust

    # Continuous image-border visibility: zero at/outside the image boundary and
    # one after image_margin_px inside it.
    x, y = projected_px[..., 0], projected_px[..., 1]
    edge = torch.stack((x, y, (width - 1.0) - x, (height - 1.0) - y), dim=-1).amin(dim=-1)
    edge = torch.nan_to_num(edge, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
    in_frame = _smoothstep01(edge / c["image_margin_px"])
    in_front = torch.isfinite(camera_depth_m) & (camera_depth_m > 0.0)
    image_visibility = in_frame * in_front.to(in_frame.dtype)

    # 3-D body occlusion from the exact camera→joint ray gap. A gap past occ_margin_m
    # of surface in front fades the joint out over occ_soft_m; a NaN gap (no surface in
    # front) is fully visible. Applied to both classes and after their evidence, so
    # motion can no longer bypass occlusion.
    no_occluder = torch.isnan(occlusion_gap_m)
    gap = torch.nan_to_num(occlusion_gap_m, nan=0.0, posinf=0.0, neginf=0.0)
    visible_3d = torch.sigmoid((c["occ_margin_m"] - gap) / c["occ_soft_m"])
    visible_3d = torch.where(no_occluder, torch.ones_like(visible_3d), visible_3d)
    visibility = image_visibility * visible_3d

    # Source-conditional evidence: how the label was decided caps how much any
    # witness may trust it (the v6 change). Surface/motion above stay the raw,
    # uncapped base terms; the caps and force corroboration live in the combiner.
    depth_present = depth_points >= float(min_depth_points)
    label_evidence = source_conditional_evidence(
        labels, surface, motion, contact_source, force_share, depth_present, c)
    label_confidence = visibility * label_evidence

    def as_numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float32)

    return {
        "image_visibility": as_numpy(image_visibility),
        "occlusion_visibility": as_numpy(visible_3d),
        "occlusion_gap": as_numpy(occlusion_gap_m),
        "visibility": as_numpy(visibility),
        "surface": as_numpy(surface),
        "motion": as_numpy(motion),
        "label_confidence": as_numpy(label_confidence),
    }
