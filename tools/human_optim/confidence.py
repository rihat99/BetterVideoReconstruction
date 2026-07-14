"""Continuous per-joint confidence for the final contact/no-contact label.

The contact stage has already produced a binary label.  This module answers one
smaller question: *how much should a learner trust that label in this frame?*

Three continuous terms are enough:

``visibility``
    The joint is in frame and not hidden by another body part in 3-D.  Visibility
    multiplies every label.  A fully occluded or out-of-frame joint therefore has
    zero confidence; neither motion nor scene distance can bypass it.

``surface``
    The guarded signed distance lies in a plausible contact band.  The score
    falls outside the wall and is reduced under implausibly deep penetration.
    Burial is not evidence that a contact is certain, but it retains a small
    floor because imperfect scene normals can put a real contact inside the wall.

``motion``
    Net joint speed over the contact readout's short window.  It rises from zero
    immediately, rather than having the old 0.15 m/s dead zone.  This is the
    strong witness for a no-contact limb approaching or leaving the wall.

For a contact label the evidence is ``surface``.  For a no-contact label it is
only ``motion``. Final confidence is simply ``visibility * label_evidence``.
There is no distance/floating-body detector and no binary-label persistence
term: a stationary no-contact joint is deliberately untrusted even when the
reconstruction places it far from the wall.

The result is a heuristic training weight in ``[0, 1]``, not a calibrated
probability.  The binary label itself remains unchanged.
"""
from __future__ import annotations

import numpy as np
import torch


CONTACT_CONFIDENCE_DEFAULTS: dict[str, float] = {
    # Plausible signed-distance band for a contact.  Soft sigmoids avoid hard
    # thresholds while making deep burial lose, rather than gain, confidence.
    "surface_inner_m": -0.02,
    "surface_outer_m": 0.05,
    "surface_soft_m": 0.01,
    "penetration_floor": 0.40,
    # p_motion = 1-exp(-speed / motion_scale_mps).  At 0.1 m/s it is 0.63,
    # versus the old formula which stayed exactly zero below 0.15 m/s.
    "motion_scale_mps": 0.10,
    # Visibility.  The cubic smoothstep reaches exactly zero at occ_full body
    # vertices, so a clearly body-occluded joint is never exported as trustworthy.
    "image_margin_px": 10.0,
    "occ_perp_m": 0.02,
    "occ_front_m": 0.005,
    "occ_subsample": 2500.0,
    "occ_full": 5.0,
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
    for key in ("surface_soft_m", "motion_scale_mps",
                "image_margin_px", "occ_perp_m", "occ_front_m", "occ_subsample",
                "occ_full"):
        if numeric[key] <= 0.0:
            raise ValueError(f"contact_confidence.{key} must be > 0")
    for key in ("penetration_floor",):
        if not 0.0 <= numeric[key] <= 1.0:
            raise ValueError(f"contact_confidence.{key} must be in [0, 1]")
    if not numeric["occ_subsample"].is_integer():
        raise ValueError("contact_confidence.occ_subsample must be an integer")
    return numeric


# Joint groups used only to exclude a limb's own skin from body occlusion. Every
# joint not listed here receives its own unique group; unlike the previous -1
# sentinel, unrelated ungrouped joints can occlude one another.
_FINGERS = ("index", "middle", "pinky", "ring", "thumb")
_LOCAL_OCCLUSION_GROUPS: tuple[tuple[str, ...], ...] = (
    tuple(["left_wrist"] + [f"left_{finger}{k}" for finger in _FINGERS for k in (1, 2, 3)]),
    tuple(["right_wrist"] + [f"right_{finger}{k}" for finger in _FINGERS for k in (1, 2, 3)]),
    ("left_ankle", "left_foot"),
    ("right_ankle", "right_foot"),
)


def occlusion_group_ids(joint_names: list[str]) -> np.ndarray:
    """Return a nonnegative local-surface group ID for every model joint.

    Each complete hand and each ankle/foot pair shares a group so its own local
    skin does not hide its joint centre. A hand is one coarse contact state, so
    fingers hiding one another must not erase the whole-hand free label. All
    other joints are unique. No sentinel value is shared by unrelated joints.
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


def joint_label_confidence(
    labels: torch.Tensor,
    guarded_sdf_m: torch.Tensor,
    drift_m: torch.Tensor,
    projected_px: torch.Tensor,
    camera_depth_m: torch.Tensor,
    occluding_vertices: torch.Tensor,
    *,
    height: int,
    width: int,
    window_s: float,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Compute continuous confidence terms for one person's ``(N,J)`` labels.

    Returns the image, 3-D occlusion, and combined visibility terms; the raw
    occluder count; the surface and motion evidence terms; and final confidence.
    """
    c = confidence_config(cfg)
    labels = labels.bool()
    shape = labels.shape
    if labels.ndim != 2:
        raise ValueError(f"expected per-joint labels (N,J), got {tuple(shape)}")
    for name, value in (
        ("guarded_sdf_m", guarded_sdf_m), ("drift_m", drift_m),
        ("camera_depth_m", camera_depth_m),
        ("occluding_vertices", occluding_vertices),
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

    # Motion starts contributing immediately and smoothly approaches one.
    speed = torch.nan_to_num(drift_m, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    speed = speed / float(window_s)
    motion = -torch.expm1(-speed / c["motion_scale_mps"])

    # Continuous image-border visibility: zero at/outside the image boundary and
    # one after image_margin_px inside it.
    x, y = projected_px[..., 0], projected_px[..., 1]
    edge = torch.stack((x, y, (width - 1.0) - x, (height - 1.0) - y), dim=-1).amin(dim=-1)
    edge = torch.nan_to_num(edge, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
    in_frame = _smoothstep01(edge / c["image_margin_px"])
    in_front = torch.isfinite(camera_depth_m) & (camera_depth_m > 0.0)
    image_visibility = in_frame * in_front.to(in_frame.dtype)

    # 3-D body occlusion reaches exactly zero at occ_full.  It is applied to both
    # classes and after their evidence, so motion can no longer bypass occlusion.
    n_occ = torch.nan_to_num(occluding_vertices, nan=c["occ_full"],
                             posinf=c["occ_full"], neginf=c["occ_full"]).clamp_min(0.0)
    visible_3d = _smoothstep01(1.0 - n_occ / c["occ_full"])
    visibility = image_visibility * visible_3d

    # No-contact confidence is motion-only. Geometry must not make a stationary,
    # floating, or depth-distorted limb look trustworthy merely because it is far
    # from the reconstructed wall.
    label_evidence = torch.where(labels, surface, motion)
    label_confidence = visibility * label_evidence

    def as_numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float32)

    return {
        "image_visibility": as_numpy(image_visibility),
        "occlusion_visibility": as_numpy(visible_3d),
        "occluding_vertices": as_numpy(n_occ),
        "visibility": as_numpy(visibility),
        "surface": as_numpy(surface),
        "motion": as_numpy(motion),
        "label_confidence": as_numpy(label_confidence),
    }
