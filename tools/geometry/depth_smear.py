"""Per-frame human-cloud depth smear — is this frame's human depth trustworthy?

VGGT's per-frame human depth has one characteristic catastrophic failure: fast
motion (or fast zoom) smears the person's depth points ALONG THE VIEWING RAY,
turning a ~0.3 m-deep body into a metres-long streak pointing at the camera
(pommel_horse: streaks of 4+ m). Everything that consumes those points — the
chamfer data term, the metric-scale ruler — then pulls the body toward garbage.

The detector is one scale-invariant ratio per frame::

    smear = (depth extent along the view ray) / (largest lateral extent)

both as robust p90 − p10 extents about the cloud centroid. A real person seen
from any angle has more lateral extent (height) than depth, so a healthy frame
sits well below 1; a smeared frame reads several. Being a ratio it works in
metric and in up-to-scale units alike, and in world or camera frame (pass the
camera position as ``origin``; camera-frame points use the default origin 0).

Measured on every processed clip (2026-08-24): all 13 annotated climbing scenes
have max smear ≤ 1.6; pommel_horse has 88 % of frames above 2.5 (median 7.9),
floor_exercise 45 %, high_jump 27 %, olympics_2 16 %. The weight law therefore
has a **dead zone**: frames at or below :data:`SMEAR_STAR` = 2 keep weight
exactly 1 (climbing behaviour is bit-identical), and only genuinely smeared
frames are downweighted::

    weight = min(1, (SMEAR_STAR / smear) ** SMEAR_POWER)

(≈ 0.4 at smear 2.5, 0.2 at 3, ~0.004 at 8). A frame with too few points to
measure gets weight 1 — absence of a witness is not evidence of bad depth.
"""

from __future__ import annotations

import numpy as np
import torch

#: Smear ratio at which downweighting starts (the dead-zone edge). All clean
#: clips measured sit below it; the catastrophic ones sit far above.
SMEAR_STAR = 2.0
#: Sharpness of the falloff beyond the dead zone.
SMEAR_POWER = 4.0
#: Fewer points than this and the frame's smear is unmeasurable (weight 1).
MIN_POINTS = 30


def alongray_smear(points: np.ndarray, origin: np.ndarray | None = None) -> float:
    """One frame's smear ratio from its ``(n, 3)`` human points; NaN if unmeasurable.

    ``origin`` is the camera position in the same coordinates as ``points``
    (``None`` = the origin, i.e. camera-frame points). The view ray is taken
    through the cloud centroid — the camera is far relative to body size, so one
    direction per frame is enough.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < MIN_POINTS:
        return float("nan")
    center = points.mean(0)
    ray = center - (np.zeros(3) if origin is None else np.asarray(origin, np.float64))
    norm = float(np.linalg.norm(ray))
    if norm < 1e-9:
        return float("nan")
    ray /= norm
    offsets = points - center
    z = offsets @ ray
    ray_extent = float(np.percentile(z, 90) - np.percentile(z, 10))
    lateral = offsets - z[:, None] * ray
    # Largest lateral extent: project onto the dominant eigenvector of the (rank-2)
    # lateral covariance — for a person this is the height axis in view.
    eigvals, eigvecs = np.linalg.eigh(np.cov(lateral.T))
    u = lateral @ eigvecs[:, -1]
    lat_extent = float(np.percentile(u, 90) - np.percentile(u, 10))
    return ray_extent / max(lat_extent, 1e-6)


def smear_weight(smear: float) -> float:
    """The per-frame weight in ``(0, 1]``; NaN smear (unmeasurable) → 1."""
    if not np.isfinite(smear) or smear <= SMEAR_STAR:
        return 1.0
    return float((SMEAR_STAR / smear) ** SMEAR_POWER)


@torch.no_grad()
def padded_frame_weights(
    cloud: torch.Tensor,            # (T, M, 3) padded per-frame points
    valid: torch.Tensor,            # (T, M) bool — real (vs padding) points
    origins: torch.Tensor | None = None,   # (T, 3) camera positions; None = origin
) -> torch.Tensor:
    """Per-frame smear weights ``(T,)`` for a padded cloud, on ``cloud``'s device."""
    pts = cloud.detach().cpu().numpy()
    ok = valid.detach().cpu().numpy()
    org = None if origins is None else origins.detach().cpu().numpy()
    weights = np.ones(pts.shape[0], dtype=np.float32)
    for t in range(pts.shape[0]):
        weights[t] = smear_weight(alongray_smear(
            pts[t][ok[t]], None if org is None else org[t]))
    return torch.as_tensor(weights, device=cloud.device, dtype=cloud.dtype)
