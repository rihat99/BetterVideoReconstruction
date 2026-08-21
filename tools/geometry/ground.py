"""Fit a floor plane to the fused scene cloud.

The gravity stage uses this to measure gravity directly from geometry: when the
floor is visible in the reconstruction, its upward normal is a better world-down
reference than the camera orientation. The caller supplies a *prior* down
direction (camera down, or GeoCalib's estimate) that only steers the search —
RANSAC hypotheses must be roughly horizontal and lie in the lower half of the
scene — while the final refinement is unconstrained, so the fitted normal is
honest evidence even when the prior is tilted. The caller decides how much
prior/fit disagreement to accept.

The RANSAC + total-least-squares recipe is the one validated on the climb_wall_2
rig (``scratch/cw2_gravity_fit.py``, 2026-08-20), where the camera is 16.6° off
true down: a simple lowest-band fit collapses onto the wall's bottom edge there,
while this one recovers the floor to within ~1° of the multi-view ground truth.
"""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def fit_ground_plane(points: np.ndarray, prior_down: np.ndarray, *,
                     max_tilt_deg: float = 30.0, iters: int = 4000,
                     min_planarity: float = 0.15, seed: int = 0
                     ) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    """RANSAC + TLS floor fit over a world-frame point cloud.

    Hypothesis planes come from random point triples, gated to within
    ``max_tilt_deg`` of horizontal (w.r.t. ``prior_down``) and to the lower half
    of the scene (a floor, not a ceiling); the best-supported one is refined by
    total least squares on its inliers, with the inlier set re-selected each
    round. The inlier distance is 0.8 % of the median point radius from the
    world origin (the anchor camera), which tracks the scene scale whether or
    not the cloud has been metricized.

    :param points: ``(N, 3)`` world-frame scene points.
    :param prior_down: ``(3,)`` prior world-down direction (search seed only).
    :param max_tilt_deg: hypothesis gate — candidate normals must sit within
        this angle of the prior vertical.
    :param iters: RANSAC iterations.
    :param min_planarity: reject when the inliers' middle-to-largest singular
        ratio falls below this — the "plane" is really a line (a wall's bottom
        edge), so its normal is undetermined.
    :returns: ``(centroid (3,), up_normal (3,), extent, n_inliers)`` — the plane
        centroid, its unit normal oriented against the prior down, a planar size
        for drawing a grid, and the final inlier count — or ``None`` when no
        supported floor plane exists.
    """
    pts = np.asarray(points, np.float64)
    if len(pts) < 300:
        return None
    g = _unit(np.asarray(prior_down, np.float64))
    rng = np.random.default_rng(seed)
    sub = pts[rng.choice(len(pts), min(len(pts), 60_000), replace=False)]
    tau = 0.008 * float(np.median(np.linalg.norm(sub, axis=1)))
    cos_gate = np.cos(np.radians(max_tilt_deg))
    height = sub @ g                                    # larger ⇒ lower
    mid_h = float(np.median(height))
    best_n, best_a, best = None, None, 0
    for _ in range(iters):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12 or abs((n @ g) / norm) < cos_gate:
            continue
        n = n / norm
        inl = np.abs((sub - a) @ n) < tau
        cnt = int(inl.sum())
        if cnt > best and np.median(height[inl]) > mid_h:  # low half: floor, not ceiling
            best_n, best_a, best = n, a, cnt
    if best_n is None:
        return None
    n, d = best_n, float(best_n @ best_a)
    for _ in range(3):                                  # TLS refine, unconstrained
        inl = np.abs(pts @ n - d) < tau
        if inl.sum() < 30:
            return None
        c = pts[inl].mean(0)
        n = np.linalg.svd(pts[inl] - c, full_matrices=False)[2][-1]
        if n @ g > 0.0:                                 # orient against the prior down
            n = -n
        d = float(n @ c)
    keep = pts[np.abs(pts @ n - d) < tau]
    centroid = keep.mean(0)
    centered = keep - centroid
    svals = np.linalg.svd(centered, full_matrices=False)[1]
    if svals[0] < 1e-9 or svals[1] / svals[0] < min_planarity:
        return None                                     # degenerate: a line, not a floor
    planar = centered - np.outer(centered @ n, n)
    extent = max(2.0 * float(np.quantile(np.linalg.norm(planar, axis=1), 0.9)), 0.5)
    return centroid, n, extent, len(keep)
