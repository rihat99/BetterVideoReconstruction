"""Robust metric scale between up-to-scale (VGGT) and metric (MoGe2) depth."""
from __future__ import annotations

import cv2
import numpy as np


def frame_scale(
    vggt_depth: np.ndarray,
    moge_depth: np.ndarray,
    vggt_mask: np.ndarray,
    moge_mask: np.ndarray,
) -> tuple[float, int]:
    """Median ``moge / vggt`` depth ratio over co-visible valid pixels.

    The MoGe maps (original resolution) are nearest-resized onto the VGGT grid.
    Returns ``(ratio, n_inliers)``; ``ratio`` is ``nan`` when nothing overlaps.
    """
    h, w = vggt_depth.shape
    md = cv2.resize(np.asarray(moge_depth, np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    mm = cv2.resize(moge_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    co = vggt_mask & mm & (md > 0)
    n = int(co.sum())
    return (float(np.median(md[co] / vggt_depth[co])) if n else float("nan")), n


def weighted_median(values, weights) -> float:
    """Weighted median of ``values``, ignoring non-finite values / zero weights."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(values) & (weights > 0)
    values, weights = values[ok], weights[ok]
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, 0.5 * cum[-1])])
