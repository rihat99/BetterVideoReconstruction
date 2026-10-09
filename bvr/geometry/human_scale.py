"""Robust aggregation of per-frame log-scales into one metric scale.

The human-body ruler stage (:mod:`scripts.stages.estimate_scale`) turns VGGT's
single up-to-scale factor into a metric one by comparing the SAM-3D-Body SMPL-X
body (which carries real metric size) against VGGT's relative depth over the
person. Per frame it reduces the per-vertex log-ratios ``log(d_metric) −
log(d_rel)`` to one robust center; this module combines those per-frame
log-scales across the clip.

VGGT has **one** scale for the whole clip (depth + camera translations share it),
so the per-frame estimates are noisy observations of a single constant. The
combine is therefore a confidence-weighted robust mean in log space: a weighted
MAD gate around the weighted median drops whole bad frames, then the survivors'
log-scales are averaged weighted by each frame's confidence mass. The gate must
be weighted too — the smear downweighting hands garbage frames near-zero weight,
and on a clip where they are the *majority* an unweighted median would centre on
the garbage cluster and reject the clean frames as outliers. Log space keeps the
estimate symmetric under too-near / too-far errors (scale is multiplicative).
"""
from __future__ import annotations

import numpy as np


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of 1-D ``values`` (``weights ≥ 0``).

    The smallest value whose cumulative weight reaches half the total — a robust
    center that lets a per-vertex weight (e.g. VGGT confidence × body-part emphasis)
    steer which vertices set a frame's log-scale. Falls back to the plain median if
    all weights are zero; ``nan`` for empty input.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    total = float(cw[-1])
    if total <= 0.0:
        return float(np.median(values))
    idx = int(np.searchsorted(cw, 0.5 * total))
    return float(v[min(idx, v.size - 1)])


def combine_log_scales(
    frame_log: np.ndarray,
    frame_weight: np.ndarray,
    *,
    mad_k: float = 3.0,
) -> dict:
    """Confidence-weighted robust mean of per-frame log-scales → one global scale.

    :param frame_log: ``(F,)`` per-frame log-scale ``log(s_k)`` (``nan`` where a
        frame produced no estimate).
    :param frame_weight: ``(F,)`` per-frame confidence weight (``≥ 0``; the
        summed VGGT depth-confidence of that frame's inlier vertices).
    :param mad_k: reject a frame whose log-scale deviates from the median by more
        than ``mad_k`` median-absolute-deviations before averaging.
    :returns: dict with ``scale`` (``exp`` of the combined log-scale),
        ``log_scale``, ``mad`` (of the per-frame log-scales, in log units),
        ``n_used`` (inlier frame count) and ``inlier_mask`` ``(F,)``.
    """
    frame_log = np.asarray(frame_log, dtype=np.float64)
    frame_weight = np.asarray(frame_weight, dtype=np.float64)
    finite = np.isfinite(frame_log) & np.isfinite(frame_weight) & (frame_weight > 0.0)
    if not finite.any():
        raise ValueError("no valid per-frame log-scales to combine")

    logs, w = frame_log[finite], frame_weight[finite]
    # Weighted centre + weighted MAD: a downweighted frame must not steer the gate
    # any more than it steers the average (see the module docstring).
    center = weighted_median(logs, w)
    mad = weighted_median(np.abs(logs - center), w)
    # A near-zero MAD (all frames agree) leaves every frame an inlier.
    inlier = np.abs(logs - center) <= mad_k * mad if mad > 0.0 else np.ones_like(logs, bool)

    log_scale = float(np.average(logs[inlier], weights=w[inlier]))
    full_inlier = np.zeros_like(frame_log, dtype=bool)
    full_inlier[np.flatnonzero(finite)[inlier]] = True
    return {
        "scale": float(np.exp(log_scale)),
        "log_scale": log_scale,
        "mad": mad,
        "n_used": int(inlier.sum()),
        "inlier_mask": full_inlier,
    }
