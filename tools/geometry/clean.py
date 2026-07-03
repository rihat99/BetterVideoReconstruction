"""Depth-map cleaning: confidence gating + boundary-edge removal.

Uses the ``utils3d`` edge filter to drop flying boundary pixels from VGGT depth.
"""
from __future__ import annotations

import numpy as np


def clean_depth(
    depth: np.ndarray,
    conf: np.ndarray | None = None,
    *,
    conf_drop_q: float = 0.0,
    edge_rtol: float = 0.04,
    normal: np.ndarray | None = None,
    normal_tol: float = 0.0,
) -> np.ndarray:
    """Boolean validity mask for ``depth`` ``(H, W)``.

    Drops non-finite / non-positive depth, flying boundary pixels
    (``utils3d.np.depth_map_edge``) and — when ``conf`` is given — the lowest
    ``conf_drop_q`` fraction of pixels by confidence (a per-map quantile, since
    the confidence scale drifts between frames).
    """
    mask = np.isfinite(depth) & (depth > 0)
    if conf is not None and conf_drop_q > 0 and mask.any():
        mask &= conf >= np.quantile(conf[mask], conf_drop_q)
    if edge_rtol > 0:
        import utils3d

        edge = utils3d.np.depth_map_edge(depth, rtol=edge_rtol, mask=mask)
        if normal_tol > 0 and normal is not None:
            edge &= utils3d.np.normal_map_edge(normal, tol=normal_tol, mask=mask)
        mask &= ~edge
    return mask
