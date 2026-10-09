"""Depth-map cleaning: boundary-edge removal.

Uses the ``utils3d`` edge filter to drop flying boundary pixels from VGGT depth.
"""
from __future__ import annotations

import numpy as np


def clean_depth(depth: np.ndarray, *, edge_rtol: float = 0.04) -> np.ndarray:
    """Boolean validity mask for ``depth`` ``(H, W)``.

    Drops non-finite / non-positive depth and flying boundary pixels
    (``utils3d.np.depth_map_edge``).
    """
    mask = np.isfinite(depth) & (depth > 0)
    if edge_rtol > 0:
        import utils3d

        edge = utils3d.np.depth_map_edge(depth, rtol=edge_rtol, mask=mask)
        mask &= ~edge
    return mask
