"""Voxel-grid merging of point clouds (GPU, single pass)."""
from __future__ import annotations

import torch


def voxel_downsample(points: torch.Tensor, voxel_size: float):
    """Collapse ``points`` ``(N, 3)`` sharing a ``voxel_size`` cell to their centroid.

    Returns ``(centroids (M, 3), inverse (N,), counts (M,))`` — ``inverse`` maps
    each input point to its voxel row, for reducing per-point attributes with
    :func:`group_reduce`. ``O(N)``, one ``torch.unique``.
    """
    ijk = torch.floor((points - points.amin(0)) / voxel_size).long()
    dims = ijk.amax(0) + 1
    keys = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    _, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
    centroids = torch.zeros((counts.numel(), 3), dtype=points.dtype, device=points.device)
    centroids.index_add_(0, inverse, points)
    return centroids / counts.unsqueeze(1), inverse, counts


def group_reduce(values: torch.Tensor, inverse: torch.Tensor, m: int, op: str) -> torch.Tensor:
    """Reduce ``values`` ``(N,)`` into ``m`` voxel groups by ``op`` (amax|amin|mean|sum)."""
    init = {"amax": float("-inf"), "amin": float("inf"), "mean": 0.0, "sum": 0.0}[op]
    out = torch.full((m,), init, dtype=values.dtype, device=values.device)
    return out.scatter_reduce_(0, inverse, values, op, include_self=False)
