"""Voxel-grid merging of point clouds + connected-component filtering."""
from __future__ import annotations

import torch


def voxel_downsample(points: torch.Tensor, voxel_size: float, *, shift: float = 0.0,
                     origin: torch.Tensor | None = None):
    """Collapse ``points`` ``(N, 3)`` sharing a ``voxel_size`` cell to their centroid.

    Returns ``(centroids (M, 3), inverse (N,), counts (M,), coords (M, 3))`` —
    ``inverse`` maps each input point to its voxel row (for reducing per-point
    attributes with :func:`group_reduce`) and ``coords`` are the integer grid
    indices of each occupied voxel (for :func:`large_component_mask`). ``O(N)``,
    one ``torch.unique``. ``origin`` fixes the grid origin (default: the cloud's
    min corner) and ``shift`` offsets it in cell units — a second pass over the
    SAME origin with ``shift=0.5`` merges the twin centroids a
    boundary-straddling surface leaves behind (the origin must be shared, or the
    second lattice can realign with the first).
    """
    base = points.amin(0) if origin is None else origin
    ijk = torch.floor((points - base) / voxel_size + shift).long()
    ijk -= ijk.amin(0)  # re-zero so the linear key below stays non-negative
    dims = ijk.amax(0) + 1
    keys = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    _, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
    centroids = torch.zeros((counts.numel(), 3), dtype=points.dtype, device=points.device)
    centroids.index_add_(0, inverse, points)
    coords = torch.zeros((counts.numel(), 3), dtype=torch.long, device=points.device)
    coords[inverse] = ijk  # members of a voxel share ijk, so any write wins
    return centroids / counts.unsqueeze(1), inverse, counts, coords


def large_component_mask(coords: torch.Tensor, min_voxels: int) -> torch.Tensor:
    """Keep-mask ``(M,)`` dropping voxels whose 26-connected cluster has ``< min_voxels`` cells.

    ``coords`` ``(M, 3)`` are integer voxel indices (from :func:`voxel_downsample`).
    Occupied voxels are grouped into 26-connected components; only components with
    at least ``min_voxels`` voxels survive — the small ones are floating depth
    fragments / off-surface blobs. ``min_voxels <= 1`` keeps everything.

    Builds a sparse neighbour graph (not a dense grid), so a lone far-outlier
    voxel can't blow up memory.
    """
    m = coords.shape[0]
    if min_voxels <= 1 or m == 0:
        return torch.ones(m, dtype=torch.bool, device=coords.device)

    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ijk = coords.detach().cpu().numpy().astype(np.int64)
    ijk = ijk - ijk.min(0)
    dims = ijk.max(0) + 2  # +2 pad so ±1 neighbours never wrap the linear index
    lin = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    order = np.argsort(lin)
    lin_sorted = lin[order]

    # 13 forward neighbour offsets (half of 26-connectivity → each edge once).
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) > (0, 0, 0)]
    rows, cols = [], []
    for dx, dy, dz in offsets:
        neigh = lin + (dx * dims[1] + dy) * dims[2] + dz
        pos = np.clip(np.searchsorted(lin_sorted, neigh), 0, m - 1)
        hit = lin_sorted[pos] == neigh
        rows.append(np.nonzero(hit)[0])
        cols.append(order[pos[hit]])
    rows, cols = np.concatenate(rows), np.concatenate(cols)

    graph = coo_matrix((np.ones(rows.size, np.uint8), (rows, cols)), shape=(m, m))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    keep = sizes[labels] >= min_voxels
    return torch.from_numpy(keep).to(coords.device)


def group_reduce(values: torch.Tensor, inverse: torch.Tensor, m: int, op: str) -> torch.Tensor:
    """Reduce ``values`` ``(N,)`` into ``m`` voxel groups by ``op`` (amax|amin|mean|sum)."""
    init = {"amax": float("-inf"), "amin": float("inf"), "mean": 0.0, "sum": 0.0}[op]
    out = torch.full((m,), init, dtype=values.dtype, device=values.device)
    return out.scatter_reduce_(0, inverse, values, op, include_self=False)
