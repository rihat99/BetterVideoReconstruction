"""Surface normals: of an unstructured point cloud, and of a mesh's vertices.

The fused scene cloud (``geometry/scene.npz``) is points without a surface, yet the scene terms need
normals to *sign* a distance: a point behind the local surface, on the far side of the outward normal,
is inside it. One normal per point comes from local PCA (the smallest principal axis of the k nearest
neighbours), turned toward the cameras that observed the scene, so "outward" means "toward the camera".

The cloud estimate runs in NumPy on the CPU: the neighbour search is a ``cKDTree``, and the batched
3x3 eigendecomposition is ``np.linalg.eigh`` (the GPU ``torch.linalg.eigh`` hits a cuSOLVER internal
error on ~10^5 tiny matrices). It runs once per clip.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree


def estimate_point_normals(
    points: torch.Tensor,          # (M, 3) world points
    camera_centers: torch.Tensor,  # (T, 3) world camera centers
    *,
    k: int = 16,
) -> torch.Tensor:
    """Unit normals ``(M, 3)`` for an unstructured cloud, oriented toward cameras.

    The normal at each point is the eigenvector of its k-NN covariance with the
    smallest eigenvalue (the local surface normal). It is then flipped to face
    the *nearest* camera center, giving a globally consistent outward
    orientation for the camera-observed scene surface — which is what lets the
    penetration loss read "behind the surface" as a negative signed distance.
    """
    dev = points.device
    pts = np.ascontiguousarray(points.detach().cpu().numpy(), dtype=np.float64)   # (M, 3)
    if pts.shape[0] == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=dev)
    cams = np.ascontiguousarray(camera_centers.detach().cpu().numpy(), dtype=np.float64)
    kk = min(int(k), pts.shape[0])

    _, idx = cKDTree(pts).query(pts, k=kk)               # (M, kk) — includes self
    idx = idx.reshape(pts.shape[0], kk)                  # k=1 queries drop the axis
    nb = pts[idx]                                        # (M, kk, 3)
    X = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", X, X) / kk           # (M, 3, 3)
    n = np.linalg.eigh(cov)[1][..., 0]                   # smallest-eigenvalue axis (M, 3)

    _, cami = cKDTree(cams).query(pts, k=1)              # nearest camera per point
    flip = ((cams[cami] - pts) * n).sum(axis=-1) < 0     # normal points away from camera
    n[flip] *= -1.0
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    return torch.as_tensor(n, dtype=torch.float32, device=dev)


def load_scene_cloud(
    sc: dict,                      # loaded geometry/scene.npz arrays
    camera_centers: torch.Tensor,  # (T, 3) world camera centers (fallback orientation)
    *,
    conf_drop_q: float,
    k: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(points, normals, confidence)`` from a scene.npz dict, confidence-gated.

    Drops the lowest ``conf_drop_q`` fraction of points by VGGT confidence (the
    raw scale is arbitrary, hence a quantile). Normals come from the ``normals``
    fuse stored (subset with the same mask, so they stay index-aligned); for
    artifacts written before fuse saved normals they are PCA-estimated here with
    :func:`estimate_point_normals`.
    """
    pts = torch.as_tensor(sc["points"], dtype=torch.float32, device=device)
    conf = torch.as_tensor(sc["confidence"], dtype=torch.float32, device=device)
    normals = sc.get("normals")
    if normals is not None:
        normals = torch.as_tensor(normals, dtype=torch.float32, device=device)
    if conf_drop_q > 0.0 and conf.numel():
        keep = conf >= torch.quantile(conf, conf_drop_q)
        pts, conf = pts[keep], conf[keep]
        normals = normals[keep] if normals is not None else None
    if normals is None:
        normals = estimate_point_normals(pts, camera_centers, k=k)
    return pts, normals, conf


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Unit normals ``(..., V, 3)`` of mesh vertices ``(..., V, 3)``: the area-weighted mean of their faces' normals."""
    a, b, c = (vertices[..., faces[:, corner], :] for corner in range(3))
    normals = torch.zeros_like(vertices).index_add_(
        -2, faces.reshape(-1), torch.linalg.cross(b - a, c - a).repeat_interleave(3, dim=-2))
    return torch.nn.functional.normalize(normals, dim=-1)
