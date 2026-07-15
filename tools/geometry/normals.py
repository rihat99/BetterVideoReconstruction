"""Per-point surface normals for an unstructured point cloud.

The fused scene cloud (``geometry/scene.npz``) stores points but no normals, yet
the penetration SDF in :mod:`tools.human_optim.losses` needs them to *sign* the
distance: a body vertex behind the local surface — on the far side of the
outward normal — is penetrating. We estimate one normal per point by local PCA
(the smallest principal axis of the k nearest neighbours) and orient it toward
the cameras that observed the scene, so "outward" consistently means "toward the
camera".

The whole estimate runs in NumPy on CPU: neighbour search is a ``cKDTree``, and
the batched 3×3 eigendecomposition is ``np.linalg.eigh`` (the GPU
``torch.linalg.eigh`` hits a cuSOLVER internal error on ~10⁵ tiny matrices). It
is a one-time precompute per run, so CPU is fine and keeps the GPU free.
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
