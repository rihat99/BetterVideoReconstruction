"""Backproject a z-depth map through a pinhole camera into world coordinates."""
from __future__ import annotations

import torch


def backproject(
    depth: torch.Tensor,
    K_px: torch.Tensor,
    extrinsic: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pixels with z-depth → world points ``(N, 3)``.

    ``K_px`` is pixel-unit intrinsics on ``depth``'s grid; ``extrinsic`` is the
    ``(4, 4)`` camera-from-world matrix (OpenCV). With ``mask`` only those pixels
    are returned, in row-major order. All tensors share one device.
    """
    h, w = depth.shape
    ys = torch.arange(h, device=depth.device, dtype=depth.dtype)
    xs = torch.arange(w, device=depth.device, dtype=depth.dtype)
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    if mask is not None:
        uu, vv, z = uu[mask], vv[mask], depth[mask]
    else:
        uu, vv, z = uu.reshape(-1), vv.reshape(-1), depth.reshape(-1)
    x = (uu - K_px[0, 2]) / K_px[0, 0] * z
    y = (vv - K_px[1, 2]) / K_px[1, 1] * z
    cam = torch.stack((x, y, z), dim=-1)            # (N, 3) camera space
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    return (cam - t) @ R                            # Rᵀ (X − t)
