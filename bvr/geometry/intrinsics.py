"""Camera-intrinsics normalization helpers.

VGGT-Omega and the fused geometry carry *normalized* intrinsics
(``fx/W, fy/H, cx/W, cy/H``) so camera parameters are resolution-independent; the
pixel-unit ``K`` the body / geometry stages consume is recovered per frame with
:func:`denormalize_K`. Both accept a single ``(3, 3)`` or a batched
``(..., 3, 3)`` intrinsics array.
"""
from __future__ import annotations

import numpy as np


def denormalize_K(intrinsics: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalized intrinsics ``(..., 3, 3)`` → pixel-unit ``K`` (float32).

    ``fx, cx`` scale by image width; ``fy, cy`` by image height.
    """
    K = np.asarray(intrinsics, dtype=np.float32).copy()
    K[..., 0, 0] *= width
    K[..., 0, 2] *= width
    K[..., 1, 1] *= height
    K[..., 1, 2] *= height
    return K


def normalize_K(K_px: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pixel-unit ``K`` ``(..., 3, 3)`` → normalized convention (``fx/W, fy/H``).

    Inverse of :func:`denormalize_K`; resolution-independent, so the result maps
    cleanly back onto full-resolution frames.
    """
    K = np.asarray(K_px).copy()
    K[..., 0, 0] /= width
    K[..., 0, 2] /= width
    K[..., 1, 1] /= height
    K[..., 1, 2] /= height
    return K
