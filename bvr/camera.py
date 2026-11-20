"""The cameras of a clip: world points to the camera frame and to pixels.

A clip has one pinhole camera per frame. ``extrinsics (T, 4, 4)`` take world points to the camera
frame (x right, y down, z forward) and ``intrinsics (T, 3, 3)`` take camera points to pixels of the
original video frame; both are ``geometry/transform.npz``'s. Points carry the frame axis before
their own: ``(..., T, K, 3)``, with any axes (people) in front.
"""
from __future__ import annotations

import torch

#: The cameras of a clip, as functions that take both receive them: ``(extrinsics, intrinsics)``.
Cameras = tuple[torch.Tensor, torch.Tensor]

#: Depth at which a point at or behind the camera is projected (m): it lands far off the image, not at NaN.
NEAR = 1e-6


def to_camera(points: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """World points ``(..., T, K, 3)`` in the camera frames of ``extrinsics (T, 4, 4)``."""
    return points @ extrinsics[:, :3, :3].mT + extrinsics[:, None, :3, 3]


def project(points: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Pixels ``(..., T, K, 2)`` of world points ``(..., T, K, 3)``; ``intrinsics`` is ``(T, 3, 3)``."""
    camera = to_camera(points, extrinsics)
    focal = torch.stack((intrinsics[:, 0, 0], intrinsics[:, 1, 1]), dim=-1)[:, None]
    return focal * camera[..., :2] / camera[..., 2:].clamp_min(NEAR) + intrinsics[:, None, :2, 2]


def centers(extrinsics: torch.Tensor) -> torch.Tensor:
    """World positions ``(T, 3)`` of the cameras of ``extrinsics (T, 4, 4)``."""
    return -(extrinsics[:, None, :3, 3] @ extrinsics[:, :3, :3])[:, 0]
