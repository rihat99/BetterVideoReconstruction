"""Camera / frame math for the SMPL-X kinematic optimizer.

The optimizer runs in the **world** frame (so smoothness and chamfer compare a
body against a single consistent space across the clip). The SMPL-X params from
the converter are in the **camera** frame, so we fold the per-frame
``world_from_cam`` into the root before optimizing, and project world joints back
through the per-frame extrinsic + pixel intrinsics for the reprojection loss.

Frame conventions (see ``docs/geometry_fusion_plan.md``): ``extrinsics`` are
OpenCV ``cam_from_world`` (X right, Y down, Z forward); ``intrinsics_px_orig``
are pixel intrinsics at the original frame resolution.
"""

from __future__ import annotations

import torch

from better_robot.lie import se3, so3


def world_from_cam(extrinsics: torch.Tensor) -> torch.Tensor:
    """Invert ``cam_from_world (T, 4, 4)`` → ``world_from_cam (T, 4, 4)``."""
    return torch.linalg.inv(extrinsics)


def matrix_to_se3(mat: torch.Tensor) -> torch.Tensor:
    """``(..., 4, 4)`` homogeneous transform → BetterRobot SE3 ``(..., 7)``."""
    t = mat[..., :3, 3]
    quat = so3.from_matrix(mat[..., :3, :3])
    return torch.cat([t, quat], dim=-1)


def fold_root_to_world(q_cam: torch.Tensor, world_from_cam_mat: torch.Tensor) -> torch.Tensor:
    """Move a per-frame camera-frame ``q (T, nq)`` into the world frame.

    Only the root SE3 (``q[..., :7]``) changes: ``root_world =
    world_from_cam ∘ root_cam``. Parent-local joint quaternions are unchanged.
    """
    wfc = matrix_to_se3(world_from_cam_mat)  # (T, 7)
    out = q_cam.clone()
    out[..., :7] = se3.compose(wfc, q_cam[..., :7])
    return out


def transform_points(points_world: torch.Tensor, extrinsic: torch.Tensor) -> torch.Tensor:
    """Map world points to the camera frame via ``cam_from_world``.

    ``points_world`` is ``(T, K, 3)``; ``extrinsic`` is ``(T, 4, 4)``.
    """
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]
    return torch.einsum("tij,tkj->tki", R, points_world) + t[:, None, :]


def project(points_cam: torch.Tensor, K: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pinhole-project camera-frame points ``(T, K, 3)`` to pixels ``(T, K, 2)``.

    ``K`` is ``(T, 3, 3)`` pixel intrinsics. Depth is clamped to ``eps`` so
    points at/behind the image plane don't produce NaNs (they get large pixel
    error and are down-weighted by their keypoint score).
    """
    fx, fy = K[:, 0, 0, None], K[:, 1, 1, None]
    cx, cy = K[:, 0, 2, None], K[:, 1, 2, None]
    z = points_cam[..., 2].clamp_min(eps)
    u = fx * points_cam[..., 0] / z + cx
    v = fy * points_cam[..., 1] / z + cy
    return torch.stack([u, v], dim=-1)
