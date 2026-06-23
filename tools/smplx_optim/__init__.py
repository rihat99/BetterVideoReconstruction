"""Visually-guided SMPL-X kinematic optimization: losses + camera/keypoint utils.

The genuinely new work behind ``scripts/stages/optimize_smplx.py``: project
SMPL-X joints to the image (reprojection), pull visible vertices onto the metric
depth cloud (chamfer), anchor the pose (prior), and limit acceleration + jerk
(smoothness). Posed-mesh inference stays in ``better_human``; the BetterRobot
robot model (``tools.smplx_robot``) supplies the Lie smoothness only.
"""

from .camera import (
    fold_root_to_world,
    matrix_to_se3,
    project,
    transform_points,
    world_from_cam,
)
from .keypoints import Correspondence, build_correspondence
from .losses import chamfer_loss, pose_prior_loss, reprojection_loss, smoothness_loss
from .qmath import normalize_q
from .visibility import front_facing_mask, subsample_indices, vertex_normals

__all__ = [
    "fold_root_to_world",
    "matrix_to_se3",
    "project",
    "transform_points",
    "world_from_cam",
    "Correspondence",
    "build_correspondence",
    "chamfer_loss",
    "pose_prior_loss",
    "reprojection_loss",
    "smoothness_loss",
    "normalize_q",
    "front_facing_mask",
    "subsample_indices",
    "vertex_normals",
]
