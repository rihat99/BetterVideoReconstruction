"""Visually-guided SMPL-X kinematic optimization: losses + camera/keypoint utils.

The genuinely new work behind ``scripts/stages/human_optimization.py`` (kinematics_1): project
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
from .contacts import (
    contact_from_sdf,
    drift_gated_contact,
    hysteresis_spans,
    joint_drift,
    joint_guarded_min_sdf,
    lift_joint_min_distance,
    joint_occlusion_counts,
    lift_joint_contact,
    merge_contact_spans,
    scene_sdf,
    stillness_weight,
    vertex_to_joint,
)
from .keypoints import Correspondence, build_correspondence
from .losses import (
    chamfer_loss,
    geman_mcclure,
    init_reprojection_gm,
    joint_velocity_loss,
    pose_prior_loss,
    sapiens_reprojection_gm,
    scene_penetration,
    smoothness_loss,
)
from .motion import tangent_difference
from .qmath import normalize_q
from .visibility import front_facing_mask, subsample_indices, vertex_normals

__all__ = [
    "fold_root_to_world",
    "matrix_to_se3",
    "project",
    "transform_points",
    "world_from_cam",
    "contact_from_sdf",
    "drift_gated_contact",
    "hysteresis_spans",
    "joint_drift",
    "joint_guarded_min_sdf",
    "lift_joint_min_distance",
    "joint_occlusion_counts",
    "lift_joint_contact",
    "merge_contact_spans",
    "scene_sdf",
    "stillness_weight",
    "vertex_to_joint",
    "Correspondence",
    "build_correspondence",
    "chamfer_loss",
    "geman_mcclure",
    "init_reprojection_gm",
    "joint_velocity_loss",
    "pose_prior_loss",
    "sapiens_reprojection_gm",
    "scene_penetration",
    "smoothness_loss",
    "tangent_difference",
    "normalize_q",
    "front_facing_mask",
    "subsample_indices",
    "vertex_normals",
]
