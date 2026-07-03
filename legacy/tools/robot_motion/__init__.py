"""SMPL params → BetterRobot robot motion: offsets helper + trajectory helpers."""
from .contact import (
    aggregate_vertex_contacts_to_joints,
    align_to_frames,
    load_joint_contacts,
    load_manual_joint_contacts,
    smpl_vertex_to_joint,
)
from .dynamics import (
    rebuild_inertial_model,
    velocity_acceleration_from_trajectory,
)
from .motion import (
    average_betas,
    build_trajectory,
    compute_smpl_joint_offsets,
    ground_centered_trajectory,
    remap_model_contacts_to_smpl,
    remap_model_q_to_smpl,
    remap_smpl_contacts_to_model,
    remap_smpl_q_to_model,
    y_up_to_z_up,
)
from .playback import play_trajectory
from .smoothing import (
    TrajectorySmoother,
    kernel_smooth,
    make_average_kernel,
    make_gaussian_kernel,
    smooth_trajectory,
)
from .smpl_inertia import (
    DENSITIES,
    assert_human_mass_in_range,
    compute_smpl_body_inertia,
    list_density_systems,
    load_smpl_face_segmentation,
)

__all__ = [
    "aggregate_vertex_contacts_to_joints",
    "align_to_frames",
    "assert_human_mass_in_range",
    "average_betas",
    "build_trajectory",
    "compute_smpl_body_inertia",
    "compute_smpl_joint_offsets",
    "DENSITIES",
    "ground_centered_trajectory",
    "kernel_smooth",
    "list_density_systems",
    "load_joint_contacts",
    "load_manual_joint_contacts",
    "load_smpl_face_segmentation",
    "make_average_kernel",
    "make_gaussian_kernel",
    "play_trajectory",
    "rebuild_inertial_model",
    "remap_model_contacts_to_smpl",
    "remap_model_q_to_smpl",
    "remap_smpl_contacts_to_model",
    "remap_smpl_q_to_model",
    "smooth_trajectory",
    "smpl_vertex_to_joint",
    "TrajectorySmoother",
    "velocity_acceleration_from_trajectory",
    "y_up_to_z_up",
]
