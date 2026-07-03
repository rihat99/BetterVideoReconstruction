"""SMPL-X robot-model tools.

Build a BetterRobot kinematic ``Model`` (free-flyer root + spherical joints)
matching ``better_human``'s :class:`~better_human.smpl.SMPLXMid` 52-joint body, and
remap ``q`` between ``better_human`` kintree order and BetterRobot's DFS order.

Only the Lie-accurate smoothness term in the kinematic stage needs the robot
model; all posed-mesh / joint inference stays in ``better_human``.
"""

from .inertia import (
    DEFAULT_TOTAL_MASS_KG,
    compute_smplx_body_inertia,
    make_smplx_inertial_model,
)
from .model import (
    load_smplx_mid,
    make_smplx_robot_model,
    remap_q_bh_to_model,
    remap_q_model_to_bh,
)

__all__ = [
    "DEFAULT_TOTAL_MASS_KG",
    "compute_smplx_body_inertia",
    "load_smplx_mid",
    "make_smplx_inertial_model",
    "make_smplx_robot_model",
    "remap_q_bh_to_model",
    "remap_q_model_to_bh",
]
