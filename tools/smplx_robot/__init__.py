"""SMPL-X robot-model tools.

The SMPL-X counterpart of :mod:`tools.robot_motion`'s SMPL helpers: build a
BetterRobot kinematic ``Model`` (free-flyer root + spherical joints) matching
``better_human``'s :class:`~better_human.smpl.SMPLXMid` 52-joint body, and remap
``q`` between ``better_human`` kintree order and BetterRobot's DFS order.

Only the Lie-accurate smoothness term in the kinematic stage needs the robot
model; all posed-mesh / joint inference stays in ``better_human``.
"""

from .model import (
    load_smplx_mid,
    make_smplx_robot_model,
    remap_q_bh_to_model,
    remap_q_model_to_bh,
)

__all__ = [
    "load_smplx_mid",
    "make_smplx_robot_model",
    "remap_q_bh_to_model",
    "remap_q_model_to_bh",
]
