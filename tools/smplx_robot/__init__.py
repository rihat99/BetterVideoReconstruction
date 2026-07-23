"""SMPL-X robot-model tools.

The body itself is a BetterRobot-backed :class:`better_human.SMPLX` (52 joints,
free-flyer root + spherical joints), so posing, the Lie smoothness ``difference``
and the baked per-part inertia all come off the loaded body directly. This package
now only holds the body loader and the frozen-``q`` trajectory finite differences
the dynamics stage feeds to RNEA.
"""

from .dynamics import velocity_acceleration_from_trajectory
from .model import load_smplx_mid

__all__ = [
    "load_smplx_mid",
    "velocity_acceleration_from_trajectory",
]
