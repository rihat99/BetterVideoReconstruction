"""Trajectory finite differences for the SMPL-X force-only dynamics stage.

``optimize_dynamics_smplx`` freezes the kinematic ``q(t)`` and needs generalised
velocity / acceleration to feed RNEA. Central differences on the BetterRobot
manifold (via ``model.difference``) give both.
"""
from __future__ import annotations

import torch


def velocity_acceleration_from_trajectory(
    model,
    q: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference generalised velocity and acceleration from ``q(t)``.

    :param model: BetterRobot ``Model`` — supplies ``difference`` (tangent
        subtraction) and ``nv``.
    :param q: ``(T, nq)`` configuration trajectory.
    :param dt: frame interval, seconds.
    :returns: ``(v, a)``, each ``(T, nv)`` tangent-space; the boundary frames are
        pinned to zero so RNEA stays well-defined at the endpoints (quasi-static).

    Matches ``better_robot.residuals.acceleration``::

        a[t] = (difference(q[t], q[t+1]) − difference(q[t−1], q[t])) / dt²
    """
    T = q.shape[0]
    if T < 3:
        raise ValueError(f"need at least 3 frames for central differences, got T={T}")

    diff_fwd = model.difference(q[:-1], q[1:])  # (T-1, nv)

    v = torch.zeros(T, model.nv, dtype=q.dtype, device=q.device)
    v[1:-1] = 0.5 * (diff_fwd[:-1] + diff_fwd[1:]) / dt

    a = torch.zeros(T, model.nv, dtype=q.dtype, device=q.device)
    a[1:-1] = (diff_fwd[1:] - diff_fwd[:-1]) / (dt * dt)
    return v, a
