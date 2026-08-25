"""Trajectory finite differences for the SMPL-X dynamics solves.

kindyn and the force witness freeze the kinematic ``q(t)`` and need generalised
velocity / acceleration to feed RNEA. Central differences on the BetterRobot
manifold (via ``model.difference``) give both.
"""
from __future__ import annotations

import torch


def velocity_acceleration_from_trajectory(
    model,
    q: torch.Tensor,
    dt: float,
    spacing: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference generalised velocity and acceleration from ``q(t)``.

    :param model: BetterRobot ``Model`` — supplies ``difference`` (tangent
        subtraction) and ``nv``.
    :param q: ``(T, nq)`` configuration trajectory.
    :param dt: frame interval, seconds.
    :param spacing: stencil step in frames. ``1`` is the classic per-frame central
        difference. A high-fps clip amplifies per-frame pose noise into the
        acceleration by ``fps²``, so callers pass ``round(fps/30)`` to keep the
        stencil's TIME step (and with it the noise amplification) at the 30 fps
        reference — the estimate a 30 fps camera of the same motion would give.
    :returns: ``(v, a)``, each ``(T, nv)`` tangent-space; the first and last
        ``spacing`` frames are pinned to zero so RNEA stays well-defined at the
        endpoints (quasi-static).

    At ``spacing=1`` matches ``better_robot.residuals.acceleration``::

        a[t] = (difference(q[t], q[t+h]) − difference(q[t−h], q[t])) / (h·dt)²
    """
    T = q.shape[0]
    h = max(1, int(spacing))
    if T < 2 * h + 1:
        raise ValueError(f"need at least {2 * h + 1} frames for spacing={h} central "
                         f"differences, got T={T}")

    diff_fwd = model.difference(q[:-h], q[h:])  # (T-h, nv) step-h tangent differences

    v = torch.zeros(T, model.nv, dtype=q.dtype, device=q.device)
    v[h:-h] = 0.5 * (diff_fwd[:-h] + diff_fwd[h:]) / (h * dt)

    a = torch.zeros(T, model.nv, dtype=q.dtype, device=q.device)
    a[h:-h] = (diff_fwd[h:] - diff_fwd[:-h]) / (h * dt) ** 2
    return v, a
