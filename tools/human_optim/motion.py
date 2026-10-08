"""Manifold finite-difference derivatives of a configuration trajectory.

The kinematic optimizer penalizes how fast the SMPL-X trajectory ``q(t)`` moves
on the joint manifold. :func:`tangent_difference` is the one non-trivial piece:
the frame-to-frame Lie tangent ``q_{t+1} ⊖ q_t`` (the per-step velocity).
Acceleration and jerk are then ordinary time-differences of that velocity
(``d[1:] - d[:-1]``), taken inline at the call sites — a one-line diff does not
earn a dedicated helper.

Conventions this primitive commits to (must match everywhere it feeds):

* ``q`` is in **better_human** order — root SE3 free-flyer at ``q[:7]`` then the
  ``num_joints-1`` parent-local joint quaternions stride-4 in ``q[7:]`` — and is
  expected already normalized (:func:`tools.human_optim.qmath.normalize_q`).
* The result is a **raw per-step tangent** ``(T-1, 6 + 3·(num_joints-1))``; the
  ``dt`` factor is folded into the smoothness weights, not divided out here.
* Interior differencing only — length shrinks by one per derivative order.

This is the vectorized twin of BetterRobot's ``Model.difference`` (root via
``se3.log(q0⁻¹∘q1)``, each spherical joint via ``so3.log(q0⁻¹∘q1)``): identical
value and gradient, but batched over frames *and* joints in a handful of ops
instead of a per-joint Python loop — several-fold faster (≈4–10× on CPU here),
and needing no BetterRobot model build/remap just to difference.

One *other* trajectory-derivative path is kept deliberately separate because it
answers a different question: kindyn and the force witness difference ``q`` through the
BetterRobot model in DFS order with a physical ``dt`` and zero-padded endpoints
(:func:`velocity_acceleration_from_trajectory`, below).
"""

from __future__ import annotations

import torch
from better_robot.lie import se3, so3


def tangent_difference(q: torch.Tensor, num_joints: int) -> torch.Tensor:
    """Batched Lie tangent ``q_{t+1} ⊖ q_t`` → ``(T-1, 6 + 3·(num_joints-1))``.

    Root SE3 (``q[:7]``) via ``se3.log(q0⁻¹∘q1)``; each parent-local joint
    quaternion (``q[7:]``, stride-4) via ``so3.log(q0⁻¹∘q1)``, vectorized over
    frames and joints. The joint block stays in ``better_human`` order (not the
    model's DFS order) — irrelevant to the smoothness terms, which weight every
    joint tangent identically.
    """
    q0, q1 = q[:-1], q[1:]
    root = se3.log(se3.compose(se3.inverse(q0[..., :7]), q1[..., :7]))   # (T-1, 6)
    j0 = q0[..., 7:].unflatten(-1, (num_joints - 1, 4))                  # (T-1, J-1, 4)
    j1 = q1[..., 7:].unflatten(-1, (num_joints - 1, 4))
    dj = so3.log(so3.normalize(so3.compose(so3.inverse(j0), j1)))        # (T-1, J-1, 3)
    return torch.cat([root, dj.flatten(-2)], dim=-1)                     # (T-1, 6+3(J-1))


def velocity_acceleration_from_trajectory(
    model,
    q: torch.Tensor,
    dt: float,
    spacing: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference generalised velocity and acceleration from ``q(t)``.

    The frozen-``q`` trajectory derivatives kindyn and the force witness feed to RNEA.

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

    At ``spacing=1``::

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
