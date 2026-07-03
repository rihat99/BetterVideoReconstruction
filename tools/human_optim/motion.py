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

Two *other* trajectory-derivative paths are kept deliberately separate because
they answer different questions: the dynamics stage differences ``q`` through the
BetterRobot model in DFS order with a physical ``dt`` and zero-padded endpoints
(:func:`tools.smplx_robot.dynamics.velocity_acceleration_from_trajectory`), and
the velocity-gated contact term needs each joint's local *linear* speed from FK
Jacobians (:mod:`tools.human_optim.velocity`).
"""

from __future__ import annotations

import torch
from better_robot.lie import se3, so3


def tangent_difference(q: torch.Tensor, num_joints: int) -> torch.Tensor:
    """Batched Lie tangent ``q_{t+1} ⊖ q_t`` → ``(T-1, 6 + 3·(num_joints-1))``.

    Root SE3 (``q[:7]``) via ``se3.log(q0⁻¹∘q1)``; each parent-local joint
    quaternion (``q[7:]``, stride-4) via ``so3.log(q0⁻¹∘q1)``, vectorized over
    frames and joints. The joint block stays in ``better_human`` order (not the
    model's DFS order) — irrelevant to
    :func:`~tools.human_optim.losses.smoothness_loss`, which weights every joint
    tangent identically.
    """
    q0, q1 = q[:-1], q[1:]
    root = se3.log(se3.compose(se3.inverse(q0[..., :7]), q1[..., :7]))   # (T-1, 6)
    j0 = q0[..., 7:].unflatten(-1, (num_joints - 1, 4))                  # (T-1, J-1, 4)
    j1 = q1[..., 7:].unflatten(-1, (num_joints - 1, 4))
    dj = so3.log(so3.normalize(so3.compose(so3.inverse(j0), j1)))        # (T-1, J-1, 3)
    return torch.cat([root, dj.flatten(-2)], dim=-1)                     # (T-1, 6+3(J-1))
