"""Per-joint SE3 (post-FK) linear velocity via the BetterRobot kinematic Jacobian.

The velocity-gated contact-repulsion loss
(:func:`tools.human_optim.losses.contact_velocity_loss`) needs each joint's
*spatial* speed — how fast the joint moves through the world — **not** the SO3
tangent of its own quaternion (the quantity the smoothness term penalizes). This
is the **LOCAL-frame linear part of the joint twist after forward kinematics**;
its magnitude equals the world-space joint speed ``‖ṗ_joint‖`` (the body-frame
rotation preserves norm), computed here the standard robotics way::

    v_j = Ad(T_j^{-1}) · J_j · q̇            (local spatial Jacobian × config velocity)

with BetterRobot's kinematic Jacobian and the tangent-space configuration
velocity ``q̇`` (finite-difference of consecutive ``q`` via ``model.difference``).

The BetterRobot ``Model`` is built **once per person** (betas fixed) by the caller
(:func:`tools.smplx_robot.make_smplx_robot_model`) and reused; everything here is
batched over the ``T`` frames and runs under ``no_grad`` — the speed is a detached
gate weight, not a gradient path.
"""

from __future__ import annotations

import torch
from better_robot.kinematics import compute_joint_jacobians, forward_kinematics
from better_robot.lie import se3

from tools.smplx_robot import remap_q_bh_to_model

from .qmath import normalize_q


@torch.no_grad()
def joint_local_linear_speed(model, mid, q_bh: torch.Tensor, fps: float) -> torch.Tensor:
    """Per-joint LOCAL-frame linear speed ``(T, J)`` in metres / second.

    Parameters
    ----------
    model
        BetterRobot SMPL-X-mid ``Model`` (free-flyer root + 51 sphericals), built
        once at the person's fixed betas (:func:`tools.smplx_robot.make_smplx_robot_model`).
    mid
        The ``better_human`` :class:`SMPLXMid` (for ``num_joints`` + the q remap).
    q_bh
        ``(T, 211)`` better_human configuration trajectory (renormalized here).
    fps
        Clip frame rate (``q̇`` is a per-second tangent difference).

    Returns
    -------
    speed : ``(T, J)``
        The norm of the LOCAL linear (translational) 3-vector of each joint twist.
        Forward-difference in time — frame ``t`` carries the outgoing speed
        ``t → t+1``; the final frame repeats the penultimate.
    """
    T = int(q_bh.shape[0])
    J = int(mid.num_joints)
    if T < 2:
        return q_bh.new_zeros(T, J)

    q_model = remap_q_bh_to_model(normalize_q(q_bh, J), mid, model)   # (T, nq)
    data = forward_kinematics(model, q_model)
    compute_joint_jacobians(model, data)
    Jw = data.joint_jacobians                                        # (T, J, 6, nv) [lin; ang]
    Ad_inv = se3.adjoint_inv(data.joint_pose_world)                  # (T, J, 6, 6)
    J_local = Ad_inv @ Jw                                            # (T, J, 6, nv) local twist Jacobian

    qdot = model.difference(q_model[:-1], q_model[1:]) * float(fps)  # (T-1, nv) tangent velocity
    v_local = (J_local[:-1] @ qdot[:, None, :, None]).squeeze(-1)    # (T-1, J, 6)
    speed = v_local[..., :3].norm(dim=-1)                            # (T-1, J) local linear speed
    return torch.cat([speed, speed[-1:]], dim=0)                     # pad to (T, J)
