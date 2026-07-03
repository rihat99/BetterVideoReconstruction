"""``q`` manifold utilities for the SMPL-X kinematic optimizer.

The optimizer treats ``q (T, nq)`` as a free leaf tensor (no exp retraction —
that has a zero-tangent autograd singularity in BetterRobot's Lie backend). The
quaternion blocks therefore drift off the unit sphere between steps and must be
re-normalized *inside* the forward pass. Normalizing by division is fully
autograd-clean (no singularity at unit norm).
"""

from __future__ import annotations

import torch


def normalize_q(q: torch.Tensor, num_joints: int, eps: float = 1e-8) -> torch.Tensor:
    """Return ``q`` with the root and per-joint quaternions renormalized.

    ``q[..., 3:7]`` is the root SO(3) quaternion; ``q[..., 7:]`` packs
    ``num_joints - 1`` parent-local quaternions (stride 4). Translations
    (``q[..., :3]``) pass through unchanged.
    """
    shape = q.shape[:-1]
    root_t = q[..., :3]
    root_quat = q[..., 3:7]
    joints = q[..., 7:].reshape(*shape, num_joints - 1, 4)

    root_quat = root_quat / root_quat.norm(dim=-1, keepdim=True).clamp_min(eps)
    joints = joints / joints.norm(dim=-1, keepdim=True).clamp_min(eps)

    return torch.cat([root_t, root_quat, joints.reshape(*shape, (num_joints - 1) * 4)], dim=-1)
