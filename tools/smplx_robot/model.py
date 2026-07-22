"""Build a BetterRobot model for the SMPL-X-mid body + q remapping.

Computes the 52-joint :class:`SMPLXMid` joint offsets and remaps ``q`` between
``better_human`` kintree order and BetterRobot's DFS order. The robot model
exists purely so the kinematic stage can measure acceleration / jerk smoothness
with BetterRobot's exact Lie ``model.difference``; the posed mesh and joints
used by the visual losses come from ``better_human`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from better_human.smpl import SMPLXMid
    from better_robot.data_model.model import Model


def load_smplx_mid(model_path: str | Path | None = None, device: str | torch.device = "cpu"):
    """Load the 52-joint SMPL-X body used across the pipeline.

    The body now comes from the rewritten BetterHuman via :func:`tools.body.load_body`
    (a ``use_face=False`` :class:`better_human.SMPLX`, which reproduces the old
    ``SMPLXMid`` 52-joint / ``q (211)`` surface — same ``forward`` / ``forward_shape``
    / ``from_classic`` / ``to_classic`` and topology attributes). ``model_path`` is
    kept for call-site compatibility but ignored: BetterHuman bundles its own
    validated ``SMPLX_NEUTRAL.npz`` (the old ``../better_human`` weights are gone).
    """
    from tools.body import load_body

    return load_body(device=device)


def _parents_from_model(smplx_mid: "SMPLXMid") -> list[int]:
    """Return ``(52,)`` parent indices in ``better_human`` joint order (-1 root)."""
    parent_row = smplx_mid.parent_tree[0]
    return [-1] + [int(parent_row[i]) for i in range(1, smplx_mid.num_joints)]


def compute_smplx_joint_offsets(smplx_mid: "SMPLXMid", betas: torch.Tensor) -> torch.Tensor:
    """Return ``(52, 3)`` parent→child bone translations in the rest pose.

    Runs ``forward_shape(betas)`` for the neutral-pose joint positions, then
    differences them along the SMPL-X-mid parent indices. The root row is zero
    (a free-flyer carries root placement in ``q``).
    """
    betas_b = betas.view(1, -1).to(
        dtype=smplx_mid.vertices_template.dtype,
        device=smplx_mid.vertices_template.device,
    )
    with torch.no_grad():
        _, neutral_joints = smplx_mid.forward_shape(betas_b)
    j = neutral_joints[0]  # (52, 3)

    parents = _parents_from_model(smplx_mid)
    offsets = torch.zeros_like(j)
    for child in range(1, smplx_mid.num_joints):
        offsets[child] = j[child] - j[parents[child]]
    return offsets


def make_smplx_robot_model(
    smplx_mid: "SMPLXMid",
    betas: torch.Tensor,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> "Model":
    """Build a frozen BetterRobot ``Model`` for the SMPL-X-mid body at ``betas``.

    Free-flyer root + 51 spherical joints, bone offsets from the shaped rest
    pose. ``mass_per_body=1.0`` (inertia is unused — this stage is kinematic).
    """
    from better_robot.io.builders import build_kinematic_tree_model

    offsets = compute_smplx_joint_offsets(smplx_mid, betas).to(dtype=dtype)
    return build_kinematic_tree_model(
        name="smplx_mid_body",
        joint_names=tuple(smplx_mid.joint_names),
        parents=tuple(_parents_from_model(smplx_mid)),
        translations=offsets,
        root_kind="free_flyer",
        child_kind="spherical",
        mass_per_body=1.0,
        device=device,
        dtype=dtype,
    )


def remap_q_bh_to_model(
    q: torch.Tensor,
    smplx_mid: "SMPLXMid",
    model: "Model",
) -> torch.Tensor:
    """Permute ``better_human`` ``q (..., 211)`` into BetterRobot DFS order.

    ``better_human`` stores the 51 spherical joints as a stride-4 block in
    kintree order (``q[..., 7:]``); BetterRobot reorders bodies depth-first, so
    each joint lands at ``model.idx_qs[body_id]``. The 7-element root pose
    (``q[..., :7]``) is identical in both conventions and copies across.
    """
    if q.shape[-1] != smplx_mid.nq:
        raise ValueError(f"expected q (..., {smplx_mid.nq}), got {tuple(q.shape)}")
    out = torch.zeros_like(q)
    out[..., :7] = q[..., :7]
    for j in range(1, smplx_mid.num_joints):
        name = smplx_mid.joint_names[j]
        br_j = model.body_name_to_id[name]
        iq = model.idx_qs[br_j]
        bh = 7 + 4 * (j - 1)
        out[..., iq : iq + 4] = q[..., bh : bh + 4]
    return out


def remap_q_model_to_bh(
    q: torch.Tensor,
    smplx_mid: "SMPLXMid",
    model: "Model",
) -> torch.Tensor:
    """Inverse of :func:`remap_q_bh_to_model`."""
    if q.shape[-1] != smplx_mid.nq:
        raise ValueError(f"expected q (..., {smplx_mid.nq}), got {tuple(q.shape)}")
    out = torch.zeros_like(q)
    out[..., :7] = q[..., :7]
    for j in range(1, smplx_mid.num_joints):
        name = smplx_mid.joint_names[j]
        br_j = model.body_name_to_id[name]
        iq = model.idx_qs[br_j]
        bh = 7 + 4 * (j - 1)
        out[..., bh : bh + 4] = q[..., iq : iq + 4]
    return out
