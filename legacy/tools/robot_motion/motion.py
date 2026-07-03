"""Motion helpers: average betas, axis-convention fix, assemble a Trajectory."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pypose as pp
import torch

from better_robot.io.builders.smpl_like import PARENTS
from better_robot.lie import se3

if TYPE_CHECKING:
    from better_human.smpl.model.smpl import SMPL
    from better_robot.data_model.model import Model
    from better_robot.tasks.trajectory import Trajectory


def compute_smpl_joint_offsets(
    smpl_model: "SMPL",
    betas: torch.Tensor,
) -> torch.Tensor:
    """Return ``(24, 3)`` parent→child bone translations in the SMPL rest pose.

    Runs ``smpl_model.forward_shape(betas)`` to obtain the neutral-pose joint
    positions at the given shape parameters, then differences them along the
    SMPL parent indices to get each bone's translation offset. Output dtype /
    device match the SMPL model's template buffer.
    """
    betas_b = betas.view(1, -1).to(
        dtype=smpl_model.vertices_template.dtype,
        device=smpl_model.vertices_template.device,
    )
    with torch.no_grad():
        _, neutral_joints = smpl_model.forward_shape(betas_b)
    j = neutral_joints[0]  # (24, 3)

    offsets = torch.zeros((24, 3), dtype=j.dtype, device=j.device)
    for child in range(1, 24):
        offsets[child] = j[child] - j[PARENTS[child]]
    return offsets


def y_up_to_z_up(q: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Rotate SMPL-like ``q (N, 99)`` from camera frame to robot Z-up world.

    Camera Y points toward the ground (OpenCV convention); rotating by −90°
    about X sends camera +Y → world −Z, putting up on +Z. Only the SE3 root
    (``q[..., :7]``) is affected; joint quaternions are parent-local and
    unchanged. Input/output dtype matches (numpy in → numpy out).
    """
    is_numpy = isinstance(q, np.ndarray)
    q_t = torch.as_tensor(q, dtype=torch.float32) if is_numpy else q
    if q_t.ndim != 2 or q_t.shape[-1] < 7:
        raise ValueError(f"expected q of shape (N, >=7), got {tuple(q_t.shape)}")

    # Euler (roll, pitch, yaw) = (−90°, 0, 0): pure rotation about X.
    euler = torch.tensor([-math.pi / 2, 0.0, 0.0], dtype=q_t.dtype, device=q_t.device)
    t_cam_to_world = torch.zeros(7, dtype=q_t.dtype, device=q_t.device)
    t_cam_to_world[3:] = pp.euler2SO3(euler).tensor()

    out = q_t.clone()
    out[..., :7] = se3.compose(
        t_cam_to_world.expand_as(q_t[..., :7]),
        q_t[..., :7],
    )
    return out.cpu().numpy() if is_numpy else out


def remap_smpl_q_to_model(
    q: torch.Tensor | np.ndarray,
    model: "Model",
) -> torch.Tensor | np.ndarray:
    """Permute an SMPL-canonical ``q (..., 99)`` into BetterRobot model order.

    SMPL stores 23 spherical joints as a stride-4 block at ``q[..., 7:99]``
    in canonical order (left_hip, right_hip, spine1, ...). Since BetterRobot
    switched to DFS topological ordering, ``model.idx_qs`` places each joint
    in a different slot (e.g. under DFS, the slot at ``q[..., 11:15]`` holds
    left_knee, not right_hip). Feeding SMPL-ordered ``q`` straight into
    ``forward_kinematics`` therefore produces scrambled FK.

    This function permutes the 23 quaternion slots by body name. The
    7-element base pose at ``q[..., :7]`` is identical in both conventions
    and copies across unchanged.
    """
    from better_robot.io.builders import JOINT_NAMES

    if q.shape[-1] != 99:
        raise ValueError(f"expected q of shape (..., 99), got {tuple(q.shape)}")
    if model.nq != 99:
        raise ValueError(f"expected model.nq=99, got {model.nq}")

    is_numpy = isinstance(q, np.ndarray)
    out = np.zeros_like(q) if is_numpy else torch.zeros_like(q)

    out[..., :7] = q[..., :7]
    for smpl_j in range(1, 24):
        br_j = model.body_name_to_id.get(JOINT_NAMES[smpl_j])
        if br_j is None:
            continue
        iq_br = model.idx_qs[br_j]
        smpl_slot = 7 + 4 * (smpl_j - 1)
        out[..., iq_br : iq_br + 4] = q[..., smpl_slot : smpl_slot + 4]
    return out


def remap_model_q_to_smpl(
    q: torch.Tensor | np.ndarray,
    model: "Model",
) -> torch.Tensor | np.ndarray:
    """Inverse of :func:`remap_smpl_q_to_model`. See its docstring."""
    from better_robot.io.builders import JOINT_NAMES

    if q.shape[-1] != 99:
        raise ValueError(f"expected q of shape (..., 99), got {tuple(q.shape)}")
    if model.nq != 99:
        raise ValueError(f"expected model.nq=99, got {model.nq}")

    is_numpy = isinstance(q, np.ndarray)
    out = np.zeros_like(q) if is_numpy else torch.zeros_like(q)

    out[..., :7] = q[..., :7]
    for smpl_j in range(1, 24):
        br_j = model.body_name_to_id.get(JOINT_NAMES[smpl_j])
        if br_j is None:
            continue
        iq_br = model.idx_qs[br_j]
        smpl_slot = 7 + 4 * (smpl_j - 1)
        out[..., smpl_slot : smpl_slot + 4] = q[..., iq_br : iq_br + 4]
    return out


def remap_smpl_contacts_to_model(
    arr: np.ndarray,
    model: "Model",
) -> np.ndarray:
    """Permute an ``(T, 24)`` SMPL-canonical per-body array into BR DFS order.

    Column ``j`` of the input corresponds to SMPL joint ``JOINT_NAMES[j]``.
    The output is ``(T, 24)`` with column ``i`` matching BR body
    ``model.body_names[i + 1]`` (DFS order, universe excluded). This is the
    layout consumed by :func:`optimize_motion` when indexing into saved
    ``joint_contact`` arrays by BR body id.
    """
    from better_robot.io.builders import JOINT_NAMES

    if arr.ndim != 2 or arr.shape[1] != 24:
        raise ValueError(f"expected arr of shape (T, 24), got {tuple(arr.shape)}")

    smpl_idx = {name: k for k, name in enumerate(JOINT_NAMES)}
    T = arr.shape[0]
    out = np.zeros((T, model.nbodies - 1), dtype=arr.dtype)
    for i in range(model.nbodies - 1):
        j = smpl_idx.get(model.body_names[i + 1])
        if j is not None:
            out[:, i] = arr[:, j]
    return out


def remap_model_contacts_to_smpl(
    arr: np.ndarray,
    model: "Model",
) -> np.ndarray:
    """Inverse of :func:`remap_smpl_contacts_to_model`."""
    from better_robot.io.builders import JOINT_NAMES

    if arr.ndim != 2 or arr.shape[1] != (model.nbodies - 1):
        raise ValueError(
            f"expected arr of shape (T, {model.nbodies - 1}), got {tuple(arr.shape)}"
        )

    smpl_idx = {name: k for k, name in enumerate(JOINT_NAMES)}
    T = arr.shape[0]
    out = np.zeros((T, 24), dtype=arr.dtype)
    for i in range(model.nbodies - 1):
        j = smpl_idx.get(model.body_names[i + 1])
        if j is not None:
            out[:, j] = arr[:, i]
    return out


def ground_centered_trajectory(traj: "Trajectory", model: "Model") -> "Trajectory":
    """Re-anchor the trajectory for display: pelvis at frame 0 sits at XY=0
    and the lowest joint across the whole motion sits at Z=0.

    This is a world-frame translation only — joint angles, bone offsets, and
    root orientation are unchanged. Intended for visualization; downstream
    kinematics are identical to the input.
    """
    from better_robot import forward_kinematics
    from better_robot.tasks.trajectory import Trajectory

    model_device = model.joint_placements.device
    q = traj.q.to(model_device)
    if q.ndim != 3 or q.shape[0] != 1:
        raise ValueError(f"expected q of shape (1, T, nq), got {tuple(q.shape)}")

    data = forward_kinematics(model, q[0])            # oMi: (T, njoints, 7)
    z_min = float(data.oMi[..., 2].min())
    x0 = float(q[0, 0, 0])
    y0 = float(q[0, 0, 1])

    q_shifted = q.clone()
    q_shifted[..., :3] -= torch.tensor(
        [x0, y0, z_min], dtype=q.dtype, device=q.device
    )
    return Trajectory(t=traj.t.to(model_device), q=q_shifted)


def average_betas(betas: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Return the mean of ``betas`` over valid frames, shape ``(10,)``."""
    if valid_mask.sum() == 0:
        raise ValueError("no valid frames — cannot average betas")
    return betas[valid_mask].mean(axis=0).astype(np.float32)


def _forward_fill(q: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Replace invalid-frame q rows with the most recent valid row.

    If the clip starts with invalid frames, fill them backward from the
    first valid frame. Guarantees the returned array has no NaNs.
    """
    if valid_mask.all():
        return q.copy()

    first_valid = int(np.argmax(valid_mask))
    filled = q.copy()

    # Backfill the leading invalid window with the first valid q.
    if first_valid > 0:
        filled[:first_valid] = q[first_valid]

    # Forward-fill anything after the first valid index.
    last = q[first_valid]
    for i in range(first_valid, len(q)):
        if valid_mask[i]:
            last = q[i]
        else:
            filled[i] = last
    return filled


def build_trajectory(
    q: np.ndarray,
    valid_mask: np.ndarray,
    fps: float,
) -> "Trajectory":
    """Pack ``q (N, 99)`` into a BetterRobot ``Trajectory(1, N, 99)``.

    Invalid frames are forward-filled so the Trajectory contains no NaNs.
    """
    from better_robot.tasks.trajectory import Trajectory

    q_filled = _forward_fill(q, valid_mask).astype(np.float32)
    n = q_filled.shape[0]
    t = np.arange(n, dtype=np.float32) / float(fps)

    return Trajectory(
        t=torch.from_numpy(t).unsqueeze(0),
        q=torch.from_numpy(q_filled).unsqueeze(0),
    )


