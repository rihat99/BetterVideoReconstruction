"""SMPL per-vertex contacts → per-joint contacts.

Reads an InteractVLM-style contact ``.npz`` (per-frame, per-vertex confidence)
and collapses it to a per-joint confidence matrix aligned with the 24 SMPL
joints used by ``better_robot.io.builders.JOINT_NAMES``. Vertex→joint assignment
uses ``smpl_model.lbs_weights`` (argmax), the canonical SMPL LBS matrix from
``SMPL_NEUTRAL.npz["weights"]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from better_human.smpl.model.smpl import SMPL


_SMPL_NUM_VERTS = 6890


def smpl_vertex_to_joint(smpl_model: "SMPL") -> np.ndarray:
    """Return ``(V,) int64`` — dominant joint per SMPL vertex.

    Argmax of the LBS weight matrix ``smpl_model.lbs_weights (V, J)`` picks
    the single most-influential joint per vertex. Indices are 0..J-1 in the
    canonical SMPL order (pelvis, left_hip, right_hip, ...), matching
    ``better_robot.io.builders.JOINT_NAMES``.
    """
    w = smpl_model.lbs_weights.detach().cpu().numpy()
    return np.argmax(w, axis=1).astype(np.int64)


def aggregate_vertex_contacts_to_joints(
    vertex_contact: np.ndarray,
    vertex_to_joint: np.ndarray,
    *,
    num_joints: int = 24,
    reduce: Literal["max", "mean"] = "max",
) -> np.ndarray:
    """Reduce per-vertex contacts ``(M, V) float`` to per-joint ``(M, J) float32``.

    ``reduce="max"`` flags a joint in-contact when any of its vertices is hot —
    preserves small patches (e.g. a single fingertip). ``"mean"`` averages the
    joint's vertex patch and yields a smoother signal.
    """
    if vertex_contact.ndim != 2:
        raise ValueError(
            f"expected vertex_contact of shape (M, V), got {vertex_contact.shape}"
        )
    if vertex_contact.shape[1] != vertex_to_joint.shape[0]:
        raise ValueError(
            f"vertex count mismatch: contact has {vertex_contact.shape[1]}, "
            f"vertex→joint map has {vertex_to_joint.shape[0]}"
        )

    m = vertex_contact.shape[0]
    out = np.zeros((m, num_joints), dtype=np.float32)
    conf = vertex_contact.astype(np.float32, copy=False)

    for j in range(num_joints):
        idx = np.where(vertex_to_joint == j)[0]
        if idx.size == 0:
            continue
        patch = conf[:, idx]
        if reduce == "max":
            out[:, j] = patch.max(axis=1)
        elif reduce == "mean":
            out[:, j] = patch.mean(axis=1)
        else:
            raise ValueError(f"unknown reduce={reduce!r}; expected 'max' or 'mean'")
    return out


def align_to_frames(
    joint_contact_m: np.ndarray,
    frame_indices: np.ndarray | None,
    num_frames: int,
) -> np.ndarray:
    """Pad a sparse ``(M, J)`` matrix (rows indexed by ``frame_indices``) to a
    dense ``(num_frames, J) float32``. Missing frames are zero-filled;
    out-of-range frame ids are skipped.
    """
    m, j = joint_contact_m.shape
    if frame_indices is None:
        fidx = np.arange(m, dtype=np.int64)
    else:
        fidx = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if fidx.shape[0] != m:
            raise ValueError(
                f"frame_indices length {fidx.shape[0]} != contact rows {m}"
            )

    out = np.zeros((num_frames, j), dtype=np.float32)
    for k, fi in enumerate(fidx):
        fi_i = int(fi)
        if 0 <= fi_i < num_frames:
            out[fi_i] = joint_contact_m[k]
    return out


def load_joint_contacts(
    npz_path: str | Path,
    smpl_model: "SMPL",
    num_frames: int,
    *,
    contact_key: str = "pred_contact_3d_smplh",
    reduce: Literal["max", "mean"] = "max",
    num_joints: int = 24,
) -> np.ndarray:
    """Load an InteractVLM ``.npz``, collapse to per-joint, align to frames.

    Returns ``(num_frames, num_joints) float32`` in ``[0, 1]``. Raises if
    ``contact_key`` is missing or does not have SMPL's 6890-vertex shape.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    if contact_key not in data.files:
        raise KeyError(
            f"{contact_key!r} not in {npz_path} — available keys: {list(data.files)}"
        )

    conf = np.asarray(data[contact_key], dtype=np.float32)
    if conf.ndim != 2 or conf.shape[1] != _SMPL_NUM_VERTS:
        raise ValueError(
            f"{contact_key!r} has shape {conf.shape}; expected "
            f"(M, {_SMPL_NUM_VERTS}). The 24-joint SMPL robot only supports "
            f"SMPL / SMPL-H vertex contacts — not SMPL-X's 10475-vert format."
        )

    fidx = data["frame_indices"] if "frame_indices" in data.files else None

    vert_to_joint = smpl_vertex_to_joint(smpl_model)
    per_joint_m = aggregate_vertex_contacts_to_joints(
        conf, vert_to_joint, num_joints=num_joints, reduce=reduce
    )
    return align_to_frames(per_joint_m, fidx, num_frames)


# Manual-label → SMPL joint-index mapping. Feet populate both the ankle
# and the distal-foot joints; hands populate the distal hand joint only.
_MANUAL_CONTACT_JOINTS: dict[str, tuple[int, ...]] = {
    "left_hand":  (22,),
    "right_hand": (23,),
    "left_foot":  (7, 10),
    "right_foot": (8, 11),
}


def load_manual_joint_contacts(
    npz_path: str | Path,
    num_frames: int,
    *,
    num_joints: int = 24,
) -> np.ndarray:
    """Load a hand-labeled contact ``.npz`` and expand to per-joint.

    The ``.npz`` must contain four per-frame 0/1 arrays — ``left_hand``,
    ``right_hand``, ``left_foot``, ``right_foot`` — each of shape ``(M,)``
    where ``M == num_frames``. Hand labels set SMPL joint 22/23; foot
    labels set both the ankle (7/8) and distal-foot (10/11) joints. All
    other joints remain zero.

    Returns ``(num_frames, num_joints) float32`` with values in ``{0.0, 1.0}``.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    missing = [k for k in _MANUAL_CONTACT_JOINTS if k not in data.files]
    if missing:
        raise KeyError(
            f"manual contact keys {missing} not in {npz_path} — "
            f"available keys: {list(data.files)}"
        )

    out = np.zeros((num_frames, num_joints), dtype=np.float32)
    for label, joint_indices in _MANUAL_CONTACT_JOINTS.items():
        vec = np.asarray(data[label], dtype=np.float32).reshape(-1)
        if vec.shape[0] != num_frames:
            raise ValueError(
                f"manual contact {label!r} has shape {data[label].shape}; "
                f"expected ({num_frames},)"
            )
        for j in joint_indices:
            out[:, j] = np.maximum(out[:, j], vec)
    return out
