"""The body of the pipeline: BetterHuman's SMPL-X with its 35 shipped contact frames.

A body is three BetterHuman objects: the ``HumanModel`` (:func:`load`, once per process), an
``Identity`` per person (:func:`bake`) and the ``HumanState`` of a trajectory
(``identity.forward(q)``), which reports joints, vertices and frames. ``q (P, N, 211)`` holds, per
person and frame, the root pose in the world (translation, quaternion) and 51 joint quaternions.
A contact frame is a named point of the skin whose z axis is the outward normal there; contacts
are detected, labelled and loaded on these frames. This module adds what BetterHuman leaves to
the pipeline: the contact frames of a state, the step from a camera to the world, and the body file.
"""
from __future__ import annotations

from pathlib import Path

import better_human as bh
import numpy as np
import torch
from better_robot.lie import se3, so3

from bvr import files

#: SMPL-X joints 0 to 21 are the body; the 30 after them are the fingers, the tail of ``q``.
BODY_JOINTS = 22


def load(device: torch.device | str) -> bh.HumanModel:
    """The neutral SMPL-X body with 10 shape coefficients, no face, and the shipped contact frames."""
    spec = bh.families.smplx(num_betas=10, face=False, device=device)
    return bh.HumanModel.from_spec(spec, contact_frames=bh.io.shipped_frame_set("smplx_contacts"))


def bake(human: bh.HumanModel, betas: torch.Tensor) -> bh.Identity:
    """One baked body per person from ``betas (P, 10)``, with batch ``(P, 1)`` so that ``q (P, N, 211)`` poses it."""
    return human.bake(betas=betas[:, None])


def contact_frames(state: bh.HumanState) -> torch.Tensor:
    """World poses ``(..., F, 7)`` of the contact frames, in the order of ``human.contact_frames.names``."""
    ids = torch.as_tensor(state.identity.human.contact_frames.frame_ids, device=state.q.device)
    return state.frame_poses.index_select(-2, ids)


def frame_normals(frames: torch.Tensor) -> torch.Tensor:
    """Outward skin normals ``(..., F, 3)`` of contact-frame poses ``(..., F, 7)``: each frame's z axis."""
    return so3.to_matrix(frames[..., 3:])[..., 2]


def to_world(q: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """A trajectory ``(..., T, 211)`` given in the camera frames of ``extrinsics (T, 4, 4)``, in the world.

    Only the root pose changes; the joint quaternions are relative to their parents.
    """
    root = se3.compose(se3.inverse(se3.from_matrix(extrinsics)), q[..., :7])
    return torch.cat((root, q[..., 7:]), dim=-1)


def save(path: Path, format: str, identity: bh.Identity, q: torch.Tensor, **arrays: np.ndarray) -> None:
    """Write a body file: ``q (P, N, 211)``, its classic SMPL-X parameters, the skeleton, and ``arrays``.

    The classic parameters are what the ``smplx`` package takes; the face ones are zero, because this body has no
    face joints.
    """
    spec, classic = identity.human.spec, bh.to_classic(identity, q)
    people, frames = q.shape[:2]
    pose = {name: getattr(classic, name).cpu().numpy()
            for name in ("global_orient", "transl", "body_pose", "left_hand_pose", "right_hand_pose")}
    face = {name: np.zeros((people, frames, width), np.float32)
            for name, width in (("jaw_pose", 3), ("leye_pose", 3), ("reye_pose", 3), ("expression", 10))}
    files.save_npz(
        path, format, q=q.cpu().numpy(), betas=identity.values["betas"][:, 0].cpu().numpy(), **pose, **face,
        scale=np.ones(people, np.float32), joint_names=np.array(spec.joint_names),
        parents=np.array(spec.parents, np.int32), contact_frame_names=np.array(identity.human.contact_frames.names),
        num_people=np.int32(people), num_frames=np.int32(frames), num_betas=np.int32(10),
        frame=np.array("world"), model_type=np.array("smplx_mid"), **arrays)
