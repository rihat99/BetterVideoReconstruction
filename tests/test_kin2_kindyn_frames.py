"""The frames-era kin2 / kindyn_1 chain: frame labels, the label loader, the force wrenches.

Three things this file pins down, all of which changed when contact moved off the 52
joints and onto the 12 contact frames:

* ``annotation_frame_labels`` — the manual per-joint annotation spread onto the frames
  each joint covers (and the 10- vs 14-joint schema difference on the heel);
* ``load_frame_contacts`` — the AUTO label source refuses a stale or mismatched
  ``contacts_<tag>.npz`` instead of silently refining against the wrong body;
* :class:`~tools.human_optim.kindyn.DynamicsSpec` — a contact-frame force becomes the
  right wrench on its parent joint (force + lever moment, accumulated when two frames
  share a joint), and the friction pyramid cannot be left.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import torch

from tools.human_optim.forces import cone_generators
from tools.human_optim.kindyn import DynamicsSpec
from tools.human_optim.stages import annotation_frame_labels, load_frame_contacts

# The 12 contact frames in their stored order — note slots 10/11 are RIGHT then LEFT.
_FRAME_NAMES = ["toe_under_left", "toe_under_right", "heel_left", "heel_right",
                "toe_above_left", "toe_above_right", "knee_left", "knee_right",
                "palm_left", "palm_right", "finger_right", "finger_left"]

_ANN_10 = ["left_hand", "right_hand", "left_foot", "right_foot", "left_knee",
           "right_knee", "left_hip", "right_hip", "left_elbow", "right_elbow"]
_ANN_14 = ["left_hand", "right_hand", "left_foot", "right_foot", "left_ankle",
           "right_ankle", "left_knee", "right_knee", "left_elbow", "right_elbow",
           "left_shoulder", "right_shoulder", "left_hip", "right_hip"]


def _column(labels: np.ndarray, name: str) -> np.ndarray:
    return labels[:, _FRAME_NAMES.index(name)]


# ───────────────────────── annotation → contact frames ─────────────────────────

def test_hand_annotation_reaches_both_palm_and_finger():
    """One ``left_hand`` label covers the palm AND the fingertip frame of that hand."""
    contacts = np.full((len(_ANN_10), 4), -1, np.int8)
    contacts[_ANN_10.index("left_hand")] = np.array([1, 1, 0, -1], np.int8)
    labels = annotation_frame_labels(contacts, _ANN_10, _FRAME_NAMES, 4)
    for name in ("palm_left", "finger_left"):
        assert _column(labels, name).tolist() == [1, 1, 0, -1]
    for name in ("palm_right", "finger_right"):
        assert _column(labels, name).tolist() == [-1] * 4   # untouched by a left-hand label


def test_ten_joint_foot_covers_the_heel_but_fourteen_joint_does_not():
    """"foot" means the whole foot in the 10-joint schema and only the toe in the 14-joint one."""
    ten = np.full((len(_ANN_10), 3), -1, np.int8)
    ten[_ANN_10.index("left_foot")] = np.array([1, 1, 1], np.int8)
    labels_10 = annotation_frame_labels(ten, _ANN_10, _FRAME_NAMES, 3)
    assert _column(labels_10, "heel_left").tolist() == [1, 1, 1]

    fourteen = np.full((len(_ANN_14), 3), -1, np.int8)
    fourteen[_ANN_14.index("left_foot")] = np.array([1, 1, 1], np.int8)
    fourteen[_ANN_14.index("left_ankle")] = np.array([0, 0, 0], np.int8)
    labels_14 = annotation_frame_labels(fourteen, _ANN_14, _FRAME_NAMES, 3)
    assert _column(labels_14, "heel_left").tolist() == [0, 0, 0]      # the ankle label owns it
    for name in ("toe_under_left", "toe_above_left"):
        assert _column(labels_14, name).tolist() == [1, 1, 1]


def test_joints_with_no_contact_frame_leave_every_frame_unlabeled():
    """hip / elbow / shoulder carry no contact frame, so annotating them changes nothing."""
    contacts = np.full((len(_ANN_14), 2), -1, np.int8)
    for name in ("left_hip", "right_elbow", "left_shoulder"):
        contacts[_ANN_14.index(name)] = 1
    labels = annotation_frame_labels(contacts, _ANN_14, _FRAME_NAMES, 2)
    assert (labels == -1).all()


def test_frame_count_mismatch_raises():
    contacts = np.full((len(_ANN_10), 5), -1, np.int8)
    with pytest.raises(ValueError, match="5 frames but the body has 4"):
        annotation_frame_labels(contacts, _ANN_10, _FRAME_NAMES, 4)


def test_unknown_annotation_schema_raises():
    with pytest.raises(ValueError, match="unrecognized contact annotation schema"):
        annotation_frame_labels(np.zeros((2, 3), np.int8), ["left_hand", "right_hand"],
                                _FRAME_NAMES, 3)


# ───────────────────────── the AUTO label source ─────────────────────────

def _chain_tree(tmp_path, *, num_frames=4, frame_names=None, drop_frame_contact=False):
    """A minimal ``out/<scene>/`` with a body, a scene, a transform and matching contacts."""
    pred = tmp_path / "scene"
    (pred / "human_optim").mkdir(parents=True)
    (pred / "geometry").mkdir()
    body_path = pred / "human_optim" / "kinematic_1.npz"
    np.savez(body_path, q=np.zeros((1, num_frames, 211), np.float32))
    for name in ("scene", "transform"):
        np.savez(pred / "geometry" / f"{name}.npz", payload=np.arange(3, dtype=np.float32))
    digest = {}
    for key, path in (("source_body_sha256", body_path),
                      ("source_scene_sha256", pred / "geometry" / "scene.npz"),
                      ("source_transform_sha256", pred / "geometry" / "transform.npz")):
        with path.open("rb") as handle:
            digest[key] = np.str_(hashlib.file_digest(handle, "sha256").hexdigest())
    names = _FRAME_NAMES if frame_names is None else frame_names
    payload = dict(object_ids=np.array([0], np.int32),
                   contact_frame_names=np.array(names), **digest)
    if not drop_frame_contact:
        payload["frame_contact"] = np.zeros((1, num_frames, len(names)), bool)
    np.savez(pred / "human_optim" / "contacts_1.npz", **payload)
    return pred, body_path


def test_load_frame_contacts_accepts_a_matching_artifact(tmp_path):
    pred, body_path = _chain_tree(tmp_path)
    ids, contact = load_frame_contacts(pred, "1", body_path, 4, _FRAME_NAMES, "test")
    assert ids == [0]
    assert contact.shape == (1, 4, 12) and contact.dtype == bool


def test_load_frame_contacts_rejects_a_stale_artifact(tmp_path):
    """Editing the body after the contacts were measured must not go unnoticed."""
    pred, body_path = _chain_tree(tmp_path)
    np.savez(body_path, q=np.ones((1, 4, 211), np.float32))     # the body moved
    with pytest.raises(SystemExit, match="is stale for"):
        load_frame_contacts(pred, "1", body_path, 4, _FRAME_NAMES, "test")


def test_load_frame_contacts_rejects_a_different_frame_set(tmp_path):
    shuffled = list(reversed(_FRAME_NAMES))
    pred, body_path = _chain_tree(tmp_path, frame_names=shuffled)
    with pytest.raises(SystemExit, match="contact-frame order does not match"):
        load_frame_contacts(pred, "1", body_path, 4, _FRAME_NAMES, "test")


def test_load_frame_contacts_rejects_a_per_joint_era_artifact(tmp_path):
    pred, body_path = _chain_tree(tmp_path, drop_frame_contact=True)
    with pytest.raises(SystemExit, match="carries no frame_contact"):
        load_frame_contacts(pred, "1", body_path, 4, _FRAME_NAMES, "test")


def test_load_frame_contacts_rejects_a_wrong_frame_count(tmp_path):
    pred, body_path = _chain_tree(tmp_path, num_frames=4)
    with pytest.raises(SystemExit, match="frame_contact must be"):
        load_frame_contacts(pred, "1", body_path, 7, _FRAME_NAMES, "test")


# ───────────────────────── contact-frame forces → RNEA wrenches ─────────────────────────

class _StubModel:
    """Just enough of a BetterRobot ``Model`` for :meth:`DynamicsSpec.build_fext`."""

    njoints = 3


def _spec(parent_ids, *, num_steps=2, cone_axis=None, mu=1.0) -> DynamicsSpec:
    count = len(parent_ids)
    mask = torch.ones(num_steps, count)
    gen = None if cone_axis is None else cone_generators(cone_axis, mu)
    return DynamicsSpec(
        model=_StubModel(), frame_ids=torch.arange(count),
        parent_ids=torch.as_tensor(parent_ids, dtype=torch.long),
        parent_body=list(range(count)), frame_names=[f"f{i}" for i in range(count)],
        mask=mask, fpair=mask[1:] * mask[:-1], valid_pair=torch.ones(num_steps - 1),
        dt=1 / 30, weights={}, dyn_opt={}, gravity_world=np.zeros(3, np.float32),
        valid=np.ones(num_steps, bool), total_mass=70.0,
        cone_axis=cone_axis, cone_gen=None if gen is None else gen.detach(), mu=mu)


def _identity_joints(num_steps: int, positions: torch.Tensor) -> torch.Tensor:
    """``(T, njoints, 7)`` placements with identity rotation at the given positions."""
    placements = torch.zeros(num_steps, positions.shape[0], 7)
    placements[..., :3] = positions
    placements[..., 6] = 1.0                       # scalar-last unit quaternion
    return placements


def test_frame_force_becomes_force_plus_lever_moment_on_its_parent_joint():
    """A world force at a contact frame lands on the parent joint as ``[f, r × f]``."""
    spec = _spec([1])
    joints = _identity_joints(2, torch.tensor([[0.0, 0.0, 0.0],
                                               [1.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0]]))
    frames = torch.zeros(2, 1, 7)
    frames[..., :3] = torch.tensor([1.0, 0.2, 0.0])          # 0.2 m out along +y from joint 1
    force = torch.tensor([0.0, 0.0, 30.0]).expand(2, 1, 3).contiguous()
    fext, f_eff = spec.build_fext(force, joints, frames)
    assert torch.allclose(f_eff, force)
    assert torch.allclose(fext[:, 1, :3], force[:, 0])                       # linear part
    expected = torch.linalg.cross(torch.tensor([0.0, 0.2, 0.0]), force[0, 0])
    assert torch.allclose(fext[:, 1, 3:], expected.expand(2, 3), atol=1e-6)  # r × f
    assert torch.allclose(fext[:, 0], torch.zeros(2, 6))                     # nothing elsewhere
    assert torch.allclose(fext[:, 2], torch.zeros(2, 6))


def test_two_frames_on_one_joint_accumulate_rather_than_overwrite():
    """The two toes of a foot share a parent joint — both forces must land, not the last one."""
    spec = _spec([1, 1])
    joints = _identity_joints(2, torch.zeros(3, 3))
    frames = torch.zeros(2, 2, 7)
    frames[:, 0, :3] = torch.tensor([0.1, 0.0, 0.0])
    frames[:, 1, :3] = torch.tensor([0.0, 0.1, 0.0])
    force = torch.zeros(2, 2, 3)
    force[:, 0] = torch.tensor([0.0, 0.0, 10.0])
    force[:, 1] = torch.tensor([0.0, 0.0, 4.0])
    fext, _ = spec.build_fext(force, joints, frames)
    assert torch.allclose(fext[:, 1, :3], torch.tensor([0.0, 0.0, 14.0]).expand(2, 3))
    moment = (torch.linalg.cross(frames[0, 0, :3], force[0, 0])
              + torch.linalg.cross(frames[0, 1, :3], force[0, 1]))
    assert torch.allclose(fext[:, 1, 3:], moment.expand(2, 3), atol=1e-6)


def test_a_rotated_parent_joint_expresses_the_wrench_in_its_own_frame():
    """RNEA wants the wrench in the joint's LOCAL frame, so the joint's rotation must apply."""
    spec = _spec([0])
    angle = math.pi / 2                              # +90° about z: world x -> local -y
    joints = torch.zeros(1, 1, 7)
    joints[..., 5] = math.sin(angle / 2)             # rows 3:7 are the quaternion (x, y, z, w)
    joints[..., 6] = math.cos(angle / 2)
    frames = torch.zeros(1, 1, 7)                    # frame sits at the joint origin
    force = torch.tensor([[[5.0, 0.0, 0.0]]])
    fext, _ = spec.build_fext(force, joints, frames)
    assert torch.allclose(fext[0, 0, :3], torch.tensor([0.0, -5.0, 0.0]), atol=1e-5)
    assert torch.allclose(fext[0, 0, 3:], torch.zeros(3), atol=1e-6)   # no lever, no moment


def test_masked_out_steps_carry_no_wrench():
    spec = _spec([1])
    spec.mask = torch.tensor([[1.0], [0.0]])
    joints = _identity_joints(2, torch.zeros(3, 3))
    frames = torch.zeros(2, 1, 7)
    frames[..., 1] = 0.3
    force = torch.tensor([0.0, 0.0, 12.0]).expand(2, 1, 3).contiguous()
    fext, f_eff = spec.build_fext(force, joints, frames)
    assert torch.allclose(f_eff[1], torch.zeros(1, 3))
    assert torch.allclose(fext[1], torch.zeros(3, 6))
    assert not torch.allclose(fext[0], torch.zeros(3, 6))


# ───────────────────────── the friction pyramid ─────────────────────────

def test_composed_forces_never_leave_the_friction_cone():
    """Any softplus activation of the four pyramid edges stays inside the true 45° cone."""
    torch.manual_seed(0)
    axis = torch.randn(5, 12, 3)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    spec = _spec(list(range(12)), num_steps=5, cone_axis=axis, mu=math.tan(math.radians(45.0)))
    activations = torch.randn(5, 12, 4) * 3.0
    force = spec.compose_forces(torch.zeros(5, 12, 3), activations)
    assert force.dtype == torch.float32
    assert float(spec.cone_violation(force).max()) < 1e-4
    # …and the force really is pushing along its axis, not merely inside a degenerate cone.
    assert float((force * axis).sum(-1).min()) > 0.0


def test_zero_activation_is_reachable_in_the_limit_and_never_negative():
    """softplus keeps every pyramid coefficient positive, so the cone constraint is strict."""
    axis = torch.tensor([[[0.0, 0.0, 1.0]]])
    spec = _spec([0], num_steps=1, cone_axis=axis, mu=1.0)
    tiny = spec.compose_forces(torch.zeros(1, 1, 3), torch.full((1, 1, 4), -30.0))
    assert float(tiny.norm()) < 1e-6
    assert float(spec.cone_violation(tiny).max()) < 1e-6


def test_cones_off_leaves_the_force_free():
    spec = _spec([0], num_steps=1)
    free = torch.tensor([[[1.0, -2.0, 3.0]]])
    assert torch.allclose(spec.compose_forces(free, None), free)
