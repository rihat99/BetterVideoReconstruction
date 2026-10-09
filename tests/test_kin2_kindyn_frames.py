"""The frames-era kin2 / kindyn_1 chain: frame labels, the label loader, the force wrenches.

Two things this file pins down, both of which changed when contact moved off the 52
joints and onto the contact frames:

* ``load_frame_contacts`` — the AUTO label source refuses a stale or mismatched
  ``contacts_<tag>.npz`` instead of silently refining against the wrong body;
* :class:`~bvr.human_optim.kindyn.DynamicsSpec` — a contact-frame force becomes the
  right wrench on its parent joint (force + lever moment, accumulated when two frames
  share a joint), and the friction pyramid cannot be left.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import torch

from bvr.human_optim.forces import cone_generators
from bvr.human_optim.kindyn import DynamicsSpec
from bvr.human_optim.stages import load_frame_contacts

# The legacy 12-frame climbing set in its stored order — note slots 10/11 are RIGHT then LEFT.
_FRAME_NAMES = ["toe_under_left", "toe_under_right", "heel_left", "heel_right",
                "toe_above_left", "toe_above_right", "knee_left", "knee_right",
                "palm_left", "palm_right", "finger_right", "finger_left"]

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


def test_force_split_order_one_matches_the_original_decomposition():
    force = torch.randn(6, 2, 3)
    d_par, d_perp = DynamicsSpec.force_split(force)
    diff = force[1:] - force[:-1]
    mean_dir = force[1:] + force[:-1]
    mean_dir = mean_dir / mean_dir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    assert torch.allclose(d_par + d_perp, diff, atol=1e-6)
    assert torch.allclose(d_par, (diff * mean_dir).sum(-1, keepdim=True) * mean_dir, atol=1e-6)


def test_higher_order_split_vanishes_on_a_linearly_ramping_force():
    steps = torch.arange(8, dtype=torch.float32)
    direction = torch.tensor([0.0, 0.0, 1.0])
    force = (10.0 + 3.0 * steps)[:, None, None] * direction    # (8, 1, 3) linear magnitude ramp
    par1, perp1 = DynamicsSpec.force_split(force, 1)
    par2, perp2 = DynamicsSpec.force_split(force, 2)
    par3, perp3 = DynamicsSpec.force_split(force, 3)
    assert par1.norm(dim=-1).allclose(torch.full((7, 1), 3.0))   # constant slope
    assert float(perp1.abs().max()) < 1e-5                       # direction never changes
    assert float(par2.abs().max()) < 1e-4 and float(perp2.abs().max()) < 1e-4
    assert float(par3.abs().max()) < 1e-4 and float(perp3.abs().max()) < 1e-4


def test_direction_jitter_lands_in_the_perp_part_at_every_order():
    signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])     # alternating small swing
    angles = 0.05 * signs
    force = 100.0 * torch.stack([angles.cos(), angles.sin(), torch.zeros(6)], dim=-1)[:, None, :]
    for order in (1, 2, 3):
        d_par, d_perp = DynamicsSpec.force_split(force, order)
        assert float(d_perp.norm(dim=-1).max()) > 10.0 * float(d_par.norm(dim=-1).max())


def test_persisting_masks_require_the_whole_window():
    mask = torch.tensor([[1.0], [1.0], [0.0], [1.0], [1.0], [1.0], [1.0]])
    spec = _spec([0], num_steps=7)
    spec.mask = mask
    spec.fpair = mask[1:] * mask[:-1]
    spec.__post_init__()
    assert spec.fpair[:, 0].tolist() == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert spec.ftrip[:, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert spec.fquad[:, 0].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_edge_mask_marks_the_first_and_last_frame_of_each_contact_interval():
    mask = torch.tensor([[1.0], [1.0], [0.0], [1.0], [1.0], [1.0], [0.0]])
    spec = _spec([0], num_steps=7)
    spec.mask = mask
    spec.fpair = mask[1:] * mask[:-1]
    spec.__post_init__()
    # a contact already held at the clip start is not an onset; the release before a
    # gap and the onset after it are edges; a one-frame-long contact counts once
    assert spec.fedge[:, 0].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    single = torch.tensor([[0.0], [1.0], [0.0]])
    spec = _spec([0], num_steps=3)
    spec.mask, spec.fpair = single, single[1:] * single[:-1]
    spec.__post_init__()
    assert spec.fedge[:, 0].tolist() == [0.0, 1.0, 0.0]


def test_edge_mask_ignores_a_switch_across_an_untracked_step():
    mask = torch.tensor([[1.0], [0.0], [1.0]])
    spec = _spec([0], num_steps=3)
    spec.mask, spec.fpair = mask, mask[1:] * mask[:-1]
    spec.valid_pair = torch.tensor([0.0, 0.0])
    spec.__post_init__()
    assert spec.fedge[:, 0].tolist() == [0.0, 0.0, 0.0]


def test_edge_weight_falls_with_the_body_speed_across_the_switch():
    mask = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    spec = _spec([0], num_steps=4)
    spec.mask, spec.fpair = mask, mask[1:] * mask[:-1]
    spec.gravity_world = np.array([0.0, -1.0, 0.0], np.float32)      # world up is +y
    spec.__post_init__()
    # the body drops 0.6 m/s into the onset at step 1 and rises at 1.8 m/s out of the
    # release at step 2
    root = torch.zeros(4, 3)
    root[:, 1] = torch.tensor([0.6 * spec.dt, 0.0, 0.0, 1.8 * spec.dt])
    spec.scale_edges_by_speed(root, speed_scale=0.6)
    assert spec.fedge[:, 0].tolist() == pytest.approx([0.0, 0.5, 0.1, 0.0])
    # horizontal travel is not a landing: the weight stays full
    spec.__post_init__()
    root = torch.zeros(4, 3)
    root[:, 0] = torch.arange(4.0)
    spec.scale_edges_by_speed(root, speed_scale=0.6)
    assert spec.fedge[:, 0].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_fps_scaling_treats_force_accel_and_jerk_as_orders_two_and_three():
    from bvr.config import fps_scaled_weights

    scaled = fps_scaled_weights({"force_smooth_mag": 1.0, "force_accel_dir": 1.0,
                                 "force_jerk_mag": 1.0, "base_wrench_ang": 1.0}, 60.0)
    assert scaled["force_smooth_mag"] == pytest.approx(4.0)
    assert scaled["force_accel_dir"] == pytest.approx(16.0)
    assert scaled["force_jerk_mag"] == pytest.approx(64.0)
    assert scaled["base_wrench_ang"] == pytest.approx(1.0)     # not a difference term


# ───────────────────────── per-joint torque multipliers ─────────────────────────

def test_joint_group_strips_side_and_index():
    from bvr.human_optim.kindyn import joint_group

    assert joint_group("left_hip") == "hip"
    assert joint_group("spine2") == "spine"
    assert joint_group("right_index3") == "index"
    assert joint_group("pelvis") == "pelvis"


def test_torque_multipliers_land_on_each_joints_dof_rows():
    from bvr.body import load_body
    from bvr.human_optim.kindyn import joint_group, torque_multipliers

    body = load_body(device="cpu")
    model = body.shaped(torch.zeros(10)).model
    table = {"wrist": 17.0, "index": 25.0}
    groups = {joint_group(n) for n in body.joint_names[1:]}
    table.update({g: 1.0 for g in groups if g not in table})
    weight = torque_multipliers(model, body.joint_names, table)

    assert weight.shape == (int(model.nv) - 6,)
    idx_vs = [int(x) for x in model.idx_vs]
    for name, expect in (("left_wrist", 17.0), ("right_index2", 25.0), ("left_knee", 1.0)):
        start = idx_vs[int(model.body_name_to_id[name])] - 6
        assert weight[start:start + 3].tolist() == [expect] * 3
    assert float((weight == 1.0).float().mean()) > 0.5      # most joints are unweighted


def test_torque_multipliers_reject_an_uncovered_joint():
    from bvr.body import load_body
    from bvr.human_optim.kindyn import torque_multipliers

    body = load_body(device="cpu")
    model = body.shaped(torch.zeros(10)).model
    with pytest.raises(KeyError):
        torque_multipliers(model, body.joint_names, {"hip": 1.0})


def test_torque_terms_read_the_weighted_square():
    spec = _spec([0], num_steps=4)
    spec.tau_weight = torch.tensor([1.0, 4.0])
    tau = torch.tensor([[1.0, 1.0], [2.0, 3.0]])
    assert spec.torque_sq(tau).tolist() == [[1.0, 4.0], [4.0, 36.0]]
    spec.tau_weight = None
    assert spec.torque_sq(tau).tolist() == [[1.0, 1.0], [4.0, 9.0]]
    assert spec.dyn_trip.tolist() == [1.0, 1.0]


def test_fps_scaling_treats_torque_accel_as_order_two():
    from bvr.config import fps_scaled_weights

    scaled = fps_scaled_weights({"torque_magnitude": 1.0, "torque_smooth": 1.0,
                                 "torque_accel": 1.0}, 60.0)
    assert scaled["torque_magnitude"] == pytest.approx(1.0)
    assert scaled["torque_smooth"] == pytest.approx(4.0)
    assert scaled["torque_accel"] == pytest.approx(16.0)
