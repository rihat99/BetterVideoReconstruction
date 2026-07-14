from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.human_optim.confidence import (
    confidence_config,
    joint_label_confidence,
    occlusion_group_ids,
)
from tools.human_optim.contacts import joint_occlusion_counts, lift_joint_min_distance
from tools.viewer.loading import _load_skeleton


def _terms(*, labels, sdf, drift, occ=None, px=None, camera_depth=None):
    labels = torch.as_tensor(labels, dtype=torch.bool)
    sdf = torch.as_tensor(sdf, dtype=torch.float32)
    drift = torch.as_tensor(drift, dtype=torch.float32)
    shape = labels.shape
    if occ is None:
        occ = torch.zeros(shape)
    else:
        occ = torch.as_tensor(occ, dtype=torch.float32)
    if px is None:
        px = torch.full((*shape, 2), 50.0)
    else:
        px = torch.as_tensor(px, dtype=torch.float32)
    if camera_depth is None:
        camera_depth = torch.ones(shape)
    else:
        camera_depth = torch.as_tensor(camera_depth, dtype=torch.float32)
    return joint_label_confidence(
        labels, sdf, drift, px, camera_depth, occ,
        height=100, width=100, window_s=0.35, cfg=confidence_config())


def test_full_3d_occlusion_zeroes_both_label_classes():
    terms = _terms(
        labels=[[True], [False]],
        sdf=[[0.0], [0.0]],
        drift=[[0.0], [0.35]],       # second row has very strong motion evidence
        occ=[[5.0], [20.0]],
    )
    np.testing.assert_array_equal(terms["visibility"], np.zeros((2, 1), np.float32))
    np.testing.assert_array_equal(terms["label_confidence"], np.zeros((2, 1), np.float32))


def test_partial_occlusion_fades_monotonically_before_full_occlusion():
    terms = _terms(
        labels=[[True]] * 6, sdf=[[0.0]] * 6, drift=[[0.0]] * 6,
        occ=[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
    )
    visibility = terms["occlusion_visibility"][:, 0]
    assert np.all(np.diff(visibility) < 0.0)
    assert visibility[0] == pytest.approx(1.0)
    assert visibility[-1] == pytest.approx(0.0)


def test_approach_and_contact_transition_have_no_persistence_dip():
    terms = _terms(
        labels=[[False], [True]],
        sdf=[[0.02], [0.02]],
        drift=[[0.07], [0.0]],       # 0.20 m/s approach, then settled contact
    )
    assert terms["motion"][0, 0] > 0.8
    assert terms["label_confidence"][0, 0] > 0.8
    assert terms["label_confidence"][1, 0] > 0.8


def test_stationary_near_wall_no_contact_stays_uncertain():
    terms = _terms(labels=[[False]], sdf=[[0.02]], drift=[[0.0]])
    assert terms["motion"][0, 0] == pytest.approx(0.0)
    assert terms["label_confidence"][0, 0] == pytest.approx(0.0)


def test_stationary_far_no_contact_is_also_zero_confidence():
    # Reconstructed wall distance is intentionally absent from the free-label
    # formula: it must not rescue a stationary floating limb.
    terms = _terms(labels=[[False]], sdf=[[1.0]], drift=[[0.0]])
    assert terms["motion"][0, 0] == pytest.approx(0.0)
    assert terms["label_confidence"][0, 0] == pytest.approx(0.0)


def test_joint_clearance_lifts_true_euclidean_distance():
    distances = torch.tensor([[0.30, 0.10, float("inf")]])
    result = lift_joint_min_distance(distances, torch.tensor([0, 0, 1]), 2)
    assert result[0, 0] == pytest.approx(0.10)
    assert torch.isnan(result[0, 1])


def test_implausibly_deep_contact_is_not_treated_as_certain():
    shallow = _terms(labels=[[True]], sdf=[[-0.005]], drift=[[0.0]])
    deep = _terms(labels=[[True]], sdf=[[-0.10]], drift=[[0.0]])
    assert shallow["surface"][0, 0] > 0.7
    assert deep["surface"][0, 0] == pytest.approx(0.40, abs=0.01)
    assert deep["label_confidence"][0, 0] < shallow["label_confidence"][0, 0]


def test_motion_evidence_is_continuous_and_monotone_without_dead_zone():
    terms = _terms(
        labels=[[False], [False], [False], [False]],
        sdf=[[0.0], [0.0], [0.0], [0.0]],
        drift=[[0.0], [0.0035], [0.035], [0.35]],  # 0, .01, .1, 1 m/s
    )
    motion = terms["motion"][:, 0]
    assert motion[0] == pytest.approx(0.0)
    assert 0.0 < motion[1] < motion[2] < motion[3] < 1.0
    assert motion[1] == pytest.approx(1.0 - np.exp(-0.1), abs=1e-6)
    assert motion[2] == pytest.approx(1.0 - np.exp(-1.0), abs=1e-6)


def test_out_of_frame_joint_has_zero_confidence():
    terms = _terms(
        labels=[[True]], sdf=[[0.0]], drift=[[0.0]], px=[[[-1.0, 50.0]]],
    )
    assert terms["visibility"][0, 0] == pytest.approx(0.0)
    assert terms["label_confidence"][0, 0] == pytest.approx(0.0)


def test_image_border_visibility_is_continuous():
    terms = _terms(
        labels=[[True], [True], [True]], sdf=[[0.0], [0.0], [0.0]],
        drift=[[0.0], [0.0], [0.0]],
        px=[[[0.0, 50.0]], [[5.0, 50.0]], [[10.0, 50.0]]],
    )
    image = terms["image_visibility"][:, 0]
    np.testing.assert_allclose(image, [0.0, 0.5, 1.0], atol=1e-6)


def test_joint_behind_camera_has_zero_confidence_even_if_projection_is_in_frame():
    terms = _terms(
        labels=[[True]], sdf=[[0.0]], drift=[[0.0]],
        px=[[[50.0, 50.0]]], camera_depth=[[-1.0]],
    )
    assert terms["image_visibility"][0, 0] == pytest.approx(0.0)
    assert terms["label_confidence"][0, 0] == pytest.approx(0.0)


def test_nonfinite_projection_and_scene_distance_fail_closed():
    terms = _terms(
        labels=[[True], [True]], sdf=[[float("nan")], [float("inf")]],
        drift=[[0.0], [0.0]], px=[[[float("nan"), 50.0]], [[50.0, 50.0]]],
    )
    np.testing.assert_array_equal(terms["surface"], np.zeros((2, 1), np.float32))
    np.testing.assert_array_equal(terms["label_confidence"], np.zeros((2, 1), np.float32))


def test_occlusion_groups_never_share_an_unmapped_sentinel():
    names = ["pelvis", "spine1", "left_wrist", "left_index1", "left_index2", "left_middle1"]
    groups = occlusion_group_ids(names)
    assert np.all(groups >= 0)
    assert groups[0] != groups[1]
    assert groups[2] == groups[3] == groups[4] == groups[5]  # one coarse hand surface
    assert groups[0] != groups[2]


def test_front_vertex_from_another_group_counts_as_occluder():
    vertices = torch.tensor([[[0.0, 0.0, 1.0]]])
    joints = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]])
    camera = torch.zeros((1, 3))
    v2j = torch.tensor([0])
    groups = torch.tensor([0, 1])
    count = joint_occlusion_counts(
        vertices, joints, camera, v2j, groups,
        perp_m=0.04, front_m=0.05, subsample=10, chunk=1)
    assert count[0, 0] == 0       # its own local surface is excluded
    assert count[0, 1] == 1       # the same vertex occludes the other joint


def test_other_person_mesh_can_occlude_a_joint():
    own_vertices = torch.tensor([[[1.0, 0.0, 1.0]]])
    other_vertices = torch.tensor([[[0.0, 0.0, 1.0]]])
    joints = torch.tensor([[[0.0, 0.0, 2.0]]])
    count = joint_occlusion_counts(
        own_vertices, joints, torch.zeros((1, 3)), torch.tensor([0]),
        torch.tensor([0]), other_vertices=other_vertices, subsample=10, chunk=1)
    assert count[0, 0] == 1


def test_each_other_person_keeps_its_own_subsample_budget():
    own_vertices = torch.tensor([[[1.0, 0.0, 1.0]]])
    other_vertices = torch.tensor([[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.2]]]])
    joints = torch.tensor([[[0.0, 0.0, 2.0]]])
    count = joint_occlusion_counts(
        own_vertices, joints, torch.zeros((1, 3)), torch.tensor([0]),
        torch.tensor([0]), other_vertices=other_vertices, subsample=1, chunk=1)
    assert count[0, 0] == 2


def test_invalid_other_person_mesh_does_not_occlude_a_joint():
    own_vertices = torch.tensor([[[1.0, 0.0, 1.0]]])
    other_vertices = torch.full((1, 1, 3), float("nan"))
    joints = torch.tensor([[[0.0, 0.0, 2.0]]])
    count = joint_occlusion_counts(
        own_vertices, joints, torch.zeros((1, 3)), torch.tensor([0]),
        torch.tensor([0]), other_vertices=other_vertices, subsample=10, chunk=1)
    assert count[0, 0] == 0


def test_occlusion_has_no_five_centimetre_blind_zone():
    vertices = torch.tensor([[[0.0, 0.0, 1.99]]])
    joints = torch.tensor([[[1.0, 0.0, 2.0], [0.0, 0.0, 2.0]]])
    count = joint_occlusion_counts(
        vertices, joints, torch.zeros((1, 3)), torch.tensor([0]), torch.tensor([0, 1]),
        subsample=10, chunk=1)
    assert count[0, 1] == 1       # a surface 1 cm in front genuinely hides the joint


def test_occlusion_is_measured_in_front_of_local_skin_not_joint_centre():
    # Target-group skin at z=1.95 is the visible local surface for joint 1.
    # The other-group vertex at z=1.98 is in front of the joint centre but behind
    # that skin, so it is not an occluder. Moving it to z=1.90 makes it one.
    joints = torch.tensor([[[1.0, 0.0, 2.0], [0.0, 0.0, 2.0]]])
    camera = torch.zeros((1, 3))
    groups = torch.tensor([0, 1])
    v2j = torch.tensor([1, 0])
    behind_skin = torch.tensor([[[0.0, 0.0, 1.95], [0.0, 0.0, 1.98]]])
    in_front = torch.tensor([[[0.0, 0.0, 1.95], [0.0, 0.0, 1.90]]])
    count_behind = joint_occlusion_counts(
        behind_skin, joints, camera, v2j, groups, subsample=10, chunk=1)
    count_front = joint_occlusion_counts(
        in_front, joints, camera, v2j, groups, subsample=10, chunk=1)
    assert count_behind[0, 1] == 0
    assert count_front[0, 1] == 1


def test_local_skin_behind_joint_cannot_make_behind_joint_vertex_an_occluder():
    joints = torch.tensor([[[1.0, 0.0, 2.0], [0.0, 0.0, 2.0]]])
    # Target-group skin is behind joint 1 at z=2.10. The unrelated vertex at
    # z=2.05 is also behind the joint, so neither can occlude it from z=0.
    vertices = torch.tensor([[[0.0, 0.0, 2.10], [0.0, 0.0, 2.05]]])
    count = joint_occlusion_counts(
        vertices, joints, torch.zeros((1, 3)), torch.tensor([1, 0]),
        torch.tensor([0, 1]), subsample=10, chunk=1)
    assert count[0, 1] == 0


def test_confidence_config_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="surface_inner_m"):
        confidence_config({"surface_inner_m": 0.1, "surface_outer_m": 0.0})
    with pytest.raises(ValueError, match="penetration_floor"):
        confidence_config({"penetration_floor": -0.1})
    with pytest.raises(ValueError, match="unknown contact_confidence"):
        confidence_config({"obsolete_gate": 0.5})
    with pytest.raises(ValueError, match="must be finite"):
        confidence_config({"motion_scale_mps": float("nan")})


def test_joint_drift_rejects_invalid_window_endpoints():
    from tools.human_optim.contacts import joint_drift

    joints = torch.arange(5, dtype=torch.float32).view(5, 1, 1).expand(-1, -1, 3)
    valid = torch.tensor([True, False, True, True, True])
    drift = joint_drift(joints, fps=2.0, window_s=1.0, valid_mask=valid)
    assert torch.isnan(drift[0]).all()   # current frame's right endpoint is invalid
    assert torch.isnan(drift[1]).all()   # current frame itself is invalid
    assert torch.isnan(drift[2]).all()   # left endpoint is invalid
    assert torch.isfinite(drift[3:]).all()


def test_viewer_ignores_legacy_confidence_but_keeps_contact_labels(tmp_path):
    import hashlib

    human = tmp_path / "human_optim"
    human.mkdir()
    names = np.asarray(["pelvis", "left_wrist", "right_wrist"])
    body = human / "kinematic_1.npz"
    np.savez(
        body, parents=np.asarray([-1, 0, 0], np.int32), joint_names=names,
        object_ids=np.asarray([0], np.int32), num_frames=np.int32(2),
    )
    body_hash = hashlib.sha256(body.read_bytes()).hexdigest()
    shape = (1, 2, 3)
    np.savez(
        human / "contacts_1.npz",
        joint_contact=np.zeros(shape, bool), label_confidence=np.ones(shape, np.float32),
        confidence_schema=np.int32(2), confidence_method=np.str_("legacy"),
        source_body_sha256=np.str_(body_hash), joint_names=names,
        object_ids=np.asarray([0], np.int32),
    )
    with pytest.warns(RuntimeWarning, match="stale/invalid joint confidence"):
        loaded = _load_skeleton(tmp_path)
    assert loaded["contact"] is not None
    assert loaded["confidence"] is None
