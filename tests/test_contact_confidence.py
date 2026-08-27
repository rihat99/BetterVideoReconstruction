from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.human_optim.confidence import (
    confidence_config,
    fold_to_joints,
    frame_report_joints,
    joint_label_confidence,
    occlusion_group_ids,
)
from tools.human_optim.contacts import joint_occlusion_gap
from tools.viewer.loading import _load_skeleton

# One triangle spanning the +z axis at depth ``z``; the camera at the origin looking
# down +z sees the origin ray pierce it, so a joint at ``(0, 0, d>z)`` is occluded.
_FACES = torch.tensor([[0, 1, 2]])


def _tri(z: float) -> torch.Tensor:
    return torch.tensor([[[-0.5, -0.5, z], [0.5, -0.5, z], [0.0, 0.6, z]]])


def _gap(vertices, joints, faces, v2j, groups, *, camera=None, other_vertices=None,
         target_group=None):
    vertices = torch.as_tensor(vertices, dtype=torch.float32)
    if camera is None:
        camera = torch.zeros((vertices.shape[0], 3))
    other = None if other_vertices is None else torch.as_tensor(other_vertices, dtype=torch.float32)
    return joint_occlusion_gap(
        vertices, torch.as_tensor(joints, dtype=torch.float32),
        torch.as_tensor(camera, dtype=torch.float32), torch.as_tensor(faces, dtype=torch.long),
        torch.as_tensor(v2j, dtype=torch.long), torch.as_tensor(groups, dtype=torch.long),
        other_vertices=other, target_group=target_group)


_MIN_DEPTH_POINTS = 5


def _terms(*, labels, sdf, drift, depth_trust=None, depth_points=None, source=None,
           force_share=None, gap=None, px=None, camera_depth=None, caps=None):
    """Run :func:`joint_label_confidence` with v6-neutral defaults.

    The defaults reduce v6 to the earlier algebra: every contact is detector
    source (top tier), every joint has a present depth witness (``depth_points``
    above the minimum), and there is no solved-force witness. Individual tests
    override ``source``/``force_share``/``depth_points`` to exercise the caps.
    """
    labels = torch.as_tensor(labels, dtype=torch.bool)
    sdf = torch.as_tensor(sdf, dtype=torch.float32)
    drift = torch.as_tensor(drift, dtype=torch.float32)
    shape = labels.shape
    depth_trust = (torch.ones(shape) if depth_trust is None
                   else torch.as_tensor(depth_trust, dtype=torch.float32))
    depth_points = (torch.full(shape, float(_MIN_DEPTH_POINTS)) if depth_points is None
                    else torch.as_tensor(depth_points, dtype=torch.float32))
    source = (labels.to(torch.int64) if source is None            # detector bit for contacts
              else torch.as_tensor(source, dtype=torch.int64))
    force_share = (torch.full(shape, float("nan")) if force_share is None
                   else torch.as_tensor(force_share, dtype=torch.float32))
    gap = (torch.full(shape, float("nan")) if gap is None         # no surface in front ⇒ visible
           else torch.as_tensor(gap, dtype=torch.float32))
    px = torch.full((*shape, 2), 50.0) if px is None else torch.as_tensor(px, dtype=torch.float32)
    camera_depth = (torch.ones(shape) if camera_depth is None
                    else torch.as_tensor(camera_depth, dtype=torch.float32))
    return joint_label_confidence(
        labels, sdf, drift, depth_trust, depth_points, source, force_share,
        px, camera_depth, gap, height=100, width=100, window_s=0.35,
        min_depth_points=_MIN_DEPTH_POINTS, cfg=confidence_config(caps))


def test_full_3d_occlusion_zeroes_both_label_classes():
    terms = _terms(
        labels=[[True], [False]],
        sdf=[[0.0], [0.0]],
        drift=[[0.0], [0.35]],       # second row has very strong motion evidence
        gap=[[0.5], [0.5]],          # 0.5 m of body in front ≫ occ_margin ⇒ fully hidden
    )
    assert np.all(terms["visibility"] < 1e-6)
    assert np.all(terms["label_confidence"] < 1e-6)


def test_partial_occlusion_fades_monotonically_with_the_gap():
    terms = _terms(
        labels=[[True]] * 6, sdf=[[0.0]] * 6, drift=[[0.0]] * 6,
        gap=[[-0.10], [0.0], [0.03], [0.05], [0.08], [0.30]],
    )
    visibility = terms["occlusion_visibility"][:, 0]
    assert np.all(np.diff(visibility) < 0.0)
    assert visibility[0] > 0.999                             # surface behind the joint ⇒ visible
    assert visibility[3] == pytest.approx(0.5, abs=1e-3)     # gap == occ_margin_m ⇒ half
    assert visibility[-1] < 1e-6                             # deep occlusion ⇒ hidden


def test_no_occluder_gap_is_fully_visible():
    terms = _terms(labels=[[True]], sdf=[[0.0]], drift=[[0.0]], gap=[[float("nan")]])
    assert terms["occlusion_visibility"][0, 0] == pytest.approx(1.0)
    assert terms["visibility"][0, 0] == pytest.approx(1.0)


def test_approach_and_contact_transition_have_no_persistence_dip():
    terms = _terms(
        labels=[[False], [True]],
        sdf=[[0.02], [0.02]],
        drift=[[0.07], [0.0]],       # 0.20 m/s approach, then settled contact
    )
    assert terms["motion"][0, 0] > 0.8
    assert terms["label_confidence"][0, 0] > 0.8
    assert terms["label_confidence"][1, 0] > 0.8


def test_close_slow_no_contact_is_approximately_zero_confidence():
    terms = _terms(labels=[[False]], sdf=[[0.02]], drift=[[0.014]])  # 0.04 m/s
    assert 0.0 < terms["motion"][0, 0] < 0.025
    assert terms["label_confidence"][0, 0] == terms["motion"][0, 0]


def test_fast_near_wall_no_contact_stays_high_confidence():
    terms = _terms(labels=[[False]], sdf=[[0.02]], drift=[[0.07]])  # 0.20 m/s
    assert terms["motion"][0, 0] > 0.9
    assert terms["label_confidence"][0, 0] > 0.9


def test_slow_far_no_contact_stays_low_without_clearance_evidence():
    near = _terms(labels=[[False]], sdf=[[0.02]], drift=[[0.014]])
    far = _terms(labels=[[False]], sdf=[[1.0]], drift=[[0.014]])
    assert near["motion"][0, 0] < 0.025
    assert 0.3 < far["motion"][0, 0] < 0.35


def test_zero_depth_trust_zeroes_no_contact_confidence():
    terms = _terms(
        labels=[[False]], sdf=[[1.0]], drift=[[0.35]], depth_trust=[[0.0]])
    assert terms["motion"][0, 0] == pytest.approx(0.0)
    assert terms["label_confidence"][0, 0] == pytest.approx(0.0)


def test_depth_trust_does_not_change_contact_label_confidence():
    trusted = _terms(labels=[[True]], sdf=[[0.0]], drift=[[0.0]], depth_trust=[[1.0]])
    untrusted = _terms(labels=[[True]], sdf=[[0.0]], drift=[[0.0]], depth_trust=[[0.0]])
    assert trusted["label_confidence"][0, 0] == pytest.approx(
        untrusted["label_confidence"][0, 0])


def test_stationary_far_no_contact_is_also_zero_confidence():
    # Reconstructed wall distance is intentionally absent from the free-label
    # formula: it must not rescue a stationary floating limb.
    terms = _terms(labels=[[False]], sdf=[[1.0]], drift=[[0.0]])
    assert terms["motion"][0, 0] == pytest.approx(0.0)
    assert terms["label_confidence"][0, 0] == pytest.approx(0.0)


def test_implausibly_deep_contact_is_not_treated_as_certain():
    shallow = _terms(labels=[[True]], sdf=[[-0.005]], drift=[[0.0]])
    deep = _terms(labels=[[True]], sdf=[[-0.10]], drift=[[0.0]])
    assert shallow["surface"][0, 0] > 0.7
    assert deep["surface"][0, 0] == pytest.approx(0.40, abs=0.01)
    assert deep["label_confidence"][0, 0] < shallow["label_confidence"][0, 0]


def test_motion_evidence_is_continuous_and_monotone_without_dead_zone():
    terms = _terms(
        labels=[[False], [False], [False], [False]],
        sdf=[[1.0], [1.0], [1.0], [1.0]],
        drift=[[0.0], [0.0035], [0.035], [0.35]],  # 0, .01, .1, 1 m/s
    )
    motion = terms["motion"][:, 0]
    assert motion[0] == pytest.approx(0.0)
    assert 0.0 < motion[1] < motion[2] < motion[3] <= 1.0
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


def test_occlusion_groups_are_nonnegative_and_group_the_axial_torso():
    names = ["pelvis", "spine1", "head", "left_wrist", "left_index1", "left_index2"]
    groups = occlusion_group_ids(names)
    assert np.all(groups >= 0)
    assert groups[0] == groups[1]                        # pelvis + spine1 share the axial group
    assert groups[3] == groups[4] == groups[5]           # one coarse hand surface
    assert groups[2] != groups[0]                        # head stays outside the axial torso
    assert groups[2] != groups[3]                        # head vs hand distinct


def test_occluder_gap_is_joint_depth_minus_hit_depth():
    joints = torch.tensor([[[0.0, 0.0, 2.0], [3.0, 0.0, 2.0]]])   # target = joint 0
    v2j = torch.tensor([1, 1, 1])                                 # triangle owned by joint 1
    groups = torch.tensor([0, 1])                                 # distinct groups
    gap = _gap(_tri(1.0), joints, _FACES, v2j, groups)
    assert gap[0, 0] == pytest.approx(1.0, abs=1e-4)              # 2 − 1 m of surface in front


def test_own_group_surface_is_not_an_occluder():
    joints = torch.tensor([[[0.0, 0.0, 2.0], [3.0, 0.0, 2.0]]])
    v2j = torch.tensor([1, 1, 1])                                 # triangle owned by joint 1
    groups = torch.tensor([0, 0])                                 # joint 1 shares joint 0's group
    gap = _gap(_tri(1.0), joints, _FACES, v2j, groups)
    assert torch.isnan(gap[0, 0])                                 # own skin excluded ⇒ nothing in front


def test_nearest_surface_in_front_sets_the_gap():
    verts = torch.cat([_tri(0.8), _tri(1.2)], dim=1)              # (1, 6, 3)
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    joints = torch.tensor([[[0.0, 0.0, 2.0], [3.0, 0.0, 2.0]]])
    v2j = torch.tensor([1, 1, 1, 1, 1, 1])                        # both owned by joint 1
    groups = torch.tensor([0, 1])
    gap = _gap(verts, joints, faces, v2j, groups)
    assert gap[0, 0] == pytest.approx(1.2, abs=1e-4)             # nearest (z=0.8) wins: 2 − 0.8


def test_surface_behind_the_joint_is_not_an_occluder():
    joints = torch.tensor([[[0.0, 0.0, 2.0], [3.0, 0.0, 2.0]]])
    v2j = torch.tensor([1, 1, 1])                                 # owned by joint 1 (non-own for joint 0)
    groups = torch.tensor([0, 1])
    gap = _gap(_tri(2.5), joints, _FACES, v2j, groups)
    assert torch.isnan(gap[0, 0])                               # nearest surface 0.5 m BEHIND ⇒ not an occluder (clear)


def test_other_person_mesh_occludes_a_joint():
    joints = torch.tensor([[[0.0, 0.0, 2.0]]])
    v2j = torch.tensor([0, 0, 0])                                 # target triangle is own
    groups = torch.tensor([0])
    other = _tri(1.0)[:, None]                                    # (1, 1, 3, 3) — one other person, z=1
    gap = _gap(_tri(3.0), joints, _FACES, v2j, groups, other_vertices=other)
    assert gap[0, 0] == pytest.approx(1.0, abs=1e-4)            # other person's surface 1 m in front


def test_invalid_other_person_mesh_does_not_occlude_a_joint():
    joints = torch.tensor([[[0.0, 0.0, 2.0]]])
    v2j = torch.tensor([0, 0, 0])
    groups = torch.tensor([0])
    other = torch.full((1, 1, 3, 3), float("nan"))
    gap = _gap(_tri(3.0), joints, _FACES, v2j, groups, other_vertices=other)
    assert torch.isnan(gap[0, 0])                                # NaN mesh never hits ⇒ visible


def test_confidence_config_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="surface_inner_m"):
        confidence_config({"surface_inner_m": 0.1, "surface_outer_m": 0.0})
    with pytest.raises(ValueError, match="penetration_floor"):
        confidence_config({"penetration_floor": -0.1})
    with pytest.raises(ValueError, match="unknown contact_confidence"):
        confidence_config({"obsolete_gate": 0.5})
    with pytest.raises(ValueError, match="must be finite"):
        confidence_config({"motion_scale_mps": float("nan")})
    with pytest.raises(ValueError, match="motion_near_scale_mps"):
        confidence_config({"motion_scale_mps": 0.2, "motion_near_scale_mps": 0.1})
    with pytest.raises(ValueError, match="motion_near_power"):
        confidence_config({"motion_near_power": 0.5})
    with pytest.raises(ValueError, match="gate_force_cap"):
        confidence_config({"gate_only_cap": 0.8, "gate_force_cap": 0.5})
    with pytest.raises(ValueError, match="nodata_force_cap"):
        confidence_config({"nodata_cap": 0.8, "nodata_force_cap": 0.5})
    with pytest.raises(ValueError, match="force_share_scale"):
        confidence_config({"force_share_scale": 0.0})


# ── v6 source-conditional caps ──────────────────────────────────────────────
# A well-placed contact has strong surface evidence (d = 0, so ``surface`` ≈ 0.92);
# the caps must hold that evidence back for the less trustworthy label sources.

_GATE, _FORCE, _DETECTOR = 2, 4, 1


def test_gate_only_contact_never_exceeds_its_cap():
    terms = _terms(labels=[[True]], sdf=[[0.0]], drift=[[0.0]], source=[[_GATE]])
    cap = confidence_config()["gate_only_cap"]
    assert terms["surface"][0, 0] > 0.9                     # raw evidence is strong
    assert terms["label_confidence"][0, 0] == pytest.approx(cap, abs=1e-4)


def test_force_corroboration_lifts_a_gate_contact_above_the_gate_cap():
    c = confidence_config()
    corroborated = _terms(labels=[[True]], sdf=[[0.0]], drift=[[0.0]],
                          source=[[_GATE | _FORCE]], force_share=[[0.5]])
    value = corroborated["label_confidence"][0, 0]
    assert value > c["gate_only_cap"] + 1e-3
    assert value <= c["gate_force_cap"] + 1e-4


def test_force_only_contact_reads_force_share_when_surface_is_absent():
    # A misregistered limb 20 cm off the wall: surface evidence is ~0, yet a solved
    # force share supplies mid, capped, evidence.
    c = confidence_config()
    terms = _terms(labels=[[True]], sdf=[[0.20]], drift=[[0.0]],
                   source=[[_FORCE]], force_share=[[0.6]])
    assert terms["surface"][0, 0] < 1e-3
    value = terms["label_confidence"][0, 0]
    assert 0.1 < value <= c["force_only_cap"] + 1e-4


def test_force_only_contact_never_exceeds_its_cap():
    c = confidence_config()
    terms = _terms(labels=[[True]], sdf=[[0.0]], drift=[[0.0]],
                   source=[[_FORCE]], force_share=[[10.0]])   # saturating force
    assert terms["label_confidence"][0, 0] <= c["force_only_cap"] + 1e-4


def test_detector_contact_ignores_the_source_caps():
    terms = _terms(labels=[[True]], sdf=[[0.0]], drift=[[0.0]], source=[[_DETECTOR]])
    assert terms["label_confidence"][0, 0] > confidence_config()["gate_force_cap"]


def test_missing_depth_no_contact_cap_is_off_by_default():
    # The shipped defaults neutralize the missing-depth caps (both 1.0): a strong
    # motion witness passes through even without a depth witness.
    c = confidence_config()
    assert c["nodata_cap"] == 1.0 and c["nodata_force_cap"] == 1.0
    terms = _terms(labels=[[False]], sdf=[[1.0]], drift=[[0.35]], depth_points=[[0.0]])
    assert terms["motion"][0, 0] > 0.9
    assert terms["label_confidence"][0, 0] == pytest.approx(terms["motion"][0, 0], abs=1e-6)


def test_missing_depth_no_contact_never_exceeds_the_cap_when_enabled():
    caps = {"nodata_cap": 0.50, "nodata_force_cap": 0.75}
    terms = _terms(
        labels=[[False]], sdf=[[1.0]], drift=[[0.35]],       # very strong motion evidence
        depth_points=[[0.0]], caps=caps)                     # no depth witness matched
    assert terms["motion"][0, 0] > 0.9                       # raw evidence is strong
    assert terms["label_confidence"][0, 0] == pytest.approx(caps["nodata_cap"], abs=1e-4)


def test_force_absence_lifts_the_missing_depth_no_contact_cap_when_enabled():
    caps = {"nodata_cap": 0.50, "nodata_force_cap": 0.75}
    terms = _terms(
        labels=[[False]], sdf=[[1.0]], drift=[[0.35]], depth_points=[[0.0]],
        force_share=[[0.0]], caps=caps)                      # rep joint, solve allocates ~no force
    value = terms["label_confidence"][0, 0]
    assert value > caps["nodata_cap"] + 1e-3
    assert value <= caps["nodata_force_cap"] + 1e-4


def test_present_depth_no_contact_is_uncapped():
    terms = _terms(labels=[[False]], sdf=[[1.0]], drift=[[0.35]],
                   depth_points=[[100.0]])
    assert terms["label_confidence"][0, 0] > 0.9


def test_joint_drift_rejects_invalid_window_endpoints():
    from tools.human_optim.contacts import joint_drift

    joints = torch.arange(5, dtype=torch.float32).view(5, 1, 1).expand(-1, -1, 3)
    valid = torch.tensor([True, False, True, True, True])
    drift = joint_drift(joints, fps=2.0, window_s=1.0, valid_mask=valid)
    assert torch.isnan(drift[0]).all()   # current frame's right endpoint is invalid
    assert torch.isnan(drift[1]).all()   # current frame itself is invalid
    assert torch.isnan(drift[2]).all()   # left endpoint is invalid
    assert torch.isfinite(drift[3:]).all()


_JOINT_NAMES = np.asarray(["pelvis", "left_wrist", "right_wrist"])
_FRAME_NAMES = np.asarray(["palm_left", "palm_right"])


def _viewer_clip(tmp_path, *, contacts: dict | None) -> None:
    """A minimal ``out/<stem>/`` tree for the viewer loader: a body npz + contacts."""
    human = tmp_path / "human_optim"
    human.mkdir()
    np.savez(
        human / "kinematic_1.npz", parents=np.asarray([-1, 0, 0], np.int32),
        joint_names=_JOINT_NAMES, object_ids=np.asarray([0], np.int32),
        num_frames=np.int32(2),
    )
    if contacts is not None:
        np.savez(human / "contacts_1.npz", **contacts)


def test_viewer_reads_contact_frame_labels(tmp_path):
    contact = np.asarray([[[True, False], [False, False]]])          # (P=1, N=2, F=2)
    _viewer_clip(tmp_path, contacts=dict(
        contact_label_schema=np.int32(2), contact_label_method=np.str_("frames-v1"),
        frame_contact=contact, contact_frame_names=_FRAME_NAMES,
        frame_parent_joint=np.asarray([1, 2], np.int64),
        joint_names=_JOINT_NAMES, object_ids=np.asarray([0], np.int32)))

    loaded = _load_skeleton(tmp_path, frame_names=list(_FRAME_NAMES))
    assert loaded["joint_names"] == list(_JOINT_NAMES)
    assert np.array_equal(loaded["frame_contact"], contact)
    assert loaded["frame_names"] == list(_FRAME_NAMES)
    assert np.array_equal(loaded["frame_parent"], [1, 2])
    assert np.array_equal(loaded["contact_oids"], [0])


def test_viewer_drops_contact_labels_for_a_different_frame_set(tmp_path):
    _viewer_clip(tmp_path, contacts=dict(
        contact_label_schema=np.int32(2), contact_label_method=np.str_("frames-v1"),
        frame_contact=np.zeros((1, 2, 2), bool), contact_frame_names=_FRAME_NAMES,
        frame_parent_joint=np.asarray([1, 2], np.int64),
        joint_names=_JOINT_NAMES, object_ids=np.asarray([0], np.int32)))

    with pytest.warns(RuntimeWarning, match="incompatible contact artifact"):
        loaded = _load_skeleton(tmp_path, frame_names=["heel_left", "heel_right"])
    assert loaded["frame_contact"] is None
    assert loaded["joint_names"] == list(_JOINT_NAMES)   # the skeleton still loads


def test_viewer_ignores_legacy_per_joint_contact_file(tmp_path):
    _viewer_clip(tmp_path, contacts=dict(
        joint_contact=np.zeros((1, 2, 3), bool), label_confidence=np.ones((1, 2, 3), np.float32),
        contact_label_schema=np.int32(1), joint_names=_JOINT_NAMES,
        object_ids=np.asarray([0], np.int32)))

    with pytest.warns(RuntimeWarning, match="label schema 1"):
        loaded = _load_skeleton(tmp_path)
    assert loaded["frame_contact"] is None
    assert loaded["joint_names"] == list(_JOINT_NAMES)   # the skeleton still loads


def test_viewer_skeleton_without_a_contact_file(tmp_path):
    _viewer_clip(tmp_path, contacts=None)

    loaded = _load_skeleton(tmp_path)
    assert loaded["frame_contact"] is None
    assert np.array_equal(loaded["parents"], [-1, 0, 0])


# ── frames-era confidence: surface-point targets + the frame → joint fold ──


def test_target_group_lets_a_frame_target_keep_its_own_skin():
    # The target is a contact FRAME on joint 1's skin, not a model joint: its
    # target_group is joint 1's group, so joint-1 skin is its own and never occludes
    # it — while a target with a different group is occluded by the same triangle.
    targets = torch.tensor([[[0.0, 0.0, 2.0]]])
    v2j = torch.tensor([1, 1, 1])                                 # triangle owned by joint 1
    groups = torch.tensor([0, 1])                                 # per-JOINT groups
    own = _gap(_tri(1.0), targets, _FACES, v2j, groups, target_group=torch.tensor([1]))
    assert torch.isnan(own[0, 0])                                 # its own skin ⇒ clear
    other = _gap(_tri(1.0), targets, _FACES, v2j, groups, target_group=torch.tensor([0]))
    assert other[0, 0] == pytest.approx(1.0, abs=1e-4)            # foreign skin ⇒ occluded


def test_frame_report_joints_folds_hand_frames_to_the_wrist():
    names = ["pelvis", "left_elbow", "left_wrist", "left_index1", "left_index2", "left_foot"]
    parents = np.array([-1, 0, 1, 2, 3, 0])
    frame_parents = [4, 2, 5, 1]      # fingertip, palm (parent = wrist), foot, elbow frame
    report = frame_report_joints(frame_parents, names, parents)
    assert report.tolist() == [2, 2, 5, 1]   # both hand frames land on the wrist


def test_fold_to_joints_uses_the_label_algebra():
    report = np.array([0, 0, 1])
    contact = np.zeros((2, 3), bool)
    confidence = np.zeros((2, 3), np.float32)
    contact[0, 0] = True
    confidence[0] = [0.6, 0.9, 0.4]   # joint 0 in contact: one touching (0.6), one free (0.9)
    confidence[1] = [0.8, 0.3, 0.5]   # joint 0 free: least sure member is 0.3
    joint_contact, joint_conf = fold_to_joints(contact, confidence, report, 3)
    assert joint_contact[0, 0] and not joint_contact[1, 0]
    assert joint_conf[0, 0] == pytest.approx(0.6)   # surest TOUCHING frame, not the free 0.9
    assert joint_conf[1, 0] == pytest.approx(0.3)   # weakest link of the AND
    assert np.isnan(joint_conf[:, 2]).all() and not joint_contact[:, 2].any()  # owns no frame
