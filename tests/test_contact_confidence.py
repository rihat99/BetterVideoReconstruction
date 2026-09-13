from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools.config import load_toml
from tools.human_optim.confidence import (
    CONTACT_CONFIDENCE_METHOD,
    confidence_coefficients,
    fold_contacts,
    fold_drift,
    frame_report_joints,
    label_confidence,
    transition_seconds,
)
from tools.viewer.loading import _load_skeleton

_REPO_ROOT = Path(__file__).resolve().parents[1]

# A hand-made coefficient set: the two heads read drift with opposite signs, and only
# the four calibrated groups appear.
_COEFFS = {
    "drift_floor_m": 0.005,
    "transition_cap_s": 1.0,
    "fit": "unit test",
    "contact": {"intercept": 0.0, "log_drift": -1.0, "log_transition": 0.5,
                "hand": 1.0, "foot": 0.0, "ankle": -1.0, "knee": 0.0},
    "free": {"intercept": 0.0, "log_drift": 1.0, "log_transition": 0.5,
             "hand": 1.0, "foot": 0.0, "ankle": -1.0, "knee": 0.0},
}


# ── the frame → joint fold ─────────────────────────────────────────────────


def test_frame_report_joints_folds_hand_frames_to_the_wrist():
    names = ["pelvis", "left_elbow", "left_wrist", "left_index1", "left_index2", "left_foot"]
    parents = np.array([-1, 0, 1, 2, 3, 0])
    frame_parents = [4, 2, 5, 1]      # fingertip, palm (parent = wrist), foot, elbow frame
    report = frame_report_joints(frame_parents, names, parents)
    assert report.tolist() == [2, 2, 5, 1]   # both hand frames land on the wrist


def test_fold_contacts_is_an_or_over_the_joints_frames():
    report = np.array([0, 0, 1])
    contact = np.zeros((2, 3), bool)
    contact[0, 1] = True                      # one of joint 0's two frames touches
    joint_contact = fold_contacts(contact, report, 3)
    assert joint_contact[0, 0] and not joint_contact[1, 0]
    assert not joint_contact[:, 1].any()      # joint 1's only frame never touches
    assert not joint_contact[:, 2].any()      # joint 2 owns no frame at all


def test_fold_drift_keeps_the_stillest_frame_and_stays_nan_when_all_are():
    report = np.array([0, 0, 1])
    drift = np.array([[0.4, 0.1, np.nan],
                      [np.nan, np.nan, 0.2]], np.float32)
    joint_drift = fold_drift(drift, report, 3)
    assert joint_drift[0, 0] == pytest.approx(0.1)     # stillest of the two frames
    assert np.isnan(joint_drift[1, 0])                 # both frames unknown
    assert np.isnan(joint_drift[0, 1])                 # its only frame is unknown
    assert np.isnan(joint_drift[:, 2]).all()           # owns no frame


# ── time to the nearest label change ───────────────────────────────────────


def test_transition_seconds_measures_both_ends_and_ignores_untracked_gaps():
    valid = np.array([True, True, True, False, True, True, True, True])
    contact = np.zeros((8, 3), bool)
    contact[2:5, 0] = True        # flips at 1.5 (on) and 4.5 (off), both inside the track
    contact[4:, 1] = True         # flips only across the untracked frame 3 — not a change
    contact[1:, 2] = True         # flips at 0.5, right at the start of the track
    seconds = transition_seconds(contact, valid, fps=2.0)

    np.testing.assert_allclose(seconds[:, 0], [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 1.0])
    assert np.isinf(seconds[:, 1]).all()
    np.testing.assert_allclose(seconds[:, 2], [0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])


# ── the confidence itself ──────────────────────────────────────────────────


def _confidence(contact, drift, valid=None, names=("left_wrist",), fps=30.0):
    contact = np.asarray(contact, bool)
    valid = np.ones(contact.shape[0], bool) if valid is None else np.asarray(valid, bool)
    return label_confidence(contact, np.asarray(drift, np.float32), valid, fps,
                            list(names), _COEFFS)


def test_more_drift_makes_a_contact_label_less_sure_and_a_free_label_more_sure():
    drift = np.array([[0.0], [0.01], [0.1], [1.0]], np.float32)
    touching = _confidence(np.ones((4, 1), bool), drift)[:, 0]
    free = _confidence(np.zeros((4, 1), bool), drift)[:, 0]
    assert np.all(np.diff(touching) < 0.0)
    assert np.all(np.diff(free) > 0.0)


def test_a_label_far_from_a_change_is_surer_than_one_next_to_it():
    contact = np.zeros((9, 1), bool)
    contact[:4] = True                       # the label flips between frames 3 and 4
    free = _confidence(contact, np.full((9, 1), 0.05, np.float32), fps=30.0)[:, 0]
    assert free[4] < free[5] < free[8]       # the frame touching the flip is the least sure


def test_no_statement_where_there_is_no_evidence():
    contact = np.array([[True, True], [True, True], [True, True]])
    drift = np.array([[0.05, 0.05], [np.nan, np.nan], [0.05, 0.05]], np.float32)
    valid = np.array([True, True, False])
    out = _confidence(contact, drift, valid, names=("left_wrist", "left_elbow"))
    assert np.isfinite(out[0, 0])            # tracked, drift known, calibrated joint
    assert np.isnan(out[1, 0])               # drift unknown
    assert np.isnan(out[2, 0])               # untracked frame
    assert np.isnan(out[:, 1]).all()         # the elbow has no fitted group


def test_each_group_uses_its_own_coefficients():
    names = ("left_wrist", "left_foot", "left_ankle")
    drift = np.full((1, 3), 0.05, np.float32)
    out = _confidence(np.ones((1, 3), bool), drift, names=names)[0]
    assert out[0] > out[1] > out[2]          # hand +1, foot 0, ankle −1 in _COEFFS


def test_confidence_coefficients_rejects_an_incomplete_or_broken_section():
    with pytest.raises(ValueError, match="empty"):
        confidence_coefficients({})
    with pytest.raises(ValueError, match="drift_floor_m"):
        confidence_coefficients({k: v for k, v in _COEFFS.items() if k != "drift_floor_m"})
    with pytest.raises(ValueError, match="drift_floor_m"):
        confidence_coefficients({**_COEFFS, "drift_floor_m": 0.0})
    with pytest.raises(ValueError, match="free head is missing"):
        confidence_coefficients({k: v for k, v in _COEFFS.items() if k != "free"})
    with pytest.raises(ValueError, match="contact.ankle is missing"):
        confidence_coefficients(
            {**_COEFFS, "contact": {k: v for k, v in _COEFFS["contact"].items() if k != "ankle"}})
    with pytest.raises(ValueError, match="must be finite"):
        confidence_coefficients({**_COEFFS, "free": {**_COEFFS["free"], "log_drift": float("nan")}})


def test_the_shipped_coefficients_read_drift_the_way_the_two_classes_need():
    shipped = confidence_coefficients(
        load_toml(_REPO_ROOT / "configs" / "human_optimization.toml")["contact_confidence"])
    assert shipped["contact"]["log_drift"] < 0.0 < shipped["free"]["log_drift"]
    assert shipped["contact"]["log_transition"] > 0.0
    assert shipped["free"]["log_transition"] > 0.0


# ── the recompute script writes what the stage would write ─────────────────


def test_recompute_reproduces_the_stage_fields(tmp_path):
    from scripts.recompute_contact_confidence import recompute

    joint_names = np.asarray(["pelvis", "left_wrist", "left_index1", "left_foot"])
    parents = np.asarray([-1, 0, 1, 0], np.int32)
    frame_parent = np.asarray([2, 1, 3], np.int64)          # fingertip, palm, foot
    report_joint = frame_report_joints(frame_parent, list(joint_names), parents)
    frame_contact = np.zeros((1, 5, 3), bool)
    frame_contact[0, 1:4, 0] = True                          # the fingertip touches
    frame_drift = np.full((1, 5, 3), 0.04, np.float32)
    frame_drift[0, 2, 1] = 0.01
    valid = np.ones((1, 5), bool)
    valid[0, 4] = False

    path = tmp_path / "contacts_1.npz"
    np.savez(path, joint_names=joint_names, parents=parents, frame_parent_joint=frame_parent,
             frame_contact=frame_contact, frame_drift_m=frame_drift, valid_mask=valid,
             fps=np.float32(30.0), object_ids=np.asarray([0], np.int32),
             frame_depth_trust=np.ones((1, 5, 3), np.float32),   # a retired witness
             confidence_method=np.str_("source-conditional-v1"))
    recompute(path, _COEFFS)

    saved = dict(np.load(path, allow_pickle=True))
    joint_contact = fold_contacts(frame_contact, report_joint, len(parents))
    expected = label_confidence(
        joint_contact[0], fold_drift(frame_drift, report_joint, len(parents))[0],
        valid[0], 30.0, list(joint_names), _COEFFS)

    np.testing.assert_array_equal(saved["joint_contact"], joint_contact)
    np.testing.assert_allclose(saved["joint_label_confidence"][0], expected,
                               rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(saved["frame_label_confidence"][0], expected[:, report_joint],
                               rtol=1e-6, equal_nan=True)
    assert "frame_depth_trust" not in saved
    assert str(saved["confidence_method"]) == CONTACT_CONFIDENCE_METHOD
    np.testing.assert_array_equal(saved["frame_contact"], frame_contact)   # labels untouched


def test_recompute_refuses_an_artifact_whose_fold_does_not_match(tmp_path):
    from scripts.recompute_contact_confidence import recompute

    joint_names = np.asarray(["pelvis", "left_wrist"])
    path = tmp_path / "contacts_1.npz"
    np.savez(path, joint_names=joint_names, parents=np.asarray([-1, 0], np.int32),
             frame_parent_joint=np.asarray([1], np.int64),
             frame_contact=np.zeros((1, 3, 1), bool),
             joint_contact=np.ones((1, 3, 2), bool),          # disagrees with the OR fold
             frame_drift_m=np.full((1, 3, 1), 0.02, np.float32),
             valid_mask=np.ones((1, 3), bool), fps=np.float32(30.0))
    with pytest.raises(ValueError, match="does not reproduce"):
        recompute(path, _COEFFS)


# ── unchanged neighbours: the drift window and the viewer's contact reader ──


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
