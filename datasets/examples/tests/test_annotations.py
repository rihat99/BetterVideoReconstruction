"""The manual annotations of the example clips, spread onto the contact frames.

``annotation_frame_labels`` copies each annotated joint's tri-state label onto the contact
frames it covers, with the 10- vs 14-joint schema difference on the heel.
"""

from __future__ import annotations

import numpy as np
import pytest

from datasets.examples.annotations import annotation_frame_labels

# The legacy 12-frame climbing set in its stored order — note slots 10/11 are RIGHT then LEFT.
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
