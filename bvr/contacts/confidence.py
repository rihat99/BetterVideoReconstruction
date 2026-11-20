"""How far to trust a readout: a calibrated confidence per contact label, and one per body for its forces.

**Labels.** The confidence of a label is the probability that a human annotator gives the same
one, fitted on the manually annotated scenes; the label itself is never changed. It is computed
per report joint, the joint a contact frame folds onto (every frame of a hand reports at its
wrist), because that is what people annotate. Three witnesses go in: which joint it is (hand,
forefoot, heel or knee), how still it was (its drift, measured before any contact term acted), and
how far the frame of the clip is from a change of the joint's own label, since labels are least
reliable where they switch. One logistic head per label value::

    logit = intercept + group + log_drift * log(drift + drift_floor_m)
                      + log_transition * log(1 + 30 * min(seconds to the nearest change, transition_cap_s))

The coefficients are ``[contact_confidence]``. Signed distance, solved load, keypoint scores,
reprojection residuals, visibility and the clip's quality each added at most 0.01 AUROC to these
three, and several are the label's own inputs, which would endorse it in a circle.

**Forces.** Contact forces are solved for the whole body at once, so their trust is a property of
the body in a frame of the clip (:func:`force_confidence`): the root residual the solve left, and
whether the forces add up to more than a person supports.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from bvr.terms.dynamics import LEVER

#: Report joint -> the group whose coefficients it reads. Any other joint has no fitted confidence (NaN).
REPORT_GROUPS = {
    "left_wrist": "hand", "right_wrist": "hand", "left_foot": "foot", "right_foot": "foot",
    "left_ankle": "ankle", "right_ankle": "ankle", "left_knee": "knee", "right_knee": "knee"}

#: The transition witness counts frames of a 30 fps clip, so a clip's own rate does not change the coefficient.
TRANSITION_RATE = 30.0

#: Force confidence halves at this root residual force and at this root residual torque over the lever arm
#: (body weights). The torque is held to less: friction cones have little moment authority, so some is normal.
RESIDUAL_FORCE, RESIDUAL_TORQUE = 0.5, 1.5
#: Total force up to this many body weights is plausible support; the confidence halves one body weight above.
LOAD_CAP = 3.0


def report_joints(frame_parents: Sequence[int], joint_names: Sequence[str], parents: Sequence[int]) -> np.ndarray:
    """The joint each contact frame reports at, ``(F,)``: its parent joint, or the wrist for a frame of a hand."""
    wrists = {joint for joint, name in enumerate(joint_names) if name in ("left_wrist", "right_wrist")}
    reports = []
    for parent in frame_parents:
        joint = parent
        while joint >= 0 and joint not in wrists:
            joint = parents[joint]
        reports.append(joint if joint in wrists else parent)
    return np.array(reports, np.int64)


def fold_contacts(frame_contact: np.ndarray, report_joint: np.ndarray, joints: int) -> np.ndarray:
    """Frame labels ``(..., F)`` on their report joints ``(..., joints)``: a joint touches where any frame does."""
    contact = np.zeros((*frame_contact.shape[:-1], joints), bool)
    for joint in np.unique(report_joint):
        contact[..., joint] = frame_contact[..., report_joint == joint].any(axis=-1)
    return contact


def fold_drift(frame_drift: np.ndarray, report_joint: np.ndarray, joints: int) -> np.ndarray:
    """The smallest drift among a joint's frames, ``(..., joints)``: a hand whose palm held still was still.

    NaN where none of the joint's frames has a drift, and at joints that own no frame.
    """
    drift = np.full((*frame_drift.shape[:-1], joints), np.nan, np.float32)
    for joint in np.unique(report_joint):
        own = frame_drift[..., report_joint == joint]
        stillest = np.where(np.isnan(own), np.inf, own).min(axis=-1)
        drift[..., joint] = np.where(np.isnan(own).all(axis=-1), np.nan, stillest)
    return drift


def transition_seconds(joint_contact: np.ndarray, tracked: np.ndarray, fps: float) -> np.ndarray:
    """Seconds from each frame to the nearest change of the joint's own label, ``(T, J)``.

    A change between two tracked neighbours sits halfway between them, so both are 0 s away. A tracking gap and
    the ends of the clip are no change; a joint whose label never changes is infinitely far from one.
    """
    frame = np.arange(len(joint_contact))
    both = tracked[1:] & tracked[:-1]
    seconds = np.full(joint_contact.shape, np.inf)
    for joint in range(joint_contact.shape[1]):
        changes = np.flatnonzero(both & (joint_contact[1:, joint] != joint_contact[:-1, joint])) + 0.5
        if len(changes):
            seconds[:, joint] = np.abs(frame[:, None] - changes).min(axis=1) - 0.5
    return (seconds / fps).astype(np.float32)


def witness_features(drift: np.ndarray, seconds: np.ndarray, *, drift_floor_m: float,
                     transition_cap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """The two fitted inputs of a head: the one place their transforms live, for the readout and the calibration."""
    with np.errstate(invalid="ignore"):
        stillness = np.log(drift.astype(np.float64) + drift_floor_m)
    return stillness, np.log1p(TRANSITION_RATE * np.minimum(seconds, transition_cap_s))


def label_confidence(joint_contact: np.ndarray, drift: np.ndarray, tracked: np.ndarray, fps: float,
                     joint_names: Sequence[str], coefficients: dict) -> np.ndarray:
    """The probability that each joint label agrees with an annotator, ``(T, J)``, for one person.

    ``joint_contact (T, J)`` and ``drift (T, J)`` are the folded labels and drifts, ``coefficients`` is
    ``[contact_confidence]``. NaN, no statement, on an untracked frame, where the drift is unknown, and at every
    joint outside :data:`REPORT_GROUPS`.
    """
    stillness, transition = witness_features(
        drift, transition_seconds(joint_contact, tracked, fps), drift_floor_m=coefficients["drift_floor_m"],
        transition_cap_s=coefficients["transition_cap_s"])
    confidence = np.full(joint_contact.shape, np.nan)
    for joint, name in enumerate(joint_names):
        if name in REPORT_GROUPS:
            for head, rows in (("contact", joint_contact[:, joint]), ("free", ~joint_contact[:, joint])):
                fitted = coefficients[head]
                logit = (fitted["intercept"] + fitted[REPORT_GROUPS[name]]
                         + fitted["log_drift"] * stillness[rows, joint]
                         + fitted["log_transition"] * transition[rows, joint])
                confidence[rows, joint] = 1.0 / (1.0 + np.exp(-logit))
    confidence[~tracked] = np.nan
    confidence[~np.isfinite(drift)] = np.nan
    return confidence.astype(np.float32)


def force_confidence(root: np.ndarray, forces: np.ndarray, tracked: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """How far to trust the solved forces of each person in each frame, ``(P, T)`` in ``[0, 1]``.

    ``root (P, T, 6)`` is the root residual after the solve (N, N m), ``forces (P, T, F, 3)`` the contact forces
    (N), ``weight (P,)`` the body weights (N). Three things each cost confidence along ``1 / (1 + (x / x*)^2)``,
    and the result is their product: the residual force (where the solve could not balance the motion, what it
    did assign is an allocation of error), the residual torque, and total force beyond what a person supports,
    which explains reconstruction error and not motion. 0 on untracked frames.
    """
    weight = weight[:, None]
    excess = np.maximum(np.linalg.norm(forces, axis=-1).sum(axis=-1) / weight - LOAD_CAP, 0.0)
    confidence = 1.0 / ((1.0 + np.square(np.linalg.norm(root[..., :3], axis=-1) / (weight * RESIDUAL_FORCE)))
                        * (1.0 + np.square(np.linalg.norm(root[..., 3:], axis=-1) / (weight * LEVER * RESIDUAL_TORQUE)))
                        * (1.0 + np.square(excess)))
    return np.where(tracked, confidence, 0.0).astype(np.float32)
