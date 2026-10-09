"""How much a contact label can be trusted — one calibrated number per joint.

``estimate_contacts`` writes a binary label for every contact frame. This module
answers the smaller question a learner needs: **how often is a label like this one
the same as a human annotator's?** The answer is a probability in ``[0, 1]``,
fitted on the manually annotated corpus scenes, never a hand-built ceiling. The
binary label is never changed.

The number is computed per **report joint** — the joint a contact frame folds onto
(every hand frame reports at its wrist), because that is the unit people annotate
and the unit the learner trains on. Three witnesses go in:

* **which joint it is** — hand, foot (the forefoot/toes), ankle (the heel) or knee;
* **how still it was** — the joint's net displacement over the readout window,
  measured before any contact enforcement acted (``drift_pre``);
* **how far it is from a label change** — seconds to the nearest frame where this
  joint's own label flips. Labels are least reliable right where they switch.

One logistic head per label class (the label says contact / the label says free)::

    x_drift = log(drift_m + drift_floor_m)
    x_trans = log1p(30 * min(seconds_to_nearest_label_change, transition_cap_s))
    logit   = intercept + group_offset + log_drift * x_drift + log_transition * x_trans
    conf    = 1 / (1 + exp(-logit))

The coefficients live in ``[contact_confidence]`` in the human-optimization TOML and
are refitted by ``datasets/climbingvideos/calibrate_contact_confidence.py``. See
``docs/contact_confidence.md``.

Everything else was tried and dropped. On the annotated corpus rows — one per
(scene, person, frame, joint) — the signed distance to the scene, the solved force share, the Sapiens
keypoint score, the limb's reprojection residual, image and body-occlusion
visibility, the depth-fit trust and the per-clip quality metrics each added at most
0.01 AUROC on top of these three — and several of them are the label's own decision
inputs, so they would endorse it in a circle. The three that stayed reach
video-grouped cross-validated AUROC 0.82 (contact labels) / 0.87 (free labels) with
a calibration error under 0.02.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Artifact version of the label confidence. ``source-conditional-v1`` was the
#: hand-built visibility × capped-evidence algebra this replaced.
CONTACT_CONFIDENCE_METHOD = "calibrated-v2"

#: Report joint → the group whose coefficients it uses. Left and right share one
#: group. Every joint outside this table is uncalibrated: the stage makes no
#: statement there and writes ``NaN`` (elbows, hips, torso, head …).
REPORT_GROUPS: dict[str, str] = {
    "left_wrist": "hand", "right_wrist": "hand",
    "left_foot": "foot", "right_foot": "foot",
    "left_ankle": "ankle", "right_ankle": "ankle",
    "left_knee": "knee", "right_knee": "knee",
}

#: Groups in the order the calibration script writes them.
GROUP_NAMES = ("hand", "foot", "ankle", "knee")

#: The transition feature is measured in 30-fps-equivalent frames, so a clip's own
#: frame rate cannot change the meaning of the fitted coefficient.
TRANSITION_RATE_HZ = 30.0

_HEAD_KEYS = ("intercept", "log_drift", "log_transition", *GROUP_NAMES)


def frame_report_joints(
    frame_parents: Sequence[int],
    joint_names: Sequence[str],
    parents: np.ndarray,
) -> np.ndarray:
    """Map each contact frame to the joint that reports it ``(F,)``.

    A frame reports at its parent joint, with one exception: every frame on a hand
    (palms, fingertips, thumbs — anything whose parent chain passes through a wrist)
    reports at that wrist, the joint we usually call "the hand". Downstream a hand is
    one coarse contact state, so its many frames must land on one joint. Implemented
    as an ancestor walk up the kinematic ``parents`` tree, so no name list of hand
    frames is needed.
    """
    names = [str(n) for n in joint_names]
    wrists = {i for i, n in enumerate(names) if n in ("left_wrist", "right_wrist")}
    out = []
    for parent in frame_parents:
        k = int(parent)
        while k >= 0 and k not in wrists:
            k = int(parents[k])
        out.append(k if k in wrists else int(parent))
    return np.asarray(out, dtype=np.int64)


def fold_contacts(
    frame_contact: np.ndarray,  # (..., F) bool — final frame labels
    report_joint: np.ndarray,   # (F,) — from :func:`frame_report_joints`
    num_joints: int,
) -> np.ndarray:
    """OR the frame labels onto their report joints ``(..., num_joints)`` bool.

    A joint is in contact when ANY of its frames is. A joint that owns no frame
    stays ``False``.
    """
    report_joint = np.asarray(report_joint)
    contact = np.zeros((*frame_contact.shape[:-1], num_joints), dtype=bool)
    for j in np.unique(report_joint):
        contact[..., j] = frame_contact[..., report_joint == j].any(axis=-1)
    return contact


def fold_drift(
    frame_drift_m: np.ndarray,  # (..., F) net displacement per contact frame, NaN where unknown
    report_joint: np.ndarray,   # (F,) — from :func:`frame_report_joints`
    num_joints: int,
) -> np.ndarray:
    """Smallest drift among a joint's frames ``(..., num_joints)`` float32.

    The stillest of a joint's frames speaks for the joint: a hand whose palm barely
    moved was still, whatever a fingertip did. ``NaN`` where every frame is ``NaN``
    (an untracked window) and at joints that own no frame.
    """
    report_joint = np.asarray(report_joint)
    drift = np.full((*frame_drift_m.shape[:-1], num_joints), np.nan, dtype=np.float32)
    for j in np.unique(report_joint):
        values = np.asarray(frame_drift_m)[..., report_joint == j]
        missing = np.isnan(values).all(axis=-1)
        smallest = np.where(np.isnan(values), np.inf, values).min(axis=-1)
        drift[..., j] = np.where(missing, np.nan, smallest)
    return drift


def transition_seconds(
    joint_contact: np.ndarray,  # (N, J) bool — the folded label
    valid: np.ndarray,          # (N,) bool — tracked frames
    fps: float,
) -> np.ndarray:
    """Seconds from each frame to the nearest change of that joint's own label ``(N, J)``.

    A change between two consecutive tracked frames ``k-1`` and ``k`` sits halfway
    between them, so both of those frames are 0 s away and the distance then grows by
    one frame at a time. An untracked gap is not a change, and neither is the start or
    the end of the track — only a real flip counts. A joint whose label never changes
    is ``inf`` seconds away from one.
    """
    contact = np.asarray(joint_contact, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if contact.ndim != 2 or valid.shape != contact.shape[:1]:
        raise ValueError(
            f"expected joint_contact (N,J) and valid (N,), got {contact.shape} and {valid.shape}")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    index = np.arange(contact.shape[0], dtype=np.float64)
    tracked_pair = valid[1:] & valid[:-1]
    out = np.full(contact.shape, np.inf, dtype=np.float64)
    for j in range(contact.shape[1]):
        edges = np.flatnonzero(tracked_pair & (contact[1:, j] != contact[:-1, j])) + 1
        if edges.size:
            distance = np.abs(index[:, None] - (edges[None, :] - 0.5)).min(axis=1) - 0.5
            out[:, j] = np.maximum(distance, 0.0)
    return (out / float(fps)).astype(np.float32)


def witness_features(
    drift_m: np.ndarray,          # net displacement over the readout window, metres
    seconds: np.ndarray,          # seconds to the nearest label change (inf = never)
    drift_floor_m: float,
    transition_cap_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The two fitted inputs ``(x_drift, x_trans)`` — the ONE place the transforms live.

    The stage, the offline recompute and the calibration script all call this, so a
    coefficient can never be fitted on one transform and applied through another.
    """
    with np.errstate(invalid="ignore"):
        x_drift = np.log(np.asarray(drift_m, dtype=np.float64) + drift_floor_m)
    x_trans = np.log1p(TRANSITION_RATE_HZ * np.minimum(seconds, transition_cap_s))
    return x_drift, x_trans


def confidence_coefficients(section: dict) -> dict:
    """Validate the ``[contact_confidence]`` TOML section and return it as plain floats.

    Both heads (``contact`` and ``free``) must be present, each with an intercept, the
    two feature weights and one offset per group, all finite.
    """
    if not isinstance(section, dict) or not section:
        raise ValueError("contact_confidence section is empty")
    out: dict = {}
    for key in ("drift_floor_m", "transition_cap_s"):
        if key not in section:
            raise ValueError(f"contact_confidence.{key} is missing")
        value = float(section[key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"contact_confidence.{key} must be finite and > 0")
        out[key] = value
    out["fit"] = str(section.get("fit", ""))
    for head in ("contact", "free"):
        if head not in section:
            raise ValueError(f"contact_confidence.{head} head is missing — "
                             "run datasets/climbingvideos/calibrate_contact_confidence.py --write")
        values = {}
        for key in _HEAD_KEYS:
            if key not in section[head]:
                raise ValueError(f"contact_confidence.{head}.{key} is missing")
            values[key] = float(section[head][key])
            if not np.isfinite(values[key]):
                raise ValueError(f"contact_confidence.{head}.{key} must be finite")
        out[head] = values
    return out


def label_confidence(
    joint_contact: np.ndarray,  # (N, J) bool — the folded label
    drift_m: np.ndarray,        # (N, J) net displacement over the readout window
    valid: np.ndarray,          # (N,) bool — tracked frames
    fps: float,
    joint_names: Sequence[str],
    coeffs: dict,
) -> np.ndarray:
    """P(the label agrees with a human annotator) per joint and frame ``(N, J)`` float32.

    ``NaN`` — no statement — at an untracked frame, where the drift is unknown, and at
    every joint outside :data:`REPORT_GROUPS`. ``coeffs`` comes from
    :func:`confidence_coefficients`.
    """
    contact = np.asarray(joint_contact, dtype=bool)
    drift = np.asarray(drift_m, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if drift.shape != contact.shape:
        raise ValueError(f"drift_m must have shape {contact.shape}, got {drift.shape}")
    if len(joint_names) != contact.shape[1]:
        raise ValueError(f"expected {contact.shape[1]} joint names, got {len(joint_names)}")

    x_drift, x_trans = witness_features(drift, transition_seconds(contact, valid, fps),
                                        coeffs["drift_floor_m"], coeffs["transition_cap_s"])

    out = np.full(contact.shape, np.nan)
    for j, name in enumerate(joint_names):
        group = REPORT_GROUPS.get(str(name))
        if group is None:
            continue
        for head, rows in (("contact", contact[:, j]), ("free", ~contact[:, j])):
            w = coeffs[head]
            logit = (w["intercept"] + w[group]
                     + w["log_drift"] * x_drift[rows, j]
                     + w["log_transition"] * x_trans[rows, j])
            out[rows, j] = 1.0 / (1.0 + np.exp(-logit))
    out[~valid] = np.nan
    out[~np.isfinite(drift)] = np.nan
    return out.astype(np.float32)
