"""The example clips' manual contact annotations: the joint schemas.

The annotated examples (``<data root>/examples/data{,_2}/<clip>.npz``) label contact per video
frame on coarse joints, in the legacy 10-joint schema or the 14-joint one. :func:`ann_mapping`
picks the schema from the exact joint set; :data:`ANN_TO_SMPLX` / :data:`ANN_TO_SMPLX_14` give
each annotated joint's SMPLXMid joints. ``score_frame_contacts.py`` scores the pipeline's
contact-frame labels against these annotations.
"""
from __future__ import annotations

_FINGER = ("index", "middle", "pinky", "ring", "thumb")


def _hand_smplx_joints(side: str) -> list[str]:
    """The SMPLXMid joints one ``<side>_hand`` annotation covers: wrist + 15 fingers."""
    return [f"{side}_wrist"] + [f"{side}_{f}{k}" for f in _FINGER for k in (1, 2, 3)]


ANN_TO_SMPLX: dict[str, list[str]] = {
    "left_hand": _hand_smplx_joints("left"),
    "right_hand": _hand_smplx_joints("right"),
    "left_foot": ["left_ankle", "left_foot"],
    "right_foot": ["right_ankle", "right_foot"],
    "left_knee": ["left_knee"],
    "right_knee": ["right_knee"],
    "left_hip": ["left_hip"],
    "right_hip": ["right_hip"],
    "left_elbow": ["left_elbow"],
    "right_elbow": ["right_elbow"],
}

# The 14-joint schema (its own l/r ankle + shoulder labels, and its 'foot' means the TOE,
# not the whole ankle+foot region). Hands stay wrist+fingers; foot maps to the toe joint
# only (ankle is its own label); the rest map to their same-named SMPLXMid joints.
ANN_TO_SMPLX_14: dict[str, list[str]] = {
    "left_hand": _hand_smplx_joints("left"),
    "right_hand": _hand_smplx_joints("right"),
    "left_foot": ["left_foot"],
    "right_foot": ["right_foot"],
    "left_ankle": ["left_ankle"],
    "right_ankle": ["right_ankle"],
    "left_knee": ["left_knee"],
    "right_knee": ["right_knee"],
    "left_elbow": ["left_elbow"],
    "right_elbow": ["right_elbow"],
    "left_shoulder": ["left_shoulder"],
    "right_shoulder": ["right_shoulder"],
    "left_hip": ["left_hip"],
    "right_hip": ["right_hip"],
}


def ann_mapping(ann_names: list[str]) -> dict[str, list[str]]:
    """The annotation→SMPLXMid group mapping for this schema, by EXACT joint-name set.

    The ``ann_names`` set must match the legacy 10-joint schema (:data:`ANN_TO_SMPLX`) or the
    14-joint schema (:data:`ANN_TO_SMPLX_14`) exactly — a partial or misspelled annotation
    raises (with the observed names) rather than silently dropping the unmatched joints."""
    names = set(ann_names)
    if names == set(ANN_TO_SMPLX):
        return ANN_TO_SMPLX
    if names == set(ANN_TO_SMPLX_14):
        return ANN_TO_SMPLX_14
    raise ValueError(
        f"unrecognized contact annotation schema {sorted(ann_names)}; expected the 10-joint "
        f"{sorted(ANN_TO_SMPLX)} or the 14-joint {sorted(ANN_TO_SMPLX_14)} joint set")

