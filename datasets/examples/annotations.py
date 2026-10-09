"""The example clips' manual contact annotations: the joint schemas, the frame map, the files.

The annotated examples (``<data root>/examples/data{,_2}/<clip>.npz``) label contact per video
frame on coarse joints, in the legacy 10-joint schema or the 14-joint one. :func:`ann_mapping`
picks the schema from the exact joint set; :data:`ANN_TO_SMPLX` / :data:`ANN_TO_SMPLX_14` give
each annotated joint's SMPLXMid joints and :data:`ANN_TO_FRAMES` its contact frames;
:func:`annotation_frame_labels` puts one person's annotation on the contact-frame axis;
:func:`resolve_annotation_path` finds a clip's file. ``score_frame_contacts.py`` scores the
pipeline's contact-frame labels against these annotations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bvr.paths import EXAMPLES

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


#: Which contact frames one coarse annotation joint covers — the inverse of the fold
#: ``datasets/examples/score_frame_contacts.py`` scores with. Used to put MANUAL per-joint
#: annotations on the frame axis for the physics stages, whose contact points ARE the
#: frames. Names cover BOTH frame sets (legacy ``toe_under`` and the full-body set's
#: ``ball_* / big_toe`` / elbow / forearm / shoulder / hip_side frames);
#: :func:`annotation_frame_labels` keeps only the names the body actually carries.
#: ``hip`` maps to the greater-trochanter frame only — the sit bones, torso and head
#: frames have no annotation joint and stay unlabeled.
ANN_TO_FRAMES: dict[str, tuple[str, ...]] = {
    "left_hand": ("palm_left", "finger_left", "palm_distal_left", "thumb_left"),
    "right_hand": ("palm_right", "finger_right", "palm_distal_right", "thumb_right"),
    "left_knee": ("knee_left",),
    "right_knee": ("knee_right",),
    # 14-joint schema: "foot" means the forefoot, "ankle" the heel. In the 10-joint
    # schema "foot" is the whole foot, so the heel joins it (see :func:`annotation_frame_labels`).
    "left_foot": ("toe_under_left", "toe_above_left", "ball_inner_left", "ball_outer_left",
                  "big_toe_left"),
    "right_foot": ("toe_under_right", "toe_above_right", "ball_inner_right",
                   "ball_outer_right", "big_toe_right"),
    "left_ankle": ("heel_left",),
    "right_ankle": ("heel_right",),
    "left_elbow": ("elbow_left", "forearm_left"),
    "right_elbow": ("elbow_right", "forearm_right"),
    "left_shoulder": ("shoulder_left",),
    "right_shoulder": ("shoulder_right",),
    "left_hip": ("hip_side_left",),
    "right_hip": ("hip_side_right",),
}


def annotation_frame_labels(contacts_p: np.ndarray, ann_names: list[str],
                            frame_names: list[str], N: int) -> np.ndarray:
    """Expand one person's ``(J_ann, N)`` annotation onto the contact frames ``(N, F)``.

    Tri-state in, tri-state out: an annotated joint's value is copied onto every contact
    frame it covers (:data:`ANN_TO_FRAMES`); a frame no annotated joint covers stays
    ``-1`` (unlabeled). The schema is resolved by :func:`ann_mapping`, and in the
    10-joint schema — where ``foot`` means the whole foot — the heel inherits the foot's
    label too.
    """
    mapping = ann_mapping(ann_names)          # validates the schema (raises on anything else)
    if contacts_p.shape[1] != N:
        raise ValueError(f"annotation has {contacts_p.shape[1]} frames but the body has {N}")
    ten_joint = mapping is ANN_TO_SMPLX
    frame_idx = {n: i for i, n in enumerate(frame_names)}
    ann_idx = {n: i for i, n in enumerate(ann_names)}
    labels = np.full((N, len(frame_names)), -1, dtype=np.int8)
    for ann_name, frames in ANN_TO_FRAMES.items():
        if ann_name not in ann_idx:
            continue
        if ten_joint and ann_name.endswith("_ankle"):
            continue                                    # no ankle label in the 10-joint schema
        covered = list(frames)
        if ten_joint and ann_name.endswith("_foot"):
            covered.append(f"heel_{ann_name.split('_')[0]}")
        series = contacts_p[ann_idx[ann_name]]
        for name in covered:
            if name in frame_idx:
                labels[:, frame_idx[name]] = series
    return labels


def resolve_annotation_path(source, annotations) -> Path:
    """The annotation npz: explicit ``--annotations``, else ``<source>.npz`` / ``data/<stem>.npz`` / ``data_2/<stem>.npz``."""
    if annotations is not None:
        return Path(annotations)
    src = Path(source)
    cand = src.with_suffix(".npz")
    if cand.exists():
        return cand
    legacy = EXAMPLES / "data" / f"{src.stem}.npz"
    if legacy.exists():
        return legacy
    return EXAMPLES / "data_2" / f"{src.stem}.npz"
