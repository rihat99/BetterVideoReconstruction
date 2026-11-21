"""Score contact predictions against the manual per-joint annotations.

The contact steps label the body's 35 named contact frames. The
manual annotations in ``<data root>/examples/data/<stem>.npz`` label coarse joints (a 10-joint
or a 14-joint schema). To compare them we fold each coarse joint's frames together with an OR — a
joint is predicted in contact if ANY of its frames is:

    hand  <- palm | finger                    (both schemas)
    knee  <- knee                             (both schemas)
    foot  <- all toe/ball frames              (14-joint schema: its "foot" means the forefoot)
    ankle <- heel                             (14-joint schema only)
    foot  <- toe/ball frames | heel           (10-joint schema: "foot" is the whole foot)

hip, elbow, shoulder and the torso/head frames are excluded from scoring, so the metric
stays comparable across frame sets.

Annotations are tri-state: 1 = contact, 0 = no contact, -1 = unlabeled. Only cells that
are 0 or 1 and whose frame is in the prediction's ``valid_mask`` are scored. Annotated
people flagged ``ignored`` are skipped. People are matched by ``object_ids``.

    uv run python bvr_datasets/examples/score_frame_contacts.py --predictions <run root> --contacts 1|2
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


from bvr import files, paths

#: The contact file each ``--contacts`` tag scores.
CONTACTS = {"1": paths.CONTACTS_1, "2": paths.CONTACTS_2}

#: The joints of the two annotation schemas. The 14-joint one adds ankles and shoulders, and its
#: "foot" is the forefoot.
SCHEMA_10 = frozenset({"left_hand", "right_hand", "left_foot", "right_foot", "left_knee", "right_knee",
                       "left_hip", "right_hip", "left_elbow", "right_elbow"})
SCHEMA_14 = SCHEMA_10 | {"left_ankle", "right_ankle", "left_shoulder", "right_shoulder"}

# Which contact frames fold onto which annotated coarse joint, per annotation schema. The
# elbow / shoulder / hip / torso frames stay unscored so the metric keeps meaning what it
# always meant.
_HANDS = {
    "left_hand": ("palm_left", "finger_left", "palm_distal_left", "thumb_left"),
    "right_hand": ("palm_right", "finger_right", "palm_distal_right", "thumb_right"),
}
_KNEES = {"left_knee": ("knee_left",), "right_knee": ("knee_right",)}
_TOES_L = ("toe_above_left", "ball_inner_left", "ball_outer_left", "big_toe_left")
_TOES_R = ("toe_above_right", "ball_inner_right", "ball_outer_right", "big_toe_right")
_FOLD_10: dict[str, tuple[str, ...]] = {
    **_HANDS,
    "left_foot": (*_TOES_L, "heel_left"),
    "right_foot": (*_TOES_R, "heel_right"),
    **_KNEES,
}
_FOLD_14: dict[str, tuple[str, ...]] = {
    **_HANDS,
    "left_foot": _TOES_L,
    "right_foot": _TOES_R,
    "left_ankle": ("heel_left",),
    "right_ankle": ("heel_right",),
    **_KNEES,
}

# Stable print order, and the coarse groups the pooled summary breaks out.
_JOINT_ORDER = ("left_hand", "right_hand", "left_foot", "right_foot",
                "left_ankle", "right_ankle", "left_knee", "right_knee")
_GROUPS: dict[str, tuple[str, ...]] = {
    "hands": ("left_hand", "right_hand"),
    "feet_ankles": ("left_foot", "right_foot", "left_ankle", "right_ankle"),
    "knees": ("left_knee", "right_knee"),
}


def fold_for_schema(ann_names: list[str]) -> tuple[dict[str, tuple[str, ...]], int]:
    """Pick the frame→coarse-joint fold for one annotation, by EXACT joint-name set.

    Returns ``(fold, schema)`` where ``schema`` is 10 or 14. Raises on any other name set
    rather than silently dropping joints.
    """
    names = set(ann_names)
    if names == SCHEMA_10:
        return _FOLD_10, 10
    if names == SCHEMA_14:
        return _FOLD_14, 14
    raise ValueError(f"unrecognized annotation schema {sorted(ann_names)}")


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _add_counts(into: dict[str, int], predicted: np.ndarray, annotated: np.ndarray) -> None:
    """Accumulate the confusion counts of one ``(M,)`` bool prediction vs ``(M,)`` bool truth."""
    into["tp"] += int(np.count_nonzero(predicted & annotated))
    into["fp"] += int(np.count_nonzero(predicted & ~annotated))
    into["fn"] += int(np.count_nonzero(~predicted & annotated))
    into["tn"] += int(np.count_nonzero(~predicted & ~annotated))


def _pool(rows: list[dict[str, int]]) -> dict[str, int]:
    out = _empty_counts()
    for row in rows:
        for key in out:
            out[key] += row[key]
    return out


def metrics(counts: dict[str, int]) -> dict[str, float | int | None]:
    """Precision / recall / F1 of a confusion count dict.

    A rate is ``None`` when its denominator is empty (nothing predicted / nothing annotated);
    F1 is ``None`` only when the joint was neither predicted nor annotated positive anywhere.
    """
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None and recall is None:
        f1 = None
    else:
        p, r = precision or 0.0, recall or 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def score_scene(scene_dir: Path, tag: str) -> dict:
    """Score one clip directory ``<run root>/<scene>/`` against ``<data root>/examples/data/<scene>.npz``:
    :func:`score_prediction` of its ``human_optim/contacts_<tag>.npz``."""
    pred = files.load_npz(scene_dir / CONTACTS[tag], "bvr-contacts")
    ann_path = paths.data_root() / "examples" / "data" / f"{scene_dir.name}.npz"
    if not ann_path.exists():
        raise FileNotFoundError(f"no manual annotation at {ann_path}")
    return score_prediction(pred, np.load(ann_path, allow_pickle=True))


def score_prediction(pred: Mapping[str, np.ndarray], ann: Mapping[str, np.ndarray]) -> dict:
    """Score one labelling against one manual annotation.

    ``pred`` holds ``frame_contact (P,N,F)``, ``valid_mask (P,N)``, ``object_ids (P,)`` and
    ``contact_frame_names (F,)`` (a contacts file, or the labels of a kindyn_1 file); ``ann`` holds
    ``contacts (P_ann,J_ann,N_ann)`` — note the annotation's time axis is LAST — ``joint_names``,
    ``object_ids`` and ``ignored``. Returns the per-joint counts pooled over the scored people.
    """
    frame_contact = pred["frame_contact"].astype(bool)          # (P, N, F)
    valid_mask = pred["valid_mask"].astype(bool)                # (P, N)
    pred_ids = [int(o) for o in pred["object_ids"]]
    frame_idx = {str(n): i for i, n in enumerate(pred["contact_frame_names"])}

    ann_names = [str(n) for n in ann["joint_names"]]
    ann_contacts = np.transpose(ann["contacts"].astype(np.int8), (0, 2, 1))  # (P_ann, N_ann, J_ann)
    ann_ids = [int(o) for o in ann["object_ids"]]
    ignored = ann["ignored"].astype(bool)
    fold, schema = fold_for_schema(ann_names)
    ann_col = {name: i for i, name in enumerate(ann_names)}

    num_frames_pred, num_frames_ann = frame_contact.shape[1], ann_contacts.shape[1]
    if num_frames_ann != num_frames_pred:
        raise ValueError(f"frame-count mismatch: prediction has {num_frames_pred} frames, "
                         f"annotation has {num_frames_ann} — refusing to trim")

    joints = {name: _empty_counts() for name in fold}
    scored_ids: list[int] = []
    skipped: list[int] = []
    for person, object_id in enumerate(pred_ids):
        if object_id not in ann_ids:
            skipped.append(object_id)
            continue
        ann_person = ann_ids.index(object_id)
        if bool(ignored[ann_person]):
            skipped.append(object_id)
            continue
        scored_ids.append(object_id)
        for name, frames in fold.items():
            cols = [frame_idx[f] for f in frames]
            predicted = np.any(frame_contact[person][:, cols], axis=1)
            labels = ann_contacts[ann_person][:, ann_col[name]]
            keep = (labels >= 0) & valid_mask[person]
            _add_counts(joints[name], predicted[keep], labels[keep] == 1)

    return {
        "schema": schema,
        "scored_object_ids": scored_ids,
        "skipped_object_ids": skipped,
        "uncovered_joints": sorted(set(ann_names) - set(fold)),
        "num_frames": int(num_frames_pred),
        "joints": joints,
        "pooled": _pool(list(joints.values())),
    }


def _find_scenes(predictions: list[Path], tag: str) -> list[Path]:
    """Expand each ``--predictions`` path into scene dirs holding a ``contacts_<tag>.npz``."""
    scenes: list[Path] = []
    for path in predictions:
        if (path / CONTACTS[tag]).exists():
            scenes.append(path)
        else:
            scenes += sorted(c for c in path.iterdir() if (c / CONTACTS[tag]).exists())
    return scenes


_HEADER = f"{'scene':22s} {'joint':12s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'TN':>6s} " \
          f"{'prec':>7s} {'rec':>7s} {'F1':>7s}"


def _row(scene: str, joint: str, counts: dict) -> str:
    m = metrics(counts)
    fmt = lambda v: "      -" if v is None else f"{v:7.3f}"  # noqa: E731
    return (f"{scene:22s} {joint:12s} {m['tp']:6d} {m['fp']:6d} {m['fn']:6d} {m['tn']:6d} "
            f"{fmt(m['precision'])}{fmt(m['recall'])}{fmt(m['f1'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, nargs="+", required=True,
                        help="clip dirs <run root>/<scene>/ or a run root holding them")
    parser.add_argument("--contacts", default="1", choices=tuple(CONTACTS), help="contacts_<tag>.npz")
    parser.add_argument("--json", type=Path, default=None, help="write the full numbers here")
    args = parser.parse_args()

    scenes = _find_scenes(args.predictions, args.contacts)
    report: dict[str, dict] = {"contacts_tag": args.contacts, "scenes": {}}
    uncovered: set[str] = set()

    print(_HEADER)
    print("-" * len(_HEADER))
    for scene_dir in scenes:
        result = score_scene(scene_dir, args.contacts)
        report["scenes"][scene_dir.name] = result
        uncovered |= set(result["uncovered_joints"])
        for name in _JOINT_ORDER:
            if name in result["joints"]:
                print(_row(scene_dir.name, name, result["joints"][name]))
        print(_row(scene_dir.name, "ALL", result["pooled"]))
        print()

    scored = list(report["scenes"].values())
    by_joint = {name: _pool([r["joints"][name] for r in scored if name in r["joints"]])
                for name in _JOINT_ORDER}
    by_group = {group: _pool([by_joint[n] for n in names]) for group, names in _GROUPS.items()}
    overall = _pool(list(by_joint.values()))

    print("=" * len(_HEADER))
    for name in _JOINT_ORDER:
        print(_row("POOLED", name, by_joint[name]))
    print("-" * len(_HEADER))
    for group, counts in by_group.items():
        print(_row("POOLED", group, counts))
    print(_row("POOLED", "ALL", overall))
    print(f"\nscenes scored: {len(scored)}    "
          f"uncovered (never scored): {', '.join(sorted(uncovered))}")

    report["pooled"] = {"by_joint": {n: metrics(c) for n, c in by_joint.items()},
                        "by_group": {g: metrics(c) for g, c in by_group.items()},
                        "all": metrics(overall)}
    report["uncovered_joints"] = sorted(uncovered)
    for result in scored:
        result["joints"] = {n: metrics(c) for n, c in result["joints"].items()}
        result["pooled"] = metrics(result["pooled"])
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
