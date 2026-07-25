"""Score the PIPELINE's contacts against the campus_board force plates.

``score_prediction_contacts.py`` scores the two external models. This script adds
the two things our own reconstruction produces, on exactly the same ground truth
and with exactly the same bookkeeping, so all four sit in one table:

- **pipeline contacts** — the per-joint labels the contacts stage wrote
  (``human_optim/contacts_1.npz``, key ``joint_contact``), folded to four limbs: a
  hand is in contact when its wrist or any of its fingers is, a foot when the foot
  or the ankle is. This is the label the force solve is given.
- **pipeline force** — the downward load ``kindyn_1`` puts on the hold, thresholded
  at 50 N, the same rule that turned the plate readings into the ground truth. A
  limb's load is the sum over its force groups (a hand is the wrist; a foot is the
  foot **and** the ankle, which are two separate contact points).

"Downward load" is one number per limb: the part of the contact force that points
along gravity, ``-(f . g_hat)``. The plates measure one axis, so this is the only
comparison the two sides can share.

Ground truth and the choice of which limbs to score are unchanged: only limbs that
had a plate count, and session 1 of campus_board instrumented the hands alone.

Writes into ``out_campus_board_single/results/`` (new names, next to the model
tables, which are left alone):

    pipeline_metrics.json        every count and rate, four rows, per limb and pooled
    table_pipeline_overall.png   the four rows side by side
    table_pipeline_per_limb.png  the same split by limb

Usage:
    uv run --no-sync python scripts/diagnostics/score_pipeline_contacts.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_DIAGNOSTICS = Path(__file__).resolve().parent
if str(_DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(_DIAGNOSTICS))

from score_prediction_contacts import (  # noqa: E402
    GT_ROOT, LIMBS, MODELS, MODEL_LABEL, PRED_ROOT, RESULTS,
    _fmt, _rates, _render_table, _wired, collect,
)

FORCE_THRESHOLD_N = 50.0  # on the pipeline's downward load, matching the plate rule
CONTACTS_FILE = "human_optim/contacts_1.npz"
KINDYN_FILE = "human_optim/kindyn_1.npz"
PIPELINE_ROWS = ["pipeline_contacts", "pipeline_force"]
ROWS = MODELS + PIPELINE_ROWS
ROW_LABEL = {**MODEL_LABEL,
             "pipeline_contacts": "Pipeline contacts  (contacts_1 labels)",
             "pipeline_force": f"Pipeline force  (kindyn_1 load > {FORCE_THRESHOLD_N:.0f} N)"}
_FINGER = ("index", "middle", "pinky", "ring", "thumb")


def _limb_members(joint_names: list[str]) -> list[list[int]]:
    """Joint columns per limb, in ``LIMBS`` order — hand = wrist + fingers, foot = foot + ankle.

    Mirrors the contact stage's own fold (``tools.human_optim.dynamics._contact_catalog``)
    by name rather than by index, and without pulling torch into a CPU script.
    """
    index = {name: column for column, name in enumerate(joint_names)}
    hands = [[index[f"{side}_wrist"]]
             + [column for name, column in index.items()
                if name.startswith(f"{side}_") and any(finger in name for finger in _FINGER)]
             for side in ("left", "right")]
    feet = [[index[f"{side}_foot"], index[f"{side}_ankle"]] for side in ("left", "right")]
    return hands + feet


def _limb_load(kindyn: np.lib.npyio.NpzFile, person: int) -> np.ndarray:
    """``(N, 4)`` downward load in newtons per limb — the plate-comparable scalar.

    ``kindyn_1`` solves a force at six points: the two wrists, the two feet and the
    two ankles. A hand limb is its wrist; a foot limb is the foot plus the ankle,
    because a planted foot can push through both.
    """
    gravity = np.asarray(kindyn["gravity_world"], float)
    down = gravity / np.linalg.norm(gravity)
    groups = [str(name) for name in np.asarray(kindyn["contact_force_joints"]).reshape(-1)]
    load = -(np.asarray(kindyn["contact_forces"][person], float) @ down)  # (N, 6)
    columns = [["left_wrist"], ["right_wrist"], ["left_foot", "left_ankle"],
               ["right_foot", "right_ankle"]]
    return np.stack([load[:, [groups.index(name) for name in limb]].sum(1) for limb in columns], 1)


def _pipeline_predictions(clip: str) -> tuple[dict[str, np.ndarray], int, dict[str, str]]:
    """Both pipeline contact readings for one clip, plus the frame count and provenance."""
    tree = PRED_ROOT / clip
    contacts = np.load(tree / CONTACTS_FILE, allow_pickle=True)
    kindyn = np.load(tree / KINDYN_FILE, allow_pickle=True)
    frame_indices = np.load(tree / "geometry" / "transform.npz")["frame_indices"]
    n_frames = len(frame_indices)
    if not np.array_equal(frame_indices, np.arange(n_frames)):
        raise ValueError("pipeline frames are not the video frames — the plate truth cannot be aligned")

    contact_ids = [int(x) for x in contacts["object_ids"]]
    kindyn_ids = [int(x) for x in kindyn["object_ids"]]
    if contact_ids != kindyn_ids or len(contact_ids) != 1:
        raise ValueError(f"campus_board is one climber, but the tree holds {contact_ids} / {kindyn_ids}")

    joint_contact = np.asarray(contacts["joint_contact"][0], bool)  # (N, 52)
    members = _limb_members([str(name) for name in contacts["joint_names"]])
    predicted = {
        "pipeline_contacts": np.stack([joint_contact[:, limb].any(1) for limb in members], 1),
        "pipeline_force": _limb_load(kindyn, 0) > FORCE_THRESHOLD_N,
    }
    provenance = {
        "fusion_method": str(contacts["fusion_method"]) if "fusion_method" in contacts.files else "none",
        "contact_label_method": str(contacts["contact_label_method"]),
    }
    return predicted, n_frames, provenance


def collect_pipeline() -> dict:
    """Score the two pipeline rows over every clip and return their metrics tree."""
    counts = {row: {limb: dict(tp=0, fp=0, fn=0, tn=0) for limb in LIMBS} for row in PIPELINE_ROWS}
    clips: list[str] = []
    skipped: dict[str, str] = {}
    provenance: dict[str, Counter] = {"fusion_method": Counter(), "contact_label_method": Counter()}

    for clip in sorted(p.name for p in PRED_ROOT.iterdir() if (p / "predictions" / "contacts.npz").exists()):
        try:
            predicted, n_frames, sources = _pipeline_predictions(clip)
        except Exception as error:  # noqa: BLE001 — one bad clip must not kill the batch
            skipped[clip] = f"{type(error).__name__}: {error}"
            print(f"[score] {clip}: SKIPPED — {error}")
            continue
        truth = np.load(GT_ROOT / clip / "contacts.npz")["contacts"][:n_frames]
        for key, counter in provenance.items():
            counter[sources[key]] += 1
        for row in PIPELINE_ROWS:
            for limb_index, wired in enumerate(_wired(clip)):
                if not wired:
                    continue
                got, want = predicted[row][:len(truth), limb_index], truth[:, limb_index]
                cell = counts[row][LIMBS[limb_index]]
                cell["tp"] += int(np.sum(got & want))
                cell["fp"] += int(np.sum(got & ~want))
                cell["fn"] += int(np.sum(~got & want))
                cell["tn"] += int(np.sum(~got & ~want))
        clips.append(clip)

    metrics = {}
    for row in PIPELINE_ROWS:
        pooled = {key: sum(counts[row][limb][key] for limb in LIMBS) for key in ("tp", "fp", "fn", "tn")}
        metrics[row] = {
            "source": CONTACTS_FILE if row == "pipeline_contacts" else KINDYN_FILE,
            "rule": ROW_LABEL[row],
            "overall": _rates(pooled),
            "per_limb": {limb: _rates(counts[row][limb]) for limb in LIMBS},
        }
    return {
        "clips": len(clips),
        "clip_names": clips,
        "skipped": skipped,
        "contacts_tag": "contacts_1",
        "fusion_method": dict(provenance["fusion_method"]),
        "contact_label_method": dict(provenance["contact_label_method"]),
        "force_threshold_newton": FORCE_THRESHOLD_N,
        "rows": metrics,
    }


def write_tables(metrics: dict) -> list[Path]:
    """Render the two four-row tables and return their paths."""
    overall_rows = [[ROW_LABEL[row].replace("  ", " "),
                     _fmt(metrics["rows"][row]["overall"]["precision"]),
                     _fmt(metrics["rows"][row]["overall"]["recall"]),
                     _fmt(metrics["rows"][row]["overall"]["f1"]),
                     _fmt(metrics["rows"][row]["overall"]["accuracy"]),
                     f"{metrics['rows'][row]['overall']['frames']:,}"] for row in ROWS]
    overall_path = RESULTS / "table_pipeline_overall.png"
    _render_table(overall_path,
                  [("Contact reading", 3.5, "l"), ("P", 0.85, "r"), ("R", 0.85, "r"),
                   ("F1", 0.85, "r"), ("Acc", 0.85, "r"), ("Frames", 1.1, "r")],
                  overall_rows)

    limb_rows = []
    for limb in [*LIMBS, "overall"]:
        cells = [limb.replace("_", " ")]
        for row in ROWS:
            entry = (metrics["rows"][row]["overall"] if limb == "overall"
                     else metrics["rows"][row]["per_limb"][limb])
            cells += [_fmt(entry["precision"]), _fmt(entry["recall"]), _fmt(entry["f1"])]
        limb_rows.append(cells)

    limb_path = RESULTS / "table_pipeline_per_limb.png"
    _render_table(limb_path,
                  [("Limb", 1.9, "l")] + [(name, 0.8, "r") for name in ("P", "R", "F1")] * len(ROWS),
                  limb_rows,
                  groups=(("Contact model", 1, 3), ("Force model", 4, 6),
                          ("Pipeline contacts", 7, 9), ("Pipeline force", 10, 12)),
                  rule_before=len(LIMBS))
    return [overall_path, limb_path]


if __name__ == "__main__":
    models = collect()
    pipeline = collect_pipeline()
    metrics = {**pipeline, "rows": {**{row: models["models"][row] for row in MODELS}, **pipeline["rows"]},
               "ground_truth": models["ground_truth"], "scored_limbs": models["scored_limbs"],
               "contact_threshold": models["contact_threshold"],
               "model_clips": models["clips"]}
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "pipeline_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    paths = write_tables(metrics)
    print(f"[score] {metrics['clips']} clips,  contacts_1 fusion: "
          f"{', '.join(f'{k}x{v}' for k, v in metrics['fusion_method'].items())}")
    for row in ROWS:
        entry = metrics["rows"][row]["overall"]
        print(f"[score] {ROW_LABEL[row]:40s} P {_fmt(entry['precision'])}  R {_fmt(entry['recall'])}  "
              f"F1 {_fmt(entry['f1'])}  over {entry['frames']:,} limb-frames")
    for path in [json_path, *paths]:
        print(f"[score] wrote {path}")
