"""Compare the contact readings we have on the ClimbingVideos **test split**.

Thirty scenes are held out of training and hand-annotated. Five readings of
contact are put on one ruler — the same scenes, the same four limbs, the same
manual labels:

- **Optimization** — what the geometric pipeline says on its own after
  ``kinematics_1``: wall distance, stillness and the support-force completion,
  **before** the image model's say is fused in. This is not a file on disk —
  ``contacts_1.npz`` is saved after the fusion — but the fusion records every
  limb-frame it flipped in ``fusion_changed``, so the pre-fusion reading is
  recovered exactly by flipping those limb-frames back.
- **Learned model** — ``features/predictions/<scene>/contacts.npz``, the
  image-conditioned network (``climb4_t5mid_v2``), thresholded at its own 0.5.
- **Optim + model** — ``contacts_1.npz`` as saved: the two witnesses above
  combined by adding their log-odds (see :mod:`tools.human_optim.fusion`).
- **Second optim + model** — ``contacts_2.npz``: the body is refined again with
  the fused labels (``kinematics_2``), and the same readout + fusion runs on
  that better body.
- **Supervised force model** — ``features/predictions/<scene>/forces_sup.npz``,
  a network trained to predict each limb's contact force (in body weights)
  straight from the video. A limb counts as in contact once it carries more than
  50 N; the scale comes from the body weight our own fit estimated
  (``contacts_1.npz``, ``body_weight_n``). A foot and its ankle are one limb on
  this ruler, so their force vectors are summed before thresholding — the
  dynamics targets split one foot's load across the two joints, and the sum
  undoes that split.

Ground truth is the manual annotation (``features/annotation/<scene>/annotation.npz``),
a 14-joint tri-state label per frame: 1 contact, 0 free, **-1 never annotated**. The
four limbs are folded exactly the way the learned model's own training data folds
them: a hand is its wrist (fingers already folded in), and a foot is the foot **or**
the ankle. A limb-frame counts only where the annotation actually says something.

Writes into ``<data_root>/results/``:

    contact_metrics.json     every count and rate, per limb and pooled
    table_overall.png        the four rows, pooled over the four limbs
    table_per_limb.png       the same four rows, one small table per limb

Usage:
    uv run --no-sync python scripts/diagnostics/score_test_contacts.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tomllib
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIAGNOSTICS = Path(__file__).resolve().parent
if str(_DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(_DIAGNOSTICS))

from score_prediction_contacts import (  # noqa: E402
    _draw_table, _fmt, _rates, _render_table, _table_size)

with (_REPO_ROOT / "configs" / "corpus.toml").open("rb") as _fh:
    DATA_ROOT = (_REPO_ROOT / tomllib.load(_fh)["corpus"]["data_root"]).resolve()
FEATURES = DATA_ROOT / "features"
RESULTS = DATA_ROOT / "results"

LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]
# Annotation labels each limb is built from. A foot is the toe or the ankle — either
# one touching the wall is a foot in contact (the learned model's own label fold).
LIMB_ANNOTATION = [("left_hand",), ("right_hand",),
                   ("left_foot", "left_ankle"), ("right_foot", "right_ankle")]
_FINGER = ("index", "middle", "pinky", "ring", "thumb")
CONTACT_THRESHOLD = 0.5  # the learned model's stored threshold
FORCE_THRESHOLD_N = 50.0  # on the supervised force model's per-limb load
# The supervised force model's channel order, asserted against the file.
SUP_LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot",
             "left_ankle", "right_ankle"]

ROWS = ["optim", "learned", "fused1", "fused2", "force_sup"]
ROW_LABEL = {
    "optim": "Optimization",
    "learned": f"Learned model  (p > {CONTACT_THRESHOLD})",
    "fused1": "Optim + model",
    "fused2": "Second optim + model",
    "force_sup": f"Supervised force  (|F| > {FORCE_THRESHOLD_N:.0f} N)",
}


def scene_dir(stage: str, scene: str) -> Path:
    """The sharded feature directory for one scene, e.g. ``annotation/CR/qu/CRquJ8H1QLI_0041``."""
    return FEATURES / stage / scene[:2] / scene[2:4] / scene


def test_scenes() -> list[str]:
    """The held-out scenes, from the scene DB's ``dataset_split`` column.

    Predictions now exist for the whole corpus, so "has a prediction file" no
    longer identifies the test split — the DB does.
    """
    db = sqlite3.connect(f"file:{DATA_ROOT / 'scenes' / 'scenes.db'}?mode=ro", uri=True)
    try:
        return sorted(row[0] for row in db.execute(
            "SELECT scene_id FROM scenes WHERE dataset_split = 'test'"))
    finally:
        db.close()


# ─────────────────────────────── ground truth ────────────────────────────────

def truth(scene: str, object_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Manual labels for one scene as ``(contact, known)``, each ``(P, N, 4)`` bool.

    ``known`` is False wherever the annotator never labelled that limb-frame; those
    are dropped from every count. A foot is in contact when the toe **or** the ankle
    is, and is known-free only when both are labelled free — a half-labelled foot
    stays unknown rather than being guessed at.
    """
    annotation = np.load(scene_dir("annotation", scene) / "annotation.npz", allow_pickle=True)
    names = [str(name) for name in annotation["joint_names"]]
    labels = np.asarray(annotation["contacts"], np.int8)          # (P_ann, J, N) tri-state
    annotated_ids = [int(o) for o in annotation["object_ids"]]
    ignored = np.asarray(annotation["ignored"], bool)

    contact = np.zeros((len(object_ids), labels.shape[2], len(LIMBS)), bool)
    known = np.zeros_like(contact)
    for person, object_id in enumerate(object_ids):
        if object_id not in annotated_ids:
            continue                                              # person never annotated
        row = annotated_ids.index(object_id)
        if ignored[row]:
            continue                                              # annotator marked the track junk
        for limb, members in enumerate(LIMB_ANNOTATION):
            values = np.stack([labels[row, names.index(name)] for name in members])  # (M, N)
            contact[person, :, limb] = (values == 1).any(axis=0)
            known[person, :, limb] = contact[person, :, limb] | (values != -1).all(axis=0)
    return contact, known


# ────────────────────────────── the predictions ──────────────────────────────

def _limb_members(joint_names: list[str]) -> list[list[int]]:
    """Joint columns per limb in ``LIMBS`` order — hand = wrist + fingers, foot = foot + ankle.

    Matched by name rather than by index, so a change to the 52-joint order cannot
    silently score the wrong joints.
    """
    column = {name: index for index, name in enumerate(joint_names)}
    hands = [[column[f"{side}_wrist"]]
             + [index for name, index in column.items()
                if name.startswith(f"{side}_") and any(finger in name for finger in _FINGER)]
             for side in ("left", "right")]
    feet = [[column[f"{side}_foot"], column[f"{side}_ankle"]] for side in ("left", "right")]
    return hands + feet


def folded_contacts(scene: str, tag: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """One saved ``contacts_<tag>.npz`` read at the limb level.

    :returns: ``(fused, changed, object_ids)`` — the saved (post-fusion) limb state
        ``(P, N, 4)`` bool, the fusion's flip record ``(P, N, 4)`` int8 (+1 turned
        the limb on, -1 turned it off, 0 untouched), and the person ids.
    """
    contacts = np.load(scene_dir("human_optim", scene) / f"contacts_{tag}.npz",
                       allow_pickle=True)
    limbs = [str(name) for name in contacts["pred_limbs"]]
    if limbs != LIMBS:
        raise SystemExit(f"[score] {scene}: contacts_{tag} limb order {limbs} is not {LIMBS}")
    joint_contact = np.asarray(contacts["joint_contact"], bool)
    members = _limb_members([str(name) for name in contacts["joint_names"]])
    fused = np.stack([joint_contact[..., group].any(axis=-1) for group in members], axis=-1)
    changed = np.asarray(contacts["fusion_changed"], np.int8)
    # The flip record must agree with the saved labels, or the pre-fusion
    # reconstruction below would be wrong.
    if (((changed == 1) & ~fused) | ((changed == -1) & fused)).any():
        raise SystemExit(f"[score] {scene}: fusion_changed disagrees with contacts_{tag} labels")
    return fused, changed, [int(o) for o in contacts["object_ids"]]


def learned_contacts(scene: str) -> tuple[np.ndarray, list[int]]:
    """``(P, N, 4)`` bool from the learned model's probabilities plus its person ids."""
    predictions = np.load(scene_dir("predictions", scene) / "contacts.npz", allow_pickle=True)
    limbs = [str(name) for name in predictions["limbs"]]
    if limbs != LIMBS:
        raise SystemExit(f"[score] {scene}: prediction limbs {limbs} are not {LIMBS}")
    return np.asarray(predictions["probs"]) > CONTACT_THRESHOLD, \
        [int(o) for o in predictions["object_ids"]]


def sup_force_contacts(scene: str) -> tuple[np.ndarray, list[int], str]:
    """``(P, N, 4)`` bool — a limb the supervised force model loads past 50 N.

    The model predicts a force vector per hand, toe and ankle in body weights,
    scaled to newtons by the fitted body's weight (``contacts_1.npz``). A foot and
    its ankle are one limb here, so their vectors are summed before taking the
    magnitude (magnitude is frame-independent, so the vectors are used as stored).

    :returns: the labels, the person ids, and the model checkpoint string.
    """
    forces = np.load(scene_dir("predictions", scene) / "forces_sup.npz", allow_pickle=True)
    limbs = [str(name) for name in forces["limbs"]]
    if limbs != SUP_LIMBS:
        raise SystemExit(f"[score] {scene}: forces_sup limbs {limbs} are not {SUP_LIMBS}")
    if str(forces["units"]) != "body_weight":
        raise SystemExit(f"[score] {scene}: forces_sup units {forces['units']!s} "
                         f"are not body_weight")
    with np.load(scene_dir("human_optim", scene) / "contacts_1.npz") as contacts:
        body_weight = np.asarray(contacts["body_weight_n"], np.float64)      # (P,)
    vec = np.asarray(forces["forces"], np.float64)                           # (P, N, 6, 3)
    united = np.concatenate([vec[..., :2, :], vec[..., 2:4, :] + vec[..., 4:6, :]], axis=-2)
    load = np.linalg.norm(united, axis=-1) * body_weight[:, None, None]      # (P, N, 4) N
    return (load > FORCE_THRESHOLD_N, [int(o) for o in forces["object_ids"]],
            str(forces["checkpoint"]))


# ─────────────────────────────────── scoring ─────────────────────────────────

def collect() -> dict:
    """Score every test scene and return the full metrics tree."""
    counts = {row: {limb: dict(tp=0, fp=0, fn=0, tn=0) for limb in LIMBS} for row in ROWS}
    scenes = test_scenes()
    checkpoint = ""
    sup_checkpoint = ""

    for scene in scenes:
        fused1, changed1, ids1 = folded_contacts(scene, 1)
        fused2, _, ids2 = folded_contacts(scene, 2)
        learned, learned_ids = learned_contacts(scene)
        sup, sup_ids, sup_ckpt = sup_force_contacts(scene)
        if not ids1 == ids2 == learned_ids == sup_ids:
            raise SystemExit(f"[score] {scene}: contacts cover {ids1} / {ids2} but the "
                             f"predictions cover {learned_ids} / {sup_ids}")
        if not fused1.shape == fused2.shape == learned.shape == sup.shape:
            raise SystemExit(f"[score] {scene}: contacts are {fused1.shape} / {fused2.shape}, "
                             f"the predictions are {learned.shape} / {sup.shape}")
        checkpoint = checkpoint or str(np.load(
            scene_dir("predictions", scene) / "contacts.npz", allow_pickle=True)["checkpoint"])
        sup_checkpoint = sup_checkpoint or sup_ckpt

        want, known = truth(scene, ids1)
        if want.shape != fused1.shape:
            raise SystemExit(f"[score] {scene}: annotation is {want.shape}, body is {fused1.shape}")
        predicted = {"optim": fused1 ^ (changed1 != 0),   # the exact pre-fusion reading
                     "learned": learned, "fused1": fused1, "fused2": fused2,
                     "force_sup": sup}
        for row, got in predicted.items():
            for limb_index, limb in enumerate(LIMBS):
                use = known[..., limb_index]
                hit, truth_ = got[..., limb_index][use], want[..., limb_index][use]
                cell = counts[row][limb]
                cell["tp"] += int(np.sum(hit & truth_))
                cell["fp"] += int(np.sum(hit & ~truth_))
                cell["fn"] += int(np.sum(~hit & truth_))
                cell["tn"] += int(np.sum(~hit & ~truth_))

    rows = {}
    for row in ROWS:
        pooled = {key: sum(counts[row][limb][key] for limb in LIMBS)
                  for key in ("tp", "fp", "fn", "tn")}
        rows[row] = {"rule": ROW_LABEL[row],
                     "overall": _rates(pooled),
                     "per_limb": {limb: _rates(counts[row][limb]) for limb in LIMBS}}
    rows["learned"]["checkpoint"] = checkpoint
    rows["force_sup"]["checkpoint"] = sup_checkpoint
    rows["optim"]["source"] = ("features/human_optim/<scene>/contacts_1.npz — limb fold of "
                               "joint_contact, flipped back where fusion_changed != 0 "
                               "(the exact pre-fusion labels)")
    rows["fused1"]["source"] = "features/human_optim/<scene>/contacts_1.npz (joint_contact)"
    rows["fused2"]["source"] = "features/human_optim/<scene>/contacts_2.npz (joint_contact)"
    rows["force_sup"]["source"] = ("features/predictions/<scene>/forces_sup.npz — |force| per "
                                   "limb (foot = toe + ankle vector sum), scaled by contacts_1 "
                                   "body_weight_n")

    return {
        "scenes": len(scenes),
        "scene_names": scenes,
        "ground_truth": "features/annotation/<scene>/annotation.npz — 14-joint manual "
                        "tri-state labels; unannotated limb-frames are dropped",
        "limb_fold": "hand = wrist (+fingers); foot = foot OR ankle",
        "contact_threshold": CONTACT_THRESHOLD,
        "force_threshold_newton": FORCE_THRESHOLD_N,
        "rows": rows,
    }


def _best(cells: dict[tuple[int, int], str]) -> set[tuple[int, int]]:
    """The ``(row, column)`` keys holding the largest value, comparing what is printed.

    Two cells that both read 0.933 are both best, whatever their hidden decimals say —
    bolding one of them would claim a difference the table does not show.
    """
    numbers = {key: float(text) for key, text in cells.items() if text != "—"}
    if not numbers:
        return set()
    return {key for key, value in numbers.items() if value == max(numbers.values())}


_LIMB_METRICS = [("P", 0.9, "r"), ("R", 0.9, "r"), ("F1", 0.9, "r")]


def _render_limb_figure(path: Path, blocks: list[tuple[str, list[list[str]], frozenset]]) -> None:
    """Draw one small table per limb into a single figure, two across and two down.

    Each block is its own table with its own rules and its own header — the limb name
    sits where a column heading would, so every table says what it is about.
    """
    columns = [(blocks[0][0], 2.7, "l"), *_LIMB_METRICS]
    width, height = _table_size(columns, blocks[0][1])
    gap_x, gap_y = 0.4, 0.2
    fig = plt.figure(figsize=(2 * width + gap_x, 2 * height + gap_y), dpi=200)
    fig.patch.set_facecolor("white")
    for index, (limb, rows, bold) in enumerate(blocks):
        _draw_table(fig, (index % 2) * (width + gap_x), (index // 2) * (height + gap_y),
                    [(limb, 2.7, "l"), *_LIMB_METRICS], rows, bold=bold)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def write_tables(metrics: dict) -> list[Path]:
    """Render the two summary tables and return their paths."""
    overall_rows = [[ROW_LABEL[row].replace("  ", " "),
                     _fmt(metrics["rows"][row]["overall"]["precision"]),
                     _fmt(metrics["rows"][row]["overall"]["recall"]),
                     _fmt(metrics["rows"][row]["overall"]["f1"])] for row in ROWS]
    # One winner per metric column.
    bold = set().union(*(_best({(index, column): overall_rows[index][column]
                                for index in range(len(ROWS))}) for column in (1, 2, 3)))
    overall_path = RESULTS / "table_overall.png"
    _render_table(overall_path,
                  [("Contact reading", 3.7, "l"), ("P", 0.9, "r"), ("R", 0.9, "r"),
                   ("F1", 0.9, "r")],
                  overall_rows, bold=frozenset(bold))

    blocks = []
    for limb in LIMBS:
        limb_rows = [[ROW_LABEL[row].replace("  ", " "),
                      *(_fmt(metrics["rows"][row]["per_limb"][limb][key])
                        for key in ("precision", "recall", "f1"))] for row in ROWS]
        bold = set().union(*(_best({(index, column): limb_rows[index][column]
                                    for index in range(len(limb_rows))})
                             for column in (1, 2, 3)))
        blocks.append((limb.replace("_", " "), limb_rows, frozenset(bold)))

    limb_path = RESULTS / "table_per_limb.png"
    _render_limb_figure(limb_path, blocks)
    return [overall_path, limb_path]


if __name__ == "__main__":
    metrics = collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "contact_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    paths = write_tables(metrics)
    print(f"[score] {metrics['scenes']} test scenes")
    for row in ROWS:
        entry = metrics["rows"][row]["overall"]
        print(f"[score] {ROW_LABEL[row]:32s} P {_fmt(entry['precision'])}  "
              f"R {_fmt(entry['recall'])}  F1 {_fmt(entry['f1'])}  "
              f"over {entry['frames']:,} limb-frames")
    for path in [json_path, *paths]:
        print(f"[score] wrote {path}")
