"""Compare the contact readings we have on the ClimbingVideos **test split**.

Thirty scenes are held out of training and hand-annotated. Three different things
predict contact on them, and this puts all three on one ruler — the same scenes,
the same four limbs, the same manual labels:

- **our optimization** — ``human_optim/contacts_1.npz`` (key ``joint_contact``), the
  labels the contacts stage reads off the fitted body and the fused scene. No
  learning involved; it never saw an annotation.
- **the learned model** — ``features/predictions/<scene>/contacts.npz``, the
  image-conditioned network (``climb4_t5mid_v2``), thresholded at its own 0.5.
- **the force model** — ``features/predictions/<scene>/forces.npz``, a different
  network (``climb4_force_t7hinge``) that predicts how hard each limb pushes rather
  than whether it touches. A limb counts as in contact once it carries more than
  50 N. It predicts in body weights, so the scale comes from the body weight our own
  fit estimated (``contacts_1.npz``, ``body_weight_n``, 534–680 N over these scenes).
- **the blind model** — the same network with the image switched off
  (``climb4_t5mid_blind``, ``blind_to_image: true``), so it sees only body pose. Its
  predictions are not on disk, so its numbers are **read out of the training log**
  at the epoch its ``best.pth`` was saved. That is the one row this script does not
  recompute; everything about it is therefore approximate to 3 decimals and has no
  per-limb breakdown, because the trainer only wrote the pooled score.

A fourth row, **always contact**, predicts contact on every limb of every frame. It
is not a model — it is the floor. A climber is usually holding on, so this scores
far above zero, and any model worth the name has to clear it.

Ground truth is the manual annotation (``features/annotation/<scene>/annotation.npz``),
a 14-joint tri-state label per frame: 1 contact, 0 free, **-1 never annotated**. The
four limbs are folded exactly the way the learned model's own training data folds
them: a hand is its wrist (fingers already folded in), and a foot is the foot **or**
the ankle. A limb-frame counts only where the annotation actually says something.

Writes into ``<data_root>/results/``:

    contact_metrics.json     every count and rate, per limb and pooled
    table_overall.png        every row, pooled over the four limbs
    table_<limb>.png         one table per limb, for the three rows that have a
                             per-limb score (the floor is not interesting per limb,
                             and the blind model only ever had a pooled number)

Usage:
    uv run --no-sync python scripts/diagnostics/score_test_contacts.py
"""

from __future__ import annotations

import json
import re
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

TRAINING_OUTPUT = _REPO_ROOT.parent / "contact_anything" / "output"
BLIND_RUN = "climb4_t5mid_blind"

LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]
# Annotation labels each limb is built from. A foot is the toe or the ankle — either
# one touching the wall is a foot in contact (the learned model's own label fold).
LIMB_ANNOTATION = [("left_hand",), ("right_hand",),
                   ("left_foot", "left_ankle"), ("right_foot", "right_ankle")]
_FINGER = ("index", "middle", "pinky", "ring", "thumb")
CONTACT_THRESHOLD = 0.5  # the learned model's stored threshold, and the one the log evaluates at
FORCE_THRESHOLD_N = 50.0  # on the force model's downward load

ROWS = ["always", "blind", "optim", "learned", "force"]
# The per-limb tables drop the floor (uninteresting limb by limb) and the blind model
# (the trainer only ever wrote its pooled score).
PER_LIMB_ROWS = ["optim", "learned", "force"]
ROW_LABEL = {
    "always": "Always contact",
    "blind": "Blind model",
    "optim": "Optimization",
    "learned": f"Learned model  (p > {CONTACT_THRESHOLD})",
    "force": f"Force model  (load > {FORCE_THRESHOLD_N:.0f} N)",
}


def scene_dir(stage: str, scene: str) -> Path:
    """The sharded feature directory for one scene, e.g. ``annotation/CR/qu/CRquJ8H1QLI_0041``."""
    return FEATURES / stage / scene[:2] / scene[2:4] / scene


def test_scenes() -> list[str]:
    """The held-out scenes, taken as the ones the learned model has predictions for."""
    return sorted(path.parent.name for path in (FEATURES / "predictions").rglob("contacts.npz"))


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


def optim_contacts(scene: str) -> tuple[np.ndarray, list[int], np.ndarray]:
    """``(P, N, 4)`` bool from ``contacts_1.npz``, its person ids, and ``(P,)`` body weight."""
    contacts = np.load(scene_dir("human_optim", scene) / "contacts_1.npz", allow_pickle=True)
    joint_contact = np.asarray(contacts["joint_contact"], bool)
    members = _limb_members([str(name) for name in contacts["joint_names"]])
    folded = np.stack([joint_contact[..., group].any(axis=-1) for group in members], axis=-1)
    return (folded, [int(o) for o in contacts["object_ids"]],
            np.asarray(contacts["body_weight_n"], float))


def learned_contacts(scene: str) -> tuple[np.ndarray, list[int]]:
    """``(P, N, 4)`` bool from the learned model's probabilities plus its person ids."""
    predictions = np.load(scene_dir("predictions", scene) / "contacts.npz", allow_pickle=True)
    limbs = [str(name) for name in predictions["limbs"]]
    if limbs != LIMBS:
        raise SystemExit(f"[score] {scene}: prediction limbs {limbs} are not {LIMBS}")
    return np.asarray(predictions["probs"]) > CONTACT_THRESHOLD, \
        [int(o) for o in predictions["object_ids"]]


def force_contacts(scene: str, body_weight_n: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """``(P, N, 4)`` bool — a limb the force model loads past 50 N — plus its person ids.

    The load is the ``+y`` component in the local-world-aligned frame, where ``+y``
    points up: that is the part of the limb's force that holds the climber against
    gravity, which is what a hold actually carries. Verified on this corpus — the four
    limbs sum to a positive load on every frame, median 0.83 body weights.
    """
    forces = np.load(scene_dir("predictions", scene) / "forces.npz", allow_pickle=True)
    limbs = [str(name) for name in forces["limbs"]]
    if limbs != LIMBS:
        raise SystemExit(f"[score] {scene}: force limbs {limbs} are not {LIMBS}")
    load = np.asarray(forces["forces"])[..., 1] * body_weight_n[:, None, None]
    return load > FORCE_THRESHOLD_N, [int(o) for o in forces["object_ids"]]


def blind_from_log() -> dict:
    """The blind model's pooled test scores, read out of its training log.

    The trainer prints one ``test`` line per epoch and keeps the epoch with the best
    F1 as ``best.pth``; that is the checkpoint anyone would run, so that is the epoch
    reported here. Only the pooled score was written — there is no per-limb split.
    """
    logs = sorted(TRAINING_OUTPUT.glob(f"{BLIND_RUN}_*.log"))
    if not logs:
        raise SystemExit(f"[score] no training log for {BLIND_RUN} in {TRAINING_OUTPUT}")
    pattern = re.compile(
        r"^epoch\s+(\d+)\s+train.*?\n\s+test loss \S+\s+joint\[f1=([\d.]+) f2=[\d.]+ "
        r"p=([\d.]+) r=([\d.]+)\]", re.MULTILINE | re.DOTALL)
    text = logs[-1].read_text(errors="replace").replace("\r", "\n")
    epochs = [(float(f1), int(epoch), float(precision), float(recall))
              for epoch, f1, precision, recall in pattern.findall(text)]
    if not epochs:
        raise SystemExit(f"[score] found no test lines in {logs[-1]}")
    f1, epoch, precision, recall = max(epochs)
    return {"source": f"{logs[-1].name} (epoch {epoch}, the checkpoint saved as best.pth)",
            "epochs_logged": len(epochs),
            "overall": {"precision": precision, "recall": recall, "f1": f1,
                        "accuracy": None, "frames": None, "positives": None},
            "per_limb": {limb: {"precision": None, "recall": None, "f1": None} for limb in LIMBS}}


# ─────────────────────────────────── scoring ─────────────────────────────────

def collect() -> dict:
    """Score every test scene and return the full metrics tree."""
    counts = {row: {limb: dict(tp=0, fp=0, fn=0, tn=0) for limb in LIMBS}
              for row in ROWS if row != "blind"}
    scenes = test_scenes()
    checkpoint = ""

    for scene in scenes:
        optim, optim_ids, body_weight = optim_contacts(scene)
        learned, learned_ids = learned_contacts(scene)
        force, force_ids = force_contacts(scene, body_weight)
        if not optim_ids == learned_ids == force_ids:
            raise SystemExit(f"[score] {scene}: contacts_1 covers {optim_ids} but the "
                             f"predictions cover {learned_ids} / {force_ids}")
        if not optim.shape == learned.shape == force.shape:
            raise SystemExit(f"[score] {scene}: contacts_1 is {optim.shape}, the predictions "
                             f"are {learned.shape} / {force.shape}")
        checkpoint = checkpoint or str(np.load(
            scene_dir("predictions", scene) / "contacts.npz", allow_pickle=True)["checkpoint"])

        want, known = truth(scene, optim_ids)
        if want.shape != optim.shape:
            raise SystemExit(f"[score] {scene}: annotation is {want.shape}, body is {optim.shape}")
        predicted = {"optim": optim, "learned": learned, "force": force,
                     "always": np.ones_like(optim)}
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
        if row == "blind":
            rows[row] = {"rule": ROW_LABEL[row], **blind_from_log()}
            continue
        pooled = {key: sum(counts[row][limb][key] for limb in LIMBS)
                  for key in ("tp", "fp", "fn", "tn")}
        rows[row] = {"rule": ROW_LABEL[row],
                     "overall": _rates(pooled),
                     "per_limb": {limb: _rates(counts[row][limb]) for limb in LIMBS}}
    rows["learned"]["checkpoint"] = checkpoint
    rows["optim"]["source"] = "features/human_optim/<scene>/contacts_1.npz (joint_contact)"
    rows["force"]["source"] = ("features/predictions/<scene>/forces.npz, +y load in the "
                               "local-world-aligned frame, scaled by contacts_1 body_weight_n")

    return {
        "scenes": len(scenes),
        "scene_names": scenes,
        "ground_truth": "features/annotation/<scene>/annotation.npz — 14-joint manual "
                        "tri-state labels; unannotated limb-frames are dropped",
        "limb_fold": "hand = wrist (+fingers); foot = foot OR ankle",
        "contact_threshold": CONTACT_THRESHOLD,
        "force_threshold_newton": FORCE_THRESHOLD_N,
        "blind_caveat": "the blind row is quoted from its training log, not recomputed here: "
                        "3 decimals, pooled only, and scored on the trainer's own sampling "
                        "(the middle frame of every 5th window, 1669 of the 8281 frames) rather "
                        "than every frame. Running the learned model both ways puts that "
                        "difference at 0.001 of F1 (0.933 all frames, 0.935 sampled), and "
                        "reproduces the log to 0.0002, so the blind row is comparable to "
                        "about the third decimal",
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
    compared = [row for row in ROWS if row != "always"]  # a floor is not in the running
    overall_rows = [[ROW_LABEL[row].replace("  ", " "),
                     _fmt(metrics["rows"][row]["overall"]["precision"]),
                     _fmt(metrics["rows"][row]["overall"]["recall"]),
                     _fmt(metrics["rows"][row]["overall"]["f1"])] for row in ROWS]
    # One winner per metric column, over the model rows.
    bold = set().union(*(_best({(ROWS.index(row), column): overall_rows[ROWS.index(row)][column]
                                for row in compared}) for column in (1, 2, 3)))
    overall_path = RESULTS / "table_overall.png"
    _render_table(overall_path,
                  [("Contact reading", 3.7, "l"), ("P", 0.9, "r"), ("R", 0.9, "r"),
                   ("F1", 0.9, "r")],
                  overall_rows, bold=frozenset(bold))

    blocks = []
    for limb in LIMBS:
        limb_rows = [[ROW_LABEL[row].replace("  ", " "),
                      *(_fmt(metrics["rows"][row]["per_limb"][limb][key])
                        for key in ("precision", "recall", "f1"))] for row in PER_LIMB_ROWS]
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
        frames = f"{entry['frames']:,} limb-frames" if entry["frames"] else "from the training log"
        print(f"[score] {ROW_LABEL[row]:42s} P {_fmt(entry['precision'])}  "
              f"R {_fmt(entry['recall'])}  F1 {_fmt(entry['f1'])}  over {frames}")
    for path in [json_path, *paths]:
        print(f"[score] wrote {path}")
