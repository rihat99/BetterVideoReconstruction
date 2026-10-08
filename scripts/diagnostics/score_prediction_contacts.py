"""Score both external models' contacts against the campus_board force plates.

Two models can each be read as a contact detector, and this compares them on the
same footing:

- the **contact model** (`predictions/contacts.npz`), thresholded at p > 0.3;
- the **force model** (`predictions/forces.npz`), which has no contact output of
  its own here, so a limb counts as in contact when the downward load it predicts
  passes 50 N — the very rule that turned the plate readings into the ground
  truth, so neither side gets an easier test. Its forces are never exactly zero,
  so a real threshold is needed rather than a non-zero test.

Ground truth is `campus_board/<clip>/contacts.npz`: the plate force resampled to
one value per video frame and thresholded at 50 N.

Only limbs that actually had a plate are scored. Session 1 instrumented the hands
alone (clips not named ``*_peter_*`` / ``*_gianna_*``), and a zero foot column
there means no sensor, not a free foot — scoring it would invent 7 000 frames of
free-foot ground truth that was never measured.

Writes into ``out_campus_board_single/results/``:

    contact_metrics.json    every count and rate, per limb and pooled
    table_overall.png       the two models side by side
    table_per_limb.png      the same split by limb

Usage:
    uv run python scripts/diagnostics/score_prediction_contacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.paths import PETER  # noqa: E402

GT_ROOT = PETER / "campus_board"
PRED_ROOT = PETER / "out_campus_board_single"
RESULTS = PRED_ROOT / "results"

LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]
CONTACT_THRESHOLD = 0.3  # on the contact model's probability
FORCE_THRESHOLD_N = 50.0  # on the force model's downward load, matching the plate rule
GRAVITY = 9.81
MODELS = ["contact_model", "force_model"]
MODEL_LABEL = {"contact_model": f"Contact model  (p > {CONTACT_THRESHOLD})",
               "force_model": f"Force model  (load > {FORCE_THRESHOLD_N:.0f} N)"}


def _wired(clip: str) -> np.ndarray:
    """Return ``(4,)`` bool — which limbs had a plate. Session 1 wired the hands only."""
    plates = np.ones(4, bool)
    if "peter" not in clip and "gianna" not in clip:
        plates[2:] = False
    return plates


def _rates(counts: dict[str, int]) -> dict[str, float | int | None]:
    """Turn a confusion count into precision, recall, F1 and accuracy.

    A rate whose denominator is zero is reported as ``None`` rather than 0.0 —
    "never predicted a contact" and "every contact was wrong" are not the same
    statement, and a JSON reader should be able to tell them apart.
    """
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {**counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            "frames": tp + fp + fn + tn,
            "positives": tp + fn}


def collect() -> dict:
    """Score every clip and return the full metrics tree."""
    counts = {model: {limb: dict(tp=0, fp=0, fn=0, tn=0) for limb in LIMBS} for model in MODELS}
    clips: list[str] = []
    checkpoints: dict[str, str] = {}

    for clip in sorted(p.name for p in PRED_ROOT.iterdir() if (p / "predictions" / "contacts.npz").exists()):
        contacts = np.load(PRED_ROOT / clip / "predictions" / "contacts.npz")
        forces = np.load(PRED_ROOT / clip / "predictions" / "forces.npz")
        truth = np.load(GT_ROOT / clip / "contacts.npz")["contacts"]
        mass = float(np.load(PRED_ROOT / clip / "human_optim" / "kindyn_1.npz")["total_mass"][0])
        checkpoints.setdefault("contact_model", str(contacts["checkpoint"]))
        checkpoints.setdefault("force_model", str(forces["checkpoint"]))

        n_frames = contacts["probs"].shape[1]
        predicted = {
            "contact_model": contacts["probs"][0] > CONTACT_THRESHOLD,
            # + F_y in the local-world-aligned frame is the downward load on the hold.
            "force_model": forces["forces"][0][..., 1] * mass * GRAVITY > FORCE_THRESHOLD_N,
        }
        truth = truth[:n_frames]
        for model in MODELS:
            for limb_index, wired in enumerate(_wired(clip)):
                if not wired:
                    continue
                got, want = predicted[model][:, limb_index], truth[:, limb_index]
                cell = counts[model][LIMBS[limb_index]]
                cell["tp"] += int(np.sum(got & want))
                cell["fp"] += int(np.sum(got & ~want))
                cell["fn"] += int(np.sum(~got & want))
                cell["tn"] += int(np.sum(~got & ~want))
        clips.append(clip)

    metrics = {}
    for model in MODELS:
        pooled = {key: sum(counts[model][limb][key] for limb in LIMBS) for key in ("tp", "fp", "fn", "tn")}
        metrics[model] = {
            "checkpoint": checkpoints[model],
            "rule": MODEL_LABEL[model],
            "overall": _rates(pooled),
            "per_limb": {limb: _rates(counts[model][limb]) for limb in LIMBS},
        }
    return {
        "clips": len(clips),
        "clip_names": clips,
        "ground_truth": "campus_board/<clip>/contacts.npz — plate force resampled per video frame, > 50 N",
        "scored_limbs": "only limbs with a plate; session 1 (clips without peter/gianna in the name) wired the hands only",
        "contact_threshold": CONTACT_THRESHOLD,
        "force_threshold_newton": FORCE_THRESHOLD_N,
        "models": metrics,
    }


_ROW_H, _MARGIN = 0.34, 0.16  # inches: one table row, and the blank border around a table


def _table_size(columns: list[tuple[str, float, str]], rows: list[list[str]],
                groups: tuple[tuple[str, int, int], ...] = ()) -> tuple[float, float]:
    """The ``(width, height)`` in inches a table occupies, border included."""
    widths = np.array([width for _, width, _ in columns], float)
    return (0.92 * widths.sum(),
            2 * _MARGIN + _ROW_H * (1 + bool(groups) + len(rows)) + 0.02)


def _draw_table(fig, block_left: float, block_top: float,
                columns: list[tuple[str, float, str]], rows: list[list[str]],
                groups: tuple[tuple[str, int, int], ...] = (), rule_before: int | None = None,
                bold: frozenset[tuple[int, int]] = frozenset()) -> None:
    """Draw one minimal paper-style table into ``fig``: horizontal rules only, no colour.

    ``block_left`` / ``block_top`` place the table's top-left corner, in inches from
    the figure's top-left, so several tables can share one figure. ``columns`` is
    ``(label, width, align)`` per column with widths in arbitrary units, ``groups``
    optional ``(label, first, last)`` spans drawn above the header, ``rule_before`` a
    row index to separate with a thin rule, and ``bold`` a set of ``(row, column)``
    cells to emphasise.
    """
    widths = np.array([width for _, width, _ in columns], float)
    edges = np.concatenate([[0.0], np.cumsum(widths / widths.sum())])
    row_h, margin = _ROW_H, _MARGIN
    fig_width, fig_height = fig.get_size_inches()
    block_width, _ = _table_size(columns, rows, groups)

    left = (block_left + margin) / fig_width
    span = (block_width - 2 * margin) / fig_width

    def place(text: str, column: int, y: float, size: float, weight: str = "normal", colour: str = "black") -> None:
        align = columns[column][2]
        anchor = {"l": edges[column], "r": edges[column + 1], "c": 0.5 * (edges[column] + edges[column + 1])}[align]
        pad = {"l": 0.004, "r": -0.004, "c": 0.0}[align]
        fig.text(left + span * (anchor + pad), y, text, fontsize=size, weight=weight, color=colour,
                 ha={"l": "left", "r": "right", "c": "center"}[align], va="center")

    def rule(y: float, thickness: float, first: int = 0, last: int | None = None) -> None:
        last = len(columns) - 1 if last is None else last
        fig.add_artist(plt.Line2D([left + span * edges[first], left + span * edges[last + 1]], [y, y],
                                  color="black", lw=thickness, transform=fig.transFigure))

    top = 1 - (block_top + margin) / fig_height
    step = row_h / fig_height
    rule(top, 1.5)
    cursor = top - 0.55 * step
    if groups:
        for label, first, last in groups:
            centre = 0.5 * (edges[first] + edges[last + 1])
            fig.text(left + span * centre, cursor, label, fontsize=10, ha="center", va="center")
            rule(cursor - 0.42 * step, 0.7, first, last)
        cursor -= step
    for column, (label, _, _) in enumerate(columns):
        place(label, column, cursor, 10)
    rule(cursor - 0.45 * step, 0.9)  # the rule sits between the rows; it does not take a row of its own

    for index, row in enumerate(rows):
        cursor -= step
        if index == rule_before:
            rule(cursor + 0.5 * step, 0.5)
        for column, cell in enumerate(row):
            emphasis = "bold" if index == rule_before or (index, column) in bold else "normal"
            place(cell, column, cursor, 10, emphasis)
    rule(cursor - 0.5 * step, 1.5)


def _render_table(path: Path, columns: list[tuple[str, float, str]], rows: list[list[str]],
                  groups: tuple[tuple[str, int, int], ...] = (), rule_before: int | None = None,
                  bold: frozenset[tuple[int, int]] = frozenset()) -> None:
    """Write one table to its own PNG, sized to fit it exactly."""
    fig = plt.figure(figsize=_table_size(columns, rows, groups), dpi=200)
    fig.patch.set_facecolor("white")
    _draw_table(fig, 0.0, 0.0, columns, rows, groups, rule_before, bold)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def write_tables(metrics: dict) -> list[Path]:
    """Render the two summary tables and return their paths."""
    overall_rows = [[MODEL_LABEL[model].replace("  ", " "),
                     _fmt(metrics["models"][model]["overall"]["precision"]),
                     _fmt(metrics["models"][model]["overall"]["recall"]),
                     _fmt(metrics["models"][model]["overall"]["f1"]),
                     _fmt(metrics["models"][model]["overall"]["accuracy"]),
                     f"{metrics['models'][model]['overall']['frames']:,}"] for model in MODELS]
    overall_path = RESULTS / "table_overall.png"
    _render_table(overall_path,
                  [("Model", 3.5, "l"), ("P", 0.85, "r"), ("R", 0.85, "r"),
                   ("F1", 0.85, "r"), ("Acc", 0.85, "r"), ("Frames", 1.1, "r")],
                  overall_rows)

    limb_rows = []
    for limb in LIMBS:
        cells = [limb.replace("_", " ")]
        for model in MODELS:
            entry = metrics["models"][model]["per_limb"][limb]
            cells += [_fmt(entry["precision"]), _fmt(entry["recall"]), _fmt(entry["f1"])]
        limb_rows.append(cells)
    pooled = ["overall"]
    for model in MODELS:
        entry = metrics["models"][model]["overall"]
        pooled += [_fmt(entry["precision"]), _fmt(entry["recall"]), _fmt(entry["f1"])]
    limb_rows.append(pooled)

    limb_path = RESULTS / "table_per_limb.png"
    _render_table(limb_path,
                  [("Limb", 1.9, "l")] + [(name, 0.8, "r") for name in ("P", "R", "F1")] * 2,
                  limb_rows,
                  groups=(("Contact model", 1, 3), ("Force model", 4, 6)),
                  rule_before=len(LIMBS))
    return [overall_path, limb_path]


if __name__ == "__main__":
    metrics = collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "contact_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    paths = write_tables(metrics)
    for model in MODELS:
        entry = metrics["models"][model]["overall"]
        print(f"[score] {MODEL_LABEL[model]:34s} P {_fmt(entry['precision'])}  "
              f"R {_fmt(entry['recall'])}  F1 {_fmt(entry['f1'])}  over {entry['frames']:,} limb-frames")
    for path in [json_path, *paths]:
        print(f"[score] wrote {path}")
