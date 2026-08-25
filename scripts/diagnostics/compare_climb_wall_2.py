"""Compare the optimization's forces (``kindyn_1``) and the learned supervised model
(``forces_sup``) against the climb_wall_2 boards, in one PNG table.

The demonstrator wall is the only rig where every limb has its own 3-axis board, so
it gives per-limb ground truth for forces AND contacts (a board above 50 N is "in
contact" — the same rule the campus plates used). One table: a row per method, a
column per metric, everything pooled over the four limbs of all seven trials.

The metrics: mean absolute error of the limb force size in newtons; the median angle
between the vectors (only where both sides read above 50 N — a near-zero vector has
no direction); the Pearson correlation of the sizes over time; the load-share error
(how far each limb's share of the summed force sits from the boards' shares, in
percentage points — this is where "the total is right but the limbs are wrong"
shows); and contact precision / recall / F1 against the board truth. The
optimization's contact claim is the label set ``kindyn_1`` actually solved with,
the model's its contact head above 0.5. The better method of each column is bold.

Writes ``peter/out_climb_wall_2_single/model_comparison.png``.

Usage:
    uv run --no-sync python scripts/diagnostics/compare_climb_wall_2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_DIAGNOSTICS = Path(__file__).resolve().parent
if str(_DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(_DIAGNOSTICS))

from render_kindyn_overlays import BOARD_CONTACT_N, WALL2_ROOT  # noqa: E402
from render_sup_force_overlays import (  # noqa: E402
    GRAVITY, GROUP_FOLD, SUP_CONTACT_PROB, SUP_GROUPS,
)
from score_pipeline_contacts import _limb_members  # noqa: E402

METRICS = ["MAE [N]", "Angle [°]", "Corr.", "Share [pp]", "Prec.", "Rec.", "F1"]
HIGHER_IS_BETTER = [False, False, True, False, True, True, True]


def _gather() -> dict[str, np.ndarray]:
    """Stack every trial's per-frame limb data into pooled arrays.

    Returns world-frame reaction vectors ``(T, 4, 3)`` in newtons for ``boards``,
    ``opt`` and ``model``, plus boolean contact claims ``(T, 4)`` for the two
    methods (``opt_contact``, ``model_contact``).
    """
    columns: dict[str, list[np.ndarray]] = {key: [] for key in
                                            ("boards", "opt", "model", "opt_contact", "model_contact")}
    trials = sorted(p for p in WALL2_ROOT.iterdir()
                    if (p / "predictions" / "forces_sup.npz").exists())
    for tree in trials:
        kindyn = np.load(tree / "human_optim" / "kindyn_1.npz", allow_pickle=True)
        sup = np.load(tree / "predictions" / "forces_sup.npz", allow_pickle=True)
        sensor = np.load(tree / "human_optim" / "sensor_forces.npz")
        if [str(x) for x in sup["limbs"]] != SUP_GROUPS:
            raise ValueError(f"{tree.name}: unexpected forces_sup groups")
        members = _limb_members([str(x) for x in kindyn["joint_names"]])
        forces = np.asarray(kindyn["contact_forces_world"][0], float)         # (N, 52, 3)
        contact = np.asarray(kindyn["joint_contact"][0], bool)                # (N, 52)
        mass = float(np.asarray(kindyn["total_mass"], float)[0])
        sup_world = np.nan_to_num(np.asarray(sup["forces_world"], float)[0]) * mass * GRAVITY
        probs = np.nan_to_num(np.asarray(sup["contact_probs"], float)[0])     # (N, 6)
        n = min(len(forces), len(sensor["reaction_world"]))
        columns["boards"].append(np.asarray(sensor["reaction_world"], float)[:n])
        columns["opt"].append(np.stack([forces[:n, m].sum(1) for m in members], 1))
        columns["opt_contact"].append(np.stack([contact[:n, m].any(1) for m in members], 1))
        columns["model"].append(np.stack([sup_world[:n, g].sum(1) for g in GROUP_FOLD], 1))
        columns["model_contact"].append(
            np.stack([probs[:n, g].max(1) for g in GROUP_FOLD], 1) > SUP_CONTACT_PROB)
    print(f"[compare] pooled {len(trials)} trials")
    return {key: np.concatenate(parts) for key, parts in columns.items()}


def _method_metrics(data: dict[str, np.ndarray], source: str) -> list[float]:
    """One row of the table: every metric for ``source`` ("opt" or "model")."""
    limb, boards = data[source], data["boards"]                               # (T, 4, 3)
    size, board_size = np.linalg.norm(limb, axis=-1), np.linalg.norm(boards, axis=-1)

    both = (size > BOARD_CONTACT_N) & (board_size > BOARD_CONTACT_N)
    cosine = (limb[both] * boards[both]).sum(-1) / (size[both] * board_size[both])
    angle = float(np.median(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))

    share = 100.0 * size.mean(0) / size.mean(0).sum()
    board_share = 100.0 * board_size.mean(0) / board_size.mean(0).sum()

    truth, claim = board_size > BOARD_CONTACT_N, data[f"{source}_contact"]
    precision = (claim & truth).sum() / max(claim.sum(), 1)
    recall = (claim & truth).sum() / max(truth.sum(), 1)

    return [float(np.abs(size - board_size).mean()), angle,
            float(np.corrcoef(size.reshape(-1), board_size.reshape(-1))[0, 1]),
            float(np.abs(share - board_share).mean()),
            float(precision), float(recall),
            float(2 * precision * recall / max(precision + recall, 1e-9))]


def main() -> None:
    """Build the comparison PNG, styled like a paper table (booktabs rules, serif)."""
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
    data = _gather()
    frames = len(data["boards"])
    values = [_method_metrics(data, source) for source in ("opt", "model")]
    formats = ["{:.1f}", "{:.1f}", "{:.2f}", "{:.1f}", "{:.3f}", "{:.3f}", "{:.3f}"]

    fig, axis = plt.subplots(figsize=(8.6, 2.15))
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    rule = lambda y, x0, x1, lw: axis.plot(  # noqa: E731 — a booktabs-style horizontal rule
        [x0, x1], [y, y], color="black", lw=lw, clip_on=False, solid_capstyle="butt")
    centre = [0.30 + 0.10 * col + 0.05 for col in range(7)]  # metric column centres

    axis.text(0.5, 1.13, "Table 1.  Force and contact accuracy against the instrumented "
              f"boards on climb_wall_2 (7 trials, {frames} frames).",
              ha="center", fontsize=11)
    rule(0.96, 0.0, 1.0, 1.4)                                       # toprule
    axis.text(0.50, 0.83, "Forces", ha="center", fontsize=11)       # spanning group heads
    axis.text(0.85, 0.83, "Contacts", ha="center", fontsize=11)
    rule(0.76, 0.31, 0.69, 0.7)                                     # cmidrules
    rule(0.76, 0.71, 0.99, 0.7)
    axis.text(0.01, 0.60, "Method", ha="left", fontsize=11)
    for col, name in enumerate(METRICS):
        arrow = "$\\uparrow$" if HIGHER_IS_BETTER[col] else "$\\downarrow$"
        axis.text(centre[col], 0.60, f"{name} {arrow}", ha="center", fontsize=11)
    rule(0.50, 0.0, 1.0, 0.7)                                       # midrule
    for row, method in enumerate(["Optimization", "Learned model"]):
        y = 0.36 - 0.19 * row
        axis.text(0.01, y, method, ha="left", fontsize=11)
        for col, (fmt, value) in enumerate(zip(formats, values[row])):
            sign = 1 if HIGHER_IS_BETTER[col] else -1
            best = max(range(2), key=lambda m: sign * values[m][col])
            axis.text(centre[col], y, fmt.format(value), ha="center", fontsize=11,
                      fontweight="bold" if row == best else "normal")
    rule(0.08, 0.0, 1.0, 1.4)                                       # bottomrule
    axis.text(0.0, -0.10, "Pooled over the four limbs. Angle: median where both sides "
              "exceed 50 N. Share: mean deviation from the boards' load distribution, "
              "percentage points.\nContact ground truth: board force above 50 N. "
              "Bold marks the better method.", ha="left", va="top", fontsize=8, color="0.25")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.86, bottom=0.22)

    path = WALL2_ROOT / "model_comparison.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[compare] -> {path}")


if __name__ == "__main__":
    main()
