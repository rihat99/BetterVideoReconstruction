"""Three runs of our own chain on climb_wall_3, against the rig's measurements.

Every clip was filmed by the one camera we reconstruct from, and measured twice over:
four instrumented holds give the contact state and the force of each limb, and a
calibrated three-camera fit gives the body. All three rows below saw identical pixels
and are scored against identical measurements. They differ by one thing each:

* **Full optimization** — the shipped chain. Contact is estimated from the video, and a
  joint torque is priced by how strong that joint is (``[torque.multipliers]``: a wrist
  torque costs 17x a hip one), so a load the body could carry either way goes to the
  limb that can carry it.
* **+ measured contacts** — the boards say which limb is loaded instead of the video
  (``docs/board_contacts.md``). The forces are still never supervised, so the force
  columns stay an honest prediction.
* **− per-joint torque weights** — every joint's torque priced the same. Only
  ``kindyn_1`` is rerun; the pose it starts from and the contact labels are inherited
  from the first row, so this row isolates the weighting and nothing else.

Sixteen columns in three families, all pooled over the 68 clips:

* **contacts** — the labels each run's dynamics consumed (``contacts_2``), folded onto
  the four limbs, against the board labels. Two of the rows share their labels, so two
  of the rows share these numbers; that is the point of the ablation, not a mistake.
* **pose** — the WHAM/EMDB set against the three-camera fit, read from the JSON
  ``score_published_pose.py --benchmark climb_wall_3`` writes. Run that first, or these
  columns come up blank.
* **forces** — each run's solved limb force against the measured one: the size error in
  newtons (mean absolute, and root-mean-square which charges a few large misses more),
  the angle between the vectors where both read above 50 N, the correlation of the sizes
  over time, and the load-share error in percentage points, where "the total is right but
  the wrong limb is carrying it" shows up.

  **The angle and share columns follow ``compare_climb_wall_2.py``** (retired 2026-10-08, see git history), so a number here can
  be read against that rig: the angle is the MEDIAN (a barely-loaded limb points nowhere
  in particular, and the median ignores it), and the share compares each limb's force
  averaged over the WHOLE CLIP. ``compare_gt_contact.py`` reads both more strictly — mean
  angle, and the share compared frame by frame, which does not let an error that swaps
  limbs back and forth cancel. On the same forces the strict pair is 14.7° / 10.8 pp where
  this table says 11.6° / 2.9 pp. Neither is wrong and they are not comparable; the strict
  pair is printed under the table and kept in the JSON under ``extra``.

Writes ``../data/peter/benchmark_tables/climb_wall_3_runs_{table.png,metrics.json}``.

Usage::

    uv run --no-sync python scripts/diagnostics/score_published_pose.py --benchmark climb_wall_3  # pose columns
    uv run --no-sync python scripts/diagnostics/compare_climb_wall_3_runs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "scripts", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from compare_gt_contact import CONTACT_N, force_metrics, limb_forces, limb_labels  # noqa: E402
from _common import _render_table  # noqa: E402
from score_published_contacts import GT_ROOT, OURS_ROOT, _count, _rates_balanced, clips, ours, truth  # noqa: E402
from tools.human_optim.boards import LIMBS  # noqa: E402
from tools.paths import PETER  # noqa: E402

POSE_JSON = PETER / "climb_wall_3_gt" / "others" / "gt_pose_metrics.json"
OUT = PETER / "benchmark_tables"

#: (row label, the run directory under ``<clip>/``, the key ``score_published_pose`` files it under).
RUNS = (("Full optimization", "human_optim", "ours/kindyn_1"),
        ("+ measured contacts", "human_optim_gt_contact", "ours/gt_contact"),
        ("− per-joint torque weights", "human_optim_flat_torque", "ours/flat_torque"))

#: Column -> (JSON key, printf format, bigger-is-better). The order the table reads in.
POSE_COLUMNS = ("MPJPE", "PA-MPJPE", "PVE", "Accel", "WA-MPJPE", "W-MPJPE", "RTE", "Jitter")
FORCE_COLUMNS = {"MAE": "limb MAE [N]", "Angle": "angle median [deg]", "Corr": "corr",
                 "Share": "share err, clip-averaged [pp]", "RMSE": "limb RMSE [N]"}
#: The stricter reading of the same two columns, printed beside the table but not in it.
STRICT_COLUMNS = {"Angle": "angle [deg]", "Share": "share err [pp]"}
#: Columns where a bigger number is the better one. They are also the rates — the
#: columns that cannot exceed one and are therefore printed to three decimals.
HIGHER_IS_BETTER = {"F1", "Precision", "Recall", "Corr"}


def contact_rates(optim_dir: str) -> dict:
    """Pooled precision / recall / F1 of one run's labels against the boards."""
    counts = {limb: dict(tp=0, fp=0, fn=0, tn=0) for limb in LIMBS}
    for clip in clips():
        n = int(np.load(GT_ROOT / clip / "cameras.npz", allow_pickle=True)["num_frames"])
        _count(counts, ours("climb_wall_3", clip, 2, n, optim_dir), truth("climb_wall_3", clip))
    return _rates_balanced({k: sum(counts[limb][k] for limb in LIMBS)
                            for k in ("tp", "fp", "fn", "tn")})


def force_rates(optim_dir: str) -> dict:
    """One run's solved limb forces against the boards, pooled over every clip.

    A limb-frame the rig never measured is NaN in the boards and drops out of every
    number; :func:`~compare_gt_contact.force_metrics` does that blanking itself.
    """
    boards, forces, residuals, labels = [], [], [], []
    aligned = True
    for clip in clips():
        run = OURS_ROOT / clip / optim_dir
        sensor = np.load(OURS_ROOT / clip / "human_optim" / "sensor_forces.npz")
        reaction = np.asarray(sensor["reaction_world"], float)
        aligned &= bool(sensor["world_aligned"]) if "world_aligned" in sensor else True
        force, residual = limb_forces(run / "kindyn_1.npz", len(reaction))
        n = min(len(reaction), len(force))
        boards.append(reaction[:n])
        forces.append(force[:n])
        residuals.append(residual[:n])
        labels.append(limb_labels(run / "contacts_2.npz", n))
    measured, solved = (np.concatenate(boards), np.concatenate(forces))
    out = force_metrics(measured, solved, np.concatenate(residuals),
                        np.concatenate(labels), aligned)
    return out | _other_conventions(measured, solved)


def _other_conventions(boards: np.ndarray, ours: np.ndarray) -> dict[str, float]:
    """The same two numbers as ``compare_climb_wall_2.py`` (retired, see git history) reports them.

    That script takes the MEDIAN angle (a near-free limb's direction is noise, and the
    median ignores it) and builds the share from each limb's force averaged over the
    whole clip FIRST — so an error that swaps limbs back and forth cancels. Both read
    much lower than the per-frame numbers in the table, and the two are not comparable;
    they are carried here so a climb_wall_2 number can be read against this rig.
    """
    m, s = np.linalg.norm(boards, axis=-1), np.linalg.norm(ours, axis=-1)
    both = (m > CONTACT_N) & (s > CONTACT_N)
    cos = np.clip((boards * ours).sum(-1)[both] / (m[both] * s[both]), -1, 1)
    return {"angle median [deg]": float(np.median(np.degrees(np.arccos(cos)))),
            "share err, clip-averaged [pp]":
                float(np.abs(100 * s.mean(0) / s.mean(0).sum()
                             - 100 * m.mean(0) / m.mean(0).sum()).mean())}


def collect() -> dict:
    pose = json.loads(POSE_JSON.read_text())["rows"] if POSE_JSON.exists() else {}
    record = {}
    for label, optim_dir, pose_key in RUNS:
        unfinished = [c for c in clips() if not (OURS_ROOT / c / optim_dir / "kindyn_1.npz").exists()]
        if unfinished:
            print(f"[cw3] {label}: {len(unfinished)}/{len(clips())} clip(s) of {optim_dir}/ "
                  "have no kindyn_1.npz — row skipped")
            continue
        contacts, forces = contact_rates(optim_dir), force_rates(optim_dir)
        row = {"F1": contacts["f1"], "Precision": contacts["precision"],
               "Recall": contacts["recall"]}
        row |= {name: pose.get(pose_key, {}).get(name) for name in POSE_COLUMNS}
        row |= {name: forces[key] for name, key in FORCE_COLUMNS.items()}
        record[label] = {"run": optim_dir, "metrics": row,
                         "extra": {k: forces[k] for k in
                                   ("total MAE [N]", "hands/feet", "root resid [BW]", "|df| [N]",
                                    "angle [deg]", "share err [pp]")}}
        print(f"[cw3] {label}: done", flush=True)
    return record


def write_table(record: dict, path: Path) -> None:
    names = list(next(iter(record.values()))["metrics"])
    cell = lambda name, v: ("—" if v is None else  # noqa: E731
                            f"{v:.3f}" if name in HIGHER_IS_BETTER else f"{v:.1f}")
    cells = [[label, *(cell(n, r["metrics"][n]) for n in names)] for label, r in record.items()]
    bold = set()
    for col, name in enumerate(names, start=1):
        vals = {i: float(c[col]) for i, c in enumerate(cells) if c[col] != "—"}
        if vals:
            best = (max if name in HIGHER_IS_BETTER else min)(vals.values())
            bold |= {(i, col) for i, v in vals.items() if v == best}
    arrow = {n: ("↑" if n in HIGHER_IS_BETTER else "↓") for n in names}
    columns = [("Method", 3.0, "l")] + [(f"{n} {arrow[n]}", 1.15, "r") for n in names]
    _render_table(path, columns, cells,
                  groups=(("contacts vs boards", 1, 3), ("pose vs 3-camera GT (mm)", 4, 11),
                          ("forces vs boards · median angle, clip-averaged share", 12, 16)),
                  bold=frozenset(bold))


if __name__ == "__main__":
    record = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "climb_wall_3_runs_metrics.json").write_text(json.dumps(record, indent=2))
    write_table(record, OUT / "climb_wall_3_runs_table.png")
    names = list(next(iter(record.values()))["metrics"])
    print(f"\n  {'':28s}" + "".join(f"{n:>11s}" for n in names))
    for label, r in record.items():
        print(f"  {label:28s}" + "".join(
            f"{'—':>11s}" if v is None else f"{v:11.3f}" for v in r["metrics"].values()))
    print("\n  the same two columns read per frame instead (see the module docstring):")
    print(f"  {'':28s}" + "".join(f"{n:>22s}" for n in STRICT_COLUMNS))
    for label, r in record.items():
        print(f"  {label:28s}" + "".join(f"{r['extra'][k]:22.3f}" for k in STRICT_COLUMNS.values()))
    for p in (OUT / "climb_wall_3_runs_table.png", OUT / "climb_wall_3_runs_metrics.json"):
        print(f"[cw3] wrote {p}")
