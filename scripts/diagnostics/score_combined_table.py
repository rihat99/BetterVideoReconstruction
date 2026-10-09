"""The published models on both benchmarks, side by side.

Combines the two evaluations that already exist:

- **ClimbingVideos** (108 held-out scenes) — hand-annotated contact on six
  end-effector joints; the pose reference is *our own* ``kindyn_1`` body, so the
  pose columns there say "how far is this model from our reconstruction", not
  from truth, and our own row is the ruler rather than a competitor.
- **climb_wall_3** (68 clips) — measured ground truth: contact from four
  instrumented holds (50 N enter / 30 N exit), pose from a calibrated
  three-camera fit. Our single-view run is a scored row like any other.

The two contact columns are **not** the same test. ClimbingVideos labels six
joints and is 52 % contact; climb_wall_3 labels four limbs and is 83 % contact.
F1 therefore starts from a much higher floor on climb_wall_3 — the
"always in contact" row makes that explicit. **Balanced accuracy** (the mean of
how often a real contact is caught and how often a free limb is left alone) is
the column that compares across the two, because guessing "contact" scores
0.500 on it whatever the prior.

For the per-vertex models the table reports the best of their four
vertex→joint mappings on each dataset, named in the JSON.

Writes ``../data/peter/benchmark_tables/final_{contact,pose}_table.png`` and
``../data/peter/benchmark_tables/final_metrics.json``.

Usage:
    uv run --no-sync python scripts/diagnostics/score_combined_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import _render_table  # noqa: E402
from bvr.paths import CLIMBING, PETER  # noqa: E402


def rate(value: float | None) -> str:
    """A rate as ``.902`` — no leading zero, which is the convention for a number
    that cannot exceed one. Values of exactly 1 keep their digit."""
    if value is None:
        return "—"
    text = f"{value:.3f}"
    return text[1:] if text.startswith("0.") else text

CV = CLIMBING / "results"
CW = PETER / "climb_wall_3_gt" / "others"
OUT = PETER / "benchmark_tables"

# (label, ClimbingVideos key, climb_wall_3 key). None = the row does not exist there.
CONTACT_ROWS = [
    ("Always in contact (floor)", "_floor", "floor/always"),
    ("Ours (contacts_1)", "ours/contacts_1", "ours/contacts_1"),
    ("Ours (contacts_2)", "ours/contacts_2", "ours/contacts_2"),
    ("GVHMR (p > 0.5)", "gvhmr/raw/p0.5", "gvhmr/raw/p0.5"),
    ("PromptHMR (p > 0.5)", "prompthmr/raw/p0.5", "prompthmr/raw/p0.5"),
    ("PromptHMR (p > 0.8)", "prompthmr/raw/p0.8", "prompthmr/raw/p0.8"),
    ("DECO", "deco/*", "deco/*"),
    ("BSTRO", "bstro/*", "bstro/*"),
    ("InteractVLM (query: wall)", "interactvlm_wall/*", "interactvlm_wall/*"),
    ("InteractVLM (query: scene)", "interactvlm_scene/*", "interactvlm_scene/*"),
]
POSE_ROWS = [
    ("SAM 3D Body init", "sam3d/init", "sam3d/init"),
    ("Ours (single view)", None, "ours/kindyn_1"),
    ("GVHMR (raw)", "gvhmr/raw", "gvhmr/raw"),
    ("GVHMR (post)", "gvhmr/post", "gvhmr/post"),
    ("PromptHMR (raw)", "prompthmr/raw", "prompthmr/raw"),
    ("PromptHMR (post)", "prompthmr/post", "prompthmr/post"),
]
# camera-frame block, then world-frame block
POSE_METRICS = ["MPJPE", "PA-MPJPE", "PVE", "Accel", "WA-MPJPE", "W-MPJPE", "RTE", "Jitter"]


def balanced(counts: dict) -> float | None:
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    if not (tp + fn) or not (tn + fp):
        return None
    return (tp / (tp + fn) + tn / (tn + fp)) / 2


def floor_row(rows: dict) -> dict:
    """Always-predict-contact, derived from any row's confusion counts."""
    c = next(iter(rows.values()))["overall"]
    pos, total = c["tp"] + c["fn"], sum(c[k] for k in ("tp", "fp", "fn", "tn"))
    precision = pos / total
    return {"precision": precision, "recall": 1.0, "f1": 2 * precision / (precision + 1),
            "balanced_accuracy": 0.5, "prior": precision}


def contact_cell(metrics: dict, key: str | None) -> tuple[dict | None, str | None]:
    if key is None:
        return None, None
    rows = metrics["rows"]
    if key == "_floor":
        return (rows["floor/always"]["overall"] if "floor/always" in rows else floor_row(rows)), None
    if key.endswith("/*"):                       # best of the vertex mappings
        model = key[:-2]
        cands = {k: v for k, v in rows.items() if k.startswith(model + "/")}
        best = max(cands, key=lambda k: cands[k]["overall"]["f1"])
        return cands[best]["overall"], best.split("/", 1)[1]
    return rows[key]["overall"], None


def group_f1(metrics: dict, key: str | None) -> dict[str, float | None]:
    """Hands and feet F1 for a row. The two datasets name the foot group differently."""
    rows = metrics["rows"]
    if key == "_floor":
        row = rows.get("floor/always")
    elif key and key.endswith("/*"):
        model = key[:-2]
        cands = {k: v for k, v in rows.items() if k.startswith(model + "/")}
        row = cands[max(cands, key=lambda k: cands[k]["overall"]["f1"])]
    else:
        row = rows.get(key)
    if row is None:                 # a derived floor: rebuild it from any row's group counts
        any_row = next(iter(rows.values()))
        out = {}
        for name, stored in (("hands", "hands"), ("feet", "feet")):
            counts = any_row.get(stored, any_row.get("feet_ankles")) if name == "feet" \
                else any_row.get("hands")
            pos = counts["tp"] + counts["fn"]
            total = sum(counts[k] for k in ("tp", "fp", "fn", "tn"))
            precision = pos / total
            out[name] = 2 * precision / (precision + 1)
        return out
    feet = row.get("feet", row.get("feet_ankles", {}))
    return {"hands": row.get("hands", {}).get("f1"), "feet": feet.get("f1")}


def render(path: Path, columns, cells, groups, better_max: bool, skip_rows: set[int]) -> None:
    bold = set()
    for col in range(1, len(columns)):
        vals = {i: float(c[col]) for i, c in enumerate(cells)
                if i not in skip_rows and c[col] not in ("—", "ref")}
        if vals:
            best = (max if better_max else min)(vals.values())
            bold |= {(i, col) for i, v in vals.items() if v == best}
    _render_table(path, columns, cells, groups=groups, bold=frozenset(bold))


def main() -> None:
    cv_c, cw_c = (json.loads((p / f).read_text()) for p, f in
                  ((CV, "baseline_contact_metrics.json"), (CW, "gt_contact_metrics.json")))
    cv_p, cw_p = (json.loads((p / f).read_text()) for p, f in
                  ((CV, "baseline_pose_metrics.json"), (CW, "gt_pose_metrics.json")))
    OUT.mkdir(parents=True, exist_ok=True)
    record: dict = {"contact": {}, "pose": {}}

    # ── contacts ──────────────────────────────────────────────────────────────
    cells, chosen = [], {}
    for label, cv_key, cw_key in CONTACT_ROWS:
        row = [label]
        for metrics, key, tag in ((cv_c, cv_key, "ClimbingVideos"), (cw_c, cw_key, "climb_wall_3")):
            entry, variant = contact_cell(metrics, key)
            if entry is None:
                row += ["—"] * 6
                continue
            bal = entry.get("balanced_accuracy") or balanced(entry)
            groups = group_f1(metrics, key)
            row += [rate(entry[m]) for m in ("precision", "recall", "f1")] + [rate(bal)]
            row += [rate(groups[g]) for g in ("hands", "feet")]
            record["contact"].setdefault(label, {})[tag] = {
                "precision": entry["precision"], "recall": entry["recall"],
                "f1": entry["f1"], "balanced_accuracy": bal,
                "hands_f1": groups["hands"], "feet_f1": groups["feet"], "variant": variant}
            if variant:
                chosen.setdefault(label, {})[tag] = variant
        cells.append(row)
    cells.append(cells.pop(0))                       # the floor belongs under the models it bounds
    columns = [("Model", 2.9, "l")] + [(n, 0.95, "r") for n in
                                       ("P", "R", "F1", "bal.Acc", "hands F1", "feet F1")] * 2
    render(OUT / "final_contact_table.png", columns, cells,
           (("ClimbingVideos · 6 joints", 1, 6), ("climb_wall_3 · 4 limbs", 7, 12)),
           better_max=True, skip_rows={len(cells) - 1})

    # ── pose ──────────────────────────────────────────────────────────────────
    cells = []
    for label, cv_key, cw_key in POSE_ROWS:
        row = [label]
        for metrics, key, tag in ((cv_p, cv_key, "ClimbingVideos"), (cw_p, cw_key, "climb_wall_3")):
            if key is None:
                row += ["ref"] * len(POSE_METRICS)
                continue
            r = metrics["rows"][key]
            row += ["—" if r[m] is None else f"{r[m]:.1f}" for m in POSE_METRICS]
            record["pose"].setdefault(label, {})[tag] = {m: r[m] for m in POSE_METRICS}
        cells.append(row)
    # The references' own jitter, so the Jitter column has a scale to be read against.
    ref = ["Reference body (jitter only)"]
    for metrics, tag in ((cv_p, "ClimbingVideos"), (cw_p, "climb_wall_3")):
        ref += ["—"] * (len(POSE_METRICS) - 1) + [f"{metrics['reference_jitter']:.1f}"]
        record["pose"].setdefault("Reference body", {})[tag] = {"Jitter": metrics["reference_jitter"]}
    cells.append(ref)
    columns = [("Model", 2.9, "l")] + [(n, 0.98, "r") for n in
                                       ("MPJPE", "PA-MPJPE", "PVE", "Accel",
                                        "WA-MPJPE", "W-MPJPE", "RTE %", "Jitter")] * 2
    render(OUT / "final_pose_table.png", columns, cells,
           (("ClimbingVideos · ref = OUR body", 1, 8),
            ("climb_wall_3 · ref = 3-camera GT", 9, 16)),
           better_max=False, skip_rows={len(cells) - 1})

    record["notes"] = {
        "contact_datasets_differ": "ClimbingVideos = 6 annotated joints, 51.9 % contact; "
                                   "climb_wall_3 = 4 instrumented limbs, 82.6 % contact. "
                                   "Compare on balanced accuracy, not F1.",
        "pose_reference_differs": "ClimbingVideos pose is measured against OUR kindyn_1 body "
                                  "(so our row is the reference); climb_wall_3 against an "
                                  "independent 3-camera fit.",
        "vertex_variant_chosen": chosen,
    }
    (OUT / "final_metrics.json").write_text(json.dumps(record, indent=2))

    for name, group in (("CONTACT", "contact"), ("POSE", "pose")):
        print(f"\n=== {name} ===")
        for label, per in record[group].items():
            for tag, vals in per.items():
                nums = " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} —"
                                for k, v in vals.items() if k != "variant" and v is not None)
                print(f"  {label:28s} {tag:16s} {nums}")
    for p in (OUT / "final_contact_table.png", OUT / "final_pose_table.png", OUT / "final_metrics.json"):
        print(f"[table] wrote {p}")


if __name__ == "__main__":
    main()
