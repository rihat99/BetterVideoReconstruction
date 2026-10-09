"""Score published contact models on one of the two contact benchmarks.

``--benchmark climbingvideos`` — the 108 held-out ClimbingVideos scenes against our
manual labels: the six annotated end-effector joints (left/right hand, foot, ankle),
tri-state (1 contact, 0 free, -1 never annotated — dropped). A "hand" here is the
annotator's hand; a "foot" is the forefoot / toes; an "ankle" is the heel region.
That is how the 14-joint annotation schema and our own contact frames use the words.

``--benchmark climb_wall_3`` — the same rows against the climb_wall_3 force boards,
measured instead of annotated. Every clip in ``peter/climb_wall_3_gt`` has four
instrumented holds, one per limb, so a limb is "in contact" when the board under it
carries load: the 50 N enter / 30 N exit hysteresis that built the GT labels stored
in ``human_optim/contacts_1.npz``. The models ran on the ORIGINAL ``cam_left.mp4``,
which is also the video our single-view pipeline used, so every row saw identical
pixels. A board carries the whole foot, so forefoot and heel are one group here.

Rows (each is one model output read per scored joint or limb):

- **Always in contact** (climb_wall_3 only) — the trivial predictor. The climber holds
  on with all four limbs about 83 % of the time, so this row is the floor any real
  model has to beat. That prior also flatters precision, recall and F1, so this table
  adds **balanced accuracy** — the mean of how often a real contact is caught and how
  often a free limb is left alone. Always guessing contact scores 0.500 on it.
- **Ours** — ``human_optim/contacts_{1,2}.npz`` (``features/`` on ClimbingVideos,
  ``out_climb_wall_3_single/<clip>/`` on the wall): the contact frames folded with an
  OR (palm/finger frames → hand, toe/ball frames → foot, heel → ankle; on the wall
  toes ∪ heel → foot), the same fold ``score_frame_contacts`` uses.
- **GVHMR / PromptHMR** — ``<model>/<scene>/pose_{raw,post}.npz``: the model's own
  six-joint stationary-joint probability, thresholded at 0.5 (and at 0.8 for
  PromptHMR, its demo's cut-off); on the wall foot and ankle are OR-ed into one foot.
  Neither model's post-processing changes it.
- **DECO / BSTRO / InteractVLM** — ``<model>/<scene>/contacts_vertex.npz``: per-vertex
  probabilities on the SMPL mesh, mapped to joints. InteractVLM is run with two text
  queries: ``scene`` (the README's word for scene/ground contact; on this model it only
  ever names the feet) and ``wall`` (names hands and feet). Two vertex→joint groupings
  × two decision rules:

    grouping ``seg``  — the SMPL part segmentation: hand = Hand ∪ HandIndex1,
                        foot = ToeBase, ankle = Foot (the rear-foot part);
    grouping ``lbs``  — each vertex goes to the joint with the largest skinning
                        weight: hand = wrist + hand joints, foot = foot joint,
                        ankle = ankle joint;
    rule ``any``      — the joint is in contact when at least one of its vertices
                        is above the model's own threshold;
    rule ``frac10``   — when at least 10 % of its vertices are.

  On the wall, once forefoot and heel are merged, the two groupings pick almost the
  same vertices (they differ by 2–3 of ~250), so their rows agree. Both are kept so
  the two tables line up.

Writes, next to the model results (or into ``--out``); models whose results are
missing are skipped and listed, so a table can be rebuilt while runs are going:

    climbingvideos  ClimbingVideos/results/baseline_contact_{metrics.json,table.png}
    climb_wall_3    peter/climb_wall_3_gt/others/gt_contact_{metrics.json,table.png,table_per_limb.png}

Usage:
    uv run --no-sync python datasets/methods/score_published_contacts.py --benchmark climbingvideos
    uv run --no-sync python datasets/methods/score_published_contacts.py --benchmark climb_wall_3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bvr.body import models_dir
from datasets._common import _fmt, _rates, _render_table
from datasets.examples.score_frame_contacts import _FOLD_10, _FOLD_14, metrics as _metrics
from bvr.paths import REPO_ROOT
from datasets.climbingvideos.paths import CLIMBING
from datasets.peter.paths import PETER

FEATURES, RESULTS = CLIMBING.resolve() / "features", CLIMBING.resolve() / "results"
GT_ROOT = PETER / "climb_wall_3_gt" / "human"
OURS_ROOT = PETER / "out_climb_wall_3_single"
OTHERS = PETER / "climb_wall_3_gt" / "others"
SEGMENTATION = REPO_ROOT.parent / "other" / "deco" / "data"  # smpl(x)_vert_segmentation.json
BODY_MODELS = models_dir()

JOINTS = ["left_hand", "right_hand", "left_foot", "right_foot", "left_ankle", "right_ankle"]
LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]
POSE_MODELS = {"gvhmr": "GVHMR", "prompthmr": "PromptHMR"}
# Probability cut-offs per six-joint model: 0.5 for every row, plus the model's own demo
# threshold where it differs (PromptHMR's demo binarises its stationary head at 0.8).
POSE_THRESHOLDS = {"gvhmr": (0.5,), "prompthmr": (0.5, 0.8)}
VERTEX_MODELS = {"deco": "DECO", "bstro": "BSTRO", "interactvlm_wall": "InteractVLM (query: wall)",
                 "interactvlm_scene": "InteractVLM (query: scene)"}
GROUPINGS, RULES, FRAC = ("seg", "lbs"), ("any", "frac10"), 0.10

#: Per benchmark: the scored columns, our contact-frame fold, the six-joint head columns
#: OR-ed into each, the SMPL parts and skinning joints of each, and the directory that
#: holds the model results and receives the scores. Dicts are in column order.
BENCHMARKS = {
    "climbingvideos": {
        "columns": JOINTS, "fold": _FOLD_14, "results": RESULTS, "prefix": "baseline",
        "six_joint": {joint: (joint,) for joint in JOINTS},
        "seg_parts": {"left_hand": ("leftHand", "leftHandIndex1"),
                      "right_hand": ("rightHand", "rightHandIndex1"),
                      "left_foot": ("leftToeBase",), "right_foot": ("rightToeBase",),
                      "left_ankle": ("leftFoot",), "right_ankle": ("rightFoot",)},
        "lbs_joints": {  # hand = wrist + everything below it
            "smpl": {"left_hand": [20, 22], "right_hand": [21, 23], "left_foot": [10],
                     "right_foot": [11], "left_ankle": [7], "right_ankle": [8]},
            "smplx": {"left_hand": [20, *range(25, 40)], "right_hand": [21, *range(40, 55)],
                      "left_foot": [10], "right_foot": [11], "left_ankle": [7], "right_ankle": [8]}},
    },
    "climb_wall_3": {
        "columns": LIMBS, "fold": _FOLD_10, "results": OTHERS, "prefix": "gt",
        "six_joint": {"left_hand": ("left_hand",), "right_hand": ("right_hand",),
                      "left_foot": ("left_foot", "left_ankle"),
                      "right_foot": ("right_foot", "right_ankle")},
        "seg_parts": {"left_hand": ("leftHand", "leftHandIndex1"),
                      "right_hand": ("rightHand", "rightHandIndex1"),
                      "left_foot": ("leftToeBase", "leftFoot"),
                      "right_foot": ("rightToeBase", "rightFoot")},
        "lbs_joints": {
            "smpl": {"left_hand": [20, 22], "right_hand": [21, 23], "left_foot": [7, 10],
                     "right_foot": [8, 11]},
            "smplx": {"left_hand": [20, *range(25, 40)], "right_hand": [21, *range(40, 55)],
                      "left_foot": [7, 10], "right_foot": [8, 11]}},
    },
}


def scene_dir(stage: str, scene: str) -> Path:
    return FEATURES / stage / scene[:2] / scene[2:4] / scene


def test_scenes() -> list[str]:
    return [line.split("\t")[0] for line in (RESULTS / "test_scenes.txt").read_text().splitlines() if line]


def clips() -> list[str]:
    return sorted(d.name for d in GT_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_"))


def truth(benchmark: str, item: str) -> np.ndarray:
    """``(T, C)`` labels in column order: the manual tri-state int8 on ClimbingVideos, the
    board labels as bool on climb_wall_3."""
    if benchmark == "climbingvideos":
        ann = np.load(scene_dir("annotation", item) / "annotation.npz", allow_pickle=True)
        if ann["contacts"].shape[0] != 1 or bool(ann["ignored"][0]):
            raise SystemExit(f"[score] {item}: expected one annotated, non-ignored person")
        names = [str(n) for n in ann["joint_names"]]
        return np.stack([ann["contacts"][0, names.index(j)] for j in JOINTS], axis=1).astype(np.int8)
    d = np.load(GT_ROOT / item / "human_optim" / "contacts_1.npz", allow_pickle=True)
    names = [str(n) for n in d["limb_names"]]
    return np.asarray(d["limb_contact"][0], bool)[:, [names.index(limb) for limb in LIMBS]]


# ────────────────────────────── readers → (T, C) bool ─────────────────────────────

def ours(benchmark: str, item: str, tag: int, n_frames: int, optim_dir: str = "human_optim") -> np.ndarray:
    """``(T, C)`` bool — one run's contact frames folded onto the benchmark's columns."""
    spec = BENCHMARKS[benchmark]
    tree = scene_dir("human_optim", item) if benchmark == "climbingvideos" else OURS_ROOT / item / optim_dir
    d = np.load(tree / f"contacts_{tag}.npz", allow_pickle=True)
    names = [str(n) for n in d["contact_frame_names"]]
    contact = np.asarray(d["frame_contact"][0], bool)                       # (T, F)
    cols = [[names.index(f) for f in spec["fold"][c] if f in names] for c in spec["columns"]]
    return np.stack([contact[:, g].any(1) for g in cols], axis=1)[:n_frames]


def six_joint(path: Path, n_frames: int, threshold: float, fold: dict[str, tuple[str, ...]]) -> np.ndarray:
    """The six-joint stationary head, each column the OR of its ``fold`` joints."""
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["contact_joint_names"]]
    probs = np.asarray(d["contacts"], np.float32) > threshold
    out = np.zeros((n_frames, len(fold)), bool)
    out[np.asarray(d["frame_indices"], int)] = np.stack(
        [probs[:, [names.index(j) for j in joints]].any(1) for joints in fold.values()], axis=1)
    return out


class VertexGroups:
    """Vertex index sets per column for one mesh topology, under both groupings."""

    def __init__(self, topology: str, spec: dict) -> None:
        self.columns = spec["columns"]
        seg = json.loads((SEGMENTATION / f"{topology}_vert_segmentation.json").read_text())
        self.seg = {c: np.unique(np.concatenate([seg[p] for p in parts])) for c, parts in spec["seg_parts"].items()}
        name = "SMPL_NEUTRAL.npz" if topology == "smpl" else "SMPLX_NEUTRAL.npz"
        owner = np.asarray(np.load(BODY_MODELS / topology / name, allow_pickle=True)["weights"]).argmax(1)
        self.lbs = {c: np.flatnonzero(np.isin(owner, ids)) for c, ids in spec["lbs_joints"][topology].items()}

    def fold(self, probs: np.ndarray, threshold: float, grouping: str, rule: str) -> np.ndarray:
        groups = getattr(self, grouping)
        hit = probs > threshold                                             # (T, V)
        frac = [hit[:, groups[c]].mean(1) for c in self.columns]
        return np.stack([f > 0 if rule == "any" else f >= FRAC for f in frac], axis=1)


def vertex_rows(path: Path, n_frames: int, groups: dict[str, VertexGroups]) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    key, topology = ("contact_smpl", "smpl") if "contact_smpl" in d.files else ("contact_smplx", "smplx")
    probs, idx, thr = np.asarray(d[key], np.float32), np.asarray(d["frame_indices"], int), float(d["threshold"])
    out = {}
    for grouping in GROUPINGS:
        for rule in RULES:
            full = np.zeros((n_frames, len(groups[topology].columns)), bool)
            full[idx] = groups[topology].fold(probs, thr, grouping, rule)
            out[f"{grouping}/{rule}"] = full
    return out


# ─────────────────────────────────── scoring ─────────────────────────────────

def _rates_balanced(counts: dict[str, int]) -> dict:
    """``_rates`` plus balanced accuracy, which the 83 % contact prior cannot inflate."""
    r = _rates(counts)
    tpr = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None
    tnr = counts["tn"] / (counts["tn"] + counts["fp"]) if counts["tn"] + counts["fp"] else None
    r["balanced_accuracy"] = None if tpr is None or tnr is None else (tpr + tnr) / 2
    return r


def _count(counts: dict, predicted: np.ndarray, labels: np.ndarray) -> None:
    """Add one clip to ``counts`` (keyed in column order); a ``-1`` label is never scored."""
    for k, c in enumerate(counts.values()):
        known = labels[:, k] >= 0
        got, want = predicted[:len(labels)][known, k], labels[known, k] == 1
        c["tp"] += int(np.sum(got & want)); c["fp"] += int(np.sum(got & ~want))
        c["fn"] += int(np.sum(~got & want)); c["tn"] += int(np.sum(~got & ~want))


def collect(benchmark: str) -> dict:
    """Score every model row on one benchmark and return the metrics tree."""
    spec = BENCHMARKS[benchmark]
    columns, cv = spec["columns"], benchmark == "climbingvideos"
    unit = "scenes" if cv else "clips"
    groups = {"smpl": VertexGroups("smpl", spec), "smplx": VertexGroups("smplx", spec)}
    rows: dict[str, dict] = {}      # row key → {label, counts per column, scenes / clips}
    missing: dict[str, int] = {}

    def add(key: str, label: str, predicted: np.ndarray, labels: np.ndarray) -> None:
        r = rows.setdefault(key, {"label": label, unit: 0,
                                  "counts": {c: dict(tp=0, fp=0, fn=0, tn=0) for c in columns}})
        _count(r["counts"], predicted, labels)
        r[unit] += 1

    items = test_scenes() if cv else clips()
    for item in items:
        labels = truth(benchmark, item)
        n = len(labels) if cv else int(np.load(GT_ROOT / item / "cameras.npz", allow_pickle=True)["num_frames"])
        if not cv:
            add("floor/always", "Always in contact", np.ones((n, len(columns)), bool), labels)
        for tag in (1, 2):
            add(f"ours/contacts_{tag}", f"Ours (contacts_{tag})", ours(benchmark, item, tag, n), labels)
        for model, name in POSE_MODELS.items():
            for variant in ("raw", "post"):
                path = spec["results"] / model / item / f"pose_{variant}.npz"
                if not path.exists():
                    missing[f"{model}/{variant}"] = missing.get(f"{model}/{variant}", 0) + 1
                    continue
                for thr in POSE_THRESHOLDS[model]:
                    add(f"{model}/{variant}/p{thr}", f"{name} ({variant}, p > {thr})",
                        six_joint(path, n, thr, spec["six_joint"]), labels)
        for model, name in VERTEX_MODELS.items():
            path = spec["results"] / model / item / "contacts_vertex.npz"
            if not path.exists():
                missing[model] = missing.get(model, 0) + 1
                continue
            for mapping, predicted in vertex_rows(path, n, groups).items():
                add(f"{model}/{mapping}", f"{name} ({mapping})", predicted, labels)

    rate = _metrics if cv else _rates_balanced
    feet, per = ("feet_ankles", "per_joint") if cv else ("feet", "per_limb")
    for r in rows.values():
        pooled = lambda cols: {k: sum(r["counts"][c][k] for c in cols) for k in ("tp", "fp", "fn", "tn")}  # noqa: E731
        r["overall"] = rate(pooled(columns))
        r["hands"], r[feet] = rate(pooled(columns[:2])), rate(pooled(columns[2:]))
        r[per] = {c: rate(r["counts"][c]) for c in columns}
        del r["counts"]
    if cv:
        head = {"scenes": len(items), "joints": JOINTS,
                "ground_truth": "features/annotation/<scene>/annotation.npz (14-joint tri-state, first six)"}
    else:
        head = {"clips": len(items), "limbs": LIMBS,
                "ground_truth": "climb_wall_3_gt/human/<clip>/human_optim/contacts_1.npz limb_contact "
                                "— measured board load, 50 N enter / 30 N exit",
                "video": "peter/climb_wall_3/<clip>/cam_left.mp4 (original, as our single-view run used)"}
    return head | {"thresholds_six_joint": POSE_THRESHOLDS, "frac_rule": FRAC, "missing": missing, "rows": rows}


def _rate(value: float | None) -> str:
    """A rate as ``.902`` — no leading zero, the convention for a number that
    cannot exceed one. Exactly 1 keeps its digit."""
    if value is None:
        return "—"
    text = f"{value:.3f}"
    return text[1:] if text.startswith("0.") else text


def _best(cells: list[list[str]], n_columns: int) -> frozenset[tuple[int, int]]:
    """The ``(row, column)`` cells holding each numeric column's highest value."""
    bold = set()
    for col in range(2, n_columns):
        vals = {i: float(c[col]) for i, c in enumerate(cells) if c[col] != "—"}
        if vals:
            best = max(vals.values()); bold |= {(i, col) for i, v in vals.items() if v == best}
    return frozenset(bold)


def write_tables(m: dict, benchmark: str, out: Path) -> list[Path]:
    """Render the benchmark's PNG table(s) into ``out`` and return their paths."""
    rows = list(m["rows"].values())
    if benchmark == "climbingvideos":
        cells = [[r["label"], str(r["scenes"]),
                  *(_fmt(r["overall"][x]) for x in ("precision", "recall", "f1")),
                  _fmt(r["hands"]["f1"]), _fmt(r["feet_ankles"]["f1"]),
                  *(_fmt(r["per_joint"][j]["f1"]) for j in JOINTS)] for r in rows]
        columns = [("Model", 3.6, "l"), ("n", 0.6, "r"), ("P", 0.9, "r"), ("R", 0.9, "r"), ("F1", 0.9, "r"),
                   ("hands F1", 1.1, "r"), ("feet F1", 1.1, "r"),
                   *((j.replace("_", " "), 1.2, "r") for j in JOINTS)]
        path = out / "baseline_contact_table.png"
        _render_table(path, columns, cells, bold=_best(cells, len(columns)))
        return [path]

    cells = [[r["label"], str(r["clips"]),
              *(_rate(r["overall"][x]) for x in ("precision", "recall", "f1", "accuracy", "balanced_accuracy")),
              _rate(r["hands"]["f1"]), _rate(r["feet"]["f1"])] for r in rows]
    columns = [("Model", 3.9, "l"), ("n", 0.5, "r"), ("P", 0.85, "r"), ("R", 0.85, "r"),
               ("F1", 0.85, "r"), ("Acc", 0.85, "r"), ("bal. Acc", 1.1, "r"),
               ("hands F1", 1.1, "r"), ("feet F1", 1.1, "r")]
    overall = out / "gt_contact_table.png"
    _render_table(overall, columns, cells, bold=_best(cells, len(columns)))
    limb_cells = [[r["label"], *(_rate(r["per_limb"][limb]["f1"]) for limb in LIMBS)] for r in rows]
    per_limb = out / "gt_contact_table_per_limb.png"
    _render_table(per_limb, [("Model", 3.9, "l")] + [(x.replace("_", " "), 1.2, "r") for x in LIMBS], limb_cells)
    return [overall, per_limb]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS))
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: the benchmark's model-results directory)")
    args = parser.parse_args()
    spec = BENCHMARKS[args.benchmark]
    out = args.out or spec["results"]
    metrics = collect(args.benchmark)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{spec['prefix']}_contact_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    paths = write_tables(metrics, args.benchmark, out)
    unit = "scenes" if args.benchmark == "climbingvideos" else "clips"
    print(f"[score] {metrics[unit]} {unit}; missing: {metrics['missing'] or 'none'}")
    print(f"{'row':38s} {'n':>4s} {'P':>6s} {'R':>6s} {'F1':>6s} {'hands':>6s} {'feet':>6s} {'balAcc':>7s}")
    for r in metrics["rows"].values():
        o, feet = r["overall"], r.get("feet_ankles", r.get("feet"))
        print(f"{r['label']:38s} {r[unit]:4d} {_fmt(o['precision']):>6s} {_fmt(o['recall']):>6s} "
              f"{_fmt(o['f1']):>6s} {_fmt(r['hands']['f1']):>6s} {_fmt(feet['f1']):>6s} "
              f"{_fmt(o.get('balanced_accuracy')):>7s}")
    for p in [json_path, *paths]:
        print(f"[score] wrote {p}")
