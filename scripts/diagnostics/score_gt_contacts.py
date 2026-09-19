"""Score published contact models against the climb_wall_3 force boards.

This is the ClimbingVideos baseline table run again, but with measured ground
truth instead of hand annotation. Every clip in ``peter/climb_wall_3_gt`` has four
instrumented holds — one per limb — so a limb is "in contact" when the board under
it carries load: the same 50 N enter / 30 N exit hysteresis that built the GT
labels stored in ``human_optim/contacts_1.npz``. Nothing here is an opinion; a
frame is a contact because a sensor said so.

The models run on the ORIGINAL ``cam_left.mp4``, which is also the video our own
single-view pipeline used, so every row saw identical pixels.

Rows:

- **Always in contact** — the trivial predictor. On a climbing wall the climber
  holds on with all four limbs about 83 % of the time, so this row is the floor
  any real model has to beat. Read the table against it, not against zero.
- **Ours** — ``out_climb_wall_3_single/<clip>/human_optim/contacts_{1,2}.npz``:
  the 18 contact frames folded onto the four limbs with an OR (palm/finger/thumb
  → hand, toes ∪ heel → foot).
- **GVHMR / PromptHMR** — their six-joint stationary-joint probability, with the
  foot and ankle columns OR-ed into one foot (the board measures the whole foot).
- **DECO / BSTRO / InteractVLM** — per-vertex contact probability on the SMPL mesh,
  mapped to limbs two ways (``seg`` = part segmentation, ``lbs`` = largest skinning
  weight) under two rules (``any`` = one vertex above threshold, ``frac10`` = a
  tenth of them). InteractVLM is queried with both its text prompts. Note that a
  board carries the whole foot, so forefoot and heel are one group here — and once
  they are merged the two mappings pick out almost the same vertices (they differ
  by 2–3 of ~250), so their rows agree. Both are kept so this table lines up with
  the ClimbingVideos one, where the split foot made them genuinely differ.

Because the climber is holding on with all four limbs about 83 % of the time,
precision, recall and F1 are all flattered by simply guessing "contact". The
table therefore also reports **balanced accuracy** — the average of how often a
real contact is caught and how often a free limb is correctly left alone. Always
guessing contact scores 0.500 on it, so anything at or below that has learnt
nothing about this dataset.

Writes ``peter/climb_wall_3_gt/others/gt_contact_metrics.json`` and two PNG tables.

Usage:
    uv run --no-sync python scripts/diagnostics/score_gt_contacts.py
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

from score_frame_contacts import _FOLD_10  # noqa: E402
from score_prediction_contacts import _rates, _render_table  # noqa: E402


def _fmt(value: float | None) -> str:
    """A rate as ``.902`` — no leading zero, the convention for a number that
    cannot exceed one. Exactly 1 keeps its digit."""
    if value is None:
        return "—"
    text = f"{value:.3f}"
    return text[1:] if text.startswith("0.") else text

GT_ROOT = _REPO / "peter" / "climb_wall_3_gt" / "human"
OURS_ROOT = _REPO / "peter" / "out_climb_wall_3_single"
OTHERS = _REPO / "peter" / "climb_wall_3_gt" / "others"
SEGMENTATION = _REPO.parent / "other" / "deco" / "data"
BODY_MODELS = _REPO.parent / "BetterHuman" / "models"

LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]
POSE_MODELS = {"gvhmr": "GVHMR", "prompthmr": "PromptHMR"}
POSE_THRESHOLDS = {"gvhmr": (0.5,), "prompthmr": (0.5, 0.8)}
VERTEX_MODELS = {"deco": "DECO", "bstro": "BSTRO", "interactvlm_wall": "InteractVLM (query: wall)",
                 "interactvlm_scene": "InteractVLM (query: scene)"}
GROUPINGS, RULES, FRAC = ("seg", "lbs"), ("any", "frac10"), 0.10

# A board carries the whole foot, so the foot group is forefoot + heel on both mappings.
_SEG_PARTS = {"left_hand": ("leftHand", "leftHandIndex1"), "right_hand": ("rightHand", "rightHandIndex1"),
              "left_foot": ("leftToeBase", "leftFoot"), "right_foot": ("rightToeBase", "rightFoot")}
_LBS_JOINTS = {
    "smpl": {"left_hand": [20, 22], "right_hand": [21, 23], "left_foot": [7, 10], "right_foot": [8, 11]},
    "smplx": {"left_hand": [20, *range(25, 40)], "right_hand": [21, *range(40, 55)],
              "left_foot": [7, 10], "right_foot": [8, 11]},
}
_SIX_TO_FOUR = {"left_hand": ("left_hand",), "right_hand": ("right_hand",),
                "left_foot": ("left_foot", "left_ankle"), "right_foot": ("right_foot", "right_ankle")}


def clips() -> list[str]:
    return sorted(d.name for d in GT_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_"))


def truth(clip: str) -> np.ndarray:
    """``(T, 4)`` bool — the board labels, in ``LIMBS`` order."""
    d = np.load(GT_ROOT / clip / "human_optim" / "contacts_1.npz", allow_pickle=True)
    names = [str(n) for n in d["limb_names"]]
    return np.asarray(d["limb_contact"][0], bool)[:, [names.index(limb) for limb in LIMBS]]


def ours(clip: str, tag: int, n_frames: int, optim_dir: str = "human_optim") -> np.ndarray:
    """``(T, 4)`` bool — one run's contact frames folded onto the four limbs."""
    d = np.load(OURS_ROOT / clip / optim_dir / f"contacts_{tag}.npz", allow_pickle=True)
    names = [str(n) for n in d["contact_frame_names"]]
    contact = np.asarray(d["frame_contact"][0], bool)                       # (T, F)
    cols = [[names.index(f) for f in _FOLD_10[limb] if f in names] for limb in LIMBS]
    return np.stack([contact[:, g].any(1) for g in cols], axis=1)[:n_frames]


def six_joint(path: Path, n_frames: int, threshold: float) -> np.ndarray:
    """The six-joint stationary head, with foot and ankle OR-ed into one foot."""
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["contact_joint_names"]]
    probs = np.asarray(d["contacts"], np.float32) > threshold
    out = np.zeros((n_frames, 4), bool)
    hit = np.stack([probs[:, [names.index(j) for j in _SIX_TO_FOUR[limb]]].any(1) for limb in LIMBS], axis=1)
    out[np.asarray(d["frame_indices"], int)] = hit
    return out


class VertexGroups:
    """Vertex index sets per limb for one mesh topology, under both groupings."""

    def __init__(self, topology: str) -> None:
        seg = json.loads((SEGMENTATION / f"{topology}_vert_segmentation.json").read_text())
        self.seg = {j: np.unique(np.concatenate([seg[p] for p in parts])) for j, parts in _SEG_PARTS.items()}
        name = "SMPL_NEUTRAL.npz" if topology == "smpl" else "SMPLX_NEUTRAL.npz"
        owner = np.asarray(np.load(BODY_MODELS / topology / name, allow_pickle=True)["weights"]).argmax(1)
        self.lbs = {j: np.flatnonzero(np.isin(owner, ids)) for j, ids in _LBS_JOINTS[topology].items()}

    def fold(self, probs: np.ndarray, threshold: float, grouping: str, rule: str) -> np.ndarray:
        groups = getattr(self, grouping)
        hit = probs > threshold                                             # (T, V)
        frac = [hit[:, groups[limb]].mean(1) for limb in LIMBS]
        return np.stack([f > 0 if rule == "any" else f >= FRAC for f in frac], axis=1)


def vertex_rows(path: Path, n_frames: int, groups: dict[str, VertexGroups]) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    key, topology = ("contact_smpl", "smpl") if "contact_smpl" in d.files else ("contact_smplx", "smplx")
    probs, idx, thr = np.asarray(d[key], np.float32), np.asarray(d["frame_indices"], int), float(d["threshold"])
    out = {}
    for grouping in GROUPINGS:
        for rule in RULES:
            full = np.zeros((n_frames, 4), bool)
            full[idx] = groups[topology].fold(probs, thr, grouping, rule)
            out[f"{grouping}/{rule}"] = full
    return out


def _rates_balanced(counts: dict[str, int]) -> dict:
    """``_rates`` plus balanced accuracy, which the 83 % contact prior cannot inflate."""
    r = _rates(counts)
    tpr = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None
    tnr = counts["tn"] / (counts["tn"] + counts["fp"]) if counts["tn"] + counts["fp"] else None
    r["balanced_accuracy"] = None if tpr is None or tnr is None else (tpr + tnr) / 2
    return r


def _count(counts: dict, predicted: np.ndarray, labels: np.ndarray) -> None:
    for k, limb in enumerate(LIMBS):
        got, want = predicted[:len(labels), k], labels[:, k]
        c = counts[limb]
        c["tp"] += int(np.sum(got & want)); c["fp"] += int(np.sum(got & ~want))
        c["fn"] += int(np.sum(~got & want)); c["tn"] += int(np.sum(~got & ~want))


def collect() -> dict:
    groups = {"smpl": VertexGroups("smpl"), "smplx": VertexGroups("smplx")}
    rows: dict[str, dict] = {}
    missing: dict[str, int] = {}

    def row(key: str, label: str) -> dict:
        return rows.setdefault(key, {"label": label, "clips": 0,
                                     "counts": {limb: dict(tp=0, fp=0, fn=0, tn=0) for limb in LIMBS}})

    names = clips()
    for clip in names:
        labels = truth(clip)
        n = int(np.load(GT_ROOT / clip / "cameras.npz", allow_pickle=True)["num_frames"])

        r = row("floor/always", "Always in contact")
        _count(r["counts"], np.ones((n, 4), bool), labels); r["clips"] += 1
        for tag in (1, 2):
            r = row(f"ours/contacts_{tag}", f"Ours (contacts_{tag})")
            _count(r["counts"], ours(clip, tag, n), labels); r["clips"] += 1
        for model, label in POSE_MODELS.items():
            for variant in ("raw", "post"):
                path = OTHERS / model / clip / f"pose_{variant}.npz"
                if not path.exists():
                    missing[f"{model}/{variant}"] = missing.get(f"{model}/{variant}", 0) + 1
                    continue
                for thr in POSE_THRESHOLDS[model]:
                    r = row(f"{model}/{variant}/p{thr}", f"{label} ({variant}, p > {thr})")
                    _count(r["counts"], six_joint(path, n, thr), labels); r["clips"] += 1
        for model, label in VERTEX_MODELS.items():
            path = OTHERS / model / clip / "contacts_vertex.npz"
            if not path.exists():
                missing[model] = missing.get(model, 0) + 1
                continue
            for mapping, predicted in vertex_rows(path, n, groups).items():
                r = row(f"{model}/{mapping}", f"{label} ({mapping})")
                _count(r["counts"], predicted, labels); r["clips"] += 1

    for r in rows.values():
        pooled = {k: sum(r["counts"][limb][k] for limb in LIMBS) for k in ("tp", "fp", "fn", "tn")}
        hands = {k: sum(r["counts"][limb][k] for limb in LIMBS[:2]) for k in ("tp", "fp", "fn", "tn")}
        feet = {k: sum(r["counts"][limb][k] for limb in LIMBS[2:]) for k in ("tp", "fp", "fn", "tn")}
        r["overall"] = _rates_balanced(pooled)
        r["hands"], r["feet"] = _rates_balanced(hands), _rates_balanced(feet)
        r["per_limb"] = {limb: _rates_balanced(r["counts"][limb]) for limb in LIMBS}
        del r["counts"]
    return {"clips": len(names), "limbs": LIMBS,
            "ground_truth": "climb_wall_3_gt/human/<clip>/human_optim/contacts_1.npz limb_contact "
                            "— measured board load, 50 N enter / 30 N exit",
            "video": "peter/climb_wall_3/<clip>/cam_left.mp4 (original, as our single-view run used)",
            "thresholds_six_joint": POSE_THRESHOLDS, "frac_rule": FRAC,
            "missing": missing, "rows": rows}


def write_tables(m: dict) -> list[Path]:
    keys = list(m["rows"])
    cells = [[m["rows"][k]["label"], str(m["rows"][k]["clips"]),
              *(_fmt(m["rows"][k]["overall"][x])
                for x in ("precision", "recall", "f1", "accuracy", "balanced_accuracy")),
              _fmt(m["rows"][k]["hands"]["f1"]), _fmt(m["rows"][k]["feet"]["f1"])] for k in keys]
    columns = [("Model", 3.9, "l"), ("n", 0.5, "r"), ("P", 0.85, "r"), ("R", 0.85, "r"),
               ("F1", 0.85, "r"), ("Acc", 0.85, "r"), ("bal. Acc", 1.1, "r"),
               ("hands F1", 1.1, "r"), ("feet F1", 1.1, "r")]
    bold = set()
    for col in range(2, len(columns)):
        vals = {i: float(c[col]) for i, c in enumerate(cells) if c[col] != "—"}
        if vals:
            best = max(vals.values()); bold |= {(i, col) for i, v in vals.items() if v == best}
    overall = OTHERS / "gt_contact_table.png"
    _render_table(overall, columns, cells, bold=frozenset(bold))

    limb_cells = [[m["rows"][k]["label"], *(_fmt(m["rows"][k]["per_limb"][limb]["f1"]) for limb in LIMBS)]
                  for k in keys]
    limb = OTHERS / "gt_contact_table_per_limb.png"
    _render_table(limb, [("Model", 3.9, "l")] + [(x.replace("_", " "), 1.2, "r") for x in LIMBS], limb_cells)
    return [overall, limb]


if __name__ == "__main__":
    metrics = collect()
    OTHERS.mkdir(parents=True, exist_ok=True)
    json_path = OTHERS / "gt_contact_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    paths = write_tables(metrics)
    print(f"[score] {metrics['clips']} clips; missing: {metrics['missing'] or 'none'}")
    print(f"{'row':38s} {'n':>3s} {'P':>6s} {'R':>6s} {'F1':>6s} {'Acc':>6s} {'balAcc':>7s} "
          f"{'hands':>6s} {'feet':>6s}")
    for r in metrics["rows"].values():
        o = r["overall"]
        print(f"{r['label']:38s} {r['clips']:3d} {_fmt(o['precision']):>6s} {_fmt(o['recall']):>6s} "
              f"{_fmt(o['f1']):>6s} {_fmt(o['accuracy']):>6s} {_fmt(o['balanced_accuracy']):>7s} "
              f"{_fmt(r['hands']['f1']):>6s} {_fmt(r['feet']['f1']):>6s}")
    for p in [json_path, *paths]:
        print(f"[score] wrote {p}")
