"""Score published contact models against our manual labels on the ClimbingVideos test split.

Every model is put on the same ruler: the 108 held-out scenes, the six annotated
end-effector joints (left/right hand, foot, ankle), the manual tri-state labels
(1 contact, 0 free, -1 never annotated — dropped). A "hand" here is the annotator's
hand; a "foot" is the forefoot / toes; an "ankle" is the heel region. That is how
the 14-joint annotation schema and our own contact frames use the words.

Rows (each is one model output read at the joint level):

- **Ours** — ``features/human_optim/<scene>/contacts_{1,2}.npz``: the contact
  frames folded onto the six joints with an OR (palm/finger frames → hand,
  toe/ball frames → foot, heel → ankle), the same fold ``score_frame_contacts`` uses.
- **GVHMR / PromptHMR** — ``results/<model>/<scene>/pose_{raw,post}.npz``: the
  model's own six-joint stationary-joint probability, thresholded at 0.5 (and at
  0.8 for PromptHMR, its demo's cut-off). Neither model's post-processing changes it.
- **DECO / BSTRO / InteractVLM** — ``results/<model>/<scene>/contacts_vertex.npz``:
  per-vertex probabilities on the SMPL mesh, mapped to joints. InteractVLM is run
  with two text queries: ``scene`` (the README's word for scene/ground contact; on
  this model it only ever names the feet) and ``wall`` (names hands and feet), saved
  as ``interactvlm_wall`` and ``interactvlm_scene``. Two vertex→joint
  groupings × two decision rules:

    grouping ``seg``  — the SMPL part segmentation: hand = Hand ∪ HandIndex1,
                        foot = ToeBase, ankle = Foot (the rear-foot part);
    grouping ``lbs``  — each vertex goes to the joint with the largest skinning
                        weight: hand = wrist + hand joints, foot = foot joint,
                        ankle = ankle joint;
    rule ``any``      — the joint is in contact when at least one of its vertices
                        is above the model's own threshold;
    rule ``frac10``   — when at least 10 % of its vertices are.

Writes ``results/baseline_contact_metrics.json`` and ``results/baseline_contact_table.png``.
Models whose results are missing are skipped (and listed), so the table can be
rebuilt while the runs are still going.

Usage:
    uv run --no-sync python scripts/diagnostics/score_baseline_contacts.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from score_frame_contacts import _FOLD_14, metrics as _metrics  # noqa: E402
from score_prediction_contacts import _fmt, _render_table  # noqa: E402

DATA_ROOT = (_REPO_ROOT.parent / "data" / "ClimbingVideos").resolve()
FEATURES, RESULTS = DATA_ROOT / "features", DATA_ROOT / "results"
SEGMENTATION = _REPO_ROOT.parent / "other" / "deco" / "data"  # smpl(x)_vert_segmentation.json
BODY_MODELS = _REPO_ROOT.parent / "BetterHuman" / "models"

JOINTS = ["left_hand", "right_hand", "left_foot", "right_foot", "left_ankle", "right_ankle"]
POSE_MODELS = {"gvhmr": "GVHMR", "prompthmr": "PromptHMR"}
# Probability cut-offs per six-joint model: 0.5 for every row, plus the model's own demo
# threshold where it differs (PromptHMR's demo binarises its stationary head at 0.8).
POSE_THRESHOLDS = {"gvhmr": (0.5,), "prompthmr": (0.5, 0.8)}
VERTEX_MODELS = {"deco": "DECO", "bstro": "BSTRO", "interactvlm_wall": "InteractVLM (query: wall)",
                 "interactvlm_scene": "InteractVLM (query: scene)"}
GROUPINGS, RULES = ("seg", "lbs"), ("any", "frac10")
FRAC = 0.10

# SMPL / SMPL-X skinning joints per annotated joint (hand = wrist + everything below it).
_LBS_JOINTS = {
    "smpl": {"left_hand": [20, 22], "right_hand": [21, 23], "left_foot": [10], "right_foot": [11],
             "left_ankle": [7], "right_ankle": [8]},
    "smplx": {"left_hand": [20, *range(25, 40)], "right_hand": [21, *range(40, 55)],
              "left_foot": [10], "right_foot": [11], "left_ankle": [7], "right_ankle": [8]},
}
_SEG_PARTS = {"left_hand": ("leftHand", "leftHandIndex1"), "right_hand": ("rightHand", "rightHandIndex1"),
              "left_foot": ("leftToeBase",), "right_foot": ("rightToeBase",),
              "left_ankle": ("leftFoot",), "right_ankle": ("rightFoot",)}


def scene_dir(stage: str, scene: str) -> Path:
    return FEATURES / stage / scene[:2] / scene[2:4] / scene


def test_scenes() -> list[str]:
    return [line.split("\t")[0] for line in (RESULTS / "test_scenes.txt").read_text().splitlines() if line]


def truth(scene: str) -> np.ndarray:
    """``(T, 6)`` int8 tri-state manual labels for the single tracked person."""
    ann = np.load(scene_dir("annotation", scene) / "annotation.npz", allow_pickle=True)
    if ann["contacts"].shape[0] != 1 or bool(ann["ignored"][0]):
        raise SystemExit(f"[score] {scene}: expected one annotated, non-ignored person")
    names = [str(n) for n in ann["joint_names"]]
    return np.stack([ann["contacts"][0, names.index(j)] for j in JOINTS], axis=1).astype(np.int8)


# ────────────────────────────── readers → (T, 6) bool ─────────────────────────────

def ours(scene: str, tag: int) -> np.ndarray:
    c = np.load(scene_dir("human_optim", scene) / f"contacts_{tag}.npz", allow_pickle=True)
    names = [str(n) for n in c["contact_frame_names"]]
    contact = np.asarray(c["frame_contact"][0], bool)                      # (T, F)
    cols = [[names.index(f) for f in _FOLD_14[j] if f in names] for j in JOINTS]
    return np.stack([contact[:, g].any(1) for g in cols], axis=1)


def six_joint(path: Path, n_frames: int, threshold: float) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["contact_joint_names"]]
    probs = np.asarray(d["contacts"], np.float32)[:, [names.index(j) for j in JOINTS]]
    out = np.zeros((n_frames, 6), bool)
    out[np.asarray(d["frame_indices"], int)] = probs > threshold
    return out


class VertexGroups:
    """Vertex index sets per joint for one mesh topology, under both groupings."""

    def __init__(self, topology: str) -> None:
        seg = json.loads((SEGMENTATION / f"{topology}_vert_segmentation.json").read_text())
        self.seg = {j: np.unique(np.concatenate([seg[p] for p in parts])) for j, parts in _SEG_PARTS.items()}
        if topology == "smpl":
            weights = np.load(BODY_MODELS / "smpl" / "SMPL_NEUTRAL.npz", allow_pickle=True)["weights"]
        else:
            weights = np.load(BODY_MODELS / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)["weights"]
        owner = np.asarray(weights).argmax(1)
        self.lbs = {j: np.flatnonzero(np.isin(owner, ids)) for j, ids in _LBS_JOINTS[topology].items()}

    def fold(self, probs: np.ndarray, threshold: float, grouping: str, rule: str) -> np.ndarray:
        groups = getattr(self, grouping)
        hit = probs > threshold                                             # (T, V)
        cols = []
        for j in JOINTS:
            frac = hit[:, groups[j]].mean(1)
            cols.append(frac > 0 if rule == "any" else frac >= FRAC)
        return np.stack(cols, axis=1)


def vertex_rows(path: Path, n_frames: int, groups: dict[str, VertexGroups]) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    key, topology = ("contact_smpl", "smpl") if "contact_smpl" in d.files else ("contact_smplx", "smplx")
    probs = np.asarray(d[key], np.float32)
    idx = np.asarray(d["frame_indices"], int)
    thr = float(d["threshold"])
    out = {}
    for grouping in GROUPINGS:
        for rule in RULES:
            full = np.zeros((n_frames, 6), bool)
            full[idx] = groups[topology].fold(probs, thr, grouping, rule)
            out[f"{grouping}/{rule}"] = full
    return out


# ─────────────────────────────────── scoring ─────────────────────────────────

def _count(counts: dict, predicted: np.ndarray, labels: np.ndarray) -> None:
    for k, joint in enumerate(JOINTS):
        known = labels[:, k] >= 0
        got, want = predicted[known, k], labels[known, k] == 1
        c = counts[joint]
        c["tp"] += int(np.sum(got & want)); c["fp"] += int(np.sum(got & ~want))
        c["fn"] += int(np.sum(~got & want)); c["tn"] += int(np.sum(~got & ~want))


def collect() -> dict:
    scenes = test_scenes()
    groups = {"smpl": VertexGroups("smpl"), "smplx": VertexGroups("smplx")}
    rows: dict[str, dict] = {}      # row key → {label, counts per joint, scenes}
    missing: dict[str, int] = {}

    def row(key: str, label: str) -> dict:
        return rows.setdefault(key, {"label": label, "scenes": 0,
                                     "counts": {j: dict(tp=0, fp=0, fn=0, tn=0) for j in JOINTS}})

    for scene in scenes:
        labels = truth(scene)
        n = len(labels)
        for tag in (1, 2):
            r = row(f"ours/contacts_{tag}", f"Ours (contacts_{tag})")
            _count(r["counts"], ours(scene, tag), labels); r["scenes"] += 1
        for model, name in POSE_MODELS.items():
            for variant in ("raw", "post"):
                path = RESULTS / model / scene / f"pose_{variant}.npz"
                if not path.exists():
                    missing[f"{model}/{variant}"] = missing.get(f"{model}/{variant}", 0) + 1
                    continue
                for thr in POSE_THRESHOLDS[model]:
                    r = row(f"{model}/{variant}/p{thr}", f"{name} ({variant}, p > {thr})")
                    _count(r["counts"], six_joint(path, n, thr), labels); r["scenes"] += 1
        for model, name in VERTEX_MODELS.items():
            path = RESULTS / model / scene / "contacts_vertex.npz"
            if not path.exists():
                missing[model] = missing.get(model, 0) + 1
                continue
            for mapping, predicted in vertex_rows(path, n, groups).items():
                r = row(f"{model}/{mapping}", f"{name} ({mapping})")
                _count(r["counts"], predicted, labels); r["scenes"] += 1

    for r in rows.values():
        pooled = {k: sum(r["counts"][j][k] for j in JOINTS) for k in ("tp", "fp", "fn", "tn")}
        hands = {k: sum(r["counts"][j][k] for j in JOINTS[:2]) for k in ("tp", "fp", "fn", "tn")}
        feet = {k: sum(r["counts"][j][k] for j in JOINTS[2:]) for k in ("tp", "fp", "fn", "tn")}
        r["overall"] = _metrics(pooled)
        r["hands"], r["feet_ankles"] = _metrics(hands), _metrics(feet)
        r["per_joint"] = {j: _metrics(r["counts"][j]) for j in JOINTS}
        del r["counts"]
    return {"scenes": len(scenes), "joints": JOINTS,
            "ground_truth": "features/annotation/<scene>/annotation.npz (14-joint tri-state, first six)",
            "thresholds_six_joint": POSE_THRESHOLDS, "frac_rule": FRAC,
            "missing": missing, "rows": rows}


def write_table(m: dict, path: Path) -> None:
    keys = list(m["rows"])
    cells = []
    for k in keys:
        r = m["rows"][k]
        cells.append([r["label"], str(r["scenes"]),
                      *(_fmt(r["overall"][x]) for x in ("precision", "recall", "f1")),
                      _fmt(r["hands"]["f1"]), _fmt(r["feet_ankles"]["f1"]),
                      *(_fmt(r["per_joint"][j]["f1"]) for j in JOINTS)])
    columns = [("Model", 3.6, "l"), ("n", 0.6, "r"), ("P", 0.9, "r"), ("R", 0.9, "r"), ("F1", 0.9, "r"),
               ("hands F1", 1.1, "r"), ("feet F1", 1.1, "r"),
               *((j.replace("_", " "), 1.2, "r") for j in JOINTS)]
    bold = set()
    for col in range(2, len(columns)):
        vals = {i: float(c[col]) for i, c in enumerate(cells) if c[col] != "—"}
        if vals:
            best = max(vals.values()); bold |= {(i, col) for i, v in vals.items() if v == best}
    _render_table(path, columns, cells, bold=frozenset(bold))


if __name__ == "__main__":
    m = collect()
    (RESULTS / "baseline_contact_metrics.json").write_text(json.dumps(m, indent=2))
    write_table(m, RESULTS / "baseline_contact_table.png")
    print(f"[score] {m['scenes']} test scenes; missing: {m['missing'] or 'none'}")
    print(f"{'row':34s} {'n':>4s} {'P':>6s} {'R':>6s} {'F1':>6s} {'hands':>6s} {'feet':>6s}")
    for k, r in m["rows"].items():
        o = r["overall"]
        print(f"{r['label']:34s} {r['scenes']:4d} {_fmt(o['precision']):>6s} {_fmt(o['recall']):>6s} "
              f"{_fmt(o['f1']):>6s} {_fmt(r['hands']['f1']):>6s} {_fmt(r['feet_ankles']['f1']):>6s}")
