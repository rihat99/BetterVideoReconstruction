"""Fit the contact-label confidence against the manual annotations, and write the TOML.

The confidence a label carries is P(this label says what a human annotator would say).
This script measures it: it pairs every automatic label on an annotated corpus scene
with the manual label for the same joint and frame, fits one small logistic model per
label class over three witnesses (which joint, how still it was, how far it is from a
label change), and prints how well the result is calibrated. With ``--write`` it
replaces the ``[contact_confidence]`` section of the given TOMLs with the fitted
numbers.

The annotated scenes are the downstream learner's TEST split, so these coefficients are
fitted on data that learner never trains on — but they are also not a held-out set for
the confidence itself. The printout says what the fit looks like on held-out videos
(grouped cross-validation), which is the honest number.

    uv run --no-sync python datasets/climbingvideos/calibrate_contact_confidence.py \\
        --config configs/human_optimization.toml --write
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from bvr.paths import REPO_ROOT

from datasets.examples.score_frame_contacts import fold_for_schema
from bvr.config import load_config
from bvr.human_optim.confidence import (
    CONTACT_CONFIDENCE_METHOD, GROUP_NAMES, REPORT_GROUPS, fold_contacts, fold_drift,
    frame_report_joints, transition_seconds, witness_features,
)

_DEFAULT_CONFIG = REPO_ROOT / "configs" / "human_optimization.toml"
_CORPUS_CONFIG = Path(__file__).resolve().parent / "configs" / "corpus.toml"

DRIFT_FLOOR_M = 0.005        # log(drift + floor): a perfectly still joint needs a floor
TRANSITION_CAP_S = 1.0       # a label change further away than this is just "far"
STILL_M = 0.08               # the cell table's stillness cut, the learner's own diagnostic
NBINS = 10
FOLDS = 5

#: Provenance written into the TOML. Replace when the manual labels have been checked.
FIT_NOTE = "2026-09-12 — annotated ClimbingVideos scenes after the manual-label check"


# ─────────────────────────────── the rows ──────────────────────────────────


def scene_split(data_root: Path) -> dict[str, tuple[str, str]]:
    """``scene_id`` → (video_id, dataset_split) from the scene database."""
    with sqlite3.connect(f"file:{data_root / 'scenes' / 'scenes.db'}?mode=ro", uri=True) as db:
        rows = db.execute("select scene_id, video_id, dataset_split from scenes").fetchall()
    return {r[0]: (str(r[1]), str(r[2] or "none")) for r in rows}


#: One row per (person, video frame, annotated joint). ``video_id`` and ``split``
#: come from the scene database, everything else from the scene's own two files.
COLUMNS = ("scene", "video_id", "split", "group", "drift", "x_drift", "x_trans",
           "auto", "manual", "correct", "conf_old")


def scene_rows(scene: str, data_root: Path, tag: str, video_id: str,
               split: str) -> tuple[list[dict], list[str]]:
    """The rows of one scene, one dict of equal-length arrays per annotated joint."""
    aa, bb = scene[:2], scene[2:4]
    ann = np.load(data_root / "features" / "annotation" / aa / bb / scene / "annotation.npz",
                  allow_pickle=True)
    pred = np.load(data_root / "features" / "human_optim" / aa / bb / scene / f"contacts_{tag}.npz",
                   allow_pickle=True)
    notes: list[str] = []

    ann_names = [str(n) for n in ann["joint_names"]]
    fold, _schema = fold_for_schema(ann_names)
    ann_contacts = np.transpose(ann["contacts"].astype(np.int8), (0, 2, 1))   # (P,N,J)
    ann_ids = [int(o) for o in ann["object_ids"]]
    ignored = ann["ignored"].astype(bool)
    ann_col = {n: i for i, n in enumerate(ann_names)}

    joint_names = [str(n) for n in pred["joint_names"]]
    parents = np.asarray(pred["parents"], dtype=np.int32)
    report_joint = frame_report_joints(pred["frame_parent_joint"], joint_names, parents)
    frame_index = {str(n): i for i, n in enumerate(pred["contact_frame_names"])}
    valid = pred["valid_mask"].astype(bool)
    fps = float(pred["fps"])
    num_frames = int(pred["frame_contact"].shape[1])
    if ann_contacts.shape[1] != num_frames:
        raise ValueError(f"frame-count mismatch pred={num_frames} ann={ann_contacts.shape[1]}")

    # The same fold and the same feature transforms the stage runs.
    joint_contact = fold_contacts(pred["frame_contact"].astype(bool), report_joint, len(parents))
    joint_drift = fold_drift(np.asarray(pred["frame_drift_m"]), report_joint, len(parents))
    # The previous weight, for the side-by-side — only where an artifact still carries it.
    old_method = str(pred["confidence_method"]) if "confidence_method" in pred.files else ""
    old_confidence = (np.asarray(pred["joint_label_confidence"])
                      if old_method and old_method != CONTACT_CONFIDENCE_METHOD else None)

    rows: list[dict] = []
    for person, object_id in enumerate([int(o) for o in pred["object_ids"]]):
        if object_id not in ann_ids:
            notes.append(f"{scene}: predicted object {object_id} is not annotated")
            continue
        ap = ann_ids.index(object_id)
        if bool(ignored[ap]):
            notes.append(f"{scene}: annotated object {object_id} is flagged ignored")
            continue
        x_drift, x_trans = witness_features(
            joint_drift[person], transition_seconds(joint_contact[person], valid[person], fps),
            DRIFT_FLOOR_M, TRANSITION_CAP_S)

        for name, frames in fold.items():
            columns = [frame_index[f] for f in frames if f in frame_index]
            reports = np.unique(report_joint[columns]) if columns else np.zeros(0, int)
            if reports.size != 1:
                # A whole-foot (10-joint) annotation folds toes AND heel — two report joints
                # with separate coefficients. No such scene exists; refuse rather than guess.
                raise ValueError(f"{name} folds onto {reports.size} report joints — "
                                 "only the 14-joint annotation schema is supported")
            j = int(reports[0])
            group = REPORT_GROUPS.get(joint_names[j])
            if group is None:
                notes.append(f"{scene}: {name} reports at uncalibrated {joint_names[j]} — skipped")
                continue

            labels = ann_contacts[ap][:, ann_col[name]]
            keep = (labels >= 0) & valid[person] & np.isfinite(joint_drift[person, :, j])
            if not keep.any():
                continue
            drift = joint_drift[person][keep, j].astype(np.float64)
            auto, manual = joint_contact[person][keep, j], labels[keep] == 1
            rows.append({
                "scene": np.repeat(scene, keep.sum()),
                "video_id": np.repeat(video_id, keep.sum()),
                "split": np.repeat(split, keep.sum()),
                "group": np.repeat(group, keep.sum()),
                "drift": drift,
                "x_drift": x_drift[keep, j].astype(np.float64),
                "x_trans": x_trans[keep, j].astype(np.float64),
                "auto": auto.astype(np.int64),
                "manual": manual.astype(np.int64),
                "correct": (auto == manual).astype(np.int64),
                "conf_old": (old_confidence[person][keep, j].astype(np.float64)
                             if old_confidence is not None
                             else np.full(int(keep.sum()), np.nan)),
            })
    return rows, notes


def build_table(data_root: Path, tag: str) -> dict[str, np.ndarray]:
    """Every annotated scene folded into one flat table."""
    scenes = sorted(p.parent.name for p in
                    (data_root / "features" / "annotation").glob("*/*/*/annotation.npz"))
    split = scene_split(data_root)
    rows: list[dict] = []
    notes, failed = [], []
    for scene in scenes:
        video_id, dataset_split = split.get(scene, (scene, "none"))
        if dataset_split == "none":           # removed from the corpus splits: not a witness
            notes.append(f"{scene}: no dataset_split in the scene DB — skipped")
            continue
        try:
            scene_parts, scene_notes = scene_rows(
                scene, data_root, tag, video_id, dataset_split)
        except Exception as error:            # a broken scene must not kill the fit
            failed.append(f"{scene}: {type(error).__name__}: {error}")
            continue
        rows += scene_parts
        notes += scene_notes
    table = {k: np.concatenate([r[k] for r in rows]) if rows else np.zeros(0) for k in COLUMNS}
    print(f"annotated scenes: {len(scenes)}   used: {len(np.unique(table['scene']))}   "
          f"videos: {len(np.unique(table['video_id']))}   rows: {len(table['correct'])}")
    for line in failed:
        print(f"  FAILED {line}")
    for line in sorted(set(notes)):
        print(f"  note   {line}")
    return table


# ─────────────────────────────── the model ─────────────────────────────────


def _mean(values: np.ndarray) -> float:
    """Mean of the finite entries; ``NaN`` when there are none (and no warning)."""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def design(table: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Feature matrix: the two witnesses plus one indicator column per joint group."""
    columns = [table["x_drift"][mask], table["x_trans"][mask]]
    columns += [(table["group"][mask] == g).astype(np.float64) for g in GROUP_NAMES]
    return np.column_stack(columns)


def _logistic(**kwargs) -> LogisticRegression:
    """The one solver setting, used for the fit and for the cross-validation alike.

    No global intercept: the four group indicators are the intercept, one per group. A
    shared intercept *plus* four offsets is one parameter too many — the split between
    them is then pinned only by the weak penalty, which left the fitted numbers drifting
    in the third decimal from run to run. With the intercept dropped every parameter is
    determined by the data, and a group with no rows in this class (contact at a knee)
    keeps a coefficient of exactly zero.
    """
    return LogisticRegression(C=1e3, solver="lbfgs", tol=1e-8, max_iter=5000,
                              fit_intercept=False, **kwargs)


def fit_head(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """One logistic head → the named coefficients the TOML stores."""
    weights = _logistic().fit(x, y).coef_[0]
    return {"intercept": 0.0,
            "log_drift": float(weights[0]), "log_transition": float(weights[1]),
            **{g: float(w) for g, w in zip(GROUP_NAMES, weights[2:])}}


def cross_validated(table: dict[str, np.ndarray], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities from a video-grouped K-fold fit, and the constant
    baseline (the training folds' accuracy) held out the same way."""
    x, y, groups = design(table, mask), table["correct"][mask].astype(float), table["video_id"][mask]
    out, constant = np.full(y.shape, np.nan), np.full(y.shape, np.nan)
    for train, test in GroupKFold(n_splits=FOLDS).split(x, y, groups):
        constant[test] = y[train].mean()
        if y[train].min() == y[train].max():
            out[test] = y[train].mean()
            continue
        out[test] = _logistic().fit(x[train], y[train]).predict_proba(x[test])[:, 1]
    return out, constant


def auroc(score: np.ndarray, target: np.ndarray) -> float | None:
    ok = np.isfinite(score)
    if ok.sum() < 10 or target[ok].min() == target[ok].max():
        return None
    return float(roc_auc_score(target[ok], score[ok]))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray) -> float:
    """Expected calibration error: how far the stated probability is from the truth."""
    bins = np.clip(np.digitize(p, np.linspace(0.0, 1.0, NBINS + 1)[1:-1]), 0, NBINS - 1)
    return float(sum(m.mean() * abs(p[m].mean() - y[m].mean())
                     for m in (bins == k for k in range(NBINS)) if m.any()))


def metrics_line(name: str, p: np.ndarray, y: np.ndarray) -> str:
    a = auroc(p, y)
    return f"{name:30s} {'-' if a is None else f'{a:8.4f}'} {brier(p, y):9.4f} {ece(p, y):8.4f}"


# ────────────────────────────── the printout ───────────────────────────────


def report(table: dict[str, np.ndarray], heads: dict[str, dict[str, float]]) -> None:
    correct = table["correct"].astype(float)
    classes = {"contact": table["auto"] == 1, "free": table["auto"] == 0}

    print("\nrows per split and joint group")
    print(f"{'split':8s} {'group':8s} {'rows':>8s} {'manual=1':>9s} {'auto=1':>8s} {'agree':>8s}")
    for split in ("test", "train", "none"):
        for group in GROUP_NAMES + ("ALL",):
            m = table["split"] == split
            m = m if group == "ALL" else (m & (table["group"] == group))
            if not m.any():
                continue
            print(f"{split:8s} {group:8s} {int(m.sum()):8d} {table['manual'][m].mean():9.3f} "
                  f"{table['auto'][m].mean():8.3f} {correct[m].mean():8.3f}")

    print("\nrows per label class and joint group (what each coefficient is fitted on)")
    print(f"{'class':8s} {'group':8s} {'rows':>8s} {'agree':>8s}")
    for name, cls in classes.items():
        for group in GROUP_NAMES:
            m = cls & (table["group"] == group)
            print(f"{name:8s} {group:8s} {int(m.sum()):8d} "
                  f"{correct[m].mean() if m.any() else float('nan'):8.3f}")

    out_of_fold = np.full(correct.shape, np.nan)
    print(f"\nvideo-grouped {FOLDS}-fold cross-validation (out-of-fold, "
          f"{len(np.unique(table['video_id']))} videos)")
    print(f"{'':30s} {'AUROC':>8s} {'Brier':>9s} {'ECE':>8s}")
    for name, cls in classes.items():
        p, constant = cross_validated(table, cls)
        out_of_fold[cls] = p
        y = correct[cls]
        print(metrics_line(f"{name}: fitted", p, y))
        print(metrics_line(f"{name}: constant baseline", constant, y))
        old = np.isfinite(table["conf_old"][cls])
        if old.any():
            # Nothing is fitted here, so there is no fold to hold out: this is the old
            # artifact weight read straight as P(correct), on the rows that still carry it.
            print(metrics_line(f"{name}: previous weight ({int(old.sum())} rows, not fitted)",
                               table["conf_old"][cls][old], y[old]))
    print(metrics_line("both classes pooled", out_of_fold, correct))

    print("\nreliability of the fitted confidence (out-of-fold, both classes)")
    print(f"    {'bin':12s} {'rows':>8s} {'share':>7s} {'stated':>8s} {'actual':>8s} {'gap':>8s}")
    bins = np.clip(np.digitize(out_of_fold, np.linspace(0.0, 1.0, NBINS + 1)[1:-1]), 0, NBINS - 1)
    for k in range(NBINS):
        m = bins == k
        low, high = k / NBINS, (k + 1) / NBINS
        if not m.any():
            print(f"    [{low:.1f},{high:.1f})  {0:8d}")
            continue
        print(f"    [{low:.1f},{high:.1f})  {int(m.sum()):8d} {m.mean():7.3f} "
              f"{out_of_fold[m].mean():8.3f} {correct[m].mean():8.3f} "
              f"{out_of_fold[m].mean() - correct[m].mean():8.3f}")

    print(f"\nthe four cells the redesign was about (still = drift < {STILL_M} m over the window)")
    print(f"{'cell':22s} {'rows':>8s} {'share':>7s} {'agree':>8s} {'new conf':>9s} {'old conf':>9s}")
    still = table["drift"] < STILL_M
    for is_still in (True, False):
        for is_contact in (True, False):
            m = (still == is_still) & (table["manual"] == int(is_contact))
            if not m.any():
                continue
            cell = f"{'still' if is_still else 'moving'} & {'contact' if is_contact else 'free'}"
            print(f"{cell:22s} {int(m.sum()):8d} {m.mean():7.3f} {correct[m].mean():8.3f} "
                  f"{_mean(out_of_fold[m]):9.3f} {_mean(table['conf_old'][m]):9.3f}")

    print("\nfitted coefficients (all annotated scenes). `intercept` is 0 by construction: "
          "each group\ncarries its own, so a group with no rows in a class keeps exactly 0.")
    for name, head in heads.items():
        print(f"  [{name}]")
        for key, value in head.items():
            print(f"    {key:16s} = {value:+11.5f}")


# ─────────────────────────────── the TOML ──────────────────────────────────

_HEADER = re.compile(r"^\s*\[([^\]]+)\]")
_DROP = ("contact_confidence", "depth_fit")

_BLOCK = """# Label confidence — P(the label agrees with a human annotator), one logistic head per label
# class over three witnesses: the joint group, its drift over the readout window, and the time
# to the nearest change of its own label. Fitted by datasets/climbingvideos/calibrate_contact_confidence.py on
# the annotated corpus scenes (docs/contact_confidence.md). `intercept` is 0 by construction:
# each group term IS that group's intercept. Do not hand-edit the numbers.
[contact_confidence]
drift_floor_m = {floor}         # log(drift + floor)
transition_cap_s = {cap}        # seconds to the nearest label change, capped
fit = "{note}"

[contact_confidence.contact]    # the label says contact
{contact}

[contact_confidence.free]       # the label says free
{free}
"""


def toml_block(heads: dict[str, dict[str, float]]) -> str:
    """The ``[contact_confidence]`` section to write into a config."""
    body = {name: "\n".join(f"{k} = {v:.6f}" for k, v in head.items())
            for name, head in heads.items()}
    return _BLOCK.format(floor=DRIFT_FLOOR_M, cap=TRANSITION_CAP_S, note=FIT_NOTE, **body)


def write_config(path: Path, block: str) -> None:
    """Replace the ``[contact_confidence*]`` and ``[depth_fit]`` tables by ``block``."""
    lines = path.read_text().splitlines(keepends=True)
    out: list[str] = []
    i, inserted = 0, False
    while i < len(lines):
        header = _HEADER.match(lines[i])
        if header is None or header.group(1).split(".")[0] not in _DROP:
            out.append(lines[i])
            i += 1
            continue
        while out and out[-1].startswith("#"):       # the comment block above the table
            out.pop()
        while out and not out[-1].strip():
            out.pop()
        end = next((k for k in range(i + 1, len(lines)) if _HEADER.match(lines[k])), len(lines))
        while end > i + 1 and (lines[end - 1].startswith("#") or not lines[end - 1].strip()):
            end -= 1                                 # …belongs to whatever comes next
        if not inserted:
            out.append("\n" + block)
            inserted = True
        i = end
    if not inserted:
        out.append("\n" + block)
    path.write_text("".join(out))
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, nargs="+", default=[_DEFAULT_CONFIG],
                    help="human-optimization TOML(s) to update")
    ap.add_argument("--contacts", default="1", choices=("1", "2"), help="contacts_<tag>.npz")
    ap.add_argument("--write", action="store_true", help="rewrite the given TOMLs")
    args = ap.parse_args(argv)

    data_root = Path(load_config(_CORPUS_CONFIG)["corpus"]["data_root"])
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()
    print(f"corpus: {data_root}   contacts_{args.contacts}.npz")
    print("NOTE: the annotated scenes are the downstream learner's TEST split; the fit "
          "below uses all of them,\n      so the held-out number to read is the "
          "cross-validation, not the in-sample fit.")

    table = build_table(data_root, args.contacts)
    heads = {name: fit_head(design(table, mask), table["correct"][mask].astype(float))
             for name, mask in (("contact", table["auto"] == 1), ("free", table["auto"] == 0))}
    report(table, heads)

    block = toml_block(heads)
    print("\n" + block)
    if args.write:
        for path in args.config:
            write_config(path, block)
    else:
        print("(--write not given; no config was changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
