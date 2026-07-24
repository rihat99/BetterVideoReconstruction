"""Selective-risk evaluation of the per-joint contact-label confidence.

The question the confidence answers: *drop every label below a threshold ``c`` —
how good is what remains?*  This diagnostic sweeps ``c`` and reports, per class:

* kept CONTACT labels — precision against the manual annotation (want → 1.0);
* kept NO-CONTACT labels — error rate, i.e. how often the annotation actually
  says contact (want → 0.0);

alongside coverage (fraction of labels kept) and mean confidence, so a method
cannot buy honesty by marking everything uncertain.  Frames are correlated, so
every quantity is computed per scene and then macro-averaged over scenes.

Evaluation is at the annotated-part level.  Four limb parts are scored — the two
hands (wrist + fingers) and the two feet (foot + ankle) — the only joints where
scene contact is genuinely uncertain; the rest are trivially no-contact.  A part
is predicted in contact if ANY member joint is (OR-fold, matching the exporter
and dynamics).  The part's confidence is the label-carrying joint's confidence:
the MAX over touching members for a contact, the MIN over all members for a
no-contact (a no-contact part is an AND across its members).  GT folds the same
way — the foot part is the annotation foot ∨ ankle rows.

Three confidence modes read the SAME contacts_1.npz artifacts (so the base terms
are identical):

* ``v5``   — the earlier algebra ``visibility × where(label, surface, motion)``,
  reconstructed from the saved raw terms (identical to what the pre-v6 stage
  wrote);
* ``v6``   — the stage-written ``label_confidence`` (source-conditional-v1);
* ``v6off``— v6 recomputed offline with overridden caps, for cap tuning only.

Usage::

    uv run python scripts/diagnostics/confidence_risk.py --set tune --modes v5,v6
    uv run python scripts/diagnostics/confidence_risk.py --set test --modes v6
    uv run python scripts/diagnostics/confidence_risk.py --set tune --modes v6off \\
        --caps gate_only_cap=0.4,force_only_cap=0.6 --loso
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import sqlite3  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.human_optim.confidence import (  # noqa: E402
    CONTACT_CONFIDENCE_SCHEMA, confidence_config, source_conditional_evidence)

_DB_ROOT = _REPO_ROOT.parent / "data" / "ClimbingVideos"
_CURVES_DIR = _REPO_ROOT / "scratch" / "confidence_v6" / "curves"

# GT defect (see scripts/diagnostics/missing_contacts.py): competition_2 frames 68-178
# are the annotation tool's default "no contact" while the climber is mid-wall. Masked -1.
_GT_DEFECTS: dict[str, tuple[int, int]] = {"competition_2": (68, 179)}

# The 12 tuning scenes (data/<stem>.npz, out/<stem>/ trees) and the source video each
# belongs to, for leave-one-source-out checks.
_TUNE_SCENES = [
    "2nwG1Qa5k_c_0115", "33ODIivxhF8_0025", "3DAgLpPXVCs_0132", "9RvrpYoWDoo_0001",
    "boulder_1", "boulder_3", "competition_1", "competition_2", "CRquJ8H1QLI_0001",
    "gtIGgs_y2nY_0024", "PfwFWSoPF_k_0026", "vcRBT9Q9jXM_0029",
]
# Two test farms overlap the tuning set (see heldout_completion._TUNED).
_TEST_TUNED = {"33ODIivxhF8_0025", "CRquJ8H1QLI_0001"}

_MIN_DEPTH_POINTS = 5   # matches [depth_fit].min_points; a joint below this has no depth witness
_FINGERS = ("index", "middle", "pinky", "ring", "thumb")
_THRESH = np.linspace(0.0, 1.0, 201)


def _source_video(scene: str) -> str:
    """Group scenes by their source clip for leave-one-source-out (e.g. ``boulder``)."""
    if scene.startswith("boulder"):
        return "boulder"
    if scene.startswith("competition"):
        return "competition"
    return scene.rsplit("_", 1)[0]


def _limb_members(names: list[str]) -> dict[str, list[int]]:
    """SMPL-X joint indices per scored limb part (hand = wrist+fingers, foot = foot+ankle)."""
    idx = {n: i for i, n in enumerate(names)}

    def fingers(side: str) -> list[int]:
        return [idx[f"{side}_{finger}{k}"] for finger in _FINGERS for k in (1, 2, 3)
                if f"{side}_{finger}{k}" in idx]

    parts = {
        "left_hand": [idx["left_wrist"]] + fingers("left"),
        "right_hand": [idx["right_wrist"]] + fingers("right"),
        "left_foot": [idx[n] for n in ("left_foot", "left_ankle") if n in idx],
        "right_foot": [idx[n] for n in ("right_foot", "right_ankle") if n in idx],
    }
    return {p: m for p, m in parts.items() if m}


def _gt_part(ann_person: np.ndarray, ann_names: list[str], part: str) -> np.ndarray:
    """Tri-state GT for one part. Foot = foot ∨ ankle rows (contact if either, unknown if all -1)."""
    ann_idx = {n: i for i, n in enumerate(ann_names)}
    rows = [ann_person[ann_idx[part]]]
    if part.endswith("_foot"):
        ankle = part.replace("_foot", "_ankle")
        if ankle in ann_idx:
            rows.append(ann_person[ann_idx[ankle]])
    r = np.stack(rows)
    return np.where((r == 1).any(0), 1, np.where((r == -1).all(0), -1, 0))


def _confidence_per_joint(cnt: dict, mode: str, caps: dict | None) -> np.ndarray:
    """Per-joint confidence ``(P, N, J)`` for one artifact under a confidence mode."""
    vis = cnt["joint_visibility"].astype(np.float32)
    surf = cnt["surface_evidence"].astype(np.float32)
    mot = cnt["motion_evidence"].astype(np.float32)
    lab = cnt["joint_contact"].astype(bool)
    if mode == "v5":
        return vis * np.where(lab, surf, mot)
    if mode == "v6":
        return cnt["label_confidence"].astype(np.float32)
    if mode == "v6off":
        c = confidence_config(caps or {})
        depth_present = cnt["joint_depth_points"].astype(np.float32) >= float(_MIN_DEPTH_POINTS)
        ev = source_conditional_evidence(
            torch.as_tensor(lab), torch.as_tensor(surf), torch.as_tensor(mot),
            torch.as_tensor(cnt["contact_source"].astype(np.int64)),
            torch.as_tensor(cnt["force_share"].astype(np.float32)),
            torch.as_tensor(depth_present), c).numpy()
        return vis * ev
    raise ValueError(f"unknown mode {mode!r}")


def _scene_records(cnt_path: Path, ann_path: Path, scene: str, mode: str,
                   caps: dict | None) -> dict | None:
    """Per-(part, frame) records for one scene: folded confidence, prediction, GT."""
    cnt = dict(np.load(cnt_path, allow_pickle=True))
    schema = int(cnt["confidence_schema"]) if "confidence_schema" in cnt else -1
    if schema != CONTACT_CONFIDENCE_SCHEMA:
        raise ValueError(
            f"{scene}: confidence_schema {schema} != {CONTACT_CONFIDENCE_SCHEMA}; "
            f"rerun estimate_contacts to refresh {cnt_path}")
    ann = dict(np.load(ann_path, allow_pickle=True))
    names = [str(x) for x in cnt["joint_names"]]
    members = _limb_members(names)
    conf_all = _confidence_per_joint(cnt, mode, caps)         # (P, N, J)
    lab_all = cnt["joint_contact"].astype(bool)
    valid_all = cnt["valid_mask"].astype(bool)
    cnt_oids = [int(x) for x in np.asarray(cnt["object_ids"]).ravel()]

    ann_names = [str(x) for x in ann["joint_names"]]
    ann_c = np.asarray(ann["contacts"])                       # (Pa, Jann, N)
    ann_oids = [int(x) for x in np.asarray(ann["object_ids"]).ravel()]
    ann_ignored = (np.asarray(ann["ignored"]).ravel().astype(bool)
                   if "ignored" in ann else np.zeros(len(ann_oids), bool))
    lo_hi = _GT_DEFECTS.get(scene)

    conf, pred, gt = [], [], []
    for p, oid in enumerate(cnt_oids):
        if oid not in ann_oids:
            continue
        a = ann_oids.index(oid)
        if ann_ignored[a]:
            continue
        ncnt, nann = valid_all.shape[1], ann_c.shape[2]
        if ncnt != nann:
            raise ValueError(
                f"{scene}: frame-count mismatch — contacts {ncnt} vs annotation {nann}")
        n = ncnt
        valid = valid_all[p, :n]
        for part, mcols in members.items():
            g = _gt_part(ann_c[a], ann_names, part)[:n].copy()
            if lo_hi is not None:
                g[lo_hi[0]:lo_hi[1]] = -1
            member_contact = lab_all[p, :n][:, mcols]         # (n, |M|)
            p_contact = member_contact.any(axis=1)
            member_conf = conf_all[p, :n][:, mcols]
            touch = np.where(member_contact, member_conf, -np.inf).max(axis=1)
            free = member_conf.min(axis=1)
            part_conf = np.where(p_contact, touch, free)
            keep = valid & ((g == 0) | (g == 1))
            conf.append(part_conf[keep])
            pred.append(p_contact[keep])
            gt.append((g == 1)[keep])
    if not conf or sum(len(x) for x in conf) == 0:
        return None
    conf = np.concatenate(conf)
    pred = np.concatenate(pred).astype(bool)
    gt = np.concatenate(gt).astype(bool)
    correct = pred == gt
    return dict(scene=scene, conf=conf, pred=pred, gt=gt,
                mean_conf=float(conf.mean()),
                mean_conf_correct=float(conf[correct].mean()) if correct.any() else np.nan)


def _scene_curve(rec: dict) -> dict:
    """Threshold-swept curves for one scene."""
    conf, pred, gt = rec["conf"], rec["pred"], rec["gt"]
    n = len(conf)
    npred, nnc = int(pred.sum()), int((~pred).sum())
    cov, prec, err, ccov, ncov = ([] for _ in range(5))
    for c in _THRESH:
        keep = conf >= c
        cov.append(keep.mean())
        kc = keep & pred
        prec.append((kc & gt).sum() / kc.sum() if kc.sum() else np.nan)
        knc = keep & ~pred
        err.append((knc & gt).sum() / knc.sum() if knc.sum() else np.nan)
        ccov.append(kc.sum() / npred if npred else np.nan)
        ncov.append(knc.sum() / nnc if nnc else np.nan)
    return dict(cov=np.array(cov), prec=np.array(prec), err=np.array(err),
                ccov=np.array(ccov), ncov=np.array(ncov), n=n, npred=npred, nnc=nnc,
                mean_conf=rec["mean_conf"], mean_conf_correct=rec["mean_conf_correct"])


def _macro(curves: list[dict]) -> dict:
    """Macro-average scene curves (nan-safe — a scene with no kept labels drops out)."""
    stack = lambda k: np.array([s[k] for s in curves])
    with np.errstate(invalid="ignore"):
        out = {k: np.nanmean(stack(k), axis=0) for k in ("cov", "prec", "err", "ccov", "ncov")}
    out["mean_conf"] = float(np.nanmean([s["mean_conf"] for s in curves]))
    out["mean_conf_correct"] = float(np.nanmean([s["mean_conf_correct"] for s in curves]))
    out["n_scenes"] = len(curves)
    return out


def _at_coverage(m: dict, target: float) -> dict:
    """Precision/error at a target overall coverage, linearly interpolated.

    Coverage decreases monotonically with the threshold, so a target coverage
    maps to one threshold. Reporting at matched coverage (not matched threshold)
    is the fair comparison — a method that merely rescales confidence downward
    keeps fewer labels at the same threshold, which looks better but is not.
    """
    cov = m["cov"]
    order = np.argsort(cov)
    prec = float(np.interp(target, cov[order], m["prec"][order]))
    err = float(np.interp(target, cov[order], m["err"][order]))
    if target > cov.max() or target < cov.min():
        prec = err = float("nan")
    return dict(thresh=float("nan"), cov=target, prec=prec, err=err)


def _c_at_99(m: dict) -> dict:
    ok = m["prec"] >= 0.99
    if not ok.any():
        return dict(cov=float("nan"), thresh=float("nan"), prec=float(np.nanmax(m["prec"])))
    cov_ok = np.where(ok, m["cov"], -1.0)
    i = int(np.nanargmax(cov_ok))
    return dict(cov=float(m["cov"][i]), thresh=float(_THRESH[i]), prec=float(m["prec"][i]))


def _test_scenes() -> list[str]:
    """Scene ids in the ClimbingVideos DB test split (mirrors heldout_completion.test_scenes)."""
    with sqlite3.connect(_DB_ROOT / "scenes" / "scenes.db") as db:
        return [r[0] for r in db.execute(
            "SELECT scene_id FROM scenes WHERE dataset_split = 'test' ORDER BY scene_id")]


def _scene_paths(which: str) -> list[tuple[str, Path, Path]]:
    """(scene, contacts_1.npz, annotation.npz) for the tuning or test set."""
    out = []
    if which == "tune":
        for s in _TUNE_SCENES:
            cnt = _REPO_ROOT / "out" / s / "human_optim" / "contacts_1.npz"
            ann = _REPO_ROOT / "data" / f"{s}.npz"
            if cnt.exists() and ann.exists():
                out.append((s, cnt, ann))
    else:
        farms_root = _REPO_ROOT / "scratch" / "merge_eval" / "farms"
        for s in _test_scenes():                 # DB test split ∩ built eval farms
            cnt = farms_root / s / "human_optim" / "contacts_1.npz"
            ann = _DB_ROOT / "features" / "annotation" / s[:2] / s[2:4] / s / "annotation.npz"
            if cnt.exists() and ann.exists():
                out.append((s, cnt, ann))
    return out


_COV_POINTS = (0.90, 0.80, 0.70, 0.50, 0.40, 0.30)


def _summary_line(tag: str, m: dict) -> None:
    pts = {t: _at_coverage(m, t) for t in _COV_POINTS}
    c99 = _c_at_99(m)
    print(f"  [{tag}] scenes={m['n_scenes']}  mean_conf={m['mean_conf']:.3f} "
          f"(on correct labels {m['mean_conf_correct']:.3f})")
    hdr = "  ".join(f"cov{int(t*100)}" for t in _COV_POINTS)
    print(f"        {'kept-CONTACT precision':28s} " + hdr)
    print(f"        {'':28s} " + "  ".join(f"{pts[t]['prec']:5.3f}" for t in _COV_POINTS))
    print(f"        {'kept-NOCONTACT error':28s} " + hdr)
    print(f"        {'':28s} " + "  ".join(f"{pts[t]['err']:5.3f}" for t in _COV_POINTS))
    print(f"        C@99 (max coverage with kept-contact precision>=0.99): "
          f"{c99['cov']:.3f} at thresh {c99['thresh']:.3f} "
          f"(peak precision {c99['prec']:.3f})")


def _run_mode(paths: list[tuple[str, Path, Path]], mode: str, caps: dict | None,
              exclude: set[str]) -> tuple[dict, list[dict]]:
    curves, used = [], []
    for scene, cnt, ann in paths:
        if scene in exclude:
            continue
        rec = _scene_records(cnt, ann, scene, mode, caps)
        if rec is None:
            print(f"  [skip] {scene}: no annotated person / no scored frames")
            continue
        cur = _scene_curve(rec)
        cur["gt"] = rec["gt"]
        curves.append(cur)
        used.append(scene)
    m = _macro(curves)
    m["scenes"] = used
    return m, curves


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", choices=["tune", "test"], required=True)
    ap.add_argument("--modes", default="v5,v6", help="comma list of v5,v6,v6off")
    ap.add_argument("--caps", default="", help="v6off overrides key=val,key=val")
    ap.add_argument("--exclude-tuned", action="store_true",
                    help="test set only: drop the 2 farms that overlap the tuning set")
    ap.add_argument("--loso", action="store_true",
                    help="also print leave-one-source-out C@99 spread (robustness of the caps)")
    ap.add_argument("--save", default="", help="save curve arrays to scratch/confidence_v6/curves/<name>.npz")
    args = ap.parse_args(argv)

    caps = {}
    for item in (x for x in args.caps.split(",") if x.strip()):
        k, v = item.split("=")
        caps[k.strip()] = float(v)
    exclude = _TEST_TUNED if (args.set == "test" and args.exclude_tuned) else set()
    paths = _scene_paths(args.set)
    print(f"== confidence risk :: set={args.set}  scenes={len(paths)}  "
          f"{'(excl tuning overlap)' if exclude else ''}")

    _CURVES_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}
    for mode in (m.strip() for m in args.modes.split(",") if m.strip()):
        m, curves = _run_mode(paths, mode, caps if mode == "v6off" else None, exclude)
        _summary_line(mode, m)
        for k in ("cov", "prec", "err", "ccov", "ncov"):
            saved[f"{mode}_{k}"] = m[k]
        saved[f"{mode}_mean_conf"] = np.float32(m["mean_conf"])
        if args.loso and args.set == "tune":
            groups = sorted({_source_video(s) for s in m["scenes"]})
            spread = []
            for g in groups:
                sub = [c for c, s in zip(curves, m["scenes"]) if _source_video(s) != g]
                if sub:
                    spread.append(_c_at_99(_macro(sub))["cov"])
            spread = np.array([x for x in spread if np.isfinite(x)])
            if spread.size:
                print(f"        LOSO C@99 spread over {len(groups)} sources: "
                      f"min={spread.min():.3f} median={np.median(spread):.3f} max={spread.max():.3f}")
    saved["thresh"] = _THRESH
    if args.save:
        np.savez(_CURVES_DIR / f"{args.save}.npz", **saved)
        print(f"  saved curves -> {_CURVES_DIR / (args.save + '.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
