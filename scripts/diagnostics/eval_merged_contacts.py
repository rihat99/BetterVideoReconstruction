"""Held-out evaluation of the MERGED estimate_contacts stage (completion in-stage).

The contact-completion diagnostic was merged into ``scripts/stages/estimate_contacts.py``.
This script checks that the merged stage reproduces the diagnostic's validated
held-out numbers (P 0.950 / R 0.910) on the 30 annotated ClimbingVideos test
scenes. It scores the labels the STAGE WROTE — not a standalone solve.

Per scene it builds a fresh eval farm where ``human_optim/`` is a REAL directory
(so the stage can write ``contacts_1.npz`` into it) holding a symlink to the DB's
``kinematic_1.npz`` input only, with ``geometry/`` symlinked read-only. The DB is
never written. It runs the stage (overlay off, via a per-run TOML) and scores the
completed labels against the manual annotation.

Three views are reported:
  1. **4-limb (parity target)** — hand = wrist∨fingers, foot = foot∨ankle; GT
     rows folded the same. This is where P 0.950 / R 0.910 lives.
  2. **6-group** — each contact group against its own GT row, plus the foot↔ankle
     confusion and how often a foot AND its ankle both carry a completion label.
  3. **gate-fold variant (report only)** — the coherent per-member gate
     ``OR_member(dist < τd AND drift < τv)`` vs the stage's default min-over-member
     gate, at 4-limb. The stage keeps min-min unless this matches/beats it.

Usage::

    CUDA_VISIBLE_DEVICES=3 uv run --no-sync python scripts/diagnostics/eval_merged_contacts.py
    ... --scenes s-ArwEzr-2M_0017          # one scene
    ... --score-only                       # skip the stage runs, score existing farms
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "diagnostics")):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import sqlite3  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402

from heldout_completion import _TUNED  # noqa: E402
from label_readout import prf  # noqa: E402
from missing_contacts import _GT_DEFECTS, fold_limbs, load_scene  # noqa: E402
from tools.human_optim.completion import _min_span  # noqa: E402
from tools.human_optim.config import load_config  # noqa: E402
from tools.human_optim.dynamics import _contact_catalog  # noqa: E402

_DB_ROOT = _REPO_ROOT.parent / "data" / "ClimbingVideos"
_EVAL_DIR = _REPO_ROOT / "scratch" / "merge_eval"
_STAGE = _REPO_ROOT / "scripts" / "stages" / "estimate_contacts.py"
_GROUPS = ["left_hand", "right_hand", "left_foot", "right_foot", "left_ankle", "right_ankle"]
# 4-limb fold: the group columns that make up each limb.
_LIMB_GROUPS = {"left_hand": ["left_hand"], "right_hand": ["right_hand"],
                "left_foot": ["left_foot", "left_ankle"], "right_foot": ["right_foot", "right_ankle"]}


def test_scenes() -> list[str]:
    with sqlite3.connect(_DB_ROOT / "scenes" / "scenes.db") as db:
        return [r[0] for r in db.execute(
            "SELECT scene_id FROM scenes WHERE dataset_split = 'test' ORDER BY scene_id")]


def perrun_config() -> Path:
    """A copy of the human-optimization TOML with the contact overlay off (no video in farms)."""
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    dst = _EVAL_DIR / "human_optimization_nooverlay.toml"
    text = (_REPO_ROOT / "configs" / "human_optimization.toml").read_text()
    dst.write_text(text.replace("overlay = true", "overlay = false"))   # the [contacts] overlay is the only `true`
    return dst


def build_farm(sid: str) -> Path:
    """Fresh farm: real ``human_optim/`` (kinematic_1 symlinked in), ``geometry/`` symlinked."""
    farm = _EVAL_DIR / "farms" / sid
    ho = farm / "human_optim"
    ho.mkdir(parents=True, exist_ok=True)
    db_ho = _DB_ROOT / "features" / "human_optim" / sid[:2] / sid[2:4] / sid
    link = ho / "kinematic_1.npz"
    if not link.is_symlink():
        link.symlink_to(db_ho / "kinematic_1.npz")
    geo = farm / "geometry"
    if not geo.is_symlink():
        geo.symlink_to(_DB_ROOT / "features" / "geometry" / sid[:2] / sid[2:4] / sid)
    geocalib_db = _DB_ROOT / "features" / "geocalib" / sid[:2] / sid[2:4] / sid
    gc = farm / "geocalib"                            # optional gravity input — absent is fine
    if geocalib_db.is_dir() and not gc.is_symlink():
        gc.symlink_to(geocalib_db)
    return farm


def run_stage(farm: Path, config: Path) -> None:
    subprocess.run(
        ["uv", "run", "--no-sync", "python", str(_STAGE),
         "--source", "none", "--predictions", str(farm), "--config", str(config)],
        cwd=_REPO_ROOT, check=True)


def _require_completed_labels(cnt_path: Path, sid: str) -> None:
    """Fail on a reused contacts_1.npz that predates the in-stage completion schema."""
    with np.load(cnt_path, allow_pickle=True) as cnt:
        has_schema = "contact_label_schema" in cnt.files
        method = str(cnt["contact_label_method"]) if "contact_label_method" in cnt.files else ""
    if not has_schema or method != "completed-static-support-v1":
        raise SystemExit(
            f"{sid}: stale contacts_1.npz (label method {method!r}); rerun with --force")


def _align(sid: str, farm: Path):
    """Load the stage-written per-joint signals aligned to ``load_scene``'s people order."""
    cnt = dict(np.load(farm / "human_optim" / "contacts_1.npz", allow_pickle=True))
    kin = dict(np.load(farm / "human_optim" / "kinematic_1.npz", allow_pickle=True))
    ann_path = _DB_ROOT / "features" / "annotation" / sid[:2] / sid[2:4] / sid / "annotation.npz"
    ann = dict(np.load(ann_path, allow_pickle=True))
    jn = [str(n) for n in cnt["joint_names"]]
    catalog = _contact_catalog(jn)
    members = {g: [jn.index(m) for m in catalog[g][1]] for g in _GROUPS}
    rep = {g: jn.index(catalog[g][0]) for g in _GROUPS}
    cnt_oids = list(np.asarray(cnt["object_ids"], np.int32))
    ann_oids = list(np.asarray(ann["object_ids"], np.int32))
    ann_names = [str(n) for n in ann["joint_names"]]
    return cnt, kin, ann, jn, members, rep, cnt_oids, ann_oids, ann_names


def _gt_group(ann, ann_names, a, group: str, N: int) -> np.ndarray:
    """Tri-state GT row (N,) for one contact group (each group has its own annotation row)."""
    return np.asarray(ann["contacts"][a, ann_names.index(group)], np.int8)[:N]


def score_scene(sid: str, farm: Path, device: str) -> dict | None:
    """P/R at 4-limb (parity + per-member variant) and 6-group, for one scene."""
    cfg = load_config(_REPO_ROOT / "configs" / "human_optimization.toml")
    ann_path = _DB_ROOT / "features" / "annotation" / sid[:2] / sid[2:4] / sid / "annotation.npz"
    scene = load_scene(sid, cfg, device, pred=farm, ann_path=ann_path)     # gt (N,4), valid, group_auto
    if not scene["people"]:
        return None
    cnt, kin, ann, jn, members, rep, cnt_oids, ann_oids, ann_names = _align(sid, farm)
    comp = cfg.knobs["contacts"]["completion"]
    N = int(kin["num_frames"])

    lim4, lim4_pm, grp6 = [], [], []            # accumulators (person-stacked) per view
    both_labeled = fa_conf = 0                  # 6-group foot↔ankle diagnostics (pooled)
    src_all = np.asarray(cnt["contact_source"], np.int8)
    dist_all = np.asarray(cnt["joint_distance_cm"], np.float32)
    drift_all = np.asarray(kin["drift_pre"], np.float32)
    lf_all = np.asarray(cnt["limb_force_n"], np.float32)

    for person in scene["people"]:
        oid = person["oid"]
        cp = cnt_oids.index(oid)
        a = ann_oids.index(oid)
        valid = person["valid"]                                            # (N,)
        # --- 4-limb parity: completed labels folded to 4 limbs ---
        pred4 = fold_limbs(person["group_auto"].astype(float), scene["groups"]) > 0   # (N,4)
        gt4 = person["gt"]                                                  # (N,4) defect-masked
        scored4 = valid[:, None] & (gt4 >= 0)
        lim4.append((pred4[scored4], (gt4 == 1)[scored4]))

        # --- per-member gate variant (delta D): keep detector + force from the written
        #     source bitmask, rebuild only the gate as OR over members of a coherent
        #     per-member (dist<τd AND drift<τv) test, min-span with invalid frames cleared ---
        src = src_all[cp]
        det = (src & 1) > 0                                                 # detector (all joints)
        force_bit = (src & 4) > 0                                          # force component (group joints)
        gate_pm = np.zeros((N, len(_GROUPS)), bool)
        for c, g in enumerate(_GROUPS):
            dm = dist_all[cp][:, members[g]]
            vm = drift_all[cp][:, members[g]]
            hit = (dm < comp["gate_dist_cm"]) & (vm < comp["gate_drift_m"])  # (N, |members|)
            gate_pm[:, c] = np.nan_to_num(hit, nan=False).any(1)
        gate_pm &= valid[:, None]
        gate_pm = _min_span(gate_pm, int(comp["min_span"]))
        pm_joint = det.copy()
        for c, g in enumerate(_GROUPS):
            pm_joint[:, rep[g]] |= gate_pm[:, c] | force_bit[:, rep[g]]
        grp_pm = np.stack([pm_joint[:, members[g]].any(1) for g in _GROUPS], 1)      # (N,6)
        pred4_pm = np.stack([grp_pm[:, [_GROUPS.index(x) for x in gs]].any(1)
                             for gs in _LIMB_GROUPS.values()], 1)
        lim4_pm.append((pred4_pm[scored4], (gt4 == 1)[scored4]))

        # --- 6-group: each group vs its own GT row ---
        pred6 = person["group_auto"]                                       # (N,6) completed, OR over members
        gt6 = np.stack([_gt_group(ann, ann_names, a, g, N) for g in _GROUPS], 1)      # (N,6) tri-state
        if sid in _GT_DEFECTS:
            lo, hi = _GT_DEFECTS[sid]
            gt6[lo:hi] = -1
        scored6 = valid[:, None] & (gt6 >= 0)
        grp6.append((pred6[scored6], (gt6 == 1)[scored6]))
        # foot↔ankle: a foot-side completion where GT credits only the sibling; both-labeled frames
        for foot, ankle in (("left_foot", "left_ankle"), ("right_foot", "right_ankle")):
            fi, ai = _GROUPS.index(foot), _GROUPS.index(ankle)
            v = valid
            both_labeled += int((pred6[:, fi] & pred6[:, ai] & v).sum())
            # confusion: predicted foot but GT says ankle-only (foot row 0, ankle row 1), or vice versa
            gf, ga = gt6[:, fi], gt6[:, ai]
            fa_conf += int(((pred6[:, fi] & (gf != 1) & (ga == 1))
                            | (pred6[:, ai] & (ga != 1) & (gf == 1))) [v].sum())

    def macro(pairs):
        pr = np.concatenate([p for p, _ in pairs]); gt = np.concatenate([g for _, g in pairs])
        return prf(pr, gt)

    return dict(sid=sid, prf4=macro(lim4), prf4_pm=macro(lim4_pm), prf6=macro(grp6),
                both_labeled=both_labeled, fa_conf=fa_conf,
                force_frac=float(((src_all & 4) > 0).sum()) )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", default=None, help="comma-separated scene ids (default: all test)")
    ap.add_argument("--score-only", action="store_true", help="skip stage runs; score existing farms")
    ap.add_argument("--force", action="store_true", help="re-run the stage even if contacts_1.npz exists")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    sids = ([s.strip() for s in args.scenes.split(",") if s.strip()] if args.scenes
            else test_scenes())
    config = perrun_config()
    rows = []
    for sid in sids:
        farm = build_farm(sid)
        try:
            contacts_npz = farm / "human_optim" / "contacts_1.npz"
            if not args.score_only and (args.force or not contacts_npz.exists()):
                run_stage(farm, config)
            elif contacts_npz.exists():                    # reusing → must carry completed labels
                _require_completed_labels(contacts_npz, sid)
            row = score_scene(sid, farm, args.device)
            if row is None:
                print(f"{sid:20s}  no people"); continue
            rows.append(row)
            p4, r4, f4 = row["prf4"]
            print(f"{sid:20s}{'~' if sid in _TUNED else ' '} 4limb P/R {p4:.3f}/{r4:.3f}"
                  f"  6grp P/R {row['prf6'][0]:.3f}/{row['prf6'][1]:.3f}"
                  f"  pm P/R {row['prf4_pm'][0]:.3f}/{row['prf4_pm'][1]:.3f}"
                  f"  both-fa {row['both_labeled']} conf {row['fa_conf']}", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad scene must not kill the sweep
            print(f"{sid:20s}  FAILED: {e}", flush=True)

    for label, keep in [("all test", rows), ("held-out (tuning overlap removed)",
                                             [r for r in rows if r["sid"] not in _TUNED])]:
        if not keep:
            continue
        print(f"\n== macro over {len(keep)} scenes — {label}")
        for name, key in [("4-limb (parity)", "prf4"), ("6-group", "prf6"),
                          ("4-limb per-member gate (variant)", "prf4_pm")]:
            m = np.mean([r[key] for r in keep], 0)
            print(f"  {name:34s} P {m[0]:.3f} R {m[1]:.3f} F1 {m[2]:.3f}")
        print(f"  foot↔ankle: both-labeled frames {sum(r['both_labeled'] for r in keep)}, "
              f"foot/ankle confusion frames {sum(r['fa_conf'] for r in keep)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
