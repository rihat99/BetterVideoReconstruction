"""Held-out head-to-head: force readout vs the kinematic baselines, source-aware.

Scores the frozen tuning-set configurations on the ClimbingVideos test scenes:
drift-only (< 0.12 m over the stillness window), dist AND drift (50 cm / 0.12),
and the completion-force readout (theta x body weight) — final labels =
auto ∪ new, 3-frame min-span, macro P/R/F1. Because 20 of the 30 test scenes
come from one source video (which also has a sibling clip in the tuning set),
results are also aggregated per SOURCE VIDEO and over the source-disjoint
subset (sources never seen during tuning). Reads the farms + forces the
held-out run produced; writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "diagnostics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from heldout_completion import _DB_ROOT, _OUT_DIR, _TUNED, test_scenes  # noqa: E402
from label_readout import _GRAVITY, limb_force, prf, readout  # noqa: E402
from missing_contacts import fold_limbs, load_scene  # noqa: E402
from ablation_labelers import _min_span  # noqa: E402
from tools.human_optim.config import load_config  # noqa: E402

_TAG = "v0.2_d0.05_mag_s4_rv"
# Source videos with any clip in the 12-scene tuning set.
_TUNED_SOURCES = {"2nwG1Qa5k_c", "33ODIivxhF8", "3DAgLpPXVCs", "9RvrpYoWDoo",
                  "CRquJ8H1QLI", "PfwFWSoPF_k", "gtIGgs_y2nY", "vcRBT9Q9jXM"}


def source_of(sid: str) -> str:
    return sid.rsplit("_", 1)[0]


def main() -> int:
    cfg = load_config(_REPO_ROOT / "configs" / "human_optimization.toml")
    rows = []
    for sid in test_scenes():
        farm = _OUT_DIR / "farms" / sid
        ann = _DB_ROOT / "features" / "annotation" / sid[:2] / sid[2:4] / sid / "annotation.npz"
        f_path = _OUT_DIR / "completion" / f"{sid}__{_TAG}.npz"
        if not f_path.exists():
            print(f"{sid}: no forces, skipped")
            continue
        scene = load_scene(sid, cfg, "cpu", pred=farm, ann_path=ann)
        saved = np.load(f_path)
        weight_n = np.asarray(saved["total_mass"], np.float32) * _GRAVITY
        fmag = limb_force(saved["forces"])
        auto = np.stack([fold_limbs(p["group_auto"].astype(float), scene["groups"]) > 0
                         for p in scene["people"]])
        dist = np.stack([fold_limbs(-p["group_dist"], scene["groups"])
                         for p in scene["people"]])          # negative cm
        drift = np.stack([fold_limbs(-p["group_drift"], scene["groups"])
                          for p in scene["people"]])         # negative m
        gt = np.stack([p["gt"] for p in scene["people"]])
        valid = np.stack([p["valid"] for p in scene["people"]])
        scored = valid[..., None] & (gt >= 0)
        gt_on = gt == 1
        labelers = {
            "auto": auto,
            "drift<0.12": auto | _min_span(-drift < 0.12),
            "d&v 50/0.12": auto | _min_span((-dist < 50) & (-drift < 0.12)),
            "force@0.0075": auto | readout(fmag, weight_n, 0.0075),
            "force@0.01": auto | readout(fmag, weight_n, 0.01),
        }
        rows.append((sid, {k: prf(v[scored], gt_on[scored]) for k, v in labelers.items()}))

    names = list(rows[0][1])

    def macro(sel_rows) -> dict:
        return {k: tuple(np.mean([r[1][k][i] for r in sel_rows]) for i in range(3))
                for k in names}

    def show(title: str, sel_rows) -> None:
        print(f"\n== {title} ({len(sel_rows)} scenes)")
        m = macro(sel_rows)
        for k in names:
            hit = sum(1 for r in sel_rows if r[1][k][0] > 0.9 and r[1][k][1] > 0.8)
            print(f"  {k:14s} P {m[k][0]:.3f} R {m[k][1]:.3f} F1 {m[k][2]:.3f}"
                  f"   scenes meeting P>0.9&R>0.8: {hit}/{len(sel_rows)}")

    show("all test scenes, scene-macro", rows)
    show("tuning-overlap scenes removed", [r for r in rows if r[0] not in _TUNED])

    by_src: dict[str, list] = {}
    for r in rows:
        by_src.setdefault(source_of(r[0]), []).append(r)
    print("\n== per-source macros (scene count)")
    for src, srows in sorted(by_src.items()):
        m = macro(srows)
        seen = "(tuned source)" if src in _TUNED_SOURCES else ""
        print(f"  {src:14s} ({len(srows):2d}) " + "  ".join(
            f"{k} {m[k][0]:.2f}/{m[k][1]:.2f}" for k in names if k != "auto") + f"  {seen}")
    src_macros = [macro(srows) for srows in by_src.values()]
    print("\n== source-video macro (each source counts once)")
    for k in names:
        p = np.mean([m[k][0] for m in src_macros])
        r = np.mean([m[k][1] for m in src_macros])
        print(f"  {k:14s} P {p:.3f} R {r:.3f}")

    show("source-disjoint scenes (source never in tuning)",
         [r for r in rows if source_of(r[0]) not in _TUNED_SOURCES])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
