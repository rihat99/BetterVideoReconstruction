"""Held-out contact-completion evaluation on the ClimbingVideos test scenes.

The 12 local scenes tuned every knob (v*, d*, theta); this script measures the
chosen configuration on the DB's ``dataset_split = 'test'`` scenes, whose
manual annotations the tuning never saw. Two of the 30 test scenes also live in
the local tuning set — the macro is reported with and without them.

Per scene it builds a read-only symlink farm (``human_optim`` + ``geometry``
from the DB feature tree), runs the completion solve, turns forces into labels
with the standard readout, and scores auto-only vs completed labels against the
DB annotation. Writes forces to ``scratch/missing_contacts/heldout/`` and
reuses them on rerun; the DB is never written.

Usage::

    uv run python scripts/diagnostics/heldout_completion.py --sigma 2
    uv run python scripts/diagnostics/heldout_completion.py --scenes s-ArwEzr-2M_0017
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

import numpy as np  # noqa: E402

from completion_solve import _load_cones, solve_completion  # noqa: E402
from label_readout import _GRAVITY, limb_force, prf, readout  # noqa: E402
from missing_contacts import PersonDynamics, evaluate_scene, fold_limbs, load_scene  # noqa: E402
from tools.human_optim.config import load_config  # noqa: E402

_DB_ROOT = _REPO_ROOT.parent / "data" / "ClimbingVideos"
_OUT_DIR = _REPO_ROOT / "scratch" / "missing_contacts" / "heldout"
_TUNED = {"33ODIivxhF8_0025", "CRquJ8H1QLI_0001"}   # also in the local tuning set


def test_scenes() -> list[str]:
    with sqlite3.connect(_DB_ROOT / "scenes" / "scenes.db") as db:
        rows = db.execute(
            "SELECT scene_id FROM scenes WHERE dataset_split = 'test' ORDER BY scene_id")
        return [r[0] for r in rows]


def build_farm(sid: str) -> Path:
    farm = _OUT_DIR / "farms" / sid
    farm.mkdir(parents=True, exist_ok=True)
    for stage in ("human_optim", "geometry"):
        link = farm / stage
        target = _DB_ROOT / "features" / stage / sid[:2] / sid[2:4] / sid
        if not link.is_symlink():
            link.symlink_to(target)
    return farm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", default=None, help="comma-separated scene ids (default: all test)")
    ap.add_argument("--vstar", type=float, default=0.2)
    ap.add_argument("--dstar", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--raw-vel", action="store_true",
                    help="work penalty uses unsmoothed limb velocity")
    ap.add_argument("--gravity", default="fallback", choices=("fallback", "solve"))
    ap.add_argument("--thetas", default="0.005,0.01,0.02")
    ap.add_argument("--config", type=Path,
                    default=_REPO_ROOT / "configs" / "human_optimization.toml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    sids = ([s.strip() for s in args.scenes.split(",") if s.strip()] if args.scenes
            else test_scenes())
    thetas = [float(t) for t in args.thetas.split(",")]
    tag = (f"v{args.vstar:g}_d{args.dstar:g}_mag"
           + (f"_s{args.sigma:g}" if args.sigma > 0 else "")
           + ("_rv" if args.raw_vel else "")
           + (f"_g{args.gravity}" if args.gravity != "fallback" else ""))
    print(f"{len(sids)} test scenes, config {tag}")

    from tools.human_optim.stages import _DEFAULT_SMPLX_MODEL
    from tools.smplx_robot import load_smplx_mid
    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, args.device)
    (_OUT_DIR / "completion").mkdir(parents=True, exist_ok=True)

    per_scene = []          # (sid, base_prf, {theta: prf}, auroc, ap)
    for sid in sids:
        farm = build_farm(sid)
        ann_path = _DB_ROOT / "features" / "annotation" / sid[:2] / sid[2:4] / sid / "annotation.npz"
        try:
            scene = load_scene(sid, cfg, args.device, pred=farm, ann_path=ann_path)
            if not scene["people"]:
                continue
            f_path = _OUT_DIR / "completion" / f"{sid}__{tag}.npz"
            if f_path.exists():
                forces = np.load(f_path)["forces"]
            else:
                _load_cones(scene, cfg, args.device)
                forces = []
                for person in scene["people"]:
                    pd = PersonDynamics(mid, person["betas"], person["q"], person["valid"],
                                        scene["gravity"], scene["fps"], args.device,
                                        smooth_sigma=args.sigma)
                    res = solve_completion(scene, person, pd, cfg, args.device,
                                           vstar=args.vstar, dstar=args.dstar,
                                           vel_sigma=0.0 if args.raw_vel else args.sigma,
                                           solve_gravity=args.gravity == "solve")
                    person["total_mass"] = float(
                        pd.model.values.body_inertias[..., 1:, 0].sum())
                    forces.append(res["forces"])
                forces = np.stack(forces)
                np.savez(f_path, forces=forces,
                         total_mass=np.array([p.get("total_mass", 0.0)
                                              for p in scene["people"]], np.float32))
            saved = np.load(f_path)
            masses = np.asarray(saved["total_mass"], np.float32)
            weight_n = masses * _GRAVITY

            fmag = limb_force(forces)
            limb_auto = np.stack([fold_limbs(p["group_auto"].astype(float),
                                             scene["groups"]) > 0
                                  for p in scene["people"]])
            gt = np.stack([p["gt"] for p in scene["people"]])
            valid = np.stack([p["valid"] for p in scene["people"]])
            scored = valid[..., None] & (gt >= 0)
            gt_on = gt == 1
            base = prf(limb_auto[scored], gt_on[scored])
            by_theta = {}
            for t in thetas:
                completed = limb_auto | readout(fmag, weight_n, t)
                by_theta[t] = prf(completed[scored], gt_on[scored])
            row = evaluate_scene(scene, {"completion": [
                fold_limbs(np.linalg.norm(forces[p], axis=-1), scene["groups"])
                for p in range(len(scene["people"]))]})
            per_scene.append((sid, base, by_theta) + row["completion"])
            t0 = thetas[0]
            print(f"{sid:20s}{'~' if sid in _TUNED else ' '} auto P/R {base[0]:.3f}/{base[1]:.3f}"
                  f"  completed@{t0:g} P/R/F1 "
                  f"{by_theta[t0][0]:.3f}/{by_theta[t0][1]:.3f}/{by_theta[t0][2]:.3f}"
                  f"  AUROC/AP {row['completion'][0]:.3f}/{row['completion'][1]:.3f}",
                  flush=True)
        except Exception as e:  # noqa: BLE001 - one bad scene must not kill the sweep
            print(f"{sid:20s}  FAILED: {e}", flush=True)

    for label, keep in [("all test scenes", per_scene),
                        ("held-out only (tuning overlap removed)",
                         [r for r in per_scene if r[0] not in _TUNED])]:
        if not keep:
            continue
        print(f"\n== macro over {len(keep)} {label}")
        base = np.mean([r[1] for r in keep], 0)
        print(f"  auto-only     P {base[0]:.3f} R {base[1]:.3f} F1 {base[2]:.3f}")
        for t in thetas:
            m = np.mean([r[2][t] for r in keep], 0)
            print(f"  completed@{t:<5g} P {m[0]:.3f} R {m[1]:.3f} F1 {m[2]:.3f}")
        au = np.nanmean([r[3] for r in keep])
        ap_ = np.nanmean([r[4] for r in keep])
        print(f"  completion score AUROC {au:.3f} AP {ap_:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
