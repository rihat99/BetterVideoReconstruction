"""Per-clip reconstruction-quality table — find the scenes where VGGT failed.

One row per ``out/<stem>/`` tree — raw metrics plus a ``flag`` column carrying the
calibrated verdicts (``[quality]`` in ``configs/pipeline.toml``; user-labelled
operating point, 2026-08-26). The metric columns:

* **cam/depth** — camera translation spread over the clip divided by the median
  scene-to-camera distance. A hallucinated dolly (VGGT turning a tripod pan+zoom
  into metres of travel, e.g. high_jump) reads ≫ 1; a static camera ~0.
* **smear med / >2.5** — the per-frame human-cloud along-ray smear
  (:mod:`tools.geometry.depth_smear`): median and the fraction of frames above
  2.5. The pommel_horse failure (person streaked along the view ray) reads
  median ~8; every clean climbing clip reads median < 1, fraction 0.
* **scale mad** — the scale stage's per-frame log-scale spread (``scale.npz``);
  large = the frames never agreed on one world size.
* **far lbl** — fraction of contact labels >10 cm from the scene
  (``contacts_1.npz``): how much of the contact story floats in mid-air.
* **res b/a** — median root residual force before/after the contact-force
  witness, in body weights (~1 BW before is normal support; ≫1 before AND after
  = the kinematics are garbage no force can explain).
* **f-conf** — whole-body force confidence of the kindyn_1 solve
  (:mod:`tools.human_optim.force_confidence`), mean over valid frames. Computed
  from the saved wrench + forces, so it also works on kindyn files written
  before the field existed.

Run::

    uv run python scripts/diagnostics/scene_quality.py --predictions out_2/ out/
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from tools.geometry.quality import cam_motion_ratio, quality_flag, smear_stats  # noqa: E402
from tools.human_optim.force_confidence import force_confidence  # noqa: E402

_GRAVITY_MAG = 9.81


def clip_row(pred: Path) -> dict | None:
    geo = pred / "geometry"
    if not (geo / "transform.npz").exists() or not (geo / "scene.npz").exists():
        return None
    tr = dict(np.load(geo / "transform.npz", allow_pickle=True))
    scene = dict(np.load(geo / "scene.npz", allow_pickle=True))
    row: dict = {"clip": pred.name,
                 "cam_ratio": cam_motion_ratio(tr["extrinsics"], scene["points"])}
    if (geo / "human.npz").exists():
        human = dict(np.load(geo / "human.npz", allow_pickle=True))
        row["smear_med"], row["smear_frac"] = smear_stats(
            human["points"], human["frame_id"], tr["extrinsics"])
    else:
        row["smear_med"], row["smear_frac"] = float("nan"), float("nan")
    row["scale_mad"] = float("nan")
    if (geo / "scale.npz").exists():
        with np.load(geo / "scale.npz") as sc:
            if "scale_mad" in sc.files:
                row["scale_mad"] = float(sc["scale_mad"])

    row.update(far_lbl=float("nan"), res_before=float("nan"), res_after=float("nan"))
    cpath = pred / "human_optim" / "contacts_1.npz"
    if cpath.exists():
        c = dict(np.load(cpath, allow_pickle=True))
        contact, valid = c["frame_contact"], np.asarray(c["valid_mask"], bool)
        dist = np.nan_to_num(c["frame_distance_cm"], nan=1e9)
        labels = int(contact[valid].sum())
        row["far_lbl"] = float((contact & (dist > 10.0))[valid].sum() / labels) if labels else 0.0
        bw = np.asarray(c["body_weight_n"], np.float64).reshape(-1, 1)
        row["res_before"] = float(np.median((c["residual_force_before_n"] / bw)[valid]))
        row["res_after"] = float(np.median((c["residual_force_after_n"] / bw)[valid]))

    row["f_conf"] = float("nan")
    kpath = pred / "human_optim" / "kindyn_1.npz"
    if kpath.exists():
        k = dict(np.load(kpath, allow_pickle=True))
        if "force_confidence_clip" in k:
            row["f_conf"] = float(np.nanmean(k["force_confidence_clip"]))
        elif {"base_wrench", "frame_forces", "total_mass", "valid_mask"} <= set(k):
            clips = [force_confidence(k["base_wrench"][p], k["frame_forces"][p],
                                      np.asarray(k["valid_mask"][p], bool),
                                      float(k["total_mass"][p]) * _GRAVITY_MAG)["clip"]
                     for p in range(int(k["num_people"]))]
            row["f_conf"] = float(np.nanmean(clips))
    return row


def flag_of(row: dict, q: dict) -> str:
    """The calibrated verdicts, recomputed at the CURRENT [quality] thresholds via the
    same :func:`tools.geometry.quality.quality_flag` the contacts stage saves — the
    table and the pipeline cannot drift in logic, only in threshold snapshots."""
    return quality_flag(row["cam_ratio"], row["res_before"], row["smear_med"],
                        cam_depth_max=float(q["cam_depth_max"]),
                        res_before_bw_max=float(q["res_before_bw_max"]),
                        smear_med_max=float(q["smear_med_max"]))


def main(argv: list[str] | None = None) -> int:
    from tools.config import load_toml

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", nargs="+", required=True,
                    help="out/<stem>/ trees, or roots containing them")
    ap.add_argument("--config", type=Path,
                    default=_REPO_ROOT / "configs" / "human_optimization.toml",
                    help="TOML carrying the [quality] flag thresholds")
    args = ap.parse_args(argv)
    q = (load_toml(args.config) if args.config.exists() else {}).get("quality", {})
    q.setdefault("cam_depth_max", 0.4)
    q.setdefault("res_before_bw_max", 2.5)
    q.setdefault("smear_med_max", 2.0)

    targets: list[Path] = []
    for raw in args.predictions:
        path = Path(raw)
        if (path / "geometry").is_dir():
            targets.append(path)
        elif path.is_dir():
            targets.extend(sorted(d for d in path.iterdir() if (d / "geometry").is_dir()))
    if not targets:
        raise SystemExit("[scene_quality] no out/<stem>/ trees found")

    print(f"{'clip':<28s} {'cam/depth':>9s} {'smear med':>9s} {'>2.5':>6s} {'scale mad':>9s} "
          f"{'far lbl':>7s} {'res b/a (BW)':>13s} {'f-conf':>6s} {'flag':>12s}")
    for pred in targets:
        row = clip_row(pred)
        if row is None:
            continue
        print(f"{row['clip']:<28.28s} {row['cam_ratio']:>9.2f} {row['smear_med']:>9.2f} "
              f"{row['smear_frac']:>6.1%} {row['scale_mad']:>9.3f} {row['far_lbl']:>7.1%} "
              f"{row['res_before']:>6.2f}/{row['res_after']:<6.2f} {row['f_conf']:>6.2f} "
              f"{flag_of(row, q):>12s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
