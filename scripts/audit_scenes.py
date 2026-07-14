#!/usr/bin/env python3
"""Corpus floater audit: one health row per (scene, person), flagging bodies that
float off the wall, sit behind it, or rest on a degenerate scene cloud.

Reads only artifacts already on disk in the post-T1 DB tree — the contact readout
(``contacts_1.npz``), the trust readout (``uncertainty_1.npz``) and the body
(``kinematic_1.npz``) beside it under ``features/human_optim/<scene>/``, plus the
fused ``geometry/scene.npz``. For every tracked person with at least
``--min-valid`` valid frames it computes:

- ``contact_rate``   — share of valid frames with ≥1 of 52 joints in contact.
- ``med_min_jsdf_cm``— median over frames of the closest joint's SIGNED distance to
  the wall (``joint_sdf_cm``, unguarded; negative = inside/behind the wall).
- ``p_seated``       — share of frames whose closest joint sits in [−5, +3] cm.
- ``nan_jsdf_share`` — share of frames with no usable wall distance (degenerate cloud).
- ``med_chamfer_cm`` / ``p90_chamfer_cm`` — body-to-own-cloud fit error (uncertainty).
- ``mean_obs``       — mean per-frame observability (uncertainty ``obs_score``).
- ``scene_pts``      — fused scene-cloud point count (a floor-level upstream health cue).
- ``valid_share``    — tracking coverage (valid frames / total).

Flags (thresholds tunable below): ``floater`` = contact_rate < 0.5 OR
med_min_jsdf > +5 cm; ``buried`` = med_min_jsdf < −15 cm; ``degenerate`` =
nan_jsdf_share > 0.5 OR scene_pts < 1000. ``buried`` is a diagnostic, not a clean
failure flag — most buried bodies are in contact almost every frame (the closest
joint's SDF sign flips where the nearest wall-point's normal faces away), so T3's
cloud-vs-wall probe, not this, decides whether such a body is genuinely behind the
wall. Writes a CSV, prints the flagged rows worst-first + flag counts, and
(``--hist``) histogram pngs.

    uv run python scripts/audit_scenes.py --out /tmp/audit/bodies.csv --hist
    uv run python scripts/audit_scenes.py --scenes PfwFWSoPF_k_0026,9RvrpYoWDoo_0000
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
import scene_db  # noqa: E402

# Seed flag thresholds (validated against the five named floaters + a visual sample).
FLOATER_CONTACT_RATE = 0.50    # below this share of frames-with-contact ⇒ suspect
                               # (0.50 flags all five named floaters, incl. the mild
                               #  PfwFWSoPF_k_0026 at 0.459; 0.40 would miss it)
FLOATER_HOVER_CM = 5.0         # closest joint typically this far OUTSIDE the wall ⇒ suspect
BURIED_CM = -15.0             # closest joint typically this far INSIDE/behind ⇒ suspect
DEGEN_NAN_SHARE = 0.50        # this share of frames with no wall distance ⇒ degenerate
DEGEN_SCENE_PTS = 1000        # fewer scene-cloud points than this ⇒ near-empty cloud

# The user's five named floating bodies (person index in parens where it matters).
NAMED = ["PfwFWSoPF_k_0026", "2nwG1Qa5k_c_0115", "9RvrpYoWDoo_0000",
         "CRquJ8H1QLI_0001", "3DAgLpPXVCs_0132"]

_FIELDS = ["scene_id", "oid", "pidx", "n_valid", "valid_share", "contact_rate",
           "med_min_jsdf_cm", "p_seated", "nan_jsdf_share", "med_chamfer_cm",
           "p90_chamfer_cm", "mean_obs", "scene_pts", "floater", "buried", "degenerate"]


def _finite_percentiles(values, percentiles=(5, 25, 50, 75, 95)) -> np.ndarray:
    """Percentiles over finite values, or all NaNs when no value is usable."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.full(len(percentiles), np.nan, dtype=float)
    return np.percentile(finite, percentiles)


def _person_rows(paths: scene_db.ScenePaths, sid: str, min_valid: int) -> list[dict]:
    """Health metrics for every person in one scene (empty if inputs are missing)."""
    ho = paths.feature_dir("human_optim", sid)
    kin_p, cnt_p = ho / "kinematic_1.npz", ho / "contacts_1.npz"
    if not kin_p.is_file() or not cnt_p.is_file():
        return []
    kin = np.load(kin_p, allow_pickle=True)
    cnt = np.load(cnt_p, allow_pickle=True)
    unc_p = ho / "uncertainty_1.npz"
    unc = np.load(unc_p, allow_pickle=True) if unc_p.is_file() else None

    scene_pts = 0
    scene_np = paths.feature_dir("geometry", sid) / "scene.npz"
    if scene_np.is_file():
        scene_pts = int(np.load(scene_np)["points"].shape[0])

    oids = kin["object_ids"]
    valid = np.asarray(kin["valid_mask"], bool)          # (P, N)
    jc = np.asarray(cnt["joint_contact"], bool)          # (P, N, J)
    jsdf = np.asarray(cnt["joint_sdf_cm"], np.float32)   # (P, N, J) signed, unguarded
    rows = []
    for p in range(valid.shape[0]):
        vm = valid[p]
        n_valid = int(vm.sum())
        if n_valid < min_valid:
            continue
        # Closest joint per frame (signed cm); a frame with no usable distance is NaN.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fmin = np.nanmin(jsdf[p], axis=1)             # (N,)
        fmin_v = fmin[vm]
        contact_rate = float(jc[p][vm].any(axis=1).mean())
        nan_share = float(np.isnan(fmin_v).mean())
        med_min = float(np.nanmedian(fmin_v)) if not np.isnan(fmin_v).all() else float("nan")
        seated = (fmin_v >= -5.0) & (fmin_v <= 3.0)       # NaN → False (not seated)
        p_seated = float(seated.mean())

        med_ch = p90_ch = mean_obs = float("nan")
        if unc is not None:
            uv = np.asarray(unc["valid"], bool)[p]
            if uv.any():
                ch = np.asarray(unc["chamfer_cm"], np.float32)[p][uv]
                if np.isfinite(ch).any():
                    med_ch = float(np.nanmedian(ch))
                    p90_ch = float(np.nanpercentile(ch, 90))
                mean_obs = float(np.nanmean(np.asarray(unc["obs_score"], np.float32)[p][uv]))

        floater = (contact_rate < FLOATER_CONTACT_RATE) or (med_min > FLOATER_HOVER_CM)
        buried = med_min < BURIED_CM
        degenerate = (nan_share > DEGEN_NAN_SHARE) or (scene_pts < DEGEN_SCENE_PTS)
        rows.append({
            "scene_id": sid, "oid": int(oids[p]), "pidx": p, "n_valid": n_valid,
            "valid_share": round(float(vm.mean()), 3), "contact_rate": round(contact_rate, 3),
            "med_min_jsdf_cm": round(med_min, 2), "p_seated": round(p_seated, 3),
            "nan_jsdf_share": round(nan_share, 3),
            "med_chamfer_cm": round(med_ch, 2), "p90_chamfer_cm": round(p90_ch, 2),
            "mean_obs": round(mean_obs, 3), "scene_pts": scene_pts,
            "floater": bool(floater), "buried": bool(buried), "degenerate": bool(degenerate),
        })
    return rows


def _fmt(rows: list[dict], keys: list[str]) -> str:
    """A fixed-width console table of selected columns."""
    cols = ["scene_id", "pidx", "contact_rate", "med_min_jsdf_cm", "p_seated",
            "med_chamfer_cm", "mean_obs", "scene_pts", "flags"]
    out = ["  ".join(f"{c:>15s}" if c != "scene_id" else f"{c:<22s}" for c in cols)]
    for r in rows:
        fl = ",".join(f for f in ("floater", "buried", "degenerate") if r[f]) or "-"
        vals = [f"{r['scene_id']:<22s}", f"{r['pidx']:>15d}", f"{r['contact_rate']:>15.3f}",
                f"{r['med_min_jsdf_cm']:>15.2f}", f"{r['p_seated']:>15.3f}",
                f"{r['med_chamfer_cm']:>15.2f}", f"{r['mean_obs']:>15.3f}",
                f"{r['scene_pts']:>15d}", f"{fl:>15s}"]
        out.append("  ".join(vals))
    return "\n".join(out)


def _histograms(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [("contact_rate", "contact rate", (0, 1)),
             ("med_min_jsdf_cm", "closest-joint dist to wall (cm)", (-40, 40)),
             ("med_chamfer_cm", "body-to-cloud chamfer (cm)", (0, 40))]
    for key, label, rng in specs:
        vals = np.array([r[key] for r in rows], float)
        vals = vals[np.isfinite(vals)]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(np.clip(vals, *rng), bins=40, color="#4477aa")
        ax.set_xlabel(label); ax.set_ylabel("# bodies"); ax.set_title(f"{key} (n={len(vals)})")
        fig.tight_layout(); fig.savefig(out_dir / f"hist_{key}.png", dpi=110); plt.close(fig)
    print(f"[audit] histograms → {out_dir}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, default=scene_db.DEFAULT_DATA_ROOT)
    p.add_argument("--out", type=Path, default=None, help="CSV output path (default: print only).")
    p.add_argument("--min-valid", type=int, default=5, help="skip persons with fewer valid frames.")
    p.add_argument("--categories", default=",".join(map(str, scene_db.DEFAULT_CATEGORIES)))
    p.add_argument("--rope-supported", type=int, default=scene_db.DEFAULT_ROPE_SUPPORTED)
    p.add_argument("--scenes", default=None, help="restrict to these scene ids (comma-separated).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top", type=int, default=50, help="how many flagged rows to print.")
    p.add_argument("--hist", action="store_true", help="write histogram pngs next to --out.")
    a = p.parse_args(argv)

    paths = scene_db.ScenePaths(a.data_root)
    rope = None if a.rope_supported < 0 else a.rope_supported
    with paths.connect() as conn:
        db_rows = scene_db.select_targets(
            conn, categories=scene_db.parse_int_csv(a.categories), rope_supported=rope)
    db_rows = scene_db.resolve_rows(db_rows, scenes=a.scenes, limit=a.limit)
    print(f"[audit] scanning {len(db_rows)} scene(s)…", flush=True)

    rows: list[dict] = []
    for r in db_rows:
        rows.extend(_person_rows(paths, r["scene_id"], a.min_valid))
    if not rows:
        print("[audit] no bodies found."); return 0

    n = len(rows)
    n_fl = sum(r["floater"] for r in rows)
    n_bu = sum(r["buried"] for r in rows)
    n_de = sum(r["degenerate"] for r in rows)
    n_any = sum(r["floater"] or r["buried"] or r["degenerate"] for r in rows)
    healthy = [r for r in rows if not (r["floater"] or r["buried"] or r["degenerate"])]

    print(f"\n[audit] {n} bodies over {len(db_rows)} scenes")
    print(f"  floater    {n_fl:4d}  ({n_fl/n:.0%})   [contact_rate<{FLOATER_CONTACT_RATE} or hover>{FLOATER_HOVER_CM}cm]")
    print(f"  buried     {n_bu:4d}  ({n_bu/n:.0%})   [med_min_jsdf<{BURIED_CM}cm]")
    print(f"  degenerate {n_de:4d}  ({n_de/n:.0%})   [nan_share>{DEGEN_NAN_SHARE} or scene_pts<{DEGEN_SCENE_PTS}]")
    print(f"  ANY flag   {n_any:4d}  ({n_any/n:.0%})   |   clean {len(healthy)} ({len(healthy)/n:.0%})")

    def pctl(key):
        return _finite_percentiles([r[key] for r in rows])
    for key in ("contact_rate", "med_min_jsdf_cm", "med_chamfer_cm"):
        print(f"  {key:16s} p[5,25,50,75,95] = {np.round(pctl(key), 2)}")

    flagged = sorted([r for r in rows if r["floater"] or r["buried"] or r["degenerate"]],
                     key=lambda r: (r["contact_rate"], -r["med_min_jsdf_cm"]))
    print(f"\n[audit] worst {min(a.top, len(flagged))} flagged bodies (contact_rate ascending):")
    print(_fmt(flagged[:a.top], _FIELDS))

    named_rows = [r for r in rows if r["scene_id"] in NAMED]
    print(f"\n[audit] the five named scenes ({sum(r['floater'] for r in named_rows)}/"
          f"{len(named_rows)} rows flag floater):")
    print(_fmt(sorted(named_rows, key=lambda r: r["contact_rate"]), _FIELDS))

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_FIELDS)
            w.writeheader()
            for r in sorted(rows, key=lambda r: r["contact_rate"]):
                w.writerow(r)
        print(f"\n[audit] wrote {n} rows → {a.out}")
        if a.hist:
            _histograms(rows, a.out.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
