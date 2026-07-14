#!/usr/bin/env python3
"""T3 autopsy: assign each flagged body (from ``audit_scenes``) one dominant cause.

Run ``audit_scenes.py --out features/audit/bodies.csv`` first; this reads that CSV
and adds two probes, then labels every flagged (scene, person) with the dominant
reason it floats / reads wrong:

- **d_broken**  — upstream garbage: near-empty scene cloud, no usable wall distance,
  or tracking dropout (``scene_pts<3000`` / ``nan_jsdf_share>0.3`` / ``valid_share<0.5``).
- **b_detached** — the human depth CLOUD is off the wall (VGGT depth layering):
  cloud→scene gap ``> 30 cm`` (healthy p95 ~19). Unfixable
  kinematically.
- **c_short** — the body's limbs are specially short (worst bone below the corpus
  5th percentile). Rare: the MHR→SMPL-X limb deficit is UNIFORM across the corpus
  (floaters no shorter than healthy), so short limbs do NOT distinguish floaters.
- **b_uncertain** — the severe-detachment and short-limb probes do not explain the
  flag; this may be a subtle cloud detachment below the ~15 cm human-exclusion floor.

Probe (b) cloud→scene gap is the p10 nearest-neighbour distance of per-frame human
points (``geometry/human.npz``) to the static scene cloud (``geometry/scene.npz``),
median over sampled frames; it is confounded by the human-exclusion dilation (~15 cm
floor) so it only cleanly resolves SEVERE detachment. Probe (c) is the median
MHR-vs-SMPL-X bone deficit (subagent-verified joint indices); it is cached and reused
by T5. Writes ``autopsy.csv`` + a per-cause count table + verdict; ``--refresh``
recomputes the cloud-gap cache.

    uv run python scripts/autopsy_scenes.py            # reads features/audit/bodies.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1]))
import scene_db  # noqa: E402

# MHR (pymomentum 127-joint rig) and SMPLXMid (52-joint) limb bone endpoints, verified.
_MHR = {"upperarm": ((75, 76), (39, 40)), "forearm": ((76, 78), (40, 42)),
        "thigh": ((2, 3), (18, 19)), "shin": ((3, 4), (19, 20))}
_SMPLX = {"upperarm": ((16, 18), (17, 19)), "forearm": ((18, 20), (19, 21)),
          "thigh": ((1, 4), (2, 5)), "shin": ((4, 7), (5, 8))}

# Cause thresholds.
BROKEN_SCENE_PTS = 3000
BROKEN_NAN_SHARE = 0.30
BROKEN_VALID_SHARE = 0.50
DETACHED_GAP_CM = 30.0


def _limb_deficit(paths, sid, mid, torch):
    """Per-person {bone: SMPLX_rest_cm − MHR_median_cm} + summaries (negative = shorter)."""
    sp = paths.feature_dir("sam3d", sid) / "params.npz"
    kp = paths.feature_dir("human_optim", sid) / "kinematic_1.npz"
    if not sp.is_file() or not kp.is_file():
        return {}
    s, k = np.load(sp), np.load(kp, allow_pickle=True)
    jc_all, vm_all = s["pred_joint_coords"], s["valid_mask"]
    out = {}
    for p in range(jc_all.shape[0]):
        jc = jc_all[p]
        use = vm_all[p] & np.isfinite(jc).all(axis=(1, 2))
        if use.sum() < 5:
            continue
        jc_v = jc[use]
        with torch.no_grad():
            _, jr = mid.forward_shape(torch.as_tensor(k["betas"][p], dtype=torch.float32)[None])
        jr = jr[0].numpy()
        defs = {}
        for bone in _MHR:
            for side in (0, 1):
                a, b = _MHR[bone][side]
                mhr = float(np.median(np.linalg.norm(jc_v[:, a] - jc_v[:, b], axis=-1)) * 100.0)
                sa, sb = _SMPLX[bone][side]
                sx = float(np.linalg.norm(jr[sa] - jr[sb]) * 100.0)
                defs[f"{'LR'[side]}_{bone}"] = sx - mhr
        out[p] = {**defs, "worst": min(defs.values()),
                  "arm_mean": float(np.mean([v for kk, v in defs.items() if "arm" in kk])),
                  "leg_mean": float(np.mean([v for kk, v in defs.items() if "thigh" in kk or "shin" in kk]))}
    return out


def _cloud_gap(paths, sid, want, ckdtree):
    """p10 NN distance (cm) of per-frame human points to the scene cloud, median over frames."""
    geom = paths.feature_dir("geometry", sid)
    try:
        kin = np.load(paths.feature_dir("human_optim", sid) / "kinematic_1.npz", allow_pickle=True)
        h, s = np.load(geom / "human.npz"), np.load(geom / "scene.npz")
    except Exception:
        return {p: float("nan") for p in want}
    pts_s = s["points"].astype(np.float64)
    if len(pts_s) < 50:
        return {p: float("nan") for p in want}
    tree = ckdtree(pts_s)
    hp, hoid, hfid = h["points"].astype(np.float64), h["obj_id"], h["frame_id"]
    res = {}
    for p in want:
        m = hoid == int(kin["object_ids"][p])
        if m.sum() < 100:
            res[p] = float("nan"); continue
        pts, fid = hp[m], hfid[m]
        fr = np.unique(fid)
        sel = fr[np.linspace(0, len(fr) - 1, min(12, len(fr))).astype(int)]
        g = [np.percentile(tree.query(pts[fid == f], k=1)[0], 10) * 100
             for f in sel if (fid == f).sum() >= 20]
        res[p] = float(np.median(g)) if g else float("nan")
    return res


def _cause(a, gap, dworst, worst_p05):
    if (float(a["scene_pts"]) < BROKEN_SCENE_PTS or float(a["nan_jsdf_share"]) > BROKEN_NAN_SHARE
            or float(a["valid_share"]) < BROKEN_VALID_SHARE):
        return "d_broken"
    if np.isfinite(gap) and gap > DETACHED_GAP_CM:
        return "b_detached"
    if np.isfinite(dworst) and dworst < worst_p05:
        return "c_short"
    return "b_uncertain"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, default=scene_db.DEFAULT_DATA_ROOT)
    p.add_argument("--audit-csv", type=Path, default=None,
                   help="default: <data-root>/features/audit/bodies.csv")
    p.add_argument("--refresh", action="store_true", help="recompute the cloud-gap cache.")
    a = p.parse_args(argv)

    import torch
    from scipy.spatial import cKDTree
    from tools.smplx_robot import load_smplx_mid

    paths = scene_db.ScenePaths(a.data_root)
    audit_csv = a.audit_csv or (a.data_root / "features" / "audit" / "bodies.csv")
    if not audit_csv.is_file():
        print(f"ERROR: {audit_csv} not found — run audit_scenes.py --out {audit_csv} first.", file=sys.stderr)
        return 1
    aud = {(r["scene_id"], r["pidx"]): r for r in csv.DictReader(open(audit_csv))}
    flagged = [k for k, r in aud.items()
               if r["floater"] == "True" or r["buried"] == "True" or r["degenerate"] == "True"]

    mid = load_smplx_mid(_HERE.parents[2] / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz", "cpu")

    # Probe (c): limb deficit for every scene that has a flagged body (cheap); worst-bone p05
    # is taken over ALL bodies so "specially short" means short vs the corpus.
    scenes = sorted({k[0] for k in aud})
    deficit = {}
    for sid in scenes:
        for pp, d in _limb_deficit(paths, sid, mid, torch).items():
            deficit[(sid, str(pp))] = d
    worst_p05 = float(np.percentile([d["worst"] for d in deficit.values()], 5)) if deficit else -1e9

    # Probe (b): cloud gap for flagged bodies, cached.
    cache = audit_csv.parent / "cloud_gap.csv"
    gap = {}
    if cache.is_file() and not a.refresh:
        gap = {(r["scene_id"], r["pidx"]): float(r["cloud_gap_cm"]) if r["cloud_gap_cm"] else float("nan")
               for r in csv.DictReader(open(cache))}
    if not gap or a.refresh:
        by_scene: dict = {}
        for sid, pp in flagged:
            by_scene.setdefault(sid, []).append(int(pp))
        gap = {}
        for i, (sid, want) in enumerate(sorted(by_scene.items())):
            for pp, g in _cloud_gap(paths, sid, want, cKDTree).items():
                gap[(sid, str(pp))] = g
            if (i + 1) % 25 == 0:
                print(f"  cloud_gap {i+1}/{len(by_scene)}", flush=True)
        with open(cache, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["scene_id", "pidx", "cloud_gap_cm"]); w.writeheader()
            for (sid, pp), g in gap.items():
                w.writerow({"scene_id": sid, "pidx": pp, "cloud_gap_cm": round(g, 2) if g == g else ""})

    # Assign + write.
    rows = []
    for k in flagged:
        a_row = aud[k]
        g = gap.get(k, float("nan"))
        d = deficit.get(k, {})
        rows.append({
            "scene_id": k[0], "pidx": k[1], "oid": a_row["oid"],
            "flags": ",".join(f for f in ("floater", "buried", "degenerate") if a_row[f] == "True"),
            "cause": _cause(a_row, g, d.get("worst", float("nan")), worst_p05),
            "contact_rate": a_row["contact_rate"], "hover_cm": a_row["med_min_jsdf_cm"],
            "cloud_gap_cm": round(g, 2) if g == g else "",
            "def_worst_cm": round(d["worst"], 2) if d else "", "scene_pts": a_row["scene_pts"],
        })
    out_csv = audit_csv.parent / "autopsy.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    floaters = [r for r in rows if "floater" in r["flags"]]
    fc = Counter(r["cause"] for r in floaters)
    print(f"\n[autopsy] {len(rows)} flagged bodies → {out_csv}")
    print(f"[autopsy] FLOATER causes (n={len(floaters)}):")
    for c, n in fc.most_common():
        print(f"    {c:14s} {n:3d}  ({n/len(floaters):.0%})")
    b_family = fc["b_detached"] + fc["b_uncertain"]
    print(f"    → (b) cloud-off-wall family = {b_family} ({b_family/len(floaters):.0%}); "
          f"limbs (c) = {fc['c_short']} ({fc['c_short']/len(floaters):.0%})")
    buried_only = [r for r in rows if "buried" in r["flags"] and "floater" not in r["flags"]]
    bg = np.array([float(r["cloud_gap_cm"]) for r in buried_only if r["cloud_gap_cm"] != ""])
    print(f"[autopsy] buried-only (n={len(buried_only)}): cloud_gap median "
          f"{np.median(bg):.1f}cm, #detached(>30)={int((bg>30).sum())} → SDF artifact, not behind wall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
