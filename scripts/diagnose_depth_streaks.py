#!/usr/bin/env python3
"""Round-9 T0: VGGT depth-streak meter + contact go/no-go slice.

Measures, per frame and per coarse body part, how stretched the human depth cloud
is along the camera ray versus the part's physical thickness, and flags "streaked"
part-frames. On labeled clips it then splits annotated joint-frames into streaked
vs clean and reports the contact false-positive / false-negative rates in each —
the round's go/no-go number (does bad VGGT depth actually drive the contact errors
this round targets?).

No optimization change. The init body (SAM-3D-Body converter params, independent of
VGGT) only defines the parts and their nominal along-ray depth; the cloud is loaded
*exactly* as the optimizer loads it (:func:`_person_depth_points`, same per-frame
conf gate). Part assignment is done in **image space** (nearest projected init joint)
because a streak shares its part's pixel footprint even while it runs a metre down
the ray — a 3-D assignment would misfile it.

Outputs, per ``--predictions`` clip dir:
  - ``geometry/diagnostics/depth_streaks.csv`` — one row per (frame, person, part).
  - ``geometry/diagnostics/depth_streaks_hist.png`` — ray-extent histogram.
  - a pooled summary table (printed; also appended to ``--summary-out`` if given).
  - on labeled clips (``data/<stem>.npz`` present): the streaked-vs-clean contact
    FP/FN slice.
"""

from __future__ import annotations

import argparse
import csv
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
import sys                        # noqa: E402

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent / "better_human" / "src"))

from tools.human_optim.camera import (   # noqa: E402
    fold_root_to_world, project, transform_points, world_from_cam,
)
from tools.human_optim.qmath import normalize_q   # noqa: E402
from tools.human_optim.stages import (   # noqa: E402
    _ANN_TO_SMPLX, _fill_invalid_frames, _person_depth_points,
)
from tools.smplx_robot import load_smplx_mid   # noqa: E402

_DEFAULT_MODEL = _REPO.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
_CONFIG = _REPO / "configs" / "human_optimization.toml"

# Nominal along-ray thickness (m) of each annotated coarse part. The streak
# threshold is max(mult x thickness, floor); with floor 0.30 m the thin
# extremities (hands/feet/elbows) all threshold at the floor, so the floor is the
# real knob — calibrated so healthy clips flag only a few % of part-frames.
_PART_THICKNESS_M: dict[str, float] = {
    "left_hand": 0.05, "right_hand": 0.05,
    "left_foot": 0.08, "right_foot": 0.08,
    "left_knee": 0.12, "right_knee": 0.12,
    "left_hip": 0.20, "right_hip": 0.20,
    "left_elbow": 0.08, "right_elbow": 0.08,
}
_COARSE = list(_ANN_TO_SMPLX.keys())   # fixed order = the 10 annotated coarse joints


def _coarse_maps(mid_names: list[str]) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Return ``joint_to_coarse (52,)`` (coarse index or -1) and ``coarse -> joint idxs``."""
    idx = {n: i for i, n in enumerate(mid_names)}
    j2c = np.full(len(mid_names), -1, dtype=np.int64)
    members: dict[int, list[int]] = {}
    for gi, gname in enumerate(_COARSE):
        js = [idx[j] for j in _ANN_TO_SMPLX[gname] if j in idx]
        members[gi] = js
        for j in js:
            j2c[j] = gi
    return j2c, members


@torch.no_grad()
def _init_joints_world(mid, sp: dict, p: int, wfc: torch.Tensor, device) -> torch.Tensor:
    """Pose the init (converter) body for person ``p`` → world joints ``(N, 52, 3)``.

    Mirrors ``run_kinematic``: mean betas over valid frames, invalid frames held from
    the nearest valid, camera-frame ``q`` folded to world. The 1e-6 tie-break jitter is
    omitted (diagnostic determinism); it is negligible vs the metrics here.
    """
    vmask = np.asarray(sp["valid_mask"][p], dtype=bool)
    N = int(sp["num_frames"])
    betas_mean = torch.as_tensor(sp["betas"][p][vmask].mean(0), dtype=torch.float32, device=device)
    betas_T = betas_mean[None].expand(N, -1)

    def t(k):
        return torch.as_tensor(_fill_invalid_frames(sp[k][p], vmask), dtype=torch.float32, device=device)

    q_cam = mid.from_classic(betas_T, t("body_pose"), t("transl"), t("global_orient"),
                             t("left_hand_pose"), t("right_hand_pose"))
    q_world = fold_root_to_world(q_cam, wfc)
    out = mid.forward(betas_T, normalize_q(q_world, mid.num_joints))
    return out.joints_world.tensor()[..., :3]


@torch.no_grad()
def meter_clip(pred: Path, mid, device: str, conf_drop_q: float, max_points: int,
               floor_m: float, mult: float, min_pts: int) -> list[dict]:
    """Compute per-(frame, person, part) streak rows for one clip dir."""
    sp = dict(np.load(pred / "sam3d" / "smplx_params.npz", allow_pickle=True))
    human = dict(np.load(pred / "geometry" / "human.npz", allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    N, P = int(sp["num_frames"]), int(sp["num_people"])
    object_ids = np.asarray(sp["object_ids"], dtype=np.int64)
    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)
    K = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)
    wfc = world_from_cam(extr)
    j2c, members = _coarse_maps(list(mid.joint_names))
    thr = {gi: max(mult * _PART_THICKNESS_M[g], floor_m) for gi, g in enumerate(_COARSE)}

    rows: list[dict] = []
    for p in range(P):
        oid = int(object_ids[p])
        vmask = np.asarray(sp["valid_mask"][p], dtype=bool)
        jw = _init_joints_world(mid, sp, p, wfc, device)                     # (N,52,3)
        j_cam = transform_points(jw, extr)                                   # (N,52,3)
        j_px = project(j_cam, K)                                             # (N,52,2)
        j_z = j_cam[..., 2]                                                  # (N,52)
        depth = _person_depth_points(human, oid, N, conf_drop_q, max_points, device)
        for tfr in range(N):
            if not vmask[tfr]:
                continue
            cloud = depth[tfr]
            if cloud.shape[0] < min_pts:
                continue
            c_cam = transform_points(cloud[None], extr[tfr:tfr + 1])[0]      # (M,3)
            c_z = c_cam[:, 2]
            front = c_z > 1e-3
            if int(front.sum()) < min_pts:
                continue
            c_px = project(c_cam[None], K[tfr:tfr + 1])[0]                   # (M,2)
            nn = torch.cdist(c_px[front], j_px[tfr]).argmin(1).cpu().numpy()  # (Mf,) nearest joint
            zf = c_z[front].cpu().numpy()
            part = j2c[nn]
            for gi, g in enumerate(_COARSE):
                sel = part == gi
                n = int(sel.sum())
                if n < min_pts:
                    continue
                z = zf[sel]
                extent = float(np.quantile(z, 0.95) - np.quantile(z, 0.05))
                # p99-p1 tail extent: chamfer (raw min-distance) is sensitive to a <5% far
                # tail that p95-p5 hides — logged so a variant sweep can watch it separately.
                tail = float(np.quantile(z, 0.99) - np.quantile(z, 0.01))
                part_z = float(j_z[tfr, members[gi]].mean().item())
                offset = float(np.median(z) - part_z)
                rows.append(dict(frame=tfr, person=oid, part=g, n_pts=n,
                                 ray_extent_m=extent, ray_extent_tail_m=tail,
                                 median_offset_m=offset, streaked=int(extent > thr[gi])))
    return rows


def _write_clip_outputs(pred: Path, rows: list[dict], floor_m: float) -> None:
    """Write the per-clip CSV + ray-extent histogram."""
    diag = pred / "geometry" / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    csv_path = diag / "depth_streaks.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "person", "part", "n_pts",
                                          "ray_extent_m", "ray_extent_tail_m",
                                          "median_offset_m", "streaked"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if rows:
        ext = np.array([r["ray_extent_m"] for r in rows])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(np.clip(ext, 0, 2.0), bins=60, color="#4477aa")
        ax.axvline(floor_m, color="crimson", ls="--", label=f"floor {floor_m:.2f} m")
        ax.set_xlabel("along-ray extent p95−p5 (m)")
        ax.set_ylabel("part-frames")
        ax.set_title(f"{pred.name}: depth-cloud along-ray extent")
        ax.legend()
        fig.tight_layout()
        fig.savefig(diag / "depth_streaks_hist.png", dpi=130)
        plt.close(fig)


def _slice_clip(pred: Path, rows: list[dict]) -> dict | None:
    """Contact FP/FN in streaked vs clean annotated joint-frames (labeled clips only)."""
    ann_path = _REPO / "data" / f"{pred.name}.npz"
    ct_path = pred / "human_optim" / "contacts_1.npz"
    if not ann_path.exists() or not ct_path.exists():
        return None
    ann = dict(np.load(ann_path, allow_pickle=True))
    ct = dict(np.load(ct_path, allow_pickle=True))
    ann_cts = ann["contacts"]                         # (Pa, 10, N) tri-state
    ann_oids = [int(x) for x in ann["object_ids"]]
    ann_names = [str(x) for x in ann["joint_names"]]
    ann_ignored = (np.asarray(ann["ignored"], dtype=bool) if "ignored" in ann
                   else np.zeros(len(ann_oids), dtype=bool))
    if ann_cts.shape[2] != ct["joint_contact"].shape[1]:
        raise ValueError(f"{pred.name}: annotation has {ann_cts.shape[2]} frames but the "
                         f"body has {ct['joint_contact'].shape[1]}")
    jc = ct["joint_contact"]                          # (P, N, 52) bool
    ct_oids = [int(x) for x in ct["object_ids"]]
    ct_names = list(ct["joint_names"])
    idx = {n: i for i, n in enumerate(ct_names)}
    coarse_joint_idx = {g: [idx[j] for j in _ANN_TO_SMPLX[g] if j in idx] for g in _COARSE}
    # streak flag lookup: (oid, part, frame) -> streaked bool
    flag = {(r["person"], r["part"], r["frame"]): bool(r["streaked"]) for r in rows}

    # bins: streaked / clean(cloud present) / no_cloud
    bins = {b: dict(n=0, fp=0, fn=0, pos=0, neg=0) for b in ("streaked", "clean", "no_cloud")}
    for p_ct, oid in enumerate(ct_oids):
        if oid not in ann_oids:
            continue
        pa = ann_oids.index(oid)
        if ann_ignored[pa]:                           # annotator marked this person unusable
            continue
        N = jc.shape[1]
        for gi, g in enumerate(_COARSE):
            if g not in ann_names:
                continue
            ga = ann_names.index(g)
            series = ann_cts[pa, ga]                  # (N,) tri-state
            js = coarse_joint_idx[g]
            for tfr in range(N):
                gt = int(series[tfr])
                if gt < 0:                            # unlabeled
                    continue
                pred_c = bool(jc[p_ct, tfr, js].any()) if js else False
                key = (oid, g, tfr)
                if key in flag:
                    b = "streaked" if flag[key] else "clean"
                else:
                    b = "no_cloud"
                s = bins[b]
                s["n"] += 1
                if gt == 1:
                    s["pos"] += 1
                    if not pred_c:
                        s["fn"] += 1
                else:
                    s["neg"] += 1
                    if pred_c:
                        s["fp"] += 1
    return bins


def _fmt_slice(name: str, bins: dict) -> list[str]:
    out = [f"  {name}:"]
    for b in ("streaked", "clean", "no_cloud"):
        s = bins[b]
        fpr = s["fp"] / s["neg"] if s["neg"] else float("nan")
        fnr = s["fn"] / s["pos"] if s["pos"] else float("nan")
        out.append(f"    {b:9s} n={s['n']:5d}  pos={s['pos']:5d} neg={s['neg']:5d}  "
                   f"FP={s['fp']:4d} (FPR={fpr:.3f})  FN={s['fn']:4d} (FNR={fnr:.3f})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", nargs="+", required=True, help="one or more clip dirs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--floor-m", type=float, default=0.30, help="streak threshold floor (m)")
    ap.add_argument("--thickness-mult", type=float, default=3.0)
    ap.add_argument("--min-pts", type=int, default=8, help="min cloud points for a part-frame to count")
    ap.add_argument("--summary-out", default=None, help="append the pooled summary + slice here")
    args = ap.parse_args()

    with _CONFIG.open("rb") as f:
        cham = tomllib.load(f)["chamfer"]
    conf_drop_q, max_points = float(cham["conf_drop_q"]), int(cham["max_points"])

    device = args.device if torch.cuda.is_available() else "cpu"
    mid = load_smplx_mid(_DEFAULT_MODEL, device)

    lines: list[str] = []
    pooled: dict[str, list] = {g: [] for g in _COARSE}
    print(f"{'clip':16s} {'part-frames':>11s} {'streaked':>9s} {'%':>6s} "
          f"{'med extent':>11s} {'med offset':>11s}")
    for pd_str in args.predictions:
        pred = Path(pd_str)
        rows = meter_clip(pred, mid, device, conf_drop_q, max_points,
                          args.floor_m, args.thickness_mult, args.min_pts)
        _write_clip_outputs(pred, rows, args.floor_m)
        n = len(rows)
        nstr = sum(r["streaked"] for r in rows)
        ext = np.array([r["ray_extent_m"] for r in rows]) if rows else np.array([0.0])
        off = np.array([r["median_offset_m"] for r in rows]) if rows else np.array([0.0])
        pct = 100.0 * nstr / n if n else 0.0
        line = (f"{pred.name:16s} {n:11d} {nstr:9d} {pct:5.1f}% "
                f"{np.median(ext):10.3f}m {np.median(off):10.3f}m")
        print(line)
        lines.append(line)
        for r in rows:
            pooled[r["part"]].append((r["ray_extent_m"], r["streaked"]))
        sl = _slice_clip(pred, rows)
        if sl is not None:
            for x in _fmt_slice(pred.name, sl):
                print(x)
                lines.append(x)

    print("\n=== per-part streak rate (pooled over clips) ===")
    lines.append("\n=== per-part streak rate (pooled over clips) ===")
    for g in _COARSE:
        vals = pooled[g]
        if not vals:
            continue
        arr = np.array(vals)
        rate = 100.0 * arr[:, 1].mean()
        row = f"  {g:12s} n={len(vals):5d}  streaked {rate:5.1f}%  med extent {np.median(arr[:,0]):.3f} m"
        print(row)
        lines.append(row)

    if args.summary_out:
        Path(args.summary_out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
