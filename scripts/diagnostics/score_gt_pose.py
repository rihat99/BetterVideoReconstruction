"""Pose metrics of published models against the climb_wall_3 multi-view ground truth.

Unlike the ClimbingVideos table, the reference here is not our single-view output.
It is the three-camera fit in ``peter/climb_wall_3_gt`` — calibrated cameras, a
shape frozen from the participant's measured height and mass, and Sapiens
keypoints in all three views. Its world frame *is* the left camera's frame, in
metres, so a body already in that camera needs no transform at all.

Our own single-view run is therefore a scored row like any other, not the ruler.

Every body — the reference, the SAM-3D-Body initialisation, our run, each
published model — is re-posed with the same pip ``smplx`` neutral model, so joints
and vertices compare like with like. The models and our single-view run both read
the ORIGINAL ``cam_left.mp4``; the reference was fitted on the undistorted frames,
but the two cameras share the same intrinsics (undistortion kept K), so the 3D
frames coincide exactly. The only residue is lens distortion nudging the image
evidence, which moves a pixel on the climber's body by a median 0.1–1.0 px
(99th percentile under 2.9 px) — far below the errors in this table.

Metrics are the WHAM/EMDB set, identical to ``score_baseline_pose``: MPJPE,
PA-MPJPE and PVE per frame in the camera frame (pelvis-aligned), W-/WA-MPJPE over
100-frame world segments, RTE as a percentage of the reference root's path,
Accel, and Jitter.

Frames beyond the reference's own length (it drops the last two, where the lagging
cameras run out) are not scored.

Writes ``peter/climb_wall_3_gt/others/gt_pose_metrics.json`` and a PNG table.

Usage:
    uv run --no-sync python scripts/diagnostics/score_gt_pose.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_baseline_pose import (  # noqa: E402
    METRICS, N_BODY, _cell, camera_metrics, jitter, model_params, pose, to_camera, world_metrics,
)
from score_prediction_contacts import _render_table  # noqa: E402

GT_ROOT = _REPO / "peter" / "climb_wall_3_gt" / "human"
OURS_ROOT = _REPO / "peter" / "out_climb_wall_3_single"
OTHERS = _REPO / "peter" / "climb_wall_3_gt" / "others"
POSE_MODELS = {"gvhmr": "GVHMR", "prompthmr": "PromptHMR"}
#: Our own runs on this tree: row key -> (label, the run directory under ``<clip>/``).
#: They share every stage but the last, so the rows price one change each — measured
#: contacts instead of estimated ones, and a torque priced the same at every joint
#: instead of by the joint's strength.
OURS_RUNS = {"ours/kindyn_1": ("Ours (single view)", "human_optim"),
             "ours/gt_contact": ("Ours + measured contacts", "human_optim_gt_contact"),
             "ours/flat_torque": ("Ours, flat torque weights", "human_optim_flat_torque")}
BODY_KEYS = ("global_orient", "body_pose", "transl", "left_hand_pose", "right_hand_pose",
             "jaw_pose", "leye_pose", "reye_pose", "expression", "betas")


def clips() -> list[tuple[str, float]]:
    names = sorted(d.name for d in GT_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_"))
    return [(c, float(np.load(GT_ROOT / c / "cameras.npz", allow_pickle=True)["fps"])) for c in names]


def reference(clip: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """The multi-view body: (joints, verts, world2cam, valid, reposing check in mm).

    Its world is the left camera's frame, so ``world2cam`` is the identity — kept
    explicit so the camera-frame metrics read the same as everywhere else.
    """
    k = np.load(GT_ROOT / clip / "human_optim" / "kindyn_1.npz")
    j, v = pose({key: k[key][0] for key in BODY_KEYS})
    valid = np.asarray(k["valid_mask"][0], bool)
    check = np.linalg.norm(j - k["joints_world"][0, :, :N_BODY], axis=-1)[valid].mean() * 1e3
    return j, v, np.repeat(np.eye(4)[None], len(j), 0), valid, check


def _restrict(params: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    """Keep the selected frames; per-clip parameters (PromptHMR-post's betas) pass through."""
    return {k: (v[keep] if np.ndim(v) > 1 and len(v) == len(keep) else v) for k, v in params.items()}


def score_model(d, gj_w: np.ndarray, gv_w: np.ndarray, w2c: np.ndarray, valid: np.ndarray,
                fps: float) -> dict:
    idx = np.asarray(d["frame_indices"], int)
    keep = idx < len(gj_w)                      # the reference is two frames shorter than the video
    idx = idx[keep]
    params, model_type = model_params(d, "cam")
    ok = valid[idx] & np.isfinite(params["transl"][keep]).all(1)
    pj_c, pv_c = pose(_restrict(params, keep), model_type)
    out = camera_metrics(pj_c, to_camera(gj_w[idx], w2c[idx]), pv_c, to_camera(gv_w[idx], w2c[idx]), fps, ok)
    out["reposing_check_mm"] = (
        float(np.linalg.norm(pj_c - d["joints_cam"][keep, :N_BODY], axis=-1)[ok].mean() * 1e3)
        if "joints_cam" in d.files else None)
    params_w, _ = model_params(d, "world")
    pj_w, _ = pose(_restrict(params_w, keep), model_type)
    out.update(world_metrics(pj_w, gj_w[idx], ok))
    out["Jitter"] = jitter(pj_w, fps, ok)
    return out


def score_ours(clip: str, gj_w: np.ndarray, gv_w: np.ndarray, w2c: np.ndarray, valid: np.ndarray,
               fps: float, optim_dir: str = "human_optim") -> dict:
    """Our single-view body, carried into the left camera by its own extrinsics."""
    k = np.load(OURS_ROOT / clip / optim_dir / "kindyn_1.npz")
    tr = np.load(OURS_ROOT / clip / "geometry" / "transform.npz")
    assert bool(tr["metric"]), clip
    n = len(gj_w)
    pj_w, pv_w = pose({key: k[key][0][:n] for key in BODY_KEYS if k[key].ndim > 2} |
                      {"betas": k["betas"][0]})
    ours_w2c = tr["extrinsics"].astype(np.float64)[:n]
    ok = valid & np.asarray(k["valid_mask"][0], bool)[:n]
    out = camera_metrics(to_camera(pj_w, ours_w2c), to_camera(gj_w, w2c),
                         to_camera(pv_w, ours_w2c), to_camera(gv_w, w2c), fps, ok)
    out.update(world_metrics(pj_w, gj_w, ok))
    out["Jitter"] = jitter(pj_w, fps, ok)
    return out


def score_init(clip: str, gj_w: np.ndarray, gv_w: np.ndarray, w2c: np.ndarray, valid: np.ndarray,
               fps: float) -> dict:
    """The SAM-3D-Body → SMPL-X initialisation, camera frame only."""
    d = np.load(OURS_ROOT / clip / "sam3d" / "smplx_params.npz")
    n = len(gj_w)
    pj_c, pv_c = pose({key: np.nan_to_num(d[key][0][:n]) for key in BODY_KEYS})
    ok = valid & np.asarray(d["valid_mask"][0], bool)[:n]
    return camera_metrics(pj_c, to_camera(gj_w, w2c), pv_c, to_camera(gv_w, w2c), fps, ok)


def collect() -> dict:
    sums: dict[str, dict] = {}
    missing: dict[str, int] = {}
    ref_checks, ref_jitter = [], []

    def add(key: str, label: str, m: dict) -> None:
        s = sums.setdefault(key, {"label": label, "clips": 0, "acc": {}})
        s["clips"] += 1
        for k, v in m.items():
            if v is None:
                continue
            if k in ("RTE", "Jitter", "reposing_check_mm"):
                s["acc"].setdefault(k, []).append(v)
            else:
                s["acc"][k] = s["acc"].get(k, 0.0) + v

    for clip, fps in clips():
        gj_w, gv_w, w2c, valid, check = reference(clip)
        ref_checks.append(check)
        ref_jitter.append(jitter(gj_w, fps, valid))
        add("sam3d/init", "SAM 3D Body init (camera frame)", score_init(clip, gj_w, gv_w, w2c, valid, fps))
        for key, (label, optim_dir) in OURS_RUNS.items():
            if (OURS_ROOT / clip / optim_dir / "kindyn_1.npz").exists():
                add(key, label, score_ours(clip, gj_w, gv_w, w2c, valid, fps, optim_dir))
        for model, name in POSE_MODELS.items():
            for variant in ("raw", "post"):
                path = OTHERS / model / clip / f"pose_{variant}.npz"
                if not path.exists():
                    missing[f"{model}/{variant}"] = missing.get(f"{model}/{variant}", 0) + 1
                    continue
                with np.load(path, allow_pickle=True) as d:
                    add(f"{model}/{variant}", f"{name} ({variant})",
                        score_model(d, gj_w, gv_w, w2c, valid, fps))
        print(f"[pose] {clip} done", flush=True)

    rows = {}
    for key, s in sums.items():
        a = s["acc"]
        r = {"label": s["label"], "clips": s["clips"], "frames": int(a["frames"])}
        for k in ("MPJPE", "PA-MPJPE", "PVE"):
            r[k] = a[k] / a["frames"] if k in a else None
        r["Accel"] = a["Accel"] / a["accel_frames"]
        for k in ("WA-MPJPE", "W-MPJPE"):
            r[k] = a[k] / a["segments"] if k in a else None
        for k in ("RTE", "Jitter", "reposing_check_mm"):
            r[k] = float(np.mean(a[k])) if k in a else None
        rows[key] = r
    return {"reference": "peter/climb_wall_3_gt/human/<clip>/human_optim/kindyn_1.npz "
                         "(three-camera fit, frozen participant shape, left-camera frame)",
            "clips": len(clips()),
            "reference_reposing_check_mm": float(np.mean(ref_checks)),
            "reference_jitter": float(np.mean(ref_jitter)),
            "missing": missing, "rows": rows}


def write_table(m: dict, path: Path) -> None:
    cells = [[r["label"], *(_cell(r[k]) for k in METRICS)] for r in m["rows"].values()]
    cells.append(["Multi-view GT (reference)", *["—"] * 7, _cell(m["reference_jitter"])])
    columns = [("Model", 3.6, "l"), ("MPJPE", 1.0, "r"), ("PA-MPJPE", 1.2, "r"), ("PVE", 1.0, "r"),
               ("WA-MPJPE", 1.2, "r"), ("W-MPJPE", 1.1, "r"), ("RTE %", 1.0, "r"),
               ("Accel m/s²", 1.3, "r"), ("Jitter", 1.0, "r")]
    bold = set()
    for col in range(1, len(columns)):
        vals = {i: float(c[col]) for i, c in enumerate(cells[:-1]) if c[col] != "—"}
        if vals:
            best = min(vals.values()); bold |= {(i, col) for i, v in vals.items() if v == best}
    _render_table(path, columns, cells, bold=frozenset(bold))


if __name__ == "__main__":
    m = collect()
    OTHERS.mkdir(parents=True, exist_ok=True)
    (OTHERS / "gt_pose_metrics.json").write_text(json.dumps(m, indent=2))
    write_table(m, OTHERS / "gt_pose_table.png")
    print(f"[pose] reference reposing check {m['reference_reposing_check_mm']:.2f} mm; "
          f"reference jitter {m['reference_jitter']:.1f}; missing: {m['missing'] or 'none'}")
    print(f"{'row':36s} " + " ".join(f"{k:>9s}" for k in METRICS) + "   repose-chk")
    for r in m["rows"].values():
        print(f"{r['label']:36s} " + " ".join(f"{_cell(r[k]):>9s}" for k in METRICS)
              + f"   {_cell(r['reposing_check_mm'])}")
