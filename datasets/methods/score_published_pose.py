"""Pose metrics of published models on one of the two pose benchmarks.

``--benchmark climbingvideos`` — the ClimbingVideos test split. There is no
motion-capture ground truth on the climbing clips. The reference is our own final
body, ``features/human_optim/<scene>/kindyn_1.npz`` (SMPL-X, world frame = the VGGT
camera world, metric through the human ruler). So every number there says "how far
is this model from our reconstruction", not "how far from the truth".

``--benchmark climb_wall_3`` — the reference is the three-camera fit in
``peter/climb_wall_3_gt``: calibrated cameras, a shape frozen from the participant's
measured height and mass, and Sapiens keypoints in all three views. Its world frame
*is* the left camera's frame, in metres, so a body already in that camera needs no
transform at all. Our own single-view runs are therefore scored rows like any other,
not the ruler. The models and our single-view run both read the ORIGINAL
``cam_left.mp4``; the reference was fitted on the undistorted frames, but the two
cameras share the same intrinsics (undistortion kept K), so the 3D frames coincide
exactly. The only residue is lens distortion nudging the image evidence, which moves
a pixel on the climber's body by a median 0.1–1.0 px (99th percentile under 2.9 px) —
far below the errors in this table. Frames beyond the reference's own length (it
drops the last two, where the lagging cameras run out) are not scored.

Every body — the reference, the SAM-3D-Body initialisation, our runs and each
published model — is posed with the same pip ``smplx`` neutral model
(``flat_hand_mean=False``, zero hands / expression where a model predicts none), so
joints and vertices compare like with like. Camera-frame metrics take the reference
through its per-frame camera-from-world extrinsics and the model's own camera-frame
parameters.

Metrics (mm unless said otherwise; 22 SMPL-X body joints), the WHAM/EMDB set:

- MPJPE / PA-MPJPE / PVE — per frame in the camera frame, pelvis-aligned
  (pelvis = midpoint of the hips), PA = after a per-frame similarity Procrustes fit.
- W-MPJPE / WA-MPJPE — the model's world trajectory vs the reference, in 100-frame
  segments (WHAM protocol): WA aligns each segment with one similarity transform, W
  aligns only its first two frames.
- RTE (%) — root (pelvis) trajectory error after one similarity alignment of the
  whole clip, divided by the reference root's total path length.
- Accel (m/s²) — mean ‖a_pred − a_ref‖ of the camera-frame joints, finite
  differences at the clip's fps.
- Jitter (10 m/s³) — mean jerk of the model's own world joints (EMDB formula, with
  the PIP ignore slots); the reference row shows the reference's own.

Writes, next to the model results (or into ``--out``); missing model results are
skipped and listed:

    climbingvideos  ClimbingVideos/results/baseline_pose_{metrics.json,table.png}
    climb_wall_3    peter/climb_wall_3_gt/others/gt_pose_{metrics.json,table.png}

Usage:
    uv run --no-sync python datasets/methods/score_published_pose.py --benchmark climbingvideos
    uv run --no-sync python datasets/methods/score_published_pose.py --benchmark climb_wall_3
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import smplx
import torch

from bvr.body import models_dir
from datasets._common import _render_table
from datasets.climbingvideos.paths import CLIMBING
from datasets.peter.paths import PETER

FEATURES, RESULTS = CLIMBING.resolve() / "features", CLIMBING.resolve() / "results"
GT_ROOT = PETER / "climb_wall_3_gt" / "human"
OURS_ROOT = PETER / "out_climb_wall_3_single"
OTHERS = PETER / "climb_wall_3_gt" / "others"
MODELS = models_dir()
#: Per benchmark: the directory that holds the model results and receives the scores,
#: the output file prefix, and the reference row's label.
BENCHMARKS = {"climbingvideos": (RESULTS, "baseline", "Ours (reference)"),
              "climb_wall_3": (OTHERS, "gt", "Multi-view GT (reference)")}
POSE_MODELS = {"gvhmr": "GVHMR", "prompthmr": "PromptHMR"}
#: Our own runs on climb_wall_3: row key -> (label, the run directory under ``<clip>/``).
#: They share every stage but the last, so the rows price one change each — measured
#: contacts instead of estimated ones, and a torque priced the same at every joint
#: instead of by the joint's strength.
OURS_RUNS = {"ours/kindyn_1": ("Ours (single view)", "human_optim"),
             "ours/gt_contact": ("Ours + measured contacts", "human_optim_gt_contact"),
             "ours/flat_torque": ("Ours, flat torque weights", "human_optim_flat_torque")}
BODY_KEYS = ("global_orient", "body_pose", "transl", "left_hand_pose", "right_hand_pose",
             "jaw_pose", "leye_pose", "reye_pose", "expression", "betas")
N_BODY, SEGMENT = 22, 100
JITTER_IGNORED = [0, 7, 8, 10, 11, 20, 21, 22, 23]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CACHE: dict[tuple[str, int], torch.nn.Module] = {}


def scene_dir(stage: str, scene: str) -> Path:
    return FEATURES / stage / scene[:2] / scene[2:4] / scene


def items(benchmark: str) -> list[tuple[str, float]]:
    """``(scene or clip, fps)`` for every item the benchmark scores."""
    if benchmark == "climbingvideos":
        rows = [line.split("\t") for line in (RESULTS / "test_scenes.txt").read_text().splitlines() if line]
        return [(r[0], float(r[2])) for r in rows]
    names = sorted(d.name for d in GT_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_"))
    return [(c, float(np.load(GT_ROOT / c / "cameras.npz", allow_pickle=True)["fps"])) for c in names]


def pose(params: dict[str, np.ndarray], model_type: str = "smplx") -> tuple[np.ndarray, np.ndarray | None]:
    """(joints (T, 22, 3), vertices (T, V, 3) or None for SMPL) in the parameters' own frame."""
    n = len(params["body_pose"])
    key = (model_type, n)
    if key not in _CACHE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _CACHE[key] = smplx.create(str(MODELS), model_type=model_type, gender="neutral", use_pca=False,
                                       flat_hand_mean=False, num_betas=10, num_expression_coeffs=10,
                                       batch_size=n).to(DEVICE)
    model = _CACHE[key]
    keys = ["global_orient", "body_pose", "transl", "betas"]
    if model_type == "smplx":
        keys += ["left_hand_pose", "right_hand_pose", "jaw_pose", "leye_pose", "reye_pose", "expression"]
    t = {}
    for k in keys:
        v = np.asarray(params.get(k, np.zeros((n, {"betas": 10, "expression": 10}.get(k, 3)))), np.float32)
        if v.ndim == 1:
            v = np.broadcast_to(v[None], (n, v.shape[0]))
        t[k] = torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        out = model(**t)
    joints = out.joints[:, :N_BODY].cpu().numpy().astype(np.float64)
    verts = out.vertices.cpu().numpy().astype(np.float64) if model_type == "smplx" else None
    return joints, verts


def similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (s, R, t) with s*R@src + t closest to dst; src/dst (K, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    x_s, x_d = src - mu_s, dst - mu_d
    u, sv, vt = np.linalg.svd(x_s.T @ x_d)
    sign = np.sign(np.linalg.det(u @ vt))
    d = np.diag([1.0, 1.0, sign])
    rot = (u @ d @ vt).T
    scale = np.trace(np.diag(sv) @ d) / (x_s ** 2).sum()
    return scale, rot, mu_d - scale * rot @ mu_s


def apply(tf: tuple[float, np.ndarray, np.ndarray], pts: np.ndarray) -> np.ndarray:
    scale, rot, trans = tf
    return scale * pts @ rot.T + trans


def to_camera(pts: np.ndarray, world2cam: np.ndarray) -> np.ndarray:
    return pts @ world2cam[:, :3, :3].transpose(0, 2, 1) + world2cam[:, None, :3, 3]


def camera_metrics(pj: np.ndarray, gj: np.ndarray, pv: np.ndarray | None, gv: np.ndarray, fps: float,
                   valid: np.ndarray) -> dict:
    """Pelvis-aligned per-frame errors (summed over frames, so scenes pool by frame).

    ``valid`` marks frames both bodies have; accelerations use only windows of three
    consecutive valid frames.
    """
    accel = lambda j: (j[2:] - 2 * j[1:-1] + j[:-2]) * fps ** 2  # noqa: E731
    ok = valid[2:] & valid[1:-1] & valid[:-2]
    acc = np.linalg.norm(accel(pj) - accel(gj), axis=-1).mean(1)[ok]
    pj, gj = pj[valid], gj[valid]
    if pv is not None:
        pv, gv = pv[valid], gv[valid]
    pel_p, pel_g = pj[:, 1:3].mean(1, keepdims=True), gj[:, 1:3].mean(1, keepdims=True)
    pj, gj = pj - pel_p, gj - pel_g
    mpjpe = np.linalg.norm(pj - gj, axis=-1).mean(1)
    pa = np.array([np.linalg.norm(apply(similarity(pj[i], gj[i]), pj[i]) - gj[i], axis=-1).mean()
                   for i in range(len(pj))])
    out = {"MPJPE": mpjpe.sum() * 1e3, "PA-MPJPE": pa.sum() * 1e3, "Accel": acc.sum(), "frames": len(pj),
           "accel_frames": len(acc)}
    if pv is not None:
        out["PVE"] = np.linalg.norm((pv - pel_p) - (gv - pel_g), axis=-1).mean(1).sum() * 1e3
    return out


def world_metrics(pj: np.ndarray, gj: np.ndarray, valid: np.ndarray) -> dict:
    """Segments are cut on the full timeline, then invalid frames are dropped inside each."""
    wa, w = [], []
    for start in range(0, len(pj), SEGMENT):
        keep = valid[start:start + SEGMENT]
        p, g = pj[start:start + SEGMENT][keep], gj[start:start + SEGMENT][keep]
        if len(p) < 2:
            continue
        wa.append(np.linalg.norm(apply(similarity(p.reshape(-1, 3), g.reshape(-1, 3)), p) - g, axis=-1).mean())
        w.append(np.linalg.norm(apply(similarity(p[:2].reshape(-1, 3), g[:2].reshape(-1, 3)), p) - g, axis=-1).mean())
    root_p, root_g = pj[valid, 1:3].mean(1), gj[valid, 1:3].mean(1)
    err = np.linalg.norm(apply(similarity(root_p, root_g), root_p) - root_g, axis=-1).mean()
    path = np.linalg.norm(np.diff(root_g, axis=0), axis=-1).sum()
    return {"WA-MPJPE": np.sum(wa) * 1e3, "W-MPJPE": np.sum(w) * 1e3, "segments": len(wa),
            "RTE": 100.0 * err / max(path, 1e-6)}


def jitter(joints: np.ndarray, fps: float, valid: np.ndarray) -> float:
    j = np.zeros((len(joints), 24, 3))
    j[:, :N_BODY] = joints
    j[:, JITTER_IGNORED] = 0
    ok = valid[3:] & valid[2:-1] & valid[1:-2] & valid[:-3]
    jerk = np.linalg.norm((j[3:] - 3 * j[2:-1] + 3 * j[1:-2] - j[:-3]) * fps ** 3, axis=2)[ok]
    return float(jerk.mean() / 10)


def reference(benchmark: str, item: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """The reference body: (joints_world, verts_world, world2cam, valid frames, reposing check in mm).

    On ClimbingVideos it is our body, carried into each frame's camera by our own
    extrinsics; frames the tracker lost (``valid_mask`` False; the initialisation is
    NaN there and our body only interpolates) are excluded from every metric. On
    climb_wall_3 it is the multi-view body, whose world is the left camera's frame, so
    ``world2cam`` is the identity — kept explicit so the camera-frame metrics read the
    same as everywhere else.
    """
    cv = benchmark == "climbingvideos"
    k = np.load((scene_dir("human_optim", item) if cv else GT_ROOT / item / "human_optim") / "kindyn_1.npz")
    j, v = pose({key: k[key][0] for key in BODY_KEYS})
    valid = np.asarray(k["valid_mask"][0], bool)
    check = np.linalg.norm(j - k["joints_world"][0, :, :N_BODY], axis=-1)[valid].mean() * 1e3
    if not cv:
        return j, v, np.repeat(np.eye(4)[None], len(j), 0), valid, check
    tr = np.load(scene_dir("geometry", item) / "transform.npz")
    assert bool(tr["metric"]) and np.array_equal(tr["frame_indices"], np.arange(len(j))), item
    return j, v, tr["extrinsics"].astype(np.float64), valid, check


def model_params(d, frame: str) -> tuple[dict[str, np.ndarray], str]:
    p = {"global_orient": d[f"global_orient_{frame}"], "transl": d[f"transl_{frame}"],
         "body_pose": d["body_pose"], "betas": d["betas"]}
    for hand in ("left_hand_pose", "right_hand_pose"):
        if hand in d.files:
            p[hand] = d[hand]
    return p, str(d["model_type"])


def _restrict(params: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    """Keep the selected frames; per-clip parameters (PromptHMR-post's betas) pass through."""
    return {k: (v[keep] if np.ndim(v) > 1 and len(v) == len(keep) else v) for k, v in params.items()}


def score_model(d, gj_w: np.ndarray, gv_w: np.ndarray, w2c: np.ndarray, valid: np.ndarray,
                fps: float) -> dict:
    idx = np.asarray(d["frame_indices"], int)
    keep = idx < len(gj_w)                      # the climb_wall_3 reference is two frames shorter than the video
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
    """Our single-view climb_wall_3 body, carried into the left camera by its own extrinsics."""
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


def score_init(benchmark: str, item: str, gj_w: np.ndarray, gv_w: np.ndarray, w2c: np.ndarray,
               valid: np.ndarray, fps: float) -> dict:
    """The SAM-3D-Body → SMPL-X initialisation, camera frame only."""
    sam3d = scene_dir("sam3d", item) if benchmark == "climbingvideos" else OURS_ROOT / item / "sam3d"
    d = np.load(sam3d / "smplx_params.npz")
    n = len(gj_w)
    pj_c, pv_c = pose({key: np.nan_to_num(d[key][0][:n]) for key in BODY_KEYS})
    ok = valid & np.asarray(d["valid_mask"][0], bool)[:n]
    return camera_metrics(pj_c, to_camera(gj_w, w2c), pv_c, to_camera(gv_w, w2c), fps, ok)


def collect(benchmark: str) -> dict:
    """Score every row on one benchmark and return the metrics tree."""
    cv = benchmark == "climbingvideos"
    unit, results = ("scenes" if cv else "clips"), BENCHMARKS[benchmark][0]
    sums: dict[str, dict] = {}
    missing: dict[str, int] = {}
    ref_checks, ref_jitter = [], []

    def add(key: str, label: str, m: dict) -> None:
        s = sums.setdefault(key, {"label": label, unit: 0, "acc": {}})
        s[unit] += 1
        for k, v in m.items():
            if v is None:
                continue
            if k in ("RTE", "Jitter", "reposing_check_mm"):
                s["acc"].setdefault(k, []).append(v)
            else:
                s["acc"][k] = s["acc"].get(k, 0.0) + v

    scored = items(benchmark)
    for item, fps in scored:
        gj_w, gv_w, w2c, valid, check = reference(benchmark, item)
        ref_checks.append(check)
        ref_jitter.append(jitter(gj_w, fps, valid))
        add("sam3d/init", "SAM 3D Body init (camera frame)",
            score_init(benchmark, item, gj_w, gv_w, w2c, valid, fps))
        for key, (label, optim_dir) in ({} if cv else OURS_RUNS).items():
            if (OURS_ROOT / item / optim_dir / "kindyn_1.npz").exists():
                add(key, label, score_ours(item, gj_w, gv_w, w2c, valid, fps, optim_dir))
        for model, name in POSE_MODELS.items():
            for variant in ("raw", "post"):
                path = results / model / item / f"pose_{variant}.npz"
                if not path.exists():
                    missing[f"{model}/{variant}"] = missing.get(f"{model}/{variant}", 0) + 1
                    continue
                with np.load(path, allow_pickle=True) as d:
                    add(f"{model}/{variant}", f"{name} ({variant})",
                        score_model(d, gj_w, gv_w, w2c, valid, fps))
        print(f"[pose] {item} done", flush=True)

    rows = {}
    for key, s in sums.items():
        a = s["acc"]
        r = {"label": s["label"], unit: s[unit], "frames": int(a["frames"])}
        for k in ("MPJPE", "PA-MPJPE", "PVE"):
            r[k] = a[k] / a["frames"] if k in a else None
        r["Accel"] = a["Accel"] / a["accel_frames"]
        for k in ("WA-MPJPE", "W-MPJPE"):
            r[k] = a[k] / a["segments"] if k in a else None
        for k in ("RTE", "Jitter", "reposing_check_mm"):
            r[k] = float(np.mean(a[k])) if k in a else None
        rows[key] = r
    head = ({"reference": "features/human_optim/<scene>/kindyn_1.npz (our final body)"} if cv else
            {"reference": "peter/climb_wall_3_gt/human/<clip>/human_optim/kindyn_1.npz "
                          "(three-camera fit, frozen participant shape, left-camera frame)",
             "clips": len(scored)})
    return head | {"reference_reposing_check_mm": float(np.mean(ref_checks)),
                   "reference_jitter": float(np.mean(ref_jitter)),
                   "missing": missing, "rows": rows}


METRICS = ["MPJPE", "PA-MPJPE", "PVE", "WA-MPJPE", "W-MPJPE", "RTE", "Accel", "Jitter"]


def _cell(v) -> str:
    return "—" if v is None else f"{v:.1f}"


def write_table(m: dict, path: Path, reference_label: str) -> None:
    cells = [[r["label"], *(_cell(r[k]) for k in METRICS)] for r in m["rows"].values()]
    cells.append([reference_label, *["—"] * 7, _cell(m["reference_jitter"])])
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS))
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: the benchmark's model-results directory)")
    args = parser.parse_args()
    results, prefix, reference_label = BENCHMARKS[args.benchmark]
    out = args.out or results
    m = collect(args.benchmark)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{prefix}_pose_metrics.json").write_text(json.dumps(m, indent=2))
    write_table(m, out / f"{prefix}_pose_table.png", reference_label)
    print(f"[pose] reference reposing check {m['reference_reposing_check_mm']:.2f} mm; "
          f"reference jitter {m['reference_jitter']:.1f}; missing: {m['missing'] or 'none'}")
    print(f"{'row':36s} " + " ".join(f"{k:>9s}" for k in METRICS) + "   repose-chk")
    for r in m["rows"].values():
        print(f"{r['label']:36s} " + " ".join(f"{_cell(r[k]):>9s}" for k in METRICS)
              + f"   {_cell(r['reposing_check_mm'])}")
