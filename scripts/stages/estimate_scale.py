"""Metric scale for VGGT depth, then a per-frame SMPL-X body overfit to that depth.

Two things happen here, in order:

1. **Scale.** VGGT-Omega geometry is up to one arbitrary per-clip factor (depth and
   camera translations share it). The SAM-3D-Body → SMPL-X body carries real metric
   size and was fit in the *same* VGGT camera, so it is a ruler: over the person, the
   body's camera-frame depth and VGGT's relative depth differ by exactly the unknown
   factor ``s``. Per body-valid frame we pose the body with that frame's own betas,
   keep the camera-facing vertices, read VGGT depth under them, and take the robust
   median of ``log(d_body) − log(d_vggt)``. The per-frame log-scales are pooled into
   one clip ``s`` (:func:`tools.geometry.human_scale.combine_log_scales`). No body-part
   weighting — every visible vertex counts the same.

2. **Overfit.** With ``s`` known, we back-project ``s · vggt_depth`` over each person's
   mask into a metric point cloud and **overfit a fresh SMPL-X body to it, per frame**,
   with free shape (per-frame betas) and no temporal coupling. See
   :mod:`tools.human_optim.overfit`. This is a diagnostic body — it does not feed the
   kinematic stage — written so the 3D viewer can show it next to the original and the
   kinematic fit.

Per-clip I/O (``--predictions out/<stem>/``)::

    reads:  sam3d/smplx_params.npz      (per-frame betas + pose, per person)
            sapiens/pose.npz            (Goliath-308 keypoints, for the overfit reproj)
            vggt/camera.npz             (frame_indices, intrinsics @ vggt + normalized)
            vggt/depth/frame_*.npz      (up-to-scale depth + confidence)
            sam3/<obj>/frame_*.png      (per-person silhouette)
    writes: geometry/scale.npz          (scale + per-frame diagnostics)
            geometry/overfit.npz        (per-frame overfit body, camera-frame classic params)
            geometry/diagnostics/scale_over_time.png

Standalone CLI::

    python scripts/stages/estimate_scale.py --predictions out/sport_climb/
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.config import load_toml, section  # noqa: E402
from tools.geometry.human_scale import combine_log_scales, weighted_median  # noqa: E402
from tools.geometry.intrinsics import denormalize_K  # noqa: E402
from tools.human_optim import normalize_q  # noqa: E402
from tools.human_optim.keypoints import build_correspondence  # noqa: E402
from tools.human_optim.losses import pad_frames  # noqa: E402
from tools.human_optim.overfit import OverfitWeights, camera_depth_points, overfit_person  # noqa: E402
from tools.human_optim.visibility import front_facing_mask, subsample_indices, vertex_normals  # noqa: E402
from tools.mask_io import read_binary_mask  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402

_DEFAULT_SMPLX_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
)
_CLASSIC_KEYS = ("betas", "body_pose", "global_orient", "transl",
                 "left_hand_pose", "right_hand_pose")


def _frame_log_scale(
    verts_cam: np.ndarray,
    K: np.ndarray,
    depth_rel: np.ndarray,
    conf: np.ndarray,
    mask: np.ndarray,
    conf_drop_q: float,
    min_inliers: int,
) -> tuple[float, float]:
    """One frame's robust log-scale from front-facing body vertices.

    :param verts_cam: ``(Nf, 3)`` camera-frame front-facing vertices (metric m); their
        ``z`` is the body depth ``d_body``.
    :param K: ``(3, 3)`` VGGT-resolution pixel intrinsics.
    :param depth_rel: ``(Hv, Wv)`` VGGT up-to-scale depth; ``conf`` its confidence;
        ``mask`` this person's silhouette at VGGT resolution.
    :param conf_drop_q: drop the lowest ``conf_drop_q`` fraction of in-mask vertices by
        confidence before reducing.
    :param min_inliers: skip the frame (return ``nan``) below this many inliers.
    :returns: ``(log_scale, weight)`` — ``(nan, 0.0)`` if too few inliers; the weight is
        the summed confidence of the surviving vertices.
    """
    height, width = depth_rel.shape
    depth_body = verts_cam[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    safe_z = np.where(depth_body > 0.0, depth_body, 1.0)  # avoid div-by-zero on rare z<=0 verts
    px = np.rint(fx * verts_cam[:, 0] / safe_z + cx).astype(np.int64)
    py = np.rint(fy * verts_cam[:, 1] / safe_z + cy).astype(np.int64)
    inb = (depth_body > 0) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not inb.any():
        return float("nan"), 0.0
    px, py, depth_body = px[inb], py[inb], depth_body[inb]
    drel = depth_rel[py, px]
    cvals = conf[py, px]
    valid = mask[py, px] & np.isfinite(drel) & (drel > 0)
    if int(valid.sum()) < min_inliers:
        return float("nan"), 0.0
    thresh = np.quantile(cvals[valid], conf_drop_q) if conf_drop_q > 0.0 else -np.inf
    keep = valid & (cvals >= thresh)
    if int(keep.sum()) < min_inliers:
        return float("nan"), 0.0
    log_ratio = np.log(depth_body[keep]) - np.log(drel[keep])
    w = cvals[keep]
    return float(weighted_median(log_ratio, w)), float(w.sum())


def _plot(
    path: Path,
    per_frame_scale: np.ndarray,
    per_frame_weight: np.ndarray,
    frame_indices: np.ndarray,
    object_ids: np.ndarray,
    scale_global: float,
) -> None:
    """Scatter per-frame scale (per person) + weight over time, with the global s."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_people = per_frame_scale.shape[0]
    fig, (ax_scale, ax_weight) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for p in range(num_people):
        ok = np.isfinite(per_frame_scale[p])
        if not ok.any():
            continue
        ax_scale.scatter(frame_indices[ok], per_frame_scale[p][ok], s=12, alpha=0.6,
                         label=f"person {int(object_ids[p])}")
        ax_weight.plot(frame_indices[ok], per_frame_weight[p][ok], lw=1.0,
                       label=f"person {int(object_ids[p])}")
    ax_scale.axhline(scale_global, color="k", ls="--", lw=1.2, label=f"global s={scale_global:.3f}")
    ax_scale.set_ylabel("per-frame scale")
    ax_scale.set_title("Human-anchored metric scale over time")
    ax_scale.legend(fontsize=8, ncol=2)
    ax_scale.grid(alpha=0.3)
    ax_weight.set_ylabel("frame weight (Σ conf)")
    ax_weight.set_xlabel("frame index")
    ax_weight.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _estimate_metric_scale(
    pred: Path, sp: dict, frame_indices: np.ndarray, K_vggt: np.ndarray,
    hv: int, wv: int, object_ids: np.ndarray, mid, device: str, args: argparse.Namespace,
) -> tuple[float, dict]:
    """Robust human-anchored metric scale + per-frame diagnostics."""
    num_people = len(object_ids)
    n_frames = len(frame_indices)
    per_frame_scale = np.full((num_people, n_frames), np.nan, dtype=np.float32)
    per_frame_weight = np.zeros((num_people, n_frames), dtype=np.float32)
    pooled_log: list[float] = []
    pooled_weight: list[float] = []

    for p in range(num_people):
        oid = int(object_ids[p])
        vmask = np.asarray(sp["valid_mask"][p], dtype=bool)
        cols = np.asarray([j for j, idx in enumerate(frame_indices) if vmask[int(idx)]], dtype=int)
        if cols.size == 0:
            print(f"  person obj_id={oid}: no body-valid VGGT frames — skipping")
            continue
        idxs = frame_indices[cols]

        def gather(key: str) -> torch.Tensor:
            return torch.as_tensor(sp[key][p][idxs], dtype=torch.float32, device=device)

        with torch.no_grad():
            betas = gather("betas")
            q_cam = mid.from_classic(betas, gather("body_pose"), gather("transl"),
                                     gather("global_orient"), gather("left_hand_pose"),
                                     gather("right_hand_pose"))
            verts = mid.forward(betas, normalize_q(q_cam, mid.num_joints)).vertices
            front = front_facing_mask(verts, vertex_normals(verts, mid.faces))
        verts_np = verts.cpu().numpy()
        front_np = front.cpu().numpy()

        for fi, (col, idx) in enumerate(zip(cols, idxs)):
            idx = int(idx)
            depth_path = pred / "vggt" / "depth" / f"frame_{idx:06d}.npz"
            mask = read_binary_mask(pred / "sam3" / f"{oid:02d}" / f"frame_{idx:06d}.png")
            if not depth_path.exists() or mask is None:
                continue
            vd = np.load(depth_path)
            depth = vd["depth"].astype(np.float32)
            conf = vd["depth_conf"].astype(np.float32)
            mask = cv2.resize(mask.astype(np.uint8), (wv, hv), interpolation=cv2.INTER_NEAREST) > 0
            log_s, weight = _frame_log_scale(
                verts_np[fi][front_np[fi]], K_vggt[col], depth, conf, mask,
                args.conf_drop_q, args.min_inlier_verts)
            if not np.isfinite(log_s):
                continue
            per_frame_scale[p, col] = float(np.exp(log_s))
            per_frame_weight[p, col] = weight
            pooled_log.append(log_s)
            pooled_weight.append(weight)

    if not pooled_log:
        raise SystemExit(f"[scale] no usable body frames to anchor scale under {pred}")

    res = combine_log_scales(np.asarray(pooled_log), np.asarray(pooled_weight), mad_k=args.mad_k)
    diagnostics = dict(res, per_frame_scale=per_frame_scale, per_frame_weight=per_frame_weight,
                       n_pooled=len(pooled_log))
    return res["scale"], diagnostics


def _overfit_bodies(
    pred: Path, sp: dict, pose: dict, scale: float, frame_indices: np.ndarray,
    K_vggt: np.ndarray, K_orig: np.ndarray, hv: int, wv: int, object_ids: np.ndarray,
    mid, device: str, args: argparse.Namespace,
) -> None:
    """Overfit a per-frame SMPL-X body to the scaled depth cloud → ``geometry/overfit.npz``."""
    weights = OverfitWeights(
        chamfer=args.w_chamfer, reproj=args.w_reproj, pose=args.w_pose, betas=args.w_betas,
        gm_chamfer_m=args.gm_chamfer_m, gm_reproj_px=args.gm_reproj_px)
    corr = build_correspondence(list(mid.frame_names), device=device)
    fvids = torch.as_tensor([int(x) for x in mid.frame_vertex_ids], dtype=torch.long, device=device)
    sub = subsample_indices(mid.num_vertices, args.vertex_subsample, device=device)

    out = {key: np.asarray(sp[key], dtype=np.float32).copy() for key in _CLASSIC_KEYS}

    for p in range(len(object_ids)):
        oid = int(object_ids[p])
        vmask = np.asarray(sp["valid_mask"][p], dtype=bool)
        cols = np.asarray([j for j, idx in enumerate(frame_indices) if vmask[int(idx)]], dtype=int)
        if cols.size == 0:
            continue
        idxs = frame_indices[cols]

        clouds = []
        for col, idx in zip(cols, idxs):
            idx = int(idx)
            depth_path = pred / "vggt" / "depth" / f"frame_{idx:06d}.npz"
            mask = read_binary_mask(pred / "sam3" / f"{oid:02d}" / f"frame_{idx:06d}.png")
            if not depth_path.exists() or mask is None:
                clouds.append(torch.zeros((0, 3), device=device))
                continue
            vd = np.load(depth_path)
            depth = torch.as_tensor(vd["depth"].astype(np.float32), device=device)
            conf = torch.as_tensor(vd["depth_conf"].astype(np.float32), device=device)
            mask = cv2.resize(mask.astype(np.uint8), (wv, hv), interpolation=cv2.INTER_NEAREST) > 0
            clouds.append(camera_depth_points(
                depth, conf, torch.as_tensor(mask, device=device),
                torch.as_tensor(K_vggt[col], dtype=torch.float32, device=device), scale,
                conf_drop_q=args.conf_drop_q, max_points=args.max_points))
        cloud, cloud_valid = pad_frames(clouds, channels=3, device=device, dtype=torch.float32)

        def gather(key: str) -> torch.Tensor:
            return torch.as_tensor(sp[key][p][idxs], dtype=torch.float32, device=device)

        betas0 = gather("betas")
        q0 = mid.from_classic(betas0, gather("body_pose"), gather("transl"),
                              gather("global_orient"), gather("left_hand_pose"),
                              gather("right_hand_pose"))
        K_orig_t = torch.as_tensor(K_orig[cols], dtype=torch.float32, device=device)
        kp = torch.as_tensor(pose["keypoints"][p][idxs], dtype=torch.float32, device=device)
        scores = torch.as_tensor(pose["keypoint_scores"][p][idxs], dtype=torch.float32, device=device)

        print(f"[overfit] person obj_id={oid} ({cols.size} valid frames) …")
        betas_opt, q_opt = _run_person_batched(
            mid, betas0, q0, K_orig_t, cloud, cloud_valid, kp, scores,
            corr, fvids, sub, weights, args)
        classic = mid.to_classic(betas_opt, q_opt)  # keys: betas, body_pose, translation, ...
        for key in _CLASSIC_KEYS:
            src = classic["translation"] if key == "transl" else classic[key]
            out[key][p, idxs] = src.cpu().numpy().astype(np.float32)

    out_path = pred / "geometry" / "overfit.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        valid_mask=np.asarray(sp["valid_mask"], dtype=bool),
        object_ids=np.asarray(object_ids, dtype=np.int32),
        num_people=np.int32(len(object_ids)),
        num_frames=np.int32(sp["num_frames"]),
        video_height=np.int32(sp["video_height"]),
        video_width=np.int32(sp["video_width"]),
        scale=np.float32(scale),
        **out,
    )
    print(f"  overfit body -> {out_path}")


def _run_person_batched(
    mid, betas0, q0, K_orig, cloud, cloud_valid, kp, scores,
    corr, fvids, sub, weights, args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overfit one person in frame-batches (frames are independent → exact) to bound memory."""
    num_frames = q0.shape[0]
    betas_parts, q_parts = [], []
    first = True
    for start in range(0, num_frames, args.frame_batch):
        sl = slice(start, min(start + args.frame_batch, num_frames))
        result = overfit_person(
            mid, betas0[sl], q0[sl], K_orig=K_orig[sl], cloud=cloud[sl],
            cloud_valid=cloud_valid[sl], target_kp=kp[sl], scores=scores[sl],
            corr=corr, frame_vertex_ids=fvids, sub=sub, weights=weights,
            iters=args.iters, lr=args.lr)
        betas_parts.append(result.betas)
        q_parts.append(result.q)
        if first:
            print(f"    frames[{sl.start}:{sl.stop}] before={_fmt(result.before)}  "
                  f"after={_fmt(result.after)}")
            first = False
    return torch.cat(betas_parts, 0), torch.cat(q_parts, 0)


def _fmt(terms: dict) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in terms.items())


def estimate_scale(args: argparse.Namespace) -> None:
    pred = Path(args.predictions)
    out_dir = pred / "geometry"
    scale_path = out_dir / "scale.npz"
    overfit_path = out_dir / "overfit.npz"
    done = scale_path.exists() and (args.skip_overfit or overfit_path.exists())
    if args.skip_existing and done:
        print(f"  skip (exists): {scale_path}")
        return

    sp = dict(np.load(pred / "sam3d" / "smplx_params.npz", allow_pickle=True))
    cam = np.load(pred / "vggt" / "camera.npz")
    frame_indices = cam["frame_indices"].astype(int)
    K_vggt = np.asarray(cam["intrinsics_px"], dtype=np.float32)          # (F, 3, 3) @ vggt res
    hv, wv = int(cam["vggt_height"]), int(cam["vggt_width"])
    H, W = int(sp["video_height"]), int(sp["video_width"])
    K_orig = denormalize_K(np.asarray(cam["intrinsics"], dtype=np.float32), H, W)  # (F, 3, 3) @ orig res
    object_ids = np.asarray(sp["object_ids"], dtype=np.int32)
    device = args.device

    mid = load_smplx_mid(str(_DEFAULT_SMPLX_MODEL), device)

    scale, diag = _estimate_metric_scale(pred, sp, frame_indices, K_vggt, hv, wv,
                                         object_ids, mid, device, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        scale_path,
        scale=np.float32(scale),
        log_scale=np.float32(diag["log_scale"]),
        per_frame_scale=diag["per_frame_scale"],
        per_frame_weight=diag["per_frame_weight"],
        frame_indices=frame_indices.astype(np.int32),
        object_ids=object_ids,
        scale_mad=np.float32(diag["mad"]),
        n_frames_used=np.int32(diag["n_used"]),
        n_people=np.int32(len(object_ids)),
        method=np.str_("human-log-ratio"),
    )
    _plot(out_dir / "diagnostics" / "scale_over_time.png",
          diag["per_frame_scale"], diag["per_frame_weight"], frame_indices, object_ids, scale)
    print(f"  scale={scale:.4f} (log {diag['log_scale']:.4f}, mad {diag['mad']:.4f}, "
          f"{diag['n_used']}/{diag['n_pooled']} frames) -> {scale_path}")

    if not args.skip_overfit:
        pose = dict(np.load(pred / "sapiens" / "pose.npz", allow_pickle=True))
        _overfit_bodies(pred, sp, pose, scale, frame_indices, K_vggt, K_orig, hv, wv,
                        object_ids, mid, device, args)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions", required=True, type=Path, help="clip dir, e.g. out/<stem>/")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "pipeline.toml",
                   help="pipeline TOML (default: configs/pipeline.toml)")
    cli = p.parse_args(argv)
    cfg = load_toml(cli.config)
    args = section(cfg, "scale", {
        "conf_drop_q": 0.3,
        "min_inlier_verts": 50,
        "mad_k": 3.0,
        "skip_existing": False,
        # overfit knobs
        "skip_overfit": False,
        "iters": 300,
        "lr": 0.02,
        "frame_batch": 512,
        "w_chamfer": 1.0,
        "w_reproj": 0.5,
        "w_pose": 0.02,
        "w_betas": 1e-3,
        "gm_chamfer_m": 0.3,
        "gm_reproj_px": 10.0,
        "max_points": 4000,
        "vertex_subsample": 3000,
    })
    args.device = section(cfg, "runtime", {"device": "cuda"}).device
    args.predictions, args.config = cli.predictions, cli.config
    return args


def main(argv: list[str] | None = None) -> int:
    estimate_scale(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
