"""Human-body-anchored metric scale for VGGT geometry (runs before ``fuse``).

VGGT-Omega geometry is **up to one arbitrary per-clip scale** (depth and camera
translations share it). The SAM-3D-Body → SMPL-X body carries **real metric
size** and was fit in the *same* VGGT camera (``estimate_body_sam3d`` reads
VGGT's intrinsics), so it is a metric ruler: over the person, the body's
camera-frame depth ``d_metric`` and VGGT's relative depth ``d_rel`` differ by
exactly the unknown scale ``s`` (``d_metric ≈ s · d_rel``).

Per tracked person, per frame with a body fit:

  1. pose the SMPL-X body with that frame's **own** (per-frame) betas + pose,
  2. keep the camera-facing (front) vertices — VGGT only sees the front surface,
  3. project them with VGGT's intrinsics and read ``d_rel`` at each pixel,
     gated to the person's SAM3 mask + confident, valid VGGT depth,
  4. per vertex ``log(d_metric) − log(d_rel)``; the frame's robust center is the
     median, its weight the summed VGGT depth-confidence of the inliers.

The per-frame log-scales are combined into one clip scale by a confidence-
weighted robust mean in log space (:func:`tools.geometry.human_scale.combine_log_scales`).
All tracked people are pooled — each is an independent ruler for the same ``s``;
the confidence weight down-ranks a small / occluded person. Per-frame betas are
used on purpose (each frame's ruler matches what SAM-3D-Body saw that frame);
betas are averaged only later, before the kinematic optimization.

Per-clip I/O (``--predictions out/<stem>/``)::

    reads:  sam3d/smplx_params.npz              (per-frame betas + pose, per person)
            vggt/camera.npz                      (frame_indices, intrinsics_px @ vggt res)
            vggt/depth/frame_*.npz               (up-to-scale depth + confidence)
            sam3/<obj>/frame_*.png               (per-person silhouette gate)
    writes: geometry/scale.npz                   (scale + per-frame diagnostics)
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

from tools.geometry.human_scale import combine_log_scales  # noqa: E402
from tools.mask_io import read_binary_mask  # noqa: E402
from tools.human_optim import normalize_q  # noqa: E402
from tools.human_optim.visibility import front_facing_mask, vertex_normals  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402

_DEFAULT_SMPLX_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
)


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

    :param verts_cam: ``(Nf, 3)`` camera-frame front-facing vertices (metric m);
        their ``z`` is the body (metric) depth ``d_metric``.
    :param K: ``(3, 3)`` VGGT-resolution pixel intrinsics.
    :param depth_rel: ``(Hv, Wv)`` VGGT up-to-scale depth.
    :param conf: ``(Hv, Wv)`` VGGT depth confidence (higher = better).
    :param mask: ``(Hv, Wv)`` this person's silhouette at VGGT resolution.
    :param conf_drop_q: drop the lowest ``conf_drop_q`` fraction of in-mask
        vertices by VGGT confidence before reducing.
    :param min_inliers: skip the frame (return ``nan``) below this many inliers.
    :returns: ``(log_scale, weight)`` — ``(nan, 0.0)`` if too few inliers; the
        weight is the summed VGGT confidence of the surviving vertices.
    """
    height, width = depth_rel.shape
    depth_metric = verts_cam[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    safe_z = np.where(depth_metric > 0.0, depth_metric, 1.0)  # avoid div-by-zero on rare z<=0 verts
    px = np.rint(fx * verts_cam[:, 0] / safe_z + cx).astype(np.int64)
    py = np.rint(fy * verts_cam[:, 1] / safe_z + cy).astype(np.int64)
    inb = (depth_metric > 0) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not inb.any():
        return float("nan"), 0.0
    px, py, depth_metric = px[inb], py[inb], depth_metric[inb]
    drel = depth_rel[py, px]
    cvals = conf[py, px]
    valid = mask[py, px] & np.isfinite(drel) & (drel > 0)
    if int(valid.sum()) < min_inliers:
        return float("nan"), 0.0
    thresh = np.quantile(cvals[valid], conf_drop_q) if conf_drop_q > 0.0 else -np.inf
    keep = valid & (cvals >= thresh)
    if int(keep.sum()) < min_inliers:
        return float("nan"), 0.0
    log_ratio = np.log(depth_metric[keep]) - np.log(drel[keep])
    return float(np.median(log_ratio)), float(cvals[keep].sum())


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

    P = per_frame_scale.shape[0]
    fig, (ax_scale, ax_weight) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for p in range(P):
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


def estimate_scale(args: argparse.Namespace) -> None:
    pred = Path(args.predictions)
    out_dir = pred / "geometry"
    if args.skip_existing and (out_dir / "scale.npz").exists():
        print(f"  skip (exists): {out_dir / 'scale.npz'}")
        return

    sp = dict(np.load(pred / "sam3d" / "smplx_params.npz", allow_pickle=True))
    cam = np.load(pred / "vggt" / "camera.npz")
    frame_indices = cam["frame_indices"].astype(int)          # (F,) original frame ids
    K_vggt = np.asarray(cam["intrinsics_px"], dtype=np.float32)  # (F, 3, 3) @ vggt res
    hv, wv = int(cam["vggt_height"]), int(cam["vggt_width"])
    P = int(sp["num_people"])
    object_ids = np.asarray(sp["object_ids"], dtype=np.int32)
    device = args.device

    mid = load_smplx_mid(str(_DEFAULT_SMPLX_MODEL), device)

    n_frames = len(frame_indices)
    per_frame_scale = np.full((P, n_frames), np.nan, dtype=np.float32)
    per_frame_weight = np.zeros((P, n_frames), dtype=np.float32)
    pooled_log: list[float] = []
    pooled_weight: list[float] = []

    for p in range(P):
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
            betas = gather("betas")   # (Fp, 10) — PER-FRAME betas (deliberately not averaged)
            q_cam = mid.from_classic(betas, gather("body_pose"), gather("transl"),
                                     gather("global_orient"), gather("left_hand_pose"),
                                     gather("right_hand_pose"))
            verts = mid.forward(betas, normalize_q(q_cam, mid.num_joints)).vertices  # (Fp, V, 3)
            front = front_facing_mask(verts, vertex_normals(verts, mid.faces))       # (Fp, V)
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
    scale = res["scale"]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "scale.npz",
        scale=np.float32(scale),
        log_scale=np.float32(res["log_scale"]),
        per_frame_scale=per_frame_scale,
        per_frame_weight=per_frame_weight,
        frame_indices=frame_indices.astype(np.int32),
        object_ids=object_ids,
        scale_mad=np.float32(res["mad"]),
        n_frames_used=np.int32(res["n_used"]),
        n_people=np.int32(P),
        method=np.str_("human-log-ratio"),
    )
    _plot(out_dir / "diagnostics" / "scale_over_time.png",
          per_frame_scale, per_frame_weight, frame_indices, object_ids, scale)
    print(f"  scale={scale:.4f} (log {res['log_scale']:.4f}, mad {res['mad']:.4f}, "
          f"{res['n_used']}/{len(pooled_log)} frames) -> {out_dir / 'scale.npz'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions", required=True, type=Path, help="clip dir, e.g. out/<stem>/")
    p.add_argument("--conf-drop-q", type=float, default=0.3,
                   help="drop this lowest-confidence fraction of in-mask body vertices per frame (default: 0.3)")
    p.add_argument("--min-inlier-verts", type=int, default=50,
                   help="skip a frame with fewer than this many inlier vertices (default: 50)")
    p.add_argument("--mad-k", type=float, default=3.0,
                   help="reject frames beyond this many MADs from the median log-scale (default: 3.0)")
    p.add_argument("--device", default="cuda", help="cuda | cpu (default: cuda)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip if geometry/scale.npz already exists")
    estimate_scale(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
