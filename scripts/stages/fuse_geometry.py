"""Fuse VGGT geometry with MoGe2 metric depth → metric scene + human clouds.

Runs after the ``moge2`` stage. VGGT-Omega gives cross-frame-consistent cameras
and depth but **up to one arbitrary scale** and at a **reduced resolution**;
MoGe2 gives **metric** per-frame depth. This stage:

  1. estimates one global metric scale ``s`` (per-frame median depth ratio →
     inlier-weighted median across frames — robust to MoGe2's per-frame noise);
  2. denormalizes VGGT's intrinsics to the original frame resolution (FOV is
     VGGT's and is preserved — scaling only touches depth + camera translation);
  3. backprojects every cleaned VGGT depth map to metric world points, split into
     a static **scene** cloud and per-frame **human** points via the SAM3 masks
     (union of all tracked people, each point tagged with ``obj_id``). The scene
     subtracts a *dilated* human mask (imperfect masks leak human pixels into the
     background); the human keeps every in-mask point (edge filter only — no
     confidence gate);
  4. merges near-duplicate points by voxel hashing — the scene globally (it is
     static); the human is kept per-frame, unmerged (the person moves).

Per-clip I/O (``--source`` frames + ``--predictions out/<stem>/``)::

    reads:  --source video / folder of frames          (RGB, for per-point colour)
            vggt/camera.npz, vggt/depth/frame_*.npz     (needs [vggt] save_depth)
            moge2/depth/frame_*.npz                     (needs [moge2] save_depth)
            sam3/<obj>/frame_*.png
    writes: geometry/transform.npz   scale + metric extrinsics + original-res K
            geometry/scene.npz        merged metric background points (+ colour)
            geometry/human.npz        per-frame metric human points (frame_id, obj_id, colour)

Standalone CLI::

    python scripts/stages/fuse_geometry.py --source video.mp4 --predictions out/sport_climb/
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch

from tools.adapters.moge2 import denormalize_K
from tools.geometry.backproject import backproject
from tools.geometry.clean import clean_depth
from tools.geometry.depth_align import frame_scale, weighted_median
from tools.geometry.voxel import group_reduce, voxel_downsample
from tools.mask_io import read_binary_mask
from tools.source import resolve_source


def _discover_objs(sam3_dir: Path, requested: list[int] | None) -> list[int]:
    found = sorted(int(d.name) for d in sam3_dir.iterdir() if d.is_dir() and d.name.isdigit())
    return [o for o in found if o in requested] if requested else found


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean mask by ``radius`` px (elliptical); ``radius<=0`` is a no-op."""
    if radius <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def _human_masks(sam3_dir: Path, idx: int, objs: list[int], hv: int, wv: int):
    """Union of person masks at VGGT resolution: ``(is_human bool, obj_id int16)``."""
    is_human = np.zeros((hv, wv), bool)
    obj_id = np.full((hv, wv), -1, np.int16)
    for o in objs:
        m = read_binary_mask(sam3_dir / f"{o:02d}" / f"frame_{idx:06d}.png")
        if m is None:
            continue
        m = cv2.resize(m.astype(np.uint8), (wv, hv), interpolation=cv2.INTER_NEAREST) > 0
        obj_id[m & ~is_human] = o
        is_human |= m
    return is_human, obj_id


def _save_cloud(path: Path, points: torch.Tensor, **arrays) -> None:
    np.savez(path, points=points.detach().cpu().numpy().astype(np.float32),
             **{k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in arrays.items()})


def fuse(args: argparse.Namespace) -> None:
    pred = Path(args.predictions)
    for sub in ("vggt/depth", "moge2/depth"):
        if not (pred / sub).is_dir():
            raise SystemExit(f"[fuse] missing {sub}/ under {pred} — run that stage with save_depth=true")
    out_dir = pred / "geometry"
    if args.skip_existing and (out_dir / "transform.npz").exists():
        print(f"  skip (exists): {out_dir / 'transform.npz'}")
        return

    cam = np.load(pred / "vggt" / "camera.npz")
    frame_indices = cam["frame_indices"]
    extr, K_px, K_norm = cam["extrinsics"], cam["intrinsics_px"], cam["intrinsics"]
    hv, wv = int(cam["vggt_height"]), int(cam["vggt_width"])
    ho, wo = int(cam["image_height"]), int(cam["image_width"])
    requested = [int(x) for x in args.person_obj_ids.split(",")] if args.person_obj_ids else None
    objs = _discover_objs(pred / "sam3", requested)
    dev = args.device

    scene_pts, scene_conf, scene_col = [], [], []
    human_pts, human_conf, human_obj, human_fid, human_col = [], [], [], [], []
    ratios, inliers = [], []

    with tempfile.TemporaryDirectory(prefix="fuse_frames_") as tmp:
        frame_paths, _ = resolve_source(Path(args.source), Path(tmp))
        for k, idx in enumerate(frame_indices):
            idx = int(idx)
            vd = np.load(pred / "vggt" / "depth" / f"frame_{idx:06d}.npz")
            depth, conf = vd["depth"].astype(np.float32), vd["depth_conf"].astype(np.float32)
            # Scene gets the full clean (confidence-gated); the human keeps every
            # in-mask point — edge filter only, no confidence drop.
            valid_scene = clean_depth(depth, conf, conf_drop_q=args.conf_drop_q, edge_rtol=args.edge_rtol)
            valid_human = clean_depth(depth, edge_rtol=args.edge_rtol)

            md = np.load(pred / "moge2" / "depth" / f"frame_{idx:06d}.npz")
            r, n = frame_scale(depth, md["depth"].astype(np.float32), valid_scene, md["mask"])
            ratios.append(r)
            inliers.append(n)

            rgb = cv2.cvtColor(cv2.imread(str(frame_paths[idx])), cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (wv, hv), interpolation=cv2.INTER_AREA)

            is_human, obj_id = _human_masks(pred / "sam3", idx, objs, hv, wv)
            # Imperfect masks leak human-edge pixels into the background, so subtract
            # a *dilated* human band from the scene; collect the human from the tight mask.
            scene_mask = valid_scene & ~_dilate(is_human, args.human_dilate)
            human_mask = valid_human & is_human

            depth_t, K_t, E_t = (torch.from_numpy(depth).to(dev),
                                 torch.from_numpy(K_px[k]).to(dev), torch.from_numpy(extr[k]).to(dev))
            conf_t = torch.from_numpy(conf).to(dev)
            col_t = torch.from_numpy(np.ascontiguousarray(rgb)).to(dev)

            ss = torch.from_numpy(scene_mask).to(dev)
            scene_pts.append(backproject(depth_t, K_t, E_t, mask=ss))
            scene_conf.append(conf_t[ss])
            scene_col.append(col_t[ss])
            if human_mask.any():
                sh = torch.from_numpy(human_mask).to(dev)
                human_pts.append(backproject(depth_t, K_t, E_t, mask=sh))
                human_conf.append(conf_t[sh])
                human_obj.append(torch.from_numpy(obj_id).to(dev)[sh].to(torch.int32))
                human_fid.append(torch.full((int(sh.sum()),), k, dtype=torch.int32, device=dev))
                human_col.append(col_t[sh])

    s = weighted_median(ratios, inliers)
    per_frame = np.asarray(ratios, np.float32)
    mad = float(np.nanmedian(np.abs(per_frame - s)))

    centroids, inverse, counts = voxel_downsample(torch.cat(scene_pts) * s, args.voxel_scene)
    scene_confidence = group_reduce(torch.cat(scene_conf), inverse, counts.numel(), "amax")
    scene_color = torch.zeros((counts.numel(), 3), device=dev)
    scene_color.index_add_(0, inverse, torch.cat(scene_col).float())
    scene_color = (scene_color / counts.unsqueeze(1)).round().to(torch.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    extr_metric = extr.copy()
    extr_metric[:, :3, 3] *= s
    K_orig = np.stack([denormalize_K(K_norm[k], ho, wo) for k in range(len(frame_indices))])
    np.savez(
        out_dir / "transform.npz",
        scale=np.float32(s),
        extrinsics=extr_metric.astype(np.float32),
        intrinsics_px_orig=K_orig.astype(np.float32),
        intrinsics=K_norm,
        fov_x_deg=cam["fov_x_deg"], fov_y_deg=cam["fov_y_deg"],
        frame_indices=frame_indices.astype(np.int32),
        image_height=np.int32(ho), image_width=np.int32(wo),
        vggt_height=np.int32(hv), vggt_width=np.int32(wv),
        per_frame_scale=per_frame,
        per_frame_inliers=np.asarray(inliers, np.int32),
        scale_mad=np.float32(mad),
        scale_method=np.str_("median-of-frame-medians"),
    )
    _save_cloud(out_dir / "scene.npz", centroids,
                colors=scene_color, confidence=scene_confidence,
                count=counts.to(torch.int32), voxel_size=np.float32(args.voxel_scene))

    empty = torch.zeros((0, 3), device=dev)
    _save_cloud(
        out_dir / "human.npz",
        torch.cat(human_pts) * s if human_pts else empty,
        colors=torch.cat(human_col) if human_col else empty.to(torch.uint8),
        confidence=torch.cat(human_conf) if human_conf else empty[:, 0],
        frame_id=torch.cat(human_fid) if human_fid else empty[:, 0].to(torch.int32),
        obj_id=torch.cat(human_obj) if human_obj else empty[:, 0].to(torch.int32),
        frame_indices=frame_indices.astype(np.int32),
    )
    print(f"  scale={s:.4f} (mad {mad:.4f}) | scene {centroids.shape[0]} pts | "
          f"human {sum(int(t.shape[0]) for t in human_pts)} pts -> {out_dir}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True,
                   help="clip video / folder of frames (RGB, for per-point colour)")
    p.add_argument("--predictions", required=True, type=Path, help="clip dir, e.g. out/<stem>/")
    p.add_argument("--voxel-scene", type=float, default=0.05,
                   help="scene merge voxel size in metres (default: 0.05)")
    p.add_argument("--conf-drop-q", type=float, default=0.3,
                   help="drop this lowest-confidence fraction of VGGT depth (default: 0.3)")
    p.add_argument("--edge-rtol", type=float, default=0.04,
                   help="depth-edge (flying-pixel) filter tolerance; 0 disables (default: 0.04)")
    p.add_argument("--human-dilate", type=int, default=8,
                   help="grow the human mask by N px before subtracting it from the scene, so "
                        "imperfect mask edges don't leak human points into it (default: 8; 0 disables)")
    p.add_argument("--person-obj-ids", default=None,
                   help="comma-separated SAM3 object ids to treat as people (default: all)")
    p.add_argument("--device", default="cuda", help="cuda | cpu (default: cuda)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip if geometry/transform.npz already exists")
    fuse(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
