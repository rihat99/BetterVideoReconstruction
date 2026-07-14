"""Fuse VGGT geometry into scene + human point clouds (up-to-scale).

Runs right after the body / SMPL-X stages and **before** ``estimate_scale``.
VGGT-Omega gives cross-frame-consistent cameras and depth but **up to one
arbitrary per-clip scale** and at a **reduced resolution**. This stage builds
the point clouds in that native (unscaled) world; ``estimate_scale`` then
measures the metric factor against the fused human points and rescales the
saved files in place. This stage:

  1. denormalizes VGGT's intrinsics to the original frame resolution (FOV is
     VGGT's and is preserved);
  2. backprojects every cleaned VGGT depth map to world points, split into a
     static **scene** cloud and per-frame **human** points via the SAM3 masks.
     The scene subtracts a *dilated* mask of **every** tracked person, so no one —
     climber or bystander — leaks into it; the **human** cloud keeps only the
     reconstructed climbers (``sam3d/smplx_params.npz`` ``object_ids``), tagged by
     ``obj_id`` (edge filter only — no confidence gate);
  3. merges near-duplicate scene points by voxel hashing. The voxel size is
     **relative**: ``voxel_px`` pixel footprints at the median scene depth
     (``voxel_px * median_depth / fx``), so it is meaningful in the unscaled
     world. The merge runs **twice** — the second pass on a half-voxel-shifted
     grid — because a surface straddling a cell face otherwise leaves two
     near-coincident centroids on either side. The human is kept per-frame,
     unmerged (the person moves);
  4. drops small disconnected voxel clusters from the scene — floating depth
     fragments and off-surface blobs (e.g. an untracked bystander VGGT placed off
     the wall).

Per-clip I/O (``--source`` frames + ``--predictions out/<stem>/``)::

    reads:  --source video / folder of frames          (RGB, for per-point colour)
            vggt/camera.npz, vggt/depth/frame_*.npz     (needs [vggt] save_depth)
            sam3/<obj>/frame_*.png
    writes: geometry/transform.npz   extrinsics + original-res K (scale=1, metric=False)
            geometry/scene.npz        merged background points (+ colour)
            geometry/human.npz        per-frame human points (frame_id, obj_id, colour)

Standalone CLI::

    python scripts/stages/fuse_geometry.py --source video.mp4 --predictions out/sport_climb/
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch

from tools.config import load_toml, section
from tools.geometry.backproject import backproject
from tools.geometry.clean import clean_depth
from tools.geometry.intrinsics import denormalize_K
from tools.geometry.voxel import group_reduce, large_component_mask, voxel_downsample
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


def _merge_scene(points: torch.Tensor, conf: torch.Tensor, color: torch.Tensor,
                 voxel: float):
    """Two-pass voxel merge of the scene cloud.

    Pass 1 collapses each occupied voxel to its centroid (confidence: max, colour:
    mean). A surface that straddles a cell face leaves two near-coincident
    centroids on either side, so pass 2 re-merges the centroids on the SAME grid
    shifted by half a voxel, weighted by each centroid's source-point count.
    (The origin is shared across the passes on purpose: letting pass 2 re-derive
    its origin from the centroids can realign the two lattices and let twins
    survive.) Returns
    ``(points (M,3), confidence (M,), color (M,3) uint8, count (M,), coords (M,3))``
    with ``coords`` the final integer grid indices (for the cluster filter).
    """
    origin = points.amin(0)
    cent, inv, cnt, _ = voxel_downsample(points, voxel, origin=origin)
    n1 = cnt.numel()
    conf1 = group_reduce(conf, inv, n1, "amax")
    col1 = torch.zeros((n1, 3), device=points.device)
    col1.index_add_(0, inv, color.float())
    col1 /= cnt.unsqueeze(1)

    _, inv2, _, coords = voxel_downsample(cent, voxel, origin=origin, shift=0.5)
    m = coords.shape[0]
    w = cnt.to(points.dtype)
    wsum = group_reduce(w, inv2, m, "sum")
    cent2 = torch.zeros((m, 3), dtype=points.dtype, device=points.device)
    cent2.index_add_(0, inv2, cent * w.unsqueeze(1))
    cent2 /= wsum.unsqueeze(1)
    conf2 = group_reduce(conf1, inv2, m, "amax")
    col2 = torch.zeros((m, 3), device=points.device)
    col2.index_add_(0, inv2, col1 * w.unsqueeze(1))
    col2 = (col2 / wsum.unsqueeze(1)).round().to(torch.uint8)
    return cent2, conf2, col2, wsum.to(torch.int32), coords


def _prep_frame(pred, frame_path, idx, objs, climber_ids, args, hv, wv):
    """Per-frame CPU prep (loads, cleaning, masks) — runs on a worker thread."""
    vd = np.load(pred / "vggt" / "depth" / f"frame_{idx:06d}.npz")
    depth, conf = vd["depth"].astype(np.float32), vd["depth_conf"].astype(np.float32)
    # One shared edge-filter pass (the expensive half of the cleaning). The human
    # keeps every in-mask point that survives it; the scene additionally drops
    # its lowest-confidence quantile.
    valid_human = clean_depth(depth, edge_rtol=args.edge_rtol)
    valid_scene = valid_human
    if args.conf_drop_q > 0 and valid_human.any():
        valid_scene = valid_human & (conf >= np.quantile(conf[valid_human], args.conf_drop_q))

    rgb = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (wv, hv), interpolation=cv2.INTER_AREA)

    is_human, obj_id = _human_masks(pred / "sam3", idx, objs, hv, wv)
    # Imperfect masks leak human-edge pixels into the background, so subtract
    # a *dilated* band of every tracked person from the scene; collect the
    # human from the tight mask, restricted to the reconstructed climbers.
    scene_mask = valid_scene & ~_dilate(is_human, args.human_dilate)
    human_mask = valid_human & is_human & np.isin(obj_id, climber_ids)
    med = float(np.median(depth[scene_mask])) if scene_mask.any() else None
    return depth, conf, rgb, scene_mask, human_mask, obj_id, med


def _build_geometry(args, pred, cam, frame_indices, extr, K_px, objs, climber_ids):
    """Backproject every frame, split scene/human, merge + cluster-filter the scene."""
    hv, wv = int(cam["vggt_height"]), int(cam["vggt_width"])
    dev = args.device
    scene_pts, scene_conf, scene_col, scene_depth_med = [], [], [], []
    human_pts, human_conf, human_obj, human_fid, human_col = [], [], [], [], []

    with tempfile.TemporaryDirectory(prefix="fuse_frames_") as tmp:
        frame_paths, src_fps = resolve_source(Path(args.source), Path(tmp))
        # The CPU prep (zlib, edge filter, imread) dominates the loop, so a few
        # worker threads prefetch it in a bounded window ahead of the GPU work.
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = deque()
            todo = deque(enumerate(frame_indices))

            def _submit_next():
                if todo:
                    k, idx = todo.popleft()
                    pending.append((k, pool.submit(
                        _prep_frame, pred, frame_paths[int(idx)], int(idx),
                        objs, climber_ids, args, hv, wv)))

            for _ in range(8):
                _submit_next()
            while pending:
                k, fut = pending.popleft()
                depth, conf, rgb, scene_mask, human_mask, obj_id, med = fut.result()
                _submit_next()
                if med is not None:
                    scene_depth_med.append(med)

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

    empty = torch.zeros((0, 3), device=dev)
    all_scene = torch.cat(scene_pts)
    if scene_depth_med and all_scene.shape[0]:
        # Relative voxel size: what a voxel_px-pixel patch covers at the median
        # scene depth. Scale-free — the world's unknown metric factor cancels out.
        fx_med = float(np.median(K_px[:, 0, 0]))
        voxel = args.voxel_px * float(np.median(scene_depth_med)) / fx_med
        centroids, scene_confidence, scene_color, counts, coords = _merge_scene(
            all_scene, torch.cat(scene_conf), torch.cat(scene_col), voxel)

        # Drop small disconnected clusters — floating depth fragments and off-surface
        # blobs (e.g. an untracked bystander VGGT placed off the wall).
        keep = large_component_mask(coords, args.min_scene_voxels)
        n_dropped = int((~keep).sum())
        centroids, counts = centroids[keep], counts[keep]
        scene_confidence, scene_color = scene_confidence[keep], scene_color[keep]
    else:  # e.g. the person fills every frame — the human ruler can still run
        voxel, n_dropped = 0.0, 0
        centroids, counts = empty, empty[:, 0].to(torch.int32)
        scene_confidence, scene_color = empty[:, 0], empty.to(torch.uint8)
    if human_pts:
        human_points, all_obj, all_fid = torch.cat(human_pts), torch.cat(human_obj), torch.cat(human_fid)
    else:
        human_points, all_obj, all_fid = empty, empty[:, 0].to(torch.int32), empty[:, 0].to(torch.int32)
    return dict(scene_points=centroids, scene_count=counts, scene_confidence=scene_confidence,
                scene_color=scene_color, human_points=human_points,
                human_color=torch.cat(human_col) if human_col else empty.to(torch.uint8),
                human_confidence=torch.cat(human_conf) if human_conf else empty[:, 0],
                human_frame_id=all_fid, human_obj_id=all_obj, n_dropped=n_dropped,
                n_human=sum(int(t.shape[0]) for t in human_pts), src_fps=src_fps,
                voxel=voxel)


def fuse(args: argparse.Namespace) -> None:
    pred = Path(args.predictions)
    if not (pred / "vggt" / "depth").is_dir():
        raise SystemExit(f"[fuse] missing vggt/depth/ under {pred} — run the vggt stage with save_depth=true")
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
    # Every tracked person is carved out of the scene, but only the reconstructed
    # climbers (smplx_params object_ids) go into the human cloud — a tracked
    # bystander is thus excluded from both. Falls back to all objs if smplx params
    # are absent (standalone fuse without the smplx stage).
    smplx_path = pred / "sam3d" / "smplx_params.npz"
    climber_ids = objs
    if smplx_path.exists():
        ids = np.load(smplx_path, allow_pickle=True)["object_ids"].ravel()
        climber_ids = [int(o) for o in ids if int(o) in objs]
    built = _build_geometry(args, pred, cam, frame_indices, extr, K_px, objs, climber_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    K_orig = np.stack([denormalize_K(K_norm[k], ho, wo) for k in range(len(frame_indices))])
    fps = float(built["src_fps"]) if built["src_fps"] else 30.0
    np.savez(
        out_dir / "transform.npz",
        scale=np.float32(1.0), metric=np.bool_(False),
        extrinsics=extr.astype(np.float32),
        intrinsics_px_orig=K_orig.astype(np.float32), intrinsics=K_norm,
        fov_x_deg=cam["fov_x_deg"], fov_y_deg=cam["fov_y_deg"],
        frame_indices=frame_indices.astype(np.int32), image_height=np.int32(ho),
        image_width=np.int32(wo), vggt_height=np.int32(hv), vggt_width=np.int32(wv),
        fps=np.float32(fps))
    # world_scale mirrors transform.npz's scale per file, so a partially
    # metricized state (crash mid-rescale) is detectable and self-healing.
    _save_cloud(out_dir / "scene.npz", built["scene_points"],
                colors=built["scene_color"], confidence=built["scene_confidence"],
                count=built["scene_count"].to(torch.int32),
                voxel_size=np.float32(built["voxel"]), world_scale=np.float32(1.0))
    _save_cloud(
        out_dir / "human.npz",
        built["human_points"], colors=built["human_color"],
        confidence=built["human_confidence"], frame_id=built["human_frame_id"],
        obj_id=built["human_obj_id"],
        frame_indices=frame_indices.astype(np.int32), world_scale=np.float32(1.0),
    )
    print(f"  scene {built['scene_points'].shape[0]} pts (voxel={built['voxel']:.4f}, "
          f"{built['n_dropped']} dropped: small clusters) | human {built['n_human']} pts "
          f"(climbers {climber_ids}) -> {out_dir}\n"
          f"  world is UP-TO-SCALE — run estimate_scale next to make it metric")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True,
                   help="clip video / folder of frames (RGB, for per-point colour)")
    p.add_argument("--predictions", required=True, type=Path, help="clip dir, e.g. out/<stem>/")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "pipeline.toml",
                   help="pipeline TOML (default: configs/pipeline.toml)")
    cli = p.parse_args(argv)
    cfg = load_toml(cli.config)
    args = section(cfg, "fuse", {
        "voxel_px": 5.0,
        "conf_drop_q": 0.3,
        "edge_rtol": 0.04,
        "human_dilate": 8,
        "min_scene_voxels": 25,
        "person_obj_ids": "",
        "skip_existing": False,
    })
    args.device = section(cfg, "runtime", {"device": "cuda"}).device
    args.source, args.predictions, args.config = cli.source, cli.predictions, cli.config
    fuse(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
