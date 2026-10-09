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
  4. prunes hidden scene points: VGGT's per-frame depth of the same wall differs
     by a few cm, leaving a slab; only points that are the front-most projection
     in at least one pixel of at least one frame survive (parameter-free
     z-buffer visibility over the merged cloud);
  5. drops small disconnected voxel clusters from the scene — floating depth
     fragments and off-surface blobs (e.g. an untracked bystander VGGT placed off
     the wall);
  6. estimates one camera-oriented PCA normal per surviving scene point and saves
     it with the cloud, so downstream stages (penetration SDF, contacts, viewer)
     load normals instead of recomputing them.

**Several camera views.** When the vggt stage put a synchronized rig's views in
one forward, ``camera.npz`` holds ``T`` rows tagged by ``view_ids`` and
``--source`` lists one video per view, in the same order. Every row feeds the
scene cloud — a second viewpoint sees what the first one cannot. Note the
hidden-point prune gets *weaker*, not stronger: it keeps a point that is
front-most in **any** row, so surfaces the first view would have pruned as
ghost slab survive if the second view saw them. ``transform.npz`` and ``human.npz`` stay
**view 0 only**, i.e. one row per time step: that is the contract every stage
after this one reads (the bodies are expressed in view 0's per-frame camera).

Per-clip I/O (``--source`` frames + ``--predictions out/<stem>/``)::

    reads:  --source video / folder of frames          (RGB, for per-point colour;
                                                        comma-separated per view)
            vggt/camera.npz, vggt/depth/[<view>/]frame_*.npz  ([vggt] save_depth)
            sam3/<obj>/frame_*.png                     (view 0)
            sam3/<view>/<obj>/frame_*.png              (the other views)
    writes: geometry/transform.npz   extrinsics + original-res K (scale=1, metric=False,
                                      static_camera + the image motion it came from)
            geometry/scene.npz        merged background points (+ colour + normals)
            geometry/human.npz        per-frame human points (frame_id, obj_id, colour)

Standalone CLI::

    python bvr/stages/fuse_geometry.py --source video.mp4 --predictions out/sport_climb/
    python bvr/stages/fuse_geometry.py --source cam_left.mp4,cam_right.mp4 \\
        --predictions out/campus_board/
"""
from __future__ import annotations

import argparse
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from bvr import determinism
from bvr.paths import REPO_ROOT
from bvr.config import load_toml, section
from bvr.geometry.backproject import backproject
from bvr.geometry.camera_motion import image_motion, is_static
from bvr.geometry.clean import clean_depth
from bvr.geometry.intrinsics import denormalize_K
from bvr.geometry.normals import estimate_point_normals
from bvr.geometry.voxel import group_reduce, large_component_mask, voxel_downsample
from bvr.mask_io import read_binary_mask
from bvr.npz_io import atomic_savez
from bvr.source import resolve_source


@dataclass(frozen=True)
class _View:
    """One camera of the clip: where its frames, depth, masks and people are."""

    source: Path
    depth_dir: Path
    mask_dir: Path
    objs: list[int]          # every tracked person — all of them are carved out of the scene
    climber_ids: list[int]   # the reconstructed climbers — only these enter the human cloud


def _discover_objs(sam3_dir: Path, requested: list[int] | None) -> list[int]:
    found = sorted(int(d.name) for d in sam3_dir.iterdir() if d.is_dir() and d.name.isdigit())
    return [o for o in found if o in requested] if requested else found


def _climber_ids(pred: Path, objs: list[int]) -> list[int]:
    """Tracked people that got a reconstructed body (``sam3d/smplx_params.npz``).

    A tracked bystander is excluded from the human cloud but still carved out of
    the scene. Falls back to every tracked person when the smplx stage has not
    run (standalone fuse).
    """
    smplx_path = pred / "sam3d" / "smplx_params.npz"
    if not smplx_path.exists():
        return objs
    ids = np.load(smplx_path, allow_pickle=True)["object_ids"].ravel()
    return [int(o) for o in ids if int(o) in objs]


def _view_rows(cam) -> tuple[np.ndarray, list[str]]:
    """``(view_ids (T,), view_names (V,))`` of a ``camera.npz``.

    Camera files written before multi-view have neither: they are one unnamed
    view, which is exactly ``view_ids`` all-zero and one empty name.
    """
    if "view_ids" not in cam.files:
        return np.zeros(len(cam["frame_indices"]), np.int32), [""]
    return cam["view_ids"].astype(np.int32), [str(name) for name in cam["view_names"]]


def _split_sources(source: str, view_names: list[str]) -> list[Path]:
    """One source path per camera view, in ``camera.npz``'s view order.

    A single-view clip takes the whole string, commas and all — only a camera
    file that really holds several views splits on them, so an ordinary path
    containing a comma still works. With several views the names must match, or
    a reordered ``--source`` would silently pair one camera's frames and masks
    with another camera's depth and pose.
    """
    if len(view_names) == 1:
        return [Path(source)]
    paths = [Path(s.strip()) for s in source.split(",")]
    if len(paths) != len(view_names):
        raise SystemExit(f"[fuse] --source lists {len(paths)} view(s) but camera.npz has "
                         f"{len(view_names)} {view_names} — give one per view, in that order")
    wrong = [(p.stem, n) for p, n in zip(paths, view_names) if p.stem != n]
    if wrong:
        raise SystemExit(f"[fuse] --source order does not match camera.npz's views "
                         f"{view_names}: got {[p.stem for p in paths]}")
    return paths


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
    atomic_savez(path, points=points.detach().cpu().numpy().astype(np.float32),
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


def _visible_mask(points: torch.Tensor, extr: np.ndarray, K_px: np.ndarray,
                  hv: int, wv: int, *, bin_px: float, voxel: float) -> torch.Tensor:
    """Keep-mask: a point survives iff it sits within one voxel of the FRONT of
    its z-buffer bin in at least one frame.

    VGGT's per-frame depth of the same wall disagrees by a few cm, so the merged
    wall is a slab; its back side hides behind the front surface in every view.
    The z-buffer bins are ``bin_px`` pixels wide — the merge voxel's own image
    footprint — so the lateral bin matches the cloud's resolution, and the depth
    band is one ``voxel`` so a surface slanting through a bin keeps all its
    points. A genuinely occluded surface (a wall patch behind a hold) still wins
    its own bins in the frames that actually saw it, so real geometry survives.
    Both quantities derive from the merge voxel — no extra knobs.
    """
    dev = points.device
    keep = torch.zeros(points.shape[0], dtype=torch.bool, device=dev)
    E = torch.from_numpy(extr).to(dev)
    K = torch.from_numpy(K_px).to(dev)
    bin_px = max(bin_px, 1.0)   # a sub-pixel bin would be a pixel z-buffer with a huge grid
    wb = int(np.ceil(wv / bin_px))
    nbins = int(np.ceil(hv / bin_px)) * wb
    for k in range(E.shape[0]):
        Xc = points @ E[k, :3, :3].T + E[k, :3, 3]
        z = Xc[:, 2]
        u = K[k, 0, 0] * Xc[:, 0] / z + K[k, 0, 2]
        v = K[k, 1, 1] * Xc[:, 1] / z + K[k, 1, 2]
        sel = torch.where((z > 0) & (u >= 0) & (u < wv) & (v >= 0) & (v < hv))[0]
        bins = (v[sel] / bin_px).long() * wb + (u[sel] / bin_px).long()
        zmin = torch.full((nbins,), torch.inf, dtype=z.dtype, device=dev)
        zmin.scatter_reduce_(0, bins, z[sel], "amin", include_self=False)
        keep[sel[z[sel] <= zmin[bins] + voxel]] = True
    return keep


def _prep_frame(view, frame_path, idx, args, hv, wv):
    """Per-frame CPU prep (loads, cleaning, masks) — runs on a worker thread."""
    vd = np.load(view.depth_dir / f"frame_{idx:06d}.npz")
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

    is_human, obj_id = _human_masks(view.mask_dir, idx, view.objs, hv, wv)
    # Imperfect masks leak human-edge pixels into the background, so subtract
    # a *dilated* band of every tracked person from the scene; collect the
    # human from the tight mask, restricted to the reconstructed climbers.
    # Views other than 0 have no climbers, so they contribute scene points only.
    scene_mask = valid_scene & ~_dilate(is_human, args.human_dilate)
    human_mask = valid_human & is_human & np.isin(obj_id, view.climber_ids)
    med = float(np.median(depth[scene_mask])) if scene_mask.any() else None
    return depth, conf, rgb, scene_mask, human_mask, obj_id, med


def _build_geometry(args, cam, frame_indices, extr, K_px, views, view_ids):
    """Backproject every camera row, split scene/human, merge + cluster-filter the scene.

    ``view_ids (T,)`` says which of ``views`` each camera row came from. Every row
    feeds the scene; only view 0 has climbers, so only its rows feed the human
    cloud, tagged with their position among the view-0 rows (the time step).
    """
    hv, wv = int(cam["vggt_height"]), int(cam["vggt_width"])
    dev = args.device
    time_of_row = np.cumsum(view_ids == 0) - 1     # position among the view-0 rows
    scene_pts, scene_conf, scene_col, scene_depth_med = [], [], [], []
    human_pts, human_conf, human_obj, human_fid, human_col = [], [], [], [], []

    with tempfile.TemporaryDirectory(prefix="fuse_frames_") as tmp:
        # Each view decodes into its own subdir so the frames cannot collide.
        resolved = [resolve_source(view.source, Path(tmp) / str(v))
                    for v, view in enumerate(views)]
        view_frames = [paths for paths, _ in resolved]
        src_fps = resolved[0][1]
        # The CPU prep (zlib, edge filter, imread) dominates the loop, so a few
        # worker threads prefetch it in a bounded window ahead of the GPU work.
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = deque()
            todo = deque(enumerate(frame_indices))

            def _submit_next():
                if todo:
                    k, idx = todo.popleft()
                    v = int(view_ids[k])
                    pending.append((k, pool.submit(
                        _prep_frame, views[v], view_frames[v][int(idx)], int(idx),
                        args, hv, wv)))

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
                    human_fid.append(torch.full((int(sh.sum()),), int(time_of_row[k]),
                                                dtype=torch.int32, device=dev))
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

        # Remove the slab's back side: keep only points visible from some camera.
        vis = _visible_mask(centroids, extr, K_px, hv, wv,
                            bin_px=args.voxel_px, voxel=voxel)
        n_hidden = int((~vis).sum())
        centroids, counts, coords = centroids[vis], counts[vis], coords[vis]
        scene_confidence, scene_color = scene_confidence[vis], scene_color[vis]

        # Drop small disconnected clusters — floating depth fragments and off-surface
        # blobs (e.g. an untracked bystander VGGT placed off the wall).
        keep = large_component_mask(coords, args.min_scene_voxels)
        n_dropped = int((~keep).sum())
        centroids, counts = centroids[keep], counts[keep]
        scene_confidence, scene_color = scene_confidence[keep], scene_color[keep]

        # One camera-oriented PCA normal per surviving point, saved with the cloud
        # so downstream stages (penetration SDF, contacts, viewer) just load them.
        if centroids.shape[0]:
            cam_centers = torch.from_numpy(
                -np.einsum("fji,fj->fi", extr[:, :3, :3], extr[:, :3, 3])).to(centroids)
            scene_normals = estimate_point_normals(centroids, cam_centers)
        else:
            scene_normals = empty
    else:  # e.g. the person fills every frame — the human ruler can still run
        voxel, n_dropped, n_hidden = 0.0, 0, 0
        centroids, counts = empty, empty[:, 0].to(torch.int32)
        scene_confidence, scene_color = empty[:, 0], empty.to(torch.uint8)
        scene_normals = empty
    if human_pts:
        human_points, all_obj, all_fid = torch.cat(human_pts), torch.cat(human_obj), torch.cat(human_fid)
    else:
        human_points, all_obj, all_fid = empty, empty[:, 0].to(torch.int32), empty[:, 0].to(torch.int32)
    return dict(scene_points=centroids, scene_count=counts, scene_confidence=scene_confidence,
                scene_color=scene_color, scene_normals=scene_normals,
                human_points=human_points,
                human_color=torch.cat(human_col) if human_col else empty.to(torch.uint8),
                human_confidence=torch.cat(human_conf) if human_conf else empty[:, 0],
                human_frame_id=all_fid, human_obj_id=all_obj,
                n_hidden=n_hidden, n_dropped=n_dropped,
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
    view_ids, view_names = _view_rows(cam)
    sources = _split_sources(args.source, view_names)
    requested = [int(x) for x in args.person_obj_ids.split(",")] if args.person_obj_ids else None
    sam3 = pred / "sam3"
    views = []
    for v, (source, name) in enumerate(zip(sources, view_names)):
        # View 0's masks stay in sam3/ — the layout every other stage reads; an
        # extra view's tracker pass writes into sam3/<view name>/, which the
        # digit-only object discovery ignores.
        mask_dir = sam3 if v == 0 else sam3 / name
        objs = _discover_objs(mask_dir, requested)
        views.append(_View(source, pred / "vggt" / "depth" / name, mask_dir, objs,
                           _climber_ids(pred, objs) if v == 0 else []))
    built = _build_geometry(args, cam, frame_indices, extr, K_px, views, view_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    # transform.npz is the per-time-step camera track of view 0 — one row per
    # frame, row-aligned with the bodies. The extra views only fed the cloud.
    rows0 = np.flatnonzero(view_ids == 0)
    K_orig = np.stack([denormalize_K(K_norm[k], ho, wo) for k in rows0])
    fps = float(built["src_fps"]) if built["src_fps"] else 30.0
    # Locked-off shot? — how far the fused scene slides across the picture.
    motion = image_motion(extr[rows0], K_orig, wo, ho, built["scene_points"].cpu().numpy())
    static = is_static(motion, args.static_motion_max)
    atomic_savez(
        out_dir / "transform.npz",
        scale=np.float32(1.0), metric=np.bool_(False),
        static_camera=np.bool_(static), camera_motion=np.float32(motion),
        extrinsics=extr[rows0].astype(np.float32),
        intrinsics_px_orig=K_orig.astype(np.float32), intrinsics=K_norm[rows0],
        fov_x_deg=cam["fov_x_deg"][rows0], fov_y_deg=cam["fov_y_deg"][rows0],
        frame_indices=frame_indices[rows0].astype(np.int32), image_height=np.int32(ho),
        image_width=np.int32(wo), vggt_height=np.int32(hv), vggt_width=np.int32(wv),
        fps=np.float32(fps))
    # world_scale mirrors transform.npz's scale per file, so a partially
    # metricized state (crash mid-rescale) is detectable and self-healing.
    _save_cloud(out_dir / "scene.npz", built["scene_points"],
                colors=built["scene_color"], confidence=built["scene_confidence"],
                normals=built["scene_normals"],
                count=built["scene_count"].to(torch.int32),
                voxel_size=np.float32(built["voxel"]), world_scale=np.float32(1.0))
    _save_cloud(
        out_dir / "human.npz",
        built["human_points"], colors=built["human_color"],
        confidence=built["human_confidence"], frame_id=built["human_frame_id"],
        obj_id=built["human_obj_id"],
        frame_indices=frame_indices[rows0].astype(np.int32), world_scale=np.float32(1.0),
    )
    print(f"  scene {built['scene_points'].shape[0]} pts from {len(views)} view(s) "
          f"(voxel={built['voxel']:.4f}, {built['n_hidden']} hidden pruned, "
          f"{built['n_dropped']} dropped: small clusters) "
          f"| human {built['n_human']} pts (climbers {views[0].climber_ids}) "
          f"| camera {'STATIC' if static else 'moving'} (slides {motion * 100:.2f}% of "
          f"the width) -> {out_dir}\n"
          f"  world is UP-TO-SCALE — run estimate_scale next to make it metric")


def main(argv: list[str] | None = None) -> int:
    determinism.apply()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True,
                   help="clip video / folder of frames (RGB, for per-point colour); "
                        "comma-separated, one per camera view, in camera.npz's view order")
    p.add_argument("--predictions", required=True, type=Path, help="clip dir, e.g. out/<stem>/")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.toml",
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
        "static_motion_max": 0.02,
        "skip_existing": False,
    })
    args.device = section(cfg, "runtime", {"device": "cuda"}).device
    args.source, args.predictions, args.config = cli.source, cli.predictions, cli.config
    fuse(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
