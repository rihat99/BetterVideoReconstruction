"""Step ``fuse``: VGGT's depth maps as one scene cloud and per-frame human points, up to scale.

VGGT gives every frame a camera and a depth map that agree across the clip up to one unknown factor.
This step builds the clouds in that unscaled world; ``scale`` measures the factor against the bodies
and writes the metric files. Per frame, the depth loses its flying pixels at depth edges, and the
SAM 3 masks split it: the human points are the own pixels of every person with a reconstructed body,
tagged with the person and the frame; the scene is the rest, less a band of ``human_dilate`` pixels
around every tracked person and less its least confident ``conf_drop_q``. The scene then:

1. merges into voxels ``voxel_px`` pixel footprints wide at the median scene depth, so the size holds
   in the unscaled world. It merges twice, the second time on the grid shifted by half a voxel: a
   surface on a cell face leaves two copies after the first pass;
2. keeps the points that are in front in some frame. VGGT's depth of one wall differs by a few
   centimetres between frames, which leaves a slab whose back no camera sees;
3. drops connected clusters of fewer than ``min_scene_voxels`` voxels: floating fragments;
4. gets one normal per point, turned to the nearest camera.

A shot counts as locked off when the scene slides less than ``static_motion_max`` of the image width
across the frames (``bvr.geometry.camera_motion``).

Writes, under the clip's directory::

    geometry/fused/transform.npz   bvr-transform: extrinsics (T, 4, 4) camera-from-world;
                                   intrinsics_px_orig (T, 3, 3) in pixels of the video, intrinsics
                                   normalised; fov_x_deg, fov_y_deg, frame_indices (T,); image_width,
                                   image_height, vggt_width, vggt_height; fps; static_camera,
                                   camera_motion; scale = 1 and metric = false
    geometry/fused/scene.npz       bvr-scene: points (M, 3), colors (M, 3) uint8, confidence (M,), the
                                   largest of its voxel; normals (M, 3); count (M,) depth pixels merged;
                                   voxel_size
    geometry/fused/human.npz       bvr-human-points: points (n, 3), colors, confidence; frame_id (n,) the
                                   row of the transform, obj_id (n,) the person; frame_indices (T,)
"""
from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from bvr import files, paths, run
from bvr.geometry.backproject import backproject
from bvr.geometry.camera_motion import image_motion
from bvr.geometry.clean import clean_depth
from bvr.geometry.intrinsics import denormalize_K
from bvr.geometry.normals import estimate_point_normals
from bvr.geometry.voxel import group_reduce, large_component_mask, voxel_downsample

log = logging.getLogger(__name__)


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    return lambda clip: fuse(clip, stage.config["fuse"], stage.device)


def fuse(clip: paths.Clip, options: dict, device: torch.device) -> None:
    """Build the clouds of ``clip`` and write ``geometry/fused/``."""
    camera = files.load_npz(clip.find(paths.VGGT_CAMERA), "bvr-camera")
    tracked = files.load_npz(clip.find(paths.SAM3_BBOXES), "bvr-bboxes")["object_ids"]
    climbers = files.load_npz(clip.find(paths.SMPLX_PARAMS), "bvr-smplx")["object_ids"]
    extrinsics, intrinsics = camera["extrinsics"], camera["intrinsics_px"]
    size = int(camera["vggt_height"]), int(camera["vggt_width"])

    def prepare(index: int, image: np.ndarray) -> tuple[np.ndarray, ...]:
        """The CPU part of one frame: its depth, confidence and colours, and which pixels are scene and human."""
        maps = np.load(clip.find(paths.VGGT_DEPTH) / f"frame_{index:06d}.npz")
        depth, confidence = maps["depth"].astype(np.float32), maps["depth_conf"].astype(np.float32)
        valid = clean_depth(depth, edge_rtol=options["edge_rtol"])
        owner = np.full(size, -1, np.int32)  # the tracked person seen at each pixel; the first one where masks overlap
        for person in tracked:
            mask = files.read_mask(clip.find(paths.SAM3) / f"{person:02d}" / f"frame_{index:06d}.png")
            mask = cv2.resize(mask.astype(np.uint8), size[::-1], interpolation=cv2.INTER_NEAREST) > 0
            owner[mask & (owner < 0)] = person
        scene = valid & ~_dilate(owner >= 0, options["human_dilate"])
        if options["conf_drop_q"] > 0 and valid.any():
            scene &= confidence >= np.quantile(confidence[valid], options["conf_drop_q"])
        human = valid & np.isin(owner, climbers)
        colors = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size[::-1], interpolation=cv2.INTER_AREA)
        return depth, confidence, colors, scene, human, owner

    scene, human, medians = [], [], []
    for row, maps in enumerate(_prefetched(prepare, enumerate(_frames(clip.video, len(extrinsics))))):
        depth, confidence, colors, scene_pixels, human_pixels, owner = maps
        if scene_pixels.any():
            medians.append(float(np.median(depth[scene_pixels])))
        depth, confidence, colors, scene_pixels, human_pixels, owner = (torch.from_numpy(a).to(device) for a in maps)
        cameras = torch.from_numpy(intrinsics[row]).to(device), torch.from_numpy(extrinsics[row]).to(device)
        scene.append((backproject(depth, *cameras, mask=scene_pixels), confidence[scene_pixels], colors[scene_pixels]))
        human.append((backproject(depth, *cameras, mask=human_pixels), confidence[human_pixels], colors[human_pixels],
                      owner[human_pixels], torch.full_like(owner[human_pixels], row)))
    if not medians:
        raise ValueError(f"{clip.name}: no scene depth survives the masks and the filters")

    # The voxel is what a voxel_px-pixel patch covers at the median scene depth, so the world's scale cancels.
    voxel = options["voxel_px"] * float(np.median(medians)) / float(np.median(intrinsics[:, 0, 0]))
    points, confidence, colors, count, cells = _merge(*(torch.cat(part) for part in zip(*scene)), voxel)
    seen = _visible(points, extrinsics, intrinsics, size, bin_px=options["voxel_px"], voxel=voxel)
    kept = seen.clone()
    kept[seen] = large_component_mask(cells[seen], options["min_scene_voxels"])
    points, confidence, colors, count = points[kept], confidence[kept], colors[kept], count[kept]
    centers = torch.from_numpy(-np.einsum("tji,tj->ti", extrinsics[:, :3, :3], extrinsics[:, :3, 3])).to(points)
    normals = estimate_point_normals(points, centers, k=options["normal_k"])

    height, width = int(camera["image_height"]), int(camera["image_width"])
    intrinsics_video = denormalize_K(camera["intrinsics"], height, width)
    motion = image_motion(extrinsics, intrinsics_video, width, height, points.cpu().numpy())
    human_points, human_confidence, human_colors, people, frames = (torch.cat(part) for part in zip(*human))
    with files.staged_directory(clip.path(paths.FUSED)) as folder:
        files.save_npz(
            folder / "transform.npz", "bvr-transform", extrinsics=extrinsics.astype(np.float32),
            intrinsics_px_orig=intrinsics_video, intrinsics=camera["intrinsics"], fov_x_deg=camera["fov_x_deg"],
            fov_y_deg=camera["fov_y_deg"], frame_indices=camera["frame_indices"].astype(np.int32),
            image_width=np.int32(width),
            image_height=np.int32(height), vggt_width=np.int32(size[1]), vggt_height=np.int32(size[0]),
            fps=np.float32(files.frame_rate(clip.video)), static_camera=np.bool_(motion < options["static_motion_max"]),
            camera_motion=np.float32(motion), scale=np.float32(1.0), metric=np.bool_(False))
        files.save_npz(folder / "scene.npz", "bvr-scene", voxel_size=np.float32(voxel), **_numpy(
            points=points, colors=colors, confidence=confidence, normals=normals, count=count))
        files.save_npz(folder / "human.npz", "bvr-human-points", frame_indices=camera["frame_indices"].astype(np.int32),
                       **_numpy(points=human_points, colors=human_colors, confidence=human_confidence, frame_id=frames,
                                obj_id=people))
    log.info("scene %d points (voxel %.4f, %d hidden, %d in small clusters); human %d points of %s; camera slides "
             "%.2f %% of the width", len(points), voxel, int((~seen).sum()), int((seen & ~kept).sum()),
             len(human_points), climbers.tolist(), 100 * motion)


def _merge(points: torch.Tensor, confidence: torch.Tensor, colors: torch.Tensor,
           voxel: float) -> tuple[torch.Tensor, ...]:
    """Merge a cloud ``(N, 3)`` into voxels twice: ``(points, confidence, colors, count, cells)`` per merged point.

    The first pass takes each voxel's centroid, its largest confidence and its mean colour. The second merges those
    on the same grid shifted by half a voxel, weighted by how many points each holds. Both passes share the grid's
    origin: a second origin taken from the centroids can line the grids up again and leave the copies apart.
    ``cells (M, 3)`` are the integer voxels of the second grid, for the cluster filter.
    """
    origin = points.amin(0)
    centroids, members, count, _ = voxel_downsample(points, voxel, origin=origin)
    first = count.numel()
    confidence = group_reduce(confidence, members, first, "amax")
    colors = torch.zeros((first, 3), device=points.device).index_add_(0, members, colors.float()) / count[:, None]

    _, members, _, cells = voxel_downsample(centroids, voxel, origin=origin, shift=0.5)
    second, weight = len(cells), count.to(points.dtype)
    total = group_reduce(weight, members, second, "sum")

    def mean(values: torch.Tensor) -> torch.Tensor:
        return torch.zeros((second, 3), dtype=values.dtype, device=points.device).index_add_(
            0, members, values * weight[:, None]) / total[:, None]
    return (mean(centroids), group_reduce(confidence, members, second, "amax"),
            mean(colors).round().to(torch.uint8), total.to(torch.int32), cells)


def _visible(points: torch.Tensor, extrinsics: np.ndarray, intrinsics: np.ndarray, size: tuple[int, int], *,
             bin_px: float, voxel: float) -> torch.Tensor:
    """Which points ``(M,)`` lie within one voxel of the front of their image bin in at least one frame.

    The bins are ``bin_px`` pixels wide, the merge voxel's own footprint, so a bin holds about one point of a surface;
    the depth band of one voxel keeps a surface that slants through a bin. A wall behind a hold still wins the bins of
    the frames that saw it.
    """
    height, width = size
    bin_px = max(bin_px, 1.0)  # a smaller bin would be a z-buffer of single pixels
    columns = int(np.ceil(width / bin_px))
    bins = int(np.ceil(height / bin_px)) * columns
    keep = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    for extrinsic, intrinsic in zip(torch.from_numpy(extrinsics).to(points), torch.from_numpy(intrinsics).to(points)):
        local = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        depth = local[:, 2]
        u = intrinsic[0, 0] * local[:, 0] / depth + intrinsic[0, 2]
        v = intrinsic[1, 1] * local[:, 1] / depth + intrinsic[1, 2]
        inside = torch.where((depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height))[0]
        cell = (v[inside] / bin_px).long() * columns + (u[inside] / bin_px).long()
        front = torch.full((bins,), torch.inf, dtype=depth.dtype, device=depth.device)
        front.scatter_reduce_(0, cell, depth[inside], "amin", include_self=False)
        keep[inside[depth[inside] <= front[cell] + voxel]] = True
    return keep


def _numpy(**tensors: torch.Tensor) -> dict[str, np.ndarray]:
    return {name: tensor.cpu().numpy() for name, tensor in tensors.items()}


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """``mask (H, W)`` grown by ``radius`` pixels with an elliptical element."""
    if radius <= 0:
        return mask
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), element) > 0


def _frames(video, count: int) -> Iterator[np.ndarray]:
    """The frames of ``video``, decoded in order; raises, after the last, unless there are ``count`` of them."""
    decoded = 0
    for decoded, image in enumerate(files.frames(video), 1):
        if decoded <= count:
            yield image
    if decoded != count:
        raise ValueError(f"{paths.VGGT_CAMERA} has {count} frames, the video {decoded}")


def _prefetched(function: Callable[..., tuple], items: Iterable[tuple], ahead: int = 8) -> Iterator[tuple]:
    """``function(*item)`` for each item, in order, computed on four threads at most ``ahead`` items in advance.

    Reading and cleaning a frame is CPU work (decompression, the edge filter, mask images); this keeps it off the
    loop that sends the frames to the device.
    """
    with ThreadPoolExecutor(4) as pool:
        pending = deque()
        for item in items:
            pending.append(pool.submit(function, *item))
            if len(pending) == ahead:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


if __name__ == "__main__":
    run.main(load)
