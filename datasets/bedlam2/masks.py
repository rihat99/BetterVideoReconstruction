"""Person masks, bounding boxes and per-frame validity for BEDLAM2 ground truth.

BEDLAM2 ships the pose of every body but no segmentation, and it labels people the
renderer placed outside the camera frustum exactly like the ones in shot. Both gaps
close from the same measurement: pose each body, project its 10 475 vertices through
the frame's own camera, and rasterise every person into one z-buffered instance map
(:mod:`datasets.bedlam2.rasterize`). Occlusion between people and self-occlusion both
fall out of the depth test.

What it is not: the environment is **not** rasterised, so a body behind a pillar, a
car or a wall is still painted. 8,458 of the 18,076 scenes (46.8%) are Unreal
environments with 3D occluders; the other 53.2% are ``hdri`` domes -- a photograph
plus a flat floor -- where nothing can occlude and the mask is exact. It is also the
*naked* SMPL-X surface, so clothing, hair and shoes fall outside it. These are
body-mesh masks, not the silhouette a matting model would cut from the render.

Three things come back per person-frame:

* the **instance map** — one 8-bit PNG per *frame*, ``0`` background and ``p + 1``
  for person ``p``, so a frame costs one small file however many people are in it.
  It is an id map, not a binary mask: reading it with
  :func:`bvr.mask_io.read_binary_mask` silently returns the union of everyone, so
  a consumer wanting one person compares against ``p + 1``;
* ``bbox_full`` — the axis-aligned box of the projected surface, measured from the
  vertices and deliberately **not** clipped to the image, so how far the person
  reaches past the edge is still readable;
* ``valid`` — whether that person-frame is usable ground truth: at most
  :data:`MAX_OUT_OF_FRAME` of ``bbox_full`` outside the image, at least one visible
  pixel, and no vertex closer to the lens than the near plane. Being *hidden* is
  deliberately not part of it — the rule the user set is about the frame edge — so a
  consumer that cares compares ``mask_area`` with ``surface_area``: 0.27% of valid
  person-frames keep under a quarter of their own surface, 0.02% under a tenth. The
  near-plane clause rejects nothing on its own (every frame it catches is already
  more than a third out of shot); it is there so a garbage projection cannot slip
  through.

Vertex min/max is the exact box of the projected surface, not an approximation: a
triangle is a convex combination of its corners, so the surface's bounding box is
its vertices' bounding box.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from datasets.bedlam2.rasterize import NEAR_M, instance_map

#: Share of ``bbox_full`` that may fall outside the image before the person-frame is
#: dropped. A person half a step out of shot is still good supervision; a sliver of
#: an elbow at the edge is not.
MAX_OUT_OF_FRAME = 1.0 / 3.0

#: Frames posed at once. Cost is ``people x chunk x 10475 x 3`` floats of vertices.
POSE_CHUNK = 64

#: zlib level for the mask PNGs. Level 6 is ~25% smaller than level 1 on these
#: near-empty silhouettes and still encodes a 720p frame in about 4 ms.
PNG_LEVEL = 6

#: Threads encoding PNGs alongside the rasteriser. Encoding costs 8.7 ms of the
#: 14.8 ms a frame takes when it happens inline, and it is the GPU that waits;
#: ``cv2.imwrite`` releases the GIL, so moving it aside overlaps the two.
PNG_THREADS = 4


def _pose_vertices(body, betas: np.ndarray, q: np.ndarray, device) -> torch.Tensor:
    """Posed world vertices ``(P, B, V, 3)`` for one chunk of frames."""
    out = []
    for person in range(len(betas)):
        shape = torch.as_tensor(betas[person], device=device)[None]
        pose = torch.as_tensor(q[person], device=device)
        with torch.no_grad():
            out.append(body.forward(shape.expand(len(pose), -1), pose).vertices)
    return torch.stack(out)


def scene_masks(gt_dir: Path, *, body, device, masks_dir: Path | None,
                chunk: int = POSE_CHUNK) -> dict:
    """Rasterise one BEDLAM2 scene and measure every person-frame.

    :param gt_dir: the scene's ``features/gt/<aa>/<bb>/<scene_id>`` directory.
    :param body: a :class:`bvr.body.Body` already on ``device``.
    :param masks_dir: where to write ``NNNNNN.png``; ``None`` measures without writing.
    :returns: the arrays :func:`save_boxes` stores, keyed by npz field name. Both boxes
        are ``xyxy`` with an exclusive far corner, so ``x1 - x0`` is a width:
        ``bbox_full`` in continuous pixel coordinates and unclipped, ``bbox_visible``
        on whole pixels and necessarily inside the image.
    """
    meta = np.load(gt_dir / "smplx.npz")
    cam = np.load(gt_dir / "camera.npz")
    height, width = int(cam["image_height"]), int(cam["image_width"])
    extrinsics = torch.as_tensor(cam["extrinsics"], device=device)
    intrinsics = torch.as_tensor(cam["intrinsics_px"], device=device)
    faces = torch.as_tensor(body.faces, dtype=torch.long, device=device)
    q, betas = meta["q"], meta["betas"]
    people, frames = q.shape[:2]

    bbox_full = np.zeros((people, frames, 4), np.float32)
    bbox_visible = np.zeros((people, frames, 4), np.float32)
    mask_area = np.zeros((people, frames), np.int32)
    surface_area = np.zeros((people, frames), np.int32)
    near_clip = np.zeros((people, frames), bool)
    slivers = 0
    if masks_dir is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)
    writer = ThreadPoolExecutor(PNG_THREADS) if masks_dir is not None else None

    for start in range(0, frames, chunk):
        stop = min(start + chunk, frames)
        verts = _pose_vertices(body, betas, q[:, start:stop], device)      # (P, B, V, 3)
        camera = torch.einsum("bij,pbvj->pbvi", extrinsics[start:stop, :3, :3], verts)
        camera = camera + extrinsics[start:stop, None, :3, 3]
        depth = camera[..., 2]
        focal = torch.stack((intrinsics[start:stop, 0, 0], intrinsics[start:stop, 1, 1]), -1)
        centre = torch.stack((intrinsics[start:stop, 0, 2], intrinsics[start:stop, 1, 2]), -1)
        uv = camera[..., :2] / depth.clamp_min(1e-6).unsqueeze(-1)
        uv = uv * focal[None, :, None, :] + centre[None, :, None, :]       # (P, B, V, 2)

        front = depth > NEAR_M
        near_clip[:, start:stop] = (~front).any(-1).cpu().numpy()
        # A vertex behind the near plane projects to nonsense, so the box is measured
        # over the ones in front; the flag above already condemns the frame.
        big = torch.where(front.unsqueeze(-1), uv, torch.full_like(uv, float("inf")))
        small = torch.where(front.unsqueeze(-1), uv, torch.full_like(uv, float("-inf")))
        box = torch.cat((big.amin(2), small.amax(2)), -1)                  # (P, B, 4) xyxy
        # Nothing in front of the camera leaves +-inf corners; store an empty box
        # instead, which :func:`out_of_frame_ratio` already reads as wholly outside.
        box = torch.where(front.any(-1).unsqueeze(-1), box, torch.zeros_like(box))
        bbox_full[:, start:stop] = box.cpu().numpy()

        for i in range(stop - start):
            ids, zbuf, dropped = instance_map(uv[:, i], depth[:, i], faces,
                                              height=height, width=width)
            slivers += dropped
            painted = ids.cpu().numpy()
            if writer is not None:
                writer.submit(cv2.imwrite, str(masks_dir / f"{start + i:06d}.png"),
                              painted, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
            # Each instance's own buffer is the silhouette before the other bodies
            # cover it, so mask_area / surface_area is how much of the person survives.
            surface_area[:, start + i] = torch.isfinite(zbuf).sum((1, 2)).cpu().numpy()
            for person in range(people):
                own = painted == person + 1
                area = int(own.sum())
                mask_area[person, start + i] = area
                if area:
                    rows = np.flatnonzero(own.any(1))
                    cols = np.flatnonzero(own.any(0))
                    bbox_visible[person, start + i] = (cols[0], rows[0],
                                                       cols[-1] + 1, rows[-1] + 1)

    if writer is not None:
        writer.shutdown(wait=True)                 # every PNG on disk before we return
    out_ratio = out_of_frame_ratio(bbox_full, width=width, height=height)
    valid = (out_ratio <= MAX_OUT_OF_FRAME) & (mask_area > 0) & ~near_clip
    return {
        "bbox_full": bbox_full, "bbox_visible": bbox_visible,
        "mask_area": mask_area, "surface_area": surface_area,
        "out_of_frame_ratio": out_ratio, "near_clip": near_clip,
        # ``person_valid`` is the literal rule — a person with no usable frame is gone.
        # ``valid_frac`` is beside it because that rule is very permissive: one good
        # frame in five hundred keeps a person, and 0.7% of the ones kept are kept on
        # under 5% of their frames.
        "valid": valid, "person_valid": valid.any(1),
        "valid_frac": valid.mean(1).astype(np.float32),
        "object_ids": np.asarray(meta["object_ids"], np.int32),
        "num_people": np.int32(people), "num_frames": np.int32(frames),
        "video_width": np.int32(width), "video_height": np.int32(height),
        "max_out_of_frame": np.float32(MAX_OUT_OF_FRAME),
        "near_m": np.float32(NEAR_M), "slivers_dropped": np.int32(slivers),
    }


def out_of_frame_ratio(bbox: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Share of each box's **area** that falls outside the image ``(...,)``.

    The user's rule is "more than 1/3 of the bbox outside"; area is the reading that
    treats a corner and an edge alike. ``bbox`` is ``(..., 4)`` xyxy in pixels and may
    lie entirely off-image, in which case the ratio is 1.
    """
    x0, y0, x1, y1 = (bbox[..., i] for i in range(4))
    full = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    inner = (np.clip(np.minimum(x1, width) - np.maximum(x0, 0), 0, None)
             * np.clip(np.minimum(y1, height) - np.maximum(y0, 0), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = 1.0 - inner / full
    return np.where(full > 0, ratio, 1.0).astype(np.float32)


def save_boxes(path: Path, fields: dict) -> None:
    """Write ``gt/bboxes.npz`` — one file per sequence, every person in it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **fields)


__all__ = ["MAX_OUT_OF_FRAME", "out_of_frame_ratio", "save_boxes", "scene_masks"]
