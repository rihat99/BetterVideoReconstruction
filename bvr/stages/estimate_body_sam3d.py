"""SAM 3D Body fits one MHR body to every tracked person on every frame.

Each person is fitted on its own from the frame, its SAM 3 box and mask, and the frame's VGGT
intrinsics at the video's resolution. VGGT's geometry is up to scale, but its intrinsics are not,
and SAM 3D Body predicts each person's camera-frame translation itself, so the intrinsics are all
it needs from VGGT.

The MHR body inside SAM 3D Body is the TorchScript model of the checkpoint's own snapshot
(``assets/mhr_model.pt``): ``MOMENTUM_ENABLED`` is set before ``sam_3d_body`` is imported, so the
head does not look for pymomentum.

Writes, under the clip's directory::

    sam3d/params.npz    bvr-mhr, people stacked as (P, N, ...), NaN where a person was not fitted:
                        mhr_model_params (204), shape_params (45), pred_cam_t (3), focal_length,
                        pred_joint_coords (127, 3), pred_keypoints_3d, pred_keypoints_2d, pred_pose_raw,
                        global_rot, body_pose_params, hand_pose_params, scale_params, pred_global_rots;
                        cam_intrinsics (3, 3) of frame 0, valid_mask (P, N), object_ids (P,), num_people,
                        num_frames, video_height, video_width
    sam3d/overlay.mp4   every fitted body over the video, one colour per person, with [sam3d].overlay
"""
from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from bvr import files, paths, run
from bvr.geometry.intrinsics import denormalize_K

log = logging.getLogger(__name__)

_SAVED = ("focal_length", "pred_cam_t", "pred_keypoints_3d", "pred_keypoints_2d", "pred_pose_raw", "global_rot",
          "body_pose_params", "hand_pose_params", "shape_params", "scale_params", "pred_joint_coords",
          "pred_global_rots", "mhr_model_params")
# Mesh colours (RGB), one per person, cycled.
_COLOURS = ((0.651, 0.741, 0.859), (0.95, 0.60, 0.40), (0.60, 0.85, 0.55), (0.80, 0.65, 0.88), (0.95, 0.85, 0.45))


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    """Load SAM 3D Body once; return the per-clip body fit."""
    os.environ["MOMENTUM_ENABLED"] = "0"  # any value: the head loads the TorchScript MHR, not pymomentum's
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf
    from sam_3d_body.visualization.renderer import Renderer

    settings = stage.config["sam3d"]
    model, model_config = load_sam_3d_body_hf(settings["hf_repo"])
    with contextlib.redirect_stdout(None):  # it prints which optional parts it runs without
        estimator = SAM3DBodyEstimator(sam_3d_body_model=model.to(stage.device), model_cfg=model_config)
    # process_one_image empties the CUDA cache on every call, one per person and frame: a host sync and fresh
    # allocations each time. Nothing else uses the GPU in this process, so the cache is kept.
    torch.cuda.empty_cache = lambda: None
    chosen = [int(part) for part in settings["obj_ids"].split(",") if part.strip()]

    def process(clip: paths.Clip) -> None:
        tracked = files.load_npz(clip.find(paths.SAM3_BBOXES), "bvr-bboxes")
        height, width, count = int(tracked["video_height"]), int(tracked["video_width"]), int(tracked["num_frames"])
        object_ids = [int(k) for k in tracked["object_ids"]]
        if missing := sorted(set(chosen) - set(object_ids)):
            raise ValueError(f"[sam3d].obj_ids {missing} are not tracked; the clip has {object_ids}")
        people = chosen or object_ids
        boxes = tracked["bboxes_per_obj"][[object_ids.index(k) for k in people]]
        intrinsics = _intrinsics(files.load_npz(clip.find(paths.VGGT_CAMERA), "bvr-camera"), tracked["frame_indices"],
                                 height, width)
        masks = clip.find(paths.SAM3)
        saved: dict[str, np.ndarray] = {}
        valid = np.zeros((len(people), count), dtype=bool)
        with tempfile.TemporaryDirectory(prefix="sam3d_") as scratch:
            frame_paths = files.extract_frames(clip.video, Path(scratch) / "frames")
            if len(frame_paths) != count:
                raise ValueError(f"{paths.SAM3_BBOXES} has {count} frames, the video {len(frame_paths)}")
            # Written beside its final name and renamed at the end, so a failed clip leaves no half video.
            overlay = clip.path(paths.SAM3D_OVERLAY).with_name("overlay.partial.mp4")
            if settings["overlay"]:
                overlay.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*"mp4v"),
                                         float(settings["fps"] or files.frame_rate(clip.video)), (width, height))
            for frame, frame_path in enumerate(frame_paths):
                image = cv2.imread(str(frame_path))
                rendered = image
                for row, person in enumerate(people):
                    if not boxes[row, frame].any():
                        continue
                    mask = cv2.imread(str(masks / f"{person:02d}" / f"frame_{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
                    with contextlib.redirect_stdout(None):
                        fitted = estimator.process_one_image(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), bboxes=boxes[row, frame][None].astype(np.float32),
                            masks=mask[None], cam_int=torch.from_numpy(intrinsics[frame][None]).to(stage.device),
                            inference_type=settings["inference_type"])
                    if not fitted:
                        continue
                    for key in _SAVED:
                        value = np.asarray(fitted[0][key], dtype=np.float32)
                        saved.setdefault(key, np.full((len(people), count, *value.shape), np.nan, np.float32))
                        saved[key][row, frame] = value
                    valid[row, frame] = True
                    if settings["overlay"]:
                        renderer = Renderer(focal_length=fitted[0]["focal_length"], faces=estimator.faces)
                        rendered = (renderer(fitted[0]["pred_vertices"], fitted[0]["pred_cam_t"], rendered,
                                             mesh_base_color=_COLOURS[row % len(_COLOURS)],
                                             scene_bg_color=(1, 1, 1)) * 255).astype(np.uint8)
                if settings["overlay"]:
                    writer.write(rendered)
            files.save_npz(clip.path(paths.SAM3D_PARAMS), "bvr-mhr", **saved, cam_intrinsics=intrinsics[0],
                           valid_mask=valid, object_ids=np.array(people, dtype=np.int32),
                           num_people=np.int32(len(people)), num_frames=np.int32(count),
                           video_height=np.int32(height), video_width=np.int32(width))
            if settings["overlay"]:
                writer.release()
                overlay.replace(clip.path(paths.SAM3D_OVERLAY))
        log.info("%d person(s) x %d frame(s), %d fitted", len(people), count, int(valid.sum()))

    return process


def _intrinsics(camera: dict, frame_indices: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pixel intrinsics ``(N, 3, 3)`` at the video's resolution for every tracked frame, from VGGT's normalised ones."""
    rows = {int(frame): row for row, frame in enumerate(camera["frame_indices"])}
    if missing := [int(frame) for frame in frame_indices if int(frame) not in rows]:
        raise ValueError(f"{paths.VGGT_CAMERA} has no camera for {len(missing)} tracked frame(s), first {missing[0]}; "
                         "VGGT subsampled the clip ([vggt].max_frames)")
    return denormalize_K(camera["intrinsics"][[rows[int(frame)] for frame in frame_indices]], height, width)


if __name__ == "__main__":
    run.main(load)
