"""VGGT-Omega gives every frame of a clip a camera and a depth map, in one forward over the clip.

VGGT-Omega is a multi-view model: cross-frame attention sees the whole clip at once, so memory
grows with the frame count. A clip longer than ``[vggt].max_frames`` is subsampled uniformly
(``frame_indices`` keeps the original frame of each row); ``frame_chunk`` and ``token_chunk``
split the forward when even that does not fit.

Its geometry is up to one unknown scale per clip: depth and camera translation share it, the
intrinsics do not. ``estimate_scale`` makes it metric later. Intrinsics come out in pixels of the
resolution VGGT saw; ``intrinsics`` holds them normalised (``fx/W``, ``fy/H``) so they map onto
the full-resolution frames.

Writes, under the clip's directory::

    vggt/camera.npz         bvr-camera: extrinsics (T, 4, 4) camera-from-world, OpenCV;
                            intrinsics (T, 3, 3) normalised; intrinsics_px (T, 3, 3) at the VGGT
                            resolution; fov_x_deg, fov_y_deg (T,); frame_indices (T,);
                            image_height, image_width (the video's); vggt_height, vggt_width;
                            num_frames (T), num_frames_total; image_resolution, mode, checkpoint;
                            scale = "up_to_scale"
    vggt/depth/frame_<k:06d>.npz   depth, depth_conf (H, W) float16 at the VGGT resolution, for
                            original frame k (with [vggt].save_depth)
"""
from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from bvr import files, paths, run
from bvr.geometry.intrinsics import normalize_K

log = logging.getLogger(__name__)

_CHECKPOINT_FILE = "vggt_omega_1b_512.pt"


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    """Load VGGT-Omega once; return the per-clip camera and depth estimate."""
    from huggingface_hub import hf_hub_download
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    settings = stage.config["vggt"]
    checkpoint = settings["checkpoint"]
    weights = (checkpoint if Path(checkpoint).is_file()
               else hf_hub_download(repo_id=checkpoint, filename=_CHECKPOINT_FILE))
    model = VGGTOmega(enable_alignment=False).eval()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model = model.to(stage.device)
    model.aggregator.frame_chunk = settings["frame_chunk"] or None
    model.aggregator.token_chunk = settings["token_chunk"] or None

    def process(clip: paths.Clip) -> None:
        with tempfile.TemporaryDirectory(prefix="vggt_") as scratch:
            frame_paths = files.extract_frames(clip.video, Path(scratch) / "frames")
            height, width = cv2.imread(str(frame_paths[0])).shape[:2]
            kept = _uniform(len(frame_paths), settings["max_frames"])
            images = load_and_preprocess_images([str(frame_paths[k]) for k in kept], mode=settings["mode"],
                                                image_resolution=settings["image_resolution"]).to(stage.device)
            with torch.inference_mode():
                prediction = model(images)
        vggt_height, vggt_width = (int(size) for size in prediction["images"].shape[-2:])
        extrinsics, intrinsics_px = encoding_to_camera(prediction["pose_enc"], (vggt_height, vggt_width))
        extrinsics = extrinsics[0].float().cpu().numpy()  # (T, 3, 4)
        intrinsics_px = intrinsics_px[0].float().cpu().numpy()
        homogeneous = np.tile(np.eye(4, dtype=np.float32), (len(kept), 1, 1))
        homogeneous[:, :3] = extrinsics
        with files.staged_directory(clip.path(paths.VGGT)) as out:
            files.save_npz(
                out / Path(paths.VGGT_CAMERA).name, "bvr-camera",
                extrinsics=homogeneous,
                intrinsics=normalize_K(intrinsics_px, vggt_height, vggt_width).astype(np.float32),
                intrinsics_px=intrinsics_px,
                fov_x_deg=np.degrees(2 * np.arctan(vggt_width / 2 / intrinsics_px[:, 0, 0])).astype(np.float32),
                fov_y_deg=np.degrees(2 * np.arctan(vggt_height / 2 / intrinsics_px[:, 1, 1])).astype(np.float32),
                frame_indices=kept.astype(np.int32), image_height=np.int32(height), image_width=np.int32(width),
                vggt_height=np.int32(vggt_height), vggt_width=np.int32(vggt_width),
                num_frames=np.int32(len(kept)), num_frames_total=np.int32(len(frame_paths)),
                image_resolution=np.int32(settings["image_resolution"]), mode=np.str_(settings["mode"]),
                checkpoint=np.str_(checkpoint), scale=np.str_("up_to_scale"))
            if settings["save_depth"]:
                depth_dir = out / Path(paths.VGGT_DEPTH).name
                depth_dir.mkdir()
                depth = prediction["depth"][0, ..., 0].float().cpu().numpy()
                confidence = prediction["depth_conf"][0].float().cpu().numpy()
                for row, frame in enumerate(kept):
                    np.savez_compressed(depth_dir / f"frame_{int(frame):06d}.npz", depth=depth[row].astype(np.float16),
                                        depth_conf=confidence[row].astype(np.float16))
        log.info("%d/%d frame(s) %dx%d -> VGGT %dx%d, mean horizontal field of view %.2f deg", len(kept),
                 len(frame_paths), width, height, vggt_width, vggt_height,
                 float(np.degrees(2 * np.arctan(vggt_width / 2 / intrinsics_px[:, 0, 0])).mean()))

    return process


def _uniform(count: int, most: int) -> np.ndarray:
    """Indices of at most ``most`` frames of ``count``, uniformly spaced, first and last kept; 0 keeps all."""
    if most <= 0 or count <= most:
        return np.arange(count)
    return np.unique(np.round(np.linspace(0, count - 1, num=most)).astype(np.int64))


if __name__ == "__main__":
    run.main(load)
