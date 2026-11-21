"""VGGT-Omega gives every frame of a clip a camera and a depth map, in one forward over the clip.

VGGT-Omega is a multi-view model: cross-frame attention sees the whole clip at once, so memory
grows with the frame count. A clip longer than ``[vggt].max_frames`` stops the step, and so does a
frame whose aspect ratio the authors' loader would crop: every later step takes row ``k`` for
frame ``k`` and the camera for the whole frame.

Its geometry is up to one unknown scale per clip: depth and camera translation share it, the
intrinsics do not. ``estimate_scale`` makes it metric later. Intrinsics come out in pixels of the
resolution VGGT saw; ``intrinsics`` holds them normalised (``fx/W``, ``fy/H``) so they map onto
the full-resolution frames.

Writes, under the clip's directory::

    vggt/camera.npz         bvr-camera: extrinsics (T, 4, 4) camera-from-world, OpenCV;
                            intrinsics (T, 3, 3) normalised; intrinsics_px (T, 3, 3) at the VGGT
                            resolution; fov_x_deg, fov_y_deg (T,); frame_indices (T,) = 0..T-1;
                            image_height, image_width (the video's); vggt_height, vggt_width;
                            num_frames (T), num_frames_total; image_resolution, mode, checkpoint;
                            scale = "up_to_scale"
    vggt/depth/frame_<k:06d>.npz   depth, depth_conf (H, W) float16 at the VGGT resolution, for
                            frame k (with [vggt].save_depth)
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
# Frame height over width that VGGT-Omega takes whole; it centre-crops a frame outside this range
# (``_crop_to_supported_aspect_ratio`` in ``third_party/vggt-omega/vggt_omega/utils/load_fn.py``).
_ASPECT_RANGE = (0.5, 2.0)


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    """Load VGGT-Omega once; return the per-clip camera and depth estimate."""
    settings = stage.config["vggt"]
    forward = model(settings, stage.device)

    def process(clip: paths.Clip) -> None:
        with tempfile.TemporaryDirectory(prefix="vggt_") as scratch:
            frame_paths = files.extract_frames(clip.video, Path(scratch) / "frames")
            height, width = cv2.imread(str(frame_paths[0])).shape[:2]
            count = len(frame_paths)
            if count > settings["max_frames"]:
                raise ValueError(f"{count} frames, more than [vggt].max_frames = {settings['max_frames']}: raise the "
                                 "key (VGGT's peak memory grows by about 0.076 GB a frame) or trim the clip")
            if not _ASPECT_RANGE[0] <= height / width <= _ASPECT_RANGE[1]:
                raise ValueError(f"frames of {width}x{height}, height/width {height / width:.3f} outside "
                                 f"[{_ASPECT_RANGE[0]}, {_ASPECT_RANGE[1]}]: VGGT-Omega would centre-crop them and the "
                                 "cameras would not fit the whole frame; crop or pad the video into that range")
            prediction = forward(frame_paths)
        vggt_height, vggt_width = prediction["depth"].shape[-2:]
        intrinsics_px = prediction["intrinsics_px"]
        with files.staged_directory(clip.path(paths.VGGT)) as out:
            files.save_npz(
                out / Path(paths.VGGT_CAMERA).name, "bvr-camera",
                extrinsics=prediction["extrinsics"],
                intrinsics=normalize_K(intrinsics_px, vggt_height, vggt_width).astype(np.float32),
                intrinsics_px=intrinsics_px,
                fov_x_deg=np.degrees(2 * np.arctan(vggt_width / 2 / intrinsics_px[:, 0, 0])).astype(np.float32),
                fov_y_deg=np.degrees(2 * np.arctan(vggt_height / 2 / intrinsics_px[:, 1, 1])).astype(np.float32),
                frame_indices=np.arange(count, dtype=np.int32), image_height=np.int32(height),
                image_width=np.int32(width), vggt_height=np.int32(vggt_height), vggt_width=np.int32(vggt_width),
                num_frames=np.int32(count), num_frames_total=np.int32(count),
                image_resolution=np.int32(settings["image_resolution"]), mode=np.str_(settings["mode"]),
                checkpoint=np.str_(settings["checkpoint"]), scale=np.str_("up_to_scale"))
            if settings["save_depth"]:
                depth_dir = out / Path(paths.VGGT_DEPTH).name
                depth_dir.mkdir()
                for frame in range(count):
                    np.savez_compressed(depth_dir / f"frame_{frame:06d}.npz",
                                        depth=prediction["depth"][frame].astype(np.float16),
                                        depth_conf=prediction["depth_conf"][frame].astype(np.float16))
        log.info("%d frame(s) %dx%d -> VGGT %dx%d, mean horizontal field of view %.2f deg", count,
                 width, height, vggt_width, vggt_height,
                 float(np.degrees(2 * np.arctan(vggt_width / 2 / intrinsics_px[:, 0, 0])).mean()))

    return process


def model(settings: dict, device: torch.device) -> Callable[[list[Path]], dict[str, np.ndarray]]:
    """Load VGGT-Omega; return its forward over a list of images, which it sees as one scene.

    ``settings`` is the ``[vggt]`` table. The forward returns ``extrinsics (N, 4, 4)``, camera-from-world (OpenCV),
    ``intrinsics_px (N, 3, 3)``, ``depth`` and ``depth_conf (N, H, W)``, all on the canvas the model saw, and
    ``window (N, 4)``: where each image sits on that canvas, as top, left, height, width. The frames of a clip fill
    the canvas. A scene of still images may mix aspect ratios (a camera mounted on its side): the authors' loader
    resizes each image on its own and pads it to the common canvas with white, and the window is what of the canvas
    is the image.
    """
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    checkpoint = settings["checkpoint"]
    weights = (checkpoint if Path(checkpoint).is_file()
               else hf_hub_download(repo_id=checkpoint, filename=_CHECKPOINT_FILE))
    network = VGGTOmega(enable_alignment=False).eval()
    network.load_state_dict(torch.load(weights, map_location="cpu"))
    network = network.to(device)

    def preprocess(images: list[Path]) -> torch.Tensor:
        return load_and_preprocess_images([str(path) for path in images], mode=settings["mode"],
                                          image_resolution=settings["image_resolution"])

    def forward(images: list[Path]) -> dict[str, np.ndarray]:
        with torch.inference_mode():
            prediction = network(preprocess(images).to(device))
        height, width = (int(size) for size in prediction["images"].shape[-2:])
        extrinsics, intrinsics_px = encoding_to_camera(prediction["pose_enc"], (height, width))
        homogeneous = np.tile(np.eye(4, dtype=np.float32), (len(images), 1, 1))
        homogeneous[:, :3] = extrinsics[0].float().cpu().numpy()
        # One image of each size, loaded alone, has the shape that size takes on the canvas.
        sizes = []
        for path in images:
            with Image.open(path) as image:
                sizes.append(image.size)
        shapes = {size: preprocess([images[sizes.index(size)]]).shape[-2:] for size in set(sizes)}
        return {"extrinsics": homogeneous, "intrinsics_px": intrinsics_px[0].float().cpu().numpy(),
                "depth": prediction["depth"][0, ..., 0].float().cpu().numpy(),
                "depth_conf": prediction["depth_conf"][0].float().cpu().numpy(),
                "window": np.array([((height - h) // 2, (width - w) // 2, h, w) for h, w in map(shapes.get, sizes)],
                                   np.int32)}

    return forward


if __name__ == "__main__":
    run.main(load)
