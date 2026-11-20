"""Sapiens2-5B finds the 308 Goliath keypoints of every tracked person on every frame.

Pose is top-down: each person is cropped by the box SAM 3 tracked (``sam3/bboxes.npz``, the
objects labelled ``[sapiens].label``), the crop is forwarded with its mirror image (flip test),
and the heatmaps are decoded on the CPU. Keypoints 0..69 are SAM 3D Body's MHR70 layout.

Decoding blurs 308 heatmaps per crop on the CPU and costs about as much as the two forwards, so it
runs in a thread pool while the GPU forwards the next crops. Its cost is mostly system time: memory
handed back to the system and faulted in again on every blur. With OpenCV's pool at every core of
the host (its default) or at one thread a decode took 3.4 to 4.2 s, at ``[run].threads`` = 8 (the
cap ``bvr.run`` applies) 0.25 to 0.85 s.

Writes, under the clip's directory::

    sapiens/pose.npz          bvr-keypoints: keypoints (P, N, 308, 2) pixels, keypoint_scores
                              (P, N, 308) in [0, 1], bboxes (P, N, 4), valid_mask (P, N), object_ids (P,),
                              labels (P,), num_people, num_frames, image_height, image_width,
                              keypoint_scheme = "goliath_308"
    sapiens/overlay/pose.mp4  the skeletons over the video, with [sapiens].overlay
"""
from __future__ import annotations

import contextlib
import logging
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from bvr import files, paths, run

log = logging.getLogger(__name__)

# Sapiens2 reads its configs relative to its pose folder.
_POSE_FOLDER = paths.REPO_ROOT / "third_party/sapiens2/sapiens/pose"
_CONFIG = "configs/keypoints308/shutterstock_goliath_3po/sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py"
_WEIGHTS = ("facebook/sapiens2-pose-5b", "sapiens2_5b_pose.safetensors")
_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
_KEYPOINTS = 308


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    """Load Sapiens2-5B pose once; return the per-clip keypoint estimate."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from sapiens.engine.config import Config
    from sapiens.engine.datasets import Compose
    from sapiens.pose import models  # noqa: F401  registers the pose models with MODELS
    from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
    from sapiens.registry import MODELS

    settings = stage.config["sapiens"]
    local = Path(settings["checkpoint_root"]) / "pose" / _WEIGHTS[1] if settings["checkpoint_root"] else None
    weights = str(local) if local is not None and local.is_file() else hf_hub_download(*_WEIGHTS)
    with contextlib.chdir(_POSE_FOLDER):
        # What sapiens.pose.models.init_model does, except that the model is built on the meta device: its random
        # initialisation of the 5B weights took 50 s, only to be overwritten by the checkpoint.
        config = Config.fromfile(_CONFIG)
        config.model["backbone"].pop("init_cfg", None)
        with torch.device("meta"):
            model = MODELS.build(config.model)
        model.load_state_dict(load_file(weights), assign=True)
        model.cfg, model.data_preprocessor = config, MODELS.build(config.data_preprocessor)
        model.pipeline = Compose(config.test_pipeline)
        metainfo = parse_pose_metainfo(dict(from_file="configs/_base_/keypoints308.py"))
    codec = dict(model.cfg.codec)
    if codec.pop("type") != "UDPHeatmap":
        raise NotImplementedError(f"Sapiens2 pose codec {model.cfg.codec['type']} is not UDPHeatmap")
    codec = UDPHeatmap(**codec)
    # Cast on the CPU: on the GPU the float32 model is a 20 GB transient.
    model = model.to(dtype=_DTYPES[settings["dtype"]]).to(stage.device).eval()
    flip_indices = metainfo["flip_indices"] if model.cfg.val_cfg.get("flip_test", False) else None

    def crop(image: np.ndarray, box: np.ndarray) -> tuple[torch.Tensor, dict]:
        data = model.data_preprocessor(model.pipeline(
            dict(img=image, bbox=box[None], bbox_score=np.ones(1, np.float32))))
        return data["inputs"], data["data_samples"]["meta"]

    def decode(heatmaps: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
        keypoints, scores = codec.decode(heatmaps)  # (1, 308, 2) in the crop's input pixels
        size, centre, scale = (np.asarray(meta[key], np.float32) for key in ("input_size", "bbox_center", "bbox_scale"))
        return keypoints[0] / size * scale + centre - 0.5 * scale, scores[0]

    def process(clip: paths.Clip) -> None:
        tracked = files.load_npz(clip.find(paths.SAM3_BBOXES), "bvr-bboxes")
        people = np.flatnonzero(tracked["labels"] == settings["label"])
        if not people.size:
            raise ValueError(f"no object labelled {settings['label']!r} in {paths.SAM3_BBOXES}: "
                             f"{list(tracked['labels'])}")
        boxes = tracked["bboxes_per_obj"][people].astype(np.float32)  # (P, N, 4)
        count = boxes.shape[1]
        keypoints = np.zeros((len(people), count, _KEYPOINTS, 2), np.float32)
        scores = np.zeros((len(people), count, _KEYPOINTS), np.float32)
        valid = boxes.any(axis=-1)
        with tempfile.TemporaryDirectory(prefix="sapiens_") as scratch:
            frame_paths = files.extract_frames(clip.video, Path(scratch) / "frames")
            if len(frame_paths) != count:
                raise ValueError(f"{paths.SAM3_BBOXES} has {count} frames, the video {len(frame_paths)}")
            height, width = cv2.imread(str(frame_paths[0])).shape[:2]
            batch, pending = settings["batch_size"], []
            with ThreadPoolExecutor(8) as pool:
                # Frames go in windows of ``batch`` frames, so a forward holds about ``batch`` crops.
                for first in range(0, count, batch):
                    jobs = [(person, frame) for frame in range(first, min(count, first + batch))
                            for person in np.flatnonzero(valid[:, frame])]
                    images = {frame: cv2.imread(str(frame_paths[frame])) for _, frame in jobs}
                    crops = [crop(images[frame], boxes[person, frame]) for person, frame in jobs]
                    if not crops:
                        continue
                    inputs = torch.cat([inputs for inputs, _ in crops]).to(next(model.parameters()).dtype)
                    with torch.no_grad():
                        for start in range(0, len(jobs), batch):
                            chunk = inputs[start:start + batch]
                            heatmaps = model(chunk)
                            if flip_indices is not None:
                                heatmaps = (heatmaps + model(chunk.flip(-1)).flip(-1)[:, flip_indices]) / 2.0
                            heatmaps = heatmaps.float().cpu().numpy()
                            for k, job in enumerate(jobs[start:start + batch]):
                                pending.append((job, pool.submit(decode, heatmaps[k], crops[start + k][1])))
                for (person, frame), decoded in pending:
                    keypoints[person, frame], scores[person, frame] = decoded.result()
            with files.staged_directory(clip.path(paths.SAPIENS)) as out:
                files.save_npz(
                    out / Path(paths.SAPIENS_POSE).name, "bvr-keypoints",
                    keypoints=keypoints, keypoint_scores=scores, bboxes=boxes, valid_mask=valid,
                    object_ids=tracked["object_ids"][people].astype(np.int32),
                    labels=tracked["labels"][people].astype(str),
                    num_people=np.int32(len(people)), num_frames=np.int32(count), image_height=np.int32(height),
                    image_width=np.int32(width), keypoint_scheme=np.str_("goliath_308"))
                if settings["overlay"]:
                    (out / "overlay").mkdir()
                    _overlay(out / "overlay" / "pose.mp4", frame_paths, keypoints, scores, valid, metainfo,
                             settings["score_thresh"], float(settings["fps"] or 30.0))
        log.info("%d person(s) x %d frame(s), %d with a box", len(people), count, int(valid.sum()))

    return process


def _overlay(path: Path, frame_paths: list[Path], keypoints: np.ndarray, scores: np.ndarray, valid: np.ndarray,
             metainfo: dict, threshold: float, fps: float) -> None:
    """Draw every person's skeleton and keypoints above ``threshold`` in the model's own colours."""
    links = np.asarray(metainfo["skeleton_links"], np.int32)
    link_colours = np.asarray(metainfo["skeleton_link_colors"], np.int32)[:, ::-1]  # RGB -> BGR
    point_colours = np.asarray(metainfo["keypoint_colors"], np.int32)[:, ::-1]
    first = cv2.imread(str(frame_paths[0]))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, first.shape[1::-1])
    for frame, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path))
        for person in np.flatnonzero(valid[:, frame]):
            points, confident = keypoints[person, frame], scores[person, frame] >= threshold
            for (a, b), colour in zip(links, link_colours):
                if confident[a] and confident[b]:
                    cv2.line(image, tuple(np.round(points[a]).astype(int)), tuple(np.round(points[b]).astype(int)),
                             colour.tolist(), 2, cv2.LINE_AA)
            for point, colour in zip(points[confident], point_colours[confident]):
                cv2.circle(image, tuple(np.round(point).astype(int)), 2, colour.tolist(), -1, cv2.LINE_AA)
        writer.write(image)
    writer.release()


if __name__ == "__main__":
    run.main(load)
