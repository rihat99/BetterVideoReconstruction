"""SAM 3 tracks the people of a clip: one mask per person and frame, and their boxes.

Frame 0 seeds the tracker. By default SAM 3's grounding finds every ``[sam3].text`` object
("person") on it, largest first, and each detection becomes one tracked object. A box, a mask,
points or a prompts file (``[sam3].bbox``, ``mask``, ``points``, ``prompts``) name the objects
instead. The tracker then carries every object through the clip in parts of
``[sam3].part_length`` frames; each part starts from the last masks of the one before.

A prompts file is a JSON list with one entry per object, each with an optional ``label`` and one of
``text`` (with optional ``confidence``, ``top_k``, ``min_area_frac``), ``bbox`` (xyxy pixels),
``select_bbox`` (ground ``text``, default "person", and keep the detection whose box best matches,
with ``iou_thresh``), ``mask`` (a PNG path) or ``points`` (``{"positive": [[x, y]], "negative": []}``).

Writes, under the clip's directory::

    sam3/<object:02d>/frame_<k:06d>.png   mask of an object on frame k, 0 or 255
    sam3/bboxes.npz                       bvr-bboxes: bboxes_per_obj (P, N, 4) int32 xyxy, zero where
                                          the mask is empty; object_ids (P,), labels (P,),
                                          frame_indices (N,), num_objects, num_frames,
                                          video_height, video_width
    sam3/mask_overlay.mp4                 masks and boxes over the video, with [sam3].overlay
"""
from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from bvr import files, paths, run

log = logging.getLogger(__name__)

# Overlay colours (BGR), one per object, cycled.
_PALETTE = ((230, 200, 60), (60, 80, 230), (90, 200, 100), (200, 100, 230), (60, 200, 230), (230, 130, 60))


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    """Build SAM 3's video tracker and grounding model once; return the per-clip tracking."""
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_video_model

    settings = stage.config["sam3"]
    device = str(stage.device)
    video_model = build_sam3_video_model(checkpoint_path=settings["checkpoint"] or None, device=device)
    video_model.tracker.backbone = video_model.detector.backbone
    tracker = video_model.tracker
    # The tracker entered a process-wide bf16 autocast when it was built, so grounding runs in bf16 too.
    grounding = Sam3Processor(build_sam3_image_model(), device=device, confidence_threshold=settings["confidence"])
    prompts = _prompts(settings)

    def process(clip: paths.Clip) -> None:
        with tempfile.TemporaryDirectory(prefix="sam3_") as scratch:
            frame_paths = files.extract_frames(clip.video, Path(scratch) / "frames")
            first = np.asarray(Image.open(frame_paths[0]).convert("RGB"))
            height, width = first.shape[:2]
            seeds, labels = _seeds(grounding, settings, prompts, frame_paths[0], (height, width))
            log.info("%d object(s) on frame 0: %s", len(seeds), ", ".join(labels))
            with files.staged_directory(clip.path(paths.SAM3)) as out:
                boxes = _track(tracker, frame_paths, seeds, settings["part_length"], out, labels,
                               overlay_fps=(float(settings["fps"] or files.frame_rate(clip.video))
                                            if settings["overlay"] else None))
                files.save_npz(
                    out / Path(paths.SAM3_BBOXES).name, "bvr-bboxes",
                    bboxes_per_obj=boxes, object_ids=np.arange(len(seeds), dtype=np.int32), labels=np.array(labels),
                    frame_indices=np.arange(len(frame_paths), dtype=np.int32), num_objects=np.int32(len(seeds)),
                    num_frames=np.int32(len(frame_paths)), video_height=np.int32(height),
                    video_width=np.int32(width))
        log.info("%d object(s) x %d frame(s), %dx%d", len(seeds), len(frame_paths), width, height)

    return process


def _prompts(settings: dict) -> list[dict]:
    """The prompt entries of the config: its prompts file, else one box, mask or point set, else the text."""
    if settings["prompts"]:
        return json.loads(Path(settings["prompts"]).read_text())
    label = settings["label"] or settings["text"]
    if settings["bbox"]:
        return [{"label": label, "bbox": list(settings["bbox"])}]
    if settings["mask"]:
        return [{"label": label, "mask": settings["mask"]}]
    if settings["points"]:
        tokens = [point.split(",") for point in settings["points"]]  # "X,Y,+" or "X,Y,-"
        return [{"label": label, "points": {
            "positive": [[float(x), float(y)] for x, y, sign in tokens if sign == "+"],
            "negative": [[float(x), float(y)] for x, y, sign in tokens if sign == "-"]}}]
    return [{"label": settings["label"], "text": settings["text"]}]


def _seeds(grounding, settings: dict, prompts: list[dict], frame: Path,
           size: tuple[int, int]) -> tuple[list[tuple[str, object]], list[str]]:
    """Turn prompt entries into the tracker's frame-0 seeds ``(kind, payload)`` and one label per object.

    Text is grounded now and seeds masks. A labelled box is grounded with its label as text, so
    SAM 3 has the category as well as the place; without a detection it seeds the box itself.
    """
    image = Image.open(frame).convert("RGB")
    seeds: list[tuple[str, object]] = []
    labels: list[str] = []
    for entry in prompts:
        label = entry.get("label") or ""
        confidence = entry.get("confidence", settings["confidence"])
        min_area_frac = entry.get("min_area_frac", settings["min_area_frac"])
        if "select_bbox" in entry:
            box = [float(v) for v in entry["select_bbox"]]
            masks, _ = _ground_text(grounding, image, entry.get("text", "person"), confidence, 0, min_area_frac)
            ious = [_box_iou(_mask_box(mask), box) for mask in masks]
            mask = masks[int(np.argmax(ious))] if ious and max(ious) >= entry.get("iou_thresh", 0.2) else None
            if mask is None:  # nothing matched, e.g. a fade-in frame: ground the box, else seed it raw
                mask = _ground_box(grounding, image, label or entry.get("text", "person"), box, confidence)
                log.info("select_bbox %s matched no detection; seeding %s", box,
                         "text+box" if mask is not None else "the box")
            seeds.append(("mask", mask) if mask is not None else ("bbox", tuple(box)))
            labels.append(label or "person")
        elif "text" in entry:
            masks, phrases = _ground_text(grounding, image, entry["text"], confidence,
                                          entry.get("top_k", settings["top_k"]), min_area_frac)
            if not masks:
                raise RuntimeError(f"no object matched the text prompt {entry['text']!r} on frame 0; lower "
                                   "[sam3].confidence or min_area_frac, or prompt a box, mask or points")
            seeds += [("mask", mask) for mask in masks]
            labels += [label or phrase for phrase in phrases]
        elif "bbox" in entry:
            box = [float(v) for v in entry["bbox"]]
            mask = _ground_box(grounding, image, label, box, confidence) if label else None
            seeds.append(("mask", mask) if mask is not None else ("bbox", tuple(box)))
            labels.append(label or "bbox-object")
        elif "mask" in entry:
            mask = np.asarray(Image.open(entry["mask"]).convert("L").resize(size[::-1], Image.NEAREST)) > 127
            seeds.append(("mask", mask))
            labels.append(label or "mask-object")
        elif "points" in entry:
            positive = [tuple(point) for point in entry["points"]["positive"]]
            negative = [tuple(point) for point in entry["points"].get("negative", [])]
            if not positive:
                raise ValueError(f"a points prompt needs a positive point: {entry}")
            seeds.append(("points", (positive, negative)))
            labels.append(label or "point-object")
        else:
            raise ValueError(f"a prompt entry needs text, bbox, select_bbox, mask or points: {entry}")
    return seeds, labels


def _ground_text(grounding, image: Image.Image, text: str, confidence: float, top_k: int,
                 min_area_frac: float) -> tuple[list[np.ndarray], list[str]]:
    """Masks of every detection of each comma-separated phrase, largest first, and their phrases.

    Detections smaller than ``min_area_frac`` of the image are dropped (a crowd behind the athlete);
    ``top_k > 0`` keeps the largest ``top_k``.
    """
    grounding.set_confidence_threshold(confidence)
    state = grounding.set_image(image)
    min_area = max(1, round(min_area_frac * image.size[0] * image.size[1]))
    masks, phrases = [], []
    for phrase in (part.strip() for part in text.split(",") if part.strip()):
        grounding.reset_all_prompts(state)
        state = grounding.set_text_prompt(phrase, state)
        found = state.get("masks")
        if found is None or len(found) == 0:
            continue
        found = found.squeeze(1).to(torch.bool)
        areas = found.flatten(1).sum(dim=1)
        for k in torch.argsort(areas, descending=True).tolist():
            if int(areas[k]) >= min_area:
                masks.append(found[k].cpu().numpy())
                phrases.append(phrase)
    return (masks[:top_k], phrases[:top_k]) if top_k > 0 else (masks, phrases)


def _ground_box(grounding, image: Image.Image, label: str, box: list[float], confidence: float) -> np.ndarray | None:
    """The union of the detections of ``label`` that overlap ``box``, or ``None`` if there is none.

    The text gives the category: a box alone has no prior and bleeds into neighbouring regions of
    the same texture.
    """
    grounding.set_confidence_threshold(confidence)
    width, height = image.size
    state = grounding.set_image(image)
    grounding.reset_all_prompts(state)
    state = grounding.set_text_prompt(label, state)
    x1, y1, x2, y2 = box
    state = grounding.add_geometric_prompt(
        [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, max(1e-6, (x2 - x1) / width), max(1e-6, (y2 - y1) / height)],
        True, state)
    found = state.get("masks")
    if found is None or len(found) == 0:
        return None
    inside = np.zeros((height, width), dtype=bool)
    inside[max(0, round(y1)):max(0, round(y2)), max(0, round(x1)):max(0, round(x2))] = True
    overlapping = [mask for mask in found.squeeze(1).to(torch.bool).cpu().numpy() if (mask & inside).any()]
    return np.logical_or.reduce(overlapping) if overlapping else None


def _track(tracker, frame_paths: list[Path], seeds: list[tuple[str, object]], part_length: int, out: Path,
           labels: list[str], overlay_fps: float | None) -> np.ndarray:
    """Propagate every seed through the clip, write the masks and the overlay; return the boxes ``(P, N, 4)``.

    Parts overlap by one frame: the last masks of a part seed the next.
    """
    if part_length < 2:
        raise ValueError(f"[sam3].part_length must be at least 2, got {part_length}")
    count = len(frame_paths)
    height, width = cv2.imread(str(frame_paths[0])).shape[:2]
    boxes = np.zeros((len(seeds), count, 4), dtype=np.int32)
    for k in range(len(seeds)):
        (out / f"{k:02d}").mkdir()
    writer = None
    if overlay_fps is not None:
        writer = files.video_writer(out / "mask_overlay.mp4", overlay_fps, (width, height))
    start = 0
    with tempfile.TemporaryDirectory(prefix="sam3_part_") as scratch:
        while True:
            end = min(start + part_length, count)
            part = Path(scratch) / f"{start}"
            part.mkdir()
            for local, frame in enumerate(frame_paths[start:end]):
                (part / f"{local}.jpg").symlink_to(frame.resolve())
            masks = _propagate(tracker, part, seeds, (height, width))  # (P, end - start, H, W)
            for local in range(1 if start else 0, end - start):
                frame = start + local
                for k in range(len(seeds)):
                    cv2.imwrite(str(out / f"{k:02d}" / f"frame_{frame:06d}.png"),
                                masks[k, local].astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                    boxes[k, frame] = _mask_box(masks[k, local])
                if writer is not None:
                    image = cv2.imread(str(frame_paths[frame]))
                    writer.write(_overlay(image, masks[:, local], boxes[:, frame], labels))
            if end == count:
                break
            seeds = [("mask", masks[k, -1]) for k in range(len(seeds))]
            start = end - 1
    if writer is not None:
        writer.release()
    return boxes


def _propagate(tracker, folder: Path, seeds: list[tuple[str, object]], size: tuple[int, int]) -> np.ndarray:
    """Masks ``(P, L, H, W)`` of every seeded object on the ``L`` frames of ``folder``."""
    height, width = size
    state = tracker.init_state(video_path=str(folder))
    for object_id, (kind, payload) in enumerate(seeds, start=1):
        if kind == "mask":
            tracker.add_new_mask(state, frame_idx=0, obj_id=object_id,
                                 mask=torch.as_tensor(payload, dtype=torch.float32))
        elif kind == "bbox":
            x1, y1, x2, y2 = payload
            tracker.add_new_points_or_box(state, frame_idx=0, obj_id=object_id, rel_coordinates=True,
                                          box=torch.tensor([x1 / width, y1 / height, x2 / width, y2 / height]))
        else:
            positive, negative = payload
            tracker.add_new_points_or_box(
                state, frame_idx=0, obj_id=object_id,
                points=torch.tensor(positive + negative, dtype=torch.float32).reshape(-1, 2),
                labels=torch.tensor([1] * len(positive) + [0] * len(negative), dtype=torch.int32))
    masks = np.zeros((len(seeds), state["num_frames"], height, width), dtype=bool)
    for frame, _, _, logits, _ in tracker.propagate_in_video(state, start_frame_idx=0, reverse=False,
                                                             max_frame_num_to_track=state["num_frames"],
                                                             propagate_preflight=True):
        masks[:, frame] = (logits[:, 0] > 0).cpu().numpy()
    return masks


def _mask_box(mask: np.ndarray) -> np.ndarray:
    """The tight xyxy box ``(4,)`` int32 of a mask (right and bottom exclusive); zeros if it is empty."""
    if not mask.any():
        return np.zeros(4, dtype=np.int32)
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.int32)


def _box_iou(a, b) -> float:
    """Intersection over union of two xyxy boxes; 0 when they do not overlap."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if inter > 0 and union > 0 else 0.0


def _overlay(frame: np.ndarray, masks: np.ndarray, boxes: np.ndarray, labels: list[str]) -> np.ndarray:
    """Each object's mask, outline, box and label drawn over a BGR frame."""
    for k, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
        colour = _PALETTE[k % len(_PALETTE)]
        if mask.any():
            fill = np.zeros_like(frame)
            fill[mask] = colour
            frame = cv2.addWeighted(frame, 1.0, fill, 0.4, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.drawContours(frame, contours, -1, colour, 2, cv2.LINE_AA)
        x1, y1, x2, y2 = (int(v) for v in box)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1, cv2.LINE_AA)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            top = max(0, y1 - text_height - 12)
            cv2.rectangle(frame, (x1, top), (min(frame.shape[1], x1 + text_width + 12), y1), colour, -1)
            cv2.putText(frame, label, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


if __name__ == "__main__":
    run.main(load)
