"""SAM 3 object tracker — text / bbox / points / mask prompts on images & videos.

Single image, default text::

    uv run python scripts/stages/track_sam3.py \\
        --source data/images/man_couch.jpg --save-dir out/

Single video, default text::

    uv run python scripts/stages/track_sam3.py \\
        --source data/videos/short.mp4 --save-dir out/

Folder of images — load SAM 3 weights once, run on each (benchmark mode)::

    uv run python scripts/stages/track_sam3.py \\
        --source data/prox_50/ --save-dir out/track_batch --text "person"

Mixed-type multi-object via JSON (one image, two objects, different prompt
types — tracked together as separate ``obj_id``s)::

    cat > prompts.json <<JSON
    [{"label":"person","text":"person"},
     {"label":"couch","bbox":[100,200,800,900]}]
    JSON
    uv run python scripts/stages/track_sam3.py \\
        --source img.jpg --save-dir out/ --prompts prompts.json

Per-item prompts via ``--source <json>`` manifest (each entry can carry its
own ``prompts`` field — useful for batches where every input has a different
target)::

    cat > batch.json <<JSON
    [{"source": "a.jpg", "prompts": [{"text": "person"}]},
     {"source": "b.mp4", "prompts": [{"bbox": [10, 20, 100, 200]}]}]
    JSON
    uv run python scripts/stages/track_sam3.py \\
        --source batch.json --save-dir out/

Prompt CLI flags (mutually exclusive — combine multiple via ``--prompts``)::

    --text "person, ladder"    # SAM 3 grounding (default; comma = multi-phrase)
    --bbox X1 Y1 X2 Y2         # single bounding box prompt (pixel xyxy)
    --mask path.png            # single mask file prompt
    --points X,Y,+ X,Y,+ X,Y,- # one or more points (+ positive, - negative)
    --label NAME               # display label override for single-object CLI prompts

JSON prompt schema (``--prompts`` file or per-item ``prompts`` field) — list of::

    {"label": str?, "text": str, "confidence": float?, "top_k": int?}
    {"label": str?, "bbox": [x1, y1, x2, y2]}
    {"label": str?, "select_bbox": [x1, y1, x2, y2], "text": str?, "iou_thresh": float?}
        # text-ground `text` (default "person") on frame 0, keep the detection
        # whose mask best matches select_bbox, then track by that mask (falls
        # back to a plain bbox seed when nothing clears iou_thresh)
    {"label": str?, "mask": "path/to/mask.png"}
    {"label": str?, "points": {"positive": [[x,y],...], "negative": [[x,y],...]}}

Outputs (per item, at ``<save-dir>/<source.stem>/``)::

    sam3/<obj_id:02d>.png           per-object uint8 0/255 (single-image input)
    sam3/<obj_id:02d>/frame_NNNNNN.png
                                    per-object per-frame mask (multi-frame input)
    sam3/union_mask.png             OR over all per-object masks (single-frame
                                    only; downstream erasers consume this)
    sam3/mask_overlay.png OR .mp4   multi-object mask+bbox overlay (default
                                    ON; --no-overlay to disable)
    sam3/bboxes.npz                 bboxes_per_obj (N_obj,T,4), object_ids,
                                    labels, num_objects, num_frames,
                                    video_height/width, plus legacy `bboxes`
                                    (T,4) when N_obj==1
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Repo root on sys.path so ``tools.*`` imports resolve when run as a script.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

from tools.mask_io import union_obj_masks, write_mask_png
from tools.precompute_runner import InputItem, SkipItem, add_standard_args, run_batch
from tools.source import frame_stem, resolve_source


# ------------------------------ overlay rendering -----------------------------

# Cycled per object index — first colour is the legacy single-person look.
_OVERLAY_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (230, 200, 60),    # teal/cyan
    (60, 80, 230),     # red
    (90, 200, 100),    # green
    (200, 100, 230),   # magenta
    (60, 200, 230),    # yellow
    (230, 130, 60),    # orange
)
_MASK_ALPHA = 0.40
_OUTLINE_THICKNESS = 2
_BBOX_THICKNESS = 2


def _draw_overlay_one(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: np.ndarray,
    color_bgr: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    """Render one mask + bracketed bbox onto ``frame_bgr`` (returns a copy)."""
    out = frame_bgr.copy()

    if mask.any():
        fill = np.zeros_like(out)
        fill[mask] = color_bgr
        out = cv2.addWeighted(out, 1.0, fill, _MASK_ALPHA, 0)

        m_u8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), _OUTLINE_THICKNESS + 2, cv2.LINE_AA)
        cv2.drawContours(out, contours, -1, color_bgr, _OUTLINE_THICKNESS, cv2.LINE_AA)

    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 > x1 and y2 > y1:
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)
        cl = max(12, int(0.12 * min(x2 - x1, y2 - y1)))
        t = _BBOX_THICKNESS + 1
        for cx, cy, dx, dy in (
            (x1, y1, +1, +1), (x2, y1, -1, +1),
            (x1, y2, +1, -1), (x2, y2, -1, -1),
        ):
            cv2.line(out, (cx, cy), (cx + dx * cl, cy), color_bgr, t, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy + dy * cl), color_bgr, t, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick, pad = 0.6, 1, 6
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        by2 = y1
        by1 = max(0, y1 - th - 2 * pad)
        bx1 = x1
        bx2 = min(out.shape[1], x1 + tw + 2 * pad)
        roi = out[by1:by2, bx1:bx2]
        if roi.size:
            bg_patch = np.full_like(roi, color_bgr)
            out[by1:by2, bx1:bx2] = cv2.addWeighted(roi, 0.15, bg_patch, 0.85, 0)
        cv2.putText(
            out, label, (bx1 + pad, by2 - pad),
            font, scale, (20, 20, 20), thick, cv2.LINE_AA,
        )

    return out


def _draw_overlay_multi(
    frame_bgr: np.ndarray,
    masks: list[np.ndarray],
    bboxes: list[np.ndarray],
    labels: list[str],
) -> np.ndarray:
    """Stack per-object overlays in palette order on a single frame."""
    out = frame_bgr
    for i, (m, bb, lab) in enumerate(zip(masks, bboxes, labels)):
        color = _OVERLAY_PALETTE_BGR[i % len(_OVERLAY_PALETTE_BGR)]
        out = _draw_overlay_one(out, m, bb, color, lab)
    return out


def _prepare_part_folder(frame_paths: list[Path], start: int, end: int, dest: Path) -> None:
    """Symlink ``frame_paths[start:end]`` into ``dest`` as ``0.jpg .. (end-start-1).jpg``."""
    dest.mkdir(parents=True, exist_ok=True)
    for local_idx, p in enumerate(frame_paths[start:end]):
        link = dest / f"{local_idx}.jpg"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(p.resolve())


def _mask_to_bbox_xyxy(mask: np.ndarray) -> np.ndarray:
    """Tight enclosing box of a bool mask — (4,) int32 xyxy; zeros if empty."""
    if not mask.any():
        return np.zeros(4, dtype=np.int32)
    ys, xs = np.where(mask)
    return np.array(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.int32
    )


def _bbox_iou(a, b) -> float:
    """IoU of two xyxy boxes (pixel coords). 0 when disjoint or degenerate."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) \
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return float(inter / union) if union > 0.0 else 0.0


def _load_mask_file(path: Path) -> np.ndarray:
    """Load a PNG mask (any channels) as ``(H, W)`` bool."""
    img = np.asarray(Image.open(path))
    if img.ndim == 3:
        img = img[..., 0]
    return img > 127


def _resize_mask_to(mask_bool: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize of a bool mask to ``(H, W)``."""
    H, W = target_hw
    pil = Image.fromarray((mask_bool.astype(np.uint8) * 255)).resize((W, H), Image.NEAREST)
    return np.asarray(pil) > 127


def _parse_point_arg(s: str) -> tuple[float, float, bool]:
    """Parse one ``X,Y,+|-`` token from ``--points``."""
    parts = s.split(",")
    if len(parts) != 3 or parts[2] not in ("+", "-"):
        raise argparse.ArgumentTypeError(
            f"--points expects 'X,Y,+|-' tokens, got {s!r}"
        )
    try:
        return float(parts[0]), float(parts[1]), parts[2] == "+"
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--points parse error in {s!r}: {e}")


# ------------------------------ tracker class --------------------------------


@dataclass
class TrackResult:
    num_frames: int
    height: int
    width: int
    num_parts: int
    num_objects: int
    output_dir: Path
    overlay_path: Path | None = None
    object_labels: list[str] = field(default_factory=list)


class Sam3ObjectTracker:
    """Step-protocol SAM 3 multi-instance tracker (text/bbox/points/mask prompts)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._tracker = None
        self._image_proc = None
        # CLI/JSON-derived prompt list shared across items unless an item carries
        # its own ``prompts`` via the ``--source <json>`` manifest entry.
        self._cli_prompts: list[dict] | None = self._build_cli_prompts()

    # ------- lazy model loaders -------

    def _get_tracker(self):
        if self._tracker is None:
            from sam3.model_builder import build_sam3_video_model

            print("[sam3] building video tracker")
            model = build_sam3_video_model(
                checkpoint_path=self.args.checkpoint, device=self.args.device,
            )
            model.tracker.backbone = model.detector.backbone
            self._tracker = model.tracker
        return self._tracker

    def _get_image_proc(self, confidence: float):
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if self._image_proc is None:
            if torch.cuda.is_available():
                torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            print("[sam3] building image processor (grounding)")
            model = build_sam3_image_model()
            self._image_proc = Sam3Processor(
                model, device=self.args.device, confidence_threshold=confidence,
            )
        else:
            self._image_proc.set_confidence_threshold(confidence)
        return self._image_proc

    def _release_image_proc(self):
        if self._image_proc is not None:
            del self._image_proc
            self._image_proc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------- prompt list construction -------

    def _build_cli_prompts(self) -> list[dict] | None:
        """Build a prompt list from CLI flags, or ``None`` when no explicit
        prompt is given (so per-item ``--source`` JSON entries can override
        cleanly; the default ``--text`` fallback is materialized at process
        time inside :meth:`_resolve_prompts`)."""
        a = self.args
        if a.prompts is not None:
            with open(a.prompts) as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise SystemExit(f"--prompts {a.prompts}: expected JSON list, got {type(data).__name__}")
            return data
        # --label wins; otherwise fall back to --text (default "person") so
        # callers like run_phy_sic.py can pass `--text person --bbox ...` and
        # have the bbox-prompted object labeled "person" in bboxes.npz.
        fallback_label = a.label or a.text or "object"
        if a.bbox is not None:
            return [{"label": fallback_label, "bbox": list(a.bbox)}]
        if a.mask is not None:
            return [{"label": fallback_label, "mask": str(a.mask)}]
        if a.points:
            pos = [(x, y) for (x, y, p) in a.points if p]
            neg = [(x, y) for (x, y, p) in a.points if not p]
            if not pos:
                raise SystemExit("--points needs at least one positive point (X,Y,+)")
            return [{"label": fallback_label,
                     "points": {"positive": pos, "negative": neg}}]
        return None

    def _resolve_prompts(self, item: InputItem) -> list[dict]:
        if item.prompts is not None:
            return item.prompts
        if self._cli_prompts is not None:
            return self._cli_prompts
        return [{
            "label": self.args.label or self.args.text,
            "text": self.args.text,
            "confidence": self.args.confidence,
            "top_k": self.args.top_k,
        }]

    # ------- frame-0 prompt → SAM 3 internal payload -------

    def _grounding_for_text(
        self,
        frame_path: Path,
        text: str,
        confidence: float,
        top_k: int,
        default_label: str | None,
        min_area_frac: float,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Run SAM 3 grounding for each comma-separated phrase in ``text``.

        ``min_area_frac`` drops any mask whose area is below
        ``min_area_frac * H * W`` — useful for crowd shots where SAM 3
        also picks up tiny background people (e.g. audience).
        """
        proc = self._get_image_proc(confidence)
        image = Image.open(frame_path).convert("RGB")
        state = proc.set_image(image)

        phrases = [p.strip() for p in text.split(",") if p.strip()]
        if not phrases:
            raise ValueError(f"text prompt {text!r} resolves to zero phrases")

        H, W = image.size[1], image.size[0]
        min_area_px = max(1, int(round(min_area_frac * H * W)))

        masks_out: list[np.ndarray] = []
        labels_out: list[str] = []
        dropped_small = 0
        for phrase in phrases:
            proc.reset_all_prompts(state)
            state = proc.set_text_prompt(phrase, state)
            masks = state.get("masks")
            if masks is None or len(masks) == 0:
                continue
            m = masks.squeeze(1).to(torch.bool)
            areas = m.flatten(1).sum(dim=1)
            order = torch.argsort(areas, descending=True)
            for k in order.tolist():
                a = int(areas[k].item())
                if a < min_area_px:
                    dropped_small += 1
                    continue
                masks_out.append(m[k].cpu().numpy())
                labels_out.append(default_label or phrase)
        if dropped_small:
            print(
                f"  [sam3] dropped {dropped_small} detection(s) below "
                f"min_area_frac={min_area_frac:.4f} ({min_area_px} px)"
            )
        if not masks_out:
            raise RuntimeError(
                f"no objects matched text prompt {text!r} on {frame_path} — "
                f"try lower confidence/min-area, different text, or --bbox/--mask/--points"
            )
        if top_k and top_k > 0:
            masks_out = masks_out[:top_k]
            labels_out = labels_out[:top_k]
        return masks_out, labels_out

    def _grounding_for_bbox_text(
        self,
        frame_path: Path,
        label: str,
        bbox_xyxy: list[float] | tuple[float, float, float, float],
        confidence: float,
    ) -> np.ndarray | None:
        """Combine text + bbox prompts via :class:`Sam3Processor` for one object.

        ``set_text_prompt(label)`` primes the grounding head with semantic
        context; ``add_geometric_prompt(box)`` then narrows the prediction to
        the input region. Calling box-only would silently fall back to a
        ``"visual"`` text prompt (see ``sam3_image_processor.py:138``), which
        is what the SAM2-style ``add_new_points_or_box`` video-tracker path
        also effectively does — leaving the model with no semantic prior and
        prone to bleeding into adjacent textures (e.g. a "countertop" bbox
        whose mask grows down into the cabinets below).

        Returns the union of every candidate that overlaps the input bbox,
        or ``None`` when no candidates pass the confidence threshold (caller
        should fall back to bbox-only prompt).
        """
        proc = self._get_image_proc(confidence)
        image = Image.open(frame_path).convert("RGB")
        state = proc.set_image(image)
        proc.reset_all_prompts(state)

        H, W = image.size[1], image.size[0]
        try:
            state = proc.set_text_prompt(str(label), state)
        except Exception as e:  # noqa: BLE001 — broad fallback by design
            print(f"  [sam3] set_text_prompt({label!r}) failed: {e}")
            return None

        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        w = max(1e-6, (x2 - x1) / W)
        h = max(1e-6, (y2 - y1) / H)
        state = proc.add_geometric_prompt([cx, cy, w, h], True, state)

        masks = state.get("masks")
        if masks is None or len(masks) == 0:
            return None

        masks_np = masks.squeeze(1).to(torch.bool).cpu().numpy()  # (K, H, W)

        bbox_mask = np.zeros((H, W), dtype=bool)
        bx1 = max(0, min(W, int(round(x1))))
        by1 = max(0, min(H, int(round(y1))))
        bx2 = max(0, min(W, int(round(x2))))
        by2 = max(0, min(H, int(round(y2))))
        bbox_mask[by1:by2, bx1:bx2] = True

        acc: np.ndarray | None = None
        for k in range(masks_np.shape[0]):
            m = masks_np[k]
            if not np.logical_and(m, bbox_mask).any():
                continue
            acc = m.copy() if acc is None else np.logical_or(acc, m)
        return acc

    def _grounding_select_by_bbox(
        self,
        frame_path: Path,
        text: str,
        sel_bbox: list[float],
        confidence: float,
        min_area_frac: float,
        iou_thresh: float,
    ) -> np.ndarray | None:
        """Text-ground ``text`` on frame 0, return the detection best matching
        ``sel_bbox`` — or ``None`` when nothing clears ``iou_thresh``.

        This is the robust seed: rather than shape the mask from a (possibly
        loose or noisy) input box, SAM 3 detects every ``"person"`` on the frame
        itself, and the external box (e.g. a YOLO human-of-interest track) only
        *selects* which detection to keep — by highest bbox IoU. The returned
        mask is a clean SAM 3 person silhouette, then tracked by mask. Returns
        ``None`` (caller falls back to a plain box seed) when the target isn't
        grounded — e.g. a fade-in / black seed frame.
        """
        try:
            masks, _labels = self._grounding_for_text(
                frame_path, text, confidence, top_k=0,
                default_label=None, min_area_frac=min_area_frac,
            )
        except RuntimeError:
            return None  # nothing grounded on this frame

        best_mask: np.ndarray | None = None
        best_iou = 0.0
        for m in masks:
            iou = _bbox_iou(_mask_to_bbox_xyxy(m), sel_bbox)
            if iou > best_iou:
                best_mask, best_iou = m, iou
        return best_mask if best_iou >= iou_thresh else None

    def _resolve_internal_prompts(
        self,
        prompt_list: list[dict],
        frame0: Path,
        image_size_hw: tuple[int, int],
    ) -> tuple[list[tuple[str, object]], list[str]]:
        """Convert a JSON prompt list into ``[(kind, payload), ...]`` for :meth:`_track_part`.

        ``text`` entries call SAM 3 grounding now (frame 0) and convert the
        produced masks into ``("mask", ...)`` payloads. ``bbox`` entries with
        a meaningful ``label`` also pre-resolve via the image processor —
        ``set_text_prompt(label) → add_geometric_prompt(box)`` — so the
        grounding head sees the category in addition to the spatial cue.
        Without that, the box prompt falls through to the SAM2-style
        ``add_new_points_or_box`` path which has no semantic prior and
        tends to bleed into adjacent same-texture regions.

        On bbox+text grounding failure (no candidates pass the confidence
        threshold), we fall back to the original ``("bbox", ...)`` payload.
        By the time ``_track_part`` runs, every prompt is one of
        bbox / mask / points.
        """
        H, W = image_size_hw
        internal: list[tuple[str, object]] = []
        labels: list[str] = []

        for entry in prompt_list:
            label = entry.get("label")
            # A ``select_bbox`` entry also carries a ``text`` phrase (the class to
            # ground), so it must be dispatched to its own branch below rather
            # than the plain-text-grounding branch.
            if "text" in entry and "select_bbox" not in entry:
                detected_masks, detected_labels = self._grounding_for_text(
                    frame0,
                    entry["text"],
                    entry.get("confidence", self.args.confidence),
                    entry.get("top_k", self.args.top_k),
                    default_label=label,
                    min_area_frac=entry.get(
                        "min_area_frac", self.args.min_area_frac
                    ),
                )
                for m, lab in zip(detected_masks, detected_labels):
                    if m.shape != (H, W):
                        m = _resize_mask_to(m, (H, W))
                    internal.append(("mask", torch.from_numpy(m.astype(np.float32))))
                    labels.append(lab)
            elif "bbox" in entry:
                bb = entry["bbox"]
                if len(bb) != 4:
                    raise ValueError(f"bbox entry needs 4 values (xyxy), got {bb}")
                grounded_mask: np.ndarray | None = None
                if label:
                    grounded_mask = self._grounding_for_bbox_text(
                        frame0,
                        label=str(label),
                        bbox_xyxy=bb,
                        confidence=entry.get("confidence", self.args.confidence),
                    )
                if grounded_mask is not None:
                    if grounded_mask.shape != (H, W):
                        grounded_mask = _resize_mask_to(grounded_mask, (H, W))
                    internal.append((
                        "mask",
                        torch.from_numpy(grounded_mask.astype(np.float32)),
                    ))
                    labels.append(str(label))
                else:
                    if label:
                        print(
                            f"  [sam3] bbox+text grounding returned no mask for "
                            f"{label!r} (bbox={bb}); falling back to bbox-only"
                        )
                    internal.append(("bbox", tuple(float(v) for v in bb)))
                    labels.append(label or "bbox-object")
            elif "mask" in entry:
                arr = _load_mask_file(Path(entry["mask"]))
                if arr.shape != (H, W):
                    arr = _resize_mask_to(arr, (H, W))
                internal.append(("mask", torch.from_numpy(arr.astype(np.float32))))
                labels.append(label or "mask-object")
            elif "points" in entry:
                pts = entry["points"]
                pos = [tuple(p) for p in pts.get("positive", [])]
                neg = [tuple(p) for p in pts.get("negative", [])]
                if not pos:
                    raise ValueError(f"points entry needs ≥1 positive point: {entry}")
                internal.append(("points", (pos, neg)))
                labels.append(label or "point-object")
            elif "select_bbox" in entry:
                # Text-ground `text` (default "person"), then keep the detection
                # whose mask best matches `select_bbox` (external selector, e.g. a
                # YOLO track box) and track that mask. Degrades to text+box
                # geometric grounding, then a raw box, when nothing matches.
                sel = entry["select_bbox"]
                if len(sel) != 4:
                    raise ValueError(f"select_bbox entry needs 4 values (xyxy), got {sel}")
                sel = [float(v) for v in sel]
                grounded = self._grounding_select_by_bbox(
                    frame0,
                    text=entry.get("text", "person"),
                    sel_bbox=sel,
                    confidence=entry.get("confidence", self.args.confidence),
                    min_area_frac=entry.get("min_area_frac", self.args.min_area_frac),
                    iou_thresh=entry.get("iou_thresh", 0.2),
                )
                if grounded is None:
                    # Graceful degradation: nothing matched (e.g. a fade-in seed
                    # frame). Try the geometric text+box grounding the plain
                    # "bbox" prompt uses — strictly ≥ a raw box seed — before
                    # falling back to the raw box itself.
                    grounded = self._grounding_for_bbox_text(
                        frame0, str(label or entry.get("text", "person")), sel,
                        entry.get("confidence", self.args.confidence),
                    )
                    print(
                        f"  [sam3] select_bbox: no '{entry.get('text', 'person')}' "
                        f"detection matched {sel} (IoU < thresh) — "
                        + ("using text+box grounding" if grounded is not None
                           else "falling back to raw bbox seed")
                    )
                if grounded is not None:
                    if grounded.shape != (H, W):
                        grounded = _resize_mask_to(grounded, (H, W))
                    internal.append(("mask", torch.from_numpy(grounded.astype(np.float32))))
                    labels.append(label or "person")
                else:
                    internal.append(("bbox", tuple(sel)))
                    labels.append(label or "person")
            else:
                raise ValueError(
                    f"prompt entry has no text/bbox/select_bbox/mask/points: {entry}"
                )
        return internal, labels

    # ------- single part propagation -------

    def _track_part(
        self,
        part_folder: Path,
        initial_prompts: list[tuple[str, object]],
        image_size_hw: tuple[int, int],
    ) -> tuple[list[list[np.ndarray]], list[torch.Tensor]]:
        """Propagate every object in ``initial_prompts`` through ``part_folder``.

        Returns ``(per_obj_masks, last_mask_gpu)`` where the second element
        feeds the next part's seed prompts (overlap-by-one frame).
        """
        tracker = self._get_tracker()
        state = tracker.init_state(video_path=str(part_folder))

        H, W = image_size_hw
        for obj_id, (kind, payload) in enumerate(initial_prompts, start=1):
            if kind == "bbox":
                x1, y1, x2, y2 = payload
                box = torch.tensor(
                    [x1 / W, y1 / H, x2 / W, y2 / H], dtype=torch.float32,
                )
                tracker.add_new_points_or_box(
                    state, frame_idx=0, obj_id=obj_id,
                    box=box, rel_coordinates=True,
                )
            elif kind == "mask":
                mask_t = payload if isinstance(payload, torch.Tensor) else torch.as_tensor(payload)
                tracker.add_new_mask(state, frame_idx=0, obj_id=obj_id,
                                     mask=mask_t.to(torch.float32))
            elif kind == "points":
                pos, neg = payload
                pts = torch.tensor(pos + neg, dtype=torch.float32).reshape(-1, 2)
                lbl = torch.tensor([1] * len(pos) + [0] * len(neg), dtype=torch.int32)
                tracker.add_new_points_or_box(
                    state, frame_idx=0, obj_id=obj_id,
                    points=pts, labels=lbl,
                )
            else:
                raise ValueError(f"unknown prompt kind {kind!r}")

        L = state["num_frames"]
        N_obj = len(initial_prompts)
        per_obj_masks: list[list[np.ndarray | None]] = [
            [None] * L for _ in range(N_obj)
        ]
        last_mask_gpu: list[torch.Tensor | None] = [None] * N_obj

        for frame_idx, _obj_ids, _low_res, video_res, _scores in tracker.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=L,
            reverse=False,
            propagate_preflight=True,
        ):
            # video_res: (N_obj, 1, H, W) float logits
            for k in range(N_obj):
                logits = video_res[k, 0]
                bool_mask = (logits > 0).cpu().numpy()
                per_obj_masks[k][frame_idx] = bool_mask
                if frame_idx == L - 1:
                    last_mask_gpu[k] = (logits > 0).to(torch.float32).detach()

        for k in range(N_obj):
            for i, m in enumerate(per_obj_masks[k]):
                if m is None:
                    per_obj_masks[k][i] = np.zeros((H, W), dtype=bool)
            if last_mask_gpu[k] is None:
                last_mask_gpu[k] = torch.from_numpy(per_obj_masks[k][-1].astype(np.float32))

        del state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return per_obj_masks, last_mask_gpu

    # ------- Step protocol -------

    def load(self) -> None:
        # Lazy: heavy SAM 3 weights build inside process() so a fully-skipped
        # batch never triggers the multi-GB download.
        pass

    def process(self, item: InputItem) -> None:
        out_dir = item.out_dir
        # ``--masks-subdir`` chooses where SAM 3 outputs land under out_dir.
        # Default "sam3" gives the usual ``out/<stem>/sam3/…`` tree; "." writes
        # masks/overlay/bboxes.npz straight into out_dir (the DB
        # ``features/sam3/<shard>/<scene>/…`` layout).
        subdir = getattr(self.args, "masks_subdir", "sam3") or "sam3"
        sam3_dir = out_dir if subdir in (".", "") else out_dir / subdir
        if self.args.skip_existing and (sam3_dir / "bboxes.npz").exists():
            raise SkipItem(f"bboxes.npz exists: {sam3_dir / 'bboxes.npz'}")
        out_dir.mkdir(parents=True, exist_ok=True)
        masks_root = sam3_dir   # all SAM3 outputs (masks, overlay, bboxes.npz) live here
        masks_root.mkdir(parents=True, exist_ok=True)

        prompt_list = self._resolve_prompts(item)

        with tempfile.TemporaryDirectory(prefix="sam3_track_") as tmp_root:
            tmp_root = Path(tmp_root)
            frame_paths, src_fps = resolve_source(item.source_path, tmp_root)
            N = len(frame_paths)

            img0 = cv2.imread(str(frame_paths[0]))
            if img0 is None:
                raise RuntimeError(f"cannot read frame {frame_paths[0]}")
            H, W = img0.shape[:2]

            initial_prompts, object_labels = self._resolve_internal_prompts(
                prompt_list, frame_paths[0], image_size_hw=(H, W),
            )
            N_obj = len(initial_prompts)
            # Multi-frame: one subdir per object (frames go inside).
            # Single-image: skip the subdir — masks land flat as sam3/<obj_id>.png.
            if N > 1:
                for k in range(N_obj):
                    (masks_root / f"{k:02d}").mkdir(parents=True, exist_ok=True)

            # ---- split into parts (overlap by 1 frame) ----
            L = self.args.part_length
            if L < 2:
                raise ValueError("--part-length must be >= 2 (need at least one tracking step)")
            parts: list[tuple[int, int]] = []
            start = 0
            while start < N:
                end = min(start + L, N)
                parts.append((start, end))
                if end == N:
                    break
                start = end - 1

            bbox_rows = np.zeros((N_obj, N, 4), dtype=np.int32)
            frame_indices = np.arange(N, dtype=np.int32)
            num_parts = len(parts)

            # ---- optional overlay writer ----
            single_image_overlay = self.args.overlay and N == 1 and src_fps is None
            writer: cv2.VideoWriter | None = None
            overlay_path: Path | None = None
            if self.args.overlay and not single_image_overlay:
                overlay_path = masks_root / "mask_overlay.mp4"
                fps = float(self.args.fps or src_fps or 30.0)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (W, H))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open video writer at {overlay_path}")
            elif single_image_overlay:
                overlay_path = masks_root / "mask_overlay.png"

            current_prompts = initial_prompts
            for k_part, (p_start, p_end) in enumerate(parts):
                part_dir = tmp_root / f"part_{k_part}"
                _prepare_part_folder(frame_paths, p_start, p_end, part_dir)

                per_obj_masks, last_mask_gpu = self._track_part(
                    part_dir, current_prompts, image_size_hw=(H, W),
                )

                local_skip = 1 if k_part > 0 else 0
                for local_idx in range(p_end - p_start):
                    if local_idx < local_skip:
                        continue
                    abs_idx = p_start + local_idx

                    stem = frame_stem(abs_idx, item.source_path, N)
                    for obj_id in range(N_obj):
                        m = per_obj_masks[obj_id][local_idx]
                        if N > 1:
                            png_path = masks_root / f"{obj_id:02d}" / f"{stem}.png"
                        else:
                            png_path = masks_root / f"{obj_id:02d}.png"
                        cv2.imwrite(
                            str(png_path),
                            (m.astype(np.uint8) * 255),
                            [cv2.IMWRITE_PNG_COMPRESSION, 6],
                        )
                        bbox_rows[obj_id, abs_idx] = _mask_to_bbox_xyxy(m)

                    if writer is not None or single_image_overlay:
                        frame_bgr = cv2.imread(str(frame_paths[abs_idx]))
                        if frame_bgr is None:
                            raise RuntimeError(f"cannot read frame {frame_paths[abs_idx]}")
                        overlay = _draw_overlay_multi(
                            frame_bgr,
                            [per_obj_masks[obj_id][local_idx] for obj_id in range(N_obj)],
                            [bbox_rows[obj_id, abs_idx] for obj_id in range(N_obj)],
                            object_labels,
                        )
                        if writer is not None:
                            writer.write(overlay)
                        else:
                            cv2.imwrite(str(overlay_path), overlay)

                if k_part + 1 < num_parts:
                    current_prompts = [
                        ("mask", last_mask_gpu[obj_id].cpu()) for obj_id in range(N_obj)
                    ]

                shutil.rmtree(part_dir, ignore_errors=True)

            if writer is not None:
                writer.release()

        # ---- save bboxes.npz (existing schema) ----
        save_kwargs: dict[str, np.ndarray] = dict(
            frame_indices=frame_indices,
            bboxes_per_obj=bbox_rows,
            object_ids=np.arange(N_obj, dtype=np.int32),
            labels=np.array(object_labels, dtype=object),
            num_objects=np.int32(N_obj),
            video_height=np.int32(H),
            video_width=np.int32(W),
            num_frames=np.int32(N),
        )
        if N_obj == 1:
            save_kwargs["bboxes"] = bbox_rows[0]
        np.savez(sam3_dir / "bboxes.npz", **save_kwargs)

        # ---- single-frame: write union_mask.png so downstream erasers don't
        # have to OR the per-object masks themselves. Multi-frame union semantics
        # are scenario-specific (per-frame? over time?), so defer.
        if N == 1:
            union = union_obj_masks(masks_root, frame_filename=None)
            if union is not None:
                write_mask_png(masks_root / "union_mask.png", union)

        print(
            f"  → {N_obj} object(s) ({', '.join(object_labels)}) × "
            f"{N} frame(s) {W}×{H} → {out_dir}"
            + (f" + {overlay_path.name}" if overlay_path else "")
        )

    def release(self) -> None:
        self._release_image_proc()
        if self._tracker is not None:
            del self._tracker
            self._tracker = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ------------------------------ CLI --------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track multiple objects with SAM 3 — text/bbox/points/mask prompts on images & videos.",
    )
    add_standard_args(p)

    p.add_argument(
        "--text", default="person",
        help='text prompt for SAM 3 grounding on frame 0 (default: %(default)r). '
             'Comma-separated for multi-phrase grounding ("person, ladder").',
    )
    p.add_argument(
        "--top-k", type=int, default=0,
        help="keep at most K detections per text prompt (sorted by area). 0 = all (default).",
    )
    p.add_argument(
        "--confidence", type=float, default=0.5,
        help="SAM 3 grounding confidence threshold (default: %(default)s)",
    )
    p.add_argument(
        "--min-area-frac", type=float, default=0.01,
        help="drop text-grounded masks with area < min_area_frac * H * W "
             "(default: %(default)s ~= 1%% of image; useful to skip "
             "background crowds / tiny detections).",
    )

    prompt_grp = p.add_mutually_exclusive_group()
    prompt_grp.add_argument(
        "--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
        help="single bounding box prompt in pixel xyxy on frame 0 (overrides --text).",
    )
    prompt_grp.add_argument(
        "--mask", type=Path, default=None,
        help="single mask file prompt (PNG matching frame 0 dimensions; overrides --text).",
    )
    prompt_grp.add_argument(
        "--points", nargs="+", type=_parse_point_arg, default=None,
        metavar="X,Y,+|-",
        help="point prompts for one object — '+' positive, '-' negative. "
             "Pass multiple tokens to give multi-point prompts (overrides --text).",
    )
    prompt_grp.add_argument(
        "--prompts", type=Path, default=None,
        help="JSON file with a list of multi-type prompts; see module docstring.",
    )
    p.add_argument(
        "--label", type=str, default=None,
        help="display label override for single-object CLI prompts (--bbox/--mask/--points).",
    )

    p.add_argument(
        "--part-length", type=int, default=200,
        help="frames per propagation part (default 200; lower if SAM 3 OOMs).",
    )
    p.add_argument(
        "--overlay", action=argparse.BooleanOptionalAction, default=True,
        help="render per-frame mask+bbox overlay (default: ON; pass --no-overlay to disable).",
    )
    p.add_argument(
        "--fps", type=float, default=None,
        help="output overlay-video fps (default: source video fps, or 30 for image folders).",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="path to SAM 3 .pt checkpoint (default: HuggingFace download).",
    )
    p.add_argument(
        "--masks-subdir", default="sam3",
        help='subdir under each item out_dir for SAM 3 outputs (default: "sam3", '
             'giving out/<stem>/sam3/…). Pass "." to write masks/overlay/bboxes.npz '
             'directly into out_dir — used by the DB features/sam3/<shard>/<scene>/ layout.',
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_batch(Sam3ObjectTracker(args), args)


if __name__ == "__main__":
    raise SystemExit(main())
