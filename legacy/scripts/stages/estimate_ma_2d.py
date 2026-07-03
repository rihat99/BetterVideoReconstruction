"""MammaNet (MAMMA ``ma_2d``) dense 2D landmarks precompute step.

Runs the ``cuevhv/mamma`` MammaNet landmark detector for **every tracked
person** in a clip, producing **512 dense SMPL-X surface landmarks** per person
per frame — each with a **visibility** probability and a per-point
**uncertainty**. Unlike Sapiens (COCO-style sparse keypoints), these landmarks
sit on the SMPL-X surface, so they line up directly with the body model used
downstream.

Top-down + masked: it consumes the per-person bboxes/masks the ``sam3`` stage
already wrote (``sam3/<obj:02d>/frame_NNNNNN.png``) — MAMMA's own
detector/segmenter are never run. The box that drives each crop is the tight
bound of that person's mask (matching MAMMA's masked inference); the full-frame
mask feeds the model's mask encoder.

Per-item I/O::

    reads:  the source video / folder of frames / single image
            <out-dir>/sam3/bboxes.npz                 object ids + frame indices
            <out-dir>/sam3/<obj:02d>/frame_NNNNNN.png  per-person masks
    writes: <out-dir>/ma_2d/landmarks.npz   (P, N, 512, 2) landmarks + (P, N, 512)
                                            visibility + uncertainty + confidence
            <out-dir>/ma_2d/ma_2d_overlay.png|.mp4  combined overlay (all people)

MammaNet is per-frame, so on hard frames it places some landmarks confidently
but wrongly (high visibility, yet off-body or a 1-2 frame spike). ``confidence``
= visibility gated by the two priors a *tracked, masked video* gives for free —
the landmark must sit on that person's silhouette (mask membership) and move
smoothly in time (temporal spike rejection). The overlay (and downstream
weighting) use ``confidence``, not raw visibility: it draws every landmark whose
confidence > ``--vis-thresh`` (default 0.5), coloured by uncertainty (green =
confident → red = uncertain). Raw ``visibility`` is kept in the npz untouched.

Standalone::

    uv run python scripts/stages/estimate_ma_2d.py \\
        --source data/sport_climb.mp4 --save-dir out/

The checkpoint defaults to ``third_party/mamma/data/weights/ma_2d/`` (MPI
registration required — see INSTALL.md); override with ``--checkpoint``.
"""
from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from pathlib import Path

# scripts/stages -> scripts -> repo root
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.precompute_runner import (  # noqa: E402
    InputItem,
    SkipItem,
    add_standard_args,
    run_batch,
)
from tools.source import resolve_source  # noqa: E402
from tools.adapters.mamma import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    NUM_LANDMARKS,
    bbox_from_mask,
    forward_landmarks,
    load_mammanet,
)

# Skip a person on a frame whose mask covers less than this fraction of the
# image (absent / too small to be reliable).
_MASK_AREA_MIN_FRAC = 0.001

# ---- Per-point confidence -------------------------------------------------
# MammaNet runs per-frame, so on hard frames (small / occluded person) it places
# a contiguous occluded body-part group *confidently but wrongly* — high
# visibility, yet off-body or a 1-2 frame spike. The model's own visibility
# can't catch these (the bad points score ~0.85). But a *tracked, masked video*
# gives two free priors that do: a real landmark must (a) sit on that person's
# silhouette and (b) move smoothly in time. ``confidence`` folds both into the
# model visibility; the overlay and downstream weighting use it instead of raw
# visibility. (Tuned constants, not CLI flags, to keep the stage's surface small.)
_TEMPORAL_WIN = 2          # ± frames used to estimate each landmark's robust position
_TELEPORT_FRAC_LO = 0.10   # deviation (frac of body diag) still fully trusted
_TELEPORT_FRAC_HI = 0.30   # deviation fully rejected (linear ramp between lo and hi)
_MASK_DILATE_FRAC = 0.05   # person-mask dilation (frac of body diag) for membership


def _body_diag(bboxes: np.ndarray) -> np.ndarray:
    """Body scale = xyxy-bbox diagonal (px), reducing the trailing 4 to a scalar."""
    bw = bboxes[..., 2] - bboxes[..., 0]
    bh = bboxes[..., 3] - bboxes[..., 1]
    return np.sqrt(bw * bw + bh * bh)


def _mask_membership(mask: np.ndarray, pts_xy: np.ndarray, dil_px: float) -> np.ndarray:
    """``(512,)`` of 1.0 where a landmark lands inside the (dilated) mask, else 0.0.

    The dilation gives margin for surface landmarks that legitimately sit just
    outside a tight segmentation edge; points that fly well off the person (the
    visually jarring flicker) land outside even when dilated.
    """
    mb = (mask > 0).astype(np.uint8)
    r = int(round(dil_px))
    if r >= 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mb = cv2.dilate(mb, k)
    H, W = mb.shape
    xs = np.clip(np.round(pts_xy[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(pts_xy[:, 1]).astype(int), 0, H - 1)
    return (mb[ys, xs] > 0).astype(np.float32)


def _temporal_consistency(
    landmarks: np.ndarray,   # (P, N, 512, 2)
    valid: np.ndarray,       # (P, N) bool
    diag: np.ndarray,        # (P, N) body diagonal px
    *,
    win: int,
    frac_lo: float,
    frac_hi: float,
) -> np.ndarray:
    """Per-(person, frame, landmark) temporal-consistency weight in ``[0, 1]``.

    Each valid landmark is compared to the component-wise median of the *same*
    landmark over its valid neighbours within ±``win`` frames. The median of
    symmetric neighbours tracks real (smooth) motion but ignores a 1-2 frame
    outlier, so a spike lands far from it: deviation beyond ``frac_hi``·body-diag
    is rejected (0), below ``frac_lo``·body-diag is trusted (1), linear between.
    Frames with <2 neighbours can't be judged and stay trusted (1).
    """
    P, N, J, _ = landmarks.shape
    tc = np.ones((P, N, J), np.float32)
    for p in range(P):
        vf = np.flatnonzero(valid[p])
        for f in vf:
            nb = vf[(vf >= f - win) & (vf <= f + win) & (vf != f)]
            if nb.size < 2:
                continue
            center = np.median(landmarks[p, nb], axis=0)            # (J, 2)
            dev = np.linalg.norm(landmarks[p, f] - center, axis=1)  # (J,)
            d = max(float(diag[p, f]), 1.0)
            lo_px, hi_px = frac_lo * d, frac_hi * d
            tc[p, f] = np.clip((hi_px - dev) / max(hi_px - lo_px, 1e-6), 0.0, 1.0)
    return tc


# -----------------------------------------------------------------------------
# Overlay
# -----------------------------------------------------------------------------

def _draw_overlay(
    bgr: np.ndarray,
    landmarks_per_person: list[np.ndarray],   # each (512, 2)
    conf_per_person: list[np.ndarray],        # each (512,)
    unc_per_person: list[np.ndarray],         # each (512,)
    *,
    conf_thresh: float,
    radius: int = 2,
) -> np.ndarray:
    """Draw confident landmarks coloured by uncertainty (green=confident→red=uncertain)."""
    canvas = bgr.copy()
    H, W = canvas.shape[:2]
    for lm, conf, unc in zip(landmarks_per_person, conf_per_person, unc_per_person):
        for j in range(NUM_LANDMARKS):
            if conf[j] <= conf_thresh:
                continue
            x, y = int(round(lm[j, 0])), int(round(lm[j, 1]))
            if not (0 <= x < W and 0 <= y < H):
                continue
            u = float(unc[j])
            color = (0, int(round(255 * (1.0 - u))), int(round(255 * u)))  # BGR
            cv2.circle(canvas, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


# -----------------------------------------------------------------------------
# Step
# -----------------------------------------------------------------------------

class Ma2dStep:
    """Run MammaNet across a clip's tracked people, frame-major."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._model = None
        self._cfg = None

    def load(self) -> None:
        self._model, self._cfg = load_mammanet(
            config_path=self.args.mamma_config,
            checkpoint=self.args.checkpoint,
            device=self.args.device,
        )

    def release(self) -> None:
        self._model = self._cfg = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- per-item entry --------------------------------------------------

    def process(self, item: InputItem) -> None:
        out_dir = item.out_dir
        ma_dir = out_dir / "ma_2d"
        out_npz = ma_dir / "landmarks.npz"
        if self.args.skip_existing and out_npz.exists():
            raise SkipItem(f"{out_npz} exists")

        # ---- tracked people (object ids, labels, frame indices) ----
        bp = out_dir / "sam3" / "bboxes.npz"
        if not bp.exists():
            raise SkipItem(f"missing {bp} — ma_2d needs sam3 bboxes + masks")
        bd = np.load(bp, allow_pickle=True)
        all_labels = list(bd["labels"])
        all_ids = np.asarray(bd["object_ids"], dtype=np.int32)
        n_frames = int(bd["num_frames"])
        frame_indices = np.asarray(bd["frame_indices"], dtype=np.int64)

        keep = np.array([lab == self.args.label for lab in all_labels], dtype=bool)
        if self.args.obj_ids is not None:
            want = set(self.args.obj_ids)
            keep &= np.array([int(i) in want for i in all_ids], dtype=bool)
        if not keep.any():
            raise SkipItem(
                f"no '{self.args.label}' objects to run in {bp.name} "
                f"(labels={all_labels}, obj_ids={all_ids.tolist()})"
            )
        obj_ids = all_ids[keep]
        labels = np.array([all_labels[i] for i in np.flatnonzero(keep)], dtype="U16")
        P = int(obj_ids.shape[0])

        ma_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="ma2d_frames_") as _tmp:
            frame_paths, _ = resolve_source(item.source_path, Path(_tmp))
            N = len(frame_paths)
            if N == 0:
                raise SkipItem(f"no frames resolved from {item.source_path}")
            if N != n_frames:
                raise RuntimeError(
                    f"frame count mismatch: sam3 bboxes={n_frames}, frames={N}"
                )
            first = cv2.imread(str(frame_paths[0]))
            if first is None:
                raise SkipItem(f"failed to read frame {frame_paths[0]}")
            H, W = first.shape[:2]

            landmarks = np.zeros((P, N, NUM_LANDMARKS, 2), dtype=np.float32)
            visibility = np.zeros((P, N, NUM_LANDMARKS), dtype=np.float32)
            uncertainty = np.zeros((P, N, NUM_LANDMARKS), dtype=np.float32)
            in_mask = np.zeros((P, N, NUM_LANDMARKS), dtype=np.float32)
            bboxes = np.zeros((P, N, 4), dtype=np.float32)
            valid = np.zeros((P, N), dtype=bool)

            # ---- inference (frame-major); also record per-point mask membership ----
            min_area = _MASK_AREA_MIN_FRAC * H * W
            for f in range(N):
                bgr = first if f == 0 else cv2.imread(str(frame_paths[f]))
                if bgr is None:
                    raise SkipItem(f"failed to read frame {frame_paths[f]}")
                fi = int(frame_indices[f])
                for p in range(P):
                    mpath = out_dir / "sam3" / f"{int(obj_ids[p]):02d}" / f"frame_{fi:06d}.png"
                    mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
                    if mask is None or int((mask > 0).sum()) < min_area:
                        continue
                    box = bbox_from_mask(mask)
                    if box is None:
                        continue
                    lm, vis, unc = forward_landmarks(
                        self._model, self._cfg, bgr, mask, box[None],
                        device=self.args.device,
                    )
                    landmarks[p, f] = lm[0]
                    visibility[p, f] = vis[0]
                    uncertainty[p, f] = unc[0]
                    bboxes[p, f] = box
                    valid[p, f] = True
                    diag = float(np.hypot(box[2] - box[0], box[3] - box[1]))
                    in_mask[p, f] = _mask_membership(mask, lm[0], _MASK_DILATE_FRAC * diag)

            # ---- per-point confidence = model visibility gated by the two priors a
            #      tracked, masked *video* gives us: on-silhouette (mask) + smooth-in-time.
            temporal = _temporal_consistency(
                landmarks, valid, _body_diag(bboxes),
                win=_TEMPORAL_WIN, frac_lo=_TELEPORT_FRAC_LO, frac_hi=_TELEPORT_FRAC_HI,
            )
            confidence = (visibility * temporal * in_mask).astype(np.float32)
            confidence[~valid] = 0.0

            # ---- overlay (gated on confidence, so it runs after the temporal pass) ----
            ov_path = None
            if self.args.overlay:
                def _views(f):
                    sel = [p for p in range(P) if valid[p, f]]
                    return ([landmarks[p, f] for p in sel],
                            [confidence[p, f] for p in sel],
                            [uncertainty[p, f] for p in sel])
                if N > 1:
                    ov_path = ma_dir / "ma_2d_overlay.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(ov_path), fourcc, float(self.args.fps or 30.0), (W, H)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open video writer at {ov_path}")
                    try:
                        for f in range(N):
                            bgr = first if f == 0 else cv2.imread(str(frame_paths[f]))
                            ov = _draw_overlay(bgr, *_views(f), conf_thresh=self.args.vis_thresh)
                            writer.write(ov)
                    finally:
                        writer.release()
                else:
                    ov_path = ma_dir / "ma_2d_overlay.png"
                    ov = _draw_overlay(first, *_views(0), conf_thresh=self.args.vis_thresh)
                    cv2.imwrite(str(ov_path), ov)

        np.savez_compressed(
            out_npz,
            landmarks=landmarks,
            visibility=visibility,
            uncertainty=uncertainty,
            temporal_consistency=temporal,
            confidence=confidence,
            bboxes=bboxes,
            valid_mask=valid,
            object_ids=obj_ids,
            labels=labels,
            num_people=np.int32(P),
            num_frames=np.int32(N),
            image_height=np.int32(H),
            image_width=np.int32(W),
            frame_indices=frame_indices,
            keypoint_scheme=np.array("mamma_smplx_512", dtype="<U32"),
            confidence_recipe=np.array(
                "visibility * temporal_consistency * in_mask(dilated)", dtype="<U64"
            ),
        )
        n_ok = int(valid.sum())
        print(
            f"  ma_2d: P={P} (obj_ids={obj_ids.tolist()}) × N={N} "
            f"→ {n_ok} valid person-frames → {out_npz.name}"
        )
        if ov_path is not None:
            print(f"  ma_2d/{ov_path.name}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(p)
    # MammaNet runs in fp32 (small ViT-B; avoids half-precision decoder issues).
    p.set_defaults(dtype="fp32")
    p.add_argument(
        "--mamma-config", type=Path, default=DEFAULT_CONFIG,
        help="MammaNet train YAML (default: masked 512-landmark config)",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="MammaNet .ckpt (default: %(default)s)",
    )
    p.add_argument(
        "--label", default="person",
        help="bboxes.npz label to run on (default: %(default)r)",
    )
    p.add_argument(
        "--obj-ids", type=int, nargs="*", default=None,
        help="subset of sam3 object ids to run (default: all tracked people)",
    )
    p.add_argument(
        "--vis-thresh", type=float, default=0.5,
        help="overlay confidence threshold (default: %(default)s; gates the "
             "combined confidence, not raw visibility; saved arrays are unfiltered)",
    )
    p.add_argument(
        "--overlay", action=argparse.BooleanOptionalAction, default=True,
        help="render landmark overlay (default: ON; --no-overlay to disable)",
    )
    p.add_argument(
        "--fps", type=float, default=None,
        help="overlay-video fps (default: 30)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_batch(Ma2dStep(args), args)


if __name__ == "__main__":
    raise SystemExit(main())
