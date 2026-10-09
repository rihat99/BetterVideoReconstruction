"""Sapiens2 pose (+ optional pointmap / normal) precompute step.

Reads frames straight from the source video / folder / image and runs
Meta's Sapiens2-5B model for any subset of {``pose``, ``pointmap``,
``normal``}. **Default is ``pose`` only** (keypoints) — the dense
``pointmap``/``normal`` depth-like forwards are opt-in (``--tasks
pose,pointmap``). Per-item outputs land under ``<out-dir>/sapiens/`` so
they don't collide with other precompute artifacts.

Pose is top-down — it consumes the bboxes that the ``sam3`` stage
already wrote (we never run Sapiens2's own DETR-based detector). Pointmap
and normal are dense, full-image forwards. The 5B head is heavy
(~20-30 GB VRAM at 1024×768 in bf16) and the safetensors load is ~30-60 s,
so the orchestrator (:func:`run_sapiens`) loads each task's model **once**,
sweeps every item through it, then releases — flipping the inner loop
from ``load → forward → release × N`` to ``load → forward × N → release``.

The 308-keypoint Sociopticon scheme used here is a strict superset of
SAM-3D-Body's MHR70 — indices 0..69 of ``pose.npz`` are byte-for-byte the
MHR70 layout, suitable for keypoint-conditioned SAM-3D-Body inference.

Per-item I/O::

    reads:  the source video / folder of frames / single image
            <out-dir>/sam3/bboxes.npz           only when 'pose' in --tasks
    writes: <out-dir>/sapiens/pose.npz          (P, N, 308, 2) keypoints + (P, N, 308) scores
            <out-dir>/sapiens/pointmap.npz      (N, H, W, 3) f16 metric xyz
            <out-dir>/sapiens/normal.npz        (N, H, W, 3) f16 unit normals
            <out-dir>/sapiens/overlay/pose.png|.mp4   skeleton overlay (default ON)

Standalone::

    uv run python bvr/stages/estimate_sapiens.py \\
        --source data/images/wild_images/ellen2.jpg \\
        --save-dir out/images/wild_images/ \\
        --config configs/pipeline.toml

Pipeline integration is deferred — this step is opt-in until we know the
VRAM/perf profile in the static-pipeline context.
"""
from __future__ import annotations

import argparse
import gc
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from bvr import determinism
from bvr.config import load_config, section
from bvr.precompute_runner import (
    InputItem,
    SkipItem,
    add_standard_args,
    run_batch,
)
from bvr.source import resolve_source
from bvr.adapters.sapiens2 import (
    NUM_KEYPOINTS,
    forward_dense,
    forward_pose_batched,
    load_sapiens2,
)


_TASK_CHOICES = ("pose", "pointmap", "normal")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _parse_tasks(s: str) -> list[str]:
    items = [t.strip() for t in s.split(",") if t.strip()]
    bad = [t for t in items if t not in _TASK_CHOICES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown task(s) {bad}; choose from {list(_TASK_CHOICES)}"
        )
    if not items:
        raise argparse.ArgumentTypeError("--tasks must list at least one task")
    # de-dupe while preserving order
    seen: set[str] = set()
    return [t for t in items if not (t in seen or seen.add(t))]


def _draw_pose_overlay(
    bgr: np.ndarray,
    kpts_per_person: list[np.ndarray],
    scores_per_person: list[np.ndarray],
    *,
    skeleton_links: np.ndarray,            # (E, 2) joint-index pairs
    skeleton_link_colors: np.ndarray,      # (E, 3) RGB
    keypoint_colors: np.ndarray,           # (K, 3) RGB
    score_thr: float,
    radius: int = 2,
    thickness: int = 2,
) -> np.ndarray:
    """Draw keypoint dots + skeleton edges using the model's own metainfo.

    ``skeleton_links``, ``skeleton_link_colors``, and ``keypoint_colors`` come
    straight from ``model.pose_metainfo`` — using them keeps the overlay
    consistent with whichever 308-keypoint scheme the model emits (the names
    aren't COCO-style: e.g. ``9 = left_hip``, ``41 = right_wrist``,
    ``62 = left_wrist``).
    """
    canvas = bgr.copy()
    H, W = canvas.shape[:2]
    for kpts, scs in zip(kpts_per_person, scores_per_person):
        # Skeleton edges
        for (a, b), rgb in zip(skeleton_links, skeleton_link_colors):
            if scs[a] < score_thr or scs[b] < score_thr:
                continue
            pa = (int(round(kpts[a, 0])), int(round(kpts[a, 1])))
            pb = (int(round(kpts[b, 0])), int(round(kpts[b, 1])))
            color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))   # RGB → BGR
            cv2.line(canvas, pa, pb, color, thickness, lineType=cv2.LINE_AA)
        # Keypoint dots (drawn on top so they're not hidden under thick edges)
        for j in range(NUM_KEYPOINTS):
            if scs[j] < score_thr:
                continue
            x, y = int(round(kpts[j, 0])), int(round(kpts[j, 1]))
            if 0 <= x < W and 0 <= y < H:
                rgb = keypoint_colors[j]
                color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                cv2.circle(canvas, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


# -----------------------------------------------------------------------------
# Step
# -----------------------------------------------------------------------------

class Sapiens2Step:
    """Run a single Sapiens2 5B head across a batch.

    One instance handles exactly one task (``"pose"`` / ``"pointmap"`` /
    ``"normal"``). The 5B safetensors load takes ~30-60 s, so the multi-task
    orchestrator (:func:`run_sapiens`) creates one Step per task and feeds
    every item through it before releasing the model — that flips the inner
    loop from "load → forward → release × N items" to "load → forward × N → release".
    """

    def __init__(self, args: argparse.Namespace, task: str):
        if task not in _TASK_CHOICES:
            raise ValueError(f"unknown task: {task!r}; choices: {list(_TASK_CHOICES)}")
        self.args = args
        self.task = task
        self._model = None
        # Pose-only metainfo for overlay rendering. Captured in :meth:`load`.
        self._sk_links: np.ndarray | None = None
        self._sk_colors: np.ndarray | None = None
        self._kp_colors: np.ndarray | None = None

    def load(self) -> None:
        self._model = load_sapiens2(
            self.task,
            checkpoint_root=self.args.checkpoint_root,
            device=self.args.device,
            dtype=self.args.dtype,
        )
        if self.task == "pose":
            mi = self._model.pose_metainfo
            self._sk_links = np.asarray(mi["skeleton_links"], dtype=np.int32)
            self._sk_colors = np.asarray(mi["skeleton_link_colors"], dtype=np.int32)
            self._kp_colors = np.asarray(mi["keypoint_colors"], dtype=np.int32)

    def release(self) -> None:
        self._model = None
        self._sk_links = self._sk_colors = self._kp_colors = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- per-item entry --------------------------------------------------

    def process(self, item: InputItem) -> None:
        out_dir = item.out_dir
        subdir = getattr(self.args, "subdir", "sapiens") or "sapiens"
        sapiens_dir = out_dir if subdir in (".", "") else out_dir / subdir
        # Top-down pose consumes sam3 bboxes: an explicit per-item path from the
        # manifest (DB layout, where sam3 lives in its own feature tree) wins;
        # otherwise the sibling out_dir/sam3/bboxes.npz (single-clip layout).
        bboxes_npz = item.bboxes_path or (out_dir / "sam3" / "bboxes.npz")
        out_npz = sapiens_dir / f"{self.task}.npz"
        if self.args.skip_existing and out_npz.exists():
            raise SkipItem(f"{out_npz} exists")
        sapiens_dir.mkdir(parents=True, exist_ok=True)

        # ---- frames (decoded straight from the source video/folder/image) ----
        with tempfile.TemporaryDirectory(prefix="sapiens_frames_") as _tmp:
            frame_paths, _ = resolve_source(item.source_path, Path(_tmp))
            bgr_frames: list[np.ndarray] = []
            for fp in frame_paths:
                bgr = cv2.imread(str(fp))
                if bgr is None:
                    raise SkipItem(f"failed to read frame {fp}")
                bgr_frames.append(bgr)
        if not bgr_frames:
            raise SkipItem(f"no frames resolved from {item.source_path}")
        N = len(bgr_frames)
        H, W = bgr_frames[0].shape[:2]

        if self.task == "pose":
            self._run_pose(bboxes_npz, bgr_frames, sapiens_dir, N=N, H=H, W=W)
        else:
            self._run_dense(bgr_frames, sapiens_dir, N=N, H=H, W=W)

    # ----- pose ------------------------------------------------------------

    def _run_pose(
        self,
        bp: Path,
        bgr_frames: list[np.ndarray],
        sapiens_dir: Path,
        *,
        N: int,
        H: int,
        W: int,
    ) -> None:
        if not bp.exists():
            raise SkipItem(
                f"missing {bp} — pose is top-down and needs bboxes from sam3"
            )
        bd = np.load(bp, allow_pickle=True)
        all_labels: list[str] = list(bd["labels"])
        all_ids = np.asarray(bd["object_ids"], dtype=np.int32)
        all_bb = np.asarray(bd["bboxes_per_obj"], dtype=np.int32)  # (P', N, 4)
        keep = np.array(
            [lab == self.args.label for lab in all_labels], dtype=bool,
        )
        if not keep.any():
            raise SkipItem(
                f"no '{self.args.label}' objects in {bp.name} (labels={all_labels})"
            )
        n_bb = int(bd["num_frames"])
        if n_bb != N:
            raise RuntimeError(
                f"frame count mismatch in {bp}: bboxes={n_bb}, frames={N}"
            )
        bboxes = all_bb[keep].astype(np.float32)
        obj_ids = all_ids[keep]
        labels = np.array(
            [all_labels[i] for i in np.flatnonzero(keep)], dtype="U16",
        )
        P = int(bboxes.shape[0])

        keypoints = np.zeros((P, N, NUM_KEYPOINTS, 2), dtype=np.float32)
        scores = np.zeros((P, N, NUM_KEYPOINTS), dtype=np.float32)
        valid = np.zeros((P, N), dtype=bool)
        kpts_view: list[list[np.ndarray]] = [[] for _ in range(N)]
        scs_view: list[list[np.ndarray]] = [[] for _ in range(N)]

        # Frames go through the cross-image batched forward in windows of
        # ``batch_size`` frames, so the backbone sees ~batch_size crops per
        # call instead of the one or two people a single frame contributes.
        window = max(1, int(self.args.batch_size))
        for f0 in range(0, N, window):
            frames = list(range(f0, min(N, f0 + window)))
            rows_per_frame: list[list[int]] = []
            bbs_per_frame: list[np.ndarray] = []
            for f in frames:
                rows = [p for p in range(P) if not np.all(bboxes[p, f] == 0)]
                rows_per_frame.append(rows)
                bbs_per_frame.append(bboxes[rows, f] if rows else np.zeros((0, 4), np.float32))
            results = forward_pose_batched(
                self._model, [bgr_frames[f] for f in frames], bbs_per_frame,
                batch_size=self.args.batch_size,
            )
            for f, rows, (kpts, scs) in zip(frames, rows_per_frame, results):
                for k, p in enumerate(rows):
                    keypoints[p, f] = kpts[k]
                    scores[p, f] = scs[k]
                    valid[p, f] = True
                    kpts_view[f].append(kpts[k])
                    scs_view[f].append(scs[k])

        out_npz = sapiens_dir / "pose.npz"
        np.savez_compressed(
            out_npz,
            keypoints=keypoints,
            keypoint_scores=scores,
            bboxes=bboxes,
            valid_mask=valid,
            object_ids=obj_ids,
            labels=labels,
            num_people=np.int32(P),
            num_frames=np.int32(N),
            image_height=np.int32(H),
            image_width=np.int32(W),
            keypoint_scheme=np.array("goliath_308", dtype="<U32"),
        )
        n_ok = int(valid.sum())
        print(
            f"  sapiens/pose: P={P} (obj_ids={obj_ids.tolist()}) × N={N} "
            f"→ {n_ok} valid → {out_npz.name}"
        )

        if self.args.overlay:
            self._render_pose_overlay(
                bgr_frames=bgr_frames,
                kpts_per_frame=kpts_view,
                scs_per_frame=scs_view,
                sapiens_dir=sapiens_dir,
                N=N, W=W, H=H,
            )

    def _render_pose_overlay(
        self,
        *,
        bgr_frames: list[np.ndarray],
        kpts_per_frame: list[list[np.ndarray]],
        scs_per_frame: list[list[np.ndarray]],
        sapiens_dir: Path,
        N: int, W: int, H: int,
    ) -> None:
        ovr_dir = sapiens_dir / "overlay"
        ovr_dir.mkdir(parents=True, exist_ok=True)
        thr = float(self.args.score_thresh)
        draw_kw = dict(
            skeleton_links=self._sk_links,
            skeleton_link_colors=self._sk_colors,
            keypoint_colors=self._kp_colors,
            score_thr=thr,
        )
        if N == 1:
            ov = _draw_pose_overlay(
                bgr_frames[0], kpts_per_frame[0], scs_per_frame[0], **draw_kw,
            )
            out_path = ovr_dir / "pose.png"
            cv2.imwrite(str(out_path), ov)
        else:
            out_path = ovr_dir / "pose.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = float(self.args.fps or 30.0)
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
            if not writer.isOpened():
                raise RuntimeError(f"cannot open video writer at {out_path}")
            try:
                for f in range(N):
                    ov = _draw_pose_overlay(
                        bgr_frames[f], kpts_per_frame[f], scs_per_frame[f],
                        **draw_kw,
                    )
                    writer.write(ov)
            finally:
                writer.release()
        print(f"  sapiens/overlay/{out_path.name}")

    # ----- pointmap / normal ----------------------------------------------

    def _run_dense(
        self,
        bgr_frames: list[np.ndarray],
        sapiens_dir: Path,
        *,
        N: int, H: int, W: int,
    ) -> None:
        out_npz = sapiens_dir / f"{self.task}.npz"
        buffer = np.zeros((N, H, W, 3), dtype=np.float16)
        for f in range(N):
            arr = forward_dense(self._model, bgr_frames[f], head=self.task)
            buffer[f] = arr.astype(np.float16)
        payload: dict[str, np.ndarray] = {
            self.task: buffer,
            "num_frames": np.int32(N),
            "image_height": np.int32(H),
            "image_width": np.int32(W),
        }
        np.savez_compressed(out_npz, **payload)
        print(f"  sapiens/{self.task}: N={N} → {out_npz.name}")


def run_sapiens(args: argparse.Namespace) -> int:
    """Per-task batched runner: load 5B head once, sweep all items, release.

    Drives one ``run_batch`` pass per task in ``args.tasks``. Each pass
    instantiates a :class:`Sapiens2Step` for that task, which holds the
    model resident in VRAM for the whole sweep — so 24 items × 2 tasks
    needs 2 model loads instead of 48.
    """
    rc = 0
    for task in args.tasks:
        print(f"\n--- sapiens task: {task} ---")
        rc |= run_batch(Sapiens2Step(args, task), args)
    return rc


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(p)
    cli = p.parse_args(argv)
    cfg = load_config(cli.config)
    args = section(cfg, "sapiens", {
        "skip_existing": False,
        "dtype": "fp16",
        "tasks": "pose",
        "label": "person",
        "subdir": "sapiens",
        "batch_size": 1,
        "checkpoint_root": "",
        "score_thresh": 0.3,
        "overlay": True,
        "fps": "",
    })
    args.device = section(cfg, "runtime", {"device": "cuda"}).device
    args.source, args.save_dir, args.config = cli.source, cli.save_dir, cli.config
    args.skip_existing = args.skip_existing or cli.skip_existing
    args.tasks = _parse_tasks(args.tasks) if isinstance(args.tasks, str) else list(args.tasks)
    args.checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else None
    args.fps = float(args.fps) if args.fps != "" else None
    return args


def main(argv: list[str] | None = None) -> int:
    determinism.apply()
    args = _parse_args(argv)
    return run_sapiens(args)


if __name__ == "__main__":
    raise SystemExit(main())
