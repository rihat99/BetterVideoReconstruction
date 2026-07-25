"""Multi-view camera + depth via VGGT-Omega.

Runs right after the tracker. VGGT-Omega is a *multi-view* model: it ingests the
whole clip in a single forward (cross-frame attention) and predicts per-frame
camera extrinsics + intrinsics and per-frame depth + confidence. It runs at a
reduced internal resolution, so we record that resolution next to the results
for later mapping back to full-resolution frames.

Memory scales with frame count (~43 GB peak for 500 frames @ 512 on an A100, per
the upstream benchmark). Clips longer than ``--max-frames`` (default 500) are
uniformly subsampled; the selected *original* frame indices are recorded in
``frame_indices`` so depth / cameras stay aligned to the source clip.
``[vggt].frame_chunk`` / ``token_chunk`` split the forward so a long sequence
still fits in memory (0 = off, the whole sequence at once).

**Several camera views.** A manifest entry may list ``sources`` — one video per
camera of a synchronized rig. All views then go into ONE forward, so every
camera lands in one common world frame. The saved arrays are a flat list of
*views into the solve*: ``T`` rows, view 0's rows first, with ``view_ids``
saying which view each row came from. ``max_frames`` caps frames **per view**,
so the views stay aligned in time. A single-view clip is just the ``V = 1``
case: ``view_ids`` is all zeros and its view name is empty, which keeps every
path flat and every array exactly what it was before.

Per-item I/O (``<save-dir>/<stem>/``)::

    reads:  the source video / folder of frames / single image (one per view)
    writes: vggt/camera.npz
              extrinsics       (T, 4, 4) f32  camera-from-world, OpenCV
              intrinsics       (T, 3, 3) f32  *normalized* (fx/W, fy/H);
                                              denormalize with
                                              tools.geometry.intrinsics.denormalize_K
              intrinsics_px    (T, 3, 3) f32  pixel units at the VGGT resolution
              fov_x_deg / fov_y_deg (T,) f32
              frame_indices    (T,)     i32   original frame index of each row,
                                              within its OWN view
              view_ids         (T,)     i32   which view each row came from
              view_names       (V,)     str   view names; ('',) when there is one
              image_height / image_width      original frame resolution (view 0)
              vggt_height  / vggt_width       VGGT inference resolution
              num_frames / num_frames_total   processed vs. source frame counts
              image_resolution / mode / checkpoint   inference settings
              scale            'up_to_scale'  (VGGT depth/translation are NOT
                                              metric; one arbitrary per-clip scale)
            vggt/depth/<view name>/frame_NNNNNN.npz   (with [vggt].save_depth;
              NNNNNN is the original frame index. The view name is empty for a
              single-view clip, i.e. the usual flat vggt/depth/frame_*.npz)
              depth (H, W) f16, depth_conf (H, W) f16   at the VGGT resolution

Standalone CLI::

    uv run python scripts/stages/estimate_camera_vggt.py \\
        --source video.mp4 --save-dir out/ \\
        --config configs/pipeline.toml
"""
from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch

from tools.adapters.vggt_omega import (
    DEFAULT_CHECKPOINT_REPO,
    DEFAULT_IMAGE_RESOLUTION,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MODE,
    infer_sequence,
    load_vggt_omega,
    uniform_subsample,
)
from tools.config import load_toml, section
from tools.precompute_runner import InputItem, SkipItem, add_standard_args, run_batch
from tools.source import resolve_source


class CameraVggtStep:
    """Run VGGT-Omega once per clip and persist cameras (+ optional depth)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = None

    def load(self) -> None:
        self.model = load_vggt_omega(self.args.checkpoint or None, device=self.args.device)

    def release(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process(self, item: InputItem) -> None:
        subdir = getattr(self.args, "subdir", "vggt") or "vggt"
        vggt_dir = item.out_dir if subdir in (".", "") else item.out_dir / subdir
        out_npz = vggt_dir / "camera.npz"
        if self.args.skip_existing and out_npz.exists():
            raise SkipItem(f"vggt already done: {out_npz}")

        views = item.views
        # One view keeps the empty name, so every per-view path stays flat.
        view_names = [""] if len(views) == 1 else [v.stem for v in views]

        with tempfile.TemporaryDirectory(prefix="vggt_frames_") as tmp_root:
            # Each view decodes into its own subdir so the frames cannot collide.
            view_frames = []
            for v, src in enumerate(views):
                frame_paths, _ = resolve_source(src, Path(tmp_root) / str(v))
                if not frame_paths:
                    raise SkipItem(f"no frames resolved from {src}")
                view_frames.append(frame_paths)

            # Original frame resolution (before VGGT's internal resize).
            bgr0 = cv2.imread(str(view_frames[0][0]))
            if bgr0 is None:
                raise RuntimeError(f"cannot read frame {view_frames[0][0]}")
            img_h, img_w = bgr0.shape[:2]

            # Subsample PER VIEW, not across the concatenation: max_frames means
            # "frames kept per view", which is what keeps the views in step.
            sel = [uniform_subsample(len(paths), self.args.max_frames) for paths in view_frames]
            sel_paths = [paths[i] for paths, keep in zip(view_frames, sel) for i in keep]
            frame_indices = np.concatenate(sel).astype(np.int32)
            view_ids = np.concatenate(
                [np.full(len(keep), v, np.int32) for v, keep in enumerate(sel)])
            n_total = sum(len(paths) for paths in view_frames)

            res = infer_sequence(
                self.model,
                sel_paths,
                image_resolution=self.args.image_resolution,
                mode=self.args.mode,
                device=self.args.device,
                compute_depth=self.args.save_depth,
                frame_chunk=self.args.frame_chunk or None,
                token_chunk=self.args.token_chunk or None,
            )

            vggt_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_npz,
                extrinsics=res["extrinsics"],
                intrinsics=res["intrinsics"],
                intrinsics_px=res["intrinsics_px"],
                fov_x_deg=res["fov_x_deg"],
                fov_y_deg=res["fov_y_deg"],
                frame_indices=frame_indices,
                view_ids=view_ids,
                view_names=np.array(view_names),
                image_height=np.int32(img_h),
                image_width=np.int32(img_w),
                vggt_height=np.int32(res["vggt_height"]),
                vggt_width=np.int32(res["vggt_width"]),
                num_frames=np.int32(len(sel_paths)),
                num_frames_total=np.int32(n_total),
                image_resolution=np.int32(self.args.image_resolution),
                mode=np.str_(self.args.mode),
                checkpoint=np.str_(self.args.checkpoint or DEFAULT_CHECKPOINT_REPO),
                scale=np.str_("up_to_scale"),
            )
            if self.args.save_depth and "depth" in res:
                self._save_depth(vggt_dir / "depth", frame_indices, view_ids, view_names, res)

            print(
                f"  {len(sel_paths)}/{n_total} frame(s) from {len(views)} view(s) "
                f"orig {img_w}x{img_h} -> vggt {res['vggt_width']}x{res['vggt_height']}, "
                f"fov_x mean={float(res['fov_x_deg'].mean()):.2f}deg -> {out_npz}"
            )

    @staticmethod
    def _save_depth(depth_dir: Path, frame_indices: np.ndarray, view_ids: np.ndarray,
                    view_names: list[str], res: dict) -> None:
        """One ``frame_NNNNNN.npz`` per row, under that row's view subdir."""
        for name in view_names:
            (depth_dir / name).mkdir(parents=True, exist_ok=True)
        depth = res["depth"]
        conf = res.get("depth_conf")
        for k, (orig_idx, v) in enumerate(zip(frame_indices, view_ids)):
            save: dict[str, np.ndarray] = {"depth": depth[k].astype(np.float16)}
            if conf is not None:
                save["depth_conf"] = conf[k].astype(np.float16)
            out = depth_dir / view_names[int(v)] / f"frame_{int(orig_idx):06d}.npz"
            np.savez_compressed(out, **save)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(p)
    cli = p.parse_args(argv)
    cfg = load_toml(cli.config)
    args = section(cfg, "vggt", {
        "skip_existing": False,
        "dtype": "fp16",
        "checkpoint": DEFAULT_CHECKPOINT_REPO,
        "subdir": "vggt",
        "max_frames": DEFAULT_MAX_FRAMES,
        "image_resolution": DEFAULT_IMAGE_RESOLUTION,
        "mode": DEFAULT_MODE,
        "save_depth": False,
        "frame_chunk": 0,
        "token_chunk": 0,
    })
    args.device = section(cfg, "runtime", {"device": "cuda"}).device
    args.source, args.save_dir, args.config = cli.source, cli.save_dir, cli.config
    args.skip_existing = args.skip_existing or cli.skip_existing
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_batch(CameraVggtStep(args), args)


if __name__ == "__main__":
    raise SystemExit(main())
