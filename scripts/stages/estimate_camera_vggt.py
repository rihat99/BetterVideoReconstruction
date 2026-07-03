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

Per-item I/O (``<save-dir>/<stem>/``)::

    reads:  the source video / folder of frames / single image
    writes: vggt/camera.npz
              extrinsics       (T, 4, 4) f32  camera-from-world, OpenCV
              intrinsics       (T, 3, 3) f32  *normalized* (fx/W, fy/H);
                                              denormalize with
                                              tools.geometry.intrinsics.denormalize_K
              intrinsics_px    (T, 3, 3) f32  pixel units at the VGGT resolution
              fov_x_deg / fov_y_deg (T,) f32
              frame_indices    (T,)     i32   original indices of processed frames
              image_height / image_width      original frame resolution
              vggt_height  / vggt_width       VGGT inference resolution
              num_frames / num_frames_total   processed vs. source frame counts
              image_resolution / mode / checkpoint   inference settings
              scale            'up_to_scale'  (VGGT depth/translation are NOT
                                              metric; one arbitrary per-clip scale)
            vggt/depth/frame_NNNNNN.npz   (only with --save-depth; NNNNNN is the
              original frame index)
              depth (H, W) f16, depth_conf (H, W) f16   at the VGGT resolution

Standalone CLI::

    uv run python scripts/stages/estimate_camera_vggt.py \\
        --source video.mp4 --save-dir out/ \\
        --checkpoint /path/to/vggt_omega_1b_512.pt
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

        with tempfile.TemporaryDirectory(prefix="vggt_frames_") as tmp_root:
            frame_paths, _ = resolve_source(item.source_path, Path(tmp_root))
            n_total = len(frame_paths)
            if n_total == 0:
                raise SkipItem(f"no frames resolved from {item.source_path}")

            # Original frame resolution (before VGGT's internal resize).
            bgr0 = cv2.imread(str(frame_paths[0]))
            if bgr0 is None:
                raise RuntimeError(f"cannot read frame {frame_paths[0]}")
            img_h, img_w = bgr0.shape[:2]

            sel = uniform_subsample(n_total, self.args.max_frames)
            sel_paths = [frame_paths[i] for i in sel]

            res = infer_sequence(
                self.model,
                sel_paths,
                image_resolution=self.args.image_resolution,
                mode=self.args.mode,
                device=self.args.device,
                compute_depth=self.args.save_depth,
            )

            vggt_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_npz,
                extrinsics=res["extrinsics"],
                intrinsics=res["intrinsics"],
                intrinsics_px=res["intrinsics_px"],
                fov_x_deg=res["fov_x_deg"],
                fov_y_deg=res["fov_y_deg"],
                frame_indices=sel.astype(np.int32),
                image_height=np.int32(img_h),
                image_width=np.int32(img_w),
                vggt_height=np.int32(res["vggt_height"]),
                vggt_width=np.int32(res["vggt_width"]),
                num_frames=np.int32(len(sel)),
                num_frames_total=np.int32(n_total),
                image_resolution=np.int32(self.args.image_resolution),
                mode=np.str_(self.args.mode),
                checkpoint=np.str_(self.args.checkpoint or DEFAULT_CHECKPOINT_REPO),
                scale=np.str_("up_to_scale"),
            )
            if self.args.save_depth and "depth" in res:
                self._save_depth(vggt_dir / "depth", sel, res)

            print(
                f"  {len(sel)}/{n_total} frame(s) "
                f"orig {img_w}x{img_h} -> vggt {res['vggt_width']}x{res['vggt_height']}, "
                f"fov_x mean={float(res['fov_x_deg'].mean()):.2f}deg -> {out_npz}"
            )

    @staticmethod
    def _save_depth(depth_dir: Path, sel: np.ndarray, res: dict) -> None:
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth = res["depth"]
        conf = res.get("depth_conf")
        for k, orig_idx in enumerate(sel):
            save: dict[str, np.ndarray] = {"depth": depth[k].astype(np.float16)}
            if conf is not None:
                save["depth_conf"] = conf[k].astype(np.float16)
            np.savez_compressed(depth_dir / f"frame_{int(orig_idx):06d}.npz", **save)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(p)
    p.set_defaults(dtype="fp16")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_REPO,
                   help="local VGGT-Omega .pt path or HF repo id "
                        f"(default: {DEFAULT_CHECKPOINT_REPO})")
    p.add_argument("--subdir", default="vggt",
                   help="output subfolder under <save-dir>/<stem>/ ('.' writes "
                        "directly into it; default: vggt)")
    p.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                   help="cap on frames per clip; longer clips are uniformly "
                        f"subsampled, <=0 disables (default: {DEFAULT_MAX_FRAMES})")
    p.add_argument("--image-resolution", type=int, default=DEFAULT_IMAGE_RESOLUTION,
                   help=f"VGGT-Omega input resolution (default: {DEFAULT_IMAGE_RESOLUTION})")
    p.add_argument("--mode", default=DEFAULT_MODE, choices=("balanced", "max_size"),
                   help="load_and_preprocess_images sizing mode (default: balanced)")
    p.add_argument("--save-depth", action="store_true",
                   help="also write per-frame depth + confidence at the VGGT "
                        "resolution to vggt/depth/frame_NNNNNN.npz")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_batch(CameraVggtStep(args), args)


if __name__ == "__main__":
    raise SystemExit(main())
