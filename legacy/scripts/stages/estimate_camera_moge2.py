"""Per-frame camera intrinsics (and optional depth) via MoGe2.

The dedicated camera-parameter stage for the video pipeline. For each frame
MoGe2 estimates its own FOV (``fov_x=None``) and returns normalized
intrinsics; we stack them across the clip and write a single small
``moge2/camera.npz``. The SAM-3D-Body stage reads per-frame ``K`` from here
(see ``scripts/stages/estimate_body_sam3d.py``), so camera estimation is
decoupled from body fitting.

A future ``estimate_camera_vggt.py`` (VGGT-Omega, multi-view) is expected to
write the same ``moge2/camera.npz`` contract so it can drop in as a richer
replacement.

Per-item I/O (``<save-dir>/<stem>/``)::

    reads:  the source video / folder of frames / single image
    writes: moge2/camera.npz
              intrinsics    (T, 3, 3) f32  MoGe2 *normalized* convention
                                           (fx/W, fy/H; denormalize with
                                           tools.adapters.moge2.denormalize_K)
              fov_x_deg     (T,)      f32  horizontal FOV per frame
              frame_indices (T,)      i32
              image_height / image_width / num_frames  scalars
            moge2/depth/frame_NNNNNN.npz   (only with --save-depth)
              points (H,W,3) f16, depth (H,W) f16, mask? (H,W) bool,
              normals? (H,W,3) f16

Standalone CLI::

    uv run python scripts/stages/estimate_camera_moge2.py \\
        --source video.mp4 --save-dir out/
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

from tools.adapters.moge2 import DEFAULT_MODEL, fov_x_deg_from_intrinsics, infer_frame, load_moge2
from tools.precompute_runner import InputItem, SkipItem, add_standard_args, run_batch
from tools.source import resolve_source


class CameraMoge2Step:
    """Run MoGe2 per frame and persist stacked camera intrinsics."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = None

    def load(self) -> None:
        self.model = load_moge2(self.args.model, device=self.args.device)

    def release(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process(self, item: InputItem) -> None:
        subdir = getattr(self.args, "subdir", "moge2") or "moge2"
        moge2_dir = item.out_dir if subdir in (".", "") else item.out_dir / subdir
        out_npz = moge2_dir / "camera.npz"
        if self.args.skip_existing and out_npz.exists():
            raise SkipItem(f"moge2 already done: {out_npz}")

        use_fp16 = self.args.dtype != "fp32"
        depth_dir = moge2_dir / "depth"

        with tempfile.TemporaryDirectory(prefix="moge2_frames_") as tmp_root:
            frame_paths, _ = resolve_source(item.source_path, Path(tmp_root))
            max_frames = getattr(self.args, "max_frames", 0) or 0
            if max_frames > 0:
                frame_paths = frame_paths[:max_frames]  # first N frames only
            n = len(frame_paths)
            if n == 0:
                raise SkipItem(f"no frames resolved from {item.source_path}")

            intrinsics = np.zeros((n, 3, 3), dtype=np.float32)
            fov_x_deg = np.zeros((n,), dtype=np.float32)
            height = width = 0

            for i, fp in enumerate(frame_paths):
                bgr = cv2.imread(str(fp))
                if bgr is None:
                    raise RuntimeError(f"cannot read frame {fp}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                res = infer_frame(
                    self.model, rgb,
                    fov_x_deg=None,  # let MoGe2 estimate the camera per frame
                    use_fp16=use_fp16,
                    edge_rtol=self.args.edge_rtol,
                    edge_normal_tol=self.args.edge_normal_tol,
                    device=self.args.device,
                )
                intrinsics[i] = res["intrinsics"]
                fov_x_deg[i] = fov_x_deg_from_intrinsics(res["intrinsics"])
                if self.args.save_depth:
                    self._save_depth(depth_dir / f"frame_{i:06d}.npz", res)

            moge2_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_npz,
                intrinsics=intrinsics,
                fov_x_deg=fov_x_deg,
                frame_indices=np.arange(n, dtype=np.int32),
                image_height=np.int32(height),
                image_width=np.int32(width),
                num_frames=np.int32(n),
            )
            print(
                f"  {n} frame(s) {width}x{height}, "
                f"fov_x mean={float(fov_x_deg.mean()):.2f}deg -> {out_npz}"
            )

    @staticmethod
    def _save_depth(path: Path, res: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save: dict[str, np.ndarray] = {
            "points": res["points"].astype(np.float16),
            "depth": res["depth"].astype(np.float16),
            "intrinsics": res["intrinsics"],
        }
        if "mask" in res:
            save["mask"] = res["mask"]
        if "normals" in res:
            save["normals"] = res["normals"].astype(np.float16)
        np.savez_compressed(path, **save)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(p)
    p.set_defaults(dtype="fp16")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"MoGe2 HuggingFace model id (default: {DEFAULT_MODEL})")
    p.add_argument("--subdir", default="moge2",
                   help="output subfolder under <save-dir>/<stem>/ ('.' writes "
                        "directly into it; default: moge2)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="process only the first N frames (0 = all, the default); "
                        "N=1 gives a single first-frame camera + depth.")
    p.add_argument("--save-depth", action="store_true",
                   help="also write per-frame depth/points to moge2/depth/frame_NNNNNN.npz")
    p.add_argument("--edge-rtol", type=float, default=0.04,
                   help="MoGe depth-edge filter relative tolerance (0 disables)")
    p.add_argument("--edge-normal-tol", type=float, default=0.0,
                   help="optional normal-edge tolerance in degrees (0 disables)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_batch(CameraMoge2Step(args), args)


if __name__ == "__main__":
    raise SystemExit(main())
