"""Convert SAM-3D-Body MHR predictions to better_human SMPL parameters.

Run from CLI:

    uv run python scripts/stages/convert_mhr_to_smpl.py \\
        --source path/to/{video.mp4|frames_folder|photo.jpg} \\
        --predictions out/video/ \\
        [--smpl-model /home/rikhat/better/better_human/models/smpl/SMPL_NEUTRAL.npz] \\
        [--device cuda] [--iters 300] [--render-video]

Or import:

    from scripts.stages.convert_mhr_to_smpl import MhrToSmplScript
    script = MhrToSmplScript()
    script.run("video.mp4", "out/video/", render_video=True)

Inputs:
    source             video file, folder of frames, or single image
    predictions_dir    folder containing sam3d/params.npz (+ sam3/). The body
                       params stack all tracked people on a (P, N, ...) axis;
                       this single-person path picks one (``--person``, default 0).

Outputs (into <predictions_dir>/):
    smpl_params.npz               per-frame SMPL params (betas, body_pose,
                                  global_orient, transl, q)
    <source-stem>_smpl.mp4        SMPL mesh overlay video (only when
                                  render_video=True)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts/ to sys.path so `import _pym_preload` works from this subfolder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pym_preload  # noqa: E402, F401  (must be first — re-execs to fix libstdc++ ABI)

import argparse
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from tqdm import tqdm


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAPPING_NPZ = (
    _REPO_ROOT
    / "third_party"
    / "MHR"
    / "tools"
    / "mhr_smpl_conversion"
    / "assets"
    / "mhr2smpl_mapping.npz"
)
_DEFAULT_SMPL_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smpl" / "SMPL_NEUTRAL.npz"
)

# Same overlay color the SAM-3D-Body script uses.
_MESH_COLOR = (0.65098039, 0.74117647, 0.85882353)


def _add_sys_paths() -> None:
    """Ensure better_human (editable src layout) and tools/ are importable."""
    better_human_src = _REPO_ROOT.parent / "better_human" / "src"
    if better_human_src.exists():
        sys.path.insert(0, str(better_human_src))
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class ConvertResult:
    num_frames: int
    num_valid: int
    height: int
    width: int
    output_path: Path
    rendered_video: Path | None
    median_fit_error_m: float


class MhrToSmplScript:
    def __init__(
        self,
        device: str = "cuda",
        smpl_model_path: str | Path | None = None,
        num_iters: int = 300,
        batch_size: int = 64,
    ) -> None:
        self.device = device
        self.smpl_model_path = Path(smpl_model_path or _DEFAULT_SMPL_MODEL)
        self.num_iters = num_iters
        self.batch_size = batch_size
        self._mhr = None
        self._smpl = None
        self._converter = None

    def _build_converter(self):
        _add_sys_paths()
        from mhr.mhr import MHR
        from better_human.smpl import SMPL
        from tools.mhr_to_smpl import MhrToSmplConverter

        if self._mhr is None:
            self._mhr = MHR.from_files(device=torch.device(self.device), lod=1)
        if self._smpl is None:
            self._smpl = SMPL(
                str(self.smpl_model_path), gender="neutral", num_betas=10
            ).to(self.device)
        if self._converter is None:
            self._converter = MhrToSmplConverter(
                mhr_model=self._mhr,
                smpl_model=self._smpl,
                mapping_npz=_MAPPING_NPZ,
                device=self.device,
                batch_size=self.batch_size,
            )
        return self._converter

    # -- main entry point ----------------------------------------------------

    def run(
        self,
        source: str | Path,
        predictions_dir: str | Path,
        render_video: bool = False,
        video_fps: float | None = None,
        person: int = 0,
    ) -> ConvertResult:
        source = Path(source)
        predictions_dir = Path(predictions_dir)

        body_path = predictions_dir / "sam3d" / "params.npz"
        if not body_path.exists():
            raise FileNotFoundError(body_path)
        bd = np.load(body_path)

        mhr_model_params = bd["mhr_model_params"]  # (P, N, 204) — legacy: (N, 204)
        shape_params = bd["shape_params"]  # (P, N, 45)
        pred_cam_t = bd["pred_cam_t"]  # (P, N, 3)
        focal_lengths = bd["focal_length"]  # (P, N)
        cam_intrinsics = bd["cam_intrinsics"]  # (3, 3) — shared camera
        valid_mask = bd["valid_mask"]  # (P, N) bool
        N = int(bd["num_frames"])
        H = int(bd["video_height"])
        W = int(bd["video_width"])

        # sam3d/params.npz stacks all tracked people on a leading (P, N, ...)
        # axis; the SMPL motion path is single-person, so select one person
        # (default index 0 = the primary tracked person).
        if mhr_model_params.ndim == 3:
            num_people = mhr_model_params.shape[0]
            if not 0 <= person < num_people:
                raise ValueError(f"--person {person} out of range (P={num_people})")
            mhr_model_params = mhr_model_params[person]
            shape_params = shape_params[person]
            pred_cam_t = pred_cam_t[person]
            focal_lengths = focal_lengths[person]
            valid_mask = valid_mask[person]

        # Some per-frame arrays may be NaN at invalid frames; replace with zeros
        # so torch tensors are safe to construct. Converter skips them anyway.
        invalid = ~valid_mask
        mhr_model_params = np.where(
            invalid[:, None], 0.0, np.nan_to_num(mhr_model_params)
        ).astype(np.float32)
        shape_params = np.where(
            invalid[:, None], 0.0, np.nan_to_num(shape_params)
        ).astype(np.float32)
        pred_cam_t = np.where(
            invalid[:, None], 0.0, np.nan_to_num(pred_cam_t)
        ).astype(np.float32)

        conv = self._build_converter()

        print(f"converting {int(valid_mask.sum())}/{N} valid frames...")
        result = conv.convert(
            mhr_model_params=mhr_model_params,
            shape_params=shape_params,
            pred_cam_t=pred_cam_t,
            valid_mask=valid_mask,
            num_iters=self.num_iters,
        )

        output_path = predictions_dir / "smpl_params.npz"
        np.savez(
            output_path,
            betas=result["betas"],
            body_pose=result["body_pose"],
            global_orient=result["global_orient"],
            transl=result["transl"],
            q=result["q"],
            fit_error=result["fit_error"],
            cam_intrinsics=cam_intrinsics.astype(np.float32),
            valid_mask=valid_mask,
            num_frames=np.int32(N),
            video_height=np.int32(H),
            video_width=np.int32(W),
        )

        err = result["fit_error"][valid_mask]
        median_err = float(np.median(err)) if err.size else float("nan")

        rendered_path: Path | None = None
        if render_video:
            rendered_path = predictions_dir / f"{source.stem}_smpl.mp4"
            self._render_overlay(
                source=source,
                predictions_dir=predictions_dir,
                smpl_params=result,
                valid_mask=valid_mask,
                focal_lengths=focal_lengths,
                cam_intrinsics=cam_intrinsics,
                W=W,
                H=H,
                N=N,
                out_path=rendered_path,
                fps=video_fps,
            )

        return ConvertResult(
            num_frames=N,
            num_valid=int(valid_mask.sum()),
            height=H,
            width=W,
            output_path=output_path,
            rendered_video=rendered_path,
            median_fit_error_m=median_err,
        )

    # -- overlay rendering ---------------------------------------------------

    def _render_overlay(
        self,
        source: Path,
        predictions_dir: Path,
        smpl_params: dict[str, np.ndarray],
        valid_mask: np.ndarray,
        focal_lengths: np.ndarray,
        cam_intrinsics: np.ndarray,
        W: int,
        H: int,
        N: int,
        out_path: Path,
        fps: float | None,
    ) -> None:
        _add_sys_paths()
        from sam_3d_body.visualization.renderer import Renderer
        from tools.source import resolve_source

        # SMPL faces (shared across frames)
        faces = self._smpl.faces.detach().cpu().numpy().astype(np.int32)

        # Pre-compute SMPL vertices in camera frame for all valid frames.
        valid_idx = np.flatnonzero(valid_mask)
        smpl_verts_cam = np.full((N, 6890, 3), np.nan, dtype=np.float32)
        with torch.no_grad():
            dev = self._smpl.vertices_template.device
            betas = torch.from_numpy(smpl_params["betas"][valid_idx]).to(dev)
            body_pose = torch.from_numpy(smpl_params["body_pose"][valid_idx]).to(dev)
            global_orient = torch.from_numpy(
                smpl_params["global_orient"][valid_idx]
            ).to(dev)
            transl = torch.from_numpy(smpl_params["transl"][valid_idx]).to(dev)
            bs = self.batch_size
            for s in range(0, valid_idx.size, bs):
                e = min(s + bs, valid_idx.size)
                q = self._smpl.from_classic(
                    betas[s:e], body_pose[s:e], transl[s:e], global_orient[s:e]
                )
                v = self._smpl(betas=betas[s:e], q=q).vertices
                smpl_verts_cam[valid_idx[s:e]] = v.cpu().numpy()

        fx = float(cam_intrinsics[0, 0])

        def _render_one(i: int, bgr: np.ndarray) -> np.ndarray:
            verts = smpl_verts_cam[i].astype(np.float32)
            cam_t = np.zeros(3, dtype=np.float32)
            renderer = Renderer(
                focal_length=float(focal_lengths[i])
                if not np.isnan(focal_lengths[i])
                else fx,
                faces=faces,
            )
            return (
                renderer(
                    verts,
                    cam_t,
                    bgr,
                    mesh_base_color=_MESH_COLOR,
                    scene_bg_color=(1, 1, 1),
                )
                * 255
            ).astype(np.uint8)

        with tempfile.TemporaryDirectory(prefix="convert_smpl_render_") as tmp_root:
            tmp_root = Path(tmp_root)
            frame_paths, src_fps = resolve_source(source, tmp_root)

            # Single-image input → write a PNG, not a video.
            if len(frame_paths) == 1 and N == 1:
                bgr = cv2.imread(str(frame_paths[0]))
                if bgr is None:
                    raise RuntimeError(f"cannot read {frame_paths[0]}")
                if valid_mask[0] and not np.isnan(smpl_verts_cam[0]).any():
                    bgr = _render_one(0, bgr)
                png_path = out_path.with_suffix(".png")
                cv2.imwrite(str(png_path), bgr)
                return

            fps = float(fps or src_fps or 30.0)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
            if not writer.isOpened():
                raise RuntimeError(f"cannot open video writer at {out_path}")
            try:
                for i in tqdm(range(N), desc="Rendering SMPL overlay"):
                    bgr = cv2.imread(str(frame_paths[i]))
                    if bgr is None:
                        continue
                    if not valid_mask[i] or np.isnan(smpl_verts_cam[i]).any():
                        writer.write(bgr)
                        continue
                    writer.write(_render_one(i, bgr))
            finally:
                writer.release()


# ------------------------------ CLI -------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert MHR predictions (body_params.npz) to better_human SMPL parameters.",
    )
    p.add_argument(
        "--source", required=True,
        help="video file or folder of frames (same source used upstream)",
    )
    p.add_argument(
        "--predictions", required=True,
        help="folder containing body_params.npz (from estimate_body_sam3d.py)",
    )
    p.add_argument(
        "--smpl-model", default=None,
        help=f"path to better_human SMPL .npz (default: {_DEFAULT_SMPL_MODEL})",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--iters", type=int, default=300, help="fine-stage iterations")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--person", type=int, default=0,
        help="which tracked person to convert from the stacked sam3d/params.npz "
             "(default: 0, the primary tracked person)",
    )
    p.add_argument(
        "--render-video", action="store_true",
        help="render SMPL mesh overlay .mp4 into predictions folder as <stem>_smpl.mp4",
    )
    p.add_argument(
        "--fps", type=float, default=None,
        help="output video fps (default: source video fps, or 30 for folders)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    script = MhrToSmplScript(
        device=args.device,
        smpl_model_path=args.smpl_model,
        num_iters=args.iters,
        batch_size=args.batch_size,
    )
    result = script.run(
        source=args.source,
        predictions_dir=args.predictions,
        render_video=args.render_video,
        video_fps=args.fps,
        person=args.person,
    )
    msg = (
        f"converted {result.num_valid}/{result.num_frames} frames "
        f"({result.width}x{result.height}), median fit error "
        f"{result.median_fit_error_m*1000:.1f} mm -> {result.output_path}"
    )
    if result.rendered_video is not None:
        msg += f"\nrendered SMPL overlay video -> {result.rendered_video}"
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
