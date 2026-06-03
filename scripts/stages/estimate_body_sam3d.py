"""SAM-3D-Body MHR fit over tracked video frames (single person).

For each frame, runs SAM-3D-Body on the tracked person's bbox + mask using
**per-frame camera intrinsics from the MoGe2 camera stage**
(``moge2/camera.npz``). Camera estimation is therefore decoupled from body
fitting; if ``moge2/camera.npz`` is absent the stage falls back to running
MoGe2 once on frame 0 (run ``estimate_camera_moge2.py`` first for proper
per-frame intrinsics).

Run from CLI::

    uv run python scripts/stages/estimate_body_sam3d.py \\
        --source path/to/{video.mp4|frames_folder|photo.jpg} \\
        --predictions out/video/ \\
        [--obj-id 0] [--hf-repo facebook/sam-3d-body-dinov3] \\
        [--device cuda] [--inference-type full] [--render-video]

Inputs (under ``--predictions``, written by the sam3 + moge2 stages):
    sam3/bboxes.npz       tracked bboxes (bboxes_per_obj / legacy bboxes)
    sam3/<obj_id:02d>/    per-frame person masks
    moge2/camera.npz      per-frame intrinsics (optional; see fallback above)

Outputs (into ``--predictions``):
    body_params.npz       per-frame MHR pose/shape params (single person)
    <source-stem>_mesh.mp4   mesh overlay video (only with --render-video)
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
# scripts/ on path so `import _pym_preload` works; repo root for `tools.*`.
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT))
import _pym_preload  # noqa: E402, F401  (must be first — re-execs to fix libstdc++ ABI)

import argparse  # noqa: E402
import gc  # noqa: E402
import importlib.util  # noqa: E402
import tempfile  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from tools.source import find_frame_file, resolve_source  # noqa: E402

_SAM3D_ROOT = _REPO_ROOT / "third_party" / "sam-3d-body"


def _import_from_sam3d(module_name: str, file_path: Path):
    """Import a module from sam-3d-body by explicit path (avoids venv 'tools' clash)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Keys to extract from per-person output dict (shapes discovered at runtime).
_SAVE_KEYS = (
    "focal_length",
    "pred_cam_t",
    "pred_keypoints_3d",
    "pred_keypoints_2d",
    "pred_pose_raw",
    "global_rot",
    "body_pose_params",
    "hand_pose_params",
    "shape_params",
    "scale_params",
    "pred_joint_coords",
    "pred_global_rots",
    "mhr_model_params",
)

_MESH_COLOR = (0.65098039, 0.74117647, 0.85882353)  # light blue (sam-3d-body default)


@dataclass
class BodyEstimateResult:
    num_frames: int
    height: int
    width: int
    output_path: Path
    camera_source: str
    rendered_video: Path | None = None


class Sam3dBodyEstimator:
    """SAM-3D-Body estimator wrapper for per-frame single-person body fit."""

    def __init__(
        self,
        hf_repo: str = "facebook/sam-3d-body-dinov3",
        device: str = "cuda",
        inference_type: str = "full",
    ):
        self.hf_repo = hf_repo
        self.device = device
        self.inference_type = inference_type
        self._estimator = None
        self._fov_estimator = None

    def _load_model(self):
        if self._estimator is None:
            from sam_3d_body import SAM3DBodyEstimator as _Estimator
            from sam_3d_body import load_sam_3d_body_hf

            model, model_cfg = load_sam_3d_body_hf(self.hf_repo)
            if self.device != "cuda":
                model = model.to(self.device)
            # No detector/segmentor/fov_estimator — we supply bboxes, masks,
            # and cam_int ourselves.
            self._estimator = _Estimator(sam_3d_body_model=model, model_cfg=model_cfg)
        return self._estimator

    def _load_fov_estimator(self):
        if self._fov_estimator is None:
            fov_mod = _import_from_sam3d(
                "build_fov_estimator",
                _SAM3D_ROOT / "tools" / "build_fov_estimator.py",
            )
            self._fov_estimator = fov_mod.FOVEstimator(name="moge2", device=self.device)
        return self._fov_estimator

    def _release_fov_estimator(self):
        if self._fov_estimator is not None:
            del self._fov_estimator
            self._fov_estimator = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _resolve_cam_intrinsics(
        self, predictions_dir: Path, frame_paths: list[Path], *, H: int, W: int, N: int
    ) -> tuple[list[torch.Tensor], np.ndarray, str]:
        """Per-frame pixel-unit ``(1, 3, 3)`` intrinsics for the whole clip.

        Prefers ``moge2/camera.npz`` (the MoGe2 camera stage, per-frame).
        Falls back to running MoGe2 once on frame 0 if that file is absent.
        Returns ``(cam_int_per_frame, frame0_K_np, source_str)``.
        """
        cam_npz = predictions_dir / "moge2" / "camera.npz"
        if cam_npz.exists():
            from tools.adapters.moge2 import denormalize_K

            cd = np.load(cam_npz)
            intr_norm = np.asarray(cd["intrinsics"], dtype=np.float32)  # (N, 3, 3) normalized
            if intr_norm.shape[0] != N:
                raise RuntimeError(
                    f"{cam_npz} has {intr_norm.shape[0]} frames but bboxes.npz says {N}"
                )
            cam_int_list = [
                torch.from_numpy(denormalize_K(intr_norm[i], H, W)).unsqueeze(0).to(self.device)
                for i in range(N)
            ]
            frame0 = cam_int_list[0].squeeze(0).cpu().numpy().astype(np.float32)
            return cam_int_list, frame0, "moge2/camera.npz (per-frame)"

        # Fallback: estimate FOV once on frame 0 and reuse it for every frame.
        fov_est = self._load_fov_estimator()
        img0_bgr = cv2.imread(str(frame_paths[0]))
        if img0_bgr is None:
            raise RuntimeError(f"cannot read frame {frame_paths[0]}")
        img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
        cam_int = fov_est.get_cam_intrinsics(img0_rgb).to(self.device)  # (1, 3, 3)
        self._release_fov_estimator()
        frame0 = cam_int.squeeze(0).cpu().numpy().astype(np.float32)
        return [cam_int] * N, frame0, "MoGe2 frame-0 (fallback; run estimate_camera_moge2 for per-frame K)"

    def estimate(
        self,
        source: str | Path,
        predictions_dir: str | Path,
        render_video: bool = False,
        video_fps: float | None = None,
        obj_id: int | None = 0,
    ) -> BodyEstimateResult:
        """Run SAM-3D-Body per-frame for one tracked person; save body_params.npz."""
        source = Path(source)
        predictions_dir = Path(predictions_dir)

        bboxes_path = predictions_dir / "sam3" / "bboxes.npz"
        masks_dir = predictions_dir / "sam3"
        if not bboxes_path.exists():
            raise FileNotFoundError(f"sam3/bboxes.npz not found in {predictions_dir}")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"sam3/ not found in {predictions_dir}")

        data = np.load(bboxes_path, allow_pickle=True)
        frame_indices = data["frame_indices"]  # (N,) int32
        if obj_id is None:
            if "bboxes" not in data.files:
                raise RuntimeError(
                    f"{bboxes_path} has no legacy 'bboxes' field "
                    f"(num_objects={int(data.get('num_objects', 0))}); pass --obj-id"
                )
            bboxes = data["bboxes"]  # (N, 4) int32 xyxy
            masks_subdir = masks_dir
        else:
            bboxes = data["bboxes_per_obj"][obj_id]  # (N, 4)
            masks_subdir = masks_dir / f"{obj_id:02d}"
        H = int(data["video_height"])
        W = int(data["video_width"])
        N = int(data["num_frames"])

        with tempfile.TemporaryDirectory(prefix="sam3d_body_") as tmp_root:
            tmp_root = Path(tmp_root)
            frame_paths, src_fps = resolve_source(source, tmp_root)
            if len(frame_paths) != N:
                raise RuntimeError(
                    f"frame count mismatch: source has {len(frame_paths)} frames "
                    f"but bboxes.npz says {N}"
                )

            estimator = self._load_model()
            cam_int_list, cam_int_np, cam_source = self._resolve_cam_intrinsics(
                predictions_dir, frame_paths, H=H, W=W, N=N
            )
            print(f"  camera intrinsics: {cam_source}")

            # ---- optional video / png writer ----
            single_image_render = render_video and N == 1 and src_fps is None
            writer: cv2.VideoWriter | None = None
            rendered_video_path: Path | None = None
            faces = None
            if render_video:
                from sam_3d_body.visualization.renderer import Renderer

                faces = estimator.faces  # (F, 3) int array
                if single_image_render:
                    rendered_video_path = predictions_dir / f"{source.stem}_mesh.png"
                else:
                    rendered_video_path = predictions_dir / f"{source.stem}_mesh.mp4"
                    fps_out = float(video_fps or src_fps or 30.0)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(rendered_video_path), fourcc, fps_out, (W, H))
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open video writer at {rendered_video_path}")

            results: dict[str, np.ndarray] | None = None
            valid_mask = np.zeros(N, dtype=bool)

            def _write_plain_frame(idx: int) -> None:
                if writer is None:
                    return
                bgr = cv2.imread(str(frame_paths[idx]))
                if bgr is not None:
                    writer.write(bgr)

            for i in tqdm(range(N), desc="Estimating body params"):
                idx = int(frame_indices[i])
                bbox = bboxes[i]
                if np.all(bbox == 0):
                    _write_plain_frame(i)
                    continue

                mask_path = find_frame_file(masks_subdir, idx, source=source, suffix=".png")
                mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
                if mask_img is None:
                    _write_plain_frame(i)
                    continue
                mask = mask_img.reshape(1, H, W).astype(np.uint8)  # (1, H, W)
                bbox_f = bbox.reshape(1, 4).astype(np.float32)

                outputs = estimator.process_one_image(
                    str(frame_paths[i]),
                    bboxes=bbox_f,
                    masks=mask,
                    cam_int=cam_int_list[i],
                    inference_type=self.inference_type,
                )
                if not outputs:
                    _write_plain_frame(i)
                    continue

                person = outputs[0]
                if results is None:
                    results = {}
                    for key in _SAVE_KEYS:
                        val = np.asarray(person[key], dtype=np.float32)
                        results[key] = np.full((N, *val.shape), np.nan, dtype=np.float32)
                for key in _SAVE_KEYS:
                    results[key][i] = np.asarray(person[key], dtype=np.float32)
                valid_mask[i] = True

                if writer is not None or single_image_render:
                    frame_bgr = cv2.imread(str(frame_paths[i]))
                    if frame_bgr is not None:
                        mesh_renderer = Renderer(focal_length=person["focal_length"], faces=faces)
                        overlay = (
                            mesh_renderer(
                                person["pred_vertices"],
                                person["pred_cam_t"],
                                frame_bgr,
                                mesh_base_color=_MESH_COLOR,
                                scene_bg_color=(1, 1, 1),
                            )
                            * 255
                        ).astype(np.uint8)
                        if writer is not None:
                            writer.write(overlay)
                        else:
                            cv2.imwrite(str(rendered_video_path), overlay)

            if writer is not None:
                writer.release()

        # ---- save (flat, so convert_mhr_to_smpl finds it) ----
        output_path = predictions_dir / "body_params.npz"
        save_dict: dict[str, np.ndarray] = {}
        if results is not None:
            save_dict.update(results)
        save_dict.update(
            cam_intrinsics=cam_int_np,
            valid_mask=valid_mask,
            num_frames=np.int32(N),
            video_height=np.int32(H),
            video_width=np.int32(W),
        )
        np.savez(output_path, **save_dict)

        return BodyEstimateResult(
            num_frames=N,
            height=H,
            width=W,
            output_path=output_path,
            camera_source=cam_source,
            rendered_video=rendered_video_path,
        )


# ------------------------------ CLI -------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate single-person 3D body (MHR) with SAM-3D-Body over tracked frames.",
    )
    p.add_argument(
        "--source", required=True,
        help="video file or folder of frames (same source used for tracking)",
    )
    p.add_argument(
        "--predictions", required=True,
        help="sam3 + moge2 output folder (contains sam3/bboxes.npz, sam3/<obj>/, moge2/camera.npz)",
    )
    p.add_argument(
        "--obj-id", type=int, default=0,
        help="tracked object to fit: reads bboxes_per_obj[obj_id] + sam3/<obj_id:02d>/ "
             "(default: 0, the primary tracked person). Use --obj-id -1 for the legacy "
             "flat 'bboxes' + sam3/ layout.",
    )
    p.add_argument(
        "--hf-repo", default="facebook/sam-3d-body-dinov3",
        help="HuggingFace repo ID for SAM-3D-Body (default: %(default)s)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--inference-type", default="full", choices=["full", "body"],
        help="'full' for body+hands, 'body' for body only (default: %(default)s)",
    )
    p.add_argument(
        "--render-video", action="store_true",
        help="render mesh overlay into predictions folder as <stem>_mesh.mp4",
    )
    p.add_argument(
        "--fps", type=float, default=None,
        help="output video fps (default: source fps, or 30 for frame folders)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    obj_id = None if args.obj_id is not None and args.obj_id < 0 else args.obj_id
    est = Sam3dBodyEstimator(
        hf_repo=args.hf_repo,
        device=args.device,
        inference_type=args.inference_type,
    )
    result = est.estimate(
        source=args.source,
        predictions_dir=args.predictions,
        render_video=args.render_video,
        video_fps=args.fps,
        obj_id=obj_id,
    )
    msg = (
        f"estimated body params for {result.num_frames} frames "
        f"({result.width}x{result.height}) [{result.camera_source}] -> {result.output_path}"
    )
    if result.rendered_video is not None:
        msg += f"\nrendered mesh overlay -> {result.rendered_video}"
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
