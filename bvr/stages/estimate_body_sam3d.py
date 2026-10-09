"""SAM-3D-Body MHR fit over tracked video frames (all tracked people).

For each frame, runs SAM-3D-Body on **every** tracked person's bbox + mask
(``--obj-ids`` defaults to all SAM 3 objects) using **per-frame camera
intrinsics from VGGT-Omega** at the original frame resolution. VGGT geometry is
up-to-scale, but the camera *intrinsics* are scale-invariant (the per-clip scale
multiplies depth + camera translation, never the focal length / principal
point), and SAM-3D-Body consumes only intrinsics — it predicts each person's
camera-frame translation itself — so the scaleless VGGT camera is exact here.
Intrinsics come from the fuse stage's ``geometry/transform.npz``
(``intrinsics_px_orig``) when present, else ``vggt/camera.npz`` (normalized,
**denormalized to the original resolution** on the fly). Camera estimation is
**VGGT-only**.

Two run modes:

  * **single clip** — ``--source clip --predictions out/video/``; reads the
    ``sam3/`` + ``vggt/`` (or ``geometry/``) siblings under ``--predictions``.
  * **batch / manifest** — ``--source manifest.json`` with ``[sam3d].save_dir`` (a
    ``precompute_runner`` Step): loads SAM-3D-Body once and sweeps every scene,
    taking each scene's ``bboxes`` / ``camera`` paths from the manifest. This is
    how ``scripts/corpus.py`` fans the stage across the ClimbingVideos
    scene DB, where ``sam3`` / ``vggt`` live in their own feature trees.

Run from CLI::

    uv run python bvr/stages/estimate_body_sam3d.py \\
        --source path/to/{video.mp4|frames_folder|photo.jpg} \\
        --predictions out/video/ \\
        --config configs/pipeline.toml

Inputs:
    sam3/bboxes.npz          tracked bboxes (bboxes_per_obj / legacy bboxes)
    sam3/<obj_id:02d>/       per-frame person masks (one dir per tracked person)
    geometry/transform.npz   VGGT per-frame intrinsics at orig res (preferred)
    vggt/camera.npz          VGGT intrinsics, denormalized on the fly (fallback)

Outputs (``sam3d/`` under ``--predictions``, or the manifest ``out_dir`` /
``[sam3d].subdir`` in batch mode):
    sam3d/params.npz      per-frame MHR pose/shape params for every tracked
                          person, stacked along a leading axis ``(P, N, ...)``
                          (``object_ids (P,)`` records which SAM 3 id each row is)
    sam3d/overlay.mp4     combined mesh overlay — all people composited per
                          frame, one colour each (on by default; --no-overlay)
"""

from __future__ import annotations

from pathlib import Path

from bvr.stages import _pym_preload  # noqa: F401  (must be first — re-execs to fix libstdc++ ABI)

import argparse
import gc
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from tqdm import tqdm

from bvr import determinism
from bvr.paths import REPO_ROOT
from bvr.config import load_toml, section
from bvr.geometry.intrinsics import denormalize_K
from bvr.precompute_runner import InputItem, SkipItem, run_batch
from bvr.source import find_frame_file, resolve_source


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

# Per-person overlay colours (RGB) so people are distinguishable in the combined
# overlay; cycles if there are more people than colours. Index 0 is the
# sam-3d-body default light blue.
_MESH_COLORS = (
    (0.65098039, 0.74117647, 0.85882353),  # light blue
    (0.95, 0.60, 0.40),                    # salmon
    (0.60, 0.85, 0.55),                    # light green
    (0.80, 0.65, 0.88),                    # lilac
    (0.95, 0.85, 0.45),                    # gold
)


@dataclass
class BodyEstimateResult:
    num_frames: int
    num_people: int
    height: int
    width: int
    output_path: Path
    camera_source: str
    rendered_video: Path | None = None


class Sam3dBodyEstimator:
    """SAM-3D-Body estimator wrapper for per-frame multi-person body fit."""

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
            # process_one_image prints two lines and flushes the CUDA cache on
            # EVERY call (one per person per frame); over a corpus that is a
            # sync + re-allocation storm, so neutralize both in this process.
            import sam_3d_body.sam_3d_body_estimator as _estimator_module

            _estimator_module.print = lambda *args, **kwargs: None
            torch.cuda.empty_cache = lambda: None
        return self._estimator

    def _resolve_cam_intrinsics(
        self, *, geometry_npz: Path | None, vggt_npz: Path | None,
        H: int, W: int, frame_indices: np.ndarray,
    ) -> tuple[list[torch.Tensor], np.ndarray, str]:
        """Per-frame pixel-unit ``(1, 3, 3)`` intrinsics for the whole clip.

        **VGGT-Omega only** (its intrinsics are scale-invariant, so the
        up-to-scale ambiguity is irrelevant): prefers the fuse stage's
        ``geometry/transform.npz`` (``intrinsics_px_orig`` — already at orig
        res), else ``vggt/camera.npz`` (normalized) denormalized on the fly to
        ``(H, W)``. Raises if neither VGGT source covers all tracked frames.
        Returns ``(cam_int_per_frame, frame0_K_np,
        source_str)``.
        """
        def _from_idx_map(idx2k: dict[int, np.ndarray]) -> list[torch.Tensor] | None:
            """Build the per-frame list iff every tracked frame is covered."""
            if not all(int(fi) in idx2k for fi in frame_indices):
                return None
            return [
                torch.from_numpy(idx2k[int(fi)].astype(np.float32)).unsqueeze(0).to(self.device)
                for fi in frame_indices
            ]

        # Priority 1: VGGT intrinsics at original resolution (fuse stage output).
        if geometry_npz is not None and geometry_npz.exists():
            gd = np.load(geometry_npz)
            K_orig = np.asarray(gd["intrinsics_px_orig"], dtype=np.float32)  # (M,3,3) px @ orig res
            gi = np.asarray(gd["frame_indices"], dtype=np.int64)
            cam_int_list = _from_idx_map({int(f): K_orig[j] for j, f in enumerate(gi)})
            if cam_int_list is not None:
                frame0 = cam_int_list[0].squeeze(0).cpu().numpy().astype(np.float32)
                return cam_int_list, frame0, "geometry/transform.npz (VGGT, per-frame @ orig res)"
            print("  [body] geometry/transform.npz doesn't cover all tracked frames — falling back")

        # Priority 2: VGGT camera (normalized) → denormalize to (H, W).
        if vggt_npz is not None and vggt_npz.exists():
            vd = np.load(vggt_npz)
            Kn = np.asarray(vd["intrinsics"], dtype=np.float32)             # (M,3,3) normalized
            vi = np.asarray(vd["frame_indices"], dtype=np.int64)
            cam_int_list = _from_idx_map({int(f): denormalize_K(Kn[j], H, W) for j, f in enumerate(vi)})
            if cam_int_list is not None:
                frame0 = cam_int_list[0].squeeze(0).cpu().numpy().astype(np.float32)
                return cam_int_list, frame0, "vggt/camera.npz (VGGT, denormalized to orig res)"
            print("  [body] vggt/camera.npz doesn't cover all tracked frames")

        raise RuntimeError(
            "no usable VGGT camera: need geometry/transform.npz (intrinsics_px_orig) "
            "or vggt/camera.npz covering all tracked frames "
            f"(geometry={geometry_npz}, vggt={vggt_npz})."
        )

    def _resolve_people(
        self, data, masks_dir: Path, obj_ids: list[int] | None,
    ) -> list[tuple[int, np.ndarray, Path]]:
        """Resolve which tracked people to fit.

        Returns ``[(object_id, bboxes (N,4), mask_subdir), ...]`` — one entry per
        person. ``obj_ids=None`` selects **all** tracked objects (the default);
        pass a list to restrict. Falls back to the legacy flat ``bboxes`` layout
        (a single object in ``sam3/``) when ``bboxes_per_obj`` is absent.
        """
        if "bboxes_per_obj" in data.files:
            all_ids = [int(o) for o in data["object_ids"]]
            per_obj = data["bboxes_per_obj"]  # (P_all, N, 4)
            if obj_ids:
                missing = [o for o in obj_ids if o not in all_ids]
                if missing:
                    raise RuntimeError(
                        f"requested --obj-ids {missing} not tracked (have {all_ids})"
                    )
                sel = [o for o in obj_ids if o in all_ids]
            else:
                sel = all_ids
            return [(o, per_obj[all_ids.index(o)], masks_dir / f"{o:02d}") for o in sel]
        if "bboxes" not in data.files:
            raise RuntimeError(
                f"{masks_dir.parent / 'sam3' / 'bboxes.npz'} has neither "
                "'bboxes_per_obj' nor a legacy 'bboxes' field"
            )
        return [(0, data["bboxes"], masks_dir)]

    def estimate(
        self,
        source: str | Path,
        predictions_dir: str | Path | None = None,
        *,
        sam3_dir: str | Path | None = None,
        out_dir: str | Path | None = None,
        camera_npz: str | Path | None = None,
        render_video: bool = False,
        video_fps: float | None = None,
        obj_ids: list[int] | None = None,
    ) -> BodyEstimateResult:
        """Run SAM-3D-Body per-frame for every tracked person; save sam3d/params.npz.

        All requested people (``obj_ids=None`` → all tracked) are fit and stacked
        along a leading person axis ``(P, N, ...)``. When ``render_video`` is on,
        a single combined overlay is written with every person composited onto
        each frame (one mesh colour per person).

        Paths: pass ``predictions_dir`` (single-clip layout — reads its ``sam3``
        / ``vggt`` / ``geometry`` siblings, writes ``sam3d/``), or the explicit
        ``sam3_dir`` (holds ``bboxes.npz`` + ``<obj>/`` masks), ``camera_npz``
        (a VGGT ``camera.npz``) and ``out_dir`` (batch / manifest layout, where
        those live in separate feature trees). Explicit paths win.
        """
        source = Path(source)
        pred = Path(predictions_dir) if predictions_dir is not None else None
        # Explicit paths (batch / manifest) win; else derive from --predictions.
        if sam3_dir is not None:
            sam3_dir = Path(sam3_dir)
        elif pred is not None:
            sam3_dir = pred / "sam3"
        else:
            raise ValueError("estimate() needs predictions_dir or an explicit sam3_dir")
        sam3d_dir = Path(out_dir) if out_dir is not None else (
            pred / "sam3d" if pred is not None else None)
        if sam3d_dir is None:
            raise ValueError("estimate() needs predictions_dir or an explicit out_dir")
        # Camera intrinsics are VGGT-only (scale-invariant):
        # an explicit vggt camera.npz wins, else the geometry/vggt siblings.
        geometry_npz = (pred / "geometry" / "transform.npz") if pred is not None else None
        vggt_npz = Path(camera_npz) if camera_npz is not None else (
            pred / "vggt" / "camera.npz" if pred is not None else None)

        bboxes_path = sam3_dir / "bboxes.npz"
        masks_dir = sam3_dir
        if not bboxes_path.exists():
            raise FileNotFoundError(f"bboxes.npz not found: {bboxes_path}")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"sam3 masks dir not found: {masks_dir}")

        data = np.load(bboxes_path, allow_pickle=True)
        frame_indices = data["frame_indices"]  # (N,) int32
        H = int(data["video_height"])
        W = int(data["video_width"])
        N = int(data["num_frames"])

        people = self._resolve_people(data, masks_dir, obj_ids)
        P = len(people)
        obj_ids_arr = np.array([o for o, _, _ in people], dtype=np.int32)
        print(f"  fitting {P} person(s): obj_ids={obj_ids_arr.tolist()}")

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
                geometry_npz=geometry_npz, vggt_npz=vggt_npz,
                H=H, W=W, frame_indices=frame_indices,
            )
            print(f"  camera intrinsics: {cam_source}")

            # ---- optional combined overlay (all people) writer ----
            single_image_render = render_video and N == 1 and src_fps is None
            writer: cv2.VideoWriter | None = None
            rendered_video_path: Path | None = None
            faces = None
            Renderer = None
            if render_video:
                from sam_3d_body.visualization.renderer import Renderer as _Renderer

                Renderer = _Renderer
                sam3d_dir.mkdir(parents=True, exist_ok=True)
                faces = estimator.faces  # (F, 3) int array
                if single_image_render:
                    rendered_video_path = sam3d_dir / "overlay.png"
                else:
                    rendered_video_path = sam3d_dir / "overlay.mp4"
                    fps_out = float(video_fps or src_fps or 30.0)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(rendered_video_path), fourcc, fps_out, (W, H))
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open video writer at {rendered_video_path}")

            results: dict[str, np.ndarray] | None = None
            valid_mask = np.zeros((P, N), dtype=bool)

            for i in tqdm(range(N), desc="Estimating body params"):
                idx = int(frame_indices[i])
                overlay_bgr = None  # lazily copied from the decoded frame on first render
                # Decode the frame ONCE for every person and the overlay (the
                # estimator re-reads a path per call). It expects RGB arrays.
                frame_bgr = cv2.imread(str(frame_paths[i]))
                if frame_bgr is None:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                for pi, (obj, bseq, msub) in enumerate(people):
                    bbox = bseq[i]
                    if np.all(bbox == 0):
                        continue
                    mask_path = find_frame_file(msub, idx, source=source, suffix=".png")
                    mask_img = (
                        cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
                    )
                    if mask_img is None:
                        continue
                    mask = mask_img.reshape(1, H, W).astype(np.uint8)  # (1, H, W)
                    bbox_f = bbox.reshape(1, 4).astype(np.float32)

                    outputs = estimator.process_one_image(
                        frame_rgb,
                        bboxes=bbox_f,
                        masks=mask,
                        cam_int=cam_int_list[i],
                        inference_type=self.inference_type,
                    )
                    if not outputs:
                        continue

                    person = outputs[0]
                    if results is None:
                        results = {}
                        for key in _SAVE_KEYS:
                            val = np.asarray(person[key], dtype=np.float32)
                            results[key] = np.full((P, N, *val.shape), np.nan, dtype=np.float32)
                    for key in _SAVE_KEYS:
                        results[key][pi, i] = np.asarray(person[key], dtype=np.float32)
                    valid_mask[pi, i] = True

                    if render_video:
                        if overlay_bgr is None:
                            overlay_bgr = frame_bgr.copy()
                        if overlay_bgr is not None:
                            color = _MESH_COLORS[pi % len(_MESH_COLORS)]
                            mesh_renderer = Renderer(
                                focal_length=person["focal_length"], faces=faces
                            )
                            overlay_bgr = (
                                mesh_renderer(
                                    person["pred_vertices"],
                                    person["pred_cam_t"],
                                    overlay_bgr,
                                    mesh_base_color=color,
                                    scene_bg_color=(1, 1, 1),
                                )
                                * 255
                            ).astype(np.uint8)

                # ---- emit the (possibly multi-person) overlay frame ----
                if render_video:
                    if overlay_bgr is None:  # no valid person this frame → plain frame
                        overlay_bgr = frame_bgr
                    if single_image_render:
                        if overlay_bgr is not None:
                            cv2.imwrite(str(rendered_video_path), overlay_bgr)
                    elif writer is not None and overlay_bgr is not None:
                        writer.write(overlay_bgr)

            if writer is not None:
                writer.release()

        # ---- save stacked params under sam3d/ (named after the model) ----
        sam3d_dir.mkdir(parents=True, exist_ok=True)
        output_path = sam3d_dir / "params.npz"
        save_dict: dict[str, np.ndarray] = {}
        if results is not None:
            save_dict.update(results)
        save_dict.update(
            cam_intrinsics=cam_int_np,
            valid_mask=valid_mask,
            object_ids=obj_ids_arr,
            num_people=np.int32(P),
            num_frames=np.int32(N),
            video_height=np.int32(H),
            video_width=np.int32(W),
        )
        np.savez(output_path, **save_dict)

        return BodyEstimateResult(
            num_frames=N,
            num_people=P,
            height=H,
            width=W,
            output_path=output_path,
            camera_source=cam_source,
            rendered_video=rendered_video_path,
        )


class Sam3dBodyStep:
    """precompute_runner Step: load SAM-3D-Body once, fit each manifest scene.

    Batch / manifest mode for the ClimbingVideos scene DB. Each
    :class:`InputItem` carries the clip (``source_path``), the output dir
    (``out_dir``), the sam3 ``bboxes.npz`` (``bboxes_path`` — its parent holds
    the ``<obj>/`` mask dirs) and the VGGT ``camera.npz`` (``camera_path``).
    ``--subdir`` picks the subfolder of ``out_dir`` to write into (``.`` → write
    straight into it, as ``corpus.py`` does for the stage-first tree).
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.est = Sam3dBodyEstimator(
            hf_repo=args.hf_repo,
            device=args.device,
            inference_type=args.inference_type,
        )

    def load(self) -> None:
        self.est._load_model()

    def release(self) -> None:
        self.est._estimator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process(self, item: InputItem) -> None:
        subdir = getattr(self.args, "subdir", "sam3d") or "sam3d"
        out_dir = item.out_dir if subdir in (".", "") else item.out_dir / subdir
        if self.args.skip_existing and (out_dir / "params.npz").exists():
            raise SkipItem(f"sam3d already done: {out_dir / 'params.npz'}")
        if item.bboxes_path is None:
            raise SkipItem(f"no sam3 bboxes path in manifest for {item.stem}")
        self.est.estimate(
            source=item.source_path,
            sam3_dir=item.bboxes_path.parent,   # bboxes.npz + <obj>/ masks live here
            camera_npz=item.camera_path,
            out_dir=out_dir,
            render_video=self.args.overlay,
            video_fps=self.args.fps,
            obj_ids=self.args.obj_ids,
        )


# ------------------------------ CLI -------------------------------------------


def _parse_obj_ids(s: str) -> list[int] | None:
    """``""`` → None (all tracked people); ``"0,1"`` → [0, 1]."""
    s = s.strip()
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate 3D bodies (MHR) for all tracked people with SAM-3D-Body.",
    )
    p.add_argument(
        "--source", required=True,
        help="single clip (video / folder of frames / image), or a .json manifest",
    )
    p.add_argument(
        "--predictions", default=None,
        help="single-clip mode: folder holding sam3/bboxes.npz + sam3/<obj>/ + "
             "vggt/camera.npz (or geometry/transform.npz); writes sam3d/ here",
    )
    p.add_argument(
        "--save-dir", default=None, type=Path,
        help="manifest batch mode: output root (per-item outputs under <save-dir>/<stem>/)",
    )
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.toml",
                   help="pipeline TOML (default: configs/pipeline.toml)")
    cli = p.parse_args(argv)
    cfg = load_toml(cli.config)
    args = section(cfg, "sam3d", {
        "subdir": "sam3d",
        "skip_existing": False,
        "obj_ids": "",
        "hf_repo": "facebook/sam-3d-body-dinov3",
        "inference_type": "full",
        "overlay": True,
        "fps": "",
    })
    args.device = section(cfg, "runtime", {"device": "cuda"}).device
    args.source, args.predictions, args.config = cli.source, cli.predictions, cli.config
    args.save_dir = cli.save_dir
    args.obj_ids = (_parse_obj_ids(args.obj_ids) if isinstance(args.obj_ids, str)
                    else list(args.obj_ids) or None)
    args.fps = float(args.fps) if args.fps != "" else None
    return args


def main(argv: list[str] | None = None) -> int:
    determinism.apply()
    args = _parse_args(argv)

    # Batch / manifest mode: precompute_runner Step (load model once, sweep scenes).
    if args.save_dir is not None:
        return run_batch(Sam3dBodyStep(args), args)

    # Single-clip mode.
    if args.predictions is None:
        raise SystemExit(
            "estimate_body_sam3d: pass --predictions <clip-dir> (single clip) "
            "or set sam3d.save_dir in the config for a .json --source manifest (batch)."
        )
    est = Sam3dBodyEstimator(
        hf_repo=args.hf_repo,
        device=args.device,
        inference_type=args.inference_type,
    )
    result = est.estimate(
        source=args.source,
        predictions_dir=args.predictions,
        render_video=args.overlay,
        video_fps=args.fps,
        obj_ids=args.obj_ids,
    )
    msg = (
        f"estimated body params for {result.num_people} person(s) × {result.num_frames} "
        f"frames ({result.width}x{result.height}) [{result.camera_source}] -> {result.output_path}"
    )
    if result.rendered_video is not None:
        msg += f"\nrendered mesh overlay -> {result.rendered_video}"
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
