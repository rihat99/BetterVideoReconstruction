"""Convert SAM-3D-Body MHR predictions to SMPL-X parameters (pip ``smplx``).

SMPL-X adds the hands (15 + 15 joints) and face channels (jaw / eyes /
expression) on top of the 21-joint body. ``better_human``'s native ``SMPLX`` has no jaw/eye/face
deformation yet, so this stage fits the **standard ``smplx`` PyPI package**
(loading ``better_human/models/smplx/SMPLX_NEUTRAL.npz``) via the unchanged
``bvr.mhr_to_smplx.MhrToSmplxConverter``. Jaw/eye/expression are left at
zero (MHR doesn't predict them); the BetterRobot-friendly ``q`` packing is
intentionally deferred, so this stage is the **end of the default pipeline**
(the SMPL → motion/dynamics path stays SMPL-only and runs standalone).

Every tracked person from ``sam3d/params.npz`` (stacked ``(P, N, ...)``) is
converted; the outputs are stacked the same way and written **inside**
``sam3d/`` (alongside the body params), not at the prediction-folder root.

The conversion reads only ``params.npz`` (MHR params + camera) — the ``source``
video is needed **only for the overlay**, so ``--no-overlay`` runs need no frames.

Two run modes (like ``estimate_body_sam3d``):

  * **single clip** — ``--source clip --predictions out/video/``; reads/writes
    ``out/video/sam3d/``.
  * **batch / manifest** — ``--source manifest.json`` with ``[smplx].save_dir`` (a
    ``precompute_runner`` Step): builds the MHR→SMPL-X converter once and sweeps
    every scene, each manifest ``out_dir`` being that scene's sam3d feature dir.
    This is how ``datasets/climbingvideos/corpus.py`` fans the stage across the
    ClimbingVideos scene DB.

Run from CLI::

    uv run python bvr/stages/convert_mhr_to_smplx.py \\
        --source path/to/{video.mp4|frames_folder|photo.jpg} \\
        --predictions out/video/ \\
        --config configs/pipeline.toml

Inputs:
    source             video file, folder of frames, or single image
    predictions_dir    folder containing sam3d/params.npz (from estimate_body_sam3d.py)

Outputs (into <predictions_dir>/sam3d/):
    sam3d/smplx_params.npz   per-person/-frame SMPL-X params, stacked (P, N, ...)
                             (betas, body_pose, global_orient, transl,
                             left/right_hand_pose, jaw/leye/reye_pose, expression;
                             object_ids (P,) records which SAM 3 id each row is)
    sam3d/smplx_overlay.mp4  combined SMPL-X overlay — all people composited per
                             frame, one colour each (on by default; --no-overlay)
"""

from __future__ import annotations

from pathlib import Path

from bvr.stages import _pym_preload  # noqa: F401  (must be first — re-execs to fix libstdc++ ABI)

import argparse
import gc
import tempfile
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from tqdm import tqdm

from bvr import determinism
from bvr.paths import REPO_ROOT
from bvr.config import load_config, section
from bvr.precompute_runner import InputItem, SkipItem, run_batch

_MAPPING_NPZ = (
    REPO_ROOT
    / "third_party"
    / "MHR"
    / "tools"
    / "mhr_smpl_conversion"
    / "assets"
    / "mhr2smplx_mapping.npz"
)
# Parent of the ``smplx/`` model folder (smplx.create appends model_type).
# The licensed weights live with BetterHuman, same tree bvr/body.py loads from.
_DEFAULT_SMPLX_MODELS = REPO_ROOT.parent / "BetterHuman" / "models"

# Per-person overlay colours (RGB), matching estimate_body_sam3d so a person
# keeps the same colour across the body and SMPL-X overlays. Cycles if needed.
_MESH_COLORS = (
    (0.65098039, 0.74117647, 0.85882353),  # light blue
    (0.95, 0.60, 0.40),                    # salmon
    (0.60, 0.85, 0.55),                    # light green
    (0.80, 0.65, 0.88),                    # lilac
    (0.95, 0.85, 0.45),                    # gold
)

# Per-frame keys produced by ``MhrToSmplxConverter.convert``.
_FIT_KEYS: tuple[str, ...] = (
    "betas",
    "body_pose",
    "global_orient",
    "transl",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "expression",
    "fit_error",
)


@dataclass
class ConvertResult:
    num_frames: int
    num_people: int
    num_valid: int
    height: int
    width: int
    output_path: Path
    rendered_video: Path | None
    median_fit_error_m: float


class MhrToSmplxScript:
    def __init__(
        self,
        device: str = "cuda",
        smplx_model_folder: str | Path | None = None,
        gender: str = "neutral",
        num_iters: int = 300,
        batch_size: int = 64,
    ) -> None:
        self.device = device
        self.smplx_model_folder = Path(smplx_model_folder or _DEFAULT_SMPLX_MODELS)
        self.gender = gender
        self.num_iters = num_iters
        self.batch_size = batch_size
        self._mhr = None
        self._converter = None

    def _make_factory(self):
        """``factory(batch_size)`` → a fresh pip-``smplx`` SMPL-X model.

        pip ``smplx`` pre-allocates batch-shaped buffers, so the model must be
        rebuilt per batch size; the converter caches by size.
        """
        import smplx as smplx_lib

        folder, gender = str(self.smplx_model_folder), self.gender

        def factory(b: int):
            # Silence numpy 2.4 deprecation when unpickling SMPLX_*.npz
            # (the MPI files carry ``align=0`` in their pickled dtypes).
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=r"dtype\(\): align should be passed",
                )
                return smplx_lib.create(
                    model_path=folder,
                    model_type="smplx",
                    gender=gender,
                    use_pca=False,
                    flat_hand_mean=False,
                    num_betas=10,
                    num_expression_coeffs=10,
                    batch_size=b,
                )

        return factory

    def _build_converter(self):
        from mhr.mhr import MHR

        from bvr.mhr_to_smplx import MhrToSmplxConverter

        if self._mhr is None:
            self._mhr = MHR.from_files(device=torch.device(self.device), lod=1)
        if self._converter is None:
            self._converter = MhrToSmplxConverter(
                mhr_model=self._mhr,
                smplx_model_factory=self._make_factory(),
                mapping_npz=_MAPPING_NPZ,
                device=self.device,
                batch_size=self.batch_size,
            )
        return self._converter

    # -- main entry point ----------------------------------------------------

    def run(
        self,
        source: str | Path | None = None,
        predictions_dir: str | Path | None = None,
        *,
        sam3d_dir: str | Path | None = None,
        render_video: bool = True,
        video_fps: float | None = None,
    ) -> ConvertResult:
        """Convert ``<sam3d_dir>/params.npz`` → ``<sam3d_dir>/smplx_params.npz``.

        Pass ``predictions_dir`` (single-clip; ``sam3d_dir = predictions_dir/sam3d``)
        or an explicit ``sam3d_dir`` (batch / DB layout, where the sam3d feature
        dir holds ``params.npz`` and receives ``smplx_params.npz``). ``source`` is
        only needed for the overlay.
        """
        pred = Path(predictions_dir) if predictions_dir is not None else None
        if sam3d_dir is not None:
            sam3d_dir = Path(sam3d_dir)
        elif pred is not None:
            sam3d_dir = pred / "sam3d"
        else:
            raise ValueError("run() needs predictions_dir or an explicit sam3d_dir")

        body_path = sam3d_dir / "params.npz"
        if not body_path.exists():
            raise FileNotFoundError(body_path)
        bd = np.load(body_path)

        mhr_all = bd["mhr_model_params"]  # (P, N, 204) — legacy single-person: (N, 204)
        shape_all = bd["shape_params"]    # (P, N, 45)
        cam_t_all = bd["pred_cam_t"]      # (P, N, 3)
        focal_all = bd["focal_length"]    # (P, N)
        cam_intrinsics = bd["cam_intrinsics"]  # (3, 3) — shared camera
        valid_all = bd["valid_mask"]      # (P, N) bool
        N = int(bd["num_frames"])
        H = int(bd["video_height"])
        W = int(bd["video_width"])

        # Normalize the legacy un-stacked (single-person) layout to (P=1, N, ...).
        if mhr_all.ndim == 2:
            mhr_all, shape_all, cam_t_all = mhr_all[None], shape_all[None], cam_t_all[None]
            focal_all, valid_all = focal_all[None], valid_all[None]
        valid_all = valid_all.astype(bool)
        P = mhr_all.shape[0]
        object_ids = (
            np.asarray(bd["object_ids"], dtype=np.int32)
            if "object_ids" in bd.files else np.arange(P, dtype=np.int32)
        )

        conv = self._build_converter()

        # Convert each tracked person independently, then stack to (P, N, ...).
        per_person: list[dict[str, np.ndarray]] = []
        for p in range(P):
            vmask = valid_all[p]
            invalid = ~vmask
            mhr = np.where(invalid[:, None], 0.0, np.nan_to_num(mhr_all[p])).astype(np.float32)
            shp = np.where(invalid[:, None], 0.0, np.nan_to_num(shape_all[p])).astype(np.float32)
            ct = np.where(invalid[:, None], 0.0, np.nan_to_num(cam_t_all[p])).astype(np.float32)
            print(f"  person obj_id={int(object_ids[p])}: "
                  f"converting {int(vmask.sum())}/{N} valid frames...")
            per_person.append(
                conv.convert(
                    mhr_model_params=mhr,
                    shape_params=shp,
                    pred_cam_t=ct,
                    valid_mask=vmask,
                    num_iters=self.num_iters,
                )
            )
        stacked = {k: np.stack([r[k] for r in per_person], axis=0) for k in _FIT_KEYS}

        output_path = sam3d_dir / "smplx_params.npz"
        save: dict[str, np.ndarray] = {k: stacked[k] for k in _FIT_KEYS}
        save.update(
            cam_intrinsics=cam_intrinsics.astype(np.float32),
            valid_mask=valid_all,
            object_ids=object_ids,
            num_people=np.int32(P),
            num_frames=np.int32(N),
            video_height=np.int32(H),
            video_width=np.int32(W),
            gender=np.array(self.gender, dtype="U8"),
            model_type=np.array("smplx", dtype="U8"),
            num_betas=np.int32(10),
            num_expression_coeffs=np.int32(10),
        )
        np.savez(output_path, **save)

        err = stacked["fit_error"][valid_all]
        median_err = float(np.median(err)) if err.size else float("nan")

        rendered_path: Path | None = None
        if render_video:
            if source is None:
                raise ValueError("overlay requested but no --source video provided")
            source = Path(source)
            rendered_path = sam3d_dir / "smplx_overlay.mp4"
            self._render_overlay(
                source=source,
                smplx_params=stacked,
                valid_mask=valid_all,
                focal_lengths=focal_all,
                cam_intrinsics=cam_intrinsics,
                W=W,
                H=H,
                N=N,
                P=P,
                out_path=rendered_path,
                fps=video_fps,
            )

        return ConvertResult(
            num_frames=N,
            num_people=P,
            num_valid=int(valid_all.sum()),
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
        smplx_params: dict[str, np.ndarray],
        valid_mask: np.ndarray,
        focal_lengths: np.ndarray,
        cam_intrinsics: np.ndarray,
        W: int,
        H: int,
        N: int,
        P: int,
        out_path: Path,
        fps: float | None,
    ) -> None:
        """Combined overlay: every person composited onto each frame, one colour each.

        ``smplx_params`` holds stacked ``(P, N, ...)`` arrays; ``valid_mask`` and
        ``focal_lengths`` are ``(P, N)``. Within a frame, people are chain-rendered
        (each composited on top of the previous), so meshes share the camera but
        don't cross-occlude — fine for spatially separated subjects.
        """
        from sam_3d_body.visualization.renderer import Renderer
        from bvr.source import resolve_source

        conv = self._converter
        faces = np.asarray(conv.faces, dtype=np.int32)  # SMPL-X faces (shared)
        V = int(faces.max()) + 1

        # Pre-compute SMPL-X vertices in camera frame for every valid (person, frame).
        # transl was fitted in the camera frame, so verts come out camera-framed.
        verts_cam = np.full((P, N, V, 3), np.nan, dtype=np.float32)
        with torch.no_grad():
            dev = torch.device(self.device)
            bs = self.batch_size
            for p in range(P):
                valid_idx = np.flatnonzero(valid_mask[p])
                if valid_idx.size == 0:
                    continue
                betas = torch.from_numpy(smplx_params["betas"][p][valid_idx]).to(dev)
                body_pose = torch.from_numpy(smplx_params["body_pose"][p][valid_idx]).to(dev)
                global_orient = torch.from_numpy(smplx_params["global_orient"][p][valid_idx]).to(dev)
                transl = torch.from_numpy(smplx_params["transl"][p][valid_idx]).to(dev)
                lh = torch.from_numpy(smplx_params["left_hand_pose"][p][valid_idx]).to(dev)
                rh = torch.from_numpy(smplx_params["right_hand_pose"][p][valid_idx]).to(dev)
                for s in range(0, valid_idx.size, bs):
                    e = min(s + bs, valid_idx.size)
                    # Reuse the converter's per-size model cache + forward (zeros the
                    # jaw/eye/expression channels, exactly as the fit did).
                    model = conv._get_smplx(e - s)
                    v = conv._smplx_forward(
                        model, betas[s:e], body_pose[s:e], global_orient[s:e],
                        transl[s:e], lh[s:e], rh[s:e],
                    )
                    verts_cam[p, valid_idx[s:e]] = v.cpu().numpy()

        fx = float(cam_intrinsics[0, 0])

        def _render_person(p: int, i: int, bgr: np.ndarray) -> np.ndarray:
            f = float(focal_lengths[p, i]) if not np.isnan(focal_lengths[p, i]) else fx
            renderer = Renderer(focal_length=f, faces=faces)
            return (
                renderer(
                    verts_cam[p, i].astype(np.float32),
                    np.zeros(3, dtype=np.float32),
                    bgr,
                    mesh_base_color=_MESH_COLORS[p % len(_MESH_COLORS)],
                    scene_bg_color=(1, 1, 1),
                )
                * 255
            ).astype(np.uint8)

        def _render_all(i: int, bgr: np.ndarray) -> np.ndarray:
            for p in range(P):
                if valid_mask[p, i] and not np.isnan(verts_cam[p, i]).any():
                    bgr = _render_person(p, i, bgr)
            return bgr

        with tempfile.TemporaryDirectory(prefix="convert_smplx_render_") as tmp_root:
            tmp_root = Path(tmp_root)
            frame_paths, src_fps = resolve_source(source, tmp_root)

            # Single-image input → write a PNG, not a video.
            if len(frame_paths) == 1 and N == 1:
                bgr = cv2.imread(str(frame_paths[0]))
                if bgr is None:
                    raise RuntimeError(f"cannot read {frame_paths[0]}")
                bgr = _render_all(0, bgr)
                cv2.imwrite(str(out_path.with_suffix(".png")), bgr)
                return

            fps = float(fps or src_fps or 30.0)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
            if not writer.isOpened():
                raise RuntimeError(f"cannot open video writer at {out_path}")
            try:
                for i in tqdm(range(N), desc="Rendering SMPL-X overlay"):
                    bgr = cv2.imread(str(frame_paths[i]))
                    if bgr is None:
                        continue
                    writer.write(_render_all(i, bgr))
            finally:
                writer.release()


class MhrToSmplxStep:
    """precompute_runner Step: build the converter once, sweep each manifest scene.

    Batch / manifest mode for the ClimbingVideos scene DB. Each :class:`InputItem`
    points ``out_dir`` at the scene's **sam3d feature dir** (which holds
    ``params.npz`` and receives ``smplx_params.npz``). The conversion reads only
    ``params.npz``; ``source`` is carried through for the overlay but unused when
    ``--no-overlay``. ``--subdir`` picks the subfolder of ``out_dir`` (``.`` → it
    directly, as ``corpus.py`` does for the stage-first tree).
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.script = MhrToSmplxScript(
            device=args.device,
            smplx_model_folder=args.smplx_model_folder,
            gender=args.gender,
            num_iters=args.iters,
            batch_size=args.batch_size,
        )

    def load(self) -> None:
        self.script._build_converter()

    def release(self) -> None:
        self.script._mhr = None
        self.script._converter = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process(self, item: InputItem) -> None:
        subdir = getattr(self.args, "subdir", "sam3d") or "sam3d"
        sam3d_dir = item.out_dir if subdir in (".", "") else item.out_dir / subdir
        if self.args.skip_existing and (sam3d_dir / "smplx_params.npz").exists():
            raise SkipItem(f"smplx already done: {sam3d_dir / 'smplx_params.npz'}")
        if not (sam3d_dir / "params.npz").exists():
            raise SkipItem(f"no sam3d params for {item.stem}: {sam3d_dir / 'params.npz'}")
        self.script.run(
            source=item.source_path,
            sam3d_dir=sam3d_dir,
            render_video=self.args.overlay,
            video_fps=self.args.fps,
        )


# ------------------------------ CLI -------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert MHR predictions (sam3d/params.npz) to SMPL-X parameters (pip smplx).",
    )
    p.add_argument(
        "--source", required=True,
        help="single clip (video / folder of frames) for --predictions mode, or a "
             ".json manifest for configured batch mode. Only used for the overlay.",
    )
    p.add_argument(
        "--predictions", default=None,
        help="single-clip mode: folder containing sam3d/params.npz; writes "
             "sam3d/smplx_params.npz",
    )
    p.add_argument(
        "--save-dir", default=None, type=Path,
        help="manifest batch mode: output root (per-item outputs under <save-dir>/<stem>/)",
    )
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.toml",
                   help="pipeline TOML (default: configs/pipeline.toml)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip items whose outputs already exist (overrides the config key)")
    cli = p.parse_args(argv)
    cfg = load_config(cli.config)
    args = section(cfg, "smplx", {
        "subdir": "sam3d",
        "skip_existing": False,
        "smplx_model_folder": "",
        "gender": "neutral",
        "iters": 300,
        "batch_size": 64,
        "overlay": True,
        "fps": "",
    })
    args.device = section(cfg, "runtime", {"device": "cuda"}).device
    args.source, args.predictions, args.config = cli.source, cli.predictions, cli.config
    args.save_dir = cli.save_dir
    args.skip_existing = args.skip_existing or cli.skip_existing
    args.smplx_model_folder = args.smplx_model_folder or None
    args.fps = float(args.fps) if args.fps != "" else None
    return args


def main(argv: list[str] | None = None) -> int:
    determinism.apply()
    args = _parse_args(argv)

    # Batch / manifest mode: precompute_runner Step (build converter once, sweep).
    if args.save_dir is not None:
        return run_batch(MhrToSmplxStep(args), args)

    # Single-clip mode.
    if args.predictions is None:
        raise SystemExit(
            "convert_mhr_to_smplx: pass --predictions <clip-dir> (single clip) "
            "or set smplx.save_dir in the config for a .json --source manifest (batch)."
        )
    script = MhrToSmplxScript(
        device=args.device,
        smplx_model_folder=args.smplx_model_folder,
        gender=args.gender,
        num_iters=args.iters,
        batch_size=args.batch_size,
    )
    result = script.run(
        source=args.source,
        predictions_dir=args.predictions,
        render_video=args.overlay,
        video_fps=args.fps,
    )
    msg = (
        f"converted {result.num_people} person(s) × {result.num_frames} frames "
        f"({result.num_valid} valid people-frames, {result.width}x{result.height}), "
        f"median fit error {result.median_fit_error_m * 1000:.1f} mm -> {result.output_path}"
    )
    if result.rendered_video is not None:
        msg += f"\nrendered SMPL-X overlay -> {result.rendered_video}"
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
