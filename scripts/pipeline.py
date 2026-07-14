"""End-to-end video reconstruction orchestrator (file-based, small-batch).

Runs the active stages in order for a video (or a folder of videos):

    sam3 → vggt → sapiens → sam3d → smplx → scale → fuse → ™

Each stage is invoked as a subprocess of the current interpreter — this
isolates per-stage model memory (SAM 3, VGGT-Omega, Sapiens2 and SAM-3D-Body
never co-reside) and lets the pymomentum ``_pym_preload`` re-exec happen cleanly
in the stages that need it.

Stages run in the **listed order**. The model **precompute** stages (``sam3``,
``vggt``, ``sapiens``) run once over ``--source`` — given a *folder* of videos
they load each model once and sweep the whole batch
(``tools/precompute_runner`` semantics). The **per-clip** stages (``sam3d``,
``smplx``, ``scale``, ``fuse``, ``human_optimization``) loop over each clip's
``<save-dir>/<stem>/``. ``scale`` (``estimate_scale``) anchors VGGT's metric
scale to the SAM-3D-Body human ruler, so it runs **after** the body / SMPL-X
stages and **before** ``fuse`` (which applies that scale); ``sam3d`` reads the
VGGT camera from ``vggt/camera.npz`` directly.

The run ends at ``human_optimization`` — the four-stage SMPL-X refinement (default
chain: root, kinematics_1 → ``human_optim/kinematic_1.npz``). ``--steps`` runs any
subset of the chain.

Per-stage knobs and the step selection live in ``configs/pipeline.toml``; the
CLI only takes ``--source``, ``--save-dir``, ``--config``, ``--steps``,
and ``--skip-existing``. Each stage is also runnable standalone
(see ``scripts/stages/``).

Usage::

    uv run python scripts/pipeline.py --source video.mp4 --save-dir out/
    uv run python scripts/pipeline.py --source clips/ --save-dir out/ --skip-existing
    uv run python scripts/pipeline.py --source video.mp4 --save-dir out/ --steps sam3,vggt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from tools.config import load_toml  # noqa: E402
from tools.source import IMAGE_EXTS, VIDEO_EXTS  # noqa: E402

_STAGES_DIR = _HERE.parent / "stages"
_MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS

# Canonical order. "precompute" stages operate on --source / --save-dir; "perclip"
# stages operate on a single clip's --predictions dir. ``--steps`` runs any subset;
# the chain ends at human_optimization (visually-guided SMPL-X kinematics).
DEFAULT_RUN = ["sam3", "vggt", "sapiens", "sam3d", "smplx", "scale", "fuse", "human_optimization"]
_PRECOMPUTE = {"sam3", "vggt", "sapiens"}
_KNOWN = set(DEFAULT_RUN)


def _default_config() -> dict:
    return {
        "steps": {"run": DEFAULT_RUN},
        "runtime": {"device": "cuda"},
        "sam3": {"text": "person"},
        "vggt": {"max_frames": 500, "image_resolution": 512, "mode": "balanced",
                 "save_depth": False, "checkpoint": ""},
        "sapiens": {"tasks": "pose", "overlay": True},
        "scale": {"conf_drop_q": 0.3, "min_inlier_verts": 50, "mad_k": 3.0},
        "fuse": {"voxel_scene": 0.05, "conf_drop_q": 0.3, "edge_rtol": 0.04,
                 "human_dilate": 8, "min_scene_voxels": 25, "person_obj_ids": ""},
        "sam3d": {"obj_ids": "", "inference_type": "full", "overlay": True},
        "smplx": {"iters": 300, "gender": "neutral", "overlay": True},
        "human_optimization": {"config": "configs/human_optimization.toml"},
    }


def _load_config(path: Path | None) -> dict:
    cfg = _default_config()
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        user = load_toml(path)
        for section, vals in user.items():
            cfg.setdefault(section, {}).update(vals)
    return cfg


def _resolve_clips(source: Path) -> list[Path]:
    if source.is_dir():
        clips = sorted(p for p in source.iterdir() if p.suffix.lower() in _MEDIA_EXTS)
        if not clips:
            raise FileNotFoundError(f"no media files in {source}")
        return clips
    if source.is_file():
        return [source]
    raise FileNotFoundError(f"--source not found: {source}")


def _build_precompute_args(source: Path, save_dir: Path, config: Path,
                           skip_existing: bool) -> list[str]:
    args = ["--source", str(source), "--save-dir", str(save_dir),
            "--config", str(config)]
    if skip_existing:
        args.append("--skip-existing")   # per-item resume inside the load-once sweep
    return args


def _perclip_output(step: str, clip_dir: Path, cfg: dict) -> Path:
    if step == "human_optimization":  # skip-check the LAST configured sub-stage's body
        ho_path = Path(cfg["human_optimization"]["config"])
        if not ho_path.is_absolute():
            ho_path = _REPO_ROOT / ho_path
        st = list(load_toml(ho_path).get("stages", {}).get("run", ["root", "kinematics_1"]))
        name = ("dynamics" if "dynamics" in st
                else "kinematic_2" if "kinematics_2" in st
                else "kinematic_1" if "kinematics_1" in st
                else "root")
        return clip_dir / "human_optim" / f"{name}.npz"
    return {
        "scale": clip_dir / "geometry" / "scale.npz",
        "fuse": clip_dir / "geometry" / "transform.npz",
        "sam3d": clip_dir / "sam3d" / "params.npz",
        "smplx": clip_dir / "sam3d" / "smplx_params.npz",
    }[step]


def _build_perclip_args(step: str, cfg: dict, clip: Path, clip_dir: Path,
                        config: Path) -> list[str]:
    if step == "scale":
        return ["--predictions", str(clip_dir), "--config", str(config)]
    if step in {"fuse", "sam3d", "smplx"}:
        return ["--source", str(clip), "--predictions", str(clip_dir),
                "--config", str(config)]
    if step == "human_optimization":
        s = cfg["human_optimization"]
        return ["--source", str(clip), "--predictions", str(clip_dir),
                "--config", str(s["config"])]
    raise ValueError(step)


_SCRIPTS = {
    "sam3": "track_sam3.py",
    "sapiens": "estimate_sapiens.py",
    "vggt": "estimate_camera_vggt.py",
    "scale": "estimate_scale.py",
    "fuse": "fuse_geometry.py",
    "sam3d": "estimate_body_sam3d.py",
    "smplx": "convert_mhr_to_smplx.py",
    "human_optimization": "human_optimization.py",
}


def _run(step: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(_STAGES_DIR / _SCRIPTS[step]), *extra_args]
    print(f"\n=== [{step}] {' '.join(cmd[1:])} ===", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"[pipeline] stage '{step}' failed (exit {rc})")


def run_pipeline(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    source = Path(args.source)
    save_dir = Path(args.save_dir)

    requested = (
        [s.strip() for s in args.steps.split(",") if s.strip()]
        if args.steps else list(cfg["steps"]["run"])
    )
    bad = [s for s in requested if s not in _KNOWN]
    if bad:
        raise SystemExit(f"[pipeline] unknown step(s) {bad}; known: {sorted(_KNOWN)}")

    clips = _resolve_clips(source)
    print(f"[pipeline] {len(clips)} clip(s); steps: {requested}")

    # Execute in list order. Precompute stages run once over --source (load-once
    # sweep across all clips); per-clip stages loop the clips. Honouring the list
    # order lets a per-clip stage (fuse) run before a precompute one (sapiens).
    for step in requested:
        if step in _PRECOMPUTE:
            _run(step, _build_precompute_args(source, save_dir, args.config, args.skip_existing))
            continue
        for clip in clips:
            clip_dir = save_dir / clip.stem
            if args.skip_existing and _perclip_output(step, clip_dir, cfg).exists():
                print(f"[pipeline] skip {step} for {clip.stem} (output exists)")
                continue
            _run(step, _build_perclip_args(step, cfg, clip, clip_dir, args.config))

    print("\n[pipeline] done.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True,
                   help="video file or folder of video files (each a clip)")
    p.add_argument("--save-dir", required=True, type=Path,
                   help="per-clip outputs land in <save-dir>/<stem>/")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "pipeline.toml",
                   help="orchestrator TOML (default: configs/pipeline.toml)")
    p.add_argument("--steps", default=None,
                   help="comma-separated subset of steps to run (default: config's [steps].run)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip stages whose outputs already exist")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
