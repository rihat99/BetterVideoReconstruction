"""End-to-end video reconstruction orchestrator (file-based, small-batch).

Runs the active stages in order for a video (or a folder of videos):

    sam3 → vggt → moge2 → fuse → sapiens → sam3d → smplx

Each stage is invoked as a subprocess of the current interpreter — this
isolates per-stage model memory (SAM 3, VGGT-Omega, MoGe2, Sapiens2 and
SAM-3D-Body never co-reside) and lets the pymomentum ``_pym_preload`` re-exec
happen cleanly in the stages that need it.

Stages run in the **listed order**. The model **precompute** stages (``sam3``,
``vggt``, ``moge2``, ``sapiens``) run once over ``--source`` — given a *folder*
of videos they load each model once and sweep the whole batch
(``tools/precompute_runner`` semantics). The **per-clip** stages (``fuse``,
``sam3d``, ``smplx``) loop over each clip's ``<save-dir>/<stem>/``. Because list
order is honoured, a per-clip stage can precede a precompute one — ``fuse`` runs
before ``sapiens`` (geometry before human keypoints), and ``sam3d`` reads the
VGGT camera that ``fuse`` writes.

The default run ends at ``smplx``. The SMPL → motion/dynamics path
(``smpl`` → ``motion`` → ``optimize_motion`` → ``optimize_dynamics``) is
SMPL-only (it consumes ``smpl_params``, not SMPL-X) and is run standalone or via
``--steps``.

Per-stage knobs and the step selection live in ``configs/pipeline.toml``; the
CLI only takes ``--source``, ``--save-dir``, ``--config``, ``--steps``,
``--skip-existing``, and ``--device``. Each stage is also runnable standalone
(see ``scripts/stages/``).

Usage::

    uv run python scripts/pipeline.py --source video.mp4 --save-dir out/
    uv run python scripts/pipeline.py --source clips/ --save-dir out/ --skip-existing
    uv run python scripts/pipeline.py --source video.mp4 --save-dir out/ --steps sam3,moge2

``gemma`` (Gemma 4 triage) is registered but a **placeholder** — not in the
default run list; selecting it raises a clear NotImplementedError.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from tools.source import IMAGE_EXTS, VIDEO_EXTS  # noqa: E402

_STAGES_DIR = _HERE.parent / "stages"
_MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS

# Canonical order. "precompute" stages operate on --source / --save-dir;
# "perclip" stages operate on a single clip's --predictions dir. The default
# chain stops at smplx (the SMPL-X body); see _STANDALONE for the rest.
DEFAULT_RUN = ["sam3", "vggt", "moge2", "fuse", "sapiens", "sam3d", "smplx"]
_PRECOMPUTE = {"sam3", "vggt", "moge2", "sapiens", "gemma"}
# Registered but not in the default run — selectable via --steps. The
# SMPL → motion/dynamics path is SMPL-only (consumes smpl_params, not SMPL-X);
# gemma is a placeholder.
_STANDALONE = {"smpl", "motion", "optimize_motion", "optimize_dynamics", "gemma"}
_KNOWN = set(DEFAULT_RUN) | _STANDALONE


def _default_config() -> dict:
    return {
        "steps": {"run": DEFAULT_RUN},
        "runtime": {"device": "cuda"},
        "sam3": {"text": "person"},
        "vggt": {"max_frames": 500, "image_resolution": 512, "mode": "balanced",
                 "save_depth": False, "checkpoint": ""},
        "moge2": {"save_depth": False},
        "sapiens": {"tasks": "pose", "overlay": True},
        "fuse": {"voxel_scene": 0.05, "conf_drop_q": 0.3, "edge_rtol": 0.04,
                 "human_dilate": 8, "person_obj_ids": ""},
        "sam3d": {"obj_ids": "", "inference_type": "full", "overlay": True},
        "smplx": {"iters": 300, "gender": "neutral", "overlay": True},
        "smpl": {"iters": 300},
        "motion": {"fps": 30.0, "smooth": "gaussian"},
        "optimize_motion": {"config": "configs/kinematic.toml"},
        "optimize_dynamics": {"config": "configs/dynamics.toml"},
    }


def _load_config(path: Path | None) -> dict:
    cfg = _default_config()
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        with path.open("rb") as f:
            user = tomllib.load(f)
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


def _flag(args: list[str], cond: bool, *flag: str) -> None:
    if cond:
        args.extend(flag)


def _build_precompute_args(step: str, cfg: dict, source: Path, save_dir: Path,
                           device: str, skip_existing: bool) -> list[str]:
    base = ["--source", str(source), "--save-dir", str(save_dir), "--device", device]
    _flag(base, skip_existing, "--skip-existing")
    if step == "sam3":
        return base + ["--text", str(cfg["sam3"]["text"])]
    if step == "vggt":
        v = cfg["vggt"]
        extra = ["--max-frames", str(v.get("max_frames", 500)),
                 "--image-resolution", str(v.get("image_resolution", 512)),
                 "--mode", str(v.get("mode", "balanced"))]
        if v.get("checkpoint"):
            extra += ["--checkpoint", str(v["checkpoint"])]
        _flag(extra, v.get("save_depth", False), "--save-depth")
        return base + extra
    if step == "moge2":
        extra = ["--save-depth"] if cfg["moge2"].get("save_depth") else []
        return base + extra
    if step == "sapiens":
        extra = ["--tasks", str(cfg["sapiens"]["tasks"])]
        _flag(extra, not cfg["sapiens"].get("overlay", True), "--no-overlay")
        return base + extra
    if step == "gemma":
        return base  # placeholder — standard args only
    raise ValueError(step)


def _perclip_output(step: str, clip_dir: Path) -> Path:
    return {
        "fuse": clip_dir / "geometry" / "transform.npz",
        "sam3d": clip_dir / "sam3d" / "params.npz",
        "smplx": clip_dir / "sam3d" / "smplx_params.npz",
        "smpl": clip_dir / "smpl_params.npz",
        "motion": clip_dir / "results" / "0.npz",
        "optimize_motion": clip_dir / "results" / "1.npz",
        "optimize_dynamics": clip_dir / "results" / "2.npz",
    }[step]


def _build_perclip_args(step: str, cfg: dict, clip: Path, clip_dir: Path,
                        device: str) -> list[str]:
    if step == "fuse":
        f = cfg["fuse"]
        extra = ["--source", str(clip), "--predictions", str(clip_dir), "--device", device,
                 "--voxel-scene", str(f["voxel_scene"]),
                 "--conf-drop-q", str(f["conf_drop_q"]),
                 "--edge-rtol", str(f["edge_rtol"]),
                 "--human-dilate", str(f["human_dilate"])]
        if f.get("person_obj_ids"):
            extra += ["--person-obj-ids", str(f["person_obj_ids"])]
        return extra
    if step == "sam3d":
        s = cfg["sam3d"]
        extra = ["--source", str(clip), "--predictions", str(clip_dir),
                 "--inference-type", str(s["inference_type"]),
                 "--device", device]
        if s.get("obj_ids"):  # empty → fit all tracked people
            extra += ["--obj-ids", str(s["obj_ids"])]
        _flag(extra, not s.get("overlay", True), "--no-overlay")
        return extra
    if step == "smplx":
        s = cfg["smplx"]
        extra = ["--source", str(clip), "--predictions", str(clip_dir),
                 "--iters", str(s["iters"]),
                 "--gender", str(s.get("gender", "neutral")),
                 "--device", device]
        _flag(extra, not s.get("overlay", True), "--no-overlay")
        return extra
    if step == "smpl":
        return ["--source", str(clip), "--predictions", str(clip_dir),
                "--iters", str(cfg["smpl"]["iters"]), "--device", device]
    if step == "motion":
        return ["--predictions", str(clip_dir),
                "--fps", str(cfg["motion"]["fps"]),
                "--smooth", str(cfg["motion"]["smooth"])]
    if step == "optimize_motion":
        return [str(clip_dir), "--config", str(cfg["optimize_motion"]["config"])]
    if step == "optimize_dynamics":
        return [str(clip_dir), "--config", str(cfg["optimize_dynamics"]["config"])]
    raise ValueError(step)


_SCRIPTS = {
    "sam3": "track_sam3.py",
    "moge2": "estimate_camera_moge2.py",
    "sapiens": "estimate_sapiens.py",
    "vggt": "estimate_camera_vggt.py",
    "gemma": "classify_gemma.py",
    "fuse": "fuse_geometry.py",
    "sam3d": "estimate_body_sam3d.py",
    "smplx": "convert_mhr_to_smplx.py",
    "smpl": "convert_mhr_to_smpl.py",
    "motion": "build_robot_motion.py",
    "optimize_motion": "optimize_motion.py",
    "optimize_dynamics": "optimize_dynamics.py",
}


def _run(step: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(_STAGES_DIR / _SCRIPTS[step]), *extra_args]
    print(f"\n=== [{step}] {' '.join(cmd[1:])} ===", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"[pipeline] stage '{step}' failed (exit {rc})")


def run_pipeline(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    device = args.device or cfg["runtime"]["device"]
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
            _run(step, _build_precompute_args(step, cfg, source, save_dir, device, args.skip_existing))
            continue
        for clip in clips:
            clip_dir = save_dir / clip.stem
            if args.skip_existing and _perclip_output(step, clip_dir).exists():
                print(f"[pipeline] skip {step} for {clip.stem} (output exists)")
                continue
            _run(step, _build_perclip_args(step, cfg, clip, clip_dir, device))

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
    p.add_argument("--device", default=None, help="override runtime.device from the config")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if not args.config.exists():
        args.config = None  # fall back to built-in defaults
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
