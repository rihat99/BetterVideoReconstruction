"""Run the full reconstruction chain over the imported LAAS Parkour clips.

The clips live under ``data/Parkour-dataset/processed/`` (next to the raw
dataset, outside the repo): ``processed/<clip>/video.mp4`` →
``processed/out/<clip>/``. Single view, one subject per clip (see
``scripts/import_parkour.py`` for the import).

The chain, in order::

    sam3 → vggt → sapiens → sam3d → smplx → fuse → scale → gravity
         → kinematics_1 → contacts_1 → kinematics_2 → contacts_2 → kindyn_1

The five model stages run as ONE manifest sweep each (the model loads once and
sweeps every clip); the rest run one subprocess per clip, clip by clip. Every
step is resumable: a step whose output already exists is skipped. A clip that
fails is reported and skipped — one bad clip never kills the batch.

``gravity`` (GeoCalib) is in the chain because ``kindyn_1`` solves against
world gravity and the lab camera's tilt is unknown — the world-down fallback
would bake that tilt into every force direction.

Knobs live in the TOMLs (``configs/pipeline_peter.toml`` for the model +
geometry stages, ``configs/human_optimization_peter.toml`` for the human stages —
the static-camera recipe: kinematic chamfer off, hand/foot contact frames only);
this CLI only takes targets.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_parkour.py
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_parkour.py \
        --clips kv01_PKFC --steps fuse,scale,kinematics_1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGES_DIR = _REPO_ROOT / "scripts" / "stages"
_PARKOUR = Path("/home/rikhat.akizhanov/better/data/Parkour-dataset/processed")

# step → (stage script, output path relative to the clip's output tree).
# The first five sweep every clip in one process; the rest run per clip.
MANIFEST_STEPS: dict[str, tuple[str, str]] = {
    "sam3":    ("track_sam3.py",           "sam3/bboxes.npz"),
    "vggt":    ("estimate_camera_vggt.py", "vggt/camera.npz"),
    "sapiens": ("estimate_sapiens.py",     "sapiens/pose.npz"),
    "sam3d":   ("estimate_body_sam3d.py",  "sam3d/params.npz"),
    "smplx":   ("convert_mhr_to_smplx.py", "sam3d/smplx_params.npz"),
}
CLIP_OUTPUT: dict[str, str] = {
    "fuse":         "geometry/transform.npz",
    "scale":        "geometry/scale.npz",
    "gravity":      "geocalib/gravity.npz",
    "kinematics_1": "human_optim/kinematic_1.npz",
    "contacts_1":   "human_optim/contacts_1.npz",
    "kinematics_2": "human_optim/kinematic_2.npz",
    "contacts_2":   "human_optim/contacts_2.npz",
    "kindyn_1":     "human_optim/kindyn_1.npz",
}
ALL_STEPS = [*MANIFEST_STEPS, *CLIP_OUTPUT]


def _clip_command(step: str, video: Path, clip_dir: Path,
                  pipeline_config: Path, human_config: Path) -> list[str]:
    """Argv for one per-clip stage (all paths absolute)."""
    stage = lambda name: str(_STAGES_DIR / name)  # noqa: E731
    common = ["--source", str(video), "--predictions", str(clip_dir)]
    if step == "fuse":
        return [sys.executable, stage("fuse_geometry.py"), *common,
                "--config", str(pipeline_config)]
    if step == "scale":
        return [sys.executable, stage("estimate_scale.py"),
                "--predictions", str(clip_dir), "--config", str(pipeline_config)]
    if step == "gravity":
        return [sys.executable, stage("estimate_gravity_geocalib.py"), *common,
                "--config", str(pipeline_config)]
    if step in {"contacts_1", "contacts_2"}:
        bodies = "kinematic_1" if step == "contacts_1" else "kinematic_2"
        return [sys.executable, stage("estimate_contacts.py"), *common,
                "--bodies", bodies, "--config", str(human_config)]
    stages_arg = {"kinematics_1": "root,kinematics_1",
                  "kinematics_2": "kinematics_2",
                  "kindyn_1": "kindyn_1"}[step]
    return [sys.executable, stage("human_optimization.py"), *common,
            "--stages", stages_arg, "--config", str(human_config)]


def _run(command: list[str]) -> int:
    print(f"\n=== {' '.join(command[1:])} ===", flush=True)
    return subprocess.run(command).returncode


def _manifest_name(clips: list[Path]) -> str:
    """Manifest filename for this exact clip selection (see run_peter.py)."""
    tag = hashlib.sha1(",".join(c.name for c in clips).encode()).hexdigest()[:8]
    return f"_clips_{tag}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", default=None,
                        help="comma-separated clip names (default: all)")
    parser.add_argument("--steps", default=None,
                        help=f"comma-separated subset of {','.join(ALL_STEPS)} (default: all)")
    parser.add_argument("--config", type=Path,
                        default=_REPO_ROOT / "configs" / "pipeline_peter.toml",
                        help="model + geometry stage TOML")
    parser.add_argument("--human-config", type=Path,
                        default=_REPO_ROOT / "configs" / "human_optimization_peter.toml",
                        help="human-optimization + contacts TOML")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    clips = sorted(d for d in _PARKOUR.iterdir()
                   if d.is_dir() and (d / "video.mp4").is_file())
    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = [c for c in clips if c.name in wanted]
        missing = wanted - {c.name for c in clips}
        if missing:
            raise SystemExit(f"[run_parkour] unknown clip(s): {sorted(missing)}")
    if not clips:
        raise SystemExit(f"[run_parkour] no clips under {_PARKOUR}")

    steps = [s.strip() for s in args.steps.split(",")] if args.steps else list(ALL_STEPS)
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise SystemExit(f"[run_parkour] unknown step(s) {unknown}; available: {ALL_STEPS}")

    out_root = _PARKOUR / "out"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / _manifest_name(clips)
    manifest.write_text(json.dumps([{
        "source": str(clip / "video.mp4"),
        "out_dir": str(out_root / clip.name),
        "bboxes": str(out_root / clip.name / "sam3" / "bboxes.npz"),
        "camera": str(out_root / clip.name / "vggt" / "camera.npz"),
    } for clip in clips], indent=2))
    print(f"[run_parkour] {len(clips)} clip(s) → {out_root}\n[run_parkour] steps: {steps}")

    for step in [s for s in steps if s in MANIFEST_STEPS]:
        script, _ = MANIFEST_STEPS[step]
        started = time.monotonic()
        rc = _run([sys.executable, str(_STAGES_DIR / script),
                   "--source", str(manifest), "--save-dir", str(out_root),
                   "--config", str(args.config)])
        print(f"[run_parkour] {step}: rc={rc} in {(time.monotonic() - started) / 60:.1f} min")

    clip_steps = [s for s in steps if s in CLIP_OUTPUT]
    failures: list[tuple[str, str]] = []
    for index, clip in enumerate(clips, 1):
        if not clip_steps:
            break
        clip_dir = out_root / clip.name
        print(f"\n########## [{index}/{len(clips)}] {clip.name} ##########", flush=True)
        for step in clip_steps:
            if (clip_dir / CLIP_OUTPUT[step]).exists():
                print(f"[run_parkour] {clip.name}/{step}: exists, skipping")
                continue
            command = _clip_command(step, clip / "video.mp4", clip_dir,
                                    args.config, args.human_config)
            started = time.monotonic()
            rc = _run(command)
            print(f"[run_parkour] {clip.name}/{step}: rc={rc} "
                  f"in {(time.monotonic() - started) / 60:.1f} min")
            if rc != 0:
                failures.append((clip.name, step))
                break   # later steps of this clip depend on this one

    if failures:
        print(f"\n[run_parkour] {len(failures)} clip(s) failed:")
        for name, step in failures:
            print(f"  {name}: failed at {step}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
