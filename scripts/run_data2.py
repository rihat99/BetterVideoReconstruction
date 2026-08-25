"""Run the full reconstruction chain over the data_2/ sports clips.

``data_2/<stem>.mp4`` → ``out_2/<stem>/``. Single athlete per clip (the
pipeline_data2 config keeps only the largest frame-0 person mask).

The chain, in order::

    sam3 → vggt → sapiens → sam3d → smplx → fuse → scale → gravity
         → kinematics_1 → contacts_1 → kinematics_2 → contacts_2 → kindyn_1

The five model stages run as ONE manifest sweep each (the model loads once and
sweeps every clip; per-item resume via the config's ``skip_existing``); the
rest run one subprocess per clip. A per-clip step whose output already exists
is skipped, so re-running the human stages on a clip means deleting (or moving
aside) its ``human_optim/`` first. A clip that fails is reported and skipped —
one bad clip never kills the batch.

``gravity`` (GeoCalib) is in the chain because ``kindyn_1`` solves against
world gravity and these are in-the-wild cameras with unknown tilt.

Knobs live in the TOMLs (``configs/pipeline_data2.toml`` for the model +
geometry stages, ``configs/human_optimization.toml`` for the human
stages); this CLI only takes targets.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_data2.py
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_data2.py \
        --clips high_jump --steps kinematics_1,contacts_1
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
_DATA = _REPO_ROOT / "data_2"
_OUT = _REPO_ROOT / "out_2"

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
    tag = hashlib.sha1(",".join(c.stem for c in clips).encode()).hexdigest()[:8]
    return f"_clips_{tag}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", default=None,
                        help="comma-separated clip stems (default: all)")
    parser.add_argument("--steps", default=None,
                        help=f"comma-separated subset of {','.join(ALL_STEPS)} (default: all)")
    parser.add_argument("--config", type=Path,
                        default=_REPO_ROOT / "configs" / "pipeline_data2.toml",
                        help="model + geometry stage TOML")
    parser.add_argument("--human-config", type=Path,
                        default=_REPO_ROOT / "configs" / "human_optimization.toml",
                        help="human-optimization + contacts TOML")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    clips = sorted(_DATA.glob("*.mp4"))
    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = [c for c in clips if c.stem in wanted]
        missing = wanted - {c.stem for c in clips}
        if missing:
            raise SystemExit(f"[run_data2] unknown clip(s): {sorted(missing)}")
    if not clips:
        raise SystemExit(f"[run_data2] no clips under {_DATA}")

    steps = [s.strip() for s in args.steps.split(",")] if args.steps else list(ALL_STEPS)
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise SystemExit(f"[run_data2] unknown step(s) {unknown}; available: {ALL_STEPS}")

    _OUT.mkdir(parents=True, exist_ok=True)
    manifest = _OUT / _manifest_name(clips)
    manifest.write_text(json.dumps([{
        "source": str(clip),
        "out_dir": str(_OUT / clip.stem),
        "bboxes": str(_OUT / clip.stem / "sam3" / "bboxes.npz"),
        "camera": str(_OUT / clip.stem / "vggt" / "camera.npz"),
    } for clip in clips], indent=2))
    print(f"[run_data2] {len(clips)} clip(s) → {_OUT}\n[run_data2] steps: {steps}")

    for step in [s for s in steps if s in MANIFEST_STEPS]:
        script, _ = MANIFEST_STEPS[step]
        started = time.monotonic()
        rc = _run([sys.executable, str(_STAGES_DIR / script),
                   "--source", str(manifest), "--save-dir", str(_OUT),
                   "--config", str(args.config)])
        print(f"[run_data2] {step}: rc={rc} in {(time.monotonic() - started) / 60:.1f} min")

    clip_steps = [s for s in steps if s in CLIP_OUTPUT]
    failures: list[tuple[str, str]] = []
    for index, clip in enumerate(clips, 1):
        if not clip_steps:
            break
        clip_dir = _OUT / clip.stem
        print(f"\n########## [{index}/{len(clips)}] {clip.stem} ##########", flush=True)
        for step in clip_steps:
            if (clip_dir / CLIP_OUTPUT[step]).exists():
                print(f"[run_data2] {clip.stem}/{step}: exists, skipping")
                continue
            command = _clip_command(step, clip, clip_dir, args.config, args.human_config)
            started = time.monotonic()
            rc = _run(command)
            print(f"[run_data2] {clip.stem}/{step}: rc={rc} "
                  f"in {(time.monotonic() - started) / 60:.1f} min")
            if rc != 0:
                failures.append((clip.stem, step))
                break   # later steps of this clip depend on this one

    if failures:
        print(f"\n[run_data2] {len(failures)} clip(s) failed:")
        for name, step in failures:
            print(f"  {name}: failed at {step}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
