"""Run the full reconstruction chain over the ``peter/`` dataset.

``peter/<dataset>/<clip>/cam_left.mp4`` → ``peter/out_<dataset>_single/<clip>/``.

*Single-view experiment*: only the left camera is reconstructed. The other views
exist and SAM 3 / Sapiens could be run on each of them independently, but VGGT
and SAM-3D-Body change when they see several views, so the multi-view run is a
separate experiment writing to ``out_<dataset>_multi/`` — not implemented here.

The chain, in order::

    sam3 → vggt → sapiens → sam3d → smplx → fuse → scale
         → kinematics_1 → contacts_1 → kinematics_2 → contacts_2 → kindyn_1

The five model stages run as ONE manifest sweep each (the model loads once and
sweeps every clip); the rest run one subprocess per clip, clip by clip, so a
finished clip is inspectable while the batch is still going. Every step is
resumable: a step whose output already exists is skipped.

A clip that fails is reported and skipped — one bad clip never kills the batch.

Knobs live in the TOMLs (``configs/pipeline_peter.toml`` for the model +
geometry stages, ``configs/human_optimization.toml`` for the human stages);
this CLI only takes targets.

Usage::

    CUDA_VISIBLE_DEVICES=3 uv run python scripts/run_peter.py --dataset campus_board
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/run_peter.py --dataset campus_board \
        --clips 01_hanging --steps fuse,scale,kinematics_1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGES_DIR = _REPO_ROOT / "scripts" / "stages"
_PETER = _REPO_ROOT / "peter"

VIEW = "cam_left"   # the single-view experiment always reconstructs the left camera

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


def _write_manifest(path: Path, clips: list[Path], out_root: Path) -> Path:
    """One JSON manifest that drives all five model stages.

    ``out_dir`` sends each clip's outputs to ``out_<dataset>_single/<clip>/``
    (the videos are all named ``cam_left.mp4``, so the stem-based default would
    collide); ``bboxes`` / ``camera`` are the inputs the pose and body stages
    read — they are written by earlier steps of this same run.
    """
    entries = [{
        "source": str(clip / f"{VIEW}.mp4"),
        "out_dir": str(out_root / clip.name),
        "bboxes": str(out_root / clip.name / "sam3" / "bboxes.npz"),
        "camera": str(out_root / clip.name / "vggt" / "camera.npz"),
    } for clip in clips]
    path.write_text(json.dumps(entries, indent=2))
    return path


def _report_people(out_root: Path, clips: list[Path]) -> None:
    """Print how many people SAM 3 tracked per clip (expected: exactly one)."""
    import numpy as np

    print("\n[run_peter] tracked people per clip:")
    for clip in clips:
        bboxes = out_root / clip.name / "sam3" / "bboxes.npz"
        if not bboxes.exists():
            print(f"  {clip.name:32s} —  (no bboxes.npz)")
            continue
        with np.load(bboxes) as data:
            count = int(data["num_objects"])
        flag = "" if count == 1 else "   <-- NOT a single person"
        print(f"  {clip.name:32s} {count}{flag}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True,
                        help="dataset folder under peter/ (campus_board | climb_wall)")
    parser.add_argument("--clips", default=None,
                        help="comma-separated clip names (default: all)")
    parser.add_argument("--steps", default=None,
                        help=f"comma-separated subset of {','.join(ALL_STEPS)} (default: all)")
    parser.add_argument("--config", type=Path,
                        default=_REPO_ROOT / "configs" / "pipeline_peter.toml",
                        help="model + geometry stage TOML")
    parser.add_argument("--human-config", type=Path,
                        default=_REPO_ROOT / "configs" / "human_optimization.toml",
                        help="human-optimization + contacts TOML")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    dataset_dir = _PETER / args.dataset
    if not dataset_dir.is_dir():
        raise SystemExit(f"[run_peter] no such dataset: {dataset_dir}")
    out_root = _PETER / f"out_{args.dataset}_single"

    clips = sorted(d for d in dataset_dir.iterdir() if (d / f"{VIEW}.mp4").is_file())
    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = [c for c in clips if c.name in wanted]
        missing = wanted - {c.name for c in clips}
        if missing:
            raise SystemExit(f"[run_peter] unknown clip(s): {sorted(missing)}")
    if not clips:
        raise SystemExit(f"[run_peter] no {VIEW}.mp4 under {dataset_dir}")

    steps = [s.strip() for s in args.steps.split(",")] if args.steps else list(ALL_STEPS)
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise SystemExit(f"[run_peter] unknown step(s) {unknown}; known: {ALL_STEPS}")

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = _write_manifest(out_root / "_clips.json", clips, out_root)
    print(f"[run_peter] {len(clips)} clip(s) → {out_root}\n[run_peter] steps: {steps}")

    for step in [s for s in steps if s in MANIFEST_STEPS]:
        script, _ = MANIFEST_STEPS[step]
        started = time.monotonic()
        # resume is the config's [<stage>].skip_existing — the sam3d / smplx CLIs
        # have no --skip-existing flag, so the TOML is the one place that sets it.
        rc = _run([sys.executable, str(_STAGES_DIR / script),
                   "--source", str(manifest), "--save-dir", str(out_root),
                   "--config", str(args.config)])
        print(f"[run_peter] {step}: rc={rc} in {(time.monotonic() - started) / 60:.1f} min")
        if step == "sam3":
            _report_people(out_root, clips)

    clip_steps = [s for s in steps if s in CLIP_OUTPUT]
    failures: list[tuple[str, str]] = []
    for index, clip in enumerate(clips, 1):
        if not clip_steps:
            break
        video, clip_dir = clip / f"{VIEW}.mp4", out_root / clip.name
        print(f"\n########## [{index}/{len(clips)}] {clip.name} ##########", flush=True)
        for step in clip_steps:
            if (clip_dir / CLIP_OUTPUT[step]).exists():
                print(f"[run_peter] skip {step} for {clip.name} (output exists)")
                continue
            started = time.monotonic()
            rc = _run(_clip_command(step, video, clip_dir, args.config, args.human_config))
            ok = rc == 0 and (clip_dir / CLIP_OUTPUT[step]).exists()
            print(f"[run_peter] {clip.name} {step}: "
                  f"{'ok' if ok else 'FAILED'} in {(time.monotonic() - started) / 60:.1f} min")
            if not ok:
                failures.append((clip.name, step))
                break   # the rest of this clip's chain depends on what just failed

    print(f"\n[run_peter] done. {len(clips) - len({c for c, _ in failures})}/{len(clips)} "
          f"clip(s) completed the per-clip chain.")
    for clip_name, step in failures:
        print(f"  FAILED {clip_name} at {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
