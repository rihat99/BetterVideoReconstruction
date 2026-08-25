"""Run the full reconstruction chain over the ``peter/`` dataset.

``peter/<dataset>/<clip>/cam_left.mp4`` → ``peter/out_<dataset>_single/<clip>/``.

Every clip was filmed by two synchronized static cameras. ``--views`` says which
ones to reconstruct:

* one view (the default, ``cam_left``) — the single-view experiment: the whole
  chain, into ``out_<dataset>_single/``.
* two views (``--views cam_left,cam_right``) — both videos go into ONE VGGT
  forward, so the two cameras land in one common world, and both depth maps fuse
  into one point cloud. Into ``out_<dataset>_multi/``. Only ``sam3``, ``vggt``
  and ``fuse`` run: every later stage still assumes a single view.

cam_left is never tracked twice: a finished ``out_<dataset>_single`` run is
linked into the multi tree. Each extra view is tracked into its own
``out_<dataset>_multi/_<view>/`` tree and then linked in as
``<clip>/sam3/<view>/``, which is where fuse looks for that view's masks. So the
tracker keeps its usual layout, and stages that find people by digit-named dirs
under ``sam3/`` never see the extra view at all.

The chain, in order::

    sam3 → vggt → sapiens → sam3d → smplx → fuse → scale
         → kinematics_1 → contacts_1 → kinematics_2 → contacts_2 → kindyn_1

The five model stages run as ONE manifest sweep each (the model loads once and
sweeps every clip); the rest run one subprocess per clip, clip by clip, so a
finished clip is inspectable while the batch is still going. Every step is
resumable: a step whose output already exists is skipped.

``--shard i/n`` takes every n-th clip, so n runs on n GPUs fill one output tree
between them without hand-writing ``--clips`` lists. The manifest is named after
the clip selection (see :func:`_manifest_name`), so the shards never read each
other's.

A clip that fails is reported and skipped — one bad clip never kills the batch.

Knobs live in the TOMLs (``configs/pipeline_peter.toml`` for the model +
geometry stages, ``configs/human_optimization_peter.toml`` for the human
stages — the general config restricted to hand/foot contact frames);
this CLI only takes targets.

Usage::

    CUDA_VISIBLE_DEVICES=3 uv run python scripts/run_peter.py --dataset campus_board
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/run_peter.py --dataset campus_board \
        --clips 01_hanging --steps fuse,scale,kinematics_1
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/run_peter.py --dataset campus_board \
        --views cam_left,cam_right
    # two GPUs over one dataset: one process each, disjoint halves of the clips
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/run_peter.py --dataset climb_wall --shard 1/2
    CUDA_VISIBLE_DEVICES=6 uv run python scripts/run_peter.py --dataset climb_wall --shard 2/2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGES_DIR = _REPO_ROOT / "scripts" / "stages"
_PETER = _REPO_ROOT / "peter"

SINGLE_VIEW = "cam_left"   # the view the out_<dataset>_single tree was built from

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
MULTI_STEPS = ["sam3", "vggt", "fuse"]   # everything after fuse still assumes one view


def _clip_command(step: str, videos: list[Path], clip_dir: Path,
                  pipeline_config: Path, human_config: Path) -> list[str]:
    """Argv for one per-clip stage (all paths absolute)."""
    stage = lambda name: str(_STAGES_DIR / name)  # noqa: E731
    common = ["--source", str(videos[0]), "--predictions", str(clip_dir)]
    if step == "fuse":
        # fuse takes one source per camera view, in the camera.npz view order.
        return [sys.executable, stage("fuse_geometry.py"),
                "--source", ",".join(str(v) for v in videos),
                "--predictions", str(clip_dir), "--config", str(pipeline_config)]
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


def _manifest_name(clips: list[Path]) -> str:
    """Manifest filename for this exact clip selection.

    Named after the selection, not fixed, so two runs over different clips of the
    same tree — the normal way to use both GPUs — cannot overwrite each other's
    manifest and start processing each other's clips.
    """
    tag = hashlib.sha1(",".join(c.name for c in clips).encode()).hexdigest()[:8]
    return f"_clips_{tag}.json"


def _write_manifest(path: Path, clips: list[Path], out_root: Path, views: list[str]) -> Path:
    """One JSON manifest that drives all five model stages.

    ``out_dir`` sends each clip's outputs to ``out_<dataset>_<mode>/<clip>/``
    (the videos are all named ``cam_left.mp4``, so the stem-based default would
    collide); ``sources`` lists every camera view, which is how the vggt stage
    puts them all in one forward; ``bboxes`` / ``camera`` are the inputs the pose
    and body stages read — they are written by earlier steps of this same run.
    """
    entries = [{
        "source": str(clip / f"{views[0]}.mp4"),
        "sources": [str(clip / f"{view}.mp4") for view in views],
        "out_dir": str(out_root / clip.name),
        "bboxes": str(out_root / clip.name / "sam3" / "bboxes.npz"),
        "camera": str(out_root / clip.name / "vggt" / "camera.npz"),
    } for clip in clips]
    path.write_text(json.dumps(entries, indent=2))
    return path


def _sam3_dir(clip_dir: Path, views: list[str], v: int) -> Path:
    """Where view ``v``'s SAM 3 masks live: ``sam3/`` for the first view — the
    layout every other stage reads — and ``sam3/<view>/`` for the extra ones."""
    return clip_dir / "sam3" if v == 0 else clip_dir / "sam3" / views[v]


def _link(link: Path, target: Path) -> None:
    """Point ``link`` at ``target``; an existing entry is left alone."""
    if link.is_symlink() or link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve())


def _reuse_masks(single_root: Path, out_root: Path, clips: list[Path]) -> None:
    """Copy a finished single-view run's SAM 3 masks into this run's tree.

    Copied, not linked. The tracker writes into ``sam3/`` whenever it does not
    skip an item, and it follows symlinks — so a link here would let a re-track
    overwrite the finished single-view run in place. The masks are ~2 MB a clip.
    Clips the single-view run hasn't tracked yet are left out and tracked here.
    """
    if single_root == out_root:
        return
    for clip in clips:
        src = single_root / clip.name / "sam3"
        dst = out_root / clip.name / "sam3"
        if not src.is_dir() or (dst / "bboxes.npz").exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for entry in src.iterdir():
            if entry.is_dir():
                shutil.copytree(entry, dst / entry.name, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, dst / entry.name)


def _track_extra_views(out_root: Path, clips: list[Path], views: list[str], config: Path) -> None:
    """Run SAM 3 over every view after the first, one manifest sweep per view.

    Each view is tracked into its own ``_<view>/`` tree, so the tracker keeps its
    usual ``sam3/`` layout, and is then linked into the clip where fuse expects
    that view's masks.
    """
    for v, view in enumerate(views[1:], start=1):
        view_root = out_root / f"_{view}"
        view_root.mkdir(parents=True, exist_ok=True)
        manifest = view_root / _manifest_name(clips)
        manifest.write_text(json.dumps([{
            "source": str(clip / f"{view}.mp4"),
            "out_dir": str(view_root / clip.name),
        } for clip in clips], indent=2))
        started = time.monotonic()
        rc = _run([sys.executable, str(_STAGES_DIR / "track_sam3.py"),
                   "--source", str(manifest), "--save-dir", str(view_root),
                   "--config", str(config)])
        print(f"[run_peter] sam3 {view}: rc={rc} in {(time.monotonic() - started) / 60:.1f} min")
        if rc != 0:
            continue   # a dangling link would look like masks that are simply empty
        for clip in clips:
            _link(_sam3_dir(out_root / clip.name, views, v), view_root / clip.name / "sam3")


def _parse_shard(text: str) -> tuple[int, int]:
    """``"2/3"`` → ``(2, 3)``: this shard's 1-based index and the number of shards."""
    index, _, total = text.partition("/")
    shard, n_shards = int(index), int(total or 1)
    if not 1 <= shard <= n_shards:
        raise SystemExit(f"[run_peter] bad --shard {text!r}: need 1 <= i <= n in 'i/n'")
    return shard, n_shards


def _report_people(out_root: Path, clips: list[Path], views: list[str]) -> None:
    """Print how many people SAM 3 tracked per clip and view (expected: exactly one)."""
    import numpy as np

    print("\n[run_peter] tracked people per clip:")
    for clip in clips:
        counts = []
        for v in range(len(views)):
            bboxes = _sam3_dir(out_root / clip.name, views, v) / "bboxes.npz"
            if not bboxes.exists():
                counts.append("—")
                continue
            with np.load(bboxes) as data:
                counts.append(str(int(data["num_objects"])))
        flag = "" if counts == ["1"] * len(views) else "   <-- NOT a single person"
        print(f"  {clip.name:32s} {'/'.join(counts)}{flag}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True,
                        help="dataset folder under peter/ (campus_board | climb_wall)")
    parser.add_argument("--clips", default=None,
                        help="comma-separated clip names (default: all)")
    parser.add_argument("--views", default="cam_left",
                        help="comma-separated camera views; several views reconstruct the "
                             "clip from all of them at once (default: cam_left)")
    parser.add_argument("--steps", default=None,
                        help=f"comma-separated subset of {','.join(ALL_STEPS)} (default: all; "
                             f"several views run only {','.join(MULTI_STEPS)})")
    parser.add_argument("--shard", default="1/1",
                        help="'i/n': process every n-th clip, so n runs on n GPUs share "
                             "one output tree (default: 1/1, all clips)")
    parser.add_argument("--config", type=Path,
                        default=_REPO_ROOT / "configs" / "pipeline_peter.toml",
                        help="model + geometry stage TOML")
    parser.add_argument("--human-config", type=Path,
                        default=_REPO_ROOT / "configs" / "human_optimization_peter.toml",
                        help="human-optimization + contacts TOML")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    dataset_dir = _PETER / args.dataset
    if not dataset_dir.is_dir():
        raise SystemExit(f"[run_peter] no such dataset: {dataset_dir}")
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    if len(set(views)) != len(views):
        raise SystemExit(f"[run_peter] repeated view in {views}")
    # The single-view tree belongs to SINGLE_VIEW alone. Any other view list —
    # a different camera, or several — goes to the multi tree, so a finished
    # single-view run can never be written over by a differently-framed one.
    single_root = _PETER / f"out_{args.dataset}_single"
    out_root = single_root if views == [SINGLE_VIEW] else _PETER / f"out_{args.dataset}_multi"

    clips = sorted(d for d in dataset_dir.iterdir()
                   if all((d / f"{view}.mp4").is_file() for view in views))
    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = [c for c in clips if c.name in wanted]
        missing = wanted - {c.name for c in clips}
        if missing:
            raise SystemExit(f"[run_peter] unknown clip(s): {sorted(missing)}")
    if not clips:
        raise SystemExit(f"[run_peter] no clip under {dataset_dir} has all of {views}")
    shard, n_shards = _parse_shard(args.shard)
    clips = clips[shard - 1::n_shards]
    if not clips:
        raise SystemExit(f"[run_peter] shard {shard}/{n_shards} is empty")

    known = ALL_STEPS if len(views) == 1 else MULTI_STEPS
    steps = [s.strip() for s in args.steps.split(",")] if args.steps else list(known)
    unavailable = [s for s in steps if s not in known]
    if unavailable:
        raise SystemExit(f"[run_peter] step(s) {unavailable} unavailable with "
                         f"{len(views)} view(s); available: {known}")

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = _write_manifest(out_root / _manifest_name(clips), clips, out_root, views)
    print(f"[run_peter] {len(clips)} clip(s) x {len(views)} view(s) → {out_root}"
          f"{'' if n_shards == 1 else f' (shard {shard}/{n_shards})'}\n"
          f"[run_peter] steps: {steps}")

    for step in [s for s in steps if s in MANIFEST_STEPS]:
        script, _ = MANIFEST_STEPS[step]
        if step == "sam3" and views[0] == SINGLE_VIEW:
            _reuse_masks(single_root, out_root, clips)   # cam_left is never tracked twice
        started = time.monotonic()
        # resume is the config's [<stage>].skip_existing — the sam3d / smplx CLIs
        # have no --skip-existing flag, so the TOML is the one place that sets it.
        rc = _run([sys.executable, str(_STAGES_DIR / script),
                   "--source", str(manifest), "--save-dir", str(out_root),
                   "--config", str(args.config)])
        print(f"[run_peter] {step}: rc={rc} in {(time.monotonic() - started) / 60:.1f} min")
        if step == "sam3":
            _track_extra_views(out_root, clips, views, args.config)
            _report_people(out_root, clips, views)

    clip_steps = [s for s in steps if s in CLIP_OUTPUT]
    failures: list[tuple[str, str]] = []
    for index, clip in enumerate(clips, 1):
        if not clip_steps:
            break
        videos = [clip / f"{view}.mp4" for view in views]
        clip_dir = out_root / clip.name
        print(f"\n########## [{index}/{len(clips)}] {clip.name} ##########", flush=True)
        for step in clip_steps:
            if (clip_dir / CLIP_OUTPUT[step]).exists():
                print(f"[run_peter] skip {step} for {clip.name} (output exists)")
                continue
            started = time.monotonic()
            rc = _run(_clip_command(step, videos, clip_dir, args.config, args.human_config))
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
