"""Run the reconstruction chain over clips: a video / folder of videos, or a configured dataset.

The target is either ``--source`` (one video, or a folder of videos — each a clip) with
``--save-dir`` (clip trees land in ``<save-dir>/<stem>/``), or ``--dataset <name>``: a
``[datasets.<name>]`` table of a ``datasets/<name>/configs/dataset.toml`` — a root of clips
(``<root>/<clip>/video.mp4``, or ``<clip>/<view>.mp4`` on the camera rigs) and an output tree
``<out>/<clip>/``. Both produce the same ``out/<stem>/`` trees. The chain, in order::

    sam3 → vggt → sapiens → sam3d → smplx → fuse → scale [→ gravity]
         → kinematics_1 → contacts_1 → kinematics_2 → contacts_2 → kindyn_1

A ``--source`` run may run any of the steps above, in that order; the pipeline config's
``[steps].run`` is its default selection and ``[human_optimization].config`` its human config.
A dataset names its chain and both configs in its table. Every stage
is a subprocess of this interpreter — per-stage model memory never co-resides and the
pymomentum ``_pym_preload`` re-exec happens cleanly — and the command line of each comes from
``bvr.runner`` (``python -m bvr.stages.<name>``, the same stages run standalone).

The five model stages run as ONE manifest sweep each (the model loads once and sweeps every
clip); the rest run one subprocess per clip, clip by clip. Every step is resumable: a
per-clip step whose output exists is skipped, a sweep skips finished items through the
stage's own ``skip_existing`` knob, and ``--skip-existing`` leaves out a sweep every selected
clip has finished, so no model is loaded for nothing. A clip that fails is reported and
skipped — one bad clip never kills the batch. ``--shard i/n`` takes every n-th clip, so n
runs on n GPUs fill one output tree; the manifest is named after the clip selection, so the
shards never read each other's.

``--views`` (rigs only) says which cameras to reconstruct. One view — the default, the
rig's first camera — is the single-view experiment, the whole chain into ``out``. Any
other selection goes to ``out_multi``: all videos go into ONE VGGT forward, so the cameras
land in one world and every depth map fuses into one cloud; only sam3, vggt and fuse run,
every later stage assumes one view. The first camera is never tracked twice (a finished
single-view run's masks are copied in); each extra view is tracked into its own
``out_multi/_<view>/`` tree and linked in as ``<clip>/sam3/<view>/``, where fuse looks for
it and no other stage ever looks.

``--variant <name>`` re-runs the human chain into ``<clip>/human_optim_<name>/`` with that
variant's config, leaving ``human_optim/`` intact. The stock stages are pointed at a
symlink farm ``<clip>/_<name>/`` whose ``human_optim`` IS the variant directory and whose
inputs link back to the real tree. ``[variants.*]`` tables of the dataset configs hold the
ones with their own chain: ``floorplane`` builds a synthetic support scene the farm owns;
``gt_contact`` replaces the estimated contacts with the rigs' measured board contacts and
draws the force overlay. ``--view-farm`` builds ``<out>/_view_<variant>/`` (clip-named
aliases of the farms) for ``scripts/view_3d.py``.

Knobs live in the TOMLs; this CLI only takes targets. Usage::

    uv run --no-sync python scripts/pipeline.py --source video.mp4 --save-dir out/
    ... --source clips/ --save-dir out/ --steps sam3,vggt
    CUDA_VISIBLE_DEVICES=0 ... --dataset parkour
    ... --dataset opencap --clips subject2_DJ1_cam1 --steps fuse,scale,kinematics_1
    ... --dataset opencap --shard 1/3          # one process per GPU, disjoint thirds
    ... --dataset opencap --variant floorplane
    ... --dataset climb_wall_3 --variant gt_contact
    ... --dataset campus_board --views cam_left,cam_right
    ... --dataset campus_board --dry-run       # print the commands, run nothing
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

from bvr.config import load_config
from bvr.paths import DATA_ROOT, REPO_ROOT
from bvr.runner import (
    ALL_STEPS, CLIP_OUTPUT, GEOMETRY_STEPS, MANIFEST_STEPS, SUPPORT_STEPS, VARIANT_STEPS,
    build_farm, build_view_farm, stage_command, sweep_command,
)
from bvr.source import IMAGE_EXTS, VIDEO_EXTS

#: A clip: its name (the output directory) and one video per camera view.
Clip = tuple[str, list[Path]]
#: The full chain of a --source run, in order (``overlay`` is a variant's own renderer).
SOURCE_CHAIN = [s for s in (*MANIFEST_STEPS, *CLIP_OUTPUT) if s != "overlay"]


def load_datasets() -> tuple[dict[str, dict], dict[str, dict]]:
    """Every ``datasets/*/configs/dataset.toml``: the ``[datasets.*]`` tables, their data paths
    made absolute under ``DATA_ROOT``, and the ``[variants.*]`` tables (variants are global)."""
    datasets: dict[str, dict] = {}
    variants: dict[str, dict] = {}
    for path in sorted((REPO_ROOT / "datasets").glob("*/configs/dataset.toml")):
        config = load_config(path)
        for name, dataset in config.get("datasets", {}).items():
            if name in datasets:
                raise SystemExit(f"[pipeline] dataset {name!r} defined twice ({path})")
            for key in ("root", "out", "out_multi"):
                if key in dataset:
                    dataset[key] = DATA_ROOT / dataset[key]
            datasets[name] = dataset
        variants.update(config.get("variants", {}))
    return datasets, variants


DATASETS, VARIANTS = load_datasets()
#: Variants whose chain is not the plain :data:`VARIANT_STEPS`.
VARIANT_CHAINS: dict[str, list[str]] = {
    name: list(variant["steps"]) for name, variant in VARIANTS.items() if "steps" in variant}


def owns_scene(variant: str | None) -> bool:
    """Does this variant's farm own a real ``geometry/`` with its own ``scene.npz``?

    True exactly for the variants whose chain builds a synthetic support scene. Everywhere
    else ``geometry`` is one symlink to the clip's real tree, so a support step pointed at
    that farm would overwrite the fused cloud the baseline run and every other variant read.
    """
    return any(step in SUPPORT_STEPS for step in VARIANT_CHAINS.get(variant or "", []))


def _run(command: list[str], dry_run: bool = False) -> int:
    print(f"\n=== {' '.join(command[1:])} ===", flush=True)
    return 0 if dry_run else subprocess.run(command).returncode


def _manifest_name(clips: list[Clip]) -> str:
    """Manifest filename for this exact clip selection.

    Named after the selection, not fixed, so two runs over different clips of the same
    tree — the normal way to use several GPUs — cannot overwrite each other's manifest
    and start processing each other's clips.
    """
    tag = hashlib.sha1(",".join(name for name, _ in clips).encode()).hexdigest()[:8]
    return f"_clips_{tag}.json"


def _parse_shard(text: str) -> tuple[int, int]:
    """``"2/3"`` → ``(2, 3)``: this shard's 1-based index and the number of shards."""
    index, _, total = text.partition("/")
    shard, n_shards = int(index), int(total or 1)
    if not 1 <= shard <= n_shards:
        raise SystemExit(f"[pipeline] bad --shard {text!r}: need 1 <= i <= n in 'i/n'")
    return shard, n_shards


def _sam3_dir(clip_dir: Path, views: list[str], v: int) -> Path:
    """Where view ``v``'s SAM 3 masks live: ``sam3/`` for the first view — the layout
    every other stage reads — and ``sam3/<view>/`` for the extra ones."""
    return clip_dir / "sam3" if v == 0 else clip_dir / "sam3" / views[v]


def _link(link: Path, target: Path) -> None:
    """Point ``link`` at ``target``; an existing entry is left alone."""
    if link.is_symlink() or link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve())


def _reuse_masks(single_root: Path, out_root: Path, clips: list[Clip]) -> None:
    """Copy a finished single-view run's SAM 3 masks into this run's tree.

    Copied, not linked. The tracker writes into ``sam3/`` whenever it does not skip an
    item, and it follows symlinks — so a link here would let a re-track overwrite the
    finished single-view run in place. The masks are ~2 MB a clip. Clips the single-view
    run hasn't tracked yet are left out and tracked here.
    """
    if single_root == out_root:
        return
    for name, _ in clips:
        src = single_root / name / "sam3"
        dst = out_root / name / "sam3"
        if not src.is_dir() or (dst / "bboxes.npz").exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for entry in src.iterdir():
            if entry.is_dir():
                shutil.copytree(entry, dst / entry.name, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, dst / entry.name)


def _track_extra_views(out_root: Path, clips: list[Clip], views: list[str], config: Path,
                       *, skip_existing: bool, dry_run: bool) -> None:
    """Run SAM 3 over every view after the first, one manifest sweep per view.

    Each view is tracked into its own ``_<view>/`` tree, so the tracker keeps its usual
    ``sam3/`` layout, and is then linked into the clip where fuse expects that view's masks.
    """
    for v, view in enumerate(views[1:], start=1):
        view_root = out_root / f"_{view}"
        if skip_existing and all((view_root / name / "sam3" / "bboxes.npz").exists() for name, _ in clips):
            print(f"[pipeline] sam3 {view}: every clip has sam3/bboxes.npz, skipping the sweep")
        else:
            manifest = view_root / _manifest_name(clips)
            if not dry_run:
                view_root.mkdir(parents=True, exist_ok=True)
                manifest.write_text(json.dumps([{
                    "source": str(videos[v]),
                    "out_dir": str(view_root / name),
                } for name, videos in clips], indent=2))
            started = time.monotonic()
            rc = _run(sweep_command("sam3", manifest, view_root, config), dry_run)
            print(f"[pipeline] sam3 {view}: rc={rc} in {(time.monotonic() - started) / 60:.1f} min")
            if rc != 0:
                continue   # a dangling link would look like masks that are simply empty
        if not dry_run:
            for name, _ in clips:
                _link(_sam3_dir(out_root / name, views, v), view_root / name / "sam3")


def _report_people(out_root: Path, clips: list[Clip], views: list[str]) -> None:
    """Print how many people SAM 3 tracked per clip and view (expected: exactly one)."""
    import numpy as np

    print("\n[pipeline] tracked people per clip:")
    for name, _ in clips:
        counts = []
        for v in range(len(views)):
            bboxes = _sam3_dir(out_root / name, views, v) / "bboxes.npz"
            if not bboxes.exists():
                counts.append("—")
                continue
            with np.load(bboxes) as data:
                counts.append(str(int(data["num_objects"])))
        flag = "" if counts == ["1"] * len(views) else "   <-- NOT a single person"
        print(f"  {name:32s} {'/'.join(counts)}{flag}")


def _source_clips(source: Path) -> list[Clip]:
    """A video, or every video of a folder, each named after its stem."""
    if source.is_dir():
        videos = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
        if not videos:
            raise SystemExit(f"[pipeline] no media files in {source}")
    elif source.is_file():
        videos = [source]
    else:
        raise SystemExit(f"[pipeline] --source not found: {source}")
    return [(video.stem, [video]) for video in videos]


def _dataset_clips(dataset: dict, views: list[str] | None) -> list[Clip]:
    """The dataset's clips: every ``<root>/<clip>/`` holding the configured video(s)."""
    root = dataset["root"]
    if not root.is_dir():
        raise SystemExit(f"[pipeline] no such dataset root: {root}")
    if views:
        names = [f"{view}.mp4" for view in views]
        return [(d.name, [d / n for n in names])
                for d in sorted(root.iterdir()) if all((d / n).is_file() for n in names)]
    return [(video.parent.name, [video]) for video in sorted(root.glob(dataset["clips"]))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_argument_group("target: a video / folder, or a dataset")
    target.add_argument("--source", type=Path, default=None,
                        help="video file or folder of video files (each a clip)")
    target.add_argument("--save-dir", type=Path, default=None,
                        help="--source outputs land in <save-dir>/<stem>/")
    target.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.toml",
                        help="pipeline TOML of a --source run: stage knobs, [steps].run, "
                             "[human_optimization].config (default: configs/pipeline.toml)")
    target.add_argument("--dataset", default=None, choices=sorted(DATASETS),
                        help="a [datasets.*] table of datasets/*/configs/dataset.toml")
    parser.add_argument("--clips", default=None,
                        help="comma-separated clip names (default: all)")
    parser.add_argument("--steps", default=None,
                        help=f"comma-separated subset of this run's chain, in its order (default: "
                             f"all); steps: {','.join(ALL_STEPS)}")
    parser.add_argument("--shard", default="1/1",
                        help="'i/n': process every n-th clip, so n runs share one output tree")
    parser.add_argument("--variant", default=None,
                        help=f"re-run {'/'.join(VARIANT_STEPS)} into <clip>/human_optim_<variant>/ "
                             "with configs/human_optimization_<variant>.toml, leaving human_optim/ intact; "
                             f"{'/'.join(VARIANT_CHAINS)} run their own chain ([variants.*] of the dataset configs)")
    parser.add_argument("--views", default=None,
                        help="comma-separated cameras of a rig dataset; several reconstruct the "
                             "clip from all of them at once (default: the rig's first camera)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="also leave out a model sweep every selected clip has finished (the "
                             "stages skip finished items anyway, but only after loading the model)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands and touch nothing")
    parser.add_argument("--view-farm", action="store_true",
                        help="only build <out>/_view_<variant>/ (clip-named aliases of the variant "
                             "farms) for scripts/view_3d.py, then exit")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if (args.dataset is None) == (args.source is None):
        parser.error("give --source <video|folder> --save-dir <out>, or --dataset <name>")
    if args.source is not None and args.save_dir is None:
        parser.error("--source needs --save-dir")
    recipe = VARIANTS.get(args.variant, {}) if args.variant else {}

    dataset = DATASETS[args.dataset] if args.dataset else None
    rig_views = dataset.get("views") if dataset else None
    if args.views and not rig_views:
        raise SystemExit(f"[pipeline] --views: {args.dataset or args.source} has one video per clip, "
                         "no camera views")
    views = ([v.strip() for v in args.views.split(",") if v.strip()] if args.views
             else rig_views[:1] if rig_views else None)
    if views and len(set(views)) != len(views):
        raise SystemExit(f"[pipeline] repeated view in {views}")
    # The single-view tree belongs to the rig's first camera alone. Any other view list —
    # a different camera, or several — goes to the multi tree, so a finished single-view
    # run can never be written over by a differently-framed one.
    multi = views is not None and views != rig_views[:1]
    if dataset:
        out_root = dataset["out_multi"] if multi else dataset["out"]
        pipeline_config = REPO_ROOT / dataset["pipeline_config"]
        human_config = REPO_ROOT / dataset["human_config"]
        chain = list(dataset["multi_steps"] if views and len(views) > 1 else dataset["steps"])
        default_steps = chain
        clips = _dataset_clips(dataset, views)
    else:
        out_root = args.save_dir
        pipeline_config = args.config
        if not pipeline_config.exists():
            raise SystemExit(f"[pipeline] config not found: {pipeline_config}")
        orchestrator = load_config(pipeline_config)
        human_config = REPO_ROOT / orchestrator["human_optimization"]["config"]
        chain, default_steps = SOURCE_CHAIN, list(orchestrator["steps"]["run"])
        clips = _source_clips(args.source)
    if args.view_farm:
        if not args.variant:
            raise SystemExit("[pipeline] --view-farm needs --variant")
        print(f"[pipeline] viewer root: {build_view_farm(out_root, args.variant, dry_run=args.dry_run)}"
              f"{'  (dry run)' if args.dry_run else ''}")
        return 0
    if args.variant:
        human_config = REPO_ROOT / recipe.get("config", f"configs/human_optimization_{args.variant}.toml")
        chain = default_steps = list(VARIANT_CHAINS.get(args.variant, VARIANT_STEPS))

    if "requires" in recipe:
        clips = [c for c in clips if (out_root / c[0] / recipe["requires"]).exists()]
    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = [c for c in clips if c[0] in wanted]
        if missing := wanted - {c[0] for c in clips}:
            raise SystemExit(f"[pipeline] unknown clip(s): {sorted(missing)}")
    if not clips:
        raise SystemExit(f"[pipeline] no clip to run under {dataset['root'] if dataset else args.source}"
                         + (f" with {out_root}/<clip>/{recipe['requires']}" if "requires" in recipe else ""))
    shard, n_shards = _parse_shard(args.shard)
    clips = clips[shard - 1::n_shards]
    if not clips:
        raise SystemExit(f"[pipeline] shard {shard}/{n_shards} is empty")

    steps = [s.strip() for s in args.steps.split(",")] if args.steps else list(default_steps)
    if unknown := [s for s in steps if s not in ALL_STEPS]:
        raise SystemExit(f"[pipeline] unknown step(s) {unknown}; available: {ALL_STEPS}")
    if args.variant and any(s in MANIFEST_STEPS for s in steps):
        raise SystemExit("[pipeline] --variant re-runs the human chain only; the model "
                         f"stages ({', '.join(MANIFEST_STEPS)}) are shared with the baseline run")
    if args.variant and any(s in GEOMETRY_STEPS for s in steps):
        raise SystemExit(f"[pipeline] {', '.join(GEOMETRY_STEPS)} cannot run under "
                         "--variant: the farm links geometry/ and geocalib/ back to the real "
                         "clip, so they would rewrite the inputs every run shares")
    if any(s in SUPPORT_STEPS for s in steps) and not owns_scene(args.variant):
        raise SystemExit(
            f"[pipeline] {', '.join(SUPPORT_STEPS)} need a variant whose farm owns its "
            f"geometry/ ({', '.join(v for v in VARIANT_CHAINS if owns_scene(v))}); under "
            f"--variant {args.variant!r} that directory is a symlink and the synthetic scene "
            "would overwrite the clip's real fused geometry/scene.npz")
    # A chain is an order, not a set: running contacts_1 before the kinematics_1 it reads, or
    # kindyn_1 twice, silently scores the wrong body.
    places = [chain.index(s) for s in steps if s in chain]
    if len(places) != len(steps) or any(a >= b for a, b in zip(places, places[1:])):
        raise SystemExit(f"[pipeline] --steps must be a subset of this run's chain, in its "
                         f"order {chain}; got {steps}")
    for path in (human_config, pipeline_config):
        if not path.exists():
            raise SystemExit(f"[pipeline] config not found: {path}")

    sweeps = [s for s in steps if s in MANIFEST_STEPS]
    # --skip-existing: a sweep every selected clip has finished is not launched at all.
    finished = {s for s in sweeps if args.skip_existing
                and all((out_root / name / MANIFEST_STEPS[s][1]).exists() for name, _ in clips)}
    manifest = out_root / _manifest_name(clips)
    if set(sweeps) - finished and not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        # ``sources`` lists every camera view, which is how the vggt stage puts them all in
        # one forward; ``bboxes`` / ``camera`` are written by earlier steps of this same run.
        manifest.write_text(json.dumps([{
            "source": str(videos[0]),
            **({"sources": [str(v) for v in videos]} if views else {}),
            "out_dir": str(out_root / name),
            "bboxes": str(out_root / name / "sam3" / "bboxes.npz"),
            "camera": str(out_root / name / "vggt" / "camera.npz"),
        } for name, videos in clips], indent=2))
    print(f"[pipeline] {len(clips)} clip(s){f' x {len(views)} view(s)' if views else ''} → {out_root}"
          f"{'' if n_shards == 1 else f' (shard {shard}/{n_shards})'}\n"
          f"[pipeline] steps: {steps}{'  (dry run)' if args.dry_run else ''}")

    for step in sweeps:
        _, output = MANIFEST_STEPS[step]
        if step in finished:
            print(f"[pipeline] {step}: every clip has {output}, skipping the sweep")
        else:
            if step == "sam3" and views and views[0] == rig_views[0] and not args.dry_run:
                _reuse_masks(dataset["out"], out_root, clips)   # the first camera is never tracked twice
            started = time.monotonic()
            rc = _run(sweep_command(step, manifest, out_root, pipeline_config), args.dry_run)
            print(f"[pipeline] {step}: rc={rc} in {(time.monotonic() - started) / 60:.1f} min")
        if step == "sam3" and views:
            _track_extra_views(out_root, clips, views, pipeline_config,
                               skip_existing=args.skip_existing, dry_run=args.dry_run)
            if not args.dry_run:
                _report_people(out_root, clips, views)

    clip_steps = [s for s in steps if s not in MANIFEST_STEPS]
    failures: list[tuple[str, str]] = []
    for index, (name, videos) in enumerate(clips, 1):
        if not clip_steps:
            break
        clip_dir = out_root / name
        if args.variant:
            # The stages are pointed at the farm, so every path below — including the
            # resume check — resolves through it into human_optim_<variant>/. A dry run
            # only checks the farm's existing links.
            clip_dir = build_farm(clip_dir, args.variant, synthetic_scene=owns_scene(args.variant),
                                  inherit=tuple(recipe.get("inherit", ())), dry_run=args.dry_run)
        # A synthetic-scene variant that has not built its scene yet would run the whole human
        # chain with the scene terms silently off (bvr.human_optim.stages._load_scene prints
        # a note and carries on), which looks like a finished variant and is not one.
        if (owns_scene(args.variant) and clip_steps[0] not in SUPPORT_STEPS
                and not (clip_dir / "geometry" / "scene.npz").exists()):
            raise SystemExit(f"[pipeline] {name}: {clip_dir}/geometry/scene.npz is "
                             f"missing — run --variant {args.variant} --steps "
                             f"{next(iter(SUPPORT_STEPS))} first")
        print(f"\n########## [{index}/{len(clips)}] {name} ##########", flush=True)
        for step in clip_steps:
            done = CLIP_OUTPUT.get(step)
            if done and (clip_dir / done).exists():
                print(f"[pipeline] {name}/{step}: exists, skipping")
                continue
            command = stage_command(step, clip_dir, videos=videos, pipeline_config=pipeline_config,
                                    human_config=human_config, variant=args.variant,
                                    boards=recipe.get("contacts") == "boards",
                                    overlay_module=recipe.get("overlay"))
            started = time.monotonic()
            rc = _run(command, args.dry_run)
            ok = args.dry_run or (rc == 0 and (done is None or (clip_dir / done).exists()))
            print(f"[pipeline] {name}/{step}: {'ok' if ok else 'FAILED'} (rc={rc}) "
                  f"in {(time.monotonic() - started) / 60:.1f} min")
            if not ok:
                failures.append((name, step))
                break   # later steps of this clip depend on this one

    print(f"\n[pipeline] done. {len(clips) - len({c for c, _ in failures})}/{len(clips)} "
          f"clip(s) completed the per-clip chain.")
    for name, step in failures:
        print(f"  FAILED {name} at {step}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
