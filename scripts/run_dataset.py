"""Run the reconstruction chain over one dataset of ``configs/datasets.toml``.

A dataset is a root of clips (``<root>/<clip>/video.mp4``, or ``<clip>/<view>.mp4`` on
the camera rigs) and an output tree ``<out>/<clip>/``. The chain, in order::

    sam3 → vggt → sapiens → sam3d → smplx → fuse → scale [→ gravity]
         → kinematics_1 → contacts_1 → kinematics_2 → contacts_2 → kindyn_1

The five model stages run as ONE manifest sweep each (the model loads once and sweeps
every clip); the rest run one subprocess per clip, clip by clip. Every step is resumable:
a per-clip step whose output exists is skipped, a sweep skips finished items through the
stage's own ``skip_existing`` knob, and ``--skip-existing`` leaves out a sweep every
selected clip has finished, so no model is loaded for nothing. A clip that fails is
reported and skipped — one bad clip never kills the batch. ``--shard i/n`` takes every
n-th clip, so n runs on n GPUs fill one output tree; the manifest is named after the
clip selection, so the shards never read each other's.

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
inputs link back to the real tree. ``[variants.*]`` in datasets.toml hold the ones with
their own chain: ``floorplane`` builds a synthetic support scene the farm owns;
``gt_contact`` replaces the estimated contacts with the rigs' measured board contacts and
draws the force overlay. ``--view-farm`` builds ``<out>/_view_<variant>/`` (clip-named
aliases of the farms) for ``scripts/view_3d.py``.

Knobs live in the TOMLs; this CLI only takes targets. Usage::

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/run_dataset.py --dataset parkour
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
import tomllib
from pathlib import Path

from bvr.paths import DATA_ROOT, REPO_ROOT

# step → (stage module under bvr.stages, output path relative to the clip's output tree).
# These five sweep every clip in one process; the rest run per clip.
MANIFEST_STEPS: dict[str, tuple[str, str]] = {
    "sam3":    ("track_sam3",           "sam3/bboxes.npz"),
    "vggt":    ("estimate_camera_vggt", "vggt/camera.npz"),
    "sapiens": ("estimate_sapiens",     "sapiens/pose.npz"),
    "sam3d":   ("estimate_body_sam3d",  "sam3d/params.npz"),
    "smplx":   ("convert_mhr_to_smplx", "sam3d/smplx_params.npz"),
}
#: per-clip step → the file that means it is done.
CLIP_OUTPUT: dict[str, str] = {
    "fuse":         "geometry/transform.npz",
    "scale":        "geometry/scale.npz",
    "gravity":      "geocalib/gravity.npz",
    "kinematics_1": "human_optim/kinematic_1.npz",
    "contacts_1":   "human_optim/contacts_1.npz",
    "kinematics_2": "human_optim/kinematic_2.npz",
    "contacts_2":   "human_optim/contacts_2.npz",
    "kindyn_1":     "human_optim/kindyn_1.npz",
    "overlay":      "human_optim/kindyn_forces_overlay.mp4",
}
#: Synthetic-support-scene steps → the body their plateaus are read off (None = floor only).
#: Both write the same ``geometry/scene.npz``, so an existence check cannot tell them apart:
#: they are never resume-skipped, which costs seconds and is safe because they are idempotent.
SUPPORT_STEPS: dict[str, str | None] = {
    "support_scene_floor":    None,
    "support_scene_plateaus": "kinematic_1",
}
ALL_STEPS = [*MANIFEST_STEPS, *CLIP_OUTPUT, *SUPPORT_STEPS]
#: The plain variant chain. kinematics_1 is recomputed too — a variant that changes the
#: contact-frame allowlist changes its auto-contacts.
VARIANT_STEPS = ["kinematics_1", "contacts_1", "kinematics_2", "contacts_2", "kindyn_1"]
#: Steps that write into ``geometry/`` (or ``geocalib/``) of whatever tree they are pointed
#: at. A variant farm links both back to the real clip, so under ``--variant`` these would
#: rewrite the shared inputs every run depends on.
GEOMETRY_STEPS = ("fuse", "scale", "gravity")
FARM_LINKS = ("sam3", "vggt", "sapiens", "sam3d", "geometry", "geocalib")
#: Contents of ``geometry/`` a synthetic-scene farm links back to the real tree; its
#: ``scene.npz`` is the one file the variant owns. ``prefit.npz`` is optional (only clips
#: whose scale stage pre-fit the body have one).
GEOMETRY_LINKS = ("transform.npz", "scale.npz", "human.npz", "prefit.npz")


def _load_datasets() -> tuple[dict[str, dict], dict[str, dict]]:
    """``configs/datasets.toml``, every data path made absolute under ``DATA_ROOT``."""
    with open(REPO_ROOT / "configs" / "datasets.toml", "rb") as handle:
        config = tomllib.load(handle)
    for dataset in config["datasets"].values():
        for key in ("root", "out", "out_multi"):
            if key in dataset:
                dataset[key] = DATA_ROOT / dataset[key]
    return config["datasets"], config.get("variants", {})


DATASETS, VARIANTS = _load_datasets()
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


def _farm_link(link: Path, target: Path, *, dry_run: bool = False) -> None:
    """Point ``link`` at the relative ``target``; a link already pointing there is kept.

    Anything else in the way is refused, never re-pointed: a farm laid out by other code
    (e.g. a variant linking another variant's geometry) would otherwise be rewired under
    the results already computed through it.
    """
    if link.is_symlink() and link.resolve() == (link.parent / target).resolve():
        return
    if link.is_symlink() or link.exists():
        current = link.readlink() if link.is_symlink() else "a real file or directory"
        raise SystemExit(f"[run_dataset] {link} already points to {current}, not {target}; "
                         "move it aside to rebuild this farm")
    if not dry_run:
        link.symlink_to(target)


def build_farm(clip_dir: Path, variant: str, *, synthetic_scene: bool = False,
               inherit: tuple[str, ...] = (), dry_run: bool = False) -> Path:
    """Create (or refresh) the clip's symlink farm; return the path to point the stages at.

    :param synthetic_scene: give the farm a real ``geometry/`` directory — the variant's
        own ``scene.npz`` beside symlinks to the real tree's other geometry artifacts —
        instead of linking the whole directory.
    :param inherit: files of the normal run's ``human_optim/`` linked into the variant
        directory instead of being recomputed.
    :param dry_run: only check that the existing links agree; write nothing.
    """
    out_dir, farm = clip_dir / f"human_optim_{variant}", clip_dir / f"_{variant}"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        farm.mkdir(exist_ok=True)
    for name in inherit:
        _farm_link(out_dir / name, Path("..") / "human_optim" / name, dry_run=dry_run)
    for name in (*FARM_LINKS, "human_optim"):
        if synthetic_scene and name == "geometry":
            continue
        _farm_link(farm / name, Path("..") / (out_dir.name if name == "human_optim" else name),
                   dry_run=dry_run)
    if synthetic_scene:
        geometry = farm / "geometry"
        if geometry.is_symlink():
            raise SystemExit(f"[run_dataset] {geometry} is a link to {geometry.readlink()}, but "
                             f"--variant {variant} owns its geometry/; move it aside to rebuild")
        if not dry_run:
            geometry.mkdir(exist_ok=True)
        for name in GEOMETRY_LINKS:
            if not (clip_dir / "geometry" / name).exists():
                continue
            _farm_link(geometry / name, Path("..") / ".." / "geometry" / name, dry_run=dry_run)
    return farm


def build_view_farm(out_root: Path, variant: str, *, dry_run: bool = False) -> Path:
    """``out/_view_<variant>/<clip>`` → each clip's ``_<variant>`` farm, for the viewer.

    ``scripts/view_3d.py`` wants a root of clip-named directories and finds each clip's
    footage through ``_batch_manifest.jsonl`` beside them, so the farms (all named
    ``_<variant>``) get a clip-named alias here and the manifest is linked in.
    ``dry_run`` prints each link instead of writing it.
    """
    view = out_root / f"_view_{variant}"
    if not dry_run:
        view.mkdir(exist_ok=True)
    links = [(view / farm.parent.name, Path("..") / farm.parent.name / farm.name)
             for farm in sorted(out_root.glob(f"*/_{variant}"))]
    for link, target in [*links, (view / "_batch_manifest.jsonl", Path("..") / "_batch_manifest.jsonl")]:
        if dry_run:
            print(f"[run_dataset] would link {link} -> {target}")
        _farm_link(link, target, dry_run=dry_run)
    return view


def _clip_command(step: str, videos: list[Path], clip_dir: Path, pipeline_config: Path,
                  human_config: Path, *, variant: str | None = None,
                  boards: bool = False) -> list[str]:
    """Argv for one per-clip stage (all paths absolute).

    :param videos: one video per camera view, in the ``camera.npz`` view order; only fuse
        takes more than the first.
    :param boards: measure the contacts on the force boards instead of estimating them.
    """
    common = ["--source", str(videos[0]), "--predictions", str(clip_dir)]
    if step in SUPPORT_STEPS:
        bodies = SUPPORT_STEPS[step]
        return [sys.executable, "-m", "bvr.stages.build_support_scene",
                "--predictions", str(clip_dir), "--config", str(human_config),
                *(["--bodies", bodies] if bodies else [])]
    if step == "overlay":
        # The renderer is pointed at the run directory, not the farm: it reads the clip's
        # real video and geometry off the tree and draws the run it is given.
        run_dir = clip_dir.parent / f"human_optim_{variant}" if variant else clip_dir / "human_optim"
        return [sys.executable, "-m", "datasets.peter.render_kindyn_overlays", str(run_dir)]
    if step == "fuse":
        return [sys.executable, "-m", "bvr.stages.fuse_geometry",
                "--source", ",".join(str(v) for v in videos),
                "--predictions", str(clip_dir), "--config", str(pipeline_config)]
    if step == "scale":
        return [sys.executable, "-m", "bvr.stages.estimate_scale",
                "--predictions", str(clip_dir), "--config", str(pipeline_config)]
    if step == "gravity":
        return [sys.executable, "-m", "bvr.stages.estimate_gravity_geocalib", *common,
                "--config", str(pipeline_config)]
    if step in {"contacts_1", "contacts_2"}:
        bodies = "kinematic_1" if step == "contacts_1" else "kinematic_2"
        if boards:
            return [sys.executable, "-m", "bvr.stages.measure_contacts_boards",
                    "--predictions", str(clip_dir), "--config", str(human_config),
                    "--bodies", bodies]
        return [sys.executable, "-m", "bvr.stages.estimate_contacts", *common,
                "--bodies", bodies, "--config", str(human_config)]
    stages_arg = {"kinematics_1": "root,kinematics_1",
                  "kinematics_2": "kinematics_2",
                  "kindyn_1": "kindyn_1"}[step]
    return [sys.executable, "-m", "bvr.stages.human_optimization", *common,
            "--stages", stages_arg, "--config", str(human_config)]


def _run(command: list[str], dry_run: bool = False) -> int:
    print(f"\n=== {' '.join(command[1:])} ===", flush=True)
    return 0 if dry_run else subprocess.run(command).returncode


def _manifest_name(clips: list[Path]) -> str:
    """Manifest filename for this exact clip selection.

    Named after the selection, not fixed, so two runs over different clips of the same
    tree — the normal way to use several GPUs — cannot overwrite each other's manifest
    and start processing each other's clips.
    """
    tag = hashlib.sha1(",".join(c.name for c in clips).encode()).hexdigest()[:8]
    return f"_clips_{tag}.json"


def _parse_shard(text: str) -> tuple[int, int]:
    """``"2/3"`` → ``(2, 3)``: this shard's 1-based index and the number of shards."""
    index, _, total = text.partition("/")
    shard, n_shards = int(index), int(total or 1)
    if not 1 <= shard <= n_shards:
        raise SystemExit(f"[run_dataset] bad --shard {text!r}: need 1 <= i <= n in 'i/n'")
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


def _reuse_masks(single_root: Path, out_root: Path, clips: list[Path]) -> None:
    """Copy a finished single-view run's SAM 3 masks into this run's tree.

    Copied, not linked. The tracker writes into ``sam3/`` whenever it does not skip an
    item, and it follows symlinks — so a link here would let a re-track overwrite the
    finished single-view run in place. The masks are ~2 MB a clip. Clips the single-view
    run hasn't tracked yet are left out and tracked here.
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


def _track_extra_views(out_root: Path, clips: list[Path], views: list[str], config: Path,
                       *, skip_existing: bool, dry_run: bool) -> None:
    """Run SAM 3 over every view after the first, one manifest sweep per view.

    Each view is tracked into its own ``_<view>/`` tree, so the tracker keeps its usual
    ``sam3/`` layout, and is then linked into the clip where fuse expects that view's masks.
    """
    for v, view in enumerate(views[1:], start=1):
        view_root = out_root / f"_{view}"
        if skip_existing and all((view_root / c.name / "sam3" / "bboxes.npz").exists() for c in clips):
            print(f"[run_dataset] sam3 {view}: every clip has sam3/bboxes.npz, skipping the sweep")
        else:
            manifest = view_root / _manifest_name(clips)
            if not dry_run:
                view_root.mkdir(parents=True, exist_ok=True)
                manifest.write_text(json.dumps([{
                    "source": str(clip / f"{view}.mp4"),
                    "out_dir": str(view_root / clip.name),
                } for clip in clips], indent=2))
            started = time.monotonic()
            rc = _run([sys.executable, "-m", "bvr.stages.track_sam3",
                       "--source", str(manifest), "--save-dir", str(view_root),
                       "--config", str(config)], dry_run)
            print(f"[run_dataset] sam3 {view}: rc={rc} in {(time.monotonic() - started) / 60:.1f} min")
            if rc != 0:
                continue   # a dangling link would look like masks that are simply empty
        if not dry_run:
            for clip in clips:
                _link(_sam3_dir(out_root / clip.name, views, v), view_root / clip.name / "sam3")


def _report_people(out_root: Path, clips: list[Path], views: list[str]) -> None:
    """Print how many people SAM 3 tracked per clip and view (expected: exactly one)."""
    import numpy as np

    print("\n[run_dataset] tracked people per clip:")
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
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS),
                        help="a [datasets.*] table of configs/datasets.toml")
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
                             f"{'/'.join(VARIANT_CHAINS)} run their own chain (configs/datasets.toml)")
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
    dataset = DATASETS[args.dataset]
    recipe = VARIANTS.get(args.variant, {}) if args.variant else {}

    rig_views = dataset.get("views")
    if args.views and not rig_views:
        raise SystemExit(f"[run_dataset] --views: {args.dataset} has one video per clip, no camera views")
    views = ([v.strip() for v in args.views.split(",") if v.strip()] if args.views
             else rig_views[:1] if rig_views else None)
    if views and len(set(views)) != len(views):
        raise SystemExit(f"[run_dataset] repeated view in {views}")
    # The single-view tree belongs to the rig's first camera alone. Any other view list —
    # a different camera, or several — goes to the multi tree, so a finished single-view
    # run can never be written over by a differently-framed one.
    multi = views is not None and views != rig_views[:1]
    out_root = dataset["out_multi"] if multi else dataset["out"]
    if args.view_farm:
        if not args.variant:
            raise SystemExit("[run_dataset] --view-farm needs --variant")
        print(f"[run_dataset] viewer root: {build_view_farm(out_root, args.variant, dry_run=args.dry_run)}"
              f"{'  (dry run)' if args.dry_run else ''}")
        return 0
    pipeline_config = REPO_ROOT / dataset["pipeline_config"]
    human_config = REPO_ROOT / (recipe.get("config", f"configs/human_optimization_{args.variant}.toml")
                                if args.variant else dataset["human_config"])

    root = dataset["root"]
    if not root.is_dir():
        raise SystemExit(f"[run_dataset] no such dataset root: {root}")
    if views:
        video_names = [f"{view}.mp4" for view in views]
        clips = sorted(d for d in root.iterdir() if all((d / name).is_file() for name in video_names))
    else:
        video_names = [Path(dataset["clips"]).name]
        clips = sorted(video.parent for video in root.glob(dataset["clips"]))
    if "requires" in recipe:
        clips = [c for c in clips if (out_root / c.name / recipe["requires"]).exists()]
    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = [c for c in clips if c.name in wanted]
        if missing := wanted - {c.name for c in clips}:
            raise SystemExit(f"[run_dataset] unknown clip(s): {sorted(missing)}")
    if not clips:
        raise SystemExit(f"[run_dataset] no clip under {root} has {', '.join(video_names)}"
                         + (f" and {out_root}/<clip>/{recipe['requires']}" if "requires" in recipe else ""))
    shard, n_shards = _parse_shard(args.shard)
    clips = clips[shard - 1::n_shards]
    if not clips:
        raise SystemExit(f"[run_dataset] shard {shard}/{n_shards} is empty")

    chain = list(dataset["multi_steps"] if views and len(views) > 1 else dataset["steps"])
    if args.variant:
        chain = list(VARIANT_CHAINS.get(args.variant, VARIANT_STEPS))
    steps = [s.strip() for s in args.steps.split(",")] if args.steps else list(chain)
    if unknown := [s for s in steps if s not in ALL_STEPS]:
        raise SystemExit(f"[run_dataset] unknown step(s) {unknown}; available: {ALL_STEPS}")
    if args.variant and any(s in MANIFEST_STEPS for s in steps):
        raise SystemExit("[run_dataset] --variant re-runs the human chain only; the model "
                         f"stages ({', '.join(MANIFEST_STEPS)}) are shared with the baseline run")
    if args.variant and any(s in GEOMETRY_STEPS for s in steps):
        raise SystemExit(f"[run_dataset] {', '.join(GEOMETRY_STEPS)} cannot run under "
                         "--variant: the farm links geometry/ and geocalib/ back to the real "
                         "clip, so they would rewrite the inputs every run shares")
    if any(s in SUPPORT_STEPS for s in steps) and not owns_scene(args.variant):
        raise SystemExit(
            f"[run_dataset] {', '.join(SUPPORT_STEPS)} need a variant whose farm owns its "
            f"geometry/ ({', '.join(v for v in VARIANT_CHAINS if owns_scene(v))}); under "
            f"--variant {args.variant!r} that directory is a symlink and the synthetic scene "
            "would overwrite the clip's real fused geometry/scene.npz")
    # A chain is an order, not a set: running contacts_1 before the kinematics_1 it reads, or
    # kindyn_1 twice, silently scores the wrong body.
    places = [chain.index(s) for s in steps if s in chain]
    if len(places) != len(steps) or any(a >= b for a, b in zip(places, places[1:])):
        raise SystemExit(f"[run_dataset] --steps must be a subset of this run's chain, in its "
                         f"order {chain}; got {steps}")
    for path in (human_config, pipeline_config):
        if not path.exists():
            raise SystemExit(f"[run_dataset] config not found: {path}")

    sweeps = [s for s in steps if s in MANIFEST_STEPS]
    # --skip-existing: a sweep every selected clip has finished is not launched at all.
    finished = {s for s in sweeps if args.skip_existing
                and all((out_root / c.name / MANIFEST_STEPS[s][1]).exists() for c in clips)}
    manifest = out_root / _manifest_name(clips)
    if set(sweeps) - finished and not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        # ``sources`` lists every camera view, which is how the vggt stage puts them all in
        # one forward; ``bboxes`` / ``camera`` are written by earlier steps of this same run.
        manifest.write_text(json.dumps([{
            "source": str(clip / video_names[0]),
            **({"sources": [str(clip / name) for name in video_names]} if views else {}),
            "out_dir": str(out_root / clip.name),
            "bboxes": str(out_root / clip.name / "sam3" / "bboxes.npz"),
            "camera": str(out_root / clip.name / "vggt" / "camera.npz"),
        } for clip in clips], indent=2))
    print(f"[run_dataset] {len(clips)} clip(s){f' x {len(views)} view(s)' if views else ''} → {out_root}"
          f"{'' if n_shards == 1 else f' (shard {shard}/{n_shards})'}\n"
          f"[run_dataset] steps: {steps}{'  (dry run)' if args.dry_run else ''}")

    for step in sweeps:
        module, output = MANIFEST_STEPS[step]
        if step in finished:
            print(f"[run_dataset] {step}: every clip has {output}, skipping the sweep")
        else:
            if step == "sam3" and views and views[0] == rig_views[0] and not args.dry_run:
                _reuse_masks(dataset["out"], out_root, clips)   # the first camera is never tracked twice
            started = time.monotonic()
            rc = _run([sys.executable, "-m", f"bvr.stages.{module}",
                       "--source", str(manifest), "--save-dir", str(out_root),
                       "--config", str(pipeline_config)], args.dry_run)
            print(f"[run_dataset] {step}: rc={rc} in {(time.monotonic() - started) / 60:.1f} min")
        if step == "sam3" and views:
            _track_extra_views(out_root, clips, views, pipeline_config,
                               skip_existing=args.skip_existing, dry_run=args.dry_run)
            if not args.dry_run:
                _report_people(out_root, clips, views)

    clip_steps = [s for s in steps if s not in MANIFEST_STEPS]
    failures: list[tuple[str, str]] = []
    for index, clip in enumerate(clips, 1):
        if not clip_steps:
            break
        videos = [clip / name for name in video_names]
        clip_dir = out_root / clip.name
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
            raise SystemExit(f"[run_dataset] {clip.name}: {clip_dir}/geometry/scene.npz is "
                             f"missing — run --variant {args.variant} --steps "
                             f"{next(iter(SUPPORT_STEPS))} first")
        print(f"\n########## [{index}/{len(clips)}] {clip.name} ##########", flush=True)
        for step in clip_steps:
            done = CLIP_OUTPUT.get(step)
            if done and (clip_dir / done).exists():
                print(f"[run_dataset] {clip.name}/{step}: exists, skipping")
                continue
            command = _clip_command(step, videos, clip_dir, pipeline_config, human_config,
                                    variant=args.variant, boards=recipe.get("contacts") == "boards")
            started = time.monotonic()
            rc = _run(command, args.dry_run)
            ok = args.dry_run or (rc == 0 and (done is None or (clip_dir / done).exists()))
            print(f"[run_dataset] {clip.name}/{step}: {'ok' if ok else 'FAILED'} (rc={rc}) "
                  f"in {(time.monotonic() - started) / 60:.1f} min")
            if not ok:
                failures.append((clip.name, step))
                break   # later steps of this clip depend on this one

    print(f"\n[run_dataset] done. {len(clips) - len({c for c, _ in failures})}/{len(clips)} "
          f"clip(s) completed the per-clip chain.")
    for name, step in failures:
        print(f"  FAILED {name} at {step}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
