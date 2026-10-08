"""Run the full reconstruction chain over the imported OpenCap LabValidation clips.

The clips live under ``data/LabValidation_withVideos/processed/`` (next to the raw
dataset, outside the repo): ``processed/<clip>/video.mp4`` →
``processed/out/<clip>/``. Single static iPhone, one subject per clip, 720x1280
portrait at 60 fps (see ``scripts/import_opencap.py`` for the import and for the
force-plate / marker ground truth that ships beside each video).

The chain, in order::

    sam3 → vggt → sapiens → sam3d → smplx → fuse → scale → gravity
         → kinematics_1 → contacts_1 → kinematics_2 → contacts_2 → kindyn_1

The five model stages run as ONE manifest sweep each (the model loads once and
sweeps every clip); the rest run one subprocess per clip, clip by clip. Every
step is resumable: a step whose output already exists is skipped. A clip that
fails is reported and skipped — one bad clip never kills the batch.

``gravity`` (GeoCalib) is in the chain because ``kindyn_1`` solves against world
gravity and the lab camera's tilt is unknown — the world-down fallback would bake
that tilt into every force direction. It is also the cross-check against the
dataset's own calibration: the gravity stage's ``gravity_world`` should match
``R_world_from_mocap @ [0, -1, 0]``.

``--shard i/n`` takes every n-th clip, so n runs on n GPUs fill one output tree
between them without hand-writing ``--clips`` lists. The manifest is named after
the clip selection, so the shards never read each other's.

Knobs live in the TOMLs (``configs/pipeline_opencap.toml`` for the model +
geometry stages, ``configs/human_optimization_peter.toml`` for the human stages —
the general config restricted to hand/foot contact frames, which is what the force
plates measure); this CLI only takes targets.

Usage::

    CUDA_VISIBLE_DEVICES=5 uv run python scripts/run_opencap.py
    CUDA_VISIBLE_DEVICES=5 uv run python scripts/run_opencap.py \
        --clips subject2_DJ1_cam1 --steps fuse,scale,kinematics_1
    # three GPUs over the 12 clips: one process each, disjoint thirds
    CUDA_VISIBLE_DEVICES=5 uv run python scripts/run_opencap.py --shard 1/3
    CUDA_VISIBLE_DEVICES=6 uv run python scripts/run_opencap.py --shard 2/3
    CUDA_VISIBLE_DEVICES=7 uv run python scripts/run_opencap.py --shard 3/3
    # the human chain again with another contact recipe, beside the baseline run
    CUDA_VISIBLE_DEVICES=5 uv run python scripts/run_opencap.py --variant labvalidation
    # …and the floorplane variant, whose chain also builds its own scene cloud
    CUDA_VISIBLE_DEVICES=5 uv run python scripts/run_opencap.py --variant floorplane --shard 1/3
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
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.paths import LABVALIDATION  # noqa: E402

_OPENCAP = LABVALIDATION / "processed"

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
#: Synthetic-support-scene steps → the body their plateaus are read off (None = floor only).
#: Both write the same ``geometry/scene.npz``, so an existence check cannot tell them apart:
#: they are never resume-skipped, which costs seconds and is safe because they are idempotent.
SUPPORT_STEPS: dict[str, str | None] = {
    "support_scene_floor":    None,
    "support_scene_plateaus": "kinematic_1",
}
#: The baseline chain. The support-scene steps are deliberately NOT in it: they rewrite
#: geometry/scene.npz, which outside a variant farm is the real fused cloud.
BASE_STEPS = [*MANIFEST_STEPS, *CLIP_OUTPUT]
ALL_STEPS = [*BASE_STEPS, *SUPPORT_STEPS]

#: ``--variant <name>``: re-run the human chain into ``<clip>/human_optim_<name>/`` with
#: ``configs/human_optimization_<name>.toml``, leaving the baseline ``human_optim/`` intact
#: for comparison. kinematics_1 is recomputed too — a variant that changes the contact-frame
#: allowlist changes its auto-contacts. The stock stages are pointed at a symlink farm
#: ``<clip>/_<name>/`` whose ``human_optim`` IS the variant directory and whose inputs link
#: back to the real tree (the pattern of ``scripts/run_gt_contact.py``).
VARIANT_STEPS = ["kinematics_1", "contacts_1", "kinematics_2", "contacts_2", "kindyn_1"]
#: Variants whose chain is not the plain :data:`VARIANT_STEPS`. ``floorplane`` replaces the
#: fused cloud with a synthetic support scene, so it builds the floor before kinematics_1 can
#: attract to it and rebuilds it with the plateau patches — read off that very body — before
#: contacts_1 reads contact from it. Every other variant is untouched by all of this.
VARIANT_CHAINS: dict[str, list[str]] = {
    "floorplane": ["support_scene_floor", "kinematics_1", "support_scene_plateaus",
                   "contacts_1", "kinematics_2", "contacts_2", "kindyn_1"],
}
#: Steps that write into ``geometry/`` (or ``geocalib/``) of whatever tree they are pointed
#: at. A variant farm links both back to the real clip, so under ``--variant`` these would
#: rewrite the shared inputs every run depends on.
GEOMETRY_STEPS = ("fuse", "scale", "gravity")
FARM_LINKS = ("sam3", "vggt", "sapiens", "sam3d", "geometry", "geocalib")
#: Contents of ``geometry/`` a synthetic-scene farm links back to the real tree; its
#: ``scene.npz`` is the one file the variant owns. ``prefit.npz`` is optional (only clips
#: whose scale stage pre-fit the body have one).
GEOMETRY_LINKS = ("transform.npz", "scale.npz", "human.npz", "prefit.npz")


def owns_scene(variant: str | None) -> bool:
    """Does this variant's farm own a real ``geometry/`` with its own ``scene.npz``?

    True exactly for the variants whose chain builds a synthetic support scene. Everywhere
    else ``geometry`` is one symlink to the clip's real tree, so a support step pointed at
    that farm would overwrite the fused cloud the baseline run and every other variant read.
    """
    return any(step in SUPPORT_STEPS for step in VARIANT_CHAINS.get(variant or "", []))


def build_farm(clip_dir: Path, variant: str, *, synthetic_scene: bool = False) -> Path:
    """Create (or refresh) the clip's symlink farm; return the path to point the stages at.

    :param synthetic_scene: give the farm a real ``geometry/`` directory — the variant's
        own ``scene.npz`` beside symlinks to the real tree's other geometry artifacts —
        instead of linking the whole directory.
    """
    out_dir, farm = clip_dir / f"human_optim_{variant}", clip_dir / f"_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    farm.mkdir(exist_ok=True)
    for name in (*FARM_LINKS, "human_optim"):
        if synthetic_scene and name == "geometry":
            continue
        (farm / name).unlink(missing_ok=True)
        (farm / name).symlink_to(Path("..") / (out_dir.name if name == "human_optim" else name))
    if synthetic_scene:
        geometry = farm / "geometry"
        if geometry.is_symlink():
            geometry.unlink()
        geometry.mkdir(exist_ok=True)
        for name in GEOMETRY_LINKS:
            if not (clip_dir / "geometry" / name).exists():
                continue
            (geometry / name).unlink(missing_ok=True)
            (geometry / name).symlink_to(Path("..") / ".." / "geometry" / name)
    return farm


def build_view_farm(out_root: Path, variant: str) -> Path:
    """``out/_view_<variant>/<clip>`` → each clip's ``_<variant>`` farm, for the viewer.

    ``scripts/view_3d.py`` wants a root of clip-named directories and finds each clip's
    footage through ``_batch_manifest.jsonl`` beside them, so the farms (all named
    ``_<variant>``) get a clip-named alias here and the manifest is linked in.
    """
    view = out_root / f"_view_{variant}"
    view.mkdir(exist_ok=True)
    for farm in sorted(out_root.glob(f"*/_{variant}")):
        link = view / farm.parent.name
        link.unlink(missing_ok=True)
        link.symlink_to(Path("..") / farm.parent.name / farm.name)
    (view / "_batch_manifest.jsonl").unlink(missing_ok=True)
    (view / "_batch_manifest.jsonl").symlink_to(Path("..") / "_batch_manifest.jsonl")
    return view


def _clip_command(step: str, video: Path, clip_dir: Path,
                  pipeline_config: Path, human_config: Path) -> list[str]:
    """Argv for one per-clip stage (all paths absolute)."""
    stage = lambda name: str(_STAGES_DIR / name)  # noqa: E731
    common = ["--source", str(video), "--predictions", str(clip_dir)]
    if step in SUPPORT_STEPS:
        bodies = SUPPORT_STEPS[step]
        return [sys.executable, stage("build_support_scene.py"),
                "--predictions", str(clip_dir), "--config", str(human_config),
                *(["--bodies", bodies] if bodies else [])]
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


def _parse_shard(text: str) -> tuple[int, int]:
    """``"2/3"`` → ``(2, 3)``: this shard's 1-based index and the number of shards."""
    index, _, total = text.partition("/")
    shard, n_shards = int(index), int(total or 1)
    if not 1 <= shard <= n_shards:
        raise SystemExit(f"[run_opencap] bad --shard {text!r}: need 1 <= i <= n in 'i/n'")
    return shard, n_shards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", default=None,
                        help="comma-separated clip names (default: all)")
    parser.add_argument("--steps", default=None,
                        help=f"comma-separated subset of {','.join(ALL_STEPS)} (default: all)")
    parser.add_argument("--shard", default="1/1",
                        help="'i/n': process every n-th clip, so n runs on n GPUs share "
                             "one output tree (default: 1/1, all clips)")
    parser.add_argument("--config", type=Path,
                        default=_REPO_ROOT / "configs" / "pipeline_opencap.toml",
                        help="model + geometry stage TOML")
    parser.add_argument("--human-config", type=Path, default=None,
                        help="human-optimization + contacts TOML (default: the peter config, "
                             "or human_optimization_<variant>.toml with --variant)")
    parser.add_argument("--variant", default=None,
                        help=f"re-run {'/'.join(VARIANT_STEPS)} into <clip>/human_optim_<variant>/ "
                             "with that variant's config, leaving human_optim/ intact; "
                             f"{'/'.join(VARIANT_CHAINS)} run their own chain instead")
    parser.add_argument("--view-farm", action="store_true",
                        help="only build out/_view_<variant>/ (a clip-named alias of every "
                             "clip's variant farm) for scripts/view_3d.py, then exit")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.view_farm:
        if not args.variant:
            raise SystemExit("[run_opencap] --view-farm needs --variant")
        print(f"[run_opencap] viewer root: {build_view_farm(_OPENCAP / 'out', args.variant)}")
        return 0
    human_config = args.human_config or _REPO_ROOT / "configs" / (
        f"human_optimization_{args.variant}.toml" if args.variant else "human_optimization_peter.toml")

    clips = sorted(d for d in _OPENCAP.iterdir()
                   if d.is_dir() and (d / "video.mp4").is_file())
    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = [c for c in clips if c.name in wanted]
        missing = wanted - {c.name for c in clips}
        if missing:
            raise SystemExit(f"[run_opencap] unknown clip(s): {sorted(missing)}")
    if not clips:
        raise SystemExit(f"[run_opencap] no clips under {_OPENCAP}")
    shard, n_shards = _parse_shard(args.shard)
    clips = clips[shard - 1::n_shards]
    if not clips:
        raise SystemExit(f"[run_opencap] shard {shard}/{n_shards} is empty")

    default_steps = (VARIANT_CHAINS.get(args.variant, VARIANT_STEPS)
                     if args.variant else BASE_STEPS)
    steps = [s.strip() for s in args.steps.split(",")] if args.steps else list(default_steps)
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise SystemExit(f"[run_opencap] unknown step(s) {unknown}; available: {ALL_STEPS}")
    if args.variant and any(s in MANIFEST_STEPS for s in steps):
        raise SystemExit("[run_opencap] --variant re-runs the human chain only; the model "
                         f"stages ({', '.join(MANIFEST_STEPS)}) are shared with the baseline run")
    if args.variant and any(s in GEOMETRY_STEPS for s in steps):
        raise SystemExit(f"[run_opencap] {', '.join(GEOMETRY_STEPS)} cannot run under "
                         "--variant: the farm links geometry/ and geocalib/ back to the real "
                         "clip, so they would rewrite the inputs every run shares")
    if any(s in SUPPORT_STEPS for s in steps) and not owns_scene(args.variant):
        raise SystemExit(
            f"[run_opencap] {', '.join(SUPPORT_STEPS)} need a variant whose farm owns its "
            f"geometry/ ({', '.join(v for v in VARIANT_CHAINS if owns_scene(v))}); under "
            f"--variant {args.variant!r} that directory is a symlink and the synthetic scene "
            "would overwrite the clip's real fused geometry/scene.npz")
    # A chain is an order, not a set: running contacts_1 before the kinematics_1 it reads, or
    # kindyn_1 twice, silently scores the wrong body.
    chain = list(default_steps)
    places = [chain.index(s) for s in steps if s in chain]
    if len(places) != len(steps) or any(a >= b for a, b in zip(places, places[1:])):
        raise SystemExit(f"[run_opencap] --steps must be a subset of this run's chain, in its "
                         f"order {chain}; got {steps}")

    out_root = _OPENCAP / "out"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / _manifest_name(clips)
    manifest.write_text(json.dumps([{
        "source": str(clip / "video.mp4"),
        "out_dir": str(out_root / clip.name),
        "bboxes": str(out_root / clip.name / "sam3" / "bboxes.npz"),
        "camera": str(out_root / clip.name / "vggt" / "camera.npz"),
    } for clip in clips], indent=2))
    print(f"[run_opencap] {len(clips)} clip(s) → {out_root}"
          f"{'' if n_shards == 1 else f' (shard {shard}/{n_shards})'}\n"
          f"[run_opencap] steps: {steps}")

    for step in [s for s in steps if s in MANIFEST_STEPS]:
        script, _ = MANIFEST_STEPS[step]
        started = time.monotonic()
        rc = _run([sys.executable, str(_STAGES_DIR / script),
                   "--source", str(manifest), "--save-dir", str(out_root),
                   "--config", str(args.config)])
        print(f"[run_opencap] {step}: rc={rc} in {(time.monotonic() - started) / 60:.1f} min")

    clip_steps = [s for s in steps if s not in MANIFEST_STEPS]
    failures: list[tuple[str, str]] = []
    for index, clip in enumerate(clips, 1):
        if not clip_steps:
            break
        # With --variant the stages are pointed at the farm, so every path below —
        # including the resume check — resolves through it into human_optim_<variant>/.
        clip_dir = out_root / clip.name
        if args.variant:
            clip_dir = build_farm(clip_dir, args.variant, synthetic_scene=owns_scene(args.variant))
        # A synthetic-scene variant that has not built its scene yet would run the whole human
        # chain with the scene terms silently off (tools.human_optim.stages._load_scene prints
        # a note and carries on), which looks like a finished variant and is not one.
        if (owns_scene(args.variant) and clip_steps[0] not in SUPPORT_STEPS
                and not (clip_dir / "geometry" / "scene.npz").exists()):
            raise SystemExit(f"[run_opencap] {clip.name}: {clip_dir}/geometry/scene.npz is "
                             f"missing — run --variant {args.variant} --steps "
                             f"{next(iter(SUPPORT_STEPS))} first")
        print(f"\n########## [{index}/{len(clips)}] {clip.name} ##########", flush=True)
        for step in clip_steps:
            done = CLIP_OUTPUT.get(step)
            if done and (clip_dir / done).exists():
                print(f"[run_opencap] {clip.name}/{step}: exists, skipping")
                continue
            command = _clip_command(step, clip / "video.mp4", clip_dir,
                                    args.config, human_config)
            started = time.monotonic()
            rc = _run(command)
            print(f"[run_opencap] {clip.name}/{step}: rc={rc} "
                  f"in {(time.monotonic() - started) / 60:.1f} min")
            if rc != 0:
                failures.append((clip.name, step))
                break   # later steps of this clip depend on this one

    if failures:
        print(f"\n[run_opencap] {len(failures)} clip(s) failed:")
        for name, step in failures:
            print(f"  {name}: failed at {step}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
