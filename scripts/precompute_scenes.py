#!/usr/bin/env python3
"""DB-driven VGGT-Omega / Sapiens / body precompute over ClimbingVideos scenes.

The second wave of the ``./data`` → ``../data/ClimbingVideos`` scene-DB migration
(``track_sam3_scenes.py`` did SAM 3 first). It runs one of the per-frame /
per-clip precompute stages across the curated target set on several GPUs, reusing
the stage CLIs unchanged via ``precompute_runner``'s per-item JSON manifest.

Stages (``--stage``):

    vggt     estimate_camera_vggt.py    multi-view cameras + depth (one forward/clip)
    sapiens  estimate_sapiens.py        top-down pose; reads the SAM 3 bboxes
    sam3d    estimate_body_sam3d.py     per-person MHR body fit; reads the SAM 3
                                        bboxes (+ masks) and the VGGT camera
    smplx    convert_mhr_to_smplx.py    MHR -> SMPL-X params; reads the sam3d
                                        params.npz, writes smplx_params.npz beside it

``sapiens`` and ``sam3d`` have input dependencies passed per-item through the
manifest: ``sapiens`` reads ``features/sam3/<...>/bboxes.npz``; ``sam3d`` reads
those same bboxes (+ their ``<obj>/`` mask dirs) and ``features/vggt/<...>/
camera.npz`` — **intrinsics only**, which are scale-invariant, so VGGT's
up-to-scale geometry is irrelevant here.
``smplx`` reads only the ``sam3d`` ``params.npz`` (it writes ``smplx_params.npz``
into that same sam3d feature dir). ``vggt`` takes only the clip.
Output mirrors the SAM 3 / yolo / tracking layout (**stage-first**)::

    ../data/ClimbingVideos/features/<stage>/<aa>/<bb>/<scene_id>/...

(``<aa>/<bb> = scene_id[:2]/scene_id[2:4]``). ``vggt`` writes ``camera.npz`` + a
per-frame ``depth/``; ``sapiens`` writes ``pose.npz`` (+
``overlay/``); ``sam3d`` writes ``params.npz`` (stacked MHR params for every
tracked person); ``smplx`` writes ``smplx_params.npz`` **into the sam3d dir**
(beside ``params.npz``). Resumable with ``--skip-existing`` (skips scenes whose
stage output already exists).

The scene list is split across the configured GPUs (default 1,2,3); one stage
worker runs per GPU over a disjoint stride shard, seeded from a per-shard JSON
manifest. This script only does scene selection, manifest building, and fan-out.

    # full target set, one stage, resumable, GPUs 1,2,3:
    uv run python scripts/precompute_scenes.py --stage vggt  --skip-existing
    uv run python scripts/precompute_scenes.py --stage sapiens --skip-existing
    uv run python scripts/precompute_scenes.py --stage sam3d --skip-existing
    uv run python scripts/precompute_scenes.py --stage smplx --skip-existing

    # smoke test 2 scenes on one GPU, print the plan first:
    uv run python scripts/precompute_scenes.py --stage sapiens --sample 2 --dry-run
    uv run python scripts/precompute_scenes.py --stage sapiens --sample 2 --devices 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import scene_db  # noqa: E402  (scripts/ on path)

_STAGES_DIR = _REPO_ROOT / "scripts" / "stages"


@dataclass(frozen=True)
class StageSpec:
    """Everything that differs between the vggt / sapiens / body stages."""

    name: str
    cli: Path
    out_check: str                       # output filename marking a scene done
    extra_args: list[str] = field(default_factory=list)
    needs_bboxes: bool = False           # pull the SAM 3 bboxes.npz into the manifest
    needs_camera: bool = False           # pull the VGGT camera.npz into the manifest
    has_overlay: bool = False            # stage supports --overlay/--no-overlay


STAGES: dict[str, StageSpec] = {
    "vggt": StageSpec(
        name="vggt",
        cli=_STAGES_DIR / "estimate_camera_vggt.py",
        out_check="camera.npz",
        extra_args=["--save-depth"],     # fuse_geometry needs the vggt depth tree
    ),
    "sapiens": StageSpec(
        name="sapiens",
        cli=_STAGES_DIR / "estimate_sapiens.py",
        out_check="pose.npz",
        extra_args=["--tasks", "pose"],
        needs_bboxes=True,
        has_overlay=True,
    ),
    "sam3d": StageSpec(
        name="sam3d",
        cli=_STAGES_DIR / "estimate_body_sam3d.py",
        out_check="params.npz",
        extra_args=["--inference-type", "full"],
        # bboxes.npz (+ its <obj>/ mask siblings) and the VGGT camera.npz — both
        # in their own feature trees, so both go through the manifest. Cameras
        # are VGGT-only (scale-invariant intrinsics).
        needs_bboxes=True,
        needs_camera=True,
        has_overlay=True,
    ),
    "smplx": StageSpec(
        # name="sam3d": the MHR->SMPL-X conversion reads params.npz from and writes
        # smplx_params.npz into the sam3d feature dir (smplx artifacts live beside
        # the body params, matching the single-clip sam3d/ layout). No extra inputs
        # -- it reads only params.npz (camera + MHR params are inside it).
        name="sam3d",
        cli=_STAGES_DIR / "convert_mhr_to_smplx.py",
        out_check="smplx_params.npz",
        has_overlay=True,
    ),
}


# ------------------------------ manifest + I/O -----------------------------


def build_manifest(paths: scene_db.ScenePaths, spec: StageSpec,
                   rows: list) -> list[dict]:
    """One ``{source, out_dir[, bboxes][, camera]}`` manifest entry per usable scene."""
    entries: list[dict] = []
    for row in rows:
        sid = row["scene_id"]
        entry = {
            "source": str(paths.clip_path(row)),
            "out_dir": str(paths.feature_dir(spec.name, sid)),
        }
        if spec.needs_bboxes:
            bb = paths.sam3_bboxes(sid)
            if not bb.is_file():
                print(f"  [skip] {sid}: no sam3 bboxes at {bb}")
                continue
            entry["bboxes"] = str(bb)
        if spec.needs_camera:
            cam = paths.vggt_camera(sid)
            if not cam.is_file():
                print(f"  [skip] {sid}: no vggt camera at {cam}")
                continue
            entry["camera"] = str(cam)
        entries.append(entry)
    return entries


def scene_done(paths: scene_db.ScenePaths, spec: StageSpec, scene_id: str) -> bool:
    return (paths.feature_dir(spec.name, scene_id) / spec.out_check).is_file()


def gather_overlays(paths: scene_db.ScenePaths, spec: StageSpec,
                    rows: list, dest: Path) -> int:
    """Symlink each scene's ``overlay/pose.mp4`` into ``dest`` as ``<scene>.mp4``."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for row in rows:
        sid = row["scene_id"]
        src = paths.feature_dir(spec.name, sid) / "overlay" / "pose.mp4"
        if not src.is_file():
            continue
        link = dest / f"{sid}.mp4"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src.resolve())
        n += 1
    return n


# ------------------------------ orchestration ------------------------------


def run_worker(paths: scene_db.ScenePaths, spec: StageSpec, rows: list, gpu: int,
               i: int, n: int, work_dir: Path,
               args: argparse.Namespace) -> subprocess.Popen | None:
    """Spawn one stage worker on ``gpu`` over shard ``i/n`` of ``rows``."""
    shard_rows = scene_db.stride_shard(rows, i, n)
    entries = build_manifest(paths, spec, shard_rows)
    if not entries:
        print(f"[shard {i}/{n} gpu={gpu}] no scenes — skipping")
        return None

    # Per-shard save-dir so concurrent workers don't clobber each other's
    # throwaway _batch_manifest.jsonl (per-scene outputs go to their own out_dir).
    shard_dir = work_dir / f"shard_{i}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = shard_dir / "manifest.json"
    manifest.write_text(json.dumps(entries, indent=2))

    cmd = [
        sys.executable, str(spec.cli),
        "--source", str(manifest),
        "--save-dir", str(shard_dir),   # only used for the batch-manifest log
        "--subdir", ".",                # data straight into features/<stage>/<...>/<scene>/
        "--device", "cuda",             # GPU pinned via CUDA_VISIBLE_DEVICES below
        *spec.extra_args,
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if spec.has_overlay and not args.overlay:
        cmd.append("--no-overlay")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    # Avoid caching-allocator fragmentation: long VGGT clips OOM'd with ~17 GB
    # reserved-but-unallocated while needing only ~3 GB more (true peak ~32 GB,
    # well under 48 GB). expandable_segments reclaims that. Harmless for the
    # lighter sapiens workers. Respect an env the caller already set.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[shard {i}/{n} gpu={gpu}] {len(entries)} scene(s) → {manifest.name}")
    return subprocess.Popen(cmd, env=env)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", required=True, choices=sorted(STAGES),
                   help="which precompute stage to run.")
    p.add_argument("--data-root", type=Path, default=scene_db.DEFAULT_DATA_ROOT,
                   help=f"ClimbingVideos data root (default: {scene_db.DEFAULT_DATA_ROOT}).")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--sample", type=int, default=None,
                     help="process a random subset of N target scenes (deterministic per --seed).")
    sel.add_argument("--limit", type=int, default=None,
                     help="process the first N target scenes (by scene_id).")
    sel.add_argument("--all", action="store_true",
                     help="process every target scene (the default).")
    sel.add_argument("--scenes", default=None,
                     help="comma-separated scene_ids to process (overrides --sample/--limit).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --sample (default: 0).")
    p.add_argument("--devices", default="1,2,3",
                   help="comma-separated GPU ids, one worker each (default: 1,2,3).")
    p.add_argument("--categories", default=",".join(map(str, scene_db.DEFAULT_CATEGORIES)),
                   help="comma-separated vlm_category filter (default: 1,2 = indoor boulder+rope).")
    p.add_argument("--rope-supported", type=int, default=scene_db.DEFAULT_ROPE_SUPPORTED,
                   help="vlm_rope_supported filter (default: 0; pass -1 to not constrain).")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip scenes whose stage output already exists.")
    p.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=False,
                   help="sapiens only: render + gather the pose overlay (default: OFF — "
                        "not needed for the DB precompute; pass --overlay for QA).")
    p.add_argument("--debug-overlay-dir", type=Path, default=None,
                   help="gather sapiens overlays here as <scene>.mp4 "
                        "(default: <features>/sapiens/_debug_overlays).")
    p.add_argument("--dry-run", action="store_true",
                   help="print the selected scenes + planned outputs and exit (no GPU).")
    args = p.parse_args(argv)

    spec = STAGES[args.stage]
    paths = scene_db.ScenePaths(args.data_root)
    if not paths.scenes_db.is_file():
        print(f"ERROR: scenes.db not found at {paths.scenes_db}", file=sys.stderr)
        return 1

    categories = scene_db.parse_int_csv(args.categories)
    rope = None if args.rope_supported < 0 else args.rope_supported

    with paths.connect() as conn:
        rows = scene_db.select_targets(conn, categories=categories, rope_supported=rope)
    rows = scene_db.resolve_rows(
        rows, scenes=args.scenes, sample=args.sample, limit=args.limit, seed=args.seed,
    )

    print(f"[precompute_scenes stage={args.stage}] target scenes: {len(rows)} "
          f"(categories={categories}, rope_supported={rope})")
    if not rows:
        return 0

    if args.dry_run:
        done = sum(scene_done(paths, spec, r["scene_id"]) for r in rows)
        for row in rows:
            sid = row["scene_id"]
            out = paths.feature_dir(spec.name, sid) / spec.out_check
            flags = "done" if scene_done(paths, spec, sid) else "pending"
            bb = ""
            if spec.needs_bboxes:
                bb = "  bboxes=OK" if paths.sam3_bboxes(sid).is_file() else "  bboxes=MISSING"
            print(f"  {sid:24s} {row['width']}x{row['height']} {row['n_frames']}f  "
                  f"[{flags}]{bb} → {out}")
        print(f"[dry-run] {done}/{len(rows)} already have {spec.out_check}")
        return 0

    gpus = list(scene_db.parse_int_csv(args.devices))
    n = len(gpus)
    work_dir = paths.features_root / spec.name / "_runs"

    procs = []
    for i, gpu in enumerate(gpus):
        proc = run_worker(paths, spec, rows, gpu, i, n, work_dir, args)
        if proc is not None:
            procs.append((i, gpu, proc))

    codes = [(i, gpu, proc.wait()) for (i, gpu, proc) in procs]
    for i, gpu, code in codes:
        print(f"[shard {i}/{n} gpu={gpu}] exit code {code}")

    if spec.has_overlay and args.overlay:
        dest = args.debug_overlay_dir or (paths.features_root / spec.name / "_debug_overlays")
        got = gather_overlays(paths, spec, rows, dest)
        print(f"[overlays] gathered {got}/{len(rows)} overlay(s) → {dest}")

    # Report — count scenes with the stage output on disk; list any missing.
    done = [r["scene_id"] for r in rows if scene_done(paths, spec, r["scene_id"])]
    missing = [r["scene_id"] for r in rows if not scene_done(paths, spec, r["scene_id"])]
    print(f"\n[precompute_scenes stage={args.stage}] {len(done)}/{len(rows)} scene(s) "
          f"have {spec.out_check}")
    if missing:
        print(f"[precompute_scenes] {len(missing)} scene(s) MISSING output:")
        for sid in missing:
            print(f"   {sid}")

    all_ok = all(code == 0 for _, _, code in codes)
    return 0 if all_ok and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
