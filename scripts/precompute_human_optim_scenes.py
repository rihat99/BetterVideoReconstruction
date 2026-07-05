#!/usr/bin/env python3
"""Run ``human_optimization`` (root + kinematics_1) over ClimbingVideos scenes.

The fourth wave of the ``./data`` -> ``../data/ClimbingVideos`` scene-DB migration
(``track_sam3_scenes.py`` did SAM 3; ``precompute_scenes.py`` did vggt / sapiens /
sam3d / smplx; ``precompute_geometry_scenes.py`` did scale + fuse). It runs the
visually-guided SMPL-X kinematic refinement — the default ``root, kinematics_1``
chain — for every tracked person, writing ``human_optim/kinematic_1.npz``.

Like ``estimate_scale`` / ``fuse_geometry``, ``human_optimization`` is a single-clip
``--predictions`` stage expecting a *unified* ``out/<stem>/`` tree
(``sam3d/``, ``sapiens/``, ``geometry/`` beside a writable ``human_optim/``); the
scene DB stores those stage-first and split. This driver bridges the two with the
same per-scene **symlink farm** the geometry driver uses — read-only links into the
input feature trees plus a writable link for the ``human_optim`` output::

    features/{sam3d,sapiens,geometry}/<aa>/<bb>/<scene>/  ->  farm/{sam3d,sapiens,geometry}
    features/human_optim/<aa>/<bb>/<scene>/               ->  farm/human_optim  (output)

One worker per GPU (default 1,2,3) processes a disjoint stride shard; each worker
imports the stage **once** and loops its scenes in-process, so torch + CUDA + the
SMPL-X model start once per worker, not once per scene. Overlays are off (batch
mode). Resumable with ``--skip-existing`` (skips scenes that already have
``human_optim/kinematic_1.npz``). Per-scene failures are caught and logged; the
shard keeps going.

    # full target set on GPUs 1,2,3, resumable:
    uv run python scripts/precompute_human_optim_scenes.py --skip-existing

    # smoke test 3 scenes (one per GPU), print the plan first:
    uv run python scripts/precompute_human_optim_scenes.py --sample 3 --dry-run
    uv run python scripts/precompute_human_optim_scenes.py --sample 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import scene_db  # noqa: E402  (scripts/ on path)

# Inputs symlinked read-only into the farm; human_optim is the writable output.
_INPUT_STAGES = ("sam3d", "sapiens", "geometry")
_OUTPUT_STAGE = "human_optim"
_OUTPUT_NPZ = "kinematic_1.npz"


# ------------------------------ symlink farm -------------------------------


def _build_farm(entry: dict, root: Path) -> Path:
    """A throwaway ``pred/`` dir: input stages -> feature trees, human_optim -> output."""
    farm = root / entry["scene_id"]
    farm.mkdir(parents=True, exist_ok=True)
    for name in _INPUT_STAGES:
        link = farm / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(entry[name])
    out = Path(entry[_OUTPUT_STAGE])
    out.mkdir(parents=True, exist_ok=True)   # real, writable output dir
    link = farm / _OUTPUT_STAGE
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(out)
    return farm


def _drop_farm(farm: Path) -> None:
    """Remove only the symlinks + the farm dir; never follow into feature trees."""
    for name in (*_INPUT_STAGES, _OUTPUT_STAGE):
        link = farm / name
        if link.is_symlink() or link.exists():
            link.unlink()          # unlink the symlink; its target is untouched
    if farm.is_dir():
        farm.rmdir()


# ------------------------------ worker (one GPU) ---------------------------


def run_worker(argv: argparse.Namespace) -> int:
    """Process one shard's manifest in-process (imports the stage once)."""
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "stages"))
    import human_optimization as ho_stage   # noqa: E402  (heavy: torch/cv2, once)

    entries = json.loads(Path(argv.manifest).read_text())
    farms_root = Path(argv.manifest).parent / "farms"
    farms_root.mkdir(parents=True, exist_ok=True)

    n = len(entries)
    ok = 0
    for i, e in enumerate(entries):
        sid = e["scene_id"]
        out_npz = Path(e[_OUTPUT_STAGE]) / _OUTPUT_NPZ
        if argv.skip_existing and out_npz.is_file():
            print(f"[{argv.tag} {i + 1}/{n}] {sid}: skip ({_OUTPUT_NPZ} exists)", flush=True)
            ok += 1
            continue
        t0 = time.time()
        farm = _build_farm(e, farms_root)
        try:
            ho_stage.main(["--predictions", str(farm), "--device", argv.device,
                           "--stages", "root,kinematics_1"])
            ok += out_npz.is_file()
            print(f"[{argv.tag} {i + 1}/{n}] {sid}: done in {time.time() - t0:.1f}s", flush=True)
        except Exception:                    # noqa: BLE001 — tolerate per-scene failure
            print(f"[{argv.tag} {i + 1}/{n}] {sid}: FAILED after {time.time() - t0:.1f}s",
                  flush=True)
            traceback.print_exc()
        finally:
            _drop_farm(farm)

    print(f"[{argv.tag}] {ok}/{n} scene(s) have human_optim/{_OUTPUT_NPZ}", flush=True)
    return 0 if ok == n else 1


# ------------------------------ orchestration ------------------------------


def run_orchestrator(args: argparse.Namespace) -> int:
    paths = scene_db.ScenePaths(args.data_root)
    if not paths.scenes_db.is_file():
        print(f"ERROR: scenes.db not found at {paths.scenes_db}", file=sys.stderr)
        return 1

    categories = scene_db.parse_int_csv(args.categories)
    rope = None if args.rope_supported < 0 else args.rope_supported
    with paths.connect() as conn:
        rows = scene_db.select_targets(conn, categories=categories, rope_supported=rope)
    rows = scene_db.resolve_rows(
        rows, scenes=args.scenes, sample=args.sample, limit=args.limit, seed=args.seed)

    print(f"[precompute_human_optim] target scenes: {len(rows)} "
          f"(categories={categories}, rope_supported={rope})")
    if not rows:
        return 0

    def entry(row) -> dict:
        sid = row["scene_id"]
        e = {"scene_id": sid, "clip": str(paths.clip_path(row))}
        for stage in (*_INPUT_STAGES, _OUTPUT_STAGE):
            e[stage] = str(paths.feature_dir(stage, sid))
        return e

    def done(sid: str) -> bool:
        return (paths.feature_dir(_OUTPUT_STAGE, sid) / _OUTPUT_NPZ).is_file()

    if args.dry_run:
        n_done = sum(done(r["scene_id"]) for r in rows)
        for row in rows:
            sid = row["scene_id"]
            flag = "done" if done(sid) else "pending"
            print(f"  {sid:24s} {row['width']}x{row['height']} {row['n_frames']}f  [{flag}]")
        print(f"[dry-run] {n_done}/{len(rows)} already have human_optim/{_OUTPUT_NPZ}")
        return 0

    gpus = list(scene_db.parse_int_csv(args.devices))
    n = len(gpus)
    work_dir = paths.features_root / _OUTPUT_STAGE / "_runs"
    work_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for i, gpu in enumerate(gpus):
        shard = scene_db.stride_shard(rows, i, n)
        if not shard:
            print(f"[shard {i}/{n} gpu={gpu}] no scenes — skipping")
            continue
        shard_dir = work_dir / f"shard_{i}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        manifest = shard_dir / "manifest.json"
        manifest.write_text(json.dumps([entry(r) for r in shard], indent=2))

        cmd = [
            sys.executable, str(_HERE),
            "--worker", "--manifest", str(manifest),
            "--device", "cuda", "--tag", f"g{gpu}",
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print(f"[shard {i}/{n} gpu={gpu}] {len(shard)} scene(s) -> {manifest.name}")
        procs.append((i, gpu, subprocess.Popen(cmd, env=env)))

    codes = [(i, gpu, proc.wait()) for (i, gpu, proc) in procs]
    for i, gpu, code in codes:
        print(f"[shard {i}/{n} gpu={gpu}] exit code {code}")

    n_done = [r["scene_id"] for r in rows if done(r["scene_id"])]
    missing = [r["scene_id"] for r in rows if not done(r["scene_id"])]
    print(f"\n[precompute_human_optim] {len(n_done)}/{len(rows)} scene(s) have human_optim/{_OUTPUT_NPZ}")
    if missing:
        print(f"[precompute_human_optim] {len(missing)} scene(s) MISSING output:")
        for sid in missing:
            print(f"   {sid}")
    return 0 if all(c == 0 for _, _, c in codes) and not missing else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, default=scene_db.DEFAULT_DATA_ROOT,
                   help=f"ClimbingVideos data root (default: {scene_db.DEFAULT_DATA_ROOT}).")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--sample", type=int, default=None,
                     help="process a random subset of N target scenes (deterministic per --seed).")
    sel.add_argument("--limit", type=int, default=None,
                     help="process the first N target scenes (by scene_id).")
    sel.add_argument("--all", action="store_true", help="process every target scene (default).")
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
                   help="skip scenes that already have human_optim/kinematic_1.npz.")
    p.add_argument("--dry-run", action="store_true",
                   help="print the selected scenes + done/pending state and exit (no GPU).")
    # worker-only (spawned by the orchestrator; not for direct use):
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    p.add_argument("--tag", default="g?", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args.worker:
        if args.manifest is None:
            p.error("--worker requires --manifest")
        return run_worker(args)
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
