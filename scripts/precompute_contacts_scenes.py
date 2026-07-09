#!/usr/bin/env python3
"""DB-driven per-joint scene-contact precompute over ClimbingVideos scenes.

The contact readout (``estimate_contacts``) is a ``--predictions`` stage, not a
``precompute_runner`` Step, so — like the human-optimization DB pass — it runs on a
per-scene **symlink farm** rather than a JSON manifest: for each target scene we
stitch the optimized body (``features/human_optim/<...>/<body>.npz``) and the fused
geometry (``features/geometry/<...>/{scene,transform}.npz``) into a throwaway
``--predictions`` dir, run the stage, and move its ``contacts_<tag>.npz`` next to the
body it scores, in that body's own ``human_optim`` scene dir::

    features/human_optim/<aa>/<bb>/<scene_id>/contacts_<tag>.npz

(mirroring the local ``out/<stem>/human_optim/`` layout). Every other DB tree stays
READ-ONLY — the only output is that one ``contacts_<tag>.npz`` file. Scenes are split across
the configured GPUs (one worker process per GPU over a disjoint stride shard); per-scene
failures are reported and skipped, never fatal. Resumable with ``--skip-existing``.

    # smoke test 2 scenes on one GPU, print the plan first:
    uv run python scripts/precompute_contacts_scenes.py --sample 2 --dry-run
    uv run python scripts/precompute_contacts_scenes.py --sample 2 --devices 2

    # full target set, resumable, GPUs 1,2,3:
    uv run python scripts/precompute_contacts_scenes.py --skip-existing
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import scene_db  # noqa: E402  (scripts/ on path)

_CONTACTS_CLI = _REPO_ROOT / "scripts" / "stages" / "estimate_contacts.py"
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"


def _out_path(paths: scene_db.ScenePaths, scene_id: str, tag: str) -> Path:
    return paths.feature_dir("human_optim", scene_id) / f"contacts_{tag}.npz"


def _process_scene(paths: scene_db.ScenePaths, out_paths: scene_db.ScenePaths, row, gpu: int,
                   bodies: str, config: Path, skip_existing: bool) -> str:
    """Run the contact readout for one scene on ``gpu``. Returns 'done' / 'skip' / 'fail'.

    Inputs are resolved under ``paths`` (the read-only DB); the ``contacts`` output goes
    under ``out_paths`` (== ``paths`` for the real run, a scratch root for a smoke test).
    """
    sid = row["scene_id"]
    tag = bodies.removeprefix("kinematic_")
    out = _out_path(out_paths, sid, tag)
    if skip_existing and out.is_file():
        return "skip"
    body = paths.feature_dir("human_optim", sid) / f"{bodies}.npz"
    geom = paths.feature_dir("geometry", sid)
    if not body.is_file() or not (geom / "scene.npz").is_file() or not (geom / "transform.npz").is_file():
        print(f"  [skip] {sid}: missing body/geometry inputs")
        return "skip"

    with tempfile.TemporaryDirectory(prefix=f"contacts_{sid}_") as farm:
        farm = Path(farm)
        (farm / "human_optim").mkdir()
        (farm / "human_optim" / f"{bodies}.npz").symlink_to(body.resolve())
        (farm / "geometry").symlink_to(geom.resolve())
        # sapiens feeds the round-11 detector-belief factor; optional (contacts degrade to neutral).
        sap = paths.feature_dir("sapiens", sid)
        if sap.is_dir():
            (farm / "sapiens").symlink_to(sap.resolve())
        cmd = [sys.executable, str(_CONTACTS_CLI), "--source", str(paths.clip_path(row)),
               "--predictions", str(farm), "--bodies", bodies, "--config", str(config),
               "--device", "cuda", "--no-overlay"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        produced = farm / "human_optim" / f"contacts_{tag}.npz"
        if res.returncode != 0 or not produced.is_file():
            print(f"  [FAIL] {sid} (rc={res.returncode})\n{res.stderr.strip()[-400:]}")
            return "fail"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), str(out))
    return "done"


def _run_worker(argv: list[str]) -> int:
    """Internal: process one stride shard sequentially on one GPU."""
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=int, required=True)   # shard index i
    p.add_argument("--nshards", type=int, required=True)  # n
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--data-root", type=Path, default=scene_db.DEFAULT_DATA_ROOT)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--bodies", default="kinematic_1")
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    p.add_argument("--categories", default=",".join(map(str, scene_db.DEFAULT_CATEGORIES)))
    p.add_argument("--rope-supported", type=int, default=scene_db.DEFAULT_ROPE_SUPPORTED)
    p.add_argument("--scenes", default=None)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    a = p.parse_args(argv)
    paths = scene_db.ScenePaths(a.data_root)
    out_paths = scene_db.ScenePaths(a.out_root) if a.out_root else paths
    rope = None if a.rope_supported < 0 else a.rope_supported
    with paths.connect() as conn:
        rows = scene_db.select_targets(conn, categories=scene_db.parse_int_csv(a.categories), rope_supported=rope)
    rows = scene_db.resolve_rows(rows, scenes=a.scenes, sample=a.sample, limit=a.limit, seed=a.seed)
    shard = scene_db.stride_shard(rows, a.worker, a.nshards)
    counts = {"done": 0, "skip": 0, "fail": 0}
    for row in shard:
        counts[_process_scene(paths, out_paths, row, a.gpu, a.bodies, a.config, a.skip_existing)] += 1
    print(f"[shard {a.worker}/{a.nshards} gpu={a.gpu}] {counts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        return _run_worker(argv)

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, default=scene_db.DEFAULT_DATA_ROOT)
    p.add_argument("--out-root", type=Path, default=None,
                   help="write the contacts feature tree under here instead of --data-root "
                        "(smoke tests point this at a scratch dir; the DB trees stay read-only).")
    p.add_argument("--bodies", default="kinematic_1", choices=["kinematic_1", "kinematic_2"],
                   help="which optimized body to read contact from (contacts_1 / contacts_2).")
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    p.add_argument("--devices", default="1,2,3", help="comma-separated GPU ids, one worker each.")
    p.add_argument("--categories", default=",".join(map(str, scene_db.DEFAULT_CATEGORIES)))
    p.add_argument("--rope-supported", type=int, default=scene_db.DEFAULT_ROPE_SUPPORTED)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--sample", type=int, default=None)
    sel.add_argument("--limit", type=int, default=None)
    sel.add_argument("--scenes", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="list selected scenes + input/output state, no GPU.")
    a = p.parse_args(argv)

    paths = scene_db.ScenePaths(a.data_root)
    out_paths = scene_db.ScenePaths(a.out_root) if a.out_root else paths
    if not paths.scenes_db.is_file():
        print(f"ERROR: scenes.db not found at {paths.scenes_db}", file=sys.stderr)
        return 1
    rope = None if a.rope_supported < 0 else a.rope_supported
    with paths.connect() as conn:
        rows = scene_db.select_targets(conn, categories=scene_db.parse_int_csv(a.categories), rope_supported=rope)
    rows = scene_db.resolve_rows(rows, scenes=a.scenes, sample=a.sample, limit=a.limit, seed=a.seed)
    tag = a.bodies.removeprefix("kinematic_")
    print(f"[precompute_contacts_scenes] target scenes: {len(rows)}  bodies={a.bodies}")
    if not rows:
        return 0

    if a.dry_run:
        for row in rows:
            sid = row["scene_id"]
            body = paths.feature_dir("human_optim", sid) / f"{a.bodies}.npz"
            have = body.is_file() and (paths.feature_dir("geometry", sid) / "scene.npz").is_file()
            done = _out_path(out_paths, sid, tag).is_file()
            print(f"  {sid:24s} inputs={'OK' if have else 'MISSING'}  {'done' if done else 'pending'}")
        return 0

    gpus = list(scene_db.parse_int_csv(a.devices))
    procs = []
    for i, gpu in enumerate(gpus):
        cmd = [sys.executable, str(_HERE), "--worker", str(i), "--nshards", str(len(gpus)),
               "--gpu", str(gpu), "--data-root", str(a.data_root), "--bodies", a.bodies,
               "--config", str(a.config), "--categories", a.categories,
               "--rope-supported", str(a.rope_supported), "--seed", str(a.seed)]
        if a.out_root:
            cmd += ["--out-root", str(a.out_root)]
        if a.scenes:
            cmd += ["--scenes", a.scenes]
        if a.sample is not None:
            cmd += ["--sample", str(a.sample)]
        if a.limit is not None:
            cmd += ["--limit", str(a.limit)]
        if a.skip_existing:
            cmd.append("--skip-existing")
        procs.append(subprocess.Popen(cmd))
    codes = [pr.wait() for pr in procs]

    done = [r["scene_id"] for r in rows if _out_path(out_paths, r["scene_id"], tag).is_file()]
    print(f"\n[precompute_contacts_scenes] {len(done)}/{len(rows)} scene(s) have contacts_{tag}.npz")
    return 0 if all(c == 0 for c in codes) and len(done) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
