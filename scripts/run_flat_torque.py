"""Rerun ``kindyn_1`` with every joint's torque priced the same, beside the shipped run.

The shipped chain prices a joint torque by how strong that joint is
(``[torque.multipliers]``: a wrist torque costs 17x a hip one, a finger torque 25x), so
a load the body could carry with either limb goes where the weighted torque is cheapest.
This variant switches that off — every joint's torque costs 1 — and changes **nothing
else**, so the difference between the two trees is the per-joint weighting and only that.

Only ``kindyn_1`` is rerun. Nothing upstream of it reads the torque table: the pose, the
contact labels and the measured board forces are all inherited from the shipped run by
symlink, so the two trees start from a byte-identical body and the same contacts.

Everything lands in ``<clip>/human_optim_flat_torque/``, beside ``<clip>/human_optim/``.
The stage itself is untouched: each clip gets a small symlink farm ``<clip>/_flat_torque/``
whose ``human_optim`` points at the variant directory, so the stock stage CLI writes into
the variant tree without knowing it exists — the same trick ``run_gt_contact.py`` uses.

The config is DERIVED from ``configs/human_optimization_peter.toml`` at launch, with only
the ``[torque.multipliers]`` values rewritten to 1. It is never hand-copied: a copied
per-variant config goes stale the moment a weight changes in the shipped one.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/run_flat_torque.py
    # four GPUs over one tree: one process each, disjoint quarters of the clips
    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/run_flat_torque.py --shard 1/4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Output directory of the variant, beside the normal ``human_optim/``.
VARIANT_DIR = "human_optim_flat_torque"
#: The symlink farm the stock stage is pointed at.
FARM_DIR = "_flat_torque"
#: Stage input directories the farm links through to the real tree.
FARM_LINKS = ("sam3", "vggt", "sapiens", "sam3d", "geometry", "geocalib")
#: Everything kindyn_1 reads out of ``human_optim/`` — none of it depends on the torque table.
INHERITED = ("kinematic_1.npz", "kinematic_2.npz", "contacts_1.npz", "contacts_2.npz",
             "sensor_forces.npz")


def flat_torque_config(source: Path, dest: Path) -> Path:
    """Write ``source`` out with every ``[torque.multipliers]`` entry set to 1.

    A line-level rewrite rather than a re-serialized TOML: every comment, weight and
    section of the shipped config survives untouched, so a diff of the two files shows
    exactly one block changed.
    """
    lines, in_table = [], False
    for line in source.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_table = stripped == "[torque.multipliers]"
        elif in_table and "=" in stripped and not stripped.startswith("#"):
            line = f"{stripped.split('=')[0].strip()} = 1.0"
        lines.append(line)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    table = tomllib.loads(dest.read_text())["torque"]["multipliers"]
    if set(table.values()) != {1.0}:
        raise SystemExit(f"[flat_torque] {dest} still has non-unit multipliers: {table}")
    return dest


def build_farm(clip_dir: Path) -> Path:
    """Create (or refresh) the clip's symlink farm; return the path to point the stage at."""
    variant, farm = clip_dir / VARIANT_DIR, clip_dir / FARM_DIR
    variant.mkdir(exist_ok=True)
    farm.mkdir(exist_ok=True)
    for name in INHERITED:
        (variant / name).unlink(missing_ok=True)
        (variant / name).symlink_to(Path("..") / "human_optim" / name)
    for name in FARM_LINKS:
        (farm / name).unlink(missing_ok=True)
        (farm / name).symlink_to(Path("..") / name)
    (farm / "human_optim").unlink(missing_ok=True)
    (farm / "human_optim").symlink_to(Path("..") / VARIANT_DIR)
    return farm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="climb_wall_3", help="dataset folder under peter/")
    ap.add_argument("--clips", default=None, help="comma-separated clip names (default: all)")
    ap.add_argument("--shard", default="1/1", help="'i/n': process every n-th clip")
    ap.add_argument("--config", type=Path,
                    default=_REPO_ROOT / "configs" / "human_optimization_peter.toml")
    args = ap.parse_args(argv)

    out_root = _REPO_ROOT / "peter" / f"out_{args.dataset}_single"
    source_root = _REPO_ROOT / "peter" / args.dataset
    clips = sorted(d for d in out_root.iterdir() if (d / "human_optim" / "kindyn_1.npz").exists())
    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = [c for c in clips if c.name in wanted]
        if missing := wanted - {c.name for c in clips}:
            raise SystemExit(f"[flat_torque] unknown clip(s) {sorted(missing)}")
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    clips = clips[shard - 1::n_shards]
    if not clips:
        raise SystemExit(f"[flat_torque] no finished clip under {out_root}")

    config = flat_torque_config(args.config, out_root / "_configs" / args.config.name)
    print(f"[flat_torque] {len(clips)} clip(s) -> {out_root}/<clip>/{VARIANT_DIR} "
          f"(shard {shard}/{n_shards})\n[flat_torque] config {config}")

    failed = []
    for index, clip_dir in enumerate(clips, 1):
        print(f"\n########## [{index}/{len(clips)}] {clip_dir.name} ##########", flush=True)
        done = clip_dir / VARIANT_DIR / "kindyn_1.npz"
        if done.exists():
            print(f"[flat_torque] skip {clip_dir.name} (output exists)")
            continue
        farm = build_farm(clip_dir)
        command = [sys.executable, str(_REPO_ROOT / "scripts" / "stages" / "human_optimization.py"),
                   "--source", str(source_root / clip_dir.name / "cam_left.mp4"),
                   "--predictions", str(farm), "--stages", "kindyn_1", "--config", str(config)]
        print(f"\n=== {' '.join(command[1:])} ===", flush=True)
        if subprocess.run(command).returncode != 0 or not done.exists():
            print(f"[flat_torque] {clip_dir.name}: FAILED")
            failed.append(clip_dir.name)
            continue
        print(f"[flat_torque] {clip_dir.name}: ok")
    print(f"\n[flat_torque] done. {len(clips) - len(failed)}/{len(clips)} clip(s)."
          + (f" failed: {failed}" if failed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
