"""Run the contact-consuming chain on MEASURED board contacts, for the peter rigs.

The pipeline's own contacts are estimated from the video. On the instrumented rigs
the contact state is measured instead — one three-axis force board per limb — so the
same chain can be run with the guesswork replaced by ground truth, and the two trees
compared to see what contact detection actually costs::

    contacts_1 (boards) -> kinematics_2 -> contacts_2 (boards) -> kindyn_1 -> overlay

The last step draws ``kindyn_forces_overlay.mp4`` beside the variant's own
``kindyn_1.npz`` — the usual force overlay, solved contacts and forces against the
boards, so the two runs can be watched side by side.

Two datasets are instrumented this way: the peter rigs (``--dataset climb_wall_3``,
one board per limb) and the LAAS Parkour clips (``--dataset parkour``, the feet on floor
plates and the hands on a sensed bar). Both write the same ``sensor_forces.npz``, so the
chain below is the same for either.

``kinematic_1`` is shared with the normal run, not recomputed: it never sees a contact
label. The boards say which LIMB is loaded and when; which of that limb's contact
frames carries the label is still read off the body, which is why the board contacts
are measured twice — once on ``kinematic_1`` and again on the repaired ``kinematic_2``,
exactly as the pipeline re-reads its own contacts.

Everything lands in ``<clip>/human_optim_gt_contact/``, beside the normal
``<clip>/human_optim/``. The stages themselves are untouched: each clip gets a small
symlink farm ``<clip>/_gt_contact/`` whose ``human_optim`` points at the variant
directory and whose inputs point at the real ones, so the stock stage CLIs write into
the variant tree without knowing it exists.

Forces are NOT supervised — the boards decide contact only, so the solved force stays
an independent prediction and force error against the boards stays a real measurement.

Knobs live in ``configs/human_optimization_peter.toml`` (``[boards]`` for the label
thresholds; every weight shared with the normal run). This CLI takes only targets.

Usage::

    CUDA_VISIBLE_DEVICES=4 uv run --no-sync python scripts/run_gt_contact.py
    CUDA_VISIBLE_DEVICES=4 uv run --no-sync python scripts/run_gt_contact.py \\
        --clips a_00_01_part1 --steps contacts_1
    # four GPUs over one tree: one process each, disjoint quarters of the clips
    CUDA_VISIBLE_DEVICES=4 uv run --no-sync python scripts/run_gt_contact.py --shard 1/4
    CUDA_VISIBLE_DEVICES=6 uv run --no-sync python scripts/run_gt_contact.py --dataset parkour
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGES = _REPO_ROOT / "scripts" / "stages"

#: Output directory of the variant chain, beside the normal ``human_optim/``.
VARIANT_DIR = "human_optim_gt_contact"
#: The symlink farm the stock stages are pointed at.
FARM_DIR = "_gt_contact"
#: Stage input directories the farm links through to the real tree.
FARM_LINKS = ("sam3", "vggt", "sapiens", "sam3d", "geometry", "geocalib")
#: Artifacts the variant chain inherits from the normal run rather than recomputing.
INHERITED = ("kinematic_1.npz", "sensor_forces.npz")

#: step -> the file under ``human_optim_gt_contact/`` that means it is done.
STEP_OUTPUT: dict[str, str] = {
    "contacts_1": "contacts_1.npz",
    "kinematics_2": "kinematic_2.npz",
    "contacts_2": "contacts_2.npz",
    "kindyn_1": "kindyn_1.npz",
    "overlay": "kindyn_forces_overlay.mp4",
}


#: Clip trees that are not a peter rig: dataset -> (run tree, the clips' source root).
OTHER_TREES = {
    "parkour": Path("/home/rikhat.akizhanov/better/data/Parkour-dataset/processed"),
}


def tree_layout(dataset: str) -> tuple[Path, Path, str]:
    """The three things that differ per dataset: run tree, clip sources, video name."""
    if dataset in OTHER_TREES:
        source = OTHER_TREES[dataset]
        return source / "out", source, "video.mp4"
    return (_REPO_ROOT / "peter" / f"out_{dataset}_single",
            _REPO_ROOT / "peter" / dataset, "cam_left.mp4")


def build_farm(clip_dir: Path) -> Path:
    """Create (or refresh) the clip's symlink farm; return the path to point stages at."""
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


def step_command(step: str, video: Path, farm: Path, config: Path) -> list[str]:
    """Argv for one step of the chain (all paths absolute)."""
    if step == "overlay":
        # The renderer is pointed at the variant directory, not the farm: it reads the
        # clip's real video and geometry off the tree and draws the run it is given.
        return [sys.executable, str(_REPO_ROOT / "scripts" / "diagnostics" / "render_kindyn_overlays.py"),
                str(farm.parent / VARIANT_DIR)]
    if step in ("contacts_1", "contacts_2"):
        return [sys.executable, str(_STAGES / "measure_contacts_boards.py"),
                "--predictions", str(farm), "--config", str(config),
                "--bodies", "kinematic_1" if step == "contacts_1" else "kinematic_2"]
    return [sys.executable, str(_STAGES / "human_optimization.py"),
            "--source", str(video), "--predictions", str(farm),
            "--stages", step, "--config", str(config)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="climb_wall_3",
                    help="dataset folder under peter/, or " + ", ".join(OTHER_TREES))
    ap.add_argument("--clips", default=None, help="comma-separated clip names (default: all)")
    ap.add_argument("--steps", default=None,
                    help=f"comma-separated subset of {','.join(STEP_OUTPUT)} (default: all)")
    ap.add_argument("--shard", default="1/1", help="'i/n': process every n-th clip")
    ap.add_argument("--config", type=Path,
                    default=_REPO_ROOT / "configs" / "human_optimization_peter.toml")
    args = ap.parse_args(argv)

    out_root, source_root, video_name = tree_layout(args.dataset)
    clips = sorted(d for d in out_root.iterdir()
                   if (d / "human_optim" / "sensor_forces.npz").exists())
    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = [c for c in clips if c.name in wanted]
        if missing := wanted - {c.name for c in clips}:
            raise SystemExit(f"[gt_contact] unknown clip(s) {sorted(missing)}")
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    clips = clips[shard - 1::n_shards]
    steps = [s.strip() for s in args.steps.split(",")] if args.steps else list(STEP_OUTPUT)
    if unknown := [s for s in steps if s not in STEP_OUTPUT]:
        raise SystemExit(f"[gt_contact] unknown step(s) {unknown}; known: {list(STEP_OUTPUT)}")
    if not clips:
        raise SystemExit(f"[gt_contact] no clip with sensor_forces.npz under {out_root}")
    print(f"[gt_contact] {len(clips)} clip(s) -> {out_root}/<clip>/{VARIANT_DIR} "
          f"(shard {shard}/{n_shards})\n[gt_contact] steps: {steps}")

    failed = []
    for index, clip_dir in enumerate(clips, 1):
        print(f"\n########## [{index}/{len(clips)}] {clip_dir.name} ##########", flush=True)
        farm = build_farm(clip_dir)
        video = source_root / clip_dir.name / video_name
        for step in steps:
            done = clip_dir / VARIANT_DIR / STEP_OUTPUT[step]
            if done.exists():
                print(f"[gt_contact] skip {step} for {clip_dir.name} (output exists)")
                continue
            command = step_command(step, video, farm, args.config)
            print(f"\n=== {' '.join(command[1:])} ===", flush=True)
            if subprocess.run(command).returncode != 0 or not done.exists():
                print(f"[gt_contact] {clip_dir.name} {step}: FAILED — skipping the rest of this clip")
                failed.append(f"{clip_dir.name}/{step}")
                break
            print(f"[gt_contact] {clip_dir.name} {step}: ok")
    print(f"\n[gt_contact] done. {len(clips) - len({f.split('/')[0] for f in failed})}/{len(clips)} "
          f"clip(s) completed the chain." + (f" failed: {failed}" if failed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
