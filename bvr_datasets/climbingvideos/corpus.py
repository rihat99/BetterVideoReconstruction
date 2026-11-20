"""Run steps of the chain over the ClimbingVideos scene corpus.

The corpus keeps its files stage first, ``features/<stage>/<aa>/<bb>/<scene>/``; a run of the chain
keeps them clip first, ``<root>/<scene>/<stage>/``. A view joins the two without copying: for every
selected scene, ``<view>/<scene>.mp4`` links its video and ``<view>/<scene>/<stage>`` links each of
its stage folders, so the view is a run root the pipeline and the viewer read as they read any other.

With ``--save-dir`` the steps write a run root of their own and read everything else from the corpus,
which they leave untouched; the view is ``<save-dir>/_corpus``. Without it the view is
``<corpus>/bvr`` and the steps write through its links into the feature tree, where readers of the
corpus find them; what they replace goes to ``<corpus>/bvr/_trash``. A step whose output is a whole
stage folder (sam3, vggt, sapiens, gravity) replaces the link rather than writing through it, so it
runs only with ``--save-dir``.

    uv run python bvr_datasets/climbingvideos/corpus.py --sample 3 --save-dir <root> \\
        --steps kinematics_1,contacts_1,kinematics_2,contacts_2,kindyn_1
    uv run python scripts/view_3d.py --source <root>/_corpus --save-dir <root> --upstream <root>/_corpus

Several GPUs fill one root with ``--shard i/n``, one process per GPU. Which scenes the corpus holds is
``[corpus]`` of ``configs/corpus.toml``; ``--scenes``, ``--sample`` and ``--limit`` narrow it.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import tomllib
from pathlib import Path

from bvr import run
from bvr_datasets.climbingvideos import scene_db
from bvr_datasets.climbingvideos.paths import CLIMBING

CONFIG = Path(__file__).resolve().parent / "configs" / "corpus.toml"

#: The stage folders of a scene that a clip tree holds.
STAGES = ("sam3", "vggt", "sapiens", "sam3d", "geometry", "geocalib", "human_optim")


def view(paths: scene_db.ScenePaths, rows: list, root: Path, *, create: set[str] = frozenset()) -> None:
    """Link the scenes of ``rows`` into ``root`` as a run root: a video and the stage folders it has, per scene.

    A stage of ``create`` that a scene does not have yet gets its folder in the feature tree, for a step to write
    into through the link.
    """
    for row in rows:
        scene, video = row["scene_id"], paths.clip_path(row)
        for stage in create:
            paths.feature_dir(stage, scene).mkdir(parents=True, exist_ok=True)
        links = {root / f"{scene}{video.suffix}": video}
        links |= {root / scene / stage: paths.feature_dir(stage, scene) for stage in STAGES
                  if paths.feature_dir(stage, scene).is_dir()}
        for link, target in links.items():
            if not link.is_symlink():
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", help="steps to run, in chain order (default: only build the view)")
    parser.add_argument("--save-dir", type=Path, help="a run root of its own (default: write into the corpus)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--scenes", help="comma-separated scene ids; --scenes=<ids> when the first starts with -")
    selection.add_argument("--sample", type=int, help="a random subset of this many scenes, drawn with [corpus].seed")
    selection.add_argument("--limit", type=int, help="the first scenes, in id order")
    parser.add_argument("--config", type=Path, help="a delta over configs/bvr.toml")
    parser.add_argument("--skip-existing", action="store_true", help="skip a step whose manifest still holds")
    parser.add_argument("--shard", default="1/1", help="i/n: every n-th scene from the i-th")
    parser.add_argument("--dry-run", action="store_true", help="build the view, log the commands, run nothing")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    with CONFIG.open("rb") as file:
        corpus = tomllib.load(file)["corpus"]
    paths = scene_db.ScenePaths(CLIMBING)
    with paths.connect() as connection:
        rows = scene_db.select_targets(connection, categories=tuple(corpus["categories"]),
                                       rope_supported=None if corpus["rope_supported"] < 0 else corpus["rope_supported"])
    rows = scene_db.resolve_rows(rows, scenes=args.scenes, sample=args.sample, limit=args.limit, seed=corpus["seed"])
    index, count = (int(part) for part in args.shard.split("/"))
    rows = scene_db.stride_shard(rows, index - 1, count)     # shards link and run disjoint scenes
    if not rows:
        print("[corpus] no scene matches the selection", flush=True)
        return 1
    names = args.steps.split(",") if args.steps else []
    if unknown := sorted(set(names) - {step.name for step in run.STEPS}):
        parser.error(f"no step named {', '.join(unknown)}")
    steps = [step for step in run.STEPS if step.name in names]
    if args.save_dir is None:
        if folders := [step.name for step in steps if any("/" not in output for output in step.outputs)]:
            parser.error(f"{', '.join(folders)} write whole stage folders; run them with --save-dir")
        root, written = CLIMBING / "bvr", {output.split("/")[0] for step in steps for output in step.outputs}
    else:
        root, written = args.save_dir / "_corpus", set()
    view(paths, rows, root, create=set() if args.dry_run else written)
    print(f"[corpus] {len(rows)} scene(s) in the view {root}", flush=True)
    if not names:
        return 0
    # The source is a folder of the selected videos alone: the view may hold scenes of an earlier selection.
    with tempfile.TemporaryDirectory(prefix="corpus_") as source:
        for row in rows:
            video = paths.clip_path(row)
            (Path(source) / f"{row['scene_id']}{video.suffix}").symlink_to(video)
        return run.chain(Path(source), args.save_dir or root, config_path=args.config, steps=names,
                         upstream=None if args.save_dir is None else root, skip_existing=args.skip_existing,
                         dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
