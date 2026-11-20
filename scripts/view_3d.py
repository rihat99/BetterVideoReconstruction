"""View the clips of a run in a browser: scene, cameras and video, bodies, contacts and forces, over time.

The viewer takes the pipeline's targets: the videos of ``--source`` (one video or a folder of them) whose clips
``--save-dir`` holds, each file read from the clip's own tree, else from ``--upstream``'s. Display settings are the
config's ``[viewer]`` table. ``--record`` writes the first clip to a video (``.mp4``, ``.webm``) or a ``.png`` still
through a headless Chromium instead of serving it.

    uv run python scripts/view_3d.py --source ../data/examples/data --save-dir ../data/examples/out
    uv run python scripts/view_3d.py --source ../data/examples/data/boulder_1.mp4 --save-dir <run> --upstream <root>
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bvr import config, paths
from bvr.viewer import app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="a video or a folder of videos")
    parser.add_argument("--save-dir", type=Path, required=True, help="the run root that holds their clips")
    parser.add_argument("--upstream", type=Path, help="a run root to read what the run does not hold from")
    parser.add_argument("--config", type=Path, help="a config delta (default: configs/bvr.toml)")
    parser.add_argument("--record", type=Path, help="write the first clip to this video or still and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    clips = [clip for clip in paths.clips(args.source, args.save_dir, args.upstream)
             if clip.find(paths.TRANSFORM).exists()]
    if not clips:
        raise SystemExit(f"no clip of {args.source} under {args.save_dir} has {paths.TRANSFORM} (the scale step)")
    app.serve(clips, config.load(args.config)["viewer"], record=args.record)


if __name__ == "__main__":
    main()
