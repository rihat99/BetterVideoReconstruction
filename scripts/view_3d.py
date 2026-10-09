"""View finished ``out/<stem>/`` reconstructions in a browser (viser, through the BetterRobot viewer).

Point it at a root of clips (``out/``) or at one clip (``out/<stem>/``); with several clips a pager
in the sidebar switches between them in one process. Per clip it draws, in the metric world: the
fused scene cloud (and its normals), the per-frame human depth points, the camera (one moving
frustum on its path, or the fixed cameras of a calibrated rig), gravity, the fitted ground plane,
and every SMPL-X body source — ``original`` (converter output), ``prefit``, ``kinematics``,
``kinematics_2``, ``kindyn_1`` — as a skinned mesh with its skeleton, its contact frames (red
touching, green free) and, for ``kindyn_1``, the solved contact forces, the root's residual wrench
and the joints coloured by torque. The ``body source`` dropdown picks the source; the ``layers``
checkboxes toggle each kind; the sidebar video pane plays the source footage in sync
(``[viewer].video = false`` turns it off). Clips with measured forces (peter rigs, OpenCap) add
orange measured and blue plate-format arrows.

``--scenes`` browses the ClimbingVideos scene DB, ``--bedlam`` the BEDLAM2 ground truth (one
``ground truth`` source, contact labels and forces where ``scripts/label_bedlam2.py`` ran, video on);
both alias the selected scenes into a throwaway ``out/``-shaped symlink farm. Knobs live in the
config's ``[viewer]`` section. This script is where the dataset readers
(``bvr/viewer/datasets_layers.py``) are plugged into the generic viewer.

Usage::

    uv run python scripts/view_3d.py ../data/examples/out/             # every clip, pager in the sidebar
    uv run python scripts/view_3d.py ../data/examples/out/boulder_1/   # one clip
    uv run python scripts/view_3d.py --scenes                          # the ClimbingVideos DB
    uv run python scripts/view_3d.py --bedlam                          # BEDLAM2 ground truth
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bvr.config import load_toml, section  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("target", type=Path, nargs="?", default=None,
                   help="root of clips (out/) or a single clip dir (out/<stem>/); omit with --scenes/--bedlam")
    p.add_argument("--scenes", action="store_true", help="browse the ClimbingVideos scene DB")
    p.add_argument("--bedlam", action="store_true",
                   help="browse the BEDLAM2 ground truth (video pane on); every scene unless "
                        "[viewer].scene_ids / sample / limit picks a set")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "pipeline.toml",
                   help="pipeline TOML (default: configs/pipeline.toml)")
    cli = p.parse_args(sys.argv[1:] if argv is None else argv)
    if cli.scenes and cli.bedlam:
        p.error("--bedlam and --scenes pick different DBs; use one")
    args = section(load_toml(cli.config), "viewer", {
        "port": 8082, "point_size": 0.02, "human_point_size": 0.01, "camera_scale": 0.1, "up": "-y",
        "smplx": True, "human_max_points": 6000, "video": True, "video_dir": "",
        "data_root": "", "categories": "1,2", "rope_supported": 0, "scene_ids": "", "sample": "",
        "limit": "", "seed": 0,
    })

    from bvr.viewer import datasets_layers, view_3d
    from bvr.viewer.app import CLIP_MARKERS

    # The footage and the ground-truth body are what a BEDLAM2 scene is checked against.
    view = partial(view_3d, port=args.port, up=args.up, point_size=args.point_size,
                   human_point_size=args.human_point_size, camera_scale=args.camera_scale,
                   show_smplx=args.smplx or cli.bedlam, human_max_points=args.human_max_points,
                   video=args.video or cli.bedlam, read=datasets_layers.read_clip,
                   markers=(*CLIP_MARKERS, datasets_layers.BEDLAM_MARKER))
    if cli.scenes or cli.bedlam:
        args.scenes = "bedlam" if cli.bedlam else "climbing"
        args.data_root = Path(args.data_root) if args.data_root else None
        args.scene_ids = args.scene_ids or None
        args.sample = int(args.sample) if args.sample != "" else None
        args.limit = int(args.limit) if args.limit != "" else None
        datasets_layers.run_scene_db(args, view)
        return 0
    if cli.target is None:
        p.error("give a predictions dir (out/ or out/<stem>/) or use --scenes / --bedlam")
    view(cli.target, video_dir=Path(args.video_dir) if args.video_dir else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
