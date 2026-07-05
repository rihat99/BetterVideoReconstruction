"""View the fusion geometry (``geometry/``) for a whole batch in one viser viewer.

Point the viewer at a **root** directory of clips (``out/``) — or a single clip
(``out/<stem>/``) — and it serves them all from one process: an **example**
dropdown in the sidebar switches between clips without relaunching. Only the
selected clip's nodes live in the browser at a time (switching tears the old
clip down and rebuilds the new one); the loaded numpy is cached so re-selecting a
clip is fast.

For each clip it shows, in the metric world frame:

* the static **scene** point cloud (scaled + fused VGGT geometry),
* per-frame **human** depth points, split by climber, preloaded once and shown
  one frame at a time via the frame slider (playback flips visibility, never
  re-uploads),
* the per-frame **camera** frustums + centre path,
* the world-frame **gravity** arrow (when the geocalib stage has run),
* a **ground plane** fitted to the lowest scene points, and
* the fitted **SMPL-X bodies** for every tracked person, following the slider.

**Body source** dropdown switches between:

* ``original`` — the raw, pre-optimization SAM-3D-Body → SMPL-X converter output
  (``smplx_params.npz``, folded to the world frame);
* ``kinematics`` / ``kinematics_2`` — the refined world-frame bodies
  (``human_optim/kinematic_{1,2}.npz``);
* ``dynamics`` — the 52-joint skeleton as torque-scaled spheres + bones + cyan
  contact-force arrows (no mesh).

The three mesh sources are each one skinned mesh per person, posed per frame (LBS
in the browser, one upload) over a single averaged ``betas`` identity.

A **video** pane in the sidebar plays the clip's original footage in sync with the
frame slider (loaded from the repo's ``data/`` by default, or ``--video-dir``;
hidden when the source video is absent). The playback folder's **onboard camera**
toggle (off by default) rides the reconstructed camera — matching its pose and
field of view each frame and locking out manual orbiting — so you watch from the
filming camera's own viewpoint.

The sidebar's **layers** checkboxes toggle whole groups on/off — SMPL-X body,
human points, scene points, cameras, gravity, ground plane — so you never have to
open viser's scene tree. Layer choices persist across example switches. A **sizes**
panel scales the scene points, human points, and camera frustums independently.

The view opens framed on the clip's average human root (so orbiting pivots on the
climber, not the world origin), under viser's default lighting on a white backdrop.

It is a read-only viewer of a finished ``out/<stem>/`` tree: it loads the fused
geometry and fitted bodies and draws them, running no stage and writing nothing.

**ClimbingVideos scene DB** — with ``--scenes`` the viewer instead browses the
curated scene set from the ``../data/ClimbingVideos`` feature DB (the same targets
the precompute drivers process). Each scene's stage-first feature dirs
(``features/<stage>/<aa>/<bb>/<scene>/``) are aliased into a throwaway per-session
symlink farm shaped like ``out/<stem>/``, which the loaders read unchanged. When
more than one clip is loaded (a local ``out/`` batch or a scene set) the sidebar
gains a **◀ prev / random / next ▶** pager plus a "scene N / M" indicator.

The viewer package lives under :mod:`tools.viewer`; this file is the thin CLI.

Usage::

    uv run python scripts/view_3d.py out/            # all clips, pick in the UI
    uv run python scripts/view_3d.py out/<stem>/     # a single clip
    uv run python scripts/view_3d.py out/ [--port 8082] [--no-smplx]
    uv run python scripts/view_3d.py --scenes        # browse the ClimbingVideos DB
    uv run python scripts/view_3d.py --scenes --sample 50   # a random 50-scene subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.viewer import view_3d

# The stage feature dirs the viewer's loaders read (sapiens is stage-only, so it is
# not linked). Each is aliased as ``<farm>/<stage>`` → the scene's feature tree.
_FARM_STAGES = ("geometry", "sam3d", "vggt", "human_optim")


def _link_scene_farm(paths, row, farm_root: Path) -> bool:
    """Alias one DB scene's feature dirs into an ``out/<stem>/``-shaped symlink farm.

    Builds ``<farm_root>/<scene_id>/<stage>`` symlinks into the scene's stage-first
    feature trees (skipping stages that were never produced), plus the source clip
    as ``<farm_root>/<scene_id><ext>`` for the sidebar video pane. Returns ``True``
    only when the scene is fused (``geometry/scene.npz`` present) so the viewer can
    actually draw it — un-fused scenes are skipped.
    """
    sid = row["scene_id"]
    if not (paths.feature_dir("geometry", sid) / "scene.npz").exists():
        return False
    farm = farm_root / sid
    farm.mkdir(parents=True, exist_ok=True)
    for stage in _FARM_STAGES:
        src = paths.feature_dir(stage, sid)
        link = farm / stage
        if src.exists() and not link.exists():
            link.symlink_to(src)
    clip = paths.clip_path(row)
    if clip.exists():
        vlink = farm_root / f"{sid}{clip.suffix}"
        if not vlink.exists():
            vlink.symlink_to(clip)
    return True


def _run_scenes(args: argparse.Namespace) -> None:
    """Browse the ClimbingVideos scene DB: select targets, build the symlink farm,
    and serve it through the same :func:`view_3d` a local ``out/`` tree uses."""
    import atexit
    import shutil
    import tempfile

    import scene_db

    data_root = args.data_root or scene_db.DEFAULT_DATA_ROOT
    paths = scene_db.ScenePaths(data_root)
    rope = None if args.rope_supported < 0 else args.rope_supported
    with paths.connect() as conn:
        rows = scene_db.select_targets(
            conn, categories=scene_db.parse_int_csv(args.categories), rope_supported=rope)
    rows = scene_db.resolve_rows(rows, scenes=args.scene_ids, sample=args.sample,
                                 limit=args.limit, seed=args.seed)
    if not rows:
        print("[view_3d] no target scenes match the selection", flush=True)
        return

    farm_root = Path(tempfile.mkdtemp(prefix="bvr_view_"))
    atexit.register(shutil.rmtree, farm_root, ignore_errors=True)  # symlinks only; targets untouched
    n = sum(_link_scene_farm(paths, row, farm_root) for row in rows)
    if not n:
        print(f"[view_3d] none of {len(rows)} target scene(s) are fused yet "
              f"(need geometry/scene.npz) under {data_root}", flush=True)
        return
    print(f"[view_3d] prepared {n}/{len(rows)} scene(s) from {data_root}", flush=True)
    view_3d(farm_root, port=args.port, point_size=args.point_size,
            human_point_size=args.human_point_size, camera_scale=args.camera_scale,
            camera_stride=args.camera_stride, up=args.up, show_smplx=args.smplx,
            device=args.device, human_max_points=args.human_max_points,
            scene_surface=args.scene_surface, video_dir=farm_root)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("predictions", type=Path, nargs="?", default=None,
                   help="root of clips (out/) or a single clip dir (out/<stem>/); "
                        "omit when using --scenes")
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--point-size", type=float, default=0.02,
                   help="scene point radius, metres (default 0.02)")
    p.add_argument("--human-point-size", type=float, default=0.01,
                   help="human point radius, metres (default 0.01)")
    p.add_argument("--camera-scale", type=float, default=0.1,
                   help="initial frustum size, metres (default 0.1; adjust live under 'sizes')")
    p.add_argument("--camera-stride", type=int, default=1, help="show every Nth camera frustum (default 1)")
    p.add_argument("--up", default="-y", choices=["+x", "+y", "+z", "-x", "-y", "-z"],
                   help="viewer up direction (default -y, OpenCV world)")
    p.add_argument("--smplx", action=argparse.BooleanOptionalAction, default=True,
                   help="load the fitted SMPL-X bodies (default: ON; --no-smplx to skip)")
    p.add_argument("--human-max-points", type=int, default=6000,
                   help="cap human depth points per frame+climber for faster loading "
                        "(default 6000; 0 = keep all)")
    p.add_argument("--scene-surface", choices=["points", "bpa", "poisson"], default="points",
                   help="render the scene cloud as points or a reconstructed surface "
                        "(default: points; switch live in the 'layers' panel)")
    p.add_argument("--device", default="cuda",
                   help="device for the SMPL-X forward (default cuda; falls back to cpu)")
    p.add_argument("--video-dir", type=Path, default=None,
                   help="directory of source videos for the sidebar video pane "
                        "(default: the repo's data/; pane hides if <stem>.mp4 is absent)")
    db = p.add_argument_group("ClimbingVideos scene DB (--scenes)")
    db.add_argument("--scenes", action="store_true",
                    help="browse the ClimbingVideos feature DB instead of a local out/ tree")
    db.add_argument("--data-root", type=Path, default=None,
                    help="ClimbingVideos root (default: ../data/ClimbingVideos)")
    db.add_argument("--categories", default="1,2",
                    help="vlm_category filter, CSV (default 1,2 = indoor boulder + rope)")
    db.add_argument("--rope-supported", type=int, default=0,
                    help="vlm_rope_supported filter (default 0; -1 = any)")
    db.add_argument("--scene-ids", default=None, help="restrict to these scene ids (CSV)")
    db.add_argument("--sample", type=int, default=None, help="random subset of N target scenes")
    db.add_argument("--limit", type=int, default=None, help="first N target scenes")
    db.add_argument("--seed", type=int, default=0, help="--sample seed (default 0)")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    if args.scenes:
        _run_scenes(args)
        return 0
    if args.predictions is None:
        p.error("give a predictions dir (out/ or out/<stem>/) or use --scenes")
    view_3d(args.predictions, port=args.port, point_size=args.point_size,
            human_point_size=args.human_point_size, camera_scale=args.camera_scale,
            camera_stride=args.camera_stride, up=args.up, show_smplx=args.smplx,
            device=args.device, human_max_points=args.human_max_points,
            scene_surface=args.scene_surface, video_dir=args.video_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
