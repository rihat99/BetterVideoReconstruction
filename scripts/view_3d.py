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
* the **ground plane** the gravity stage fitted and accepted (drawn from
  ``geocalib/gravity.npz``; absent when the floor was not visible), and
* the fitted **SMPL-X bodies** for every tracked person, following the slider.

**Body source** dropdown switches between:

* ``original`` — the raw, pre-optimization SAM-3D-Body → SMPL-X converter output
  (``smplx_params.npz``, folded to the world frame);
* ``kinematics`` / ``kinematics_2`` — the refined world-frame bodies
  (``human_optim/kinematic_{1,2}.npz``); both draw the 52-joint skeleton overlay
  (finger joints small, because the contact frames hang off them) and the
  **contact frames** layer: the 12 named skin points — toes, heels, knees, palms,
  fingertips — that contact is detected on, each shown as a sphere (red touching
  the scene / green free, from its own contact artifact ``contacts_1`` /
  ``contacts_2``), a thin link to the joint it rides on, and a short arrow along
  its outward normal;
* ``kindyn_1`` / ``kindyn_manual`` — the joint kinematics+dynamics bodies
  (``human_optim/kindyn_{1,manual}.npz``; ``kindyn_manual`` is the
  manual-annotation reference chain), skinned meshes that also carry
  force/torque data.

The dynamics folder's **force display** dropdown picks the arrow regime:
``per-joint`` accumulates each joint's frame forces into one arrow at the joint
(the classic view), ``per-frame`` draws every contact frame's own solved force
where it acts on the skin (kindyn sources, which store ``frame_forces``).

The three mesh sources are each one skinned mesh per person, posed per frame (LBS
in the browser, one upload) over a single averaged ``betas`` identity.

A **video** pane in the sidebar plays the clip's original footage in sync with the
frame slider (loaded from the repo's ``data/`` by default, or ``[viewer].video_dir``;
hidden when the source video is absent). ``[viewer].video = false`` disables the
pane entirely — no decode, nothing streamed — which keeps playback smooth over a
remote connection. The playback folder's **onboard camera**
toggle (off by default) rides the reconstructed camera — matching its pose and
field of view each frame and locking out manual orbiting — so you watch from the
filming camera's own viewpoint.

The sidebar's **layers** checkboxes toggle whole groups on/off — SMPL-X body, body
skeleton, contact frames, human points, scene points, scene normals, cameras,
gravity — so you never have to open viser's scene tree. Checkboxes that do not
apply to the active body source are hidden (contact frames need a body with a
contact artifact). Layer choices persist across example switches. A **sizes** panel
scales the scene points, human points, and camera frustums independently; the mesh
turns translucent while a skeleton or the contact frames overlay it.

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
    uv run python scripts/view_3d.py out/ --config configs/pipeline.toml
    uv run python scripts/view_3d.py --scenes        # browse the ClimbingVideos DB
    # Set viewer.sample = 50 in the config for a random 50-scene subset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.config import load_toml, section
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
            video=args.video, video_dir=farm_root)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("target", type=Path, nargs="?", default=None,
                   help="root of clips (out/) or a single clip dir (out/<stem>/); "
                        "omit when using --scenes")
    p.add_argument("--scenes", action="store_true",
                   help="browse the ClimbingVideos feature DB instead of a local out/ tree")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "pipeline.toml",
                   help="pipeline TOML (default: configs/pipeline.toml)")
    cli = p.parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_toml(cli.config)
    args = section(cfg, "viewer", {
        "port": 8082,
        "point_size": 0.02,
        "human_point_size": 0.01,
        "camera_scale": 0.1,
        "camera_stride": 1,
        "up": "-y",
        "smplx": True,
        "human_max_points": 6000,
        "video": True,
        "video_dir": "",
        "data_root": "",
        "categories": "1,2",
        "rope_supported": 0,
        "scene_ids": "",
        "sample": "",
        "limit": "",
        "seed": 0,
    })
    args.device = section(cfg, "runtime", {"device": "cuda"}).device
    args.target, args.scenes, args.config = cli.target, cli.scenes, cli.config
    args.video_dir = Path(args.video_dir) if args.video_dir else None
    args.data_root = Path(args.data_root) if args.data_root else None
    args.scene_ids = args.scene_ids or None
    args.sample = int(args.sample) if args.sample != "" else None
    args.limit = int(args.limit) if args.limit != "" else None

    if args.scenes:
        _run_scenes(args)
        return 0
    if args.target is None:
        p.error("give a predictions dir (out/ or out/<stem>/) or use --scenes")
    view_3d(args.target, port=args.port, point_size=args.point_size,
            human_point_size=args.human_point_size, camera_scale=args.camera_scale,
            camera_stride=args.camera_stride, up=args.up, show_smplx=args.smplx,
            device=args.device, human_max_points=args.human_max_points,
            video=args.video, video_dir=args.video_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
