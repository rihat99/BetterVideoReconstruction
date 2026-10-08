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
* the per-frame **camera** frustums + centre path — or, for a clip filmed by a
  calibrated rig (``cameras.npz``), one named cyan frustum per fixed camera,
* the world-frame **gravity** arrow (when the geocalib stage has run),
* the **ground plane** the gravity stage fitted and accepted (drawn from
  ``geocalib/gravity.npz``; absent when the floor was not visible) — a translucent
  grey fill under grey half-metre cells, so it reads as a floor you can still see
  the scene through, and
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
  force/torque data;
* ``mhr_1_sup`` — the kindyn body re-fitted onto Meta's MHR rig
  (``human_optim/mhr_1.npz``, the pose supervision the contact model trains on).
  Mesh only: no contact frames and no forces are drawn on it.

The dynamics folder's **sensor forces (measured)** toggle draws the wall rig's own
reading — the measured reaction on each limb, in orange, at that limb's joint and on
the same scale as the solved red arrows — for the clips that ship a
``sensor_forces.npz``. When that file is a force plate's (the OpenCap
LabValidation clips) the arrow starts at the measured centre of pressure on the
floor, and the **estimated forces (plate format)** toggle draws our own foot forces
collapsed to the same force-at-centre-of-pressure form (``estimated_forces.npz``,
``scripts/export_opencap_mot.py``) in blue beside them — measured orange vs ours
blue, same scale, same floor. Its **force display** dropdown picks the solved arrows' regime:
``per-joint`` accumulates each joint's frame forces into one arrow at the joint —
each hand's forces summed at its wrist, never on individual finger joints —
``per-frame`` draws every contact frame's own solved force where it acts on the
skin (kindyn sources, which store ``frame_forces``). Joint spheres stay the
neutral skeleton grey; the ``joint torques`` toggle turns them yellow, sized by
torque magnitude.

The three mesh sources are each one skinned mesh per person, posed per frame (LBS
in the browser, one upload) over a single averaged ``betas`` identity.

A **video** pane in the sidebar plays the clip's original footage in sync with the
frame slider (loaded from ``../data/examples/data/`` by default, or ``[viewer].video_dir``;
hidden when the source video is absent). A multi-view clip plays its own
``views/<name>/video.mp4`` instead and gains a **view** dropdown to choose the
camera; each view is shifted by the lag ``cameras.npz`` measured for it, so every
view shows the same instant as the frame slider. ``[viewer].video = false`` disables the
pane entirely — no decode, nothing streamed — which keeps playback smooth over a
remote connection. The playback folder's **onboard camera**
toggle (off by default) rides the reconstructed camera — gliding along its
trajectory (pose and field of view interpolated between frames, updated at 60 Hz
independently of how fast the scene is posed) and locking out manual orbiting — so
you watch from the filming camera's own viewpoint. Playback itself runs on the wall
clock at the chosen fps: a frame the machine cannot pose in time is skipped, never
stretched, so motion keeps the video's real speed.

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
``--scenes athletics`` browses the AthleticsVideos DB instead; its sidebar gains a
**sport** dropdown (all / floor / beam / vault / pommel / high_jump / triple_jump)
that narrows the scene list.

**BEDLAM2** — ``--bedlam`` browses the synthetic ground-truth set at
``../data/BEDLAM2_our`` the same way. BEDLAM2 ships finished motion, so there is no
stage chain and no body-source choice: one **ground truth** source, the cameras, the
ground plane the renderer stood everyone on, and its gravity. The environments ship
no geometry, so there are no scene points and no human depth points, and those
layers are hidden. Where ``scripts/label_bedlam2.py`` has run, the **contact frames**
layer draws its ground-contact labels (``gt/contacts.npz``) red for touching the floor
and green for free, and the people it found no usable frame for (``gt/bboxes.npz``)
are left out of the scene entirely, and the forces ``--step forces`` solved
(``gt/forces.npz``) are drawn on that body the way ``kindyn_1``'s are.
The video pane is on (the footage is what you check the bodies against), and all
18 076 scenes are listed unless ``[viewer].scene_ids`` / ``sample`` / ``limit`` names
a set — the pager and the dropdown load a scene only when it is selected.

The viewer package lives under :mod:`tools.viewer`; this file is the thin CLI.

Usage::

    uv run python scripts/view_3d.py out/            # all clips, pick in the UI
    uv run python scripts/view_3d.py out/<stem>/     # a single clip
    uv run python scripts/view_3d.py out/ --config configs/pipeline.toml
    uv run python scripts/view_3d.py --scenes        # browse the ClimbingVideos DB
    uv run python scripts/view_3d.py --scenes athletics  # AthleticsVideos DB, sport dropdown
    uv run python scripts/view_3d.py --bedlam        # BEDLAM2 ground truth, all 18 076 scenes
    # The climb_wall_3 ground-truth clips are ordinary trees — point at one, or at
    # their root to get the example dropdown:
    uv run python scripts/view_3d.py ../data/peter/climb_wall_3_gt/human/
    # Set viewer.sample = 50 in the config for a random 50-scene subset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scene_db  # noqa: E402  (scripts/ is this file's own directory)
from tools.config import load_toml, section
from tools.viewer import view_3d

# The stage feature dirs the viewer's loaders read (sapiens is stage-only, so it is
# not linked). Each is aliased as ``<farm>/<stage>`` → the scene's feature tree.
_FARM_STAGES = ("geometry", "sam3d", "vggt", "human_optim", "geocalib")

def _link_scene_farm(paths, row, farm_root: Path, *,
                     stages: tuple[str, ...] = _FARM_STAGES,
                     needs: tuple[str, str] = ("geometry", "scene.npz")) -> bool:
    """Alias one DB scene's feature dirs into an ``out/<stem>/``-shaped symlink farm.

    Builds ``<farm_root>/<scene_id>/<stage>`` symlinks into the scene's stage-first
    feature trees (skipping stages that were never produced), plus the source clip
    as ``<farm_root>/<scene_id><ext>`` for the sidebar video pane. ``needs`` is the
    ``(stage, file)`` the viewer cannot draw without — ``geometry/scene.npz`` for a
    fused clip, ``gt/smplx.npz`` for a BEDLAM2 scene; a scene missing it is skipped.
    """
    sid = row["scene_id"]
    if not (paths.feature_dir(needs[0], sid) / needs[1]).exists():
        return False
    farm = farm_root / sid
    farm.mkdir(parents=True, exist_ok=True)
    for stage in stages:
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

    data_root = args.data_root or scene_db.DEFAULT_DATA_ROOT
    categories = scene_db.parse_int_csv(args.categories)
    rope = None if args.rope_supported < 0 else args.rope_supported
    groups = None
    stages, needs = _FARM_STAGES, ("geometry", "scene.npz")
    if args.scenes == "athletics":  # every sport; the viewer's sport dropdown narrows it
        data_root = args.data_root or scene_db.ATHLETICS_DATA_ROOT
        categories, rope = (), None
    elif args.scenes == "bedlam":   # one feature stage, and every scene is a target
        data_root = args.data_root or scene_db.BEDLAM_DATA_ROOT
        categories, rope = (), None
        stages, needs = ("gt",), ("gt", "smplx.npz")
    paths = scene_db.ScenePaths(data_root)
    with paths.connect() as conn:
        rows = scene_db.select_targets(conn, categories=categories, rope_supported=rope)
    if args.scenes == "athletics":
        sport = {v: k for k, v in scene_db.ATHLETICS_SPORTS.items()}
        groups = {r["scene_id"]: sport.get(r["vlm_category"], "other") for r in rows}
    rows = scene_db.resolve_rows(rows, scenes=args.scene_ids, sample=args.sample,
                                 limit=args.limit, seed=args.seed)
    if not rows:
        print("[view_3d] no target scenes match the selection", flush=True)
        return

    farm_root = Path(tempfile.mkdtemp(prefix="bvr_view_"))
    atexit.register(shutil.rmtree, farm_root, ignore_errors=True)  # symlinks only; targets untouched
    n = sum(_link_scene_farm(paths, row, farm_root, stages=stages, needs=needs)
            for row in rows)
    if not n:
        print(f"[view_3d] none of {len(rows)} target scene(s) have "
              f"{needs[0]}/{needs[1]} under {data_root}", flush=True)
        return
    print(f"[view_3d] prepared {n}/{len(rows)} scene(s) from {data_root}", flush=True)
    view_3d(farm_root, port=args.port, point_size=args.point_size,
            human_point_size=args.human_point_size, camera_scale=args.camera_scale,
            camera_stride=args.camera_stride, up=args.up, show_smplx=args.smplx,
            device=args.device, human_max_points=args.human_max_points,
            video=args.video, video_dir=farm_root, groups=groups)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("target", type=Path, nargs="?", default=None,
                   help="root of clips (out/) or a single clip dir (out/<stem>/); "
                        "omit when using --scenes")
    p.add_argument("--scenes", nargs="?", const="climbing", choices=("climbing", "athletics"),
                   help="browse a scene DB instead of a local out/ tree: the ClimbingVideos "
                        "DB (default) or the AthleticsVideos DB (sport dropdown in the sidebar)")
    p.add_argument("--bedlam", action="store_true",
                   help="browse the BEDLAM2 ground-truth DB: cameras, ground plane and "
                        "bodies, with the video pane on; every scene unless "
                        "[viewer].scene_ids / sample / limit picks a set")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "pipeline.toml",
                   help="pipeline TOML (default: configs/pipeline.toml)")
    cli = p.parse_args(sys.argv[1:] if argv is None else argv)
    if cli.bedlam:
        if cli.scenes:
            p.error("--bedlam and --scenes pick different DBs; use one")
        cli.scenes = "bedlam"
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
    if cli.bedlam:
        # The video pane is half of what you check a BEDLAM2 scene against, so it is
        # on here whatever the config says (it defaults to off for remote sessions).
        args.video = True

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
