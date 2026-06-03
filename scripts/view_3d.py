"""View the fusion geometry (``geometry/``) in a viser web viewer.

Loads ``out/<stem>/geometry/`` — the static **scene** point cloud, the per-frame
**camera** frustums + centre path (from ``transform.npz``), and the per-frame
**human** points split by climber (``obj_id``). The human is stored per-frame and
deliberately unmerged (it moves), so it is shown one frame at a time via a
slider / play control; the scene and cameras are static and shown in full.

This is the geometry counterpart to ``view_motion.py`` (which plays the SMPL
motion ``results/*.npz``); the two are kept separate.

Usage::

    uv run python scripts/view_3d.py out/<stem>/ [--port 8082]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _camera_world_pose(extrinsic: np.ndarray):
    """cam-from-world OpenCV ``[R|t]`` → ``(world position, world-from-cam wxyz)``."""
    import viser.transforms as vt

    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    return -R.T @ t, vt.SO3.from_matrix(R.T).wxyz


def view_3d(pred: Path, *, port: int = 8082, point_size: float = 0.02,
            camera_scale: float = 0.15, camera_stride: int = 1, up: str = "-y") -> None:
    import viser

    geo = pred / "geometry"
    if not (geo / "scene.npz").exists():
        raise FileNotFoundError(f"no fusion output under {geo} — run the 'fuse' stage first")
    transform = np.load(geo / "transform.npz")
    scene = np.load(geo / "scene.npz")
    human = np.load(geo / "human.npz")

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction(up)  # VGGT world is OpenCV y-down → up ≈ -y
    server.scene.add_point_cloud("/scene", scene["points"], scene["colors"], point_size=point_size)

    # Cameras — a frustum per (strided) frame + the full centre path.
    extr = transform["extrinsics"]
    aspect = float(transform["image_width"]) / float(transform["image_height"])
    fov_y = np.radians(transform["fov_y_deg"].astype(np.float64))
    for k in range(0, len(extr), camera_stride):
        pos, wxyz = _camera_world_pose(extr[k])
        server.scene.add_camera_frustum(
            f"/cameras/cam_{k:04d}", fov=float(fov_y[k]), aspect=aspect,
            scale=camera_scale, color=(90, 160, 250), wxyz=wxyz, position=pos)
    centers = np.stack([_camera_world_pose(e)[0] for e in extr]).astype(np.float32)
    server.scene.add_spline_catmull_rom("/cameras/path", centers, color=(255, 140, 0), line_width=2.0)

    # Human — per-frame, split by climber (each its own toggleable node). Unmerged
    # across frames, so show one frame at a time; a play loop animates it.
    pts, col = human["points"], human["colors"]
    fid, obj = human["frame_id"], human["obj_id"]
    obj_ids = [int(o) for o in np.unique(obj)]
    n_frames = len(extr)
    handles: dict[str, object] = {}

    def show(f: int) -> None:
        on = fid == f
        for o in obj_ids:
            sel = on & (obj == o)
            name = f"/human/climber_{o:02d}"
            if sel.any():
                handles[name] = server.scene.add_point_cloud(
                    name, pts[sel], col[sel], point_size=point_size)
            elif name in handles:
                handles[name].visible = False  # type: ignore[attr-defined]

    frame = play = play_fps = None
    if len(pts):
        frame = server.gui.add_slider("human frame", 0, n_frames - 1, 1, 0)
        play = server.gui.add_checkbox("play", False)
        play_fps = server.gui.add_slider("play fps", 1, 60, 1, 30)
        frame.on_update(lambda _: show(int(frame.value)))
        show(0)

    print(f"[view_3d] {pred.name}: scene {len(scene['points'])} pts | {len(extr)} cameras | "
          f"human {len(pts)} pts, climbers {obj_ids} | http://localhost:{port}")

    while True:
        if play is not None and play.value:
            frame.value = (int(frame.value) + 1) % n_frames
            show(int(frame.value))
            time.sleep(1.0 / float(play_fps.value))
        else:
            time.sleep(0.1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("predictions", type=Path, help="clip dir, e.g. out/<stem>/")
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--point-size", type=float, default=0.02, help="point radius, metres (default 0.02)")
    p.add_argument("--camera-scale", type=float, default=0.15, help="frustum size, metres (default 0.15)")
    p.add_argument("--camera-stride", type=int, default=1, help="show every Nth camera frustum (default 1)")
    p.add_argument("--up", default="-y", choices=["+x", "+y", "+z", "-x", "-y", "-z"],
                   help="viewer up direction (default -y, OpenCV world)")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    view_3d(args.predictions, port=args.port, point_size=args.point_size,
            camera_scale=args.camera_scale, camera_stride=args.camera_stride, up=args.up)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
