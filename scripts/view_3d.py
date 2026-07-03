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
* the world-frame **gravity** arrow (when the geocalib stage has run), and
* the fitted **SMPL-X bodies** for every tracked person, following the slider.

**Body source** dropdown switches between:

* ``original`` — the raw, pre-optimization SAM-3D-Body → SMPL-X converter output
  (``smplx_params.npz``, folded to the world frame);
* ``kinematics`` / ``kinematics_2`` — the refined world-frame bodies
  (``smplx_kinematic{,_2}.npz``);
* ``dynamics`` — the 52-joint skeleton as torque-scaled spheres + bones + cyan
  contact-force arrows (no mesh).

The three mesh sources are each one skinned mesh per person, posed per frame (LBS
in the browser, one upload) over a single averaged ``betas`` identity.

The sidebar's **layers** checkboxes toggle whole groups on/off — SMPL-X body,
human points, scene points, cameras, gravity — so you never have to open viser's
scene tree. Layer choices persist across example switches. A **sizes** panel scales
the scene points, human points, and camera frustums independently.

The view opens framed on the clip's average human root (so orbiting pivots on the
climber, not the world origin), lit from the camera side against a dark backdrop.

It is a read-only viewer of a finished ``out/<stem>/`` tree: it loads the fused
geometry and fitted bodies and draws them, running no stage and writing nothing.

The viewer package lives under :mod:`tools.viewer`; this file is the thin CLI.

Usage::

    uv run python scripts/view_3d.py out/            # all clips, pick in the UI
    uv run python scripts/view_3d.py out/<stem>/     # a single clip
    uv run python scripts/view_3d.py out/ [--port 8082] [--no-smplx]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.viewer import view_3d


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("predictions", type=Path,
                   help="root of clips (out/) or a single clip dir (out/<stem>/)")
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
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    view_3d(args.predictions, port=args.port, point_size=args.point_size,
            human_point_size=args.human_point_size, camera_scale=args.camera_scale,
            camera_stride=args.camera_stride, up=args.up, show_smplx=args.smplx,
            device=args.device, human_max_points=args.human_max_points,
            scene_surface=args.scene_surface)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
