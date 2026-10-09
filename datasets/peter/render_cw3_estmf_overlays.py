"""Draw the Li et al. estimator's climb_wall_3 answer on the video it was given.

One video per clip: the estimator's 24-joint body, its contact forces as **green**
arrows at the points it applies them, the force boards' measured forces as **red**
arrows at the same points, and two planes -- **cyan** for the wall stage 3 fitted,
**grey outline** for the wall the rig measured.

Nothing has to be registered here. The estimator solves in its camera frame, that
camera is cam_left, and cam_left's frame is the rig's world, so the prediction, the
boards and the wall are already in one frame and the calibrated intrinsics project
all three.

Writes ``peter/climb_wall_3_gt/others/estmf/videos/<clip>.mp4``.

Usage:
    uv run --no-sync python datasets/peter/render_cw3_estmf_overlays.py
    uv run --no-sync python datasets/peter/render_cw3_estmf_overlays.py --clips a_00_01_part3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


from datasets.methods.paths import ESTMF
from datasets.peter.paths import PETER

ESTIMATOR = ESTMF
PREDICTIONS = ESTIMATOR / "validation" / "results" / "climb_wall_3" / "run" / "evaluation"
GT = PETER / "climb_wall_3_gt" / "human"
VIDEOS = PETER / "climb_wall_3"
SINGLE = PETER / "out_climb_wall_3_single"
OUT_DIR = PETER / "climb_wall_3_gt" / "others" / "estmf" / "videos"

DEFAULT_CLIPS = ["a_00_01_part3", "b_00_02_part2", "a_16_01_part2", "b_32_02_part2"]

#: The parent of each SMPL joint, for the bones of the skeleton.
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
#: The estimator's force channels, and the board channel that measures each.
CHANNELS = ["l_sole", "r_sole", "l_hand", "r_hand"]
BOARD_OF_CHANNEL = ["left_foot", "right_foot", "left_hand", "right_hand"]
#: How long a newton is drawn: 500 N is 30 cm in the world.
METRES_PER_NEWTON = 0.30 / 500.0
#: Anything smaller than this is not drawn at all.
MIN_DRAWN_N = 20.0

MEASURED_RGB = (220, 50, 40)
SOLVED_RGB = (60, 200, 90)
BODY_RGB = (250, 210, 60)
FIT_PLANE_RGB = (70, 200, 230)
RIG_PLANE_RGB = (170, 170, 170)


def project(points: np.ndarray, intrinsics: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    """``(..., 3)`` camera points -> integer pixels in the resized pane."""
    depth = np.maximum(points[..., 2], 1e-3)
    x = (intrinsics[0, 0] * points[..., 0] / depth + intrinsics[0, 2]) * scale[0]
    y = (intrinsics[1, 1] * points[..., 1] / depth + intrinsics[1, 2]) * scale[1]
    return np.stack([x, y], -1).round().astype(np.int32)


def plane_corners(point: np.ndarray, normal: np.ndarray, half_size: float = 0.8) -> np.ndarray:
    """Four corners of a square patch of the plane, centred on ``point``."""
    normal = normal / np.linalg.norm(normal)
    seed = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return np.array([point + half_size * (su * u + sv * v)
                     for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def draw_plane(pane: np.ndarray, corners: np.ndarray, intrinsics, scale,
               colour, fill: bool) -> None:
    if (corners[:, 2] <= 0.1).any():
        return
    polygon = project(corners, intrinsics, scale)
    if fill:
        patch = pane.copy()
        cv2.fillConvexPoly(patch, polygon, colour)
        cv2.addWeighted(patch, 0.22, pane, 0.78, 0.0, pane)
    cv2.polylines(pane, [polygon], True, colour, 2, cv2.LINE_AA)


def draw_arrow(pane, origin, force, intrinsics, scale, colour, thickness=3) -> None:
    size = float(np.linalg.norm(force))
    if size < MIN_DRAWN_N or origin[2] <= 0.1:
        return
    tip = origin + force * METRES_PER_NEWTON
    if tip[2] <= 0.1:
        return
    start = project(origin, intrinsics, scale)
    end = project(tip, intrinsics, scale)
    if np.linalg.norm(end - start) < 2:
        return
    cv2.arrowedLine(pane, tuple(start), tuple(end), colour, thickness,
                    cv2.LINE_AA, tipLength=0.25)


def draw_legend(pane: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    for row, (text, colour) in enumerate(lines):
        origin = (14, 26 + 24 * row)
        cv2.putText(pane, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(pane, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)


def render(clip: str, save_name: str, width: int) -> Path:
    prediction = np.load(PREDICTIONS / f"{save_name}_s7" / f"{clip}_pred.npz")
    wall = np.load(GT / clip / "geometry" / "wall.npz")
    sensor = np.load(SINGLE / clip / "human_optim" / "sensor_forces.npz")
    intrinsics = np.asarray(np.load(GT / clip / "cameras.npz", allow_pickle=True)["K"][0], float)

    joints = prediction["joints_24"]
    forces = prediction["forces_world"]
    points = prediction["force_points_world"]
    limbs = [str(x) for x in sensor["limbs"]]
    board = np.asarray(sensor["reaction_world"], float)[:, [limbs.index(n) for n in BOARD_OF_CHANNEL]]
    frames = min(len(joints), len(board))
    fitted = plane_corners(prediction["ground_plane_point"], prediction["ground_plane_normal"])
    measured_plane = plane_corners(np.asarray(wall["point"], float),
                                   np.asarray(wall["normal"], float))

    capture = cv2.VideoCapture(str(VIDEOS / clip / "cam_left.mp4"))
    source = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pane_size = (width, int(round(width * source[1] / source[0])))
    scale = (pane_size[0] / source[0], pane_size[1] / source[1])
    angle = np.degrees(np.arccos(np.clip(abs(
        prediction["ground_plane_normal"].dot(np.asarray(wall["normal"], float))), -1, 1)))
    legend = [(f"{clip}  --  Li et al. (CVPR'19), our front end", (255, 255, 255)),
              ("measured board force", MEASURED_RGB),
              ("estimator force", SOLVED_RGB),
              (f"fitted wall plane ({angle:.0f} deg off the rig's)", FIT_PLANE_RGB),
              ("wall measured by the rig", RIG_PLANE_RGB)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{clip}.mp4"
    writer = imageio.get_writer(path, fps=float(prediction["fps"]), codec="libx264", quality=7)
    for index in tqdm(range(frames), desc=f"[estmf overlay] {clip}", leave=False):
        ok, bgr = capture.read()
        if not ok:
            break
        pane = cv2.resize(bgr, pane_size)[..., ::-1].copy()
        draw_plane(pane, measured_plane, intrinsics, scale, RIG_PLANE_RGB, fill=False)
        draw_plane(pane, fitted, intrinsics, scale, FIT_PLANE_RGB, fill=True)
        pixels = project(joints[index], intrinsics, scale)
        for child, parent in enumerate(SMPL_PARENTS):
            if parent >= 0:
                cv2.line(pane, tuple(pixels[parent]), tuple(pixels[child]), BODY_RGB, 2, cv2.LINE_AA)
        for pixel in pixels:
            cv2.circle(pane, tuple(pixel), 4, BODY_RGB, -1, cv2.LINE_AA)
        for channel in range(4):
            origin = points[index, channel]
            draw_arrow(pane, origin, board[index, channel], intrinsics, scale, MEASURED_RGB, 2)
            draw_arrow(pane, origin, forces[index, channel], intrinsics, scale, SOLVED_RGB, 3)
        draw_legend(pane, legend)
        writer.append_data(pane)
    capture.release()
    writer.close()
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", default=",".join(DEFAULT_CLIPS),
                    help="comma-separated clip names")
    ap.add_argument("--save-name", default="ours", help="the estimator run to draw")
    ap.add_argument("--width", type=int, default=1280, help="output width in pixels")
    args = ap.parse_args(argv)

    for clip in args.clips.split(","):
        print(render(clip.strip(), args.save_name, args.width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
