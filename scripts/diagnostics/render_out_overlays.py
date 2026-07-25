"""Draw the contact and force predictions on the ``out/<scene>/`` example videos.

These scenes are internet climbing clips, so there is no force plate to compare
against — the overlay carries the model output alone, with no plots and no
numbers. For each of the four contact limbs every frame gets

* a **disc** at the contact anchor, red where the contact model says contact and
  green where it says free (at the threshold stored in the file), and
* an **arrow** from that anchor along the force model's force.

The arrow is a real 3D vector, not a 2D doodle: the force is rotated into the
camera frame with the file's own axis flip, added to the anchor in camera
coordinates, and both ends are projected, so it shortens with distance. The
pinhole camera comes out of the prediction file itself, which stores every anchor
twice — once in camera coordinates and once in pixels — and that pins
``(fx, fy, cx, cy)`` down by least squares. So the arrows need no external
calibration and cannot disagree with the model's own geometry.

Frames keep their source resolution, because these scenes run from 480x854
portrait to 1920x1080 and a fixed output size would squash half of them. The
stored pixel anchors are therefore used exactly as saved, and the disc and arrow
sizes scale with the frame instead.

Everyone tracked in the scene is drawn — some of these clips hold two climbers.

An anchor can land outside the frame when the body fit floats away from the
image; those discs simply do not appear, which is the honest picture.

Predictions cover a prefix of the video (``frame_indices`` is contiguous from 0),
so prediction ``i`` is video frame ``i``.

Usage:
    uv run python scripts/diagnostics/render_out_overlays.py
    uv run python scripts/diagnostics/render_out_overlays.py boulder_1 sport_climb
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

OUT_ROOT = Path(__file__).resolve().parents[2] / "out"

CONTACT_RGB = (225, 55, 45)
FREE_RGB = (45, 185, 95)
WHITE = (255, 255, 255)

ARROW_METRES_PER_BW = 0.35  # how long one body weight of force is drawn
MIN_FORCE_BW = 0.02  # below this the force is noise, so no arrow


def _intrinsics(anchor_cam: np.ndarray, anchor_2d: np.ndarray) -> tuple[float, float, float, float]:
    """Recover ``(fx, fy, cx, cy)`` from anchors stored in both metres and pixels.

    ``anchor_cam: (N, 4, 3)`` camera coordinates, ``anchor_2d: (N, 4, 2)`` pixels.
    """
    points, pixels = anchor_cam.reshape(-1, 3), anchor_2d.reshape(-1, 2)
    keep = np.isfinite(points).all(1) & np.isfinite(pixels).all(1) & (points[:, 2] > 1e-3)
    points, pixels = points[keep], pixels[keep]
    design = np.zeros((2 * len(points), 4))
    design[0::2, 0], design[0::2, 2] = points[:, 0] / points[:, 2], 1.0
    design[1::2, 1], design[1::2, 3] = points[:, 1] / points[:, 2], 1.0
    fx, fy, cx, cy = np.linalg.lstsq(design, pixels.reshape(-1), rcond=None)[0]
    return float(fx), float(fy), float(cx), float(cy)


def _draw_caption(frame: np.ndarray, lines: list[str], unit: int) -> None:
    """Write a translucent caption box into the top-left corner, in place."""
    font, scale, weight = cv2.FONT_HERSHEY_SIMPLEX, unit / 16, max(1, unit // 6)
    row, pad = int(unit * 2.1), unit
    widths = [cv2.getTextSize(text, font, scale, weight)[0][0] for text in lines]
    box = frame[pad:pad + 2 * pad + row * len(lines), pad:pad + 2 * pad + max(widths)]
    box[:] = (0.35 * box + 0.65 * np.array([25, 25, 25])).astype(np.uint8)
    for index, text in enumerate(lines):
        cv2.putText(frame, text, (2 * pad, 2 * pad + row * index + row // 2), font, scale,
                    (245, 245, 245), weight, cv2.LINE_AA)


def _draw_limbs(frame: np.ndarray, anchor_2d: np.ndarray, anchor_cam: np.ndarray,
                force: np.ndarray, contact: np.ndarray, flip: np.ndarray,
                camera: tuple[float, float, float, float], unit: int) -> None:
    """Stamp one frame's four discs and force arrows, in place.

    All per-limb arrays are this frame's alone: ``anchor_2d (4, 2)``,
    ``anchor_cam (4, 3)``, ``force (4, 3)`` in body weights, ``contact (4,)``.
    """
    fx, fy, cx, cy = camera
    force_cam = force * flip
    for limb in range(4):
        if not np.isfinite(anchor_2d[limb]).all():
            continue
        centre = (int(round(anchor_2d[limb][0])), int(round(anchor_2d[limb][1])))
        start = anchor_cam[limb]
        if start[2] > 1e-3 and np.linalg.norm(force[limb]) >= MIN_FORCE_BW:
            end = start + force_cam[limb] * ARROW_METRES_PER_BW
            if end[2] > 1e-3:
                head = (int(round(fx * end[0] / end[2] + cx)), int(round(fy * end[1] / end[2] + cy)))
                cv2.arrowedLine(frame, centre, head, WHITE, max(2, unit // 3) + 2,
                                cv2.LINE_AA, tipLength=0.22)
                cv2.arrowedLine(frame, centre, head, CONTACT_RGB, max(2, unit // 3),
                                cv2.LINE_AA, tipLength=0.22)
        cv2.circle(frame, centre, unit + 2, WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, centre, unit, CONTACT_RGB if contact[limb] else FREE_RGB, -1, cv2.LINE_AA)


def render(scene: str) -> Path:
    """Write one scene's overlay next to its predictions and return the path."""
    pred_dir = OUT_ROOT / scene / "predictions"
    contacts = np.load(pred_dir / "contacts.npz", allow_pickle=True)
    forces = np.load(pred_dir / "forces.npz", allow_pickle=True)

    # Everyone tracked in the scene is drawn; the leading axis is the person.
    contact = contacts["contacts"]  # (P, N, 4) bool, already at the file's threshold
    tracked = contacts["valid_mask"]  # (P, N)
    anchor_2d = contacts["anchor_points_2d"]  # (P, N, 4, 2) source pixels
    force = forces["forces"]  # (P, N, 4, 3) body weights, local-world-aligned
    anchor_cam = forces["anchor_cam"]  # (P, N, 4, 3) metres in the camera frame
    flip = forces["lwa_to_opencv_cam_flip"]
    # One camera filmed them all, so every person's anchors constrain the same fit.
    camera = _intrinsics(anchor_cam, forces["anchor_points_2d"])
    fps = float(contacts["fps"])

    video = Path(str(contacts["source_video"]))
    if not video.is_file():
        raise SystemExit(f"[overlays] {scene}: source video is gone: {video}")
    capture = cv2.VideoCapture(str(video))
    unit = max(3, round(min(capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                            capture.get(cv2.CAP_PROP_FRAME_WIDTH)) / 90))
    caption = [f"disc: red = contact, green = free  (p > {float(contacts['threshold']):.2f})",
               f"arrow: force, {ARROW_METRES_PER_BW} m per body weight"]

    out_path = pred_dir / "predictions_overlay.mp4"
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7,
                               macro_block_size=1)
    for index in tqdm(range(contact.shape[1]), desc=f"[overlays] {scene}", leave=False):
        ok, bgr = capture.read()
        if not ok:
            break
        frame = bgr[..., ::-1].copy()
        for person in range(contact.shape[0]):
            if tracked[person, index]:
                _draw_limbs(frame, anchor_2d[person, index], anchor_cam[person, index],
                            force[person, index], contact[person, index], flip, camera, unit)
        _draw_caption(frame, caption, unit)
        writer.append_data(frame)
    capture.release()
    writer.close()
    return out_path


if __name__ == "__main__":
    scenes = sys.argv[1:] or sorted(p.name for p in OUT_ROOT.iterdir()
                                    if (p / "predictions" / "contacts.npz").exists())
    for name in scenes:
        try:
            print(f"[overlays] {name} -> {render(name)}", flush=True)
        except Exception as error:  # noqa: BLE001 — one bad scene must not kill the batch
            print(f"[overlays] {name} FAILED: {error}", flush=True)
