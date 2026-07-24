"""Render <cam_left | per-panel |F| curves + moving cursor> checks for climb_wall trials.

The climb_wall force log (`forces.csv`) has no annotated sync indicator, so the
video is placed on the force timeline by wall clock: the OBS filename in the
source tree carries the recording start time (CET, 1 s resolution) and the CSV
rows carry UTC timestamps. The cursor is therefore accurate to about one second.

Curves are per-panel force magnitudes with the tare removed (median of the
log's first second subtracted per panel) — in the tilted trials the sensors
carry kilonewton-scale static structure loads.

Usage:
    uv run python scripts/diagnostics/wall_force_check.py trial_03 trial_10
"""
from __future__ import annotations

import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PETER = Path(__file__).resolve().parents[2] / "peter"
SRC = Path("/data3/rikhat.akizhanov/climb_forces/data/eth/4/videos")
LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]
VIDEO_FPS = 30.0
LOCAL_UTC_OFFSET = 1.0 * 3600  # OBS filenames are CET (UTC+1), CSV timestamps are UTC


def load_forces_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps (T,) unix seconds, force vectors (T, 4, 3) N)."""
    times, vecs = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            times.append(datetime.fromisoformat(row[0].replace("Z", "+00:00")).timestamp())
            vecs.append([float(v) for v in row[2:14]])
    return np.array(times), np.array(vecs).reshape(-1, 4, 3)


def video_start_utc(trial_src: Path) -> float:
    """Recording start (unix seconds) from the OBS filename timestamp."""
    name = next(trial_src.glob("cam_left_*.mkv")).stem
    stamp = re.search(r"(\d{4}-\d{2}-\d{2}) (\d{2})-(\d{2})-(\d{2})", name)
    local = datetime.fromisoformat(f"{stamp.group(1)}T{stamp.group(2)}:{stamp.group(3)}:{stamp.group(4)}")
    return local.replace(tzinfo=timezone.utc).timestamp() - LOCAL_UTC_OFFSET


def render(trial_name: str) -> Path:
    trial = PETER / "climb_wall" / trial_name
    times, vecs = load_forces_csv(trial / "forces.csv")
    rel_time = times - times[0]
    tare = np.median(vecs[rel_time < 1.0], axis=0)
    mag = np.linalg.norm(vecs - tare, axis=2)
    offset = video_start_utc(SRC / f"trial_{int(trial_name.split('_')[1])}") - times[0]

    panel_w, panel_h = 640, 540
    fig, ax = plt.subplots(figsize=(panel_w / 100, panel_h / 100), dpi=100)
    for column, limb in enumerate(LIMBS):
        ax.plot(rel_time, mag[:, column], label=limb, linewidth=1.0)
    ax.set_xlabel("force-log time [s]")
    ax.set_ylabel("|force| [N] (tare-subtracted)")
    ax.set_title(f"{trial_name}  (video starts at {offset:.1f}s, ±1s)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    to_pixel = lambda t: int(round(ax.transData.transform((t, 0))[0]))  # noqa: E731
    plt.close(fig)

    out_path = PETER / "_force_checks" / f"{trial_name}.mp4"
    out_path.parent.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(trial / "cam_left.mp4"))
    writer = imageio.get_writer(out_path, fps=VIDEO_FPS, codec="libx264", quality=7)
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (960, panel_h))[..., ::-1]  # BGR -> RGB
        cursor = panel.copy()
        x = np.clip(to_pixel(offset + frame_index / VIDEO_FPS), 0, panel_w - 1)
        cursor[:, x] = (255, 0, 0)
        writer.append_data(np.hstack([frame, cursor]))
        frame_index += 1
    cap.release()
    writer.close()
    return out_path


if __name__ == "__main__":
    for trial_name in sys.argv[1:]:
        print(f"[wall_check] {trial_name} -> {render(trial_name)}")
