"""Write world-frame wall-sensor forces into the peter climb_wall out trees.

The climb_wall rig records four 3-axis load cells at ~98.85 Hz
(``peter/climb_wall/<clip>/forces.npz``, one channel per limb name). The cells
are the SUPPORTS of one shared structure — a force applied anywhere shows up on
all four (verified on the first-touch events, where a left-hand-only touch
registers up to 260 N on the "right foot" cell). So a single channel is a
support reaction, not that limb's contact force; only the channel SUM is a
clean physical quantity (the total force the climber applies to the wall).

One fixed rotation maps the sensor frame to the reconstruction world frame
(world = cam_left, identical in every clip). It was fit once by Kabsch between
the per-video-frame channel sum and the kindyn_1 predicted total wall force,
pooled over the 8 clips where the sensor total is trustworthy — and the SAME
rotation fits the tilted-wall trials (01-09) to within 5 degrees: the cells
sit in the rig's base and do NOT tilt with the climbing surface. Median
sensor-vs-predicted total angle on the calibration clips: 11-15 degrees,
magnitude ratio 0.86-1.14. Fit script: scratchpad ``wall_total_fit.py``
(session bafd5529, 2026-07-26).

Output ``<tree>/<clip>/human_optim/sensor_forces.npz`` per video frame, in the
REACTION sense (force on the climber, comparable to the predicted contact
forces): ``reaction_world (N, 4, 3)`` per channel and ``total_world (N, 3)``.

Usage:
    uv run python scripts/peter_sensor_forces.py [--tree out_climb_wall_single]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Sensor frame -> world (cam_left) rotation, columns = world images of the
# sensor x/y/z axes. Data-derived constant (see module docstring).
R_SENSOR_TO_WORLD = np.array([
    [-0.9801, -0.1768, +0.0907],
    [-0.0956, +0.8197, +0.5647],
    [-0.1742, +0.5448, -0.8203],
], dtype=np.float32)

CALIB_CLIPS = ["trial_10", "trial_11", "trial_12", "trial_13", "trial_14_part1",
               "trial_01", "trial_02", "trial_03"]
NOTE = ("channels are support-cell reactions of one shared structure, not per-limb "
        "contact forces; only total_world is a clean physical force")


def per_frame(forces: np.ndarray, ratio: float, n_frames: int) -> np.ndarray:
    """Average the ~98.85 Hz samples into per-video-frame values, ``(N, 4, 3)``."""
    return np.stack([forces[int(i * ratio):max(int((i + 1) * ratio), int(i * ratio) + 1)].mean(0)
                     for i in range(n_frames)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--tree", default="out_climb_wall_single",
                    help="out tree under peter/ to write into")
    args = ap.parse_args()

    tree = REPO / "peter" / args.tree
    gt = REPO / "peter" / args.tree.removeprefix("out_").removesuffix("_single").removesuffix("_multi")
    clips = sorted(p.name for p in gt.iterdir() if p.is_dir() and (p / "forces.npz").exists())
    for clip in clips:
        out_dir = tree / clip / "human_optim"
        if not out_dir.is_dir():
            print(f"{clip}: no out tree, skipped")
            continue
        z = np.load(gt / clip / "forces.npz", allow_pickle=True)
        if np.asarray(z["forces"]).ndim != 3:
            print(f"{clip}: not 3-axis data, skipped")
            continue
        n = int(z["n_frames"])
        s = per_frame(np.asarray(z["forces"], np.float32),
                      float(z["fps"]) / float(z["video_fps"]), n)
        reaction = -(s @ R_SENSOR_TO_WORLD.T)
        np.savez(
            out_dir / "sensor_forces.npz",
            reaction_world=reaction.astype(np.float32),
            total_world=reaction.sum(1).astype(np.float32),
            limbs=np.asarray(z["limbs"]),
            R_sensor_to_world=R_SENSOR_TO_WORLD,
            method=np.str_("total-kabsch-v1"),
            calib_clips=np.array(CALIB_CLIPS),
            note=np.str_(NOTE),
            video_fps=np.float32(z["video_fps"]),
            source=z["source"],
        )
        print(f"{clip}: wrote sensor_forces.npz  ({n} frames)")


if __name__ == "__main__":
    main()
