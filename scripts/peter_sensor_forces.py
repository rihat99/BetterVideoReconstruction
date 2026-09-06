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

The climb_wall_2 demonstrator rig is different: its four channels ARE per-limb
contact forces (one board per limb, verified independent), logged in the board
frame the datasheet documents (+x up along the wall plane, +y right, +z out of
the wall toward the climber; wall inclined ~16 deg). Its rotation below comes
from that DATASHEET, not from any fit: true gravity (3-view floor fit) plus an
exact 16 deg overhang tilt determine the frame up to its horizontal facing
direction (azimuth), the one scalar a datasheet cannot contain — that comes
from the fused scene's wall normal. Nothing touches our predictions, which
makes the measured-vs-predicted comparison fully independent.

Usage:
    uv run python scripts/peter_sensor_forces.py [--tree out_climb_wall_single]
    uv run python scripts/peter_sensor_forces.py --tree out_climb_wall_2_single
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

# climb_wall_2 (the 2026-08 demonstrator rig): channels ARE per-limb forces, in
# the datasheet's board frame (+x up along the wall, +y the climber's right, +z out
# toward the climber; wall inclined 16 deg). That frame is LEFT-handed (the author
# confirmed it), so this is a reflection (det -1), not a rotation. Built from the
# datasheet (scratch/cw2_wall_datasheet.py, 2026-08-20): up = -gravity from the
# 3-view floor fit (scratch/cw2_gravity_fit.py), tilt pinned to the spec's EXACT
# 16 deg overhang (z = cos16*h - sin16*up, x = sin16*h + cos16*up), y = x cross z
# (the climber's right; the camera sees the climber's back, so that is image right),
# and only the wall's horizontal facing direction h read from the fused-scene
# normal. Zero force fitting. 2026-09-06: the y column was MIRRORED until now
# (built right-handed as z cross x = the climber's left); the fix halves the
# per-limb measured-vs-kindyn angles (25/26/17/21 -> 11/10/8/11 deg) and turns the
# solve-free check corr(lateral CoM accel, summed lateral force) from -0.30 to +0.65.
R_CW2_WALL_TO_WORLD = np.array([
    [+0.0934, +0.9280, +0.3608],
    [-0.8488, -0.1152, +0.5160],
    [-0.5204, +0.3544, -0.7769],
], dtype=np.float32)
NOTE_CW2 = ("channels are TRUE per-limb contact forces (one board per limb; "
            "verified: only the touched channel responds at the sync tap, a "
            "released limb reads ~0 mid-climb); coupling is only a <1 s "
            "structural ring after impacts")

# gt dataset name -> (rotation, method tag, calibration clips, note)
RIGS = {
    "climb_wall": (R_SENSOR_TO_WORLD, "total-kabsch-v1", CALIB_CLIPS, NOTE),
    "climb_wall_2": (R_CW2_WALL_TO_WORLD, "wall-datasheet-v1",
                     [f"trial_0{n}{p}" for n, parts in [(1, 2), (2, 2), (3, 3)]
                      for p in [f"_part{i + 1}" for i in range(parts)]], NOTE_CW2),
}


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
    dataset = args.tree.removeprefix("out_").removesuffix("_single").removesuffix("_multi")
    gt = REPO / "peter" / dataset
    if dataset not in RIGS:
        raise SystemExit(f"[peter_sensor_forces] no rig calibration for {dataset!r}")
    rotation, method, calib_clips, note = RIGS[dataset]
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
        reaction = -(s @ rotation.T)
        np.savez(
            out_dir / "sensor_forces.npz",
            reaction_world=reaction.astype(np.float32),
            total_world=reaction.sum(1).astype(np.float32),
            limbs=np.asarray(z["limbs"]),
            R_sensor_to_world=rotation,
            method=np.str_(method),
            calib_clips=np.array(calib_clips),
            note=np.str_(note),
            video_fps=np.float32(z["video_fps"]),
            source=z["source"],
        )
        print(f"{clip}: wrote sensor_forces.npz  ({n} frames)")


if __name__ == "__main__":
    main()
