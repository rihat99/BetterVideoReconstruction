"""Write world-frame wall-sensor forces into the peter climb_wall out trees.

The climb_wall_2 / climb_wall_3 demonstrator rigs record four 3-axis force
boards at ~98.85 Hz (``peter/<rig>/<clip>/forces.npz``, one channel per limb
name). The four channels ARE per-limb contact forces (one board per limb,
verified independent), logged in the board frame the datasheet documents (+x up
along the wall plane, +y right, +z out of the wall toward the climber; wall
inclined ~16 deg). The rotation below comes from that DATASHEET, not from any
fit: true gravity (3-view floor fit) plus an exact overhang tilt determine the
frame up to its horizontal facing direction (azimuth), the one scalar a
datasheet cannot contain — that comes from the fused scene's wall normal.
Nothing touches our predictions, which makes the measured-vs-predicted
comparison fully independent.

Output ``<tree>/<clip>/human_optim/sensor_forces.npz`` per video frame, in the
REACTION sense (force on the climber, comparable to the predicted contact
forces): ``reaction_world (N, 4, 3)`` per channel and ``total_world (N, 3)``.

Usage:
    uv run python datasets/peter/sensor_forces.py --tree out_climb_wall_2_single
    uv run python datasets/peter/sensor_forces.py [--tree out_climb_wall_3_single]
"""
from __future__ import annotations

import argparse

import numpy as np


from datasets.peter.paths import PETER

# climb_wall_2 (the 2026-08 demonstrator rig): channels ARE per-limb forces, in
# the datasheet's board frame (+x up along the wall, +y the climber's right, +z out
# toward the climber; wall inclined 16 deg). That frame is LEFT-handed (the author
# confirmed it), so this is a reflection (det -1), not a rotation. Built from the
# datasheet (<data root>/peter/_provenance/cw2_wall_datasheet.py, 2026-08-20): up = -gravity from the
# 3-view floor fit (<data root>/peter/_provenance/cw2_gravity_fit.py), tilt pinned to the spec's EXACT
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
# climb_wall_3 (the 2026-09-03 session on the same demonstrator rig): the same
# board frame, but the camera was moved between the three wall inclinations, so
# there is one rotation per inclination, keyed by the recording name's middle
# field (a_16_02_part3 -> "16"). Built by <data root>/peter/_provenance/cw3_wall_datasheet.py exactly
# like R_CW2_WALL_TO_WORLD: up = -gravity from that set-up's own 3-view floor fit
# (<data root>/peter/_provenance/cw3_gravity_fit.py; 0 deg needed a 12 deg tilt gate), tilt pinned to
# the name's 0 / 16 / 32, azimuth from the fused scene's wall normal (the scene
# normal sat 3-4 deg from the pinned tilt in all three). Left-handed, det -1.
R_CW3_WALL_TO_WORLD = {
    "00": np.array([
        [+0.0299, +0.9807, +0.1930],
        [-0.9892, +0.0568, -0.1350],
        [+0.1434, +0.1869, -0.9719],
    ], dtype=np.float32),
    "16": np.array([
        [+0.0783, +0.9966, +0.0250],
        [-0.9743, +0.0712, +0.2138],
        [-0.2113, +0.0411, -0.9766],
    ], dtype=np.float32),
    "32": np.array([
        [+0.0444, +0.9808, +0.1899],
        [-0.8627, -0.0582, +0.5023],
        [-0.5037, +0.1862, -0.8436],
    ], dtype=np.float32),
}

RIGS = {
    "climb_wall_2": (R_CW2_WALL_TO_WORLD, "wall-datasheet-v1",
                     [f"trial_0{n}{p}" for n, parts in [(1, 2), (2, 2), (3, 3)]
                      for p in [f"_part{i + 1}" for i in range(parts)]], NOTE_CW2),
    "climb_wall_3": (R_CW3_WALL_TO_WORLD, "wall-datasheet-v1", ["(per inclination, no fit)"],
                     NOTE_CW2),
}


def per_frame(forces: np.ndarray, ratio: float, n_frames: int) -> np.ndarray:
    """Average the ~98.85 Hz samples into per-video-frame values, ``(N, 4, 3)``."""
    return np.stack([forces[int(i * ratio):max(int((i + 1) * ratio), int(i * ratio) + 1)].mean(0)
                     for i in range(n_frames)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--tree", default="out_climb_wall_3_single",
                    help="out tree under ../data/peter/ to write into")
    args = ap.parse_args()

    tree = PETER / args.tree
    dataset = args.tree.removeprefix("out_").removesuffix("_single").removesuffix("_multi")
    gt = PETER / dataset
    if dataset not in RIGS:
        raise SystemExit(f"[sensor_forces] no rig calibration for {dataset!r}")
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
        # climb_wall_3 carries one rotation per wall inclination (the name's middle field).
        clip_rotation = rotation[clip.split("_")[1]] if isinstance(rotation, dict) else rotation
        reaction = -(s @ clip_rotation.T)
        np.savez(
            out_dir / "sensor_forces.npz",
            reaction_world=reaction.astype(np.float32),
            total_world=reaction.sum(1).astype(np.float32),
            limbs=np.asarray(z["limbs"]),
            R_sensor_to_world=clip_rotation,
            method=np.str_(method),
            calib_clips=np.array(calib_clips),
            note=np.str_(note),
            video_fps=np.float32(z["video_fps"]),
            source=z["source"],
        )
        print(f"{clip}: wrote sensor_forces.npz  ({n} frames)")


if __name__ == "__main__":
    main()
