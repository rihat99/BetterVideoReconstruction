"""Write the LAAS Parkour force ground truth in the schema the board stages read.

The Parkour rig measures what the peter climbing boards measure — one 3D contact force
per limb, per video frame — but ``datasets/parkour/import_parkour.py`` stores it as a 6D wrench
with a capture mask (``processed/<clip>/forces.npz``). The contact stages read
``sensor_forces.npz``: the linear part alone, ``(N, 4, 3)`` in
:data:`~bvr.human_optim.boards.LIMBS` order, in the reaction-on-the-person sense.
This script is that translation, so ``measure_contacts_boards.py`` and the comparison
run on Parkour with no changes of their own.

Two things are unlike the peter rigs, and the file records both:

* the vectors live in the MoCap world, whose yaw is not our world's yaw, so only their
  SIZES compare — ``world_aligned = False``, and readers drop the direction metrics;
* a limb that landed off the plates was never measured. Those samples are written as
  NaN, which the boards module already reads as no force, and which keeps "not
  touching" distinguishable from "not measured" for anything that cares.

Output ``processed/out/<clip>/human_optim/sensor_forces.npz``, beside the run, exactly
where ``datasets/peter/peter_sensor_forces.py`` puts the peter rigs' own.

Usage::

    uv run --no-sync python datasets/parkour/parkour_sensor_forces.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


from datasets.parkour import paths
from bvr.human_optim.boards import LIMBS

PARKOUR = paths.PARKOUR / "processed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=PARKOUR, help="the processed clip root")
    args = ap.parse_args(argv)

    written = 0
    for clip in sorted(p for p in args.source.iterdir() if (p / "forces.npz").exists()):
        out_dir = args.source / "out" / clip.name / "human_optim"
        if not out_dir.is_dir():
            print(f"{clip.name}: no out tree, skipped")
            continue
        gt = np.load(clip / "forces.npz", allow_pickle=True)
        if [str(x) for x in gt["limbs"]] != LIMBS:
            raise SystemExit(f"{clip.name}: force channels are not {LIMBS}")
        reaction = np.asarray(gt["forces"], np.float32)[:, :, :3]
        captured = np.asarray(gt["captured"], bool)
        reaction[~captured] = np.nan
        total = reaction.sum(1)                      # a sum missing a limb is not the total
        total[~captured.all(1)] = np.nan
        np.savez(
            out_dir / "sensor_forces.npz",
            reaction_world=reaction, total_world=total.astype(np.float32),
            limbs=np.asarray(gt["limbs"]), captured=captured,
            world_aligned=np.bool_(False),
            method=np.str_("LAAS Parkour force plates + instrumented bar, MoCap world frame"),
            video_fps=np.float32(gt["fps"]), source=gt["source"])
        print(f"{clip.name}: wrote sensor_forces.npz  ({len(reaction)} frames, "
              f"{int((~captured).sum())} unmeasured limb-frames)")
        written += 1
    print(f"\n[parkour_sensor_forces] {written} clip(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
