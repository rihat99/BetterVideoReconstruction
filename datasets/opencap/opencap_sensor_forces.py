"""Put the OpenCap LabValidation force plates into OUR world frame.

``datasets/opencap/import_opencap.py`` stores each clip's plate ground truth per video frame in
the dataset's own OpenSim ground frame (right-handed, **Y up**, floor at y≈0):
``processed/<clip>/forces.npz`` with ``force``/``cop``/``free_torque`` ``(N, 4, 3)`` in
:data:`~bvr.human_optim.boards.LIMBS` order plus a ``captured (N, 4)`` mask. The
contact stages and the force comparison read ``sensor_forces.npz`` instead — the same
vectors in the reconstruction's world frame. This script is that rotation.

Our world is VGGT's first-frame camera (OpenCV convention), and the dataset ships the
calibration that reaches it: ``meta.json`` carries ``R_cam_from_mocap`` /
``t_cam_from_mocap``, composed by the importer from the camera's own extrinsics and the
session's mocap→video transform. So

    f_world = R f_mocap,   p_world = R p_mocap + t,   τ_world = R τ_mocap

with ``R = R_cam_from_mocap``. **Nothing is fitted** — no force, no pose, no scale enters
the transform — which is what makes the measured-vs-predicted comparison honest. Unlike
Parkour's MoCap yaw, this frame IS our frame, so ``world_aligned = True`` and the
direction metrics apply.

The one free check on that chain is gravity: the mocap floor's up axis is ``+y``, so
``R @ [0, -1, 0]`` must point the way the gravity stage says down is. The angle between
them lands in the file as ``gravity_check_deg`` (NaN when the clip has no
``geocalib/gravity.npz``); a few degrees is the gravity stage's own noise, tens of degrees
means the calibration chain is wrong. Read it next to ``gravity_source``, also saved:
``"ground"`` means the gravity stage fell back to fitting the floor of our own
reconstruction, so the check is against that plane and not against an independent
measurement. And note what the check CANNOT see — it constrains two of the rotation's
three degrees of freedom, leaving the yaw about the vertical unconstrained, and says
nothing at all about the translation (for which the comparison reports a ``cop gap``).

Limb-frames the plates could not see (``captured`` False: a drop jump's box phase, a
foot off the plates) are written NaN, which every reader already treats as *not
measured* rather than *no force*. The hands are written as measured zeros — these trials
have no hand contact at all.

Output ``processed/out/<clip>/<run>/sensor_forces.npz``, beside the run, exactly where
``datasets/parkour/parkour_sensor_forces.py`` and ``datasets/peter/peter_sensor_forces.py`` put their
rigs' own. ``--run`` names the run directory, so a variant chain
(``human_optim_floorplane``) gets its own copy of the same rotated plates.

Usage::

    uv run --no-sync python datasets/opencap/opencap_sensor_forces.py
    uv run --no-sync python datasets/opencap/opencap_sensor_forces.py --run human_optim_floorplane
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


from bvr.human_optim.boards import LIMBS
from bvr.npz_io import atomic_savez
from datasets.opencap.paths import LABVALIDATION

#: Where ``datasets/opencap/import_opencap.py`` writes the clips and the pipeline writes its runs.
PROCESSED = LABVALIDATION / "processed"

#: Mocap up axis — the OpenSim ground frame is Y up, so this is where "down" lives.
MOCAP_DOWN = np.array([0.0, -1.0, 0.0])

METHOD = ("OpenCap LabValidation force plates, rotated into our world by the dataset's own "
          "camera calibration (R_cam_from_mocap; no force or pose fitting)")


def world_from_mocap(clip: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(R (3, 3), t (3,))`` taking OpenSim ground coordinates to our world, from ``meta.json``."""
    meta = json.loads((clip / "meta.json").read_text())
    missing = [k for k in ("R_cam_from_mocap", "t_cam_from_mocap") if k not in meta]
    if missing:
        raise SystemExit(f"{clip.name}: meta.json has no {missing} — the importer must compose them")
    rotation = np.asarray(meta["R_cam_from_mocap"], np.float64).reshape(3, 3)
    translation = np.asarray(meta["t_cam_from_mocap"], np.float64).reshape(3)
    if np.abs(rotation @ rotation.T - np.eye(3)).max() > 1e-3 or np.linalg.det(rotation) < 0:
        raise SystemExit(f"{clip.name}: R_cam_from_mocap is not a rotation "
                         f"(det {np.linalg.det(rotation):.3f})")
    if np.linalg.norm(translation) > 50.0:      # a lab camera sits metres, not millimetres, away
        raise SystemExit(f"{clip.name}: t_cam_from_mocap is {np.linalg.norm(translation):.0f} "
                         "long — metres expected, this looks like millimetres")
    return rotation, translation


def gravity_check(out_clip: Path, rotation: np.ndarray) -> tuple[float, str]:
    """``(angle in degrees, source)`` between the mocap floor's down axis and the stage's.

    The source matters as much as the angle: ``"geocalib"`` is an independent estimate,
    ``"ground"`` is a plane fitted to our own reconstructed floor, and the fallback is
    world-down, which agrees with nothing in particular.
    """
    path = out_clip / "geocalib" / "gravity.npz"
    if not path.exists():
        return float("nan"), "none"
    gravity = np.load(path)
    ours = np.asarray(gravity["gravity_world"], np.float64)
    theirs = rotation @ MOCAP_DOWN
    cos = float(ours @ theirs / max(np.linalg.norm(ours) * np.linalg.norm(theirs), 1e-12))
    source = str(gravity["source"]) if "source" in gravity.files else "unknown"
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))), source


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=PROCESSED, help="the processed clip root")
    ap.add_argument("--run", default="human_optim", help="run directory under processed/out/<clip>/")
    args = ap.parse_args(argv)

    written = 0
    clips = sorted(p for p in args.source.iterdir()
                   if (p / "forces.npz").exists() and (p / "meta.json").exists())
    for clip in clips:
        out_clip = args.source / "out" / clip.name
        if not out_clip.is_dir():
            print(f"{clip.name}: no out tree, skipped")
            continue
        gt = np.load(clip / "forces.npz", allow_pickle=True)
        if [str(x) for x in gt["limbs"]] != LIMBS:
            raise SystemExit(f"{clip.name}: force channels are not {LIMBS}")
        rotation, translation = world_from_mocap(clip)

        captured = np.asarray(gt["captured"], bool)                       # (N, 4)
        blank = ~captured[..., None]
        reaction = np.where(blank, np.nan, np.asarray(gt["force"], np.float64) @ rotation.T)
        cop = np.where(blank, np.nan, np.asarray(gt["cop"], np.float64) @ rotation.T + translation)
        torque = np.where(blank, np.nan,
                          np.asarray(gt["free_torque"], np.float64) @ rotation.T)
        total = reaction.sum(1)                   # a sum missing a limb is not the total
        total[~captured.all(1)] = np.nan

        angle, gravity_source = gravity_check(out_clip, rotation)
        out_dir = out_clip / args.run
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_savez(
            out_dir / "sensor_forces.npz",
            reaction_world=reaction.astype(np.float32),
            total_world=total.astype(np.float32),
            cop_world=cop.astype(np.float32),
            free_torque_world=torque.astype(np.float32),
            limbs=np.asarray(gt["limbs"]), captured=captured,
            world_aligned=np.bool_(True),
            R_world_from_mocap=rotation.astype(np.float32),
            t_world_from_mocap=translation.astype(np.float32),
            gravity_check_deg=np.float32(angle), gravity_source=np.str_(gravity_source),
            mass_kg=np.float32(gt["mass_kg"]), height_m=np.float32(gt["height_m"]),
            method=np.str_(METHOD),
            video_fps=np.float32(gt["fps"]), source=gt["source"])
        print(f"{clip.name}: wrote sensor_forces.npz  ({len(reaction)} frames, "
              f"{int((~captured).sum())} unmeasured limb-frames, "
              f"gravity check {angle:.1f} deg vs {gravity_source})")
        written += 1
    print(f"\n[opencap_sensor_forces] {written} clip(s) of {len(clips)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
