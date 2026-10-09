"""Li et al. CVPR'19 foot forces -> the OpenCap plate contract, in the mocap frame.

The estimator (``data/Estimating-3D-Motion-Forces``, run by its
``scripts/run_opencap.py``) writes one ``<clip>_pred.npz`` per clip holding what it
solved for: the 16 Parkour joints, the eight sole contact points, and four 6D contact
wrenches — left foot, right foot, left hand, right hand. This turns the two feet into the
``estimated_forces.npz`` that ``datasets/opencap/compare_opencap_plates.py`` scores,
which means three conversions.

**Frame.** The estimator optimises in the camera, with the camera's ``+y`` taken to BE
gravity, and reports its wrenches in the frame the Parkour ground truth used, a fixed
rotation of that camera: ``f_camera = P f_parkour`` with
``P = [[-1,0,0],[0,0,-1],[0,-1,0]]`` (its own inverse). The lab's calibration then takes
the camera to the mocap ground: ``x_mocap = Rᵀ (x_camera − t)`` with ``R, t`` the
``R_cam_from_mocap`` / ``t_cam_from_mocap`` of ``meta.json``. Nothing is fitted here —
this is the dataset's own calibration, so the forces land in the plates' frame with no
per-clip registration of any kind.

**Reference point.** A foot channel's wrench is a spatial force with world-parallel axes
about whichever joint carried the contact, and the estimator has two per foot. A SOLE
contact is sixteen friction-cone generators built from the four sole points expressed in
the ankle frame (``person_models/limb.py``), so its wrench is about the ANKLE joint origin
and carries a real moment. A TOES contact is four generators with zero angular rows
(``dataloader_person.cc::SetFrictionConeGenerators``), so it is a pure point force at the
TOES joint origin. ``optimizer.py`` sums both into one channel and only ever rotates them,
never moves them, so the per-frame contact states saved beside the wrench are what says
which joint it is about; its moment about the ground origin is then ``τ + p_joint × f``.
The two never coexist on a frame: ``lib/contact_states.py::adjust_toe_contacts`` drops the
toes wherever the sole is in contact, and the exporter that writes the labels never claims
both anyway.

**Triple.** ``datasets/opencap/plates.py`` turns that wrench into the dataset's
``(force, cop, free_torque)`` on the floor plane the plates themselves report
(``export_opencap_mot.floor_y``, ≈6 mm). The estimator's own ground plane is not that
plane — it has no way to know where the lab floor is; it solves a single height in the
camera, its normal hard-wired to the camera's ``+y`` — so the COP is where this foot's
line of action crosses the TRUE floor, which is the only plane the plates' COP can be
compared on. The printed ``foot above floor`` says how far the two planes ended up apart.

``total_mass`` is the estimator's own: its skeleton carries a fixed inertia table
(``person_models/full_body_inertia.pkl``, 74.308 kg) that is the same for every subject
and every clip, so its newtons are the newtons of a 74.3 kg person whatever the subject
weighs. The scorer's "own-BW" row is the one that reads this.

Frames 0 and 1 are marked invalid: their forces are a copy of frame 2's, because a
finite-difference acceleration does not exist there (the estimator's own evaluation does
the same copy).

Usage::

    uv run --no-sync python datasets/opencap/estmf_export.py
    uv run --no-sync python datasets/opencap/estmf_export.py --clips <file>
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


from datasets.opencap.export_opencap_mot import floor_y, plate_triple
from datasets.opencap.plates import LOADED_N
from datasets.opencap.sensor_forces import PROCESSED, world_from_mocap
from bvr.npz_io import atomic_savez
from datasets.methods.paths import ESTMF

ESTIMATOR = ESTMF.resolve()
RESULTS = ESTIMATOR / "validation" / "results" / "opencap"
INERTIA = ESTIMATOR / "person_models" / "full_body_inertia.pkl"

#: Camera axes <- the frame the estimator reports forces in. Symmetric, hence its own
#: inverse; ``optimizer.py::ExpressLocalForcesInParkourFrame`` applies it the other way.
CAMERA_FROM_PARKOUR = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])

#: The wrench channels of ``forces_parkour``: left foot, right foot, then the two hands.
FEET = {"left_foot": 0, "right_foot": 1}
#: Where each foot's ankle and toes joints sit in ``joints_16``, and the hips, in the
#: scorer's order. ``TOES_JOINT`` is the same toes joint in the estimator's own 24-joint
#: numbering, which is how ``contact_states`` is indexed.
ANKLE = {"left_foot": 2, "right_foot": 6}
TOES = {"left_foot": 3, "right_foot": 7}
TOES_JOINT = {"left_foot": 4, "right_foot": 8}
#: ``contact_states`` value for a joint standing on the ground.
GROUND_CONTACT = 2
HIP_COLUMNS = [0, 4]
#: ``sole_points`` columns per foot, and which of the four are the heel and the toe end.
#: ``person_models/limb.py`` builds the sole box with its local ``+z`` along the ankle ->
#: toes direction and puts the four corners at ``±footWidth/2, ±footLength/2``, in the
#: order (rear-inner, front-inner, front-outer, rear-outer) — so 0 and 3 are the rear pair
#: and 1 and 2 the front pair. Measured on a solved clip, the rear pair sits 17 cm from
#: the toes joint and the front pair 4 cm, which is the same statement.
SOLE_COLUMNS = {"left_foot": [0, 1, 2, 3], "right_foot": [4, 5, 6, 7]}
HEEL_POINTS, TOE_POINTS = [0, 3], [1, 2]

#: The first two frames carry a copy of frame 2's forces, not a prediction of their own.
UNPREDICTED_FRAMES = 2


def person_mass() -> float:
    """The estimator's body mass, kg — the sum of its fixed per-link inertia table."""
    with open(INERTIA, "rb") as handle:
        return float(sum(link[0] for link in pickle.load(handle, encoding="latin-1").values()))


def prediction(clip_name: str, save_name: str) -> Path | None:
    path = RESULTS / "run" / "evaluation" / f"{save_name}_s7" / f"{clip_name}_pred.npz"
    return path if path.exists() else None


def to_mocap(pred: dict, rotation: np.ndarray, translation: np.ndarray) -> dict:
    """Everything the estimator solved, taken from its camera to the mocap ground frame."""
    wrench = np.asarray(pred["forces_parkour"], np.float64)
    turn = lambda v: (v @ CAMERA_FROM_PARKOUR.T) @ rotation  # noqa: E731  Rᵀ (P v)
    return {
        "force": turn(wrench[:, :, :3]),
        "torque": turn(wrench[:, :, 3:]),
        "joints": (np.asarray(pred["joints_16"], np.float64) - translation) @ rotation,
        "soles": (np.asarray(pred["sole_points"], np.float64) - translation) @ rotation,
    }


def export(clip: Path, run_dir: Path, save_name: str) -> int:
    """Convert one clip's prediction into the plate contract. Returns the frames written."""
    source = prediction(clip.name, save_name)
    if source is None:
        print(f"{clip.name}: no stage-7 prediction, skipped")
        return 0
    pred = np.load(source)
    rotation, translation = world_from_mocap(clip)
    # The estimator takes the camera's +y FOR gravity and freezes its ground plane's
    # normal there. This is how far that axis really is from the mocap vertical.
    tilt = float(np.degrees(np.arccos(np.clip(-rotation[1, 1], -1.0, 1.0))))
    solved = to_mocap(pred, rotation, translation)
    y0 = floor_y(clip)

    if "contact_states" not in pred.files:
        raise SystemExit(f"{source} predates the contact-state fix — rerun the estimator")
    states = np.asarray(pred["contact_states"], int)

    gt = np.load(clip / "forces.npz", allow_pickle=True)
    n_frames = len(gt["force"])
    n_solved = min(len(solved["force"]), n_frames)
    if len(solved["force"]) != n_frames:
        print(f"  {clip.name}: the estimator solved {len(solved['force'])} frames, "
              f"the plates cover {n_frames}")

    force, cop, free_torque = (np.zeros((n_frames, 2, 3)) for _ in range(3))
    foot_points = np.full((n_frames, 2, 2, 3), np.nan)
    undefined, heights, on_toes_frames = {}, {}, {}
    for column, (limb, channel) in enumerate(FEET.items()):
        on_toes = states[:n_solved, TOES_JOINT[limb]] == GROUND_CONTACT
        reference = np.where(on_toes[:, None],
                             solved["joints"][:n_solved, TOES[limb]],
                             solved["joints"][:n_solved, ANKLE[limb]])
        limb_force = solved["force"][:n_solved, channel]
        moment = solved["torque"][:n_solved, channel] + np.cross(reference, limb_force)
        soles = solved["soles"][:n_solved][:, SOLE_COLUMNS[limb]]
        fallback = soles.mean(1).copy()
        fallback[:, 1] = y0
        triple = plate_triple(limb_force, moment, fallback, y0)
        force[:n_solved, column], cop[:n_solved, column], free_torque[:n_solved, column] = triple[:3]
        undefined[limb] = triple[3]
        on_toes_frames[limb] = int(on_toes.sum())
        foot_points[:n_solved, column, 0] = soles[:, HEEL_POINTS].mean(1)
        foot_points[:n_solved, column, 1] = soles[:, TOE_POINTS].mean(1)
        carrying = limb_force[:, 1] > LOADED_N
        heights[limb] = (float(np.median(soles[carrying, :, 1]) - y0) if carrying.any()
                         else float("nan"))

    hips = np.full((n_frames, 2, 3), np.nan)
    hips[:n_solved] = solved["joints"][:n_solved, HIP_COLUMNS]
    valid = np.zeros(n_frames, bool)
    valid[UNPREDICTED_FRAMES:n_solved] = True

    mass = person_mass()
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        run_dir / "estimated_forces.npz",
        limbs=np.array(list(FEET)),
        force=force.astype(np.float32), cop=cop.astype(np.float32),
        free_torque=free_torque.astype(np.float32),
        total_mass=np.float32(mass), valid_mask=valid,
        hips_mocap=hips.astype(np.float32),
        foot_points_mocap=foot_points.astype(np.float32),
        frame=np.str_("opensim_ground_y_up"), fps=np.float32(60.0),
        method=np.str_("li2019"),
        cop_undefined_rows=np.array([undefined[k] for k in FEET], np.int32),
        floor_y_m=np.float32(y0), camera_tilt_deg=np.float32(tilt),
        toe_contact_frames=np.array([on_toes_frames[k] for k in FEET], np.int32),
        # median height of a foot's sole points above the true floor, on its loaded frames
        foot_above_floor_m=np.array([heights[k] for k in FEET], np.float32),
        R_cam_from_mocap=rotation.astype(np.float32),
        t_cam_from_mocap=translation.astype(np.float32),
        source=np.str_(str(source)))
    peak = force[valid][:, :, 1].max() if valid.any() else float("nan")
    loaded = (force[valid][:, :, 1] > 20.0).mean() if valid.any() else float("nan")
    print(f"{clip.name}: {n_solved}/{n_frames} frames, peak Fy {peak:6.0f} N, foot loaded "
          f"{loaded:4.0%}, foot above floor L {heights['left_foot'] * 1e3:5.0f} mm "
          f"R {heights['right_foot'] * 1e3:5.0f} mm, camera tilt {tilt:.1f} deg, toe "
          f"contact on {on_toes_frames['left_foot']} L / {on_toes_frames['right_foot']} R, "
          f"COP undefined on {undefined['left_foot']} L / {undefined['right_foot']} R rows")
    return n_solved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=PROCESSED, help="the processed clip root")
    ap.add_argument("--run", default="method_estmf", help="run directory under out/<clip>/")
    ap.add_argument("--save-name", default="ours", help="the estimator's --save-name")
    ap.add_argument("--clips", type=Path, default=None,
                    help="file listing the clips to export (default: every prediction found)")
    args = ap.parse_args(argv)

    if args.clips is not None:
        names = [line.strip() for line in args.clips.read_text().split() if line.strip()]
    else:
        folder = RESULTS / "run" / "evaluation" / f"{args.save_name}_s7"
        names = sorted(p.name[: -len("_pred.npz")] for p in folder.glob("*_pred.npz"))

    written = 0
    print(f"[estmf_export] mass of the estimator's person model: {person_mass():.3f} kg")
    for name in names:
        if export(args.source / name, args.source / "out" / name / args.run, args.save_name):
            written += 1
    print(f"\n[estmf_export] {written}/{len(names)} clip(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
