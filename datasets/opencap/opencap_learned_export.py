"""Turn the learned model's six-group forces into the OpenCap plate contract.

``contact_anything``'s ``predict_reconstruction.py`` writes, per scene,
``predictions/forces.npz`` (six groups ``left_hand, right_hand, left_foot (toe),
right_foot, left_ankle (heel), right_ankle``, ``forces_world`` in BODY-WEIGHT units in
OUR world) and ``predictions/smplx.npz`` (the refined 52-joint body, ``joints_world``).
The scorer ``datasets/opencap/compare_opencap_plates.py`` wants one plate-style triple
per foot in the OpenSim ground frame. This script is that conversion.

**Newtons.** The model was trained on ``force / (total_mass * 9.81)`` of the kindyn body
of the same clip (``data/climbing_videos/kindyn.py::load_forces``), so the inverse is
``forces_world * total_mass * 9.81`` with the ``total_mass`` of OUR ``kindyn_1.npz`` —
the shape-baked SMPL-X mass, NOT the subject's measured mass. That same mass is written
into the file as ``total_mass`` so the scorer can form the mass-independent "own-BW" row.

**Frames.** Forces and joints are in our world (the ``geometry/transform.npz`` world =
VGGT's first camera, which for these static-camera clips IS the camera). They are taken
into the mocap frame with the dataset's own calibration exactly as
``datasets/opencap/export_opencap_mot.py`` does it — ``f' = Rᵀ f``, ``r' = Rᵀ (r − t)`` with
``R, t`` the ``R_cam_from_mocap`` / ``t_cam_from_mocap`` of ``meta.json``.

**The triple.** A foot's force is toe + heel; its moment about the ground origin is
``r_toe × f_toe + r_heel × f_heel`` with the anchors the refined body's own toe
(``*_foot``) and ankle (``*_ankle``) joints — the points the force groups are defined at.
:func:`~datasets.opencap.export_opencap_mot.plate_triple` then inverts that wrench onto the floor
plane the dataset reports for the clip (:func:`~datasets.opencap.export_opencap_mot.floor_y`),
which is where the projection to the floor happens.

**Coverage.** The model predicts on a ~25 fps stride of the 60 fps clip, so only every
second row carries a prediction (``predicted_mask``). Scoring only those rows would score
an easier problem — the method would never be asked about the instants between its own
samples, where a drop jump's impact peak lives — so ``valid_mask`` is the whole SPAN the
model covered: every row from the first to the last prediction of a covered run, with the
holes the stride leaves filled by linear interpolation (:func:`fill_gaps`, applied to the
group forces and to every point before the triple is formed, so force, CoP and free
moment stay one consistent wrench). Only rows outside the covered runs are invalid. The
same filling is what keeps the scorer's low-pass honest: a signal holed at every second
sample comes out of a 6 Hz Butterworth at half amplitude.

**Foot landmarks.** ``foot_points_mocap`` is the heel and big-toe CONTACT FRAMES of the
same BetterHuman body our own export uses, not the ankle and toe joints, so the in-foot
CoP compares homologous landmarks across methods. They are re-posed from the refined
body's ``q_cam`` + ``betas`` (the ``predict_test`` layout is the kindyn ``q`` layout, and
the learned body carries no global scale) and put into the model's own world by the rigid
map from the posed joints onto its ``joints_world`` (:func:`contact_frame_points`).

Writes ``processed/out/<clip>/<run>/estimated_forces.npz`` plus a copy of the source
``predictions/`` directory beside it, so nothing of the method lives in the variant farm.

Usage::

    uv run --no-sync python datasets/opencap/opencap_learned_export.py \
        --clips /…/processed/out/_methods/clips_natural.txt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch


from datasets.opencap.export_opencap_mot import floor_y, plate_triple
from datasets.opencap.opencap_plates import LOADED_N
from datasets.opencap.opencap_sensor_forces import PROCESSED, world_from_mocap
from bvr.body import load_body
from bvr.human_optim.qmath import normalize_q
from bvr.npz_io import atomic_savez

GRAVITY = 9.81

#: The learned model's six force groups, in its ``forces.npz`` column order.
LEARNED_GROUPS = ("left_hand", "right_hand", "left_foot", "right_foot",
                  "left_ankle", "right_ankle")

#: The two feet the plates measure, and per foot the (toe, heel) force group. Each group
#: is anchored at the SMPL-X joint of the same name, so the group name is also the
#: point's joint name: ``*_foot`` is the big-toe joint, ``*_ankle`` the heel.
FEET = ("left_foot", "right_foot")
FOOT_GROUPS = {"left_foot": ("left_foot", "left_ankle"),
               "right_foot": ("right_foot", "right_ankle")}

#: Our hip joints, in the ``hips_mocap`` column order the scorer expects.
HIP_JOINTS = ("left_hip", "right_hip")

#: The heel and big-toe CONTACT FRAMES of our body, per foot — the ``foot_points_mocap``
#: landmarks, the same pair ``compare_opencap_plates.FOOT_FRAMES`` uses for our own runs.
FOOT_FRAMES = {"left_foot": ("heel_left", "big_toe_left"),
               "right_foot": ("heel_right", "big_toe_right")}

METHOD = "contact_anything L_body (round 10), given-gravity eval twin"


def load_prediction(pred_dir: Path, n_frames: int) -> dict:
    """One scene's predictions padded to ``n_frames``: forces, body and coverage.

    :returns: ``forces_bw (N, 6, 3)``, ``joints_world`` / ``joints_cam (N, 52, 3)``,
        ``q_cam (N, 211)``, ``betas (N, 10)``, ``covered (N,)`` and the clip ``stride``.

    The prediction arrays carry a leading person axis; these trees hold exactly one
    tracked person. Rows the model did not cover are NaN in the source and are returned
    as zero force / NaN body with ``covered = False``.
    """
    forces = np.load(pred_dir / "forces.npz", allow_pickle=True)
    body = np.load(pred_dir / "smplx.npz", allow_pickle=True)
    if [str(x) for x in forces["limbs"]] != list(LEARNED_GROUPS):
        raise SystemExit(f"{pred_dir}: unexpected force groups {list(forces['limbs'])}")
    if str(forces["units"]) != "body_weight":
        raise SystemExit(f"{pred_dir}: forces are in {forces['units']!s}, not body_weight")
    forces_bw = np.asarray(forces["forces_world"], np.float64)
    joints = np.asarray(body["joints_world"], np.float64)
    covered = np.asarray(forces["valid_mask"], bool)
    if forces_bw.shape[0] != 1:
        raise SystemExit(f"{pred_dir}: {forces_bw.shape[0]} people, expected one")
    forces_bw, joints, covered = forces_bw[0], joints[0], covered[0]
    if not np.array_equal(covered, np.asarray(body["covered"], bool)[0]):
        raise SystemExit(f"{pred_dir}: forces and body disagree about which rows are covered")
    if not (np.isfinite(forces_bw[covered]).all() and np.isfinite(joints[covered]).all()):
        raise SystemExit(f"{pred_dir}: a covered row holds non-finite forces or joints")

    def pad(values: np.ndarray, fill: float) -> np.ndarray:
        if len(values) >= n_frames:
            return values[:n_frames]
        tail = np.full((n_frames - len(values), *values.shape[1:]), fill, values.dtype)
        return np.concatenate([values, tail], 0)

    forces_bw = np.where(covered[:, None, None], np.nan_to_num(forces_bw), 0.0)
    return {"forces_bw": pad(forces_bw, 0.0), "joints_world": pad(joints, np.nan),
            "joints_cam": pad(np.asarray(body["joints_cam"], np.float64)[0], np.nan),
            "q_cam": pad(np.asarray(body["q_cam"], np.float32)[0], np.nan),
            "betas": pad(np.asarray(body["betas"], np.float32)[0], np.nan),
            "covered": pad(covered, False), "stride": int(forces["stride"])}


def covered_span(covered: np.ndarray, stride: int) -> np.ndarray:
    """``covered`` widened over the holes the prediction stride leaves inside a covered run.

    A hole of at most ``stride`` frames between two predictions belongs to the run the
    model covered; a longer one is a real break (an untracked stretch) and stays invalid.
    """
    span = covered.copy()
    index = np.flatnonzero(covered)
    for start, end in zip(index[:-1], index[1:]):
        if end - start <= stride:
            span[start:end] = True
    return span


def rigid_map(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row rotation and translation taking ``source`` onto ``target``, both ``(B, K, 3)``.

    Kabsch, one fit per row; the two point sets are the same body in two frames, so the
    fit is exact up to the posing round-off.
    """
    source_centre, target_centre = source.mean(1, keepdims=True), target.mean(1, keepdims=True)
    covariance = np.einsum("bki,bkj->bij", source - source_centre, target - target_centre)
    left, _, right = np.linalg.svd(covariance)
    flip = np.tile(np.eye(3), (len(source), 1, 1))
    flip[:, 2, 2] = np.sign(np.linalg.det(
        np.einsum("bij,bkj->bik", right.transpose(0, 2, 1), left)))
    rotation = right.transpose(0, 2, 1) @ flip @ left.transpose(0, 2, 1)
    return rotation, (target_centre[:, 0]
                      - np.einsum("bij,bj->bi", rotation, source_centre[:, 0]))


@torch.no_grad()
def contact_frame_points(prediction: dict, chunk: int = 64) -> tuple[np.ndarray, float]:
    """``((N, 4, 3) world heel / big-toe points, the posing residual in metres)``.

    The refined body is re-posed from its ``q_cam`` + ``betas`` on our own SMPL-X body
    (:func:`~bvr.body.load_body`), exactly as ``export_opencap_mot.frame_positions``
    poses ours, but with no global scale — the learned body carries none. The posed body
    lands in the camera frame, so it is put into the model's world by the rigid map from
    its posed joints onto the ``joints_world`` the prediction file carries; the returned
    residual is how well that map holds (a large one means the layouts disagree).
    """
    names = [name for foot in FEET for name in FOOT_FRAMES[foot]]
    body = load_body(device="cpu", frames_allow=names)
    order = [list(body.contact_frame_names).index(name) for name in names]
    covered = prediction["covered"]
    points = np.full((len(covered), len(names), 3), np.nan)
    rows, residual = np.flatnonzero(covered), 0.0
    for start in range(0, len(rows), chunk):
        sel = rows[start:start + chunk]
        out = body.forward(
            torch.as_tensor(prediction["betas"][sel]),
            normalize_q(torch.as_tensor(prediction["q_cam"][sel]), body.num_joints))
        posed = out.joints.numpy().astype(np.float64)
        frames = out.frames_world[..., :3].numpy().astype(np.float64)[:, order]
        target = prediction["joints_world"][sel]
        rotation, translation = rigid_map(posed, target)
        residual = max(residual, float(np.abs(
            np.einsum("bij,bkj->bki", rotation, posed) + translation[:, None] - target).max()))
        points[sel] = np.einsum("bij,bkj->bki", rotation, frames) + translation[:, None]
    return points, residual


def fill_gaps(values: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``values`` over the rows ``covered`` is False, per component.

    Outside the covered span the nearest covered row is held (``np.interp``'s clamp),
    which is the same constant extension the comparison's ``filtfilt`` pads with.
    """
    if not covered.any():
        return values
    index = np.arange(len(values), dtype=np.float64)
    flat = values.reshape(len(values), -1).copy()
    for column in range(flat.shape[1]):
        flat[:, column] = np.interp(index, index[covered], flat[covered, column])
    return flat.reshape(values.shape)


def foot_triples(forces_n: np.ndarray, joints_mocap: np.ndarray, joint_names: list[str],
                 y0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Per-foot ``(force, cop, free_torque)`` blocks ``(N, 2, 3)`` plus the undefined-row count.

    :param forces_n: ``(N, 6, 3)`` group forces in newtons, mocap frame.
    :param joints_mocap: ``(N, 52, 3)`` refined-body joints in the mocap frame.
    :param y0: the clip's floor plane, metres.
    """
    blocks, undefined = [], {}
    for foot in FEET:
        toe_group, heel_group = FOOT_GROUPS[foot]
        toe_f = forces_n[:, LEARNED_GROUPS.index(toe_group)]
        heel_f = forces_n[:, LEARNED_GROUPS.index(heel_group)]
        toe_p = joints_mocap[:, joint_names.index(toe_group)]
        heel_p = joints_mocap[:, joint_names.index(heel_group)]
        moment = np.cross(toe_p, toe_f) + np.cross(heel_p, heel_f)
        weight = np.stack([np.linalg.norm(toe_f, axis=-1), np.linalg.norm(heel_f, axis=-1)], 1)
        points = np.stack([toe_p, heel_p], 1)
        total = weight.sum(1)[:, None]
        fallback = np.where(total > 1e-6,
                            (points * weight[..., None]).sum(1) / np.maximum(total, 1e-9),
                            points.mean(1))
        fallback[:, 1] = y0
        force, cop, torque, undefined[foot] = plate_triple(
            toe_f + heel_f, moment, np.nan_to_num(fallback), y0)
        # No vertical load, no CoP: the comparison masks these rows on this same raw
        # force, so NaN says "no point here" without any value to be mistaken for one.
        cop[force[:, 1] <= LOADED_N] = np.nan
        blocks.append((force, cop, torque))
    return (np.stack([b[0] for b in blocks], 1), np.stack([b[1] for b in blocks], 1),
            np.stack([b[2] for b in blocks], 1), undefined)


def export(clip: Path, pred_dir: Path, run_dir: Path, kindyn_dir: Path) -> dict[str, float]:
    """Convert one clip's predictions into the scorer's contract file."""
    plates = np.load(clip / "forces.npz", allow_pickle=True)
    n_frames = len(plates["force"])
    kindyn = np.load(kindyn_dir / "kindyn_1.npz", allow_pickle=True)
    joint_names = [str(x) for x in kindyn["joint_names"]]
    mass = float(np.asarray(kindyn["total_mass"], np.float64).reshape(-1)[0])

    prediction = load_prediction(pred_dir, n_frames)
    covered, stride = prediction["covered"], prediction["stride"]
    span = covered_span(covered, stride)
    feet_world, posing_residual = contact_frame_points(prediction)
    rotation, translation = world_from_mocap(clip)
    y0 = floor_y(clip)

    # Fill the stride's holes on the INPUTS, so the force, the CoP and the free moment of
    # a filled row are one consistent wrench rather than three separately patched arrays.
    forces_n = fill_gaps(prediction["forces_bw"], covered) * mass * GRAVITY @ rotation
    joints_mocap = (fill_gaps(prediction["joints_world"], covered) - translation) @ rotation
    feet_mocap = (fill_gaps(feet_world, covered) - translation) @ rotation
    forces_n[~span] = 0.0
    force, cop, torque, undefined = foot_triples(
        forces_n, np.nan_to_num(joints_mocap), joint_names, y0)
    force[~span], torque[~span], cop[~span] = 0.0, 0.0, np.nan

    hips = np.stack([joints_mocap[:, joint_names.index(j)] for j in HIP_JOINTS], 1)
    foot_points = feet_mocap.reshape(n_frames, len(FEET), 2, 3)   # per foot [heel, toe]

    run_dir.mkdir(parents=True, exist_ok=True)
    if pred_dir.resolve() != (run_dir / "predictions").resolve():
        shutil.rmtree(run_dir / "predictions", ignore_errors=True)
        shutil.copytree(pred_dir, run_dir / "predictions")
    source = np.load(pred_dir / "forces.npz", allow_pickle=True)
    atomic_savez(
        run_dir / "estimated_forces.npz",
        limbs=np.array(FEET), force=force.astype(np.float32), cop=cop.astype(np.float32),
        free_torque=torque.astype(np.float32),
        total_mass=np.float32(mass), valid_mask=span, predicted_mask=covered,
        hips_mocap=np.where(span[:, None, None], hips, np.nan).astype(np.float32),
        foot_points_mocap=np.where(span[:, None, None, None], foot_points,
                                   np.nan).astype(np.float32),
        frame=np.str_("opensim_ground_y_up"), fps=np.float32(plates["fps"]),
        method=np.str_(METHOD),
        gap_fill=np.str_("valid_mask is the SPAN the model covered; the holes its ~25 fps "
                         "stride leaves inside that span are linearly interpolated from "
                         "the group forces and body points up. predicted_mask marks the "
                         "rows the model actually predicted"),
        foot_points_source=np.str_("heel / big-toe contact frames of the BetterHuman "
                                   "SMPL-X body, re-posed from q_cam + betas, scale 1.0"),
        posing_residual_m=np.float32(posing_residual),
        cop_undefined_rows=np.array([undefined[f] for f in FEET], np.int32),
        floor_y_m=np.float32(y0),
        R_world_from_mocap=rotation.astype(np.float32),
        t_world_from_mocap=translation.astype(np.float32),
        mass_source=np.str_(str(kindyn_dir / "kindyn_1.npz")),
        checkpoint=source["checkpoint"], checkpoint_epoch=source["checkpoint_epoch"],
        exp_name=source["exp_name"], stride=source["stride"],
        force_frame=source["force_frame"])

    loaded = span[:, None] & (force[..., 1] > LOADED_N)
    peak = float(force[span][..., 1].max()) if span.any() else np.nan
    return {"frames": n_frames, "valid": int(span.sum()), "predicted": int(covered.sum()),
            "peak_v_n": peak, "peak_v_bw": peak / (float(plates["mass_kg"]) * GRAVITY) * 100.0,
            "loaded_rows": int(loaded.sum()), "undefined": sum(undefined.values()),
            "posing_residual_mm": 1000.0 * posing_residual}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=PROCESSED, help="the processed clip root")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="root of predict_reconstruction out-trees "
                         "(default: <source>/out/_methods/learned_root)")
    ap.add_argument("--kindyn-run", default="human_optim_floorplane",
                    help="run directory the total_mass and joint names come from")
    ap.add_argument("--run", default="method_learned_L_body",
                    help="output directory under processed/out/<clip>/")
    ap.add_argument("--clips", type=Path, default=None,
                    help="file of clip names (default: every clip with predictions)")
    args = ap.parse_args(argv)

    pred_root = args.predictions or args.source / "out" / "_methods" / "learned_root"
    names = ([line.strip() for line in args.clips.read_text().splitlines() if line.strip()]
             if args.clips else
             sorted(d.name for d in pred_root.iterdir()
                    if (d / "predictions" / "forces.npz").is_file()))
    written, missing = 0, []
    for name in names:
        pred_dir = pred_root / name / "predictions"
        if not (pred_dir / "forces.npz").is_file():
            missing.append(name)
            print(f"{name}: no predictions, skipped")
            continue
        stats = export(args.source / name, pred_dir,
                       args.source / "out" / name / args.run,
                       args.source / "out" / name / args.kindyn_run)
        written += 1
        print(f"{name:34s} valid {stats['valid']:4d}/{stats['frames']:4d} "
              f"({stats['predicted']:4d} predicted), {stats['loaded_rows']:4d} foot-rows "
              f"loaded, peak Fy {stats['peak_v_n']:7.1f} N ({stats['peak_v_bw']:5.1f} %BW "
              f"measured), {stats['undefined']} CoP-free rows, posing residual "
              f"{stats['posing_residual_mm']:.2f} mm")
    print(f"\n[opencap_learned_export] {written} clip(s) -> {args.run}/estimated_forces.npz"
          + (f"; {len(missing)} without predictions: {missing}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
