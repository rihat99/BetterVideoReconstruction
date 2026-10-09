"""Turn the learned model's 35-contact-frame forces into the OpenCap plate contract.

``contact_anything``'s ``predict_reconstruction.py`` with ``data.contact_set: frames35`` writes,
per scene, ``predictions/<run>/forces_sup.npz`` (``forces_world (1, N, 35, 3)`` in BODY-WEIGHT
units in OUR world, ``anchor_cam (1, N, 35, 3)`` = the refined body's contact-frame points in
the camera, ``limbs`` = the 35 frame names, ``contact_probs``, ``valid_mask``, ``stride``) and
``predictions/<run>/smplx.npz`` (``joints_world (1, N, 52, 3)``, ``covered``). This is the
35-slot twin of ``opencap_learned_export.py`` (six groups); the helpers are shared.

**A foot's wrench.** The plates measure one wrench per foot. A foot here is every slot whose
parent is the foot or the ankle of that side — heel, big toe, inner / outer ball, toe-above —
so ``F = Σ f_i`` and the moment about the ground origin is ``Σ r_i × f_i`` with ``r_i`` the
slot's own point (``anchor_cam`` lifted into the world through ``geometry/transform.npz``),
not a joint: each slot is a real point of the sole, so the CoP is where the model put the
load. :func:`~scripts.export_opencap_mot.plate_triple` inverts that wrench on the plates'
floor plane. Hands and the other slots (sit, knees, back, …) have no plate and are not
exported; the sit slots are what keeps a seated body's feet unloaded, and they matter only
through the foot forces the model predicts.

**Newtons, frames, coverage** exactly as the six-group export: ``forces_world × total_mass
× 9.81`` with the RECONSTRUCTED mass of the clip's ``kindyn_1.npz`` (the scorer's
``MAE V own-BW`` column re-reads the same forces against the measured mass); our world →
mocap through ``meta.json``'s calibration; the stride's holes filled by linear interpolation
inside the covered span, ``valid_mask`` = the span, ``predicted_mask`` = the rows the model
actually predicted.

``foot_points_mocap`` is the heel and big-toe slot of each foot — the same landmarks the
six-group export re-poses, read straight off the dump here.

Writes ``processed/out/<clip>/<--run>/estimated_forces.npz`` plus a copy of the source
``predictions/<pred-dir>/`` beside it.

Usage::

    uv run --no-sync python scripts/methods/opencap_frames35_export.py \
        --pred-dir bedlam_frames35 --run method_bedlam_frames35 \
        --clips /…/processed/out/_methods/clips_natural.txt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_REPO / "scripts"), str(_REPO / "scripts" / "methods")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_opencap_mot import floor_y, plate_triple  # noqa: E402
from opencap_learned_export import GRAVITY, HIP_JOINTS, covered_span, fill_gaps  # noqa: E402
from opencap_plates import LOADED_N  # noqa: E402
from opencap_sensor_forces import PROCESSED, world_from_mocap  # noqa: E402
from bvr.npz_io import atomic_savez  # noqa: E402

FEET = ("left_foot", "right_foot")

#: Per foot: the slots the plates see under it (parent = the foot or the ankle joint of that
#: side, in ``contact_anything/model/contact_frames.py``'s ``frames35`` table), heel first,
#: big toe second — the two ``foot_points_mocap`` landmarks.
FOOT_SLOTS = {
    "left_foot": ("heel_left", "big_toe_left", "ball_inner_left", "ball_outer_left",
                  "toe_above_left"),
    "right_foot": ("heel_right", "big_toe_right", "ball_inner_right", "ball_outer_right",
                   "toe_above_right"),
}


def load_prediction(pred_dir: Path, n_frames: int) -> dict:
    """One scene's 35-slot predictions padded to ``n_frames`` (one tracked person)."""
    forces = np.load(pred_dir / "forces_sup.npz", allow_pickle=True)
    body = np.load(pred_dir / "smplx.npz", allow_pickle=True)
    if str(forces["contact_set"]) != "frames35":
        raise SystemExit(f"{pred_dir}: contact set is {forces['contact_set']!s}, not frames35")
    if str(forces["units"]) != "body_weight":
        raise SystemExit(f"{pred_dir}: forces are in {forces['units']!s}, not body_weight")
    slots = [str(x) for x in forces["limbs"]]
    forces_bw = np.asarray(forces["forces_world"], np.float64)
    anchors_cam = np.asarray(forces["anchor_cam"], np.float64)
    joints = np.asarray(body["joints_world"], np.float64)
    covered = np.asarray(forces["valid_mask"], bool)
    if forces_bw.shape[0] != 1:
        raise SystemExit(f"{pred_dir}: {forces_bw.shape[0]} people, expected one")
    forces_bw, anchors_cam, joints, covered = forces_bw[0], anchors_cam[0], joints[0], covered[0]
    if not np.array_equal(covered, np.asarray(body["covered"], bool)[0]):
        raise SystemExit(f"{pred_dir}: forces and body disagree about which rows are covered")
    if not (np.isfinite(forces_bw[covered]).all() and np.isfinite(joints[covered]).all()
            and np.isfinite(anchors_cam[covered]).all()):
        raise SystemExit(f"{pred_dir}: a covered row holds non-finite forces, anchors or joints")

    def pad(values: np.ndarray, fill: float) -> np.ndarray:
        if len(values) >= n_frames:
            return values[:n_frames]
        tail = np.full((n_frames - len(values), *values.shape[1:]), fill, values.dtype)
        return np.concatenate([values, tail], 0)

    forces_bw = np.where(covered[:, None, None], np.nan_to_num(forces_bw), 0.0)
    return {"slots": slots, "forces_bw": pad(forces_bw, 0.0),
            "anchors_cam": pad(anchors_cam, np.nan), "joints_world": pad(joints, np.nan),
            "covered": pad(covered, False), "stride": int(forces["stride"]),
            "frame_indices": np.asarray(forces["frame_indices"], int)}


def anchors_to_world(anchors_cam: np.ndarray, out_clip: Path, frame_indices: np.ndarray,
                     n_frames: int) -> np.ndarray:
    """``(N, S, 3)`` camera-frame slot points → OUR world through ``geometry/transform.npz``."""
    transform = np.load(out_clip / "geometry" / "transform.npz", allow_pickle=True)
    cam_from_world = np.asarray(transform["extrinsics"], np.float64)          # (F, 4, 4)
    rows = np.asarray(transform["frame_indices"], int)
    if not np.array_equal(rows[:n_frames], np.arange(min(len(rows), n_frames))):
        raise SystemExit(f"{out_clip.name}: transform.npz rows are not video frames 0..N-1")
    if len(cam_from_world) < n_frames:
        raise SystemExit(f"{out_clip.name}: transform.npz covers {len(cam_from_world)} of "
                         f"{n_frames} frames")
    rotation = cam_from_world[:n_frames, :3, :3]
    translation = cam_from_world[:n_frames, :3, 3]
    return np.einsum("nji,nsj->nsi", rotation, anchors_cam - translation[:, None])


def foot_triples(forces_n: np.ndarray, points: np.ndarray, slots: list[str], y0: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Per-foot ``(force, cop, free_torque)`` blocks ``(N, 2, 3)`` plus the undefined-row count.

    :param forces_n: ``(N, 35, 3)`` slot forces in newtons, mocap frame.
    :param points: ``(N, 35, 3)`` slot points in the mocap frame.
    :param y0: the clip's floor plane, metres.
    """
    blocks, undefined = [], {}
    for foot in FEET:
        cols = [slots.index(name) for name in FOOT_SLOTS[foot]]
        f, r = forces_n[:, cols], points[:, cols]
        force = f.sum(1)
        moment = np.cross(r, f).sum(1)
        weight = np.linalg.norm(f, axis=-1)[..., None]
        total = weight.sum(1)
        fallback = np.where(total > 1e-6, (r * weight).sum(1) / np.maximum(total, 1e-9),
                            r.mean(1))
        fallback[:, 1] = y0
        force, cop, torque, undefined[foot] = plate_triple(
            force, moment, np.nan_to_num(fallback), y0)
        cop[force[:, 1] <= LOADED_N] = np.nan
        blocks.append((force, cop, torque))
    return (np.stack([b[0] for b in blocks], 1), np.stack([b[1] for b in blocks], 1),
            np.stack([b[2] for b in blocks], 1), undefined)


def export(clip: Path, out_clip: Path, pred_dir: Path, run_dir: Path,
           kindyn_dir: Path) -> dict[str, float]:
    """Convert one clip's predictions into the scorer's contract file."""
    plates = np.load(clip / "forces.npz", allow_pickle=True)
    n_frames = len(plates["force"])
    kindyn = np.load(kindyn_dir / "kindyn_1.npz", allow_pickle=True)
    joint_names = [str(x) for x in kindyn["joint_names"]]
    mass = float(np.asarray(kindyn["total_mass"], np.float64).reshape(-1)[0])

    prediction = load_prediction(pred_dir, n_frames)
    covered, stride, slots = prediction["covered"], prediction["stride"], prediction["slots"]
    span = covered_span(covered, stride)
    anchors_world = anchors_to_world(prediction["anchors_cam"], out_clip,
                                     prediction["frame_indices"], n_frames)
    rotation, translation = world_from_mocap(clip)
    y0 = floor_y(clip)

    forces_n = fill_gaps(prediction["forces_bw"], covered) * mass * GRAVITY @ rotation
    points_mocap = (fill_gaps(anchors_world, covered) - translation) @ rotation
    joints_mocap = (fill_gaps(prediction["joints_world"], covered) - translation) @ rotation
    forces_n[~span] = 0.0
    force, cop, torque, undefined = foot_triples(forces_n, np.nan_to_num(points_mocap), slots, y0)
    force[~span], torque[~span], cop[~span] = 0.0, 0.0, np.nan

    hips = np.stack([joints_mocap[:, joint_names.index(j)] for j in HIP_JOINTS], 1)
    foot_points = np.stack([points_mocap[:, [slots.index(FOOT_SLOTS[f][0]),
                                             slots.index(FOOT_SLOTS[f][1])]] for f in FEET], 1)

    run_dir.mkdir(parents=True, exist_ok=True)
    if pred_dir.resolve() != (run_dir / "predictions").resolve():
        shutil.rmtree(run_dir / "predictions", ignore_errors=True)
        shutil.copytree(pred_dir, run_dir / "predictions")
    source = np.load(pred_dir / "forces_sup.npz", allow_pickle=True)
    atomic_savez(
        run_dir / "estimated_forces.npz",
        limbs=np.array(FEET), force=force.astype(np.float32), cop=cop.astype(np.float32),
        free_torque=torque.astype(np.float32),
        total_mass=np.float32(mass), valid_mask=span, predicted_mask=covered,
        hips_mocap=np.where(span[:, None, None], hips, np.nan).astype(np.float32),
        foot_points_mocap=np.where(span[:, None, None, None], foot_points,
                                   np.nan).astype(np.float32),
        frame=np.str_("opensim_ground_y_up"), fps=np.float32(plates["fps"]),
        method=np.str_(f"contact_anything {source['exp_name']!s} (frames35), "
                       f"{source['checkpoint']!s}"),
        gap_fill=np.str_("valid_mask is the SPAN the model covered; the holes its stride "
                         "leaves inside that span are linearly interpolated from the slot "
                         "forces and points up. predicted_mask marks the rows the model "
                         "actually predicted"),
        foot_slots=np.array([",".join(FOOT_SLOTS[f]) for f in FEET]),
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
            "loaded_rows": int(loaded.sum()), "undefined": sum(undefined.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=PROCESSED, help="the processed clip root")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="root of predict_reconstruction out-trees "
                         "(default: <source>/out/_methods/learned_root)")
    ap.add_argument("--pred-dir", required=True,
                    help="the run's subdirectory under <clip>/predictions/")
    ap.add_argument("--run", required=True, help="output directory under processed/out/<clip>/")
    ap.add_argument("--kindyn-run", default="human_optim_floorplane",
                    help="run directory the total_mass and joint names come from")
    ap.add_argument("--clips", type=Path, default=None,
                    help="file of clip names (default: every clip with predictions)")
    args = ap.parse_args(argv)

    pred_root = args.predictions or args.source / "out" / "_methods" / "learned_root"
    names = ([line.strip() for line in args.clips.read_text().splitlines() if line.strip()]
             if args.clips else
             sorted(d.name for d in pred_root.iterdir()
                    if (d / "predictions" / args.pred_dir / "forces_sup.npz").is_file()))
    written, missing = 0, []
    for name in names:
        pred_dir = pred_root / name / "predictions" / args.pred_dir
        if not (pred_dir / "forces_sup.npz").is_file():
            missing.append(name)
            print(f"{name}: no predictions, skipped")
            continue
        stats = export(args.source / name, args.source / "out" / name, pred_dir,
                       args.source / "out" / name / args.run,
                       args.source / "out" / name / args.kindyn_run)
        written += 1
        print(f"{name:34s} valid {stats['valid']:4d}/{stats['frames']:4d} "
              f"({stats['predicted']:4d} predicted), {stats['loaded_rows']:4d} foot-rows "
              f"loaded, peak Fy {stats['peak_v_n']:7.1f} N ({stats['peak_v_bw']:5.1f} %BW "
              f"measured), {stats['undefined']} CoP-free rows")
    print(f"\n[opencap_frames35_export] {written} clip(s) -> {args.run}/estimated_forces.npz"
          + (f"; {len(missing)} without predictions: {missing}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
