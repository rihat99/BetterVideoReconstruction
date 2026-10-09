"""PhysPT's ground reaction forces in the OpenCap plate contract.

``other/PhysPT/opencap/<clip>_output.npz`` holds, in PhysPT's own z-up world, the refined
SMPL mesh (``physics_vertices_world``, one row per frame it produced) and a force at each
of its 29 contact vertices (``pred_grf``, two rows fewer — the spring-mass model needs a
velocity). Eight of those vertices are on the feet: SMPL parts 7/10 (left heel and toe)
and 8/11 (right). Their forces summed per foot give the foot's force, and the vertex
positions are the points of application, so the wrench about the ground origin is
``Σ r_i × f_i`` — from which :mod:`scripts.opencap_plates` recovers the plate triple
(COP on the floor plane, vertical free moment).

**Frames.** PhysPT's world is its own: the camera's heading turned z-up by a fixed
rotation, then a per-window correction from ``GlobalTrajPredictor``, with the origin at
the first frame's pelvis. Its gravity axis is +z BY CONSTRUCTION, so the map into the
mocap frame is not a free 3D rotation: :data:`Z_UP_TO_Y_UP` takes +z onto the mocap
vertical, and only a yaw about that vertical and a translation are fitted, to the mocap
markers of the same anatomy (the ``JOINT_PAIRS`` of ``scripts/sync_opencap.py``). A free
rotation fitted on the same points tilts by up to 10°, which would leak vertical force
into the AP axis; it is kept in the npz as ``R_free_fit`` with its tilt for provenance
only. The fit residual is the check that the alignment is real.

**Rate.** The inputs were written at 20 Hz (every third 60 fps video frame), the rate
PhysPT's released weights assume — see ``physpt_inputs.py``. Rows therefore land
on video frames 0, 3, 6, … The foot WRENCH (force and its moment about the ground origin,
the quantity that may be averaged) and the body geometry are linearly resampled onto all
60 fps video frames, and the plate triple is inverted from the resampled wrench.
``valid_mask`` covers the whole span PhysPT covered, interpolated frames included —
scoring only on its own timestamps would hide the impact peaks that fall between them.
``predicted_mask`` keeps the frames it actually produced.

Writes ``processed/out/<clip>/method_physpt/estimated_forces.npz``. Run from
BetterVideoReconstruction::

    uv run --no-sync python scripts/methods/opencap_physpt_export.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_opencap_mot import floor_y, plate_triple  # noqa: E402
from opencap_plates import LOADED_N  # noqa: E402
from bvr.npz_io import atomic_savez  # noqa: E402
from bvr.paths import LABVALIDATION  # noqa: E402

PROCESSED = (LABVALIDATION / "processed").resolve()
PHYSPT = (_REPO.parent / "other" / "PhysPT").resolve()
CLIP_LIST = PROCESSED / "out" / "_methods" / "clips_natural.txt"

#: SMPL joint ↔ mocap marker, the same anatomy on both, for the per-clip alignment.
JOINT_PAIRS = ((1, "L_HJC"), (2, "R_HJC"), (4, "L_knee"), (5, "r_knee"),
               (7, "L_ankle"), (8, "r_ankle"), (16, "L_Shoulder"), (17, "R_Shoulder"))

#: PhysPT's world is z-up, the mocap frame y-up. This takes one onto the other; what is
#: left to fit is a yaw about the shared vertical. Any base with ``+z ↦ +y`` would do —
#: the yaw absorbs the difference — so the choice of the other two axes is immaterial.
Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

#: The SMPL body parts PhysPT's contact vertices belong to, per foot, in ``limbs`` order.
FOOT_PARTS = {"left_foot": {7, 10}, "right_foot": {8, 11}}

#: Heel and toe of each foot as the contact vertices of the heel part and the toe part —
#: the points the forces are applied at, on the sole (rest pose: ~1 cm above the lowest
#: mesh vertex, the heel group 5 cm behind and the toe group 11 cm ahead of the ankle).
FOOT_VERTEX_PARTS = {"left_foot": (7, 10), "right_foot": (8, 11)}

#: Left and right hip as SMPL joints.
HIP_JOINTS = (1, 2)

GRAVITY = 9.81


def physpt_model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PhysPT's SMPL joint regressor ``(24, 6890)``, its 29 contact vertices and their parts."""
    with open(PHYSPT / "assets" / "data" / "smpl" / "neutral.pkl", "rb") as handle:
        regressor = np.asarray(pickle.load(handle)["V_regressor"], np.float64)
    with open(PHYSPT / "assets" / "data" / "physics.pkl", "rb") as handle:
        physics = pickle.load(handle)
    vertices = np.asarray(physics["contact_vertexidxall"], np.int64)
    parts = np.concatenate([[part] * len(v) for part, v in enumerate(physics["contact_vertexidx"])])
    return regressor, vertices, parts.astype(np.int64)


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rotation, translation, residual)`` minimising ``‖R s + t − d‖`` over ALL rotations.

    Kept for provenance only — :func:`yaw_fit` is what the export uses.

    :param source: ``(K, 3)`` points in the method's world.
    :param target: ``(K, 3)`` the same anatomy measured in the mocap frame.
    """
    source_mean, target_mean = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation, np.linalg.norm(source @ rotation.T + translation - target, axis=1)


def yaw_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rotation, translation, residual)`` with gravity kept: yaw about ``+y`` only.

    ``source`` is first turned z-up → y-up by :data:`Z_UP_TO_Y_UP`; the yaw that then best
    matches ``target`` maximises ``tr(R_y H)`` with ``H = Σ s d^T``, which for a rotation
    about ``y`` is ``cos θ (H₀₀ + H₂₂) + sin θ (H₂₀ − H₀₂)`` — maximal at
    ``θ = atan2(H₂₀ − H₀₂, H₀₀ + H₂₂)``.
    """
    upright = source @ Z_UP_TO_Y_UP.T
    source_mean, target_mean = upright.mean(0), target.mean(0)
    h = (upright - source_mean).T @ (target - target_mean)
    angle = np.arctan2(h[2, 0] - h[0, 2], h[0, 0] + h[2, 2])
    cos, sin = np.cos(angle), np.sin(angle)
    yaw = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    rotation = yaw @ Z_UP_TO_Y_UP
    translation = target_mean - yaw @ source_mean
    return rotation, translation, np.linalg.norm(source @ rotation.T + translation - target, axis=1)


def align(joints: np.ndarray, markers: np.ndarray, names: list[str]) -> dict:
    """PhysPT's world → the mocap frame, from its SMPL joints and the mocap markers.

    :param joints: ``(T, 24, 3)`` PhysPT's SMPL joints over the frames it produced.
    :param markers: ``(T, M, 3)`` the mocap markers of those same video frames.
    :returns: the gravity-preserving ``rotation``/``translation``/``residual`` (m) that the
        export uses, and the free 3D fit's rotation, residual and tilt for provenance.
    """
    source = np.concatenate([joints[:, j] for j, _ in JOINT_PAIRS], 0)
    target = np.concatenate([markers[:, names.index(m)] for _, m in JOINT_PAIRS], 0)
    finite = np.isfinite(source).all(1) & np.isfinite(target).all(1)
    rotation, translation, residual = yaw_fit(source[finite], target[finite])
    free, _, free_residual = rigid_fit(source[finite], target[finite])
    up = free @ np.array([0.0, 0.0, 1.0])
    tilt = float(np.degrees(np.arccos(np.clip(up @ np.array([0.0, 1.0, 0.0]), -1.0, 1.0))))
    return {"rotation": rotation, "translation": translation, "residual": float(residual.mean()),
            "free": free, "free_residual": float(free_residual.mean()), "free_tilt_deg": tilt}


def resample(values: np.ndarray, frames: np.ndarray, n_frames: int) -> np.ndarray:
    """``(T, ...)`` sampled at video frames ``frames`` → ``(n_frames, ...)``, linearly.

    Exact at the sample points. Outside the covered range the end value is held, which is
    what ``np.interp`` does; those frames are invalid and a step to zero would ring
    through the scorer's low-pass filter.
    """
    grid = np.arange(n_frames, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    out = np.stack([np.interp(grid, frames, flat[:, k]) for k in range(flat.shape[1])], 1)
    return out.reshape(n_frames, *values.shape[1:])


def export(clip: str, source: Path, run: str, physpt_dir: Path) -> dict[str, float]:
    """Convert one clip's PhysPT output into the plate contract; returns the sanity numbers."""
    saved = np.load(physpt_dir / f"{clip}_output.npz")
    written = json.loads((physpt_dir / f"{clip}_sam3d.json").read_text())
    regressor, vertices, parts = physpt_model()

    force_rows = np.asarray(saved["pred_grf"][0], np.float64)          # (T, 29, 3) newtons
    mesh = np.asarray(saved["physics_vertices_world"], np.float64)[:len(force_rows)]
    video_frames = np.asarray(written["video_frames"], np.int64)[:len(force_rows)]
    joints = np.einsum("jv,tvc->tjc", regressor, mesh)                 # (T, 24, 3)

    clip_dir = source / clip
    plates = np.load(clip_dir / "forces.npz", allow_pickle=True)
    n_frames = len(plates["force"])
    markers = np.load(clip_dir / "markers.npz", allow_pickle=True)
    marker_names = [str(x) for x in markers["names"]]
    fit = align(joints, np.asarray(markers["markers"], np.float64)[video_frames], marker_names)

    rotation, translation = fit["rotation"], fit["translation"]
    to_mocap = lambda p: p @ rotation.T + translation  # noqa: E731
    points = to_mocap(mesh[:, vertices])                               # (T, 29, 3)
    y0 = floor_y(clip_dir)
    frames = video_frames.astype(np.float64)

    limbs = list(FOOT_PARTS)
    force, cop, free_torque, undefined = [], [], [], {}
    for limb in limbs:
        columns = [c for c, part in enumerate(parts) if part in FOOT_PARTS[limb]]
        foot_force = force_rows[:, columns] @ rotation.T               # (T, C, 3) mocap
        foot_point = points[:, columns]
        weight = np.linalg.norm(foot_force, axis=-1)[..., None]
        centre = (foot_point * weight).sum(1) / np.maximum(weight.sum(1), 1e-9)
        blind = weight.sum(1)[:, 0] <= 1e-9
        centre[blind] = foot_point[blind].mean(1)                      # no force: the foot itself
        # The wrench is what may be averaged between PhysPT's own samples (mean p × F is
        # not mean p × mean F), so the triple is inverted from the RESAMPLED wrench.
        total = resample(foot_force.sum(1), frames, n_frames)
        moment = resample(np.cross(foot_point, foot_force).sum(1), frames, n_frames)
        fallback = resample(centre, frames, n_frames)
        fallback[:, 1] = y0
        _, placed, torque, undefined[limb] = plate_triple(total, moment, fallback, y0)
        carried = (total[:, 1] > LOADED_N) | (np.linalg.norm(total, axis=-1) >= LOADED_N)
        force.append(total)
        cop.append(np.where(carried[:, None], placed, np.nan))
        free_torque.append(torque)

    predicted = np.zeros(n_frames, bool)
    predicted[video_frames[video_frames < n_frames]] = True
    valid = np.zeros(n_frames, bool)
    valid[int(video_frames[0]):min(int(video_frames[-1]) + 1, n_frames)] = True

    hips = resample(to_mocap(joints[:, list(HIP_JOINTS)]), frames, n_frames)
    sole = {part: to_mocap(mesh[:, [v for v, q in zip(vertices, parts) if q == part]]).mean(1)
            for limb in limbs for part in FOOT_VERTEX_PARTS[limb]}
    heel_toe = np.stack([np.stack([sole[part] for part in FOOT_VERTEX_PARTS[limb]], 1)
                         for limb in limbs], 1)                        # (T, 2, 2, 3)
    foot_points = resample(heel_toe, frames, n_frames)

    mass = float(saved["mass_kg"])
    run_dir = source / "out" / clip / run
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        run_dir / "estimated_forces.npz",
        limbs=np.array(limbs),
        force=np.stack(force, 1).astype(np.float32),
        cop=np.stack(cop, 1).astype(np.float32),
        free_torque=np.stack(free_torque, 1).astype(np.float32),
        total_mass=np.float32(mass),
        valid_mask=valid, predicted_mask=predicted,
        hips_mocap=hips.astype(np.float32),
        foot_points_mocap=foot_points.astype(np.float32),
        frame=np.str_("opensim_ground_y_up"), fps=np.float32(60.0),
        method=np.str_("physpt"),
        physpt_dt=np.float32(saved["dt"]), physpt_fps=np.float32(written["physpt_fps"]),
        physpt_frames=video_frames.astype(np.int32),
        R_mocap_from_physpt=rotation.astype(np.float32),
        t_mocap_from_physpt=translation.astype(np.float32),
        R_free_fit=fit["free"].astype(np.float32),
        align_residual_m=np.float32(fit["residual"]),
        free_fit_residual_m=np.float32(fit["free_residual"]),
        free_fit_tilt_deg=np.float32(fit["free_tilt_deg"]),
        gravity_angle_deg=np.float32(0.0),
        floor_y_m=np.float32(y0),
        cop_undefined_rows=np.array([undefined[k] for k in limbs], np.int32))

    peak = float(np.stack(force, 1)[valid][..., 1].sum(1).max())
    body_weight = float(plates["mass_kg"]) * GRAVITY
    print(f"{clip}: {len(force_rows)} rows -> {int(valid.sum())}/{n_frames} valid frames, "
          f"mass {mass:.1f} kg, residual {fit['residual'] * 1e3:6.1f} mm "
          f"(free fit {fit['free_residual'] * 1e3:6.1f} mm, tilt {fit['free_tilt_deg']:4.1f}°), "
          f"peak total Fv {peak:5.0f} N ({peak / body_weight * 100:4.0f} %BW)")
    return {"clip": clip, "residual_mm": fit["residual"] * 1e3,
            "free_residual_mm": fit["free_residual"] * 1e3, "tilt_deg": fit["free_tilt_deg"],
            "mass_kg": mass, "peak_bw": peak / body_weight * 100.0,
            "rows": len(force_rows), "valid": int(valid.sum()), "n_frames": n_frames}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=PROCESSED)
    ap.add_argument("--clips", type=Path, default=CLIP_LIST)
    ap.add_argument("--run", default="method_physpt")
    ap.add_argument("--physpt-dir", type=Path, default=PHYSPT / "opencap",
                    help="where PhysPT's own run_opencap.py wrote <clip>_output.npz")
    args = ap.parse_args(argv)

    rows, missing = [], []
    for clip in args.clips.read_text().split():
        if not (args.physpt_dir / f"{clip}_output.npz").exists():
            missing.append(clip)
            continue
        rows.append(export(clip, args.source, args.run, args.physpt_dir))
    residual = np.array([r["residual_mm"] for r in rows])
    free = np.array([r["free_residual_mm"] for r in rows])
    tilt = np.array([r["tilt_deg"] for r in rows])
    print(f"\n[opencap_physpt_export] {len(rows)} clip(s); missing: {missing}")
    print(f"  yaw-fit residual   mean {residual.mean():.0f} mm  median "
          f"{np.median(residual):.0f}  max {residual.max():.0f} "
          f"({rows[int(residual.argmax())]['clip']})")
    print(f"  free-fit residual  mean {free.mean():.0f} mm  median {np.median(free):.0f}  "
          f"max {free.max():.0f}  (tilt mean {tilt.mean():.1f}° max {tilt.max():.1f}°)")
    print(f"  valid frames       {sum(r['valid'] for r in rows)} of "
          f"{sum(r['n_frames'] for r in rows)} video frames")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
