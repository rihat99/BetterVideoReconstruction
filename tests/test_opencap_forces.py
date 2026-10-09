"""The OpenCap force bridge: the COP / free-torque inversion and the plate→world rotation.

The export writes our solved point forces as the dataset's own (force, COP, free moment)
triple, and the triple is only right if the inversion in
:func:`scripts.export_opencap_mot.cop_free_torque` is right sign for sign. Two checks:
synthetically, that any wrench survives triple → wrench → triple; and on the DATASET's own
force files, that its shipped triples come back to themselves. A third covers the other
direction of the same calibration — ``scripts/opencap_sensor_forces.py`` carrying the
plates into our world.

Everything here is float64 on purpose: the code under test is CPU numpy that parses text
files of 8-decimal newtons, and the point of the round-trip is to catch an algebra error
(which shows at 1e-3) rather than a precision one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_opencap_mot import COLUMNS, plate_triple, write_mot  # noqa: E402
from opencap_plates import LOADED_N, cop_free_torque, origin_moment  # noqa: E402
from opencap_sensor_forces import main as sensor_forces_main  # noqa: E402
from bvr.paths import LABVALIDATION  # noqa: E402

#: A plate triple that has no COP does not need a fallback point; pass an unused one.
NO_FALLBACK = None


def triple(force: np.ndarray, moment: np.ndarray, y0: float = 0.006, fallback=None):
    """:func:`plate_triple` with a fallback point that is never reachable by accident."""
    if fallback is None:
        fallback = np.full_like(force, np.nan)
    return plate_triple(force, moment, fallback, y0)

#: The dataset the round-trip reads; the synthetic checks run without it.
LAB_VALIDATION = LABVALIDATION

FLOOR_Y = 0.006


def read_mot(path: Path) -> tuple[list[str], np.ndarray]:
    """``(column names, (rows, columns) values)`` of an OpenSim ``.mot``."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    end = next(i for i, line in enumerate(lines) if line.strip().startswith("endheader"))
    names = [c for c in lines[end + 1].split("\t") if c.strip()]
    values = np.array([[float(x) for x in line.split("\t") if x.strip()]
                       for line in lines[end + 2:] if line.strip()])
    return names, values


def test_cop_free_torque_inverts_an_arbitrary_wrench() -> None:
    """Triple → wrench about the origin → triple returns the triple it started from."""
    rng = np.random.default_rng(0)
    force = rng.normal(0.0, 300.0, (500, 3))
    force[:, 1] = rng.uniform(5 * LOADED_N, 1500.0, 500)                 # always loaded
    cop = np.stack([rng.uniform(-1.0, 1.0, 500), np.full(500, FLOOR_Y),
                    rng.uniform(-1.0, 1.0, 500)], 1)
    free = np.stack([np.zeros(500), rng.normal(0.0, 20.0, 500), np.zeros(500)], 1)

    moment = origin_moment(cop, force, free)
    force_back, cop_back, free_back, undefined = triple(force, moment, FLOOR_Y)
    assert undefined == 0
    assert np.abs(force_back - force).max() == 0.0
    assert np.abs(cop_back - cop).max() < 1e-9
    assert np.abs(free_back - free).max() < 1e-9
    # and the wrench itself is preserved, which is the property the export actually needs
    assert np.abs(np.cross(cop_back, force_back) + free_back - moment).max() < 1e-9


def test_the_three_row_cases_of_the_export() -> None:
    """Carrying nothing → zeros; ordinary load → the inversion; no vertical load → fallback.

    The third case is the one that used to delete real force: a foot pushing almost purely
    sideways clears the load threshold in ``‖F‖`` but not in ``F_y``, so its line of action
    never meets the floor. It must keep its force and take the caller's point.
    """
    # row 0: carrying nothing. row 1: an ordinary load. row 2: 354 N almost all sideways —
    # the subject3_walking1 right foot, frame 47, that the old rule threw away.
    force = np.array([[10.0, 1.0, -5.0], [10.0, 500.0, -5.0], [-353.728, 12.709, 11.844]])
    moment = np.array([[3.0, 1.0, 2.0], [3.0, 1.0, 2.0], [40.0, -6.0, -15.0]])
    fallback = np.array([[9.0, FLOOR_Y, 9.0], [9.0, FLOOR_Y, 9.0], [0.4, FLOOR_Y, -0.3]])
    force_back, cop, free, undefined = triple(force, moment, FLOOR_Y, fallback)

    assert undefined == 1
    assert not force_back[0].any() and not cop[0].any() and not free[0].any()
    assert cop[1].any() and free[1].any()
    assert np.allclose(force_back[2], force[2]), "a real force must survive the export"
    assert np.allclose(cop[2], fallback[2]), "with no COP, the caller's point is used"
    # the vertical moment about that point is preserved; the horizontal one is dropped
    assert abs((np.cross(cop[2], force_back[2]) + free[2])[1] - moment[2][1]) < 1e-9
    assert np.isfinite(np.concatenate([force_back, cop, free])).all()


def test_the_floor_height_pairs_with_the_right_force_components() -> None:
    """The ``y0`` terms pair ``F_z`` with ``p_z`` and ``F_x`` with ``p_x``, not the other way.

    A tilted force makes the two pairings differ; the test asserts both that ours returns
    the COP it started from and that the swapped variant does NOT, so the one derivation
    error this inversion can hide is pinned down rather than merely unexercised.
    """
    force = np.array([[-260.0, 800.0, 140.0]])
    cop = np.array([[0.3, FLOOR_Y, -0.2]])
    moment = np.cross(cop, force)
    _, cop_back, free_back, _ = triple(force, moment, FLOOR_Y)
    assert np.abs(cop_back - cop).max() < 1e-12
    assert np.abs(free_back).max() < 1e-12

    fx, fy, fz = force.T
    swapped = np.stack([(moment[:, 2] + FLOOR_Y * fz) / fy, np.full_like(fx, FLOOR_Y),
                        (FLOOR_Y * fx - moment[:, 0]) / fy], 1)
    assert np.abs(swapped - cop).max() > 1e-4, "the swapped pairing must be detectable here"


def test_write_mot_round_trips_through_the_reader(tmp_path: Path) -> None:
    """The written file has the dataset's 19 columns and parses back to its own numbers."""
    rng = np.random.default_rng(1)
    table = np.concatenate([np.arange(20)[:, None] / 60.0, rng.normal(0.0, 100.0, (20, 18))], 1)
    path = tmp_path / "estimated_forces.mot"
    write_mot(path, table)
    names, values = read_mot(path)
    assert names == COLUMNS
    assert values.shape == table.shape
    assert np.abs(values - table).max() < 1e-7


@pytest.mark.parametrize("trial", ["walking1", "squats1", "DJ1", "STS1"])
def test_dataset_force_files_round_trip(trial: str) -> None:
    """The dataset's own (force, COP, free moment) triples come back to themselves.

    This is the check that the inversion matches the convention the DATA was written
    with, not merely a self-consistent one of our own.
    """
    path = LAB_VALIDATION / "subject2" / "ForceData" / f"{trial}_forces.mot"
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")     # external dataset, not an edge case
    names, values = read_mot(path)
    for side in ("R", "L"):
        column = lambda stem: values[:, names.index(f"{side}_ground_{stem}")]  # noqa: E731
        force = np.stack([column(f"force_v{a}") for a in "xyz"], 1)
        cop = np.stack([column(f"force_p{a}") for a in "xyz"], 1)
        free = np.stack([column(f"torque_{a}") for a in "xyz"], 1)
        loaded = force[:, 1] > LOADED_N
        assert loaded.any(), f"{trial} {side} is never loaded"
        force, cop, free = force[loaded], cop[loaded], free[loaded]
        assert np.abs(free[:, [0, 2]]).max() == 0.0, "a plate's free moment is vertical only"

        moment = origin_moment(cop, force, free)
        _, cop_back, free_back, _ = triple(force, moment, float(np.median(cop[:, 1])))
        assert np.abs(cop_back - cop).max() < 1e-6
        assert np.abs(free_back - free).max() < 1e-6


def _rotation(axis: list[float], degrees: float) -> np.ndarray:
    """A rotation matrix about ``axis`` — a non-trivial, non-axis-aligned test frame."""
    unit = np.asarray(axis, float) / np.linalg.norm(axis)
    theta = np.radians(degrees)
    cross = np.array([[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]])
    return np.eye(3) + np.sin(theta) * cross + (1 - np.cos(theta)) * cross @ cross


def test_sensor_forces_rotates_the_plates_and_masks_the_unseen(tmp_path: Path) -> None:
    """The plate vectors land in our world, uncaptured limb-frames land as NaN."""
    from bvr.human_optim.boards import LIMBS

    rotation = _rotation([0.3, 1.0, 0.2], 118.0) @ _rotation([1.0, 0.0, 0.0], 168.0)
    translation = np.array([0.42, 1.31, 4.05])
    rng = np.random.default_rng(2)
    force = rng.normal(0.0, 200.0, (12, 4, 3))
    cop = rng.normal(0.0, 0.5, (12, 4, 3))
    free = rng.normal(0.0, 5.0, (12, 4, 3))
    captured = np.ones((12, 4), bool)
    captured[:3, 2:] = False

    clip = tmp_path / "subject2_walking1_cam1"
    clip.mkdir()
    np.savez(clip / "forces.npz", force=force, cop=cop, free_torque=free, captured=captured,
             limbs=np.array(LIMBS), fps=np.float32(60.0), mass_kg=np.float32(72.3),
             height_m=np.float32(1.81), source=np.str_("synthetic"))
    (clip / "meta.json").write_text(json.dumps({"R_cam_from_mocap": rotation.tolist(),
                                                "t_cam_from_mocap": translation.tolist()}))
    out = tmp_path / "out" / clip.name
    (out / "geocalib").mkdir(parents=True)
    np.savez(out / "geocalib" / "gravity.npz", gravity_world=rotation @ np.array([0.0, -1.0, 0.0]))

    assert sensor_forces_main(["--source", str(tmp_path)]) == 0
    sensor = np.load(out / "human_optim" / "sensor_forces.npz", allow_pickle=True)
    seen = captured[..., None]
    assert np.allclose(sensor["reaction_world"][captured], (force @ rotation.T)[captured], atol=1e-3)
    assert np.allclose(sensor["cop_world"][captured],
                       (cop @ rotation.T + translation)[captured], atol=1e-5)
    assert np.allclose(sensor["free_torque_world"][captured],
                       (free @ rotation.T)[captured], atol=1e-4)
    assert np.isnan(np.asarray(sensor["reaction_world"])[~np.broadcast_to(seen, force.shape)]).all()
    assert np.isnan(sensor["total_world"][:3]).all() and np.isfinite(sensor["total_world"][3:]).all()
    assert bool(sensor["world_aligned"])
    assert float(sensor["gravity_check_deg"]) < 1e-3


def test_best_lag_finds_a_known_shift_and_refuses_an_unresolved_one() -> None:
    """A known delay comes back with the documented sign; a flat optimum comes back unresolved.

    Two straight ramps are perfectly aligned, yet every shift of them correlates exactly as
    well, so the argmax lands on the scan boundary and means nothing. Reporting it as a
    measured lag is how a "0 = the sync holds" reading goes wrong, so the boundary is NaN.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "diagnostics"))
    from compare_opencap_forces import MAX_LAG, best_lag

    time = np.arange(40.0)
    bump = lambda centre: np.exp(-0.5 * ((time - centre) / 2.0) ** 2)  # noqa: E731
    lag, correlation = best_lag(bump(12.0), bump(8.0))
    assert lag == 4 and correlation > 0.99, "ours peaks 4 frames after the plates: lag = +4"
    assert best_lag(bump(8.0), bump(12.0))[0] == -4

    for length in range(2, 2 * MAX_LAG + 2):     # shorter than the scan window on both sides
        lag, _ = best_lag(np.arange(float(length)), np.arange(float(length)))
        assert not np.isfinite(lag) or abs(lag) < MAX_LAG


def test_resampling_must_average_the_wrench_not_the_cop() -> None:
    """A moving COP: averaging the triple's parts loses the wrench, averaging it keeps it.

    ``mean(p × F) ≠ mean(p) × mean(F)``. The importer therefore averages force and origin
    moment and re-derives the triple; this pins that the naive route really is wrong, so
    the extra step cannot be "simplified" away again.
    """
    samples = 33                                           # one 60 Hz bin at 2000 Hz
    rng = np.random.default_rng(3)
    force = np.stack([rng.normal(0, 60.0, samples), rng.uniform(200.0, 900.0, samples),
                      rng.normal(0, 60.0, samples)], 1)
    cop = np.stack([np.linspace(-0.10, 0.18, samples), np.full(samples, FLOOR_Y),
                    np.linspace(0.05, -0.12, samples)], 1)
    free = np.stack([np.zeros(samples), rng.normal(0, 4.0, samples), np.zeros(samples)], 1)

    want_force = force.mean(0, keepdims=True)
    want_moment = origin_moment(cop, force, free).mean(0, keepdims=True)
    got_cop, got_free, defined = cop_free_torque(want_force, want_moment, FLOOR_Y)
    assert defined.all()
    assert np.abs(origin_moment(got_cop, want_force, got_free) - want_moment).max() < 1e-9

    naive_moment = origin_moment(cop.mean(0, keepdims=True), want_force,
                                 free.mean(0, keepdims=True))
    assert np.abs(naive_moment - want_moment).max() > 1e-3, "the naive route must differ"



def test_part_ranges_split_long_trials_evenly():
    from import_opencap import part_ranges
    assert part_ranges(839) == [(0, 839)]
    assert part_ranges(900) == [(0, 900)]
    assert part_ranges(1312) == [(0, 656), (656, 1312)]
    parts = part_ranges(1900)
    assert parts == [(0, 633), (633, 1267), (1267, 1900)]
    assert all(b - a <= 900 for a, b in parts)


def _step_mot(step_s: float, seconds: float = 2.0):
    """A 2000 Hz plate file whose right plate loads to 700 N at ``step_s``."""
    from import_opencap import FORCE_FPS
    time = np.arange(int(seconds * FORCE_FPS)) / FORCE_FPS
    columns = ["time"] + [f"{side}_ground_{kind}{a}" for side in "RL"
                          for kind in ("force_v", "force_p", "torque_") for a in "xyz"]
    rows = np.zeros((len(time), len(columns)))
    rows[:, 0] = time
    rows[time >= step_s, columns.index("R_ground_force_vy")] = 700.0
    rows[:, columns.index("R_ground_force_py")] = FLOOR_Y
    return columns, rows


def test_resampling_honours_the_measured_time_offset() -> None:
    """Video frame i is mocap time ``i / 60 + offset``: the plate step moves accordingly."""
    from import_opencap import FPS, LIMBS, _resample_forces, _resample_markers
    columns, rows = _step_mot(step_s=1.0)
    right = LIMBS.index("right_foot")
    force, *_ = _resample_forces(columns, rows, 90, "walking1")
    assert np.argmax(force[:, right, 1] > LOADED_N) == 60
    force, *_ = _resample_forces(columns, rows, 90, "walking1", offset_s=0.5)
    assert np.argmax(force[:, right, 1] > LOADED_N) == 30           # the plates run early
    force, _, _, captured = _resample_forces(columns, rows, 90, "DJ1", offset_s=-0.25)
    assert np.argmax(force[:, right, 1] > LOADED_N) == 75
    assert not captured[:75, right].any() and captured[75:, right].all()   # box phase moves too
    assert not captured[:15].any()          # before the mocap started: unmeasured

    times = np.arange(200) / 100.0
    xyz = np.stack([times, np.zeros(200), np.zeros(200)], 1)[:, None, :]   # x = t
    markers = _resample_markers(times, xyz, 60, offset_s=0.5)
    assert np.allclose(markers[:, 0, 0], np.arange(60) / FPS + 0.5, atol=1e-6)


def test_joint_sync_recovers_a_positive_offset_despite_a_constant_offset() -> None:
    """Video frame i shows mocap time (i + 23) / 60 and the calibration is 0.2 m off:
    the joint scan must return +23 to a fraction of a frame."""
    from sync_opencap import MAX_SHIFT, best_shift, joint_curve
    n, lag = 90, 23
    t = np.arange(n + 2 * MAX_SHIFT) / 60.0
    motion = np.stack([0.4 * np.sin(2 * np.pi * 0.9 * t), 0.05 * np.cos(2 * np.pi * 1.8 * t),
                       1.3 * t], 1)                                            # a walker
    markers = np.stack([motion + k * 0.3 for k in range(3)], 1)                # (N + 2M, 3, 3)
    ours = markers[MAX_SHIFT + lag: MAX_SHIFT + lag + n] + [0.2, -0.1, 0.15]
    shift, error = best_shift(joint_curve(ours, markers))
    assert abs(shift - lag) < 0.05 and error < 1e-6


def test_sync_scan_recovers_a_shift_despite_a_constant_offset() -> None:
    """The video runs 14 frames behind the mocap (video frame i + 14 shows mocap frame i)
    and the calibration is 40 px off: only the time shift must come out, to a fraction of
    a frame, as ``+14`` (``proj[i]`` matches ``kp[i + 14]``; the mocap offset is its
    negative)."""
    from sync_opencap import best_shift, error_curve
    n, lag = 120, 14
    t = np.arange(n + lag) / 60.0
    motion = np.stack([80 * np.sin(2 * np.pi * 1.1 * t), 30 * np.cos(2 * np.pi * 0.7 * t)], 1)
    projected = np.stack([motion[lag:lag + n] + k * 50 for k in range(4)], 1)   # (N, 4, 2)
    keypoints = np.stack([motion[:n] + k * 50 + [40.0, -25.0] for k in range(4)], 1)
    scores = np.ones((n, 4))
    curve = error_curve(projected, keypoints, scores, max_shift=30)
    shift, error = best_shift(curve)
    assert abs(shift - lag) < 0.05 and error < 1e-6
    assert curve[0] > 20
