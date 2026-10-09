"""The two measurements ``scripts/label_bedlam2.py`` turns BEDLAM2 pose into.

Both are easy to get subtly wrong in ways nothing downstream notices. The
rasteriser decides who owns a pixel, so an off-by-one in the sample grid or a
depth test that interpolates ``z`` instead of ``1/z`` produces masks that still
*look* like people. The contact readout decides what is on the floor from three
measurements that each fail alone — height labels a foot swinging low through
mid-stride, stillness labels a hand held motionless in the air, facing labels the
back of every hand that passes near the ground — so the tests below pin each of
those, the width of the height band and what keeps it honest, and that the speed
the gate reads is the one BetterRobot's Jacobian gives.
"""

from __future__ import annotations

import numpy as np
import torch

from bvr.bedlam_masks import MAX_OUT_OF_FRAME, out_of_frame_ratio
from bvr.bedlam_contacts import contact_from_floor, frame_positions
from bvr.body import load_body
from bvr.geometry.rasterize import NEAR_M, instance_map

_UP = torch.tensor([0.0, -1.0, 0.0])          # BEDLAM2's world up
_FLOOR = torch.tensor([0.0, 0.0, 0.0])
_FPS = 30.0

#: The two triangles of a quad given corner-first winding.
_QUAD = torch.tensor([[0, 1, 2], [0, 2, 3]])


def _quad(x0: float, y0: float, x1: float, y1: float) -> torch.Tensor:
    return torch.tensor([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=torch.float32)


def test_a_quad_covers_exactly_the_pixels_it_spans():
    """Sample points are pixel centres at integer coordinates — no half-pixel drift."""
    ids, zbuf, dropped = instance_map(_quad(10, 20, 50, 60), torch.full((1, 4), 3.0),
                                      _QUAD, height=80, width=80)

    covered = ids > 0
    assert int(covered.sum()) == (50 - 10 + 1) * (60 - 20 + 1)
    assert int(np.flatnonzero(covered.any(0).numpy())[0]) == 10
    assert int(np.flatnonzero(covered.any(1).numpy())[-1]) == 60
    assert torch.allclose(zbuf[0][covered], torch.tensor(3.0), atol=1e-5)
    assert dropped == 0


def test_the_nearer_instance_owns_every_shared_pixel():
    """Occlusion between people is the depth test and nothing else."""
    far, near = _quad(10, 10, 50, 50), _quad(30, 30, 70, 70)
    verts = torch.cat([far, near])
    depth = torch.stack([torch.full((4,), 3.0), torch.full((4,), 2.0)])

    ids, zbuf, _ = instance_map(verts, depth, _QUAD, height=120, width=120)

    assert int((ids == 2).sum()) == 41 * 41                     # the near one is whole
    assert int((ids == 1).sum()) == 41 * 41 - 21 * 21           # the far one loses the overlap
    # Each instance keeps its OWN buffer, so the silhouette before occlusion survives —
    # that is what separates "hidden behind someone" from "out of shot".
    assert int(torch.isfinite(zbuf[0]).sum()) == 41 * 41


def test_depth_is_interpolated_in_inverse_z():
    """A quad slanted in depth: screen-space barycentrics are exact in 1/z, not z."""
    verts = _quad(10, 10, 70, 70)
    depth = torch.tensor([[2.0, 6.0, 6.0, 2.0]])                # near at x=10, far at x=70

    _ids, zbuf, _ = instance_map(verts, depth, _QUAD, height=80, width=80)

    # Halfway across in screen space, the surface is at the harmonic mean of the ends.
    middle = float(zbuf[0, 40, 40])
    assert abs(middle - 2.0 / (1.0 - 0.5 * (1.0 - 2.0 / 6.0))) < 1e-3
    assert abs(float(zbuf[0, 40, 10]) - 2.0) < 1e-3


def test_geometry_closer_than_the_near_plane_is_dropped():
    """A vertex grazing the lens projects to nonsense; it must not paint anything."""
    ids, _z, _d = instance_map(_quad(10, 10, 50, 50),
                               torch.full((1, 4), NEAR_M / 2.0), _QUAD,
                               height=80, width=80)

    assert int((ids > 0).sum()) == 0


def test_out_of_frame_ratio_measures_the_area_outside_the_image():
    boxes = np.array([[10.0, 10.0, 50.0, 50.0],        # wholly inside
                      [-40.0, 10.0, 40.0, 50.0],       # exactly half off the left edge
                      [-90.0, 10.0, -50.0, 50.0],      # wholly outside
                      [20.0, 20.0, 20.0, 60.0]],       # degenerate, zero width
                     np.float32)

    ratio = out_of_frame_ratio(boxes, width=100, height=100)

    assert ratio.dtype == np.float32
    assert abs(float(ratio[0]) - 0.0) < 1e-6
    assert abs(float(ratio[1]) - 0.5) < 1e-6
    assert abs(float(ratio[2]) - 1.0) < 1e-6
    assert abs(float(ratio[3]) - 1.0) < 1e-6           # nothing there is nothing visible
    # The shipped rule: a person half out of shot is kept, one two-thirds out is not.
    assert bool(ratio[1] <= MAX_OUT_OF_FRAME) is False


def _trace(height_m, drift_per_frame: float = 0.0, frames: int = 90) -> torch.Tensor:
    """One contact frame's world track ``(N, 1, 3)`` at a given height, sliding in +x."""
    h = torch.as_tensor(height_m, dtype=torch.float32)
    h = h.expand(frames) if h.ndim == 0 else h
    x = torch.arange(frames, dtype=torch.float32) * drift_per_frame
    # World up is -y, so a point ``h`` above the floor sits at y = -h.
    return torch.stack([x, -h, torch.zeros(frames)], dim=-1)[:, None, :]


def _velocity(track: torch.Tensor) -> torch.Tensor:
    """The derivative of a synthetic track, m/s — what the Jacobian gives for a real body."""
    v = torch.zeros_like(track)
    v[1:-1] = (track[2:] - track[:-2]) * (_FPS / 2)
    v[0], v[-1] = v[1], v[-2]
    return v


def _normals(frames: int, tilt_deg: float) -> torch.Tensor:
    """``(N, 1, 3)`` outward normals leaning ``tilt_deg`` from straight down."""
    angle = torch.deg2rad(torch.tensor(tilt_deg))
    # Down is +y (world up is -y); leaning about x keeps the angle to down equal to tilt.
    facing = torch.stack([torch.zeros(()), angle.cos(), angle.sin()])
    return facing.expand(frames, 1, 3).clone()


def _label(track: torch.Tensor, tilt_deg: float = 0.0, **knobs) -> np.ndarray:
    out = contact_from_floor(track, _velocity(track), _normals(len(track), tilt_deg),
                             up=_UP, centroid=_FLOOR, fps=_FPS, **knobs)
    return out["contact"][:, 0].numpy()


def test_a_point_resting_on_the_floor_is_in_contact():
    assert _label(_trace(0.005)).all()


def test_stillness_alone_is_not_contact():
    """A hand held motionless a metre up is perfectly still and touching nothing."""
    assert not _label(_trace(1.0)).any()


def test_height_alone_is_not_contact():
    """A foot swinging through 1 cm at 1.5 m/s is low, moving, and not in contact."""
    assert not _label(_trace(0.01, drift_per_frame=1.5 / _FPS)).any()


def test_skin_facing_the_sky_is_not_a_floor_contact():
    """``toe_above`` is the top of the toe. With the foot flat it sits 3.7 cm up and
    its normal points 157 deg from down — inside the height band, so only the facing
    gate keeps it out. Tuck the toes under and the same point must label."""
    assert not _label(_trace(0.005), tilt_deg=157.0).any()
    assert _label(_trace(0.005), tilt_deg=10.0).all()


def test_the_facing_gate_separates_a_kneeling_knee_from_a_vertical_foot():
    """Both are still and inside the wide height band. The patella of a kneeling body
    faces the floor within 45 deg and is a contact; the ball of a foot standing on
    tucked toes leans 60-75 deg and is 7-10 cm off the ground — only its normal says
    so, which is why the threshold sits between them."""
    assert _label(_trace(0.05), tilt_deg=45.0).all()
    assert not _label(_trace(0.07), tilt_deg=65.0).any()


def test_a_toe_several_centimetres_up_is_still_standing_in_its_shoe():
    """The reason the height band is 8 cm. BEDLAM2 lifts the whole body by the shoe
    sole and poses the naked toes inside a flat shoe: with the heel down, the big toe
    of scene ``1a000787`` sits 5.4-7.5 cm above the plane and the shoe is visibly flat
    on the grass. A knee held a hand's width up is not standing on anything."""
    assert _label(_trace(0.070)).all()
    assert not _label(_trace(0.120)).any()


def test_a_point_skimming_the_floor_is_not_resting_on_it():
    """A foot that keeps sliding is not planted: at 0.25 m/s the gate kills the label,
    and it is the gate's own knob that decides where."""
    assert not _label(_trace(0.005, drift_per_frame=0.25 / _FPS)).any()
    assert _label(_trace(0.005, drift_per_frame=0.25 / _FPS), still_mps=0.20).any()


def test_a_single_frame_touch_is_flicker_not_contact():
    track = _trace(1.0)
    track[45, 0, 1] = -0.001                             # one frame on the floor
    assert not _label(track).any()


def test_hysteresis_holds_a_contact_through_a_wobble():
    """Mocap noise lifting a planted foot past the enter threshold must not end it."""
    track = _trace(0.075)
    track[40:45, 0, 1] = -0.090                          # above enter, below exit
    assert _label(track).all()


def test_the_exit_threshold_cannot_latch_onto_a_resting_point():
    """One stray low sample must not hold a label for as long as the point rests:
    a span held open at the exit threshold would outlive its contact."""
    track = _trace(0.11)
    track[30, 0, 1] = -0.07

    assert _label(track).sum() == 0


def _plant(stance: int, total: int = 90, speed: float = 2.0) -> torch.Tensor:
    """A foot travelling at ``speed``, planted dead still for ``stance`` frames."""
    track = torch.zeros(total, 1, 3)
    start, x = (total - stance) // 2, 0.0
    for t in range(total):
        track[t, 0, 0] = x
        if start <= t < start + stance:
            track[t, 0, 1] = -0.005
        else:
            x += speed / _FPS
            track[t, 0, 1] = -0.20
    return track


def test_a_short_foot_plant_survives_the_stillness_gate():
    """The failure that chose the stillness window.

    A speed averaged over a window *centred* on ``t`` needs the whole window inside
    the stance, so it erodes a contact by half a window at each end: a 0.4 s plant at
    2 m/s got zero frames, a 0.8 s plant 14 of 24. Averaging over the calmer
    half-window has no such erosion, and a running stance is only ~0.25 s.
    """
    for stance in (9, 12, 18, 24):
        start = (90 - stance) // 2
        labelled = _label(_plant(stance))[start:start + stance]
        assert labelled.all(), f"{stance}-frame plant: only {labelled.sum()} labelled"


def test_a_foot_swinging_through_the_contact_band_is_not_labelled():
    """The reason a height gate alone will not do: mid-stride a foot passes within a
    centimetre of the floor at well over a metre a second."""
    track = torch.zeros(120, 1, 3)
    track[:, 0, 0] = torch.arange(120, dtype=torch.float32) * (1.5 / _FPS)
    track[:, 0, 1] = torch.where((torch.arange(120) - 60).abs() < 3,
                                 torch.tensor(-0.01), torch.tensor(-0.30))

    assert _label(track).sum() == 0


def test_the_frame_velocity_is_the_derivative_of_the_frame_position():
    """``J(q) q̇`` from BetterRobot must agree with differencing the posed positions.

    A body drifting and turning at the root while a hip swings: every contact frame
    moves, and its Jacobian velocity has to match the central difference of the
    positions the same forward pass produced.
    """
    body = load_body(device="cpu")
    frames, fps = 12, 30.0
    q = torch.zeros(frames, body.nq)
    q[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    q[:, 7:] = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(body.num_joints - 1)
    t = torch.arange(frames, dtype=torch.float32) / fps
    q[:, 0] = 0.8 * t                                             # 0.8 m/s along x
    q[:, 3], q[:, 6] = torch.sin(0.5 * t), torch.cos(0.5 * t)     # turning about x, 1 rad/s
    hip = 7 + 4 * 0                                                # first joint after the root
    q[:, hip], q[:, hip + 3] = torch.sin(1.0 * t), torch.cos(1.0 * t)
    betas = np.zeros(10, np.float32)
    up, centroid = torch.tensor([0.0, -1.0, 0.0]), torch.zeros(3)

    positions, velocities, _normals, _low = frame_positions(
        body, betas, q.numpy(), "cpu", up=up, centroid=centroid, fps=fps)

    differenced = (positions[2:] - positions[:-2]) * (fps / 2)
    # A central difference is second-order: at 2.6 m/s under 1 rad/s it is off by a
    # few mm/s, which is the curvature of the path, not the Jacobian.
    assert torch.allclose(velocities[1:-1], differenced, atol=1e-3, rtol=1e-2)
    assert float(velocities.norm(dim=-1).max()) > 0.8            # the frames really move
