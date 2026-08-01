"""Contact-frame detection: the surface / normal-window / stillness gates."""

from __future__ import annotations

import math

import pytest
import torch
from better_robot.lie import so3

from tools.body import frame_normals
from tools.human_optim.contacts import (
    detect_frame_contacts, frame_guarded_sdf, frame_scene_signals, joint_drift)

_UP = torch.tensor([0.0, 0.0, 1.0])
_DOWN = torch.tensor([0.0, 0.0, -1.0])


def _plane(half_extent: float = 0.5, step: float = 0.05, conf: float = 1.0):
    """A horizontal plane of scene points at ``z = 0`` whose normals all point UP."""
    axis = torch.arange(-half_extent, half_extent + 1e-6, step, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1),
                          torch.zeros(grid_x.numel())), dim=-1)
    normals = _UP.expand_as(points).clone()
    return points, normals, torch.full((points.shape[0],), conf)


def _hover(num_steps: int, height: float, normal: torch.Tensor):
    """One contact frame hovering ``height`` above the plane origin, facing ``normal``."""
    points = torch.tensor([0.0, 0.0, height]).expand(num_steps, 1, 3).clone()
    return points, normal.expand(num_steps, 1, 3).clone()


def _detect(points, normals, drift, scene, **overrides):
    knobs = dict(close_thresh_m=0.03, exit_thresh_m=0.035, min_frames=3, still_m=0.03,
                 max_dist_m=0.10, conf_thresh=0.1, normal_max_angle_deg=90.0,
                 prune_margin_m=0.15)
    knobs.update(overrides)
    return detect_frame_contacts(points, normals, drift, *scene, **knobs)


# ── the signals ──────────────────────────────────────────────────────────────

def test_scene_signals_report_height_and_facing():
    scene = _plane()
    points, normals = _hover(4, 0.01, _DOWN)
    sdf, dmin, conf, normal_dot = frame_scene_signals(points, normals, *scene,
                                                      prune_margin_m=0.15)
    # 1 cm above a plane whose normal is +z: the perpendicular signed distance is +1 cm.
    assert torch.allclose(sdf, torch.full_like(sdf, 0.01), atol=1e-6)
    assert torch.allclose(dmin, torch.full_like(dmin, 0.01), atol=1e-6)
    assert torch.allclose(conf, torch.ones_like(conf))
    # skin faces straight down into the plane ⇒ dot(n_frame, −n_scene) = +1
    assert torch.allclose(normal_dot, torch.ones_like(normal_dot), atol=1e-6)
    # flipping the frame normal flips the facing signal, nothing else
    _, _, _, flipped = frame_scene_signals(points, -normals, *scene, prune_margin_m=0.15)
    assert torch.allclose(flipped, -torch.ones_like(flipped), atol=1e-6)


def test_scene_signals_empty_box_is_no_evidence():
    scene = _plane()
    points, normals = _hover(3, 5.0, _DOWN)          # far above every scene point
    sdf, dmin, conf, normal_dot = frame_scene_signals(points, normals, *scene,
                                                      prune_margin_m=0.15)
    assert torch.isinf(sdf).all() or (sdf > 1e30).all()
    assert (dmin > 1e30).all()
    assert (conf == 0.0).all()
    assert (normal_dot == -1.0).all()


def test_guards_force_infinity_not_masking():
    scene = _plane()
    points, normals = _hover(2, 0.01, _DOWN)
    sdf, dmin, conf, normal_dot = frame_scene_signals(points, normals, *scene,
                                                      prune_margin_m=0.15)
    guarded = frame_guarded_sdf(sdf, dmin, conf, normal_dot, max_dist_m=0.10,
                                conf_thresh=0.1, normal_max_angle_deg=90.0)
    assert torch.allclose(guarded, sdf)
    # each guard on its own must push the signal to +inf-ish
    for kwargs in (dict(max_dist_m=0.005), dict(conf_thresh=2.0), dict(normal_max_angle_deg=1.0)):
        knobs = dict(max_dist_m=0.10, conf_thresh=0.1, normal_max_angle_deg=90.0)
        knobs.update(kwargs)
        blocked = frame_guarded_sdf(sdf, dmin, conf, -normal_dot if "normal_max_angle_deg"
                                    in kwargs else normal_dot, **knobs)
        assert (blocked > 1e30).all(), kwargs


# ── the normal window: the toe-hook case ─────────────────────────────────────

def test_normal_window_accepts_the_frame_facing_the_scene():
    scene = _plane()
    points, normals = _hover(10, 0.01, _DOWN)
    drift = torch.zeros(10, 1)
    assert _detect(points, normals, drift, scene).contact.all()


def test_normal_window_rejects_the_frame_facing_away():
    """Same point, opposite skin direction — toe_above cannot fire on a hold below it."""
    scene = _plane()
    points, normals = _hover(10, 0.01, _UP)
    drift = torch.zeros(10, 1)
    assert not _detect(points, normals, drift, scene).contact.any()


def test_toe_hook_pair_never_both_fire():
    """toe_under and toe_above sit at the same place with opposite normals."""
    scene = _plane()
    points = torch.tensor([0.0, 0.0, 0.01]).expand(10, 2, 3).clone()
    normals = torch.stack((_DOWN.expand(10, 3), _UP.expand(10, 3)), dim=1)
    contact = _detect(points, normals, torch.zeros(10, 2), scene).contact
    assert contact[:, 0].all()
    assert not contact[:, 1].any()


def test_wide_open_window_lets_the_away_facing_frame_through():
    """The rejection is the WINDOW, not some other guard: open it and the label returns."""
    scene = _plane()
    points, normals = _hover(10, 0.01, _UP)
    drift = torch.zeros(10, 1)
    assert _detect(points, normals, drift, scene, normal_max_angle_deg=180.0).contact.all()


# ── stillness ────────────────────────────────────────────────────────────────

def test_stillness_gate_kills_a_moving_frame():
    scene = _plane()
    points, normals = _hover(10, 0.01, _DOWN)
    still = torch.zeros(10, 1)
    moving = torch.full((10, 1), 0.07)              # well past 2 · still_m = 6 cm
    assert _detect(points, normals, still, scene).contact.all()
    assert not _detect(points, normals, moving, scene).contact.any()


def test_stillness_gate_breaks_a_span_in_the_middle():
    scene = _plane()
    points, normals = _hover(12, 0.01, _DOWN)
    drift = torch.zeros(12, 1)
    drift[5:7] = 0.20                                # a two-frame lunge through the hold
    contact = _detect(points, normals, drift, scene, min_frames=1).contact[:, 0]
    assert contact[:5].all() and contact[7:].all()
    assert not contact[5:7].any()


def test_nan_drift_cannot_hold_a_contact():
    """An untracked step has no motion witness, so it must not be labelled."""
    scene = _plane()
    points, normals = _hover(10, 0.01, _DOWN)
    drift = torch.zeros(10, 1)
    drift[4:8] = float("nan")
    contact = _detect(points, normals, drift, scene, min_frames=1).contact[:, 0]
    assert not contact[4:8].any()


# ── hysteresis + minimum duration ────────────────────────────────────────────

def test_hysteresis_holds_a_span_between_enter_and_exit():
    scene = _plane()
    heights = torch.tensor([0.010] * 4 + [0.032] * 4 + [0.010] * 4)   # dips into enter..exit
    points = torch.zeros(12, 1, 3)
    points[:, 0, 2] = heights
    normals = _DOWN.expand(12, 1, 3).clone()
    contact = _detect(points, normals, torch.zeros(12, 1), scene).contact[:, 0]
    assert contact.all(), "3.2 cm is above enter (3 cm) but below exit (3.5 cm)"


def test_hysteresis_ends_a_span_past_the_exit_threshold():
    scene = _plane()
    points = torch.zeros(12, 1, 3)
    points[:, 0, 2] = torch.tensor([0.010] * 4 + [0.050] * 4 + [0.010] * 4)
    normals = _DOWN.expand(12, 1, 3).clone()
    contact = _detect(points, normals, torch.zeros(12, 1), scene).contact[:, 0]
    assert contact[:4].all() and contact[8:].all()
    assert not contact[4:8].any()


def test_min_frames_drops_a_flicker_span():
    scene = _plane()
    points = torch.full((12, 1, 3), 0.0)
    points[:, 0, 2] = 0.50                            # far above the plane all clip long
    points[5:7, 0, 2] = 0.010                         # …except a two-frame blip
    normals = _DOWN.expand(12, 1, 3).clone()
    drift = torch.zeros(12, 1)
    assert not _detect(points, normals, drift, scene, min_frames=3).contact.any()
    assert _detect(points, normals, drift, scene, min_frames=2).contact[5:7, 0].all()


def test_drift_is_measured_the_same_way_on_frames_as_on_joints():
    """The detector is K-generic: 12 contact frames use the same drift as 52 joints."""
    positions = torch.zeros(9, 2, 3)
    positions[:, 0, 0] = torch.linspace(0.0, 0.8, 9)     # a sweeping point
    drift = joint_drift(positions, fps=30.0, window_s=0.2)
    assert drift.shape == (9, 2)
    assert (drift[:, 0] > 0.0).all()
    assert torch.allclose(drift[:, 1], torch.zeros(9))


# ── the frame_normals helper ─────────────────────────────────────────────────

def test_frame_normals_is_the_third_rotation_column():
    torch.manual_seed(0)
    quaternions = torch.randn(4, 12, 4)
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    frames7 = torch.cat((torch.randn(4, 12, 3), quaternions), dim=-1)
    normals = frame_normals(frames7)
    assert normals.shape == (4, 12, 3)
    assert torch.allclose(normals, so3.to_matrix(quaternions)[..., :, 2], atol=1e-6)
    assert torch.allclose(normals.norm(dim=-1), torch.ones(4, 12), atol=1e-5)


@pytest.mark.parametrize("angle_deg, expected", [(0.0, 1.0), (180.0, -1.0), (90.0, 0.0)])
def test_frame_normals_follows_a_rotation_about_x(angle_deg: float, expected: float):
    """Rotating a frame about x tips its baked z normal by the same angle."""
    angle = math.radians(angle_deg)
    quaternion = torch.tensor([[math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)]])
    frames7 = torch.cat((torch.zeros(1, 3), quaternion), dim=-1)[None]
    assert frame_normals(frames7)[0, 0, 2] == pytest.approx(expected, abs=1e-6)
