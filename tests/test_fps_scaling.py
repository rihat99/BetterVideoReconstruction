"""fps normalization: weight scaling, span rule in seconds, v/a stencil spacing.

The contract everything else leans on: all three are the IDENTITY at fps <= 30,
so every clip at or below the tuning reference keeps its exact shipped behaviour.
"""

from __future__ import annotations

import pytest
import torch

from bvr.config import fps_scaled_weights
from bvr.human_optim.contacts import min_span_frames
from bvr.human_optim.motion import velocity_acceleration_from_trajectory


def test_fps_scaled_weights_identity_at_or_below_reference():
    w = {"accel_root": 400.0, "snap_joint": 500.0, "base_wrench": 100.0,
         "ann_velocity": 4000.0, "radial_velocity": 400.0}
    for fps in (23.98, 24.0, 25.0, 29.97, 30.0):
        assert fps_scaled_weights(w, fps) == w


def test_fps_scaled_weights_scales_by_difference_order():
    w = {"radial_velocity": 1.0, "accel_root": 1.0, "jerk_joint": 1.0,
         "snap_root": 1.0, "force_smooth_mag": 1.0, "base_wrench": 100.0}
    out = fps_scaled_weights(w, 60.0)          # s = 2
    assert out["radial_velocity"] == pytest.approx(4.0)     # (2)^2
    assert out["force_smooth_mag"] == pytest.approx(4.0)
    assert out["accel_root"] == pytest.approx(16.0)         # (2)^4
    assert out["jerk_joint"] == pytest.approx(64.0)         # (2)^6
    assert out["snap_root"] == pytest.approx(256.0)         # (2)^8
    assert out["base_wrench"] == 100.0                      # not a difference term


def test_min_span_frames_matches_legacy_three_frames_below_reference():
    for fps in (23.98, 24.0, 25.0, 29.97, 30.0):
        assert min_span_frames(0.1, fps) == 3               # the old min_frames = 3
    assert min_span_frames(0.1, 50.0) == 5
    assert min_span_frames(0.1, 59.94) == 6
    assert min_span_frames(0.1, 60.0) == 6


class _EuclidModel:
    """Minimal stand-in: Euclidean difference, nv = nq."""

    nv = 4

    @staticmethod
    def difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return b - a


def test_va_stencil_spacing_one_matches_legacy():
    q = torch.randn(9, 4)
    v1, a1 = velocity_acceleration_from_trajectory(_EuclidModel, q, 1 / 30)
    vs, as_ = velocity_acceleration_from_trajectory(_EuclidModel, q, 1 / 30, spacing=1)
    assert torch.equal(v1, vs) and torch.equal(a1, as_)


def test_va_stencil_spacing_two_equals_subsampled_rate():
    # On a smooth quadratic trajectory the h=2 stencil at dt is exact, and its noise
    # response equals the 30 fps stencil's: a per-frame alternating +/- eps wiggle that
    # the h=1 stencil amplifies to 4*eps/dt^2 vanishes entirely under h=2.
    t = torch.arange(11, dtype=torch.float32)[:, None]
    q = 0.5 * t * t * torch.ones(1, 4)
    dt = 1 / 60
    _, a2 = velocity_acceleration_from_trajectory(_EuclidModel, q, dt, spacing=2)
    assert torch.allclose(a2[2:-2], torch.full_like(a2[2:-2], 1.0 / dt**2), rtol=1e-3)
    assert torch.equal(a2[:2], torch.zeros_like(a2[:2]))    # boundary pinned to spacing
    wiggle = q + 0.01 * (-1.0) ** t
    _, aw1 = velocity_acceleration_from_trajectory(_EuclidModel, wiggle, dt, spacing=1)
    _, aw2 = velocity_acceleration_from_trajectory(_EuclidModel, wiggle, dt, spacing=2)
    noise1 = (aw1[2:-2] - 1.0 / dt**2).abs().max()
    noise2 = (aw2[2:-2] - 1.0 / dt**2).abs().max()
    assert noise2 < 1e-2 * noise1
