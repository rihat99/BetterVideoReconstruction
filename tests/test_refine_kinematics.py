"""``bvr.stages.refine_kinematics``: the finger smoothing, which runs before any fit and that no fit corrects."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from bvr.stages.refine_kinematics import BODY, _smooth_fingers


def _x_quats(angles: torch.Tensor) -> torch.Tensor:
    """Unit quaternions (scalar last) of rotations by ``angles (T,)`` about x, ``(T, 1, 4)``."""
    zeros = torch.zeros_like(angles)
    return torch.stack((torch.sin(angles / 2), zeros, zeros, torch.cos(angles / 2)), dim=-1)[:, None]


def _q_with_fingers(track: torch.Tensor) -> torch.Tensor:
    """One person's pose ``(1, T, 211)`` at identity, with every finger quaternion set from ``track (T, 30, 4)``."""
    q = torch.zeros(1, len(track), 211)
    q[..., 6:BODY:4] = 1.0  # the scalar of the root and of each body joint quaternion
    q[0, :, BODY:] = track.flatten(1)
    return q


def _fingers(q: torch.Tensor) -> torch.Tensor:
    return q[0, :, BODY:].unflatten(-1, (30, 4))


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The angle between unit quaternions, the same for a quaternion and its negative."""
    return torch.rad2deg(2.0 * torch.arccos((a * b).sum(dim=-1).abs().clamp(max=1.0)))


def test_a_still_hand_stays_still_when_half_its_quaternions_are_negated() -> None:
    still = _x_quats(torch.full((12,), 0.7))
    track = still.expand(12, 30, 4).clone()
    track[6:] *= -1.0  # the same rotations; averaging raw components across the flip would cancel them
    q = _q_with_fingers(track)

    smooth = _smooth_fingers(q, np.ones((1, 12), bool), 2.0)

    assert torch.allclose(_fingers(smooth).norm(dim=-1), torch.ones(12, 30), atol=1e-5)
    assert float(_angle_deg(_fingers(smooth), still).max()) == pytest.approx(0.0, abs=1e-3)
    assert torch.equal(smooth[..., :BODY], q[..., :BODY])
    assert _smooth_fingers(q, np.ones((1, 12), bool), 0.0) is q


def test_jitter_goes_and_the_motion_stays() -> None:
    torch.manual_seed(0)
    curl = torch.linspace(0.0, 1.0, 40)  # a slow finger curl
    noisy = _x_quats(curl + 0.05 * torch.randn(40)).expand(40, 30, 4).contiguous()
    clean = _x_quats(curl)

    smooth = _fingers(_smooth_fingers(_q_with_fingers(noisy), np.ones((1, 40), bool), 3.0))

    step = lambda track: float(_angle_deg(track[1:], track[:-1]).mean())  # noqa: E731
    assert step(smooth) - step(clean) < 0.25 * (step(noisy) - step(clean))
    assert float(_angle_deg(smooth, clean).mean()) < float(_angle_deg(noisy, clean).mean())


def test_a_tracking_gap_separates_what_is_smoothed() -> None:
    # Two tracked runs at different hand poses; each is constant, so a filter that stays inside a run leaves it.
    track = _x_quats(torch.cat((torch.full((12,), 0.2), torch.full((8,), 1.4)))).expand(20, 30, 4).contiguous()
    tracked = np.ones((1, 20), bool)
    tracked[0, 8:12] = False
    q = _q_with_fingers(track)

    smooth = _smooth_fingers(q, tracked, 4.0)

    assert float(_angle_deg(_fingers(smooth)[:8], track[:8]).max()) == pytest.approx(0.0, abs=1e-3)
    assert float(_angle_deg(_fingers(smooth)[12:], track[12:]).max()) == pytest.approx(0.0, abs=1e-3)
    assert torch.equal(smooth[0, 8:12], q[0, 8:12])
