"""The pieces of the dynamics that are easy to get wrong: the friction pyramid, the stencil, the force split."""
from __future__ import annotations

import math

import pytest
import torch

from bvr.terms import dynamics
from bvr.terms.motion import Time


def test_a_cone_force_pushes_back_on_the_skin_and_stays_inside_the_friction_cone() -> None:
    normals = torch.nn.functional.normalize(torch.randn(50, 3, generator=torch.Generator().manual_seed(0)), dim=-1)
    edges = dynamics.cone_edges(normals, 45.0)
    assert torch.allclose(edges.norm(dim=-1), torch.ones(50, 4), atol=1e-6)
    force = (torch.rand(50, 4, 1, generator=torch.Generator().manual_seed(1)) * edges).sum(dim=-2)
    along = -(force * normals).sum(dim=-1)
    across = (force + along[:, None] * normals).norm(dim=-1)
    assert (along > 0).all() and (across <= math.tan(math.radians(45.0)) * along + 1e-6).all()
    # The two sides of a foot's toes face opposite ways, so no force lies in both of their cones.
    assert torch.allclose(dynamics.cone_edges(-normals, 45.0).sum(dim=-2), -edges.sum(dim=-2), atol=1e-5)


@pytest.mark.parametrize("fps", [25.0, 30.0, 60.0, 120.0])
def test_rates_are_the_velocity_and_acceleration_per_second_at_every_frame_rate(fps: float) -> None:
    time, seconds = Time(fps), torch.arange(40) / fps
    position = (0.5 * 3.0 * seconds.square() + 2.0 * seconds)[None, :, None]  # accelerates at 3, starts at 2
    velocity = time.difference(position)
    v, a = dynamics.rates(time, velocity, time.difference(velocity))
    inside = slice(time.stride, -time.stride)
    assert torch.allclose(a[0, inside, 0], torch.tensor(3.0), atol=2e-2)
    assert torch.allclose(v[0, inside, 0], (3.0 * seconds + 2.0)[inside], atol=1e-3)
    assert not v[0, :time.stride].any() and not a[0, -time.stride:].any()


def test_inverse_dynamics_count_only_where_the_whole_stencil_is_tracked() -> None:
    tracked = torch.ones(1, 12, dtype=torch.bool)
    tracked[0, 6] = False
    # A frame the clip does not have counts as tracked (the body is taken at rest at the clip's ends); a gap does not.
    assert dynamics.stencil_rows(Time(30.0), tracked)[0].tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    assert dynamics.stencil_rows(Time(60.0), tracked)[0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]


def test_the_force_split_sends_a_change_of_size_and_a_turn_to_different_parts() -> None:
    time, frames = Time(30.0), torch.arange(8.0)
    growing = torch.stack((torch.zeros(8), torch.zeros(8), 100.0 + 10.0 * frames), dim=-1)[None, :, None]
    rate = dynamics.force_split(growing, time=time, order=1)
    assert torch.allclose(rate[..., 0].norm(dim=-1), torch.tensor(10.0)) and rate[..., 1].abs().max() < 1e-4
    assert dynamics.force_split(growing, time=time, order=2).abs().max() < 1e-4  # a ramp has no acceleration
    turning = 100.0 * torch.stack((torch.cos(0.1 * frames), torch.sin(0.1 * frames), torch.zeros(8)), dim=-1)
    rate = dynamics.force_split(turning[None, :, None], time=time, order=1)
    assert rate[..., 0].abs().max() < 1e-3 and torch.allclose(
        rate[..., 1].norm(dim=-1), torch.tensor(200.0 * math.sin(0.05)), atol=1e-3)
