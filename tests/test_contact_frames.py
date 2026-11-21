"""Contact frames: a frame's outward normal is the z axis of its pose (``bvr.body.frame_normals``)."""

from __future__ import annotations

import math

import pytest
import torch
from better_robot.lie import so3

from bvr.body import frame_normals


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
