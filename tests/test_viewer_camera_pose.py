"""The onboard camera's between-frame interpolation (``_camera_pose_at``)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.viewer.helpers import _camera_follow_target, _camera_pose_at


def _extrinsic(yaw_deg: float, centre: np.ndarray) -> np.ndarray:
    """OpenCV cam-from-world ``[R|t]`` for a camera yawed about world y at ``centre``."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R_wc = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])  # cam→world
    E = np.eye(4)
    E[:3, :3] = R_wc.T
    E[:3, 3] = -R_wc.T @ centre
    return E


@pytest.fixture
def trajectory() -> tuple[np.ndarray, np.ndarray]:
    extr = np.stack([_extrinsic(0.0, np.array([0.0, 0.0, 0.0])),
                     _extrinsic(10.0, np.array([1.0, 0.0, 0.0])),
                     _extrinsic(20.0, np.array([2.0, 0.0, 0.0]))])
    return extr, np.array([1.0, 1.2, 1.4])


def test_integer_playhead_matches_frame_pose(trajectory):
    extr, fov = trajectory
    for i in range(len(extr)):
        pos, look_at, up, f = _camera_pose_at(extr, fov, float(i))
        ref = _camera_follow_target(extr[i])
        np.testing.assert_allclose(pos, ref[0], atol=1e-9)
        np.testing.assert_allclose(look_at, ref[1], atol=1e-9)
        np.testing.assert_allclose(up, ref[2], atol=1e-9)
        assert f == fov[i]


def test_half_playhead_is_the_midpoint(trajectory):
    extr, fov = trajectory
    pos, look_at, up, f = _camera_pose_at(extr, fov, 0.5)
    np.testing.assert_allclose(pos, [0.5, 0.0, 0.0], atol=1e-9)
    # optical axis bisects the 0° and 10° yaws and stays unit length
    forward = look_at - pos
    expected = np.array([np.sin(np.radians(5.0)), 0.0, np.cos(np.radians(5.0))])
    np.testing.assert_allclose(forward, expected, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(up), 1.0, atol=1e-12)
    assert f == pytest.approx(1.1)


def test_playhead_is_clamped_to_the_trajectory(trajectory):
    extr, fov = trajectory
    last = _camera_pose_at(extr, fov, 2.0)
    for t in (2.7, 99.0):
        got = _camera_pose_at(extr, fov, t)
        for a, b in zip(got, last):
            np.testing.assert_allclose(a, b, atol=1e-9)
    first = _camera_pose_at(extr, fov, 0.0)
    for a, b in zip(_camera_pose_at(extr, fov, -3.0), first):
        np.testing.assert_allclose(a, b, atol=1e-9)


def test_single_frame_trajectory():
    extr = _extrinsic(30.0, np.array([0.0, 1.0, 2.0]))[None]
    pos, look_at, up, f = _camera_pose_at(extr, np.array([0.9]), 0.75)
    ref = _camera_follow_target(extr[0])
    np.testing.assert_allclose(pos, ref[0], atol=1e-9)
    np.testing.assert_allclose(look_at, ref[1], atol=1e-9)
    assert f == 0.9
