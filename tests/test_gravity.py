"""The gravity step's floor: a plane counts as the floor only when the camera that films it stands above it."""
from __future__ import annotations

import numpy as np

from bvr import config
from bvr.stages.estimate_gravity import floor


def test_a_plane_over_the_filming_camera_is_no_floor() -> None:
    rng = np.random.default_rng(0)
    down = np.array([0.0, 1.0, 0.0])                    # the world's y points down, as the cameras'
    ground = np.c_[rng.uniform(-3, 3, 1500), np.full(1500, 1.5), rng.uniform(2, 8, 1500)]
    wall = np.c_[rng.uniform(-3, 3, 3000), rng.uniform(-2, 1.5, 3000), np.full(3000, 8.0)]
    settings = config.load()["gravity"]

    def cameras(height: float) -> np.ndarray:
        """World-to-camera ``(10, 4, 4)`` of a camera moving sideways ``height`` metres above the plane y = 1.5."""
        extrinsics = np.tile(np.eye(4), (10, 1, 1))
        extrinsics[:, :3, 3] = -np.c_[np.linspace(-1, 1, 10), np.full(10, 1.5 - height), np.zeros(10)]
        return extrinsics

    kept, reason, _ = floor(np.concatenate([ground, wall]), down, cameras(1.5), settings)
    assert reason == "" and np.allclose(kept, down, atol=0.01)
    rejected, reason, plane = floor(np.concatenate([ground, wall]), down, cameras(-1.0), settings)
    assert rejected is None and reason.startswith("filming camera -1.00 m")
    assert plane["ground_points"] >= settings["ground_min_points"] and plane["ground_angle_deg"] < 1.0
