"""The gravity step: GeoCalib's up turned into the world, and the floor, which counts only when the camera that films
it stands above it."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from bvr import config, files, paths, run
from bvr.paths import Clip
from bvr.stages import estimate_gravity
from bvr.stages.estimate_gravity import floor


def test_geocalib_up_in_turned_cameras_is_the_world_down(tmp_path: Path, monkeypatch) -> None:
    # Four frames, each from a camera turned another way; GeoCalib reports the world's up in each camera's frame.
    down = np.array([0.3, 0.9, -0.3]) / np.linalg.norm([0.3, 0.9, -0.3])
    extrinsics = np.tile(np.eye(4), (4, 1, 1))
    for k, extrinsic in enumerate(extrinsics):
        extrinsic[:3, :3] = cv2.Rodrigues(np.array([0.4, -0.7, 0.2]) * (k + 1))[0]
        extrinsic[:3, 3] = [0.1 * k, -0.3, 2.0]
    ups = iter(extrinsics[:, :3, :3] @ -down)

    class GeoCalib:
        """Stands in for the model: the next frame's up in its camera, and the fields the stage logs."""

        def __init__(self, weights: str) -> None:
            pass

        def to(self, device: torch.device) -> GeoCalib:
            return self

        def eval(self) -> GeoCalib:
            return self

        def calibrate(self, image: torch.Tensor, camera_model: str) -> dict:
            gravity = SimpleNamespace(vec3d=torch.tensor(next(ups))[None], roll=torch.zeros(1), pitch=torch.zeros(1))
            return {"gravity": gravity, "up_confidence": torch.ones(1, 4, 4),
                    "camera": SimpleNamespace(f=torch.ones(1, 2))}

    monkeypatch.setitem(sys.modules, "geocalib", SimpleNamespace(GeoCalib=GeoCalib))
    clip = Clip("one", tmp_path / "one.avi", tmp_path / "out" / "one")
    writer = cv2.VideoWriter(str(clip.video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (16, 16))
    for _ in extrinsics:
        writer.write(np.full((16, 16, 3), 127, np.uint8))
    writer.release()
    files.save_npz(clip.path(paths.TRANSFORM), "bvr-transform", extrinsics=extrinsics, frame_indices=np.arange(4))
    files.save_npz(clip.path(paths.SCENE), "bvr-scene", points=np.random.default_rng(0).uniform(-1, 1, (200, 3)))
    configured = config.load()
    process = estimate_gravity.load(run.Stage("gravity", {"gravity": configured["gravity"], "run": configured["run"]},
                                              torch.device("cpu")))

    process(clip)

    stored = files.load_npz(clip.path(paths.GRAVITY), "bvr-gravity")
    np.testing.assert_allclose(stored["gravity_world_frames"], np.tile(down, (4, 1)), atol=1e-6)
    np.testing.assert_allclose(stored["gravity_world_geocalib"], down, atol=1e-6)


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
