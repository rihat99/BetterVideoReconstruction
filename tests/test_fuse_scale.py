"""The geometry steps: the scene merge and visibility of ``fuse``, its split of a frame, and two pre-fit terms."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from bvr import config, files, paths
from bvr.geometry.camera_motion import image_motion
from bvr.geometry.normals import estimate_point_normals
from bvr.geometry.voxel import voxel_downsample
from bvr.paths import Clip
from bvr.stages.estimate_scale import _chamfer, _start_velocity
from bvr.stages.fuse_geometry import _merge, _visible, fuse
from bvr.terms.motion import Time


def test_merge_collapses_boundary_twins():
    # The twins straddle x=2, while the pair near 0 pulls the first pass's lowest centroid to 0.499. A second pass
    # that took its origin from the centroids would put a face at 1.999 and miss the twins: both passes share one.
    xs = [0.0, 0.998, 1.98, 1.99, 2.01, 2.02, 7.0]
    points = torch.tensor([[x, 0.0, 0.0] for x in xs])
    confidence = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    merged, merged_confidence, _, count, cells = _merge(points, confidence, torch.full((7, 3), 100.0), voxel=1.0)

    assert merged.shape[0] == 3                                   # the near pair, ONE merged twin, the far point
    assert float(merged[count == 4][0, 0]) == pytest.approx(2.0, abs=1e-6)
    assert float(merged_confidence[count == 4]) == pytest.approx(0.6)  # the largest of the twins
    assert cells.shape == merged.shape


def test_voxel_downsample_shift_moves_the_grid():
    points = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    assert voxel_downsample(points, 1.0)[2].numel() == 1             # both in the cell [0, 1)
    assert voxel_downsample(points, 1.0, shift=0.5)[2].numel() == 2  # a face at 0.5 between them


def test_visible_prunes_the_back_of_the_slab():
    # A camera at the origin looking down +z, f = 100 px, 100 x 100 px. A 3 x 3 sheet at z = 2, a copy of it 0.5
    # behind (more than one voxel), and one point far behind, alone in its bin.
    xy = [(x, y) for x in (-0.1, 0.0, 0.1) for y in (-0.1, 0.0, 0.1)]
    points = torch.tensor([[x, y, 2.0] for x, y in xy] + [[x * 1.25, y * 1.25, 2.5] for x, y in xy] + [[0.8, 0.8, 4.0]])
    intrinsics = np.array([[[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]]], np.float32)

    keep = _visible(points, np.eye(4, dtype=np.float32)[None], intrinsics, (100, 100), bin_px=6.0, voxel=0.2)

    assert keep[:9].all() and not keep[9:18].any() and keep[18]


def test_fuse_splits_a_frame_into_scene_and_human_points(tmp_path: Path):
    # One 16 x 16 frame at depth 2, f = 8 px, centre (8, 8); the person covers pixel (8, 8) alone.
    clip = Clip("one", tmp_path / "one.avi", tmp_path / "out" / "one")
    writer = cv2.VideoWriter(str(clip.video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (16, 16))
    writer.write(np.full((16, 16, 3), 127, np.uint8))
    writer.release()
    intrinsics = np.array([[[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]]], np.float32)
    files.save_npz(clip.path(paths.VGGT_CAMERA), "bvr-camera", frame_indices=np.array([0]),
                   extrinsics=np.eye(4, dtype=np.float32)[None], intrinsics_px=intrinsics, intrinsics=intrinsics / 16,
                   fov_x_deg=np.array([90.0]), fov_y_deg=np.array([90.0]), vggt_height=np.int32(16),
                   vggt_width=np.int32(16), image_height=np.int32(16), image_width=np.int32(16))
    clip.path(paths.VGGT_DEPTH).mkdir()
    np.savez(clip.path(paths.VGGT_DEPTH) / "frame_000000.npz", depth=np.full((16, 16), 2.0, np.float16),
             depth_conf=np.ones((16, 16), np.float16))
    mask = np.zeros((16, 16), bool)
    mask[8, 8] = True
    files.write_mask(clip.path(paths.SAM3) / "00" / "frame_000000.png", mask)
    files.save_npz(clip.path(paths.SAM3_BBOXES), "bvr-bboxes", object_ids=np.array([0]))
    files.save_npz(clip.path(paths.SMPLX_PARAMS), "bvr-smplx", object_ids=np.array([0]))
    options = config.load()["fuse"] | {"voxel_px": 1e-3, "conf_drop_q": 0.0, "edge_rtol": 0.0, "human_dilate": 0,
                                       "min_scene_voxels": 1}

    fuse(clip, options, torch.device("cpu"))

    scene, human = (files.load_npz(clip.path(paths.FUSED) / name, format) for name, format in (
        ("scene.npz", "bvr-scene"), ("human.npz", "bvr-human-points")))
    transform = files.load_npz(clip.path(paths.FUSED) / "transform.npz", "bvr-transform")
    assert len(scene["points"]) == 255 and float(scene["voxel_size"]) == pytest.approx(1e-3 * 2 / 8)
    np.testing.assert_allclose(human["points"], [[0.0, 0.0, 2.0]])  # pixel (8, 8) is on the optical axis
    assert human["frame_id"].tolist() == [0] and human["obj_id"].tolist() == [0]
    np.testing.assert_allclose(np.linalg.norm(scene["normals"], axis=1), 1.0, atol=1e-5)
    assert (scene["normals"][:, 2] < 0).all()                        # toward the camera
    assert not transform["metric"] and transform["static_camera"] and float(transform["fps"]) == pytest.approx(30.0)


def test_estimate_point_normals_edge_cases():
    cameras = torch.zeros((1, 3))
    assert estimate_point_normals(torch.zeros((0, 3)), cameras).shape == (0, 3)
    one = estimate_point_normals(torch.tensor([[0.0, 0.0, 2.0]]), cameras)  # k collapses to 1
    assert one.shape == (1, 3) and float(one.norm()) == pytest.approx(1.0, abs=1e-5)


def test_chamfer_matches_only_visible_vertices():
    # One depth point; the visible vertex sits 0.1 from it, the hidden one exactly on it.
    vertices = torch.tensor([[[[0.0, 0.0, 1.9], [0.0, 0.0, 2.0]]]])  # (P, T, 2, 3)
    term = _chamfer(torch.tensor([[[[0.0, 0.0, 2.0]]]]), torch.ones(1, 1, 1, dtype=torch.bool), torch.ones(1, 1),
                    torch.arange(2), gm_scale_m=0.3)
    body = SimpleNamespace(points=lambda ids: vertices[..., ids, :])

    hidden = term(body, torch.zeros(1, 1), torch.tensor([[[True, False]]]))
    seen = term(body, torch.zeros(1, 1), torch.tensor([[[True, True]]]))

    assert float(hidden) == pytest.approx(0.1**2 / (0.3**2 + 0.1**2), rel=1e-4)
    assert float(seen) == pytest.approx(0.0, abs=1e-6)
    # At s = 2 the body is halved and so is the kernel: the term is measured in the cloud's units.
    halved = term(SimpleNamespace(points=lambda ids: 2 * vertices[..., ids, :]), torch.full((1, 1), np.log(2.0)),
                  torch.tensor([[[True, False]]]))
    assert float(halved) == pytest.approx(0.1**2 / (0.15**2 + 0.1**2), rel=1e-4)


def test_start_velocity_prices_a_change_of_motion_not_motion():
    start = torch.linspace(0, 1, 4)[None, :, None, None] * torch.tensor([1.0, 0.0, 0.0])  # (1, 4, 1, 3), steady
    term = _start_velocity(start, torch.ones(1, 4, dtype=torch.bool), Time(30.0))
    assert float(term(SimpleNamespace(joint_translations=start))) == 0.0
    assert float(term(SimpleNamespace(joint_translations=torch.zeros_like(start)))) > 0.0


# Is the shot locked off? -------------------------------------------------------------------------------------------

_K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]], np.float32)
_W, _H = 1280, 720


def _wall(scale: float = 1.0) -> np.ndarray:
    """A patch of scene 5 units in front of a camera at the origin looking down +z."""
    xy = np.stack(np.meshgrid(np.linspace(-2, 2, 12), np.linspace(-1, 1, 12)), -1).reshape(-1, 2)
    return (np.concatenate([xy, np.full((len(xy), 1), 5.0)], 1) * scale).astype(np.float32)


def _track(rotations: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Cam-from-world matrices for a camera with these orientations and centres."""
    extrinsics = np.zeros((len(rotations), 4, 4), np.float32)
    extrinsics[:, :3, :3] = rotations
    extrinsics[:, :3, 3] = -np.einsum("fij,fj->fi", rotations, centres)
    extrinsics[:, 3, 3] = 1.0
    return extrinsics


def _yaw(angles_deg: np.ndarray) -> np.ndarray:
    a = np.radians(angles_deg)
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([c, z, s, z, o, z, -s, z, c], -1).reshape(-1, 3, 3).astype(np.float32)


def test_a_locked_camera_does_not_move_the_picture():
    motion = image_motion(_track(_yaw(np.zeros(40)), np.zeros((40, 3), np.float32)), np.repeat(_K[None], 40, 0),
                          _W, _H, _wall())
    assert motion == pytest.approx(0.0, abs=1e-9)


def test_a_pan_counts_as_moving_even_though_the_camera_never_travels():
    # The centre is nailed to the origin; a test on the camera position alone would call this locked.
    extrinsics = _track(_yaw(np.linspace(0.0, 4.0, 40)), np.zeros((40, 3), np.float32))
    motion = image_motion(extrinsics, np.repeat(_K[None], 40, 0), _W, _H, _wall())
    # The 90th-percentile frame sits ~1.6 deg off the median one: ~30 px at f = 1000 px, ~2.5 % of the width.
    assert motion == pytest.approx(0.026, rel=0.2) and motion > config.load()["fuse"]["static_motion_max"]


def test_the_answer_survives_metricization():
    # The same clip, up to scale and metric (cloud and camera translations times 7), must get the same answer.
    centres = np.stack([np.linspace(0.0, 0.3, 40), np.zeros(40), np.zeros(40)], -1).astype(np.float32)
    K = np.repeat(_K[None], 40, 0)
    unscaled = image_motion(_track(_yaw(np.zeros(40)), centres), K, _W, _H, _wall())
    metric = image_motion(_track(_yaw(np.zeros(40)), centres * 7.0), K, _W, _H, _wall(7.0))
    assert unscaled > 0 and metric == pytest.approx(unscaled, rel=1e-6)
