"""Tests for the fuse → scale chain: voxel merge, geometry build, metricization."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from scripts.stages.estimate_scale import _metricize_geometry
from scripts.stages.fuse_geometry import _build_geometry, _merge_scene
from tools.geometry.voxel import voxel_downsample


def test_merge_scene_collapses_boundary_twins():
    # Two tight pairs straddle the cell face at x=2 (grid origin = amin = 0.0):
    # pass 1 leaves twin centroids 0.03 apart; pass 2 (half-shifted grid) merges them.
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.98, 0.0, 0.0], [1.99, 0.0, 0.0],
                        [2.01, 0.0, 0.0], [2.02, 0.0, 0.0], [7.0, 0.0, 0.0]])
    conf = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    color = torch.full((6, 3), 100.0)

    points, confidence, col, count, coords = _merge_scene(pts, conf, color, voxel=1.0)

    assert points.shape[0] == 3  # far point + lone point + ONE merged twin
    merged = points[count == 4]
    assert merged.shape == (1, 3)
    assert float(merged[0, 0]) == pytest.approx(2.0, abs=1e-6)
    assert float(confidence[count == 4]) == pytest.approx(0.5)  # max over the twin
    assert coords.shape == points.shape


def test_voxel_downsample_shift_moves_the_grid():
    pts = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    _, _, counts_plain, _ = voxel_downsample(pts, 1.0)
    _, _, counts_shift, _ = voxel_downsample(pts, 1.0, shift=0.5)
    assert counts_plain.numel() == 1   # amin-origin puts both in cell [0, 1)
    assert counts_shift.numel() == 2   # half-shift puts a face at 0.5, between them


def _tiny_clip(tmp_path: Path):
    """3x3 frame, constant depth 2.0, one human pixel at (1,1)."""
    source = tmp_path / "frames"
    source.mkdir()
    cv2.imwrite(str(source / "0.png"), np.full((3, 3, 3), 127, np.uint8))
    pred = tmp_path / "pred"
    (pred / "vggt" / "depth").mkdir(parents=True)
    (pred / "sam3" / "00").mkdir(parents=True)
    np.savez(pred / "vggt" / "depth" / "frame_000000.npz",
             depth=np.full((3, 3), 2.0, np.float32),
             depth_conf=np.ones((3, 3), np.float32))
    mask = np.zeros((3, 3), np.uint8)
    mask[1, 1] = 255
    cv2.imwrite(str(pred / "sam3" / "00" / "frame_000000.png"), mask)
    return source, pred


def test_build_geometry_splits_scene_and_human_unscaled(tmp_path):
    source, pred = _tiny_clip(tmp_path)
    args = SimpleNamespace(source=source, device="cpu", conf_drop_q=0.0, edge_rtol=0.0,
                           human_dilate=0, voxel_px=1e-4, min_scene_voxels=1)
    cam = {"vggt_height": 3, "vggt_width": 3}
    extr = np.eye(4, dtype=np.float32)[None]
    K = np.array([[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]], np.float32)

    built = _build_geometry(args, pred, cam, np.array([0]), extr, K, [0], [0])

    # voxel = voxel_px * median_depth / fx = 1e-4 * 2 / 2 — far below spacing, no merging
    assert built["voxel"] == pytest.approx(1e-4)
    assert built["scene_points"].shape[0] == 8       # 9 pixels minus the human one
    assert built["n_human"] == 1
    # (1,1) at depth 2 with cx=cy=1, fx=fy=2 backprojects to the optical axis
    assert torch.allclose(built["human_points"][0], torch.tensor([0.0, 0.0, 2.0]))


def _fake_geometry(tmp_path: Path):
    geo = tmp_path / "geometry"
    geo.mkdir()
    extr = np.eye(4, dtype=np.float32)[None]
    extr[0, :3, 3] = [1.0, 2.0, 3.0]
    np.savez(geo / "transform.npz", scale=np.float32(1.0), metric=np.bool_(False),
             extrinsics=extr, fps=np.float32(30.0))
    np.savez(geo / "scene.npz", points=np.ones((4, 3), np.float32),
             voxel_size=np.float32(0.01), colors=np.zeros((4, 3), np.uint8))
    np.savez(geo / "human.npz", points=np.full((5, 3), 2.0, np.float32),
             frame_id=np.zeros(5, np.int32))
    return geo


def test_metricize_geometry_scales_everything_with_units(tmp_path):
    geo = _fake_geometry(tmp_path)

    total = _metricize_geometry(tmp_path, 3.0)

    assert total == pytest.approx(3.0)
    tr = np.load(geo / "transform.npz")
    assert bool(tr["metric"])
    assert float(tr["scale"]) == pytest.approx(3.0)
    np.testing.assert_allclose(tr["extrinsics"][0, :3, 3], [3.0, 6.0, 9.0])
    np.testing.assert_allclose(tr["extrinsics"][0, :3, :3], np.eye(3))  # rotation untouched
    sc = np.load(geo / "scene.npz")
    np.testing.assert_allclose(sc["points"], 3.0)
    assert float(sc["voxel_size"]) == pytest.approx(0.03)
    hu = np.load(geo / "human.npz")
    np.testing.assert_allclose(hu["points"], 6.0)
    np.testing.assert_array_equal(hu["frame_id"], 0)  # unitless fields untouched

    # a second metricization accumulates into the recorded total
    total = _metricize_geometry(tmp_path, 0.5)
    assert total == pytest.approx(1.5)
    assert float(np.load(geo / "transform.npz")["scale"]) == pytest.approx(1.5)
