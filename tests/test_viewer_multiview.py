"""The viewer's multi-view reading: field of view and the calibrated camera rig."""

from __future__ import annotations

import numpy as np

from tools.viewer.loading import _load_view_cameras, fov_y_rad


def _pixel_k(fx: float, fy: float, w: int, h: int) -> np.ndarray:
    return np.array([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]], np.float32)


def test_fov_comes_straight_from_vggt_when_recorded(tmp_path):
    path = tmp_path / "transform.npz"
    np.savez(path, fov_y_deg=np.array([40.0, 50.0], np.float32),
             intrinsics_px_orig=np.stack([_pixel_k(900.0, 900.0, 1920, 1080)] * 2),
             image_height=np.int32(1080))
    with np.load(path) as t:
        np.testing.assert_allclose(np.degrees(fov_y_rad(t)), [40.0, 50.0], atol=1e-5)


def test_fov_falls_back_to_the_pixel_focal_length(tmp_path):
    # A calibrated rig saves no fov_y_deg: 2·atan(h / 2f) with h == 2f is 90°.
    path = tmp_path / "transform.npz"
    np.savez(path, intrinsics_px_orig=np.stack([_pixel_k(900.0, 540.0, 1920, 1080)] * 3),
             image_height=np.int32(1080))
    with np.load(path) as t:
        np.testing.assert_allclose(np.degrees(fov_y_rad(t)), [90.0] * 3, atol=1e-5)


def test_no_cameras_npz_is_not_a_multi_view_clip(tmp_path):
    assert _load_view_cameras(tmp_path) is None


def test_calibrated_rig_reports_fov_aspect_and_lag(tmp_path):
    np.savez(tmp_path / "cameras.npz",
             view_names=np.array(["cam_left", "cam_3"]),
             K=np.stack([_pixel_k(900.0, 540.0, 1920, 1080),
                         _pixel_k(900.0, 960.0, 1080, 1920)]),
             extrinsics=np.stack([np.eye(4, dtype=np.float32)] * 2),
             image_size=np.array([[1920, 1080], [1080, 1920]], np.int64),
             frame_offset=np.array([0, 2], np.int32))
    views = _load_view_cameras(tmp_path)
    assert views["names"] == ["cam_left", "cam_3"]
    np.testing.assert_allclose(np.degrees(views["fov_y"]), [90.0, 90.0], atol=1e-5)
    np.testing.assert_allclose(views["aspect"], [1920 / 1080, 1080 / 1920])
    np.testing.assert_array_equal(views["offset"], [0, 2])


def test_a_rig_without_a_measured_lag_reads_as_synchronised(tmp_path):
    np.savez(tmp_path / "cameras.npz",
             view_names=np.array(["a", "b"]),
             K=np.stack([_pixel_k(900.0, 900.0, 1920, 1080)] * 2),
             extrinsics=np.stack([np.eye(4, dtype=np.float32)] * 2),
             image_size=np.array([[1920, 1080]] * 2, np.int64))
    assert _load_view_cameras(tmp_path)["offset"].tolist() == [0, 0]
