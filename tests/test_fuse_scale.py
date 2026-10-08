"""Tests for the fuse → scale chain: voxel merge, geometry build, metricization."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from scripts.stages.estimate_scale import _metricize_geometry
from scripts.stages.fuse_geometry import _build_geometry, _merge_scene, _View, _visible_mask
from tools.geometry.camera_motion import image_motion, is_static
from tools.geometry.voxel import voxel_downsample
from tools.human_optim.prefit import (
    cloud_chamfer, finger_freeze_mask, joint_velocity_loss, smooth_finger_quats)


def test_merge_scene_collapses_boundary_twins():
    # Adversarial case: the twins straddle x=2, while the pair near 0 pulls the
    # pass-1 centroid minimum to 0.499. A second pass that re-derived its origin
    # from the centroids would place a face at 1.999 and MISS the twins — the
    # passes must share one grid origin.
    xs = [0.0, 0.998, 1.98, 1.99, 2.01, 2.02, 7.0]
    pts = torch.tensor([[x, 0.0, 0.0] for x in xs])
    conf = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    color = torch.full((7, 3), 100.0)

    points, confidence, col, count, coords = _merge_scene(pts, conf, color, voxel=1.0)

    assert points.shape[0] == 3  # near pair + ONE merged twin + far point
    merged = points[count == 4]
    assert merged.shape == (1, 3)
    assert float(merged[0, 0]) == pytest.approx(2.0, abs=1e-6)
    assert float(confidence[count == 4]) == pytest.approx(0.6)  # max over the twin
    assert coords.shape == points.shape


def test_voxel_downsample_shift_moves_the_grid():
    pts = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    _, _, counts_plain, _ = voxel_downsample(pts, 1.0)
    _, _, counts_shift, _ = voxel_downsample(pts, 1.0, shift=0.5)
    assert counts_plain.numel() == 1   # amin-origin puts both in cell [0, 1)
    assert counts_shift.numel() == 2   # half-shift puts a face at 0.5, between them


def test_visible_mask_prunes_the_slab_back():
    # One camera at the origin looking down +z (identity extrinsic), fx=fy=100,
    # 100x100 image. A 3x3 front sheet at z=2 with a duplicate ("ghost") sheet
    # 0.5 behind it — more than one voxel — plus one isolated point far behind
    # that is alone in its bin (nothing in front of it → visibility keeps it).
    xy = [(x, y) for x in (-0.1, 0.0, 0.1) for y in (-0.1, 0.0, 0.1)]
    front = [[x, y, 2.0] for x, y in xy]
    ghost = [[x * 1.25, y * 1.25, 2.5] for x, y in xy]   # same rays, deeper
    lone = [[0.8, 0.8, 4.0]]
    pts = torch.tensor(front + ghost + lone)
    extr = np.eye(4, dtype=np.float32)[None]
    K = np.array([[[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]]], np.float32)

    keep = _visible_mask(pts, extr, K, 100, 100, bin_px=6.0, voxel=0.2)

    assert keep[:9].all()          # the front sheet survives
    assert not keep[9:18].any()    # the ghost sheet is behind the front row everywhere
    assert keep[18]                # an isolated point wins its own bin


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
    args = SimpleNamespace(device="cpu", conf_drop_q=0.0, edge_rtol=0.0,
                           human_dilate=0, voxel_px=1e-4, min_scene_voxels=1)
    cam = {"vggt_height": 3, "vggt_width": 3}
    extr = np.eye(4, dtype=np.float32)[None]
    K = np.array([[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]], np.float32)
    views = [_View(source, pred / "vggt" / "depth", pred / "sam3", [0], [0])]

    built = _build_geometry(args, cam, np.array([0]), extr, K, views, np.zeros(1, np.int32))

    # voxel = voxel_px * median_depth / fx = 1e-4 * 2 / 2 — far below spacing, no merging
    assert built["voxel"] == pytest.approx(1e-4)
    assert built["scene_points"].shape[0] == 8       # 9 pixels minus the human one
    assert built["n_human"] == 1
    # (1,1) at depth 2 with cx=cy=1, fx=fy=2 backprojects to the optical axis
    assert torch.allclose(built["human_points"][0], torch.tensor([0.0, 0.0, 2.0]))
    # one unit normal per scene point, oriented toward the camera (plane at z=2,
    # camera at the origin ⇒ normals ≈ -z)
    normals = built["scene_normals"]
    assert normals.shape == built["scene_points"].shape
    assert torch.allclose(normals.norm(dim=-1), torch.ones(8), atol=1e-5)
    assert (normals[:, 2] < 0).all()


def _second_view(tmp_path: Path, pred: Path) -> _View:
    """A second camera: its own frames, depth and masks, and no reconstructed climber."""
    source = tmp_path / "frames_right"
    source.mkdir()
    cv2.imwrite(str(source / "0.png"), np.full((3, 3, 3), 64, np.uint8))
    depth_dir = pred / "vggt" / "depth" / "cam_right"
    mask_dir = pred / "sam3" / "cam_right"
    depth_dir.mkdir(parents=True)
    (mask_dir / "00").mkdir(parents=True)
    np.savez(depth_dir / "frame_000000.npz", depth=np.full((3, 3), 5.0, np.float32),
             depth_conf=np.ones((3, 3), np.float32))
    mask = np.zeros((3, 3), np.uint8)
    mask[1, 1] = 255
    cv2.imwrite(str(mask_dir / "00" / "frame_000000.png"), mask)
    return _View(source, depth_dir, mask_dir, [0], [])


def test_build_geometry_second_view_feeds_only_the_scene(tmp_path):
    # Camera row 0 is the SECOND view and row 1 the first, so a human frame id
    # taken from the camera row instead of the time step would come out as 1.
    source, pred = _tiny_clip(tmp_path)
    args = SimpleNamespace(device="cpu", conf_drop_q=0.0, edge_rtol=0.0,
                           human_dilate=0, voxel_px=1e-4, min_scene_voxels=1)
    cam = {"vggt_height": 3, "vggt_width": 3}
    right = np.eye(4, dtype=np.float32)
    right[0, 3] = 10.0        # far enough aside that neither view occludes the other
    extr = np.stack([right, np.eye(4, dtype=np.float32)])
    K = np.tile(np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]], np.float32),
                (2, 1, 1))
    views = [_View(source, pred / "vggt" / "depth", pred / "sam3", [0], [0]),
             _second_view(tmp_path, pred)]

    built = _build_geometry(args, cam, np.array([0, 0]), extr, K, views,
                            np.array([1, 0], np.int32))

    z = built["scene_points"][:, 2]
    assert built["scene_points"].shape[0] == 16        # 8 background pixels per view
    assert int((z == 2.0).sum()) == 8                  # view 0's depth
    assert int((z == 5.0).sum()) == 8                  # the second view's
    assert built["n_human"] == 1                       # only view 0 has a climber
    assert built["human_frame_id"].tolist() == [0]     # the time step, not the camera row


def test_estimate_point_normals_edge_cases():
    from tools.geometry.normals import estimate_point_normals

    cams = torch.zeros((1, 3))
    empty = estimate_point_normals(torch.zeros((0, 3)), cams)
    assert empty.shape == (0, 3)
    # a single point (k collapses to 1) must not crash and faces the camera
    one = estimate_point_normals(torch.tensor([[0.0, 0.0, 2.0]]), cams)
    assert one.shape == (1, 3)
    assert float(one.norm()) == pytest.approx(1.0, abs=1e-5)


def test_finger_freeze_mask_zeros_the_finger_tail():
    mask = finger_freeze_mask(211, "cpu")
    assert mask[: 7 + 4 * 21].eq(1).all()   # root SE3 + 21 body joints stay live
    assert mask[7 + 4 * 21:].eq(0).all()    # 30 finger joints frozen
    assert mask.numel() == 211


def _x_quats(angles: torch.Tensor) -> torch.Tensor:
    """``(T, 1, 4)`` scalar-last quaternions rotating ``angles`` (rad) about x."""
    half = 0.5 * angles
    zero = torch.zeros_like(half)
    return torch.stack([half.sin(), zero, zero, half.cos()], dim=-1)[:, None, :]


def _q_with_fingers(track: torch.Tensor) -> torch.Tensor:
    """Pose ``(T, 211)`` at identity, with every finger quaternion set from ``track (T, 30, 4)``."""
    num_frames = track.shape[0]
    q = torch.zeros(num_frames, 211)
    q[:, 6] = 1.0                                             # root quaternion (scalar last)
    q[:, 7: 7 + 4 * 21].reshape(num_frames, 21, 4)[..., 3] = 1.0
    q[:, 7 + 4 * 21:] = track.reshape(num_frames, -1)
    return q


def _quat_angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotation angle (deg) between unit quaternions, insensitive to the ``q ≡ −q`` sign."""
    return torch.rad2deg(2.0 * torch.arccos((a * b).sum(-1).abs().clamp(max=1.0)))


def test_smooth_finger_quats_survives_hemisphere_flips():
    # A hand held perfectly still, reported as −q on half the frames. Averaging the raw
    # components would cancel across the flip; hemisphere alignment must prevent that.
    still = _x_quats(torch.full((12,), 0.7))
    track = still.expand(12, 30, 4).clone()
    track[6:] *= -1.0
    q = _q_with_fingers(track)

    out = smooth_finger_quats(q.clone(), np.ones(12, bool), 2.0)
    fingers = out[:, 7 + 4 * 21:].reshape(12, 30, 4)

    assert torch.allclose(fingers.norm(dim=-1), torch.ones(12, 30), atol=1e-5)
    assert float(_quat_angle_deg(fingers, still).max()) == pytest.approx(0.0, abs=1e-3)
    assert torch.equal(out[:, : 7 + 4 * 21], q[:, : 7 + 4 * 21])          # body block untouched
    assert torch.equal(smooth_finger_quats(q.clone(), np.ones(12, bool), 0.0), q)


def test_smooth_finger_quats_removes_jitter_but_keeps_motion():
    torch.manual_seed(0)
    curl = torch.linspace(0.0, 1.0, 40)                        # a real, slow finger curl
    noisy = _x_quats(curl + 0.05 * torch.randn(40)).expand(40, 30, 4).contiguous()
    clean = _x_quats(curl)

    out = smooth_finger_quats(_q_with_fingers(noisy), np.ones(40, bool), 3.0)
    fingers = out[:, 7 + 4 * 21:].reshape(40, 30, 4)

    before = float(_quat_angle_deg(noisy[1:], noisy[:-1]).mean())
    after = float(_quat_angle_deg(fingers[1:], fingers[:-1]).mean())
    floor = float(_quat_angle_deg(clean[1:], clean[:-1]).mean())   # the curl's own step size
    assert after - floor < 0.25 * (before - floor)             # most of the excess step is gone
    assert float(_quat_angle_deg(fingers, clean).mean()) < \
        float(_quat_angle_deg(noisy, clean).mean())            # closer to the true curl


def test_smooth_finger_quats_does_not_smooth_across_a_tracking_gap():
    # Two valid spans at very different hand poses with untracked frames between them
    # (those hold a copy of a neighbouring pose). Each span is already constant, so a
    # correct filter leaves it exactly where it is instead of dragging the spans together.
    angles = torch.cat([torch.full((12,), 0.2), torch.full((8,), 1.4)])
    track = _x_quats(angles).expand(20, 30, 4).contiguous()
    valid = np.zeros(20, bool)
    valid[:8] = True
    valid[12:] = True

    q = _q_with_fingers(track)
    out = smooth_finger_quats(q.clone(), valid, 4.0)
    fingers = out[:, 7 + 4 * 21:].reshape(20, 30, 4)

    assert float(_quat_angle_deg(fingers[:8], track[:8]).max()) == pytest.approx(0.0, abs=1e-3)
    assert float(_quat_angle_deg(fingers[12:], track[12:]).max()) == pytest.approx(0.0, abs=1e-3)
    assert torch.equal(out[8:12], q[8:12])                     # untracked rows untouched


def test_cloud_chamfer_matches_only_visible_vertices():
    # One frame, two verts: the visible one sits 0.1 from the cloud point, the
    # hidden one exactly on it. The loss must reflect the VISIBLE distance.
    verts = torch.tensor([[[0.0, 0.0, 1.9], [0.0, 0.0, 2.0]]])   # (1, 2, 3)
    vis = torch.tensor([[True, False]])
    cloud = torch.tensor([[[0.0, 0.0, 2.0]]])                    # (1, 1, 3)
    valid = torch.tensor([[True]])

    loss = cloud_chamfer(verts, vis, cloud, valid, gm_scale_m=0.3)

    expected = 0.1 ** 2 / (0.3 ** 2 + 0.1 ** 2)                  # GM of the visible dist
    assert float(loss) == pytest.approx(expected, rel=1e-4)
    both = cloud_chamfer(verts, torch.tensor([[True, True]]), cloud, valid, gm_scale_m=0.3)
    assert float(both) == pytest.approx(0.0, abs=1e-6)           # hidden vert now allowed


def test_kinematic_init_prefers_matching_prefit(tmp_path):
    from tools.human_optim.stages import _init_params

    sp = {"num_frames": np.int32(4), "object_ids": np.array([2], np.int32),
          "transl": np.zeros((1, 4, 3), np.float32)}
    oids = np.asarray(sp["object_ids"], np.int32)

    params, name = _init_params(tmp_path, sp, oids, "t")   # no prefit.npz → converter
    assert name == "converter" and params is sp

    geo = tmp_path / "geometry"
    geo.mkdir()
    np.savez(geo / "prefit.npz", num_frames=np.int32(4), object_ids=oids,
             transl=np.ones((1, 4, 3), np.float32))
    params, name = _init_params(tmp_path, sp, oids, "t")
    assert name == "prefit"
    np.testing.assert_allclose(params["transl"], 1.0)

    np.savez(geo / "prefit.npz", num_frames=np.int32(4),
             object_ids=np.array([7], np.int32))           # stale tracking → hard error
    with pytest.raises(SystemExit):
        _init_params(tmp_path, sp, oids, "t")


def test_joint_velocity_loss_matches_converter_motion():
    t = torch.linspace(0, 1, 4)[:, None, None]
    joints0 = t * torch.tensor([[[1.0, 0.0, 0.0]]])              # constant velocity
    pairs = torch.tensor([0, 1, 2])
    assert float(joint_velocity_loss(joints0, joints0, pairs)) == 0.0   # same motion → free
    still = torch.zeros_like(joints0)
    assert float(joint_velocity_loss(still, joints0, pairs)) > 0.0     # killing motion costs


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


def test_metricize_heals_a_partial_rescale(tmp_path):
    geo = _fake_geometry(tmp_path)
    _metricize_geometry(tmp_path, 3.0)
    # Simulate a crash that rescaled transform+scene but left human.npz behind.
    np.savez(geo / "human.npz", points=np.full((5, 3), 2.0, np.float32),
             frame_id=np.zeros(5, np.int32), world_scale=np.float32(1.0))

    total = _metricize_geometry(tmp_path, 1.0)  # a re-run measures ~1 on the repaired cloud

    assert total == pytest.approx(3.0)
    hu = np.load(geo / "human.npz")
    np.testing.assert_allclose(hu["points"], 6.0)  # lagging file brought to the shared world
    assert float(hu["world_scale"]) == pytest.approx(3.0)
    np.testing.assert_allclose(np.load(geo / "scene.npz")["points"], 3.0)  # NOT double-scaled


# --- is the shot locked off? -------------------------------------------------

_K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]], np.float32)
_W, _H = 1280, 720


def _wall(scale: float = 1.0) -> np.ndarray:
    """A patch of scene 5 units in front of a camera at the origin looking down +z."""
    xy = np.stack(np.meshgrid(np.linspace(-2, 2, 12), np.linspace(-1, 1, 12)), -1).reshape(-1, 2)
    return (np.concatenate([xy, np.full((len(xy), 1), 5.0)], 1) * scale).astype(np.float32)


def _track(rotations: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Cam-from-world matrices for a camera with these orientations and centres."""
    extr = np.zeros((len(rotations), 4, 4), np.float32)
    extr[:, :3, :3] = rotations
    extr[:, :3, 3] = -np.einsum("fij,fj->fi", rotations, centres)
    extr[:, 3, 3] = 1.0
    return extr


def _yaw(angles_deg: np.ndarray) -> np.ndarray:
    a = np.radians(angles_deg)
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([c, z, s, z, o, z, -s, z, c], -1).reshape(-1, 3, 3).astype(np.float32)


def test_a_locked_camera_does_not_move_the_picture():
    extr = _track(_yaw(np.zeros(40)), np.zeros((40, 3), np.float32))
    K = np.repeat(_K[None], 40, 0)

    motion = image_motion(extr, K, _W, _H, _wall())

    assert motion == pytest.approx(0.0, abs=1e-9)
    assert is_static(motion)


def test_a_pan_counts_as_moving_even_though_the_camera_never_travels():
    # The centre is nailed to the origin the whole time — a test on the camera
    # position alone would call this static. The picture still slides.
    angles = np.linspace(0.0, 4.0, 40)
    extr = _track(_yaw(angles), np.zeros((40, 3), np.float32))
    K = np.repeat(_K[None], 40, 0)

    motion = image_motion(extr, K, _W, _H, _wall())

    centres = -np.einsum("fji,fj->fi", extr[:, :3, :3], extr[:, :3, 3])
    np.testing.assert_allclose(centres, 0.0, atol=1e-6)
    assert not is_static(motion)
    # The reading is the swing away from the MIDDLE of the sweep, not its full
    # width: the 90th-percentile frame sits ~1.6 deg off the median one, which at
    # f = 1000 px slides the picture ~30 px, i.e. ~2.5% of the width.
    assert motion == pytest.approx(0.026, rel=0.2)


def test_the_answer_survives_metricization():
    # The fused world is up-to-scale until estimate_scale rescales the cloud and
    # the camera translations together. The same clip must not change verdict.
    centres = np.stack([np.linspace(0.0, 0.3, 40), np.zeros(40), np.zeros(40)], -1).astype(np.float32)
    K = np.repeat(_K[None], 40, 0)
    unscaled = image_motion(_track(_yaw(np.zeros(40)), centres), K, _W, _H, _wall())

    metric = image_motion(_track(_yaw(np.zeros(40)), centres * 7.0), K, _W, _H, _wall(7.0))

    assert unscaled > 0
    assert metric == pytest.approx(unscaled, rel=1e-6)
