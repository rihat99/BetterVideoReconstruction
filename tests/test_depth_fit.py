from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import scripts.stages.estimate_contacts  # noqa: F401  # installs the better_human path shim
from tools.human_optim.depth_fit import NO_DATA_TRUST, joint_depth_fit


def _camera(frames: int = 1):
    extr = torch.eye(4, dtype=torch.float32).repeat(frames, 1, 1)
    K = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32).repeat(frames, 1, 1)
    return extr, K


def _mesh_and_cloud(stretch_joint: int | None = 1, stretch_m: float = 0.30):
    vertices = []
    owners = []
    for joint, x in enumerate((-0.4, 0.0, 0.4)):
        for y in (-0.16, -0.08, 0.0, 0.08, 0.16):
            vertices.append([x, y, 2.0])
            owners.append(joint)
    mesh = torch.tensor(vertices, dtype=torch.float32)[None]
    cloud = mesh[0].clone()
    if stretch_joint is not None:
        selected = torch.tensor(owners) == stretch_joint
        cloud[selected] *= (2.0 + stretch_m) / 2.0
    return mesh, torch.tensor(owners), cloud


def _fit(mesh, owners, cloud, obj_id=None, *, object_id=0, min_points=5):
    extr, K = _camera(mesh.shape[0])
    if obj_id is None:
        obj_id = torch.zeros(cloud.shape[0], dtype=torch.int32)
    return joint_depth_fit(
        mesh, cloud, torch.zeros(cloud.shape[0], dtype=torch.int32), obj_id, object_id,
        extr, K, owners, 3, cell_px=4, min_points=min_points, trust_scale_log=0.05)


def test_stretched_limb_has_signed_gap_on_owning_joint():
    mesh, owners, cloud = _mesh_and_cloud(stretch_m=0.30)
    gap, points, trust = _fit(mesh, owners, cloud)
    np.testing.assert_allclose(gap[0, [0, 2]], [0.0, 0.0], atol=1e-5)
    assert gap[0, 1].item() == pytest.approx(30.0, abs=1e-4)
    np.testing.assert_array_equal(points[0].numpy(), np.full(3, 5.0, np.float32))
    assert trust[0, 1] < trust[0, 0]


def test_too_few_points_has_nan_gap_and_no_data_trust():
    mesh, owners, cloud = _mesh_and_cloud(stretch_joint=None)
    keep = owners != 2
    keep[torch.nonzero(owners == 2)[:3, 0]] = True
    gap, points, trust = _fit(mesh, owners, cloud[keep], min_points=5)
    assert points[0, 2].item() == 3.0
    assert torch.isnan(gap[0, 2])
    assert trust[0, 2].item() == pytest.approx(NO_DATA_TRUST)


def test_camera_pose_invariance():
    # The same scene expressed in a rotated+translated world frame must give the
    # same per-joint gaps — pins the world→camera extrinsics convention, which
    # the identity-camera tests above cannot catch.
    mesh, owners, cloud = _mesh_and_cloud(stretch_m=0.30)
    gap_id, _, _ = _fit(mesh, owners, cloud)
    c, s = math.cos(0.7), math.sin(0.7)
    R = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)
    t = torch.tensor([0.3, -1.2, 0.8], dtype=torch.float32)
    world_mesh = mesh @ R.T + t                       # cam→world: x_w = R x_c + t
    world_cloud = cloud @ R.T + t
    extr = torch.eye(4, dtype=torch.float32)          # world→cam inverse
    extr[:3, :3] = R.T
    extr[:3, 3] = -R.T @ t
    _, K = _camera(1)
    zeros = torch.zeros(world_cloud.shape[0], dtype=torch.int32)
    gap, _, _ = joint_depth_fit(
        world_mesh, world_cloud, zeros, zeros, 0, extr[None], K, owners, 3,
        cell_px=4, min_points=5, trust_scale_log=0.05)
    np.testing.assert_allclose(gap[0].numpy(), gap_id[0].numpy(), atol=1e-2)


def test_obj_id_separates_two_people():
    mesh, owners, clean = _mesh_and_cloud(stretch_joint=None)
    _, _, stretched_other = _mesh_and_cloud(stretch_joint=1, stretch_m=0.60)
    cloud = torch.cat((clean, stretched_other), dim=0)
    obj_id = torch.cat((
        torch.zeros(clean.shape[0], dtype=torch.int32),
        torch.ones(stretched_other.shape[0], dtype=torch.int32)))
    gap, points, _ = _fit(mesh, owners, cloud, obj_id)
    np.testing.assert_allclose(gap[0].numpy(), np.zeros(3, np.float32), atol=1e-5)
    np.testing.assert_array_equal(points[0].numpy(), np.full(3, 5.0, np.float32))
    other_gap, other_points, _ = _fit(mesh, owners, cloud, obj_id, object_id=1)
    np.testing.assert_allclose(other_gap[0, [0, 2]], [0.0, 0.0], atol=1e-5)
    assert other_gap[0, 1].item() == pytest.approx(60.0, abs=1e-4)
    np.testing.assert_array_equal(other_points[0].numpy(), np.full(3, 5.0, np.float32))
