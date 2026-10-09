"""Force-plate loading for the viewer: measured plates keep their centre of pressure and
our ``estimated_forces.npz`` comes back in the world frame (``bvr.viewer.datasets_layers``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.run_dataset import build_view_farm
from bvr.viewer.datasets_layers import load_plate_estimate, load_sensor_forces

_R = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]], np.float32)
_T = np.array([1.0, 2.0, 3.0], np.float32)


def test_wall_sensor_file_has_no_cop(tmp_path: Path):
    (tmp_path / "human_optim").mkdir()
    np.savez(tmp_path / "human_optim" / "sensor_forces.npz",
             reaction_world=np.zeros((5, 4, 3), np.float32), limbs=np.array(["left_hand"] * 4))
    assert load_sensor_forces(tmp_path)["cop"] is None


def test_plate_sensor_file_keeps_cop(tmp_path: Path):
    (tmp_path / "human_optim").mkdir()
    cop = np.full((5, 4, 3), np.nan, np.float32)
    cop[:, 2] = [0.1, 0.0, 0.2]
    np.savez(tmp_path / "human_optim" / "sensor_forces.npz",
             reaction_world=np.zeros((5, 4, 3), np.float32), cop_world=cop,
             limbs=np.array(["left_hand", "right_hand", "left_foot", "right_foot"]))
    out = load_sensor_forces(tmp_path)
    np.testing.assert_array_equal(out["cop"][:, 2], cop[:, 2])
    assert np.isnan(out["cop"][:, 0]).all()


def test_plate_estimate_is_rotated_into_world(tmp_path: Path):
    (tmp_path / "human_optim").mkdir()
    force = np.zeros((2, 2, 3), np.float32)
    force[:, :, 1] = 100.0                                    # straight up in Y-up ground
    cop = np.tile(np.array([0.5, 0.0, 0.25], np.float32), (2, 2, 1))
    np.savez(tmp_path / "human_optim" / "estimated_forces.npz", force=force, cop=cop,
             limbs=np.array(["right_foot", "left_foot"]), R_world_from_mocap=_R,
             t_world_from_mocap=_T)
    out = load_plate_estimate(tmp_path)
    assert out["limbs"] == ["right_foot", "left_foot"]
    np.testing.assert_allclose(out["rw"][0, 0], [0.0, -100.0, 0.0])
    np.testing.assert_allclose(out["cop"][0, 0], _R @ cop[0, 0] + _T)


def test_no_estimate_file_means_no_plate_layer(tmp_path: Path):
    assert load_plate_estimate(tmp_path) is None


def test_view_farm_aliases_each_variant_farm_by_clip_name(tmp_path: Path):
    for clip in ("s2_STS1_cam1", "s3_DJ1_cam1"):
        (tmp_path / clip / "_lab" / "geometry").mkdir(parents=True)
        (tmp_path / clip / "_lab" / "geometry" / "scene.npz").write_bytes(b"")
    (tmp_path / "_batch_manifest.jsonl").write_text("{}\n")
    view = build_view_farm(tmp_path, "lab")
    assert view == tmp_path / "_view_lab"
    assert sorted(p.name for p in view.iterdir() if p.name != "_batch_manifest.jsonl") == [
        "s2_STS1_cam1", "s3_DJ1_cam1"]
    assert (view / "s2_STS1_cam1" / "geometry" / "scene.npz").exists()
    assert (view / "_batch_manifest.jsonl").read_text() == "{}\n"
    build_view_farm(tmp_path, "lab")   # idempotent: relinking does not fail
