from __future__ import annotations

import warnings
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from scripts.audit_scenes import _finite_percentiles, _person_rows
from scripts.stages.estimate_uncertainty import _mean_contact_conf, _score_person
from scripts.stages.fuse_geometry import _build_geometry, _record_anchor_veto


def test_anchor_veto_diagnostics_describe_the_baseline_that_ships():
    anchor = {
        "rescale_r": 0.8,
        "depth_refuted": np.array([True, False]),
        "anchor_pred_gap_q": np.ones((2, 3), np.float32),
        "anchor_true_touch": 0.01,
        "anchor_true_burial": -0.2,
    }

    _record_anchor_veto(anchor, touch_m=0.31, burial_m=-0.04)

    assert anchor["rescale_r"] == 1.0
    assert not anchor["depth_refuted"].any()
    assert np.isnan(anchor["anchor_pred_gap_q"]).all()
    assert anchor["anchor_true_touch"] == pytest.approx(0.31)
    assert anchor["anchor_true_burial"] == pytest.approx(-0.04)


def test_fusion_split_scale_geometry_signature_is_unchanged(tmp_path):
    """The diagnostics fix must not alter the restored fusion geometry policy."""
    source = tmp_path / "frames"
    source.mkdir()
    cv2.imwrite(str(source / "0.png"), np.full((3, 3, 3), 127, np.uint8))

    pred = tmp_path / "pred"
    (pred / "vggt" / "depth").mkdir(parents=True)
    (pred / "sam3" / "00").mkdir(parents=True)
    np.savez(
        pred / "vggt" / "depth" / "frame_000000.npz",
        depth=np.full((3, 3), 2.0, np.float32),
        depth_conf=np.ones((3, 3), np.float32),
    )
    mask = np.zeros((3, 3), np.uint8)
    mask[1, 1] = 255
    cv2.imwrite(str(pred / "sam3" / "00" / "frame_000000.png"), mask)

    args = SimpleNamespace(
        source=source,
        device="cpu",
        conf_drop_q=0.0,
        edge_rtol=0.0,
        human_dilate=0,
        voxel_scene=1e-4,
        min_scene_voxels=1,
    )
    cam = {"vggt_height": 3, "vggt_width": 3}
    extr = np.eye(4, dtype=np.float32)[None]
    intrinsics = np.array(
        [[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )

    base = _build_geometry(
        args, pred, cam, np.array([0]), extr, intrinsics, [0], [0], 1.0,
        base_scale=1.0, fixed_human_ids=set(),
    )
    candidate = _build_geometry(
        args, pred, cam, np.array([0]), extr, intrinsics, [0], [0], 0.75,
        base_scale=1.0, fixed_human_ids={0},
    )

    assert torch.allclose(candidate["human_points"], base["human_points"])
    distance = torch.cdist(candidate["scene_points"], base["scene_points"] * 0.75)
    assert float(distance.min(dim=1).values.max()) < 1e-6
    assert float(distance.min(dim=0).values.max()) < 1e-6


def test_audit_percentiles_and_person_rows_tolerate_all_nan_chamfer(tmp_path):
    got = _finite_percentiles([np.nan, np.nan])
    assert got.shape == (5,)
    assert np.isnan(got).all()

    class Paths:
        def feature_dir(self, stage, sid):
            return tmp_path / stage / sid

    sid = "scene"
    human_dir = Paths().feature_dir("human_optim", sid)
    geometry_dir = Paths().feature_dir("geometry", sid)
    human_dir.mkdir(parents=True)
    geometry_dir.mkdir(parents=True)
    np.savez(
        human_dir / "kinematic_1.npz",
        object_ids=np.array([4], np.int32),
        valid_mask=np.array([[True, True]]),
    )
    np.savez(
        human_dir / "contacts_1.npz",
        joint_contact=np.zeros((1, 2, 1), bool),
        joint_sdf_cm=np.array([[[1.0], [2.0]]], np.float32),
    )
    np.savez(
        human_dir / "uncertainty_1.npz",
        valid=np.array([[True, True]]),
        chamfer_cm=np.full((1, 2), np.nan, np.float32),
        obs_score=np.array([[0.5, 0.7]], np.float32),
    )
    np.savez(geometry_dir / "scene.npz", points=np.zeros((10, 3), np.float32))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rows = _person_rows(Paths(), sid, min_valid=1)

    assert len(rows) == 1
    assert np.isnan(rows[0]["med_chamfer_cm"])
    assert np.isnan(rows[0]["p90_chamfer_cm"])


def _reprojection_metrics(n_frames: int) -> dict[str, np.ndarray]:
    return {
        "L_sap": np.full(n_frames, 0.1, np.float32),
        "L_init": np.full(n_frames, 0.1, np.float32),
        "n_kp": np.ones(n_frames, np.float32),
        "reproj_sap_px": np.ones(n_frames, np.float32),
        "reproj_init_px": np.ones(n_frames, np.float32),
    }


def test_uncertainty_absent_evidence_is_not_observable_or_reliable():
    valid = np.array([True, True])
    result = _score_person(
        _reprojection_metrics(2),
        np.full(2, np.nan, np.float32),
        np.zeros(2, np.float32),
        np.zeros(2, np.float32),
        valid,
        (8.0, 30.0, 10.0),
    )

    np.testing.assert_array_equal(result["obs_score"], np.zeros(2, np.float32))
    assert not result["reliable"].any()

    observed = _score_person(
        _reprojection_metrics(2),
        np.full(2, 5.0, np.float32),
        np.full(2, 100.0, np.float32),
        np.ones(2, np.float32),
        valid,
        (8.0, 30.0, 10.0),
    )
    assert observed["reliable"].all()


def test_mean_contact_conf_is_warning_free_for_frames_without_contact():
    contact = np.array([[False, False], [True, True]])
    confidence = np.array([[0.9, 0.8], [0.2, 0.6]], np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        got = _mean_contact_conf(contact, confidence)

    assert np.isnan(got[0])
    assert got[1] == pytest.approx(0.4)
