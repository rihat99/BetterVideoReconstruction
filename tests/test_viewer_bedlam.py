"""The viewer's BEDLAM2 path: which directories are clips, and the ground patch.

``_bedlam_gravity`` is the only arithmetic in the BEDLAM2 loader, and it is easy
to get wrong in a way nothing else notices: BEDLAM2 stores the world origin at
floor height as its ``ground_centroid``, and an Unreal level can sit hundreds of
metres from its own origin, so anything that treats that point as "where the
people are" draws a floor kilometres wide in the wrong place.
"""

from __future__ import annotations

import numpy as np

from tools.viewer.app import _discover_clips, _is_clip
from tools.viewer.loading import _bedlam_gravity
from tools.viewer.scene import _GROUND_GRID_MULT

_FLOOR_Y = -0.015           # BEDLAM2's ground_height: up is -y, so the floor is just below 0
_UP = np.array([0.0, -1.0, 0.0], np.float32)


def _write_gravity(tmp_path, floor_y: float = _FLOOR_Y):
    """A BEDLAM2 ``gt/gravity.npz``: the floor plane through the WORLD ORIGIN."""
    gt = tmp_path / "gt"
    gt.mkdir(exist_ok=True)
    np.savez(gt / "gravity.npz",
             gravity_world=np.array([0.0, 1.0, 0.0], np.float32),
             up_world=_UP, ground_normal=_UP,
             ground_centroid=np.array([0.0, floor_y, 0.0], np.float32),
             ground_height=np.float32(floor_y), ground_accepted=np.bool_(True),
             reliable=np.bool_(True), ground_angle_deg=np.float32(0.0),
             source=np.str_("bedlam2_gt"))
    return gt


def _person(pelvis: np.ndarray):
    """One entry of the ``people`` list; only joint 0 (the pelvis) is read."""
    bone_pos = np.zeros((len(pelvis), 52, 3), np.float32)
    bone_pos[:, 0, :] = pelvis
    return {"bone_pos": bone_pos}


def _walk(origin, steps: int = 60, span: float = 2.0):
    """A pelvis walking ``span`` metres in +x, one metre above the floor."""
    walk = np.tile(np.asarray(origin, np.float32), (steps, 1))
    walk[:, 0] += np.linspace(0.0, span, steps, dtype=np.float32)
    return walk


def test_patch_sits_under_the_action_not_on_the_stored_centroid(tmp_path):
    # An Unreal level 600 m from its own origin — the common BEDLAM2 case.
    far = np.array([562.8, _FLOOR_Y - 1.0, -266.2], np.float32)
    gt = _write_gravity(tmp_path)
    centers = np.tile(np.array([562.0, _FLOOR_Y - 1.6, -263.0], np.float32), (40, 1))

    ground = _bedlam_gravity(gt, centers, [_person(_walk(far))])["ground"]

    assert abs(ground["centroid"][0] - 562.0) < 5.0               # under the people
    assert abs(ground["centroid"][2] - (-265.0)) < 5.0
    # The drawn grid must actually cover them.
    half = ground["extent"] * _GROUND_GRID_MULT / 2.0
    for point in np.concatenate([centers, _walk(far)]):
        assert abs(point[0] - ground["centroid"][0]) <= half
        assert abs(point[2] - ground["centroid"][2]) <= half


def test_patch_keeps_the_ground_truth_floor_height(tmp_path):
    gt = _write_gravity(tmp_path)
    pelvis = _walk([3.0, _FLOOR_Y - 0.95, -7.0])
    centers = np.tile(np.array([0.0, _FLOOR_Y - 1.5, 0.0], np.float32), (10, 1))

    ground = _bedlam_gravity(gt, centers, [_person(pelvis)])["ground"]

    # Re-centring moves the patch along the floor only — never off it.
    assert abs(ground["centroid"][1] - _FLOOR_Y) < 1e-9
    assert np.allclose(ground["normal"], _UP)


def test_extent_follows_the_action_not_the_distance_to_the_origin(tmp_path):
    """The same motion 600 m out must draw the same size of floor as at the origin."""
    gt = _write_gravity(tmp_path)
    centers = np.zeros((4, 3), np.float32)
    near = _bedlam_gravity(gt, centers, [_person(_walk([1.0, -1.0, 1.0]))])["ground"]
    far = _bedlam_gravity(gt, centers + np.float32(600.0),
                          [_person(_walk([601.0, -1.0, 601.0]))])["ground"]

    assert far["extent"] == near["extent"]


def test_a_still_scene_still_gets_a_floor_to_stand_on(tmp_path):
    """Everything at one spot — the patch falls back to its minimum, never zero."""
    gt = _write_gravity(tmp_path)
    still = np.tile(np.array([2.0, _FLOOR_Y - 0.9, 2.0], np.float32), (30, 1))

    ground = _bedlam_gravity(gt, still[:1], [_person(still)])["ground"]

    assert ground["extent"] * _GROUND_GRID_MULT >= 8.0


def test_gravity_is_the_unit_down_of_the_ground_truth(tmp_path):
    gt = _write_gravity(tmp_path)
    out = _bedlam_gravity(gt, np.zeros((2, 3), np.float32), [_person(np.zeros((2, 3), np.float32))])

    assert np.allclose(out["g"], [0.0, 1.0, 0.0])      # up is -y, so down is +y
    assert out["is_fallback"] is False
    assert out["source"] == "bedlam2_gt"


def test_people_that_never_pose_are_skipped(tmp_path):
    """``people`` carries ``None`` for anyone with no valid frame."""
    gt = _write_gravity(tmp_path)
    centers = np.zeros((3, 3), np.float32)
    pelvis = _walk([4.0, -1.0, 0.0])
    both = _bedlam_gravity(gt, centers, [None, _person(pelvis), None])["ground"]
    only = _bedlam_gravity(gt, centers, [_person(pelvis)])["ground"]

    assert both["extent"] == only["extent"]
    assert np.allclose(both["centroid"], only["centroid"])


def test_a_fused_tree_and_a_bedlam_scene_are_both_clips(tmp_path):
    fused = tmp_path / "out_clip"
    (fused / "geometry").mkdir(parents=True)
    (fused / "geometry" / "scene.npz").touch()
    bedlam = tmp_path / "0u000000"
    (bedlam / "gt").mkdir(parents=True)
    (bedlam / "gt" / "smplx.npz").touch()
    empty = tmp_path / "nothing"
    (empty / "gt").mkdir(parents=True)          # a gt/ dir alone is not enough

    assert _is_clip(fused)
    assert _is_clip(bedlam)
    assert not _is_clip(empty)
    assert _discover_clips(tmp_path) == [bedlam, fused]      # sorted by name
    assert _discover_clips(bedlam) == [bedlam]               # pointed at one scene
    assert _discover_clips(empty) == []
