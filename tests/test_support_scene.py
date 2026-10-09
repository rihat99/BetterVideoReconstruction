"""The synthetic support scene: the floor lattice, the plateau rule, and the variant farm.

``bvr/stages/build_support_scene.py`` replaces the fused cloud with a scene built from
the fitted ground plane plus one patch per surface the body was found resting on. Nothing
downstream knows it is synthetic, so the file has to satisfy the same contract the fuse
stage's ``geometry/scene.npz`` does — checked here by handing the written cloud straight to
the loss the kinematics stages read it with. The plateau rule carries the rest of the
weight: a surface has to be separated from the floor, flat, held without a break, and, for
the sit bones, below the knees. Each of those is a test, because each of them was a false
plateau before it was a rule.

The farm tests cover the destructive failure mode: a support step pointed at a farm whose
``geometry`` is a symlink would overwrite the clip's real fused cloud.

Everything is float32, the dtype the pipeline carries scene geometry in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts import run_dataset  # noqa: E402
from bvr.stages.build_support_scene import (  # noqa: E402
    FLOOR_MARGIN_M, PLATEAU_INPLANE_M, PLATEAU_MAX_SPAN_M, PLATEAU_MIN_STILL_S,
    PLATEAU_SIT_BELOW_KNEE_M, SPACING_M,
    detect_plateaus, disc_points, frame_groups, plane_basis, run,
)
from scripts.run_dataset import VARIANT_CHAINS, VARIANT_STEPS, build_farm, owns_scene  # noqa: E402

from bvr.human_optim.losses import scene_signed_distance  # noqa: E402

FLOORPLANE_CONFIG = _REPO_ROOT / "configs" / "human_optimization_floorplane.toml"

#: A plane that is not axis-aligned, so "on the plane" is a real claim about the lattice.
UP = np.array([0.03, 0.02, 0.999], dtype=np.float32)
UP /= np.linalg.norm(UP)
CENTROID = np.array([0.4, -1.2, 2.7], dtype=np.float32)
GROUND_EXTENT = 1.0
#: A normal the ground fit and GeoCalib disagree about, to prove the lattice follows the fit.
TILTED_UP = np.array([0.25, 0.05, 0.967], dtype=np.float32)
TILTED_UP /= np.linalg.norm(TILTED_UP)

FPS = 60.0
STILL_M = 0.04
#: The contact-frame set the LabValidation configs build, in the body's own order.
FRAME_NAMES = ["heel_left", "big_toe_left", "toe_above_left", "heel_right", "big_toe_right",
               "toe_above_right", "sit_left", "sit_right"]
GROUPS = frame_groups(FRAME_NAMES)
STILL_FRAMES = int(round(PLATEAU_MIN_STILL_S * FPS))


@pytest.fixture()
def clip_dir(tmp_path: Path) -> Path:
    """A clip tree with just what a floor-only support scene reads: gravity + transform."""
    (tmp_path / "geocalib").mkdir()
    (tmp_path / "geometry").mkdir()
    np.savez(tmp_path / "geocalib" / "gravity.npz",
             ground_accepted=np.bool_(True), ground_centroid=CENTROID,
             ground_normal=UP, up_world=TILTED_UP, ground_extent=np.float32(GROUND_EXTENT))
    np.savez(tmp_path / "geometry" / "transform.npz",
             scale=np.float32(4.25), fps=np.float32(FPS))
    return tmp_path


def plane_coordinates(points: np.ndarray, center: np.ndarray,
                      up: np.ndarray = UP) -> np.ndarray:
    """``(M, 2)`` in-plane lattice coordinates of ``points (M, 3)`` about ``center``."""
    axis_u, axis_v = plane_basis(up)
    offset = points - center
    return np.stack([offset @ axis_u, offset @ axis_v], axis=1)


def trajectory(height_m: float, still: slice, total_frames: int, *, group: str = "foot_left",
               knee_m: float = 0.5, offset_m: float = 0.0
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(points (T, F, 3), knee_height (T,), drift (T, F))`` for one group parked at a height.

    The columns of ``group`` sit ``height_m`` above the plane and drift 5 mm per window over
    the frames in ``still``; everywhere else they are a metre up and moving. Every other
    column stays still on the floor, which is what gives the floor its own height cluster.
    ``offset_m`` slides the group's in-plane position, for the in-plane clustering tests.
    """
    points = np.zeros((total_frames, len(FRAME_NAMES), 3), dtype=np.float32)
    drift = np.full((total_frames, len(FRAME_NAMES)), 0.005, dtype=np.float32)
    axis_u, axis_v = plane_basis(UP)
    columns = GROUPS[group]
    height = np.zeros((total_frames, len(FRAME_NAMES)), dtype=np.float32)
    height[:, columns] = 1.0
    height[still, columns] = np.float32(height_m)
    drift[:, columns] = 0.5
    drift[still, columns] = 0.005
    for column in range(len(FRAME_NAMES)):
        shift = np.float32(0.05 * column + (offset_m if column in columns else 0.0))
        points[:, column] = (CENTROID + shift * axis_u + np.float32(0.02) * axis_v
                             + height[:, column, None] * UP)
    return points, np.full(total_frames, np.float32(knee_m)), drift


def detect(points: np.ndarray, knee: np.ndarray, drift: np.ndarray,
           valid: np.ndarray | None = None) -> list:
    """:func:`detect_plateaus` on this module's plane and groups."""
    if valid is None:
        valid = np.ones(len(points), bool)
    return detect_plateaus(points, knee, drift, valid, GROUPS, CENTROID, UP,
                           still_m=STILL_M, fps=FPS)


def accepted(plateaus: list) -> list:
    """Only the clusters that became patches."""
    return [p for p in plateaus if p.accepted]


# ── the floor lattice ────────────────────────────────────────────────────────────────

def test_floor_lattice_lies_on_the_fitted_plane_with_its_own_normal(clip_dir: Path) -> None:
    """Every floor point is on ``ground_normal``'s plane and carries that normal."""
    scene = dict(np.load(run(clip_dir, FLOORPLANE_CONFIG), allow_pickle=True))
    points = scene["points"]
    assert points.dtype == np.float32 and scene["normals"].dtype == np.float32
    # ground_normal, NOT up_world — the two differ by 13 deg in this fixture.
    assert np.degrees(np.arccos(float(UP @ TILTED_UP))) > 10.0
    assert np.abs((points - CENTROID) @ UP).max() < 1e-5
    assert np.allclose(scene["normals"], UP[None], atol=1e-6)
    assert np.allclose(scene["floor_normal"], UP, atol=1e-6)
    assert np.array_equal(scene["confidence"], np.ones(len(points), np.float32))
    assert np.array_equal(scene["count"], np.ones(len(points), np.int32))
    assert float(scene["voxel_size"]) == pytest.approx(SPACING_M)
    assert float(scene["world_scale"]) == pytest.approx(4.25)
    assert bool(scene["synthetic"]) and int(scene["floor_points"]) == len(points)

    radius = 0.5 * GROUND_EXTENT + FLOOR_MARGIN_M
    assert float(scene["floor_radius_m"]) == pytest.approx(radius)
    planar = plane_coordinates(points, CENTROID)
    assert np.linalg.norm(planar, axis=1).max() <= radius + 1e-6
    # A disc of this radius at this pitch, to the half-cell the boundary can round by.
    assert len(points) == pytest.approx(np.pi * radius**2 / SPACING_M**2, rel=0.02)
    assert np.abs(planar - np.round(planar / SPACING_M) * SPACING_M).max() < 1e-6
    columns = np.unique(np.round(planar[:, 0] / SPACING_M).astype(np.int32))
    assert np.array_equal(columns, np.arange(columns[0], columns[-1] + 1))


def test_flipped_ground_normal_is_oriented_up(clip_dir: Path) -> None:
    """A plane fit that points into the floor is turned over, not used upside down."""
    np.savez(clip_dir / "geocalib" / "gravity.npz",
             ground_accepted=np.bool_(True), ground_centroid=CENTROID,
             ground_normal=-UP, up_world=TILTED_UP, ground_extent=np.float32(GROUND_EXTENT))
    scene = dict(np.load(run(clip_dir, FLOORPLANE_CONFIG), allow_pickle=True))
    assert np.allclose(scene["normals"], UP[None], atol=1e-6)


def test_written_scene_reads_back_through_the_penetration_loss(clip_dir: Path) -> None:
    """The consumer's signed distance is positive above the floor and negative below it."""
    scene = dict(np.load(run(clip_dir, FLOORPLANE_CONFIG), allow_pickle=True))
    points = torch.as_tensor(scene["points"], dtype=torch.float32)[None]
    normals = torch.as_tensor(scene["normals"], dtype=torch.float32)[None]
    confidence = torch.as_tensor(scene["confidence"], dtype=torch.float32)[None]
    valid = torch.ones(points.shape[:2], dtype=torch.bool)
    probes = torch.as_tensor(
        np.stack([CENTROID + np.float32(0.12) * UP, CENTROID - np.float32(0.07) * UP]),
        dtype=torch.float32)[None]
    signed, distance, _, has = scene_signed_distance(probes, points, normals, confidence, valid)
    assert bool(has.all())
    assert signed[0, 0].item() == pytest.approx(0.12, abs=1e-3)
    assert signed[0, 1].item() == pytest.approx(-0.07, abs=1e-3)
    # Nearest point is directly under the probe, up to the lattice's half-cell offset.
    assert distance[0, 0].item() == pytest.approx(0.12, abs=SPACING_M)


def test_support_scene_is_idempotent(clip_dir: Path) -> None:
    """A second floor-only run rewrites the identical cloud over the first."""
    first = dict(np.load(run(clip_dir, FLOORPLANE_CONFIG), allow_pickle=True))
    second = dict(np.load(run(clip_dir, FLOORPLANE_CONFIG), allow_pickle=True))
    assert sorted(first) == sorted(second)
    for key in first:
        assert np.array_equal(first[key], second[key]), key


def test_disc_points_are_symmetric_about_their_centre() -> None:
    """The patch lattice is centred on the point it is asked for, whatever the radius."""
    center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    points = disc_points(center, UP, 0.35, SPACING_M)
    assert points.dtype == np.float32
    assert np.abs(points.mean(0) - center).max() < 1e-5
    assert np.linalg.norm(plane_coordinates(points, center), axis=1).max() <= 0.35 + 1e-6


# ── the plateau rule ─────────────────────────────────────────────────────────────────

def test_frame_groups_drop_the_top_of_the_toes() -> None:
    """A foot rests on heel, ball and big toe — never on ``toe_above``."""
    assert GROUPS == {"foot_left": [0, 1], "foot_right": [3, 4], "sit": [6, 7]}
    assert frame_groups(["toe_above_left", "toe_above_right"]) == {}


def test_box_under_a_still_foot_is_drawn_under_it() -> None:
    """A foot parked 0.30 m up grows one patch, at or below the frames that found it."""
    points, knee, drift = trajectory(0.30, slice(0, 90), 150)
    found = accepted(detect(points, knee, drift))
    assert [(p.group, p.reject) for p in found] == [("foot_left", "")]
    plateau = found[0]
    assert plateau.height_m == pytest.approx(0.30, abs=1e-5)
    assert plateau.height_m <= plateau.height_lo_m + 1e-6
    assert (plateau.frame_start, plateau.frame_end) == (0, 89)
    assert plateau.steps == 90 and plateau.longest_run == 90 and plateau.samples == 180
    assert np.isnan(plateau.sit_below_knee_m)
    assert float((plateau.center - CENTROID) @ UP) == pytest.approx(0.30, abs=1e-5)
    expected = plane_coordinates(points[:90, GROUPS["foot_left"]].reshape(-1, 3), CENTROID)
    assert plane_coordinates(plateau.center[None], CENTROID)[0] == pytest.approx(
        np.median(expected, axis=0), abs=1e-5)


def test_moving_foot_grows_no_plateau() -> None:
    """A foot that never holds still is never eligible, so only the floor clusters."""
    points, knee, drift = trajectory(0.30, slice(0, 0), 150)
    assert accepted(detect(points, knee, drift)) == []


def test_floor_tail_never_becomes_a_shelf() -> None:
    """A foot on the floor whose reconstruction floats to 0.13 m stays one floor cluster.

    This is the failure the un-truncated clustering exists for: gate the heights at 0.10 m
    first and the tail above it looks like a surface with ``height_lo`` exactly 0.10.
    """
    total = 240
    points, knee, drift = trajectory(0.0, slice(0, total), total)
    columns = GROUPS["foot_left"]
    ramp = np.linspace(0.0, 0.13, total, dtype=np.float32)
    points[:, columns] = (points[:, columns]
                          + ramp[:, None, None] * UP[None, None, :]).astype(np.float32)
    found = detect(points, knee, drift)
    assert accepted(found) == []
    assert [p.reject for p in found if p.group == "foot_left"] == ["floor"]


def test_chair_below_the_knees_is_a_seat() -> None:
    """Sit bones held 0.45 m up with the knees at 0.52 m are sitting on something."""
    points, knee, drift = trajectory(0.45, slice(0, 120), 200, group="sit", knee_m=0.52)
    found = accepted(detect(points, knee, drift))
    assert [p.group for p in found] == ["sit"]
    assert found[0].sit_below_knee_m == pytest.approx(-0.07, abs=1e-5)
    assert found[0].sit_below_knee_m < PLATEAU_SIT_BELOW_KNEE_M


def test_standing_sit_bones_are_not_a_seat() -> None:
    """Sit bones still at 0.78 m ride far above the knees: standing, not sitting."""
    points, knee, drift = trajectory(0.78, slice(0, 120), 200, group="sit", knee_m=0.49)
    found = detect(points, knee, drift)
    assert accepted(found) == []
    sit = [p for p in found if p.group == "sit"]
    assert [p.reject for p in sit] == ["above_knees"]
    assert sit[0].sit_below_knee_m == pytest.approx(0.29, abs=1e-5)


def test_squat_bottom_is_not_a_seat() -> None:
    """At the bottom of a squat the sit bones sit level with the knees, so nothing is there."""
    points, knee, drift = trajectory(0.50, slice(0, 120), 200, group="sit", knee_m=0.505)
    found = detect(points, knee, drift)
    assert accepted(found) == []
    assert [p.reject for p in found if p.group == "sit"] == ["above_knees"]
    assert found[-1].sit_below_knee_m == pytest.approx(-0.005, abs=1e-5)


def test_scattered_still_frames_are_not_a_rest() -> None:
    """The same number of still frames, broken into blips, buys no surface."""
    total = 400
    still = np.zeros(total, bool)
    still[::4] = True                       # 100 still frames, never two in a row
    points, knee, drift = trajectory(0.30, slice(0, total), total)
    columns = GROUPS["foot_left"]
    drift[~still[:, None] & np.isin(np.arange(len(FRAME_NAMES))[None, :], columns)] = 0.5
    found = detect(points, knee, drift)
    assert accepted(found) == []
    foot = [p for p in found if p.group == "foot_left"]
    assert [p.reject for p in foot] == ["short"]
    assert foot[0].steps == 100 and foot[0].longest_run == 1


def test_two_surfaces_three_centimetres_apart_stay_apart() -> None:
    """A bridging sample between two close shelves does not weld them into one.

    Single-linkage alone would — 1.5 cm steps chain through the 2 cm gap — so the modes are
    sought instead: the bridge joins the shelf it is nearest and the far one keeps its own.
    """
    total = 200
    points, knee, drift = trajectory(0.30, slice(0, total), total)
    columns = GROUPS["foot_left"]
    lift = np.zeros(total, dtype=np.float32)
    lift[100:] = 0.03                        # second shelf, 3 cm above the first
    lift[99] = 0.015                         # one bridging sample halfway between
    points[:, columns] = (points[:, columns]
                          + lift[:, None, None] * UP[None, None, :]).astype(np.float32)
    found = accepted(detect(points, knee, drift))
    assert len(found) == 2
    heights = sorted(p.height_m for p in found)
    assert heights[0] == pytest.approx(0.30, abs=1e-5)
    assert heights[1] == pytest.approx(0.33, abs=1e-5)
    assert max(p.height_hi_m - p.height_lo_m for p in found) <= PLATEAU_MAX_SPAN_M + 1e-6


def test_a_nine_centimetre_ramp_of_still_heights_is_not_one_surface() -> None:
    """The 8-9 cm spans single-linkage used to accept are split into flat modes."""
    total = 300
    points, knee, drift = trajectory(0.30, slice(0, total), total)
    columns = GROUPS["foot_left"]
    ramp = np.linspace(0.0, 0.09, total, dtype=np.float32)
    points[:, columns] = (points[:, columns]
                          + ramp[:, None, None] * UP[None, None, :]).astype(np.float32)
    found = detect(points, knee, drift)
    foot = [p for p in found if p.group == "foot_left"]
    assert len(foot) > 1
    assert max(p.height_hi_m - p.height_lo_m for p in foot) <= PLATEAU_MAX_SPAN_M + 1e-6


def test_far_in_plane_members_do_not_drag_the_patch_centre() -> None:
    """Still frames a metre away at the same height are a different surface, not this one."""
    total = 300
    points, knee, drift = trajectory(0.30, slice(0, total), total)
    columns = GROUPS["foot_left"]
    axis_u, _ = plane_basis(UP)
    points[200:, columns] += (np.float32(2.0 * PLATEAU_INPLANE_M) * axis_u)[None, None, :]
    found = accepted(detect(points, knee, drift))
    assert len(found) == 1
    near = plane_coordinates(points[:200, columns].reshape(-1, 3), CENTROID)
    assert plane_coordinates(found[0].center[None], CENTROID)[0] == pytest.approx(
        np.median(near, axis=0), abs=1e-5)
    assert found[0].samples == 2 * 200


def test_untracked_frames_carry_no_stillness() -> None:
    """A NaN-posed, untracked frame is not evidence of a surface."""
    points, knee, drift = trajectory(0.30, slice(0, 90), 150)
    valid = np.ones(150, bool)
    valid[:90] = False
    points[:90] = np.nan
    assert accepted(detect(points, knee, drift, valid)) == []


# ── the variant farm and the run_dataset guards ──────────────────────────────────────

def _point_opencap_at(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Make ``--dataset opencap`` read ``root/<clip>/video.mp4`` and write ``root/out/``."""
    monkeypatch.setitem(run_dataset.DATASETS["opencap"], "root", root)
    monkeypatch.setitem(run_dataset.DATASETS["opencap"], "out", root / "out")


def _fake_clip_root(tmp_path: Path) -> Path:
    """A ``processed/`` root with one clip whose out tree carries a real fused scene."""
    clip = tmp_path / "subject3_DJ1_cam1"
    clip.mkdir()
    (clip / "video.mp4").write_bytes(b"")
    out = tmp_path / "out" / clip.name
    (out / "geometry").mkdir(parents=True)
    for name in ("transform.npz", "scale.npz", "human.npz"):
        (out / "geometry" / name).write_bytes(b"")
    (out / "geometry" / "scene.npz").write_bytes(b"the real fused cloud")
    for name in ("sam3", "vggt", "sapiens", "sam3d", "geocalib"):
        (out / name).mkdir()
    return out


def test_floorplane_farm_owns_its_geometry_directory(tmp_path: Path) -> None:
    """The synthetic-scene farm links every geometry artifact but ``scene.npz``, which it owns."""
    clip = _fake_clip_root(tmp_path)
    (clip / "geometry" / "scene.npz").unlink()

    plain = build_farm(clip, "labvalidation")
    assert (plain / "geometry").is_symlink()

    farm = build_farm(clip, "floorplane", synthetic_scene=True)
    assert farm == clip / "_floorplane"
    assert (farm / "human_optim").resolve() == clip / "human_optim_floorplane"
    assert (farm / "vggt").resolve() == clip / "vggt"
    geometry = farm / "geometry"
    assert geometry.is_dir() and not geometry.is_symlink()
    # prefit.npz does not exist on these clips, so nothing links to it.
    assert sorted(p.name for p in geometry.iterdir()) == ["human.npz", "scale.npz",
                                                          "transform.npz"]
    assert (geometry / "transform.npz").resolve() == clip / "geometry" / "transform.npz"

    (geometry / "scene.npz").write_bytes(b"synthetic")
    assert build_farm(clip, "floorplane", synthetic_scene=True) == farm
    assert (geometry / "scene.npz").read_bytes() == b"synthetic"


def test_owns_scene_is_true_only_for_synthetic_scene_variants() -> None:
    """The farm layout follows the chain, never the caller's word for it."""
    assert owns_scene("floorplane")
    assert not owns_scene("labvalidation")
    assert not owns_scene(None)


def test_floorplane_chain_builds_the_scene_around_kinematics_1() -> None:
    """Floor before kinematics_1, plateaus between it and contacts_1; other variants untouched."""
    chain = VARIANT_CHAINS["floorplane"]
    assert chain.index("support_scene_floor") < chain.index("kinematics_1")
    assert chain.index("kinematics_1") < chain.index("support_scene_plateaus")
    assert chain.index("support_scene_plateaus") < chain.index("contacts_1")
    assert [step for step in chain if not step.startswith("support_scene")] == VARIANT_STEPS
    assert VARIANT_CHAINS.get("labvalidation") is None


@pytest.mark.parametrize("argv", [
    ["--steps", "support_scene_floor"],
    ["--variant", "labvalidation", "--steps", "support_scene_floor"],
    ["--variant", "labvalidation", "--steps", "support_scene_plateaus"],
])
def test_support_steps_cannot_touch_a_farm_that_links_the_real_geometry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    """The one destructive mistake: a synthetic scene written over the real fused cloud."""
    out = _fake_clip_root(tmp_path)
    _point_opencap_at(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="geometry"):
        run_dataset.main(["--dataset", "opencap", *argv])
    assert (out / "geometry" / "scene.npz").read_bytes() == b"the real fused cloud"
    assert not (out / "_labvalidation").exists()


@pytest.mark.parametrize("argv, message", [
    (["--variant", "floorplane", "--steps", "fuse"], "cannot run under"),
    (["--variant", "floorplane", "--steps", "gravity"], "cannot run under"),
    (["--variant", "floorplane", "--steps", "contacts_1,kinematics_1"], "in its order"),
    (["--steps", "kindyn_1,fuse"], "in its order"),
    (["--variant", "labvalidation", "--steps", "kinematics_1,kinematics_1"], "in its order"),
])
def test_steps_must_be_this_run_s_chain_in_order(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str], message: str) -> None:
    """A chain is an order: a step that writes shared inputs or runs early is refused."""
    _fake_clip_root(tmp_path)
    _point_opencap_at(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match=message):
        run_dataset.main(["--dataset", "opencap", *argv])


def test_human_chain_refuses_to_run_without_the_synthetic_scene(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scene terms silently off would look like a finished variant and would not be one."""
    out = _fake_clip_root(tmp_path)
    _point_opencap_at(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="scene.npz is missing"):
        run_dataset.main(["--dataset", "opencap", "--variant", "floorplane", "--steps", "kinematics_1"])
    assert not (out / "_floorplane" / "geometry" / "scene.npz").exists()
