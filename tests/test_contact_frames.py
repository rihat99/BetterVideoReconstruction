"""Contact-frame detection: the surface guards, the stillness gate, the foot-side tiebreak."""

from __future__ import annotations

import math

import pytest
import torch
from better_robot.lie import so3

from bvr.body import frame_normals
from bvr.human_optim.contacts import (
    detect_frame_contacts, frame_guarded_sdf, frame_scene_signals, joint_drift,
    merge_contact_spans, toe_side_slots)

# The legacy 12-frame climbing set in its stored order — note slots 10/11 are RIGHT then LEFT.
_FRAME_NAMES = ("toe_under_left", "toe_under_right", "heel_left", "heel_right",
                "toe_above_left", "toe_above_right", "knee_left", "knee_right",
                "palm_left", "palm_right", "finger_right", "finger_left")

_UP = torch.tensor([0.0, 0.0, 1.0])
_DOWN = torch.tensor([0.0, 0.0, -1.0])


def _plane(half_extent: float = 0.5, step: float = 0.05, conf: float = 1.0):
    """A horizontal plane of scene points at ``z = 0`` whose normals all point UP."""
    axis = torch.arange(-half_extent, half_extent + 1e-6, step, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1),
                          torch.zeros(grid_x.numel())), dim=-1)
    normals = _UP.expand_as(points).clone()
    return points, normals, torch.full((points.shape[0],), conf)


def _hover(num_steps: int, height: float, normal: torch.Tensor):
    """One contact frame hovering ``height`` above the plane origin, facing ``normal``."""
    points = torch.tensor([0.0, 0.0, height]).expand(num_steps, 1, 3).clone()
    return points, normal.expand(num_steps, 1, 3).clone()


def _detect(points, normals, drift, scene, *, frame_names=None, **overrides):
    """Run the detector on ``points``; unnamed columns get non-toe names (no tiebreak)."""
    knobs = dict(close_thresh_m=0.03, exit_thresh_m=0.035, min_frames=3, still_m=0.03,
                 max_dist_m=0.10, conf_thresh=0.1, prune_margin_m=0.15)
    knobs.update(overrides)
    names = frame_names if frame_names is not None else [
        f"palm_{i}" for i in range(points.shape[1])]
    return detect_frame_contacts(points, normals, drift, *scene, frame_names=names, **knobs)


# ── the signals ──────────────────────────────────────────────────────────────

def test_scene_signals_report_height_and_facing():
    scene = _plane()
    points, normals = _hover(4, 0.01, _DOWN)
    sdf, dmin, conf, normal_dot = frame_scene_signals(points, normals, *scene,
                                                      prune_margin_m=0.15)
    # 1 cm above a plane whose normal is +z: the perpendicular signed distance is +1 cm.
    assert torch.allclose(sdf, torch.full_like(sdf, 0.01), atol=1e-6)
    assert torch.allclose(dmin, torch.full_like(dmin, 0.01), atol=1e-6)
    assert torch.allclose(conf, torch.ones_like(conf))
    # skin faces straight down into the plane ⇒ dot(n_frame, −n_scene) = +1
    assert torch.allclose(normal_dot, torch.ones_like(normal_dot), atol=1e-6)
    # flipping the frame normal flips the facing signal, nothing else
    _, _, _, flipped = frame_scene_signals(points, -normals, *scene, prune_margin_m=0.15)
    assert torch.allclose(flipped, -torch.ones_like(flipped), atol=1e-6)


def test_scene_signals_empty_box_is_no_evidence():
    scene = _plane()
    points, normals = _hover(3, 5.0, _DOWN)          # far above every scene point
    sdf, dmin, conf, normal_dot = frame_scene_signals(points, normals, *scene,
                                                      prune_margin_m=0.15)
    assert torch.isinf(sdf).all() or (sdf > 1e30).all()
    assert (dmin > 1e30).all()
    assert (conf == 0.0).all()
    assert (normal_dot == -1.0).all()


def test_guards_force_infinity_not_masking():
    scene = _plane()
    points, normals = _hover(2, 0.01, _DOWN)
    sdf, dmin, conf, normal_dot = frame_scene_signals(points, normals, *scene,
                                                      prune_margin_m=0.15)
    guarded = frame_guarded_sdf(sdf, dmin, conf, max_dist_m=0.10, conf_thresh=0.1)
    assert torch.allclose(guarded, sdf)
    # each guard on its own must push the signal to +inf-ish
    for kwargs in (dict(max_dist_m=0.005), dict(conf_thresh=2.0)):
        knobs = dict(max_dist_m=0.10, conf_thresh=0.1)
        knobs.update(kwargs)
        assert (frame_guarded_sdf(sdf, dmin, conf, **knobs) > 1e30).all(), kwargs


# ── the normal is NOT a gate ─────────────────────────────────────────────────

def test_a_still_frame_at_the_surface_fires():
    scene = _plane()
    points, normals = _hover(10, 0.01, _DOWN)
    drift = torch.zeros(10, 1)
    assert _detect(points, normals, drift, scene).contact.all()


def test_a_frame_facing_away_from_the_scene_still_fires():
    """The normal is no longer a gate: same point, flipped skin, same label.

    Reconstructed scene normals and the fitted pose are both too noisy to veto on — the
    old normal window threw away real contacts (fingertips on holds, mostly).
    """
    scene = _plane()
    points, normals = _hover(10, 0.01, _UP)
    drift = torch.zeros(10, 1)
    assert _detect(points, normals, drift, scene).contact.all()


# ── the toe pair: the one place a normal still decides ───────────────────────

def _toe_pair(num_steps: int, height: float, *, under_normal, above_normal):
    """A left toe_under / toe_above pair at the same place, ``height`` above the plane."""
    points = torch.tensor([0.0, 0.0, height]).expand(num_steps, 2, 3).clone()
    normals = torch.stack((under_normal.expand(num_steps, 3),
                           above_normal.expand(num_steps, 3)), dim=1)
    return points, normals, ["toe_under_left", "toe_above_left"]


def test_toe_pair_keeps_the_frame_looking_into_the_wall():
    """Both toes are millimetres from the same hold; only the one facing it may fire."""
    scene = _plane()
    points, normals, names = _toe_pair(10, 0.01, under_normal=_DOWN, above_normal=_UP)
    contact = _detect(points, normals, torch.zeros(10, 2), scene, frame_names=names).contact
    assert contact[:, 0].all(), "toe_under faces the plane, so it wins"
    assert not contact[:, 1].any()
    # flip which one faces the plane and the winner flips with it
    points, normals, names = _toe_pair(10, 0.01, under_normal=_UP, above_normal=_DOWN)
    contact = _detect(points, normals, torch.zeros(10, 2), scene, frame_names=names).contact
    assert contact[:, 1].all() and not contact[:, 0].any()


def test_toe_pair_tie_is_deterministic_and_keeps_toe_under():
    """Identical normals ⇒ identical evidence; the rule must still pick exactly one."""
    scene = _plane()
    points, normals, names = _toe_pair(10, 0.01, under_normal=_DOWN, above_normal=_DOWN)
    contact = _detect(points, normals, torch.zeros(10, 2), scene, frame_names=names).contact
    assert contact[:, 0].all() and not contact[:, 1].any()


def test_toe_pair_leaves_a_lone_candidate_alone():
    """Only one toe near the wall ⇒ nothing to disambiguate, whichever way it looks."""
    scene = _plane()
    points = torch.zeros(10, 2, 3)
    points[:, 0, 2] = 0.01                      # toe_under at the surface
    points[:, 1, 2] = 0.40                      # toe_above far away
    normals = torch.stack((_UP.expand(10, 3), _DOWN.expand(10, 3)), dim=1)
    detected = _detect(points, normals, torch.zeros(10, 2), scene,
                       frame_names=["toe_under_left", "toe_above_left"])
    assert detected.contact[:, 0].all()
    assert not detected.toe_suppressed.any()


def test_toe_pairs_of_one_foot_are_never_both_in_contact():
    """The output invariant, on noisy poses: at most ONE toe per foot at every step."""
    torch.manual_seed(0)
    scene = _plane()
    num_steps, num_frames = 60, len(_FRAME_NAMES)
    points = torch.zeros(num_steps, num_frames, 3)
    points[..., :2] = torch.empty(num_steps, num_frames, 2).uniform_(-0.2, 0.2)
    points[..., 2] = torch.empty(num_steps, num_frames).uniform_(-0.01, 0.05)
    normals = torch.randn(num_steps, num_frames, 3)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    detected = _detect(points, normals, torch.zeros(num_steps, num_frames), scene,
                       frame_names=list(_FRAME_NAMES), min_frames=1)
    assert detected.toe_suppressed.any(), "the stress case must actually exercise the rule"
    for sole, top in toe_side_slots(_FRAME_NAMES):
        sole_on = detected.contact[:, list(sole)].any(dim=1)
        top_on = detected.contact[:, list(top)].any(dim=1)
        assert not (sole_on & top_on).any()


def test_span_merging_cannot_resurrect_a_suppressed_toe():
    """The readout's span merge works one frame at a time, so it could undo the tiebreak.

    ``estimate_contacts`` bridges short gaps AFTER detection. Left alone it would bridge
    ``toe_under`` straight across the steps where ``toe_above`` won the pair — putting both
    toes of one foot back in contact at once. Masking the merge with ``toe_suppressed``
    keeps the verdict final; this pins that composition.
    """
    scene = _plane()
    num_steps = 15
    points = torch.tensor([0.0, 0.0, 0.01]).expand(num_steps, 2, 3).clone()
    # Both toes sit at the surface the whole time; only which one faces it changes, so the
    # tiebreak hands steps 5-7 to toe_above and every other step to toe_under.
    normals = torch.zeros(num_steps, 2, 3)
    normals[:, 0] = _DOWN                       # toe_under looks into the plane …
    normals[:, 1] = _UP
    normals[5:8, 0] = _UP                       # … except mid-clip, where they swap
    normals[5:8, 1] = _DOWN
    names = ["toe_under_left", "toe_above_left"]
    detected = _detect(points, normals, torch.zeros(num_steps, 2), scene,
                       frame_names=names, min_frames=3)
    under, above = 0, 1
    assert detected.contact[:5, under].all() and detected.contact[8:, under].all()
    assert detected.contact[5:8, above].all()
    assert not (detected.contact[:, under] & detected.contact[:, above]).any()

    bridged = merge_contact_spans(detected.contact, points, detected.guarded_sdf,
                                  merge_dist_m=0.03, max_gap_s=2.0, gap_sdf_m=0.08, fps=30.0)
    assert (bridged[5:8, under] & bridged[5:8, above]).all(), "the merge alone re-creates both-fire"
    kept = bridged & ~detected.toe_suppressed
    assert not (kept[:, under] & kept[:, above]).any()


def test_toe_side_slots_resolve_by_name_not_position():
    """Slots 10/11 are finger_RIGHT then finger_LEFT — positional arithmetic would misfire.

    On the legacy 12-frame set the underside group resolves to (toe_under, heel) per foot."""
    assert toe_side_slots(_FRAME_NAMES) == (((0, 2), (4,)), ((1, 3), (5,)))
    shuffled = tuple(reversed(_FRAME_NAMES))
    assert toe_side_slots(shuffled) == (((11, 9), (7,)), ((10, 8), (6,)))
    assert toe_side_slots(["palm_left", "knee_left"]) == ()


# ── stillness ────────────────────────────────────────────────────────────────

def test_stillness_gate_kills_a_moving_frame():
    scene = _plane()
    points, normals = _hover(10, 0.01, _DOWN)
    still = torch.zeros(10, 1)
    moving = torch.full((10, 1), 0.07)              # well past 2 · still_m = 6 cm
    assert _detect(points, normals, still, scene).contact.all()
    assert not _detect(points, normals, moving, scene).contact.any()


def test_stillness_gate_breaks_a_span_in_the_middle():
    scene = _plane()
    points, normals = _hover(12, 0.01, _DOWN)
    drift = torch.zeros(12, 1)
    drift[5:7] = 0.20                                # a two-frame lunge through the hold
    contact = _detect(points, normals, drift, scene, min_frames=1).contact[:, 0]
    assert contact[:5].all() and contact[7:].all()
    assert not contact[5:7].any()


def test_nan_drift_cannot_hold_a_contact():
    """An untracked step has no motion witness, so it must not be labelled."""
    scene = _plane()
    points, normals = _hover(10, 0.01, _DOWN)
    drift = torch.zeros(10, 1)
    drift[4:8] = float("nan")
    contact = _detect(points, normals, drift, scene, min_frames=1).contact[:, 0]
    assert not contact[4:8].any()


def test_still_override_lets_a_pressing_frame_through_the_stillness_gate():
    """A fast foot plant presses into the floor — closeness outranks stillness."""
    scene = _plane()
    pressing, normals = _hover(10, -0.005, _DOWN)     # 5 mm INSIDE the floor
    hovering, _ = _hover(10, 0.02, _DOWN)             # near, but not pressing
    moving = torch.full((10, 1), 0.50)                # a sprinting foot
    assert not _detect(pressing, normals, moving, scene).contact.any()   # default: off
    on = dict(still_override_m=0.01)
    assert _detect(pressing, normals, moving, scene, **on).contact.all()
    assert not _detect(hovering, normals, moving, scene, **on).contact.any()


def test_still_override_never_passes_a_nan_drift_step():
    """The override needs a real motion measurement — untracked steps stay out."""
    scene = _plane()
    points, normals = _hover(10, -0.005, _DOWN)
    drift = torch.full((10, 1), float("nan"))
    assert not _detect(points, normals, drift, scene,
                       still_override_m=0.01).contact.any()


# ── the load check on the penetration exception ──────────────────────────────

def test_still_frames_never_need_the_solves_blessing():
    """Near and still IS the geometric contact — zero solved load changes nothing."""
    scene = _plane()
    points, normals = _hover(10, 0.01, _DOWN)
    still = torch.zeros(10, 1)
    weightless = torch.zeros(10, 1)
    assert _detect(points, normals, still, scene,
                   load_bw=weightless, min_load_bw=0.02).contact.all()
    # …and a zero bar keeps every press (the plain still_override_m behaviour).
    assert _detect(points, normals, still, scene,
                   load_bw=weightless, min_load_bw=0.0).contact.all()


def test_load_confirms_a_pressing_but_moving_frame():
    """The press exception with the solve heard: a plant carries weight, a sweep none."""
    scene = _plane()
    pressing, normals = _hover(10, -0.005, _DOWN)     # 5 mm INSIDE the floor
    moving = torch.full((10, 1), 0.50)                # a sprinting foot
    carrying = torch.full((10, 1), 0.30)              # 30 % of body weight
    weightless = torch.zeros(10, 1)
    on = dict(still_override_m=0.01, min_load_bw=0.02)
    assert _detect(pressing, normals, moving, scene, load_bw=carrying, **on).contact.all()
    assert not _detect(pressing, normals, moving, scene,
                       load_bw=weightless, **on).contact.any()


def test_load_alone_is_not_an_override():
    """A moving frame merely hovering near the wall is not rescued by its load — the
    force witness labels those through its own thresholds, never through this gate
    (a load override was measured 2026-08-30 and found fully redundant with theta_bw)."""
    scene = _plane()
    hovering, normals = _hover(10, 0.02, _DOWN)       # near, but not pressing
    moving = torch.full((10, 1), 0.50)
    carrying = torch.full((10, 1), 0.30)
    assert not _detect(hovering, normals, moving, scene, load_bw=carrying,
                       still_override_m=0.01, min_load_bw=0.02).contact.any()


def test_a_muted_step_keeps_its_press_exception():
    """Where the solve explained nothing its silence is not evidence — the press stands."""
    scene = _plane()
    pressing, normals = _hover(10, -0.005, _DOWN)
    moving = torch.full((10, 1), 0.50)
    weightless = torch.zeros(10, 1)
    mute = torch.zeros(10, dtype=torch.bool)
    mute[3:8] = True
    contact = _detect(pressing, normals, moving, scene, load_bw=weightless,
                      still_override_m=0.01, min_load_bw=0.02, load_mute=mute,
                      min_frames=1).contact[:, 0]
    assert contact[3:8].all()
    assert not contact[:3].any() and not contact[8:].any()


# ── hysteresis + minimum duration ────────────────────────────────────────────

def test_hysteresis_holds_a_span_between_enter_and_exit():
    scene = _plane()
    heights = torch.tensor([0.010] * 4 + [0.032] * 4 + [0.010] * 4)   # dips into enter..exit
    points = torch.zeros(12, 1, 3)
    points[:, 0, 2] = heights
    normals = _DOWN.expand(12, 1, 3).clone()
    contact = _detect(points, normals, torch.zeros(12, 1), scene).contact[:, 0]
    assert contact.all(), "3.2 cm is above enter (3 cm) but below exit (3.5 cm)"


def test_hysteresis_ends_a_span_past_the_exit_threshold():
    scene = _plane()
    points = torch.zeros(12, 1, 3)
    points[:, 0, 2] = torch.tensor([0.010] * 4 + [0.050] * 4 + [0.010] * 4)
    normals = _DOWN.expand(12, 1, 3).clone()
    contact = _detect(points, normals, torch.zeros(12, 1), scene).contact[:, 0]
    assert contact[:4].all() and contact[8:].all()
    assert not contact[4:8].any()


def test_min_frames_drops_a_flicker_span():
    scene = _plane()
    points = torch.full((12, 1, 3), 0.0)
    points[:, 0, 2] = 0.50                            # far above the plane all clip long
    points[5:7, 0, 2] = 0.010                         # …except a two-frame blip
    normals = _DOWN.expand(12, 1, 3).clone()
    drift = torch.zeros(12, 1)
    assert not _detect(points, normals, drift, scene, min_frames=3).contact.any()
    assert _detect(points, normals, drift, scene, min_frames=2).contact[5:7, 0].all()


def test_penetration_exempts_a_one_frame_span_from_the_blip_killer():
    """A one-frame plant pressing into the floor is evidence, not flicker."""
    scene = _plane()
    points = torch.full((12, 1, 3), 0.0)
    points[:, 0, 2] = 0.50                            # airborne all clip long
    points[6, 0, 2] = -0.005                          # …except one frame 5 mm INSIDE
    normals = _DOWN.expand(12, 1, 3).clone()
    drift = torch.full((12, 1), 0.50)                 # sprinting throughout
    assert not _detect(points, normals, drift, scene).contact.any()   # default: off
    contact = _detect(points, normals, drift, scene, still_override_m=0.01).contact
    assert contact[6, 0] and contact.sum() == 1


def test_a_one_frame_hover_is_still_flicker_even_with_the_override():
    """The exemption needs pressing — a brief near pass without penetration stays out."""
    scene = _plane()
    points = torch.full((12, 1, 3), 0.0)
    points[:, 0, 2] = 0.50
    points[6, 0, 2] = 0.020                           # near (2 cm), never pressing
    normals = _DOWN.expand(12, 1, 3).clone()
    assert not _detect(points, normals, torch.zeros(12, 1), scene,
                       still_override_m=0.01).contact.any()


def test_drift_is_measured_the_same_way_on_frames_as_on_joints():
    """The detector is K-generic: contact frames use the same drift as 52 joints."""
    positions = torch.zeros(9, 2, 3)
    positions[:, 0, 0] = torch.linspace(0.0, 0.8, 9)     # a sweeping point
    drift = joint_drift(positions, fps=30.0, window_s=0.2)
    assert drift.shape == (9, 2)
    assert (drift[:, 0] > 0.0).all()
    assert torch.allclose(drift[:, 1], torch.zeros(9))


# ── the frame_normals helper ─────────────────────────────────────────────────

def test_frame_normals_is_the_third_rotation_column():
    torch.manual_seed(0)
    quaternions = torch.randn(4, 12, 4)
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    frames7 = torch.cat((torch.randn(4, 12, 3), quaternions), dim=-1)
    normals = frame_normals(frames7)
    assert normals.shape == (4, 12, 3)
    assert torch.allclose(normals, so3.to_matrix(quaternions)[..., :, 2], atol=1e-6)
    assert torch.allclose(normals.norm(dim=-1), torch.ones(4, 12), atol=1e-5)


@pytest.mark.parametrize("angle_deg, expected", [(0.0, 1.0), (180.0, -1.0), (90.0, 0.0)])
def test_frame_normals_follows_a_rotation_about_x(angle_deg: float, expected: float):
    """Rotating a frame about x tips its baked z normal by the same angle."""
    angle = math.radians(angle_deg)
    quaternion = torch.tensor([[math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)]])
    frames7 = torch.cat((torch.zeros(1, 3), quaternion), dim=-1)[None]
    assert frame_normals(frames7)[0, 0, 2] == pytest.approx(expected, abs=1e-6)


# ── the floor gate ───────────────────────────────────────────────────────────

def test_floor_gate_blocks_a_foot_frame_above_the_plane_and_nothing_else():
    """A foot frame above ``floor_gate_m`` is refused; a hand frame at the same height is not."""
    from bvr.stages.estimate_contacts import floor_gate

    centroid = torch.tensor([1.0, 2.0, 3.0])          # the plane need not pass the origin
    up = torch.tensor([0.0, 0.0, 1.0])
    # two frames, three video frames: column 0 is a foot, column 1 a hand.
    heights = torch.tensor([0.02, 0.10, 0.40])
    points = centroid + torch.stack((heights, heights), dim=1)[..., None] * up
    height, allowed = floor_gate(points, centroid, up, foot_columns=[0], gate_m=0.17)

    assert torch.allclose(height[:, 0], heights, atol=1e-6)
    assert allowed[:, 0].tolist() == [True, True, False]    # 0.40 m is not standing on the floor
    assert allowed[:, 1].all()                              # the hand column is never gated


def test_floor_gate_measures_height_along_up_not_along_z():
    """The plane can be tilted: height is the signed distance along ``up_world``."""
    from bvr.stages.estimate_contacts import floor_gate

    up = torch.tensor([0.0, math.sin(math.radians(30.0)), math.cos(math.radians(30.0))])
    centroid = torch.zeros(3)
    points = (0.5 * up).expand(1, 1, 3)               # half a metre up the tilted normal
    height, allowed = floor_gate(points, centroid, up, foot_columns=[0], gate_m=0.17)
    assert height[0, 0] == pytest.approx(0.5, abs=1e-6)
    assert not allowed[0, 0]
