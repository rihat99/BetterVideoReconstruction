"""The rules of the contact readout: what makes a label, and what must never make one."""
from __future__ import annotations

import numpy as np
import torch

from bvr.contacts import confidence, detector, labels
from bvr.terms.scene import Scene

NAMES = ("heel_left", "heel_right", "toe_above_left", "toe_above_right", "palm_left", "knee_left", "ball_inner_left",
         "ball_outer_left", "ball_inner_right", "ball_outer_right", "big_toe_left", "big_toe_right")
COLUMN = {name: column for column, name in enumerate(NAMES)}
RULES = {"names": NAMES, "close_thresh_m": 0.03, "exit_thresh_m": 0.035, "min_frames": 3, "still_m": 0.04,
         "still_override_m": 0.01, "max_dist_m": 0.10, "conf_thresh": 0.1}
CONTACTS = {"close_thresh_m": 0.03, "exit_thresh_m": 0.035, "min_span_s": 0.1, "max_dist_m": 0.10, "conf_thresh": 0.1,
            "merge_dist_m": 0.03, "merge_max_gap_s": 2.0, "merge_sdf_m": 0.08}
GATE = {"still_m": 0.04, "still_override_m": 0.01, "detect_min_frames": 3}
FORCE = {"theta_bw": 0.02, "theta_hold_bw": 0.01, "far_still_x": 4.0, "confirm_bw": 0.02}


def signals(signed: np.ndarray, facing: float = 1.0) -> detector.Signals:
    """Contact frames at ``signed (T, F)`` metres in front of a confident surface they face."""
    signed = torch.as_tensor(signed, dtype=torch.float32)
    return detector.Signals(signed, signed.abs(), torch.ones_like(signed), torch.full_like(signed, facing))


def at(frames: int, **distance: float | list[float]) -> np.ndarray:
    """``(T, F)`` distances: 1 m everywhere, and the given value or series in the named columns."""
    table = np.ones((frames, len(NAMES)), np.float32)
    for name, value in distance.items():
        table[:, COLUMN[name]] = value
    return table


def test_scene_signals_measure_the_offset_along_the_scene_normal_and_no_scene_is_no_evidence() -> None:
    grid = torch.cartesian_prod(torch.linspace(-1, 1, 21), torch.linspace(-1, 1, 21))
    floor = Scene(torch.cat((grid, torch.zeros(len(grid), 1)), dim=1), torch.tensor([0.0, 0.0, 1.0]).expand(len(grid), 3),
                  torch.full((len(grid),), 2.0))
    points = torch.tensor([[[0.0, 0.0, 0.02]], [[0.0, 0.0, -0.01]], [[5.0, 5.0, 5.0]]])
    down = torch.tensor([0.0, 0.0, -1.0]).expand(3, 1, 3)
    found = detector.scene_signals(points, down, floor, prune_margin_m=0.1)
    assert torch.allclose(found.signed[:2, 0], torch.tensor([0.02, -0.01]), atol=1e-6)
    assert torch.allclose(found.facing[:2, 0], torch.ones(2)) and found.confidence[0, 0] == 2.0
    assert found.signed[2, 0] == torch.inf and found.nearest[2, 0] == torch.inf
    assert found.confidence[2, 0] == 0.0 and found.facing[2, 0] == -1.0


def test_drift_is_the_net_distance_over_the_window_and_unknown_where_an_end_of_it_is_not_tracked() -> None:
    points = torch.zeros(20, 1, 3)
    points[:, 0, 0] = 0.01 * torch.arange(20)
    tracked = torch.ones(20, dtype=torch.bool)
    tracked[12] = False
    covered = detector.drift(points, tracked, fps=30.0, window_s=0.2)  # 3 frames each way
    assert torch.allclose(covered[3:9, 0], torch.full((6,), 0.06), atol=1e-6)
    assert torch.allclose(covered[0, 0], torch.tensor(0.06), atol=1e-6)  # a window cut by the clip's end is scaled up
    assert covered[:, 0].isnan().nonzero()[:, 0].tolist() == [9, 12, 15]  # the frame itself, or an end of its window


def test_a_frame_touches_where_it_is_near_and_still() -> None:
    still = torch.zeros(10, len(NAMES))
    found = detector.detect(signals(at(10, palm_left=0.02)), still, **RULES)
    assert found.contact[:, COLUMN["palm_left"]].all() and found.contact.sum() == 10
    moving = torch.full((10, len(NAMES)), 0.2)
    assert not detector.detect(signals(at(10, palm_left=0.02)), moving, **RULES).contact.any()
    unknown = torch.full((10, len(NAMES)), torch.nan)
    assert not detector.detect(signals(at(10, palm_left=-0.05)), unknown, **RULES).contact.any()


def test_a_press_outranks_stillness_and_span_length_unless_the_second_witness_denies_it() -> None:
    moving = torch.full((10, len(NAMES)), 0.2)
    pressing = at(10, palm_left=[1, 1, 1, 1, 0.005, 1, 1, 1, 1, 1])
    found = detector.detect(signals(pressing), moving, **RULES)
    assert found.contact[:, COLUMN["palm_left"]].tolist() == [False] * 4 + [True] + [False] * 5
    denied = detector.detect(signals(pressing), moving, **RULES, confirmed=np.zeros((10, len(NAMES)), bool))
    assert not denied.contact.any()
    hovering = at(10, palm_left=[1, 1, 1, 1, 0.02, 1, 1, 1, 1, 1])  # one frame, near but not pressing: flicker
    assert not detector.detect(signals(hovering), torch.zeros(10, len(NAMES)), **RULES).contact.any()


def test_a_span_holds_between_its_two_thresholds_and_a_moving_frame_breaks_it() -> None:
    series = at(8, palm_left=[0.02, 0.033, 0.033, 0.033, 0.04, 0.033, 0.033, 0.033])
    found = detector.detect(signals(series), torch.zeros(8, len(NAMES)), **RULES)
    assert found.contact[:, COLUMN["palm_left"]].tolist() == [True] * 4 + [False] * 4
    drift = torch.zeros(8, len(NAMES))
    drift[3] = 0.2
    broken = detector.detect(signals(at(8, palm_left=0.02)), drift, **RULES)
    assert broken.contact[:, COLUMN["palm_left"]].tolist() == [True] * 3 + [False] + [True] * 4


def test_one_side_of_a_foots_toes_touches_the_one_that_faces_the_surface() -> None:
    both = at(6, ball_inner_left=0.02, toe_above_left=0.02)
    facing = np.full((6, len(NAMES)), 0.2, np.float32)
    facing[:, COLUMN["toe_above_left"]] = 0.9
    found = detector.detect(signals(both)._replace(facing=torch.as_tensor(facing)), torch.zeros(6, len(NAMES)), **RULES)
    assert found.contact[:, COLUMN["toe_above_left"]].all() and not found.contact[:, COLUMN["ball_inner_left"]].any()
    assert found.suppressed[:, COLUMN["ball_inner_left"]].all()
    tie = detector.detect(signals(both), torch.zeros(6, len(NAMES)), **RULES)
    assert tie.contact[:, COLUMN["ball_inner_left"]].all() and not tie.contact[:, COLUMN["toe_above_left"]].any()
    alone = detector.detect(signals(at(6, toe_above_right=0.02), facing=-1.0), torch.zeros(6, len(NAMES)), **RULES)
    assert alone.contact[:, COLUMN["toe_above_right"]].all()  # the normal decides between sides, it gates nothing


def test_span_rules_a_contact_and_a_release_both_have_a_shortest_length() -> None:
    column = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0], bool)[:, None]
    assert detector.runs(column[:, 0]) == [(1, 3), (4, 8), (11, 14)]
    assert detector.min_span(column, 3)[:, 0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    keep = np.zeros_like(column)
    keep[2] = True
    assert detector.min_span(column, 3, keep)[:, 0].tolist() == column[:, 0].tolist()
    closed = detector.close_gaps(column, 3)[:, 0].tolist()
    assert closed == [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]  # the 3-frame release stays, the clip's ends stay


def test_a_gap_is_bridged_only_if_the_frame_stayed_put_and_near() -> None:
    contact = np.array([1, 1, 0, 0, 1, 1], bool)[:, None]
    points, guarded = np.zeros((6, 1, 3)), np.full((6, 1), 0.02)
    rules = {"merge_dist_m": 0.03, "max_gap_frames": 60, "gap_sdf_m": 0.08}
    assert detector.merge_spans(contact, points, guarded, **rules).all()
    left = guarded.copy()
    left[2] = np.inf  # it left the wall in between: a new grip of the same hold
    assert detector.merge_spans(contact, points, left, **rules)[:, 0].tolist() == contact[:, 0].tolist()
    moved = points.copy()
    moved[4:, 0, 0] = 0.1
    assert detector.merge_spans(contact, moved, guarded, **rules)[:, 0].tolist() == contact[:, 0].tolist()


def test_the_loaded_side_of_the_toes_takes_the_whole_contact_and_the_foot_keeps_its_frames() -> None:
    marked = np.zeros((8, len(NAMES)), bool)
    marked[:4, COLUMN["ball_inner_left"]] = True
    marked[4:, COLUMN["toe_above_left"]] = True
    load = np.zeros((8, len(NAMES)))
    load[:, COLUMN["toe_above_left"]] = 0.5
    load[:, COLUMN["ball_inner_left"]] = 0.1
    hooked = labels.one_toe_side(marked, load, NAMES, 3)
    assert hooked[:, COLUMN["toe_above_left"]].all() and not hooked[:, COLUMN["ball_inner_left"]].any()
    load[:, COLUMN["ball_inner_left"]] = 0.9
    standing = labels.one_toe_side(marked, load, NAMES, 3)
    assert standing[:, COLUMN["ball_inner_left"]].all() and not standing[:, COLUMN["toe_above_left"]].any()


def read(load: np.ndarray, far: float, drift: float, signed: np.ndarray | None = None) -> labels.Labels:
    frames = len(load)
    signed = at(frames) if signed is None else signed
    return labels.read(
        signals(signed), torch.full((frames, len(NAMES)), drift), torch.zeros(frames, len(NAMES), 3), load,
        np.full((frames, len(NAMES)), far), np.zeros(frames, bool), np.ones(frames, bool), names=NAMES, fps=30.0,
        contacts=CONTACTS, gate=GATE, force=FORCE)


def test_load_labels_a_frame_wherever_it_sits_if_it_is_still_or_near() -> None:
    load = np.zeros((10, len(NAMES)))
    load[2:8, COLUMN["palm_left"]] = 0.3
    far_and_still = read(load, far=0.5, drift=0.02)
    assert far_and_still.contact[:, COLUMN["palm_left"]].tolist() == [False] * 2 + [True] * 6 + [False] * 2
    assert far_and_still.loaded.sum() == 6 and not far_and_still.geometric.any()
    assert not read(load, far=0.5, drift=0.5).contact.any()         # far and moving: scraps on a flying limb
    assert read(load, far=0.05, drift=0.5).contact.sum() == 6       # near: load alone labels
    assert not read(load, far=np.nan, drift=np.nan).contact.any()   # nothing measured reads as far and moving


def test_load_never_labels_a_heel_or_a_knee_and_geometry_never_a_knee() -> None:
    load = np.zeros((10, len(NAMES)))
    load[:, [COLUMN["heel_left"], COLUMN["knee_left"]]] = 0.5
    assert not read(load, far=0.05, drift=0.0).contact.any()
    touching = read(np.zeros((10, len(NAMES))), far=0.05, drift=0.0, signed=at(10, heel_left=0.02, knee_left=0.02))
    assert touching.contact[:, COLUMN["heel_left"]].all() and not touching.contact[:, COLUMN["knee_left"]].any()


def test_a_hand_frame_reports_at_its_wrist_and_a_joint_is_as_still_as_its_stillest_frame() -> None:
    names = ("pelvis", "left_wrist", "left_index1", "left_index2", "left_knee")
    report = confidence.report_joints((3, 1, 4, 0), names, (-1, 0, 1, 2, 0))
    assert report.tolist() == [1, 1, 4, 0]
    contact = np.array([[True, False, False, False], [False, False, True, False]])
    assert confidence.fold_contacts(contact, report, 5).tolist() == [[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]]
    drift = np.array([[0.05, 0.01, np.nan, np.nan]], np.float32)
    folded = confidence.fold_drift(drift, report, 5)[0]
    assert folded[1] == np.float32(0.01) and np.isnan(folded[[0, 2, 3, 4]]).all()


def test_seconds_to_a_label_change_ignore_tracking_gaps() -> None:
    contact = np.array([0, 0, 1, 1, 1, 0, 0, 0], bool)[:, None]
    tracked = np.array([1, 1, 1, 1, 1, 1, 1, 1], bool)
    assert np.allclose(confidence.transition_seconds(contact, tracked, 10.0)[:, 0], [0.1, 0, 0, 0.1, 0, 0, 0.1, 0.2])
    tracked[5] = False  # the release happens inside a gap: it is no change anyone saw
    assert np.allclose(confidence.transition_seconds(contact, tracked, 10.0)[:, 0], [0.1, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5])


def test_drift_makes_a_contact_label_less_sure_and_a_free_label_more_sure() -> None:
    coefficients = {"drift_floor_m": 0.005, "transition_cap_s": 1.0,
                    "contact": {"intercept": 0.0, "log_drift": -0.8, "log_transition": 0.5, "hand": -0.3},
                    "free": {"intercept": 0.0, "log_drift": 1.0, "log_transition": 0.6, "hand": 0.8}}
    still, moved = confidence.label_confidence(
        np.array([[True, True]] * 4), np.array([[0.01, 0.10]] * 4, np.float32), np.ones(4, bool), 30.0,
        ("left_wrist", "right_wrist"), coefficients)[0]
    assert still > moved
    free = confidence.label_confidence(
        np.array([[False, False, True]] * 4), np.array([[0.01, 0.10, np.nan]] * 4, np.float32),
        np.array([True, True, True, False]), 30.0, ("left_wrist", "right_wrist", "left_elbow"), coefficients)
    assert free[0, 0] < free[0, 1] and np.isnan(free[:, 2]).all() and np.isnan(free[3]).all()


def test_force_confidence_falls_with_the_residual_and_with_implausible_load() -> None:
    root, forces = np.zeros((1, 4, 6), np.float32), np.zeros((1, 4, 2, 3), np.float32)
    root[0, 1, 0] = 350.0         # half a body weight of unexplained force: the confidence halves
    forces[0, 2, :, 2] = 1400.0   # four body weights of support: one above the cap
    trust = confidence.force_confidence(root, forces, np.array([[True, True, True, False]]), np.array([700.0]))
    assert np.allclose(trust[0], [1.0, 0.5, 0.5, 0.0])
