"""The force witness: cone feasibility, the readout rules, and the no-scene policy."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tools.human_optim.forces import (
    NO_SCENE_M, ForceKnobs, contact_axes, force_labels, min_span, resolve_toe_conflicts,
)
from tools.human_optim.friction import cone_generators

_TOE_NAMES = ("toe_under_left", "toe_above_left", "toe_under_right", "toe_above_right")


def _axis_batch() -> torch.Tensor:
    """A few unit axes pointing in awkward directions, ``(1, 5, 3)`` float32."""
    axes = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                         [0.6, 0.8, 0.0], [-0.5774, 0.5774, 0.5774]], dtype=torch.float32)
    return (axes / axes.norm(dim=-1, keepdim=True))[None]


def test_cone_generators_stay_inside_the_friction_cone() -> None:
    """Every nonnegative mix of the four pyramid edges obeys ‖f_tangent‖ ≤ μ·f_normal."""
    axes = _axis_batch()
    mu = math.tan(math.radians(45.0))
    generators = cone_generators(axes, mu)                       # (1, 5, 4, 3)
    assert generators.shape == (1, 5, 4, 3)
    torch.testing.assert_close(generators.norm(dim=-1), torch.ones(1, 5, 4))

    generator = torch.Generator().manual_seed(0)
    coefficients = torch.rand((200, 1, 5, 4), generator=generator, dtype=torch.float32)
    force = torch.einsum("btfg,tfgd->btfd", coefficients, generators)
    normal = (force * axes).sum(-1)
    tangential = (force - normal[..., None] * axes).norm(dim=-1)
    assert torch.all(normal >= -1e-6), "a cone force must never pull"
    assert torch.all(tangential <= mu * normal + 1e-5), "a cone force must never slip"


def test_zero_coefficients_give_exactly_zero_force() -> None:
    """The solver's start point is attainable: λ = 0 means no force at all.

    This is the property the earlier softplus parameterization did not have — it could
    only approach zero, so every frame carried a little phantom load.
    """
    force = torch.einsum("tfg,tfgd->tfd", torch.zeros(1, 5, 4), cone_generators(_axis_batch(), 1.0))
    assert torch.count_nonzero(force) == 0


def test_frame_axis_separates_the_two_toe_frames() -> None:
    """``"frame"`` axes are the negated skin normal, so a toe pair gets opposite cones."""
    points = torch.zeros(1, 2, 3)
    normals = torch.tensor([[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]])     # under faces down, above up
    axes, valid = contact_axes(points, normals, torch.zeros(0, 3), torch.zeros(0, 3),
                               mode="frame", k=4, max_dist_m=1.0)
    assert bool(valid.all()), "a frame axis never needs the scene, so it is always usable"
    torch.testing.assert_close(axes[0, 0], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(axes[0, 1], torch.tensor([0.0, -1.0, 0.0]))
    assert float((axes[0, 0] * axes[0, 1]).sum()) == pytest.approx(-1.0)


def test_toe_winner_take_all_keeps_the_stronger_toe() -> None:
    """At most one toe frame per foot survives; exact ties keep ``toe_under``."""
    magnitude = np.tile(np.array([[80.0, 100.0, 0.0, 0.0]], dtype=np.float32), (5, 1))
    labels, split = force_labels(magnitude, frame_names=_TOE_NAMES, valid=np.ones(5, bool),
                                 threshold_n=30.0, min_frames=3)
    assert labels[0].tolist() == [False, True, False, False]
    assert split[0].tolist() == [True, True, False, False], "both fired, so the pair was split"

    tied = np.tile(np.array([[100.0, 100.0, 0.0, 0.0]], dtype=np.float32), (5, 1))
    tie_labels, _ = force_labels(tied, frame_names=_TOE_NAMES, valid=np.ones(5, bool),
                                 threshold_n=30.0, min_frames=3)
    assert tie_labels[0].tolist() == [True, False, False, False]


def test_threshold_and_min_span_drop_weak_and_brief_load() -> None:
    """Below-threshold load never fires, and a run shorter than ``min_frames`` is dropped."""
    magnitude = np.zeros((6, 4), dtype=np.float32)
    magnitude[0:2, 0] = 100.0        # two frames only — too brief
    magnitude[3:6, 2] = 100.0        # three frames — survives
    magnitude[3:6, 3] = 10.0         # below threshold
    labels, _ = force_labels(magnitude, frame_names=_TOE_NAMES, valid=np.ones(6, bool),
                             threshold_n=30.0, min_frames=3)
    assert labels[:, 0].tolist() == [False] * 6
    assert labels[:, 2].tolist() == [False, False, False, True, True, True]
    assert labels[:, 3].tolist() == [False] * 6


def test_invalid_frames_can_never_be_labelled() -> None:
    """A step where the person is not tracked carries no contact, whatever the force says."""
    magnitude = np.full((6, 4), 500.0, dtype=np.float32)
    valid = np.array([True, True, True, False, False, False])
    labels, _ = force_labels(magnitude, frame_names=_TOE_NAMES, valid=valid,
                             threshold_n=30.0, min_frames=3)
    assert not labels[3:].any()


def test_min_span_is_per_column_and_leaves_long_runs_alone() -> None:
    mask = np.array([[True, True], [True, False], [False, True], [True, True], [True, True]])
    kept = min_span(mask, 2)
    assert kept[:, 0].tolist() == [True, True, False, True, True]
    assert kept[:, 1].tolist() == [False, False, True, True, True]


def test_union_of_the_two_readouts_never_holds_both_toes() -> None:
    """The regression this exists for: geometry keeps one toe, force keeps the other.

    Each readout on its own already keeps a single toe per foot, so neither catches this —
    only the union does, and it must resolve it rather than hold both.
    """
    steps = 8
    geometric = np.zeros((steps, 4), bool)
    geometric[:, 0] = True                       # detector says toe_under_left, all steps
    magnitude = np.zeros((steps, 4), dtype=np.float32)
    magnitude[2:7, 1] = 400.0                    # force says toe_above_left, overlapping
    force, _ = force_labels(magnitude, frame_names=_TOE_NAMES, valid=np.ones(steps, bool),
                            threshold_n=30.0, min_frames=3)
    union = geometric | force
    assert (union[:, 0] & union[:, 1]).sum() == 5, "the union really does hold both toes"

    resolved = resolve_toe_conflicts(union, magnitude, frame_names=_TOE_NAMES, min_frames=3)
    assert (resolved[:, 0] & resolved[:, 1]).sum() == 0, "at most one toe per foot per step"
    # Force carried the load on steps 2..6, so it wins those; the detector keeps steps 0-1
    # and 7, but neither remnant is 3 frames long, so the minimum-duration pass drops them.
    assert resolved[:, 1].tolist() == [False, False, True, True, True, True, True, False]
    assert not resolved[:, 0].any()


def test_toe_conflicts_keep_a_geometric_label_the_force_does_not_contest() -> None:
    """Exclusivity only fires on a genuine conflict — it is not a blanket force override."""
    steps = 6
    labels = np.zeros((steps, 4), bool)
    labels[:, 0] = True                          # toe_under_left only; nothing contests it
    resolved = resolve_toe_conflicts(labels, np.zeros((steps, 4), dtype=np.float32),
                                     frame_names=_TOE_NAMES, min_frames=3)
    assert resolved[:, 0].all()
    assert not resolved[:, 1].any()


def test_toe_conflict_tie_keeps_toe_under() -> None:
    """Equal force on both toes is resolved deterministically, never by column order."""
    labels = np.ones((5, 4), bool)
    resolved = resolve_toe_conflicts(labels, np.full((5, 4), 100.0, dtype=np.float32),
                                     frame_names=_TOE_NAMES, min_frames=3)
    assert resolved[:, 0].all() and not resolved[:, 1].any()
    assert resolved[:, 2].all() and not resolved[:, 3].any()


def test_no_scene_sentinel_is_not_a_distance() -> None:
    """``frame_scene_signals`` reports ``finfo.max`` for "no scene point near"; the force
    path must recognise that as missing evidence rather than as a very large gap."""
    sentinel = torch.finfo(torch.float32).max
    distance = torch.tensor([[0.02, sentinel, 0.4]])
    known = torch.isfinite(distance) & (distance < NO_SCENE_M)
    assert known.tolist() == [[True, False, True]]
    assert sentinel > NO_SCENE_M
    # Relative pricing must take its datum from a known frame, never from the sentinel.
    datum = torch.where(known, distance, torch.full_like(distance, float("inf"))).amin(dim=1)
    assert float(datum[0]) == pytest.approx(0.02)


def test_default_knobs_are_the_documented_recipe() -> None:
    """The shipped defaults are the measured recipe, not whatever was last tried."""
    knobs = ForceKnobs()
    assert knobs.cone_axis == "frame"
    assert knobs.relative_gap is True
    assert math.isinf(knobs.candidate_max_dist_m)
    assert knobs.half_angle_deg == pytest.approx(45.0)
