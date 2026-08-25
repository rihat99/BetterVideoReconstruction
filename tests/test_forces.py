"""The force witness: cone feasibility, the readout rules, and the no-scene policy."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tools.human_optim.forces import (
    NO_SCENE_M, ForceKnobs, close_gaps, far_still_veto, force_labels, hysteresis, min_span,
    resolve_toe_conflicts,
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


def test_opposite_skin_normals_give_opposite_cones() -> None:
    """The cone axis is the negated skin normal, so a toe pair can never push the same way.

    ``toe_under`` faces down out of the sole and ``toe_above`` faces up out of the top of
    the toe, so the surfaces they can be pushed by are on opposite sides of the toe. That
    separation is what makes "standing" and "hooking" different hypotheses at all.
    """
    skin = torch.tensor([[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]])       # under faces down, above up
    axes = -skin
    torch.testing.assert_close(axes[0, 0], torch.tensor([0.0, 1.0, 0.0]))
    assert float((axes[0, 0] * axes[0, 1]).sum()) == pytest.approx(-1.0)

    generators = cone_generators(axes, math.tan(math.radians(45.0)))  # (1, 2, 4, 3)
    under_up = (generators[0, 0] * axes[0, 0]).sum(-1)
    above_up = (generators[0, 1] * axes[0, 0]).sum(-1)
    assert torch.all(under_up > 0), "every edge of the sole's cone pushes the body up"
    assert torch.all(above_up < 0), "every edge of the toe-top's cone pulls the body down"


# ── the toe rule: whichever side the solve loaded takes the contact ───────────

def test_toe_rule_gives_a_supported_foot_to_the_sole() -> None:
    """The sole carried the load, so the whole contact is ``toe_under``."""
    labels = np.zeros((6, 4), bool)
    labels[:, 0] = labels[:, 1] = True                # both toe frames claim the contact
    magnitude = np.zeros((6, 4), dtype=np.float32)
    magnitude[:, 0] = 40.0                            # the load went through the sole
    magnitude[:, 1] = 5.0                             # the toe top carried a trace
    resolved = resolve_toe_conflicts(labels, magnitude, frame_names=_TOE_NAMES, min_frames=3)
    assert resolved[:, 0].all() and not resolved[:, 1].any()


def test_toe_rule_gives_a_loaded_toe_top_the_contact() -> None:
    """The load went through the toe top, so the whole contact is ``toe_above``.

    The regression behind this test is a bat hang: the climber is inverted and the hooked
    toe holds the body weight *up*. An earlier rule read the force's vertical sign
    ("hooks pull down") and renamed every bat hang to a stance — the rule must read which
    side carried the load, never which way the force pointed.
    """
    labels = np.zeros((6, 4), bool)
    labels[:, 0] = labels[:, 1] = True
    magnitude = np.zeros((6, 4), dtype=np.float32)
    magnitude[:, 0] = 5.0                             # a trace of load on the sole
    magnitude[:, 1] = 60.0                            # the toe top carried the body
    resolved = resolve_toe_conflicts(labels, magnitude, frame_names=_TOE_NAMES, min_frames=3)
    assert resolved[:, 1].all() and not resolved[:, 0].any()


def test_toe_rule_never_switches_patch_inside_one_contact() -> None:
    """The regression this rule exists for: the patch used to flip mid-foothold.

    Per step the load wanders between the two frames; the whole contact still has to come
    out as one patch, because a foot does not change which side of the toe is touching
    while it stays on the hold.
    """
    steps = 12
    labels = np.zeros((steps, 4), bool)
    labels[:, 0] = labels[:, 1] = True
    magnitude = np.zeros((steps, 4), dtype=np.float32)
    magnitude[:, 0] = 30.0
    magnitude[:, 1] = 25.0
    magnitude[4:7, 0] = 2.0                           # three steps where the sole looks weak
    magnitude[4:7, 1] = 80.0                          # and the toe top looks strong
    resolved = resolve_toe_conflicts(labels, magnitude, frame_names=_TOE_NAMES, min_frames=3)
    for column in (0, 1):
        assert len(set(resolved[:, column].tolist())) == 1, "the patch changed inside a contact"
    assert not (resolved[:, 0] & resolved[:, 1]).any()


def test_toe_rule_decides_each_contact_separately() -> None:
    """Two contacts of the same foot are two decisions — the foot may stand, then hook."""
    labels = np.zeros((14, 4), bool)
    labels[0:5, 0] = labels[0:5, 1] = True            # first contact
    labels[9:14, 0] = labels[9:14, 1] = True          # second contact, after a real release
    magnitude = np.zeros((14, 4), dtype=np.float32)
    magnitude[0:5, 0] = 50.0                          # standing
    magnitude[9:14, 1] = 50.0                         # hooking
    resolved = resolve_toe_conflicts(labels, magnitude, frame_names=_TOE_NAMES, min_frames=3)
    assert resolved[0:5, 0].all() and not resolved[0:5, 1].any()
    assert resolved[9:14, 1].all() and not resolved[9:14, 0].any()


def test_toe_rule_leaves_the_foot_contact_itself_untouched() -> None:
    """It only moves the label between the two frames; "this foot is touching" is invariant.

    That invariance is why the rule cannot change what the scorer or any downstream stage
    sees — only which patch of the toe carries the name.
    """
    generator = np.random.default_rng(0)
    labels = generator.random((40, 4)) < 0.4
    magnitude = np.abs(generator.normal(size=(40, 4))).astype(np.float32) * 30.0
    resolved = resolve_toe_conflicts(labels, magnitude, frame_names=_TOE_NAMES, min_frames=1)
    for under, above in ((0, 1), (2, 3)):
        before = labels[:, under] | labels[:, above]
        after = resolved[:, under] | resolved[:, above]
        assert (before == after).all(), "the foot's own contact must not move"
        assert not (resolved[:, under] & resolved[:, above]).any()


def test_toe_conflicts_keep_a_label_nothing_contests() -> None:
    """With no force either way the contact stays on the sole — standing is the default."""
    labels = np.zeros((6, 4), bool)
    labels[:, 0] = True
    resolved = resolve_toe_conflicts(labels, np.zeros((6, 4), dtype=np.float32),
                                     frame_names=_TOE_NAMES, min_frames=3)
    assert resolved[:, 0].all()
    assert not resolved[:, 1].any()


def test_union_of_the_two_readouts_never_holds_both_toes() -> None:
    """Geometry keeps one toe, the force keeps the other; the union must not hold both."""
    steps = 8
    geometric = np.zeros((steps, 4), bool)
    geometric[:, 0] = True                             # detector says toe_under_left
    magnitude = np.zeros((steps, 4), dtype=np.float32)
    magnitude[2:7, 1] = 400.0                          # the force loaded the toe top
    force = force_labels(magnitude, frame_names=_TOE_NAMES, valid=np.ones(steps, bool),
                            threshold_n=30.0, min_frames=3)
    union = geometric | force
    assert (union[:, 0] & union[:, 1]).sum() > 0, "the union really does hold both toes"

    resolved = resolve_toe_conflicts(union, magnitude, frame_names=_TOE_NAMES, min_frames=3)
    assert not (resolved[:, 0] & resolved[:, 1]).any(), "at most one toe per foot per step"
    assert resolved[:, 1].all(), "the hook explains the whole contact, so it takes all of it"


# ── thresholds, spans and gaps ────────────────────────────────────────────────

def test_threshold_and_min_span_drop_weak_and_brief_load() -> None:
    """Below-threshold load never fires, and a run shorter than ``min_frames`` is dropped."""
    magnitude = np.zeros((6, 4), dtype=np.float32)
    magnitude[0:2, 0] = 100.0        # two frames only — too brief
    magnitude[3:6, 2] = 100.0        # three frames — survives
    magnitude[3:6, 3] = 10.0         # below threshold
    labels = force_labels(magnitude, frame_names=_TOE_NAMES, valid=np.ones(6, bool),
                             threshold_n=30.0, min_frames=3)
    assert labels[:, 0].tolist() == [False] * 6
    assert labels[:, 2].tolist() == [False, False, False, True, True, True]
    assert labels[:, 3].tolist() == [False] * 6


def test_invalid_frames_can_never_be_labelled() -> None:
    """A step where the person is not tracked carries no contact, whatever the force says."""
    magnitude = np.full((6, 4), 500.0, dtype=np.float32)
    valid = np.array([True, True, True, False, False, False])
    labels = force_labels(magnitude, frame_names=_TOE_NAMES, valid=valid,
                             threshold_n=30.0, min_frames=3)
    assert not labels[3:].any()


def test_min_span_is_per_column_and_leaves_long_runs_alone() -> None:
    mask = np.array([[True, True], [True, False], [False, True], [True, True], [True, True]])
    kept = min_span(mask, 2)
    assert kept[:, 0].tolist() == [True, True, False, True, True]
    assert kept[:, 1].tolist() == [False, False, True, True, True]


def test_min_span_keeps_a_short_run_that_overlaps_a_keep_step() -> None:
    """The penetration exception: a pressing step exempts its run from the duration rule."""
    mask = np.zeros((6, 2), bool)
    mask[2, 0] = True                # one-frame run, pressed → survives
    mask[2, 1] = True                # one-frame run, not pressed → dropped
    keep = np.zeros((6, 2), bool)
    keep[2, 0] = True
    kept = min_span(mask, 3, keep)
    assert kept[2, 0] and not kept[:, 1].any()


def test_close_gaps_fills_short_holes_but_not_real_releases() -> None:
    """A one-frame blink is flicker; a long gap is the limb genuinely letting go."""
    mask = np.zeros((12, 2), bool)
    mask[0:4, 0] = True
    mask[5:9, 0] = True              # a one-frame hole at step 4 — flicker
    mask[0:3, 1] = True
    mask[8:12, 1] = True             # a five-frame hole — a real release
    filled = close_gaps(mask, 3)
    assert filled[:, 0].tolist() == [True] * 9 + [False] * 3
    assert filled[:, 1].tolist() == mask[:, 1].tolist()


def test_close_gaps_leaves_the_ends_of_the_clip_alone() -> None:
    """There is no contact on the far side of a leading or trailing gap to bridge to."""
    mask = np.zeros((8, 1), bool)
    mask[2:6, 0] = True
    filled = close_gaps(mask, 3)
    assert filled[:, 0].tolist() == mask[:, 0].tolist()


def test_hysteresis_holds_a_span_through_a_dip() -> None:
    """Entering needs the high threshold; staying only needs the low one."""
    series = np.array([[0.0], [10.0], [4.0], [4.0], [1.0], [4.0]], dtype=np.float32)
    held = hysteresis(series, enter_above=8.0, hold_above=3.0)
    assert held[:, 0].tolist() == [False, True, True, True, False, False]
    single = hysteresis(series, enter_above=8.0, hold_above=8.0)
    assert single[:, 0].tolist() == [False, True, False, False, False, False]


def test_force_labels_hysteresis_extends_a_contact_it_never_starts_one() -> None:
    """The hold threshold can only lengthen a span that the enter threshold already opened."""
    magnitude = np.zeros((10, 4), dtype=np.float32)
    magnitude[2:5, 2] = 100.0        # clearly loaded: opens the span
    magnitude[5:8, 2] = 40.0         # fading, but never released
    magnitude[0:2, 3] = 40.0         # never crosses the enter threshold at all
    plain = force_labels(magnitude, frame_names=_TOE_NAMES, valid=np.ones(10, bool),
                            threshold_n=60.0, min_frames=3)
    held = force_labels(magnitude, frame_names=_TOE_NAMES, valid=np.ones(10, bool),
                           threshold_n=60.0, hold_n=30.0, min_frames=3)
    assert plain[:, 2].sum() == 3
    assert held[:, 2].sum() == 6, "the fading tail stays in contact"
    assert not held[:, 3].any(), "a frame that never crossed the enter threshold stays off"


# ── the no-scene policy and the shipped recipe ────────────────────────────────

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


def test_the_solve_has_five_knobs_and_they_are_the_documented_recipe() -> None:
    """The shipped defaults are the measured recipe, and there is nothing else to tune."""
    knobs = ForceKnobs()
    assert set(vars(knobs)) == {"half_angle_deg", "stillness_vstar_mps", "gap_dstar_m",
                                "mag_weight", "iters"}
    assert knobs.half_angle_deg == pytest.approx(45.0)
    assert knobs.stillness_vstar_mps == pytest.approx(0.2)
    assert knobs.gap_dstar_m == pytest.approx(0.05)
    assert knobs.iters == 2000


# ── the far-still label veto ──────────────────────────────────────────────────

def test_far_force_labels_need_stillness_and_near_ones_do_not() -> None:
    """Near + fast (a loaded sprint plant) labels; far + still (a misregistered but
    gripping body) labels; far + fast (a bad-depth flight scrap) is vetoed."""
    dist = np.array([[0.05, 0.40, 0.40]])
    drift = np.array([[0.50, 0.10, 0.50]])
    veto = far_still_veto(dist, drift, near_m=0.10, still_m=0.04, far_still_x=4.0)
    assert veto.tolist() == [[False, False, True]]


def test_missing_evidence_reads_far_and_fast_in_the_veto() -> None:
    """NaN distance = no scene evidence = far; NaN drift = no stillness evidence = fast.
    Unknown never manufactures a label."""
    dist = np.array([[np.nan, np.nan, 0.05]])
    drift = np.array([[0.01, np.nan, np.nan]])
    veto = far_still_veto(dist, drift, near_m=0.10, still_m=0.04, far_still_x=4.0)
    assert veto.tolist() == [[False, True, False]], (
        "NaN dist + still stays (far but still); NaN dist + NaN drift is vetoed; "
        "near needs no stillness at all")


# ───────────────────────── the solve's NaN-drift policy ─────────────────────────

class _StubDyn:
    """Just enough of :class:`FrameDynamics` for :func:`solve_frame_forces`:
    two contact frames whose Jacobians map force straight onto the root force rows,
    a constant 1-BW gravity residual, and upward cone axes."""

    def __init__(self, steps: int, frames: int = 2, nv: int = 6):
        self.points = torch.zeros(steps, frames, 3)
        self.body_weight_n = 700.0
        self.tau = torch.zeros(steps, nv)
        # The solve's residual is tau − J·f, so a +W row is a load the (+y) cones
        # can carry: the no-contact RNEA says W newtons of support are missing.
        self.tau[:, 1] = self.body_weight_n
        self.jacobians = torch.zeros(steps, frames, 3, nv)
        for i in range(3):
            self.jacobians[:, :, i, i] = 1.0          # world force → root force rows
        self.cone_axes = torch.zeros(steps, frames, 3)
        self.cone_axes[:, :, 1] = 1.0                 # every cone can push +y (up)


def test_nan_drift_steps_are_not_candidates_and_never_poison_the_solve() -> None:
    """A step with NaN pre-enforcement drift has no stillness evidence: it carries
    exactly zero force, while the finite-drift steps still balance gravity — the
    9/361 corpus scenes with tracking gaps must produce a witness, not a NaN loss."""
    from tools.human_optim.forces import solve_frame_forces

    steps = 6
    dyn = _StubDyn(steps)
    drift = torch.zeros(steps, 2)
    drift[2] = float("nan")                           # a gap-window step
    dist = torch.full((steps, 2), 0.02)               # near the wall everywhere
    knobs = ForceKnobs(gap_dstar_m=0.25, stillness_vstar_mps=0.2)
    sol = solve_frame_forces(
        dyn, valid=np.ones(steps, bool), drift_m=drift, drift_window_s=0.35,
        dist_m=dist, normal_dot=torch.ones(steps, 2), knobs=knobs)
    assert np.isfinite(sol.forces).all()
    assert not sol.candidate[2].any(), "NaN drift must exclude the step from candidacy"
    assert sol.magnitude[2].sum() == 0.0
    finite_steps = [t for t in range(steps) if t != 2]
    total = sol.forces[finite_steps, :, 1].sum(axis=1)
    assert (total > 0.5 * dyn.body_weight_n).all(), "finite steps still carry the load"
