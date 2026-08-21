"""Contact forces on the 12 contact frames — the force witness for contact.

Run inverse dynamics (RNEA) on the frozen kinematic trajectory with nothing
touching the body. The result ``tau_nc`` is the generalized force the body would
need to move that way in mid-air. Its first six rows belong to the free-flyer
root, which has no motor: whatever sits there must have come in through a
contact. That leftover is the **residual**, and this module asks the inverse
question — *what world forces at the 12 contact frames explain it?*

A world force ``f`` at contact frame ``i`` adds ``J_iᵀ f`` to the generalized
force, where ``J_i`` is that frame's ``(3, nv)`` translation Jacobian. So the
residual left after contact is

    r = tau_nc − Σ_i J_iᵀ f_i

and the solve minimizes its root six rows over the unknown forces. Writing it with
Jacobians (instead of RNEA's per-joint ``fext``) keeps everything in one coordinate
system: ``J_iᵀ f`` already lives in the same tangent space as ``tau_nc``, so no
moment arms are computed by hand and the two toe frames that share a parent joint
cannot collide.

**Friction cones are what make the answer mean something.** With free forces, any
one limb can absorb almost any residual, so the solve says nothing about *which*
limb touched — that is the recorded failure of the earlier attempt. Here every
candidate frame gets a cone: its force is a nonnegative combination of the four
edges of a pyramid inscribed in a 45° friction cone about that frame's own
**negated outward skin normal**. The skin of a touching frame faces into the
surface, so the surface pushes back the other way; a frame can then only explain
the part of the residual it is physically oriented to explain. The two toe frames
of one foot carry *opposite* axes (their skin faces opposite ways), so the solve —
not a normal-vs-scene tiebreak — decides which of them is holding.

Two prices keep the rest honest. The **stillness price** makes force on a moving
frame expensive (a sliding limb carries little load), and the **gap price** makes
force far from the scene expensive. The gap price is measured **relative** to the
nearest contact frame of the same step, so a body that is entirely misregistered —
every frame far from the wall — can still be assigned force. That is exactly the
case the distance gate cannot see.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch

from .friction import cone_generators
from .qmath import normalize_q

_GRAVITY_MAG = 9.81  # m/s²

#: ``frame_scene_signals`` reports ``finfo.max`` when a step's contact frames have no
#: scene point anywhere near them. That is **no evidence**, not "far from the wall", and
#: the two must never be confused — see :func:`solve_frame_forces`. Anything above this
#: is the sentinel.
NO_SCENE_M = 1e3

#: Characteristic lever arm (m). The residual's last three root rows are torques; dividing
#: them by this puts all six root rows in force units, so the objective adds one wrench in
#: one unit instead of two quantities in two.
TORQUE_LENGTH_M = 0.2


@dataclass(frozen=True)
class ForceKnobs:
    """The force solve, in five numbers (defaults = the shipped recipe)."""

    #: Friction half-angle in degrees; μ = tan(half_angle_deg).
    half_angle_deg: float = 45.0
    #: Stillness price: force on a frame drifting faster than this is expensive (m/s).
    stillness_vstar_mps: float = 0.2
    #: Gap price: force further than this from the scene is expensive (m), measured
    #: relative to the nearest candidate frame of the same step.
    gap_dstar_m: float = 0.05
    #: Plain magnitude price (inside the square, like the dynamics stage).
    mag_weight: float = 0.01
    #: Projected-Adam iterations for the whole-sequence solve.
    iters: int = 2000


@dataclass(frozen=True)
class ForceSolution:
    """One person's solved contact forces and the residual they explain."""

    #: ``(T, 12, 3)`` world contact forces, newtons; zero off the candidate set.
    forces: np.ndarray
    #: ``(T, 12)`` force magnitudes, newtons. The toe rule reads these: which side the
    #: solve loaded is its verdict on which side really touched.
    magnitude: np.ndarray
    #: ``(T, 12)`` bool — frames the solve was allowed to load.
    candidate: np.ndarray
    #: ``(T, 12)`` ``dot(cone axis, nearest scene normal)`` ∈ [−1, 1] — ``+1`` when this
    #: patch of skin faces straight into the wall, so its cone pushes the way the wall
    #: pushes. ``−1`` where there is no scene evidence (see :data:`NO_SCENE_M`).
    normal_dot: np.ndarray
    #: ``(T,)`` root residual force / torque norms with no contact forces.
    residual_force_before_n: np.ndarray
    residual_torque_before_nm: np.ndarray
    #: ``(T,)`` the same after the solve.
    residual_force_after_n: np.ndarray
    residual_torque_after_nm: np.ndarray
    #: Total baked body mass × g, newtons — the natural scale for a force threshold.
    body_weight_n: float
    #: Worst KKT violation of the ``λ ≥ 0`` constraint at the returned point, in
    #: body-weight units. Near zero = the convex solve converged.
    kkt_violation: float = 0.0


class FrameDynamics:
    """One person's frozen-trajectory dynamics: RNEA residual + contact-frame Jacobians.

    Everything here depends only on ``q`` and the body, so it is computed once. ``tau``
    is the no-contact generalized force under finite-difference velocity and acceleration.

    :param body: the :class:`tools.body.Body` SMPL-X wrapper.
    :param betas: ``(10,)`` shape coefficients.
    :param q: ``(T, 211)`` world-frame configuration trajectory.
    :param gravity_world: ``(3,)`` world-frame down direction (any length).
    :param fps: clip frame rate, for the finite differences.
    """

    def __init__(self, body, betas: torch.Tensor, q: torch.Tensor, *,
                 gravity_world: np.ndarray, fps: float, device: str = "cuda") -> None:
        from better_robot.dynamics import rnea
        from better_robot.kinematics import (
            compute_joint_jacobians, forward_kinematics, get_frame_jacobian,
            update_frame_placements,
        )

        from ..body import frame_normals
        from ..smplx_robot.dynamics import velocity_acceleration_from_trajectory

        dev = torch.device(device)
        model = body.shaped(betas.to(dev)).robot
        gravity = torch.as_tensor(gravity_world, dtype=torch.float32, device=dev)
        gravity = gravity / (gravity.norm() + 1e-9) * _GRAVITY_MAG
        model = dataclasses.replace(model, values=dataclasses.replace(
            model.values, gravity=torch.cat([gravity, torch.zeros(3, device=dev)])))

        self.model = model
        self.q_model = normalize_q(q.to(dev), body.num_joints)
        self.frame_names = tuple(body.contact_frame_names)
        #: ``(3,)`` world up — the direction a contact has to push to hold the body.
        self.body_weight_n = float(model.values.body_inertias[..., 1:, 0].sum()) * _GRAVITY_MAG

        with torch.no_grad():
            velocity, acceleration = velocity_acceleration_from_trajectory(
                model, self.q_model, 1.0 / float(fps))
            #: ``(T, nv)`` no-contact generalized force.
            self.tau = rnea(model, self.q_model, velocity, acceleration)

            data = forward_kinematics(model, self.q_model)
            compute_joint_jacobians(model, data)      # once, else every frame redoes the pass
            frame_ids = list(body.contact_frames.frame_ids)
            #: ``(T, 12, 3, nv)`` world translation Jacobian of each contact frame.
            self.jacobians = torch.stack(
                [get_frame_jacobian(model, data, fid)[..., :3, :] for fid in frame_ids], dim=-3)
            update_frame_placements(model, data)
            placements = data.frame_pose_world[..., torch.tensor(frame_ids, device=dev), :]
            #: ``(T, 12, 3)`` world contact-frame positions and outward skin normals.
            self.points = placements[..., :3]
            self.normals = frame_normals(placements)

    @property
    def cone_axes(self) -> torch.Tensor:
        """``(T, 12, 3)`` the direction each frame's contact can push the body.

        The negated outward skin normal. It needs no scene cloud at all, which is why it
        survives a misregistered body, and it is the only source that separates the two
        toe frames — their skin faces opposite ways, so their cones are opposite. (A
        sign-corrected *scene* normal measured the same to within .002 F1, for a pile of
        extra machinery, so it is not an option here.)
        """
        return -self.normals


def solve_frame_forces(dyn: FrameDynamics, *, valid: np.ndarray, drift_m: torch.Tensor,
                       drift_window_s: float, dist_m: torch.Tensor,
                       normal_dot: torch.Tensor, knobs: ForceKnobs) -> ForceSolution:
    """Solve the whole sequence's contact forces at once → :class:`ForceSolution`.

    :param valid: ``(T,)`` bool — steps where this person is tracked.
    :param drift_m: ``(T, 12)`` net window drift of each contact frame, metres. This is
        the *pre-enforcement* drift (``drift_pre``): motion measured before any contact
        term acted, so the stillness price cannot be manufactured by the optimizer.
    :param drift_window_s: the window ``drift_m`` was measured over — drift/window is
        the frame's speed for the stillness price.
    :param dist_m: ``(T, 12)`` distance to the nearest scene point, metres. Values above
        :data:`NO_SCENE_M` are the "no scene point anywhere near" sentinel.
    :param normal_dot: ``(T, 12)`` ``dot(cone axis, that scene point's normal)``, carried
        through to the solution for the toe rule.
    """
    device = dyn.points.device
    num_steps, num_frames = dyn.points.shape[:2]
    # Everything is solved in units of body weight: forces are fractions of it and the
    # whole objective is divided by its square. That is a pure rescaling — it moves no
    # optimum, every term scales the same way — but it turns a problem whose numbers run
    # to 10⁵ N² into an O(1) one, which is the difference between L-BFGS converging and
    # its line search giving up on the first step (measured: it did, silently).
    weight = dyn.body_weight_n
    tau_nc = dyn.tau / weight
    valid_t = torch.as_tensor(np.asarray(valid, bool), device=device)

    # "No scene point near this frame" is missing evidence, not distance. Treating the
    # sentinel as a number is exactly the bug that made the relative and absolute gap
    # ablations measure the encoding instead of the pricing: relative mode subtracted
    # sentinel−sentinel = 0 and handed those steps FREE force, absolute mode clamped them
    # to the 1 m cap and blocked them. One policy, applied before any pricing: a frame with
    # no scene evidence is not a candidate. Everywhere else in this pipeline a hole in the
    # cloud reads as *unknown*, and unknown never manufactures a contact.
    known = torch.isfinite(dist_m) & (dist_m < NO_SCENE_M)
    candidate = valid_t[:, None] & known

    # Cone forces: four pyramid edges per frame, coefficients constrained nonnegative, so
    # the force cannot leave the cone and there is nothing to tune. The coefficients ARE
    # the variables — no reparameterization. The objective is a convex quadratic in them
    # over the nonnegative orthant, so projected gradient reaches the global optimum and
    # exactly zero force is attainable. (An earlier softplus reparameterization made zero
    # unreachable at finite parameters and the optimized problem nonconvex, for nothing.)
    generators = cone_generators(dyn.cone_axes,
                                 math.tan(math.radians(knobs.half_angle_deg))).detach()
    coefficients = torch.zeros((num_steps, num_frames, 4), device=device, requires_grad=True)
    active = candidate.to(torch.float32)[..., None]

    speed = drift_m / float(drift_window_s)
    # Relative pricing is the floater rescue: with every frame far from the wall, absolute
    # pricing makes all force so expensive that the solve leaves gravity unbalanced and
    # nothing crosses the threshold — on exactly the bodies the force witness exists to
    # rescue. Relative pricing keeps the price's real job (prefer the nearer frame) while
    # the weight still has to land somewhere. The datum is the nearest CANDIDATE, so a
    # frame with no scene evidence cannot set it.
    gap = torch.where(known, dist_m, torch.zeros_like(dist_m))
    near = torch.where(candidate, gap, torch.full_like(gap, float("inf"))).amin(dim=1)
    gap = gap - torch.where(torch.isfinite(near), near, torch.zeros_like(near))[:, None]
    gap = gap.clamp(0.0, 1.0)

    # Root torque rows divided by the characteristic lever arm, so all six root rows are
    # one wrench in one unit.
    row_scale = torch.ones(dyn.jacobians.shape[-1], device=device)
    row_scale[3:] = 1.0 / TORQUE_LENGTH_M

    def compose() -> torch.Tensor:
        """Contact force per ``(step, frame)``, in units of body weight."""
        return active * torch.einsum("tfg,tfgd->tfd", coefficients, generators)

    def loss_of(force: torch.Tensor) -> torch.Tensor:
        residual = (tau_nc - torch.einsum("tfkn,tfk->tn", dyn.jacobians, force)
                    ) * valid_t[:, None] * row_scale
        loss = 0.5 * residual[:, :6].pow(2).sum()
        squared = force.pow(2).sum(-1)                     # |f|², exactly quadratic
        loss = loss + 0.5 * knobs.mag_weight**2 * squared.sum()
        loss = loss + 0.5 * ((speed / knobs.stillness_vstar_mps).pow(2) * squared).sum()
        return loss + 0.5 * ((gap / knobs.gap_dstar_m).pow(2) * squared).sum()

    # Projected Adam with a decaying step. The decay is what makes the KKT check
    # meaningful: at a fixed step size Adam orbits the optimum and the gradient never
    # settles, so "converged" could not be distinguished from "still moving".
    optimizer = torch.optim.Adam([coefficients], lr=0.05)
    schedule = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=(1e-3) ** (1.0 / max(int(knobs.iters), 1)))
    for _ in range(int(knobs.iters)):
        optimizer.zero_grad()
        loss_of(compose()).backward()
        optimizer.step()
        schedule.step()
        with torch.no_grad():
            coefficients.clamp_(min=0.0)                   # projection onto λ ≥ 0

    # KKT check for the nonnegativity constraint: at the optimum each coefficient either
    # sits at zero with a non-negative gradient (pushing further down is blocked) or is
    # positive with a vanishing gradient. The worst of the two, in body-weight units, is
    # how we know the convex solve converged rather than merely stopped.
    optimizer.zero_grad()
    loss_of(compose()).backward()
    with torch.no_grad():
        gradient = coefficients.grad
        violation = torch.maximum(
            torch.where(coefficients > 0, gradient.abs(), torch.zeros_like(gradient)),
            torch.clamp(-gradient, min=0.0))
        live = active[..., 0].bool()
        kkt = float(violation[live].max()) if bool(live.any()) else 0.0

        force = compose() * weight                                    # back to newtons
        before = dyn.tau
        after = before - torch.einsum("tfkn,tfk->tn", dyn.jacobians, force)
    to_numpy = lambda t: t.detach().cpu().numpy().astype(np.float32)  # noqa: E731
    if not np.isfinite(to_numpy(force)).all():
        raise FloatingPointError("force solve produced non-finite forces")
    return ForceSolution(
        forces=to_numpy(force),
        magnitude=to_numpy(force.norm(dim=-1)),
        candidate=candidate.cpu().numpy(),
        normal_dot=to_numpy(normal_dot),
        residual_force_before_n=to_numpy(before[:, :3].norm(dim=-1)),
        residual_torque_before_nm=to_numpy(before[:, 3:6].norm(dim=-1)),
        residual_force_after_n=to_numpy(after[:, :3].norm(dim=-1)),
        residual_torque_after_nm=to_numpy(after[:, 3:6].norm(dim=-1)),
        body_weight_n=weight,
        kkt_violation=kkt,
    )


#: Nearest-scene-point search box margin for the force path. The detector's 10 cm margin
#: answers "is it touching?"; here an empty box must mean the body really is far from all
#: reconstructed geometry, not that the box was small — otherwise ordinary steps get
#: mislabelled "no scene evidence". Measured: at 10 cm, 6.4% of gtIGgs steps and 5.0% of
#: vcRBT steps had all 12 frames unknown; at 1 m, none do, on any of the 13 scenes.
SEARCH_MARGIN_M = 1.0


def solve_person_forces(body, betas: torch.Tensor, q: torch.Tensor, valid: np.ndarray, *,
                        drift_pre: torch.Tensor, drift_window_s: float,
                        gravity_world: np.ndarray, fps: float, scene_pts: torch.Tensor,
                        scene_normals: torch.Tensor, scene_conf: torch.Tensor,
                        knobs: ForceKnobs) -> ForceSolution:
    """One person, start to finish: RNEA + Jacobians + cones + the solve.

    The single entry point shared by the contact stage and the diagnostics, so the two can
    never drift apart. ``drift_pre`` is the pre-enforcement drift ``(T, 12)``; passing
    anything measured on a body that a contact term already acted on would make the
    stillness price self-fulfilling.
    """
    from .contacts import frame_scene_signals

    dynamics = FrameDynamics(body, betas, q, gravity_world=gravity_world, fps=fps,
                             device=str(scene_pts.device))
    _, dist_m, _, normal_dot = frame_scene_signals(
        dynamics.points, dynamics.normals, scene_pts, scene_normals, scene_conf,
        prune_margin_m=SEARCH_MARGIN_M)
    return solve_frame_forces(
        dynamics, valid=valid, drift_m=drift_pre, drift_window_s=drift_window_s,
        dist_m=dist_m, normal_dot=normal_dot, knobs=knobs)


def _runs(column: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of a 1-D bool array as half-open ``(start, stop)`` pairs."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], column.view(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def min_span(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Drop ``True`` runs shorter than ``min_frames`` in each column of an ``(T, C)`` bool array."""
    out = np.array(mask, dtype=bool, copy=True)
    for column in range(out.shape[1]):
        for start, stop in _runs(out[:, column]):
            if stop - start < min_frames:
                out[start:stop, column] = False
    return out


def close_gaps(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Fill ``False`` runs shorter than ``min_frames`` — the mirror image of :func:`min_span`.

    A label that blinks off for a frame or two in the middle of a contact did not really
    let go: no real grip is released and retaken that fast. Running both passes states one
    plain rule with one number — **a contact must last at least ``min_frames``, and so must
    a release.** Gaps at the very start and end of the clip are left alone; there is no
    contact on the far side of them to bridge to.
    """
    out = np.array(mask, dtype=bool, copy=True)
    for column in range(out.shape[1]):
        for start, stop in _runs(~out[:, column]):
            if start > 0 and stop < out.shape[0] and stop - start < min_frames:
                out[start:stop, column] = True
    return out


def hysteresis(series: np.ndarray, *, enter_above: float, hold_above: float) -> np.ndarray:
    """``(T, C)`` bool: start a run where ``series > enter_above``, hold while ``> hold_above``.

    With a single threshold a signal hovering at the line switches on and off every other
    frame. A second, lower threshold prices that switching: once on, the signal has to fall
    all the way to ``hold_above`` before the label lets go. ``hold_above == enter_above``
    is the plain single-threshold behaviour.
    """
    enter, hold = series > enter_above, series > hold_above
    out = np.zeros_like(enter)
    active = np.zeros(series.shape[1], dtype=bool)
    for step in range(series.shape[0]):
        active = (active & hold[step]) | enter[step]
        out[step] = active
    return out


def resolve_toe_conflicts(labels: np.ndarray, magnitude: np.ndarray, *, frame_names,
                          min_frames: int = 3) -> np.ndarray:
    """Give each foot ONE toe side per contact — the side the solved force loaded.

    A foot's contact frames come in two sides facing opposite ways through the foot:
    the underside (``heel`` plus ``toe_under`` in the legacy set; ``heel`` plus
    ``ball_inner`` / ``ball_outer`` / ``big_toe`` in the full-body set) — the foot stands on a hold; and ``toe_above``, the top of the
    toe — the foot presses it against a hold (a toe hook, or the whole body inverted in
    a bat hang). At most one side touches, so a label naming both is wrong — and the
    readouts do name both, because the geometric detector and :func:`force_labels` each
    keep a single side but can keep *different* ones.

    The verdict is the force solve's own load allocation, summed over the whole contact.
    A contact is one unbroken run of "this foot is touching" (any toe frame on); across
    that run each side is scored by how much force the solve put through it::

        standing = Σ |f|[underside frames]        hooking = Σ |f|[toe_above]

    The bigger sum takes the entire run; ties keep the underside (standing is much the
    more common contact, and the result then never depends on evaluation order). This
    works because each frame's force lives in a cone about that frame's *own* skin normal:
    the solve can only load the side whose orientation can explain the body's motion, so
    "which side carried the load" already contains the normal information — the body's
    normals, which are exact, not the cloud's. Whole-run evidence instead of one frame,
    so the side cannot change while the foot stays on the hold.

    Within the winning side each frame keeps its own label — which underside points touch
    is real information (edging vs a flat foot, heel on or off). Steps of the run where only the losing side
    was labelled are given to the winning side's most-loaded frame, so the union — which
    is what "this foot is in contact" means, and all the scorer and the downstream stages
    read — comes out unchanged.

    Two earlier readings of the same data were refuted by eyeballing the disputed spans:

    * **The force's vertical sign** ("standing pushes up, hooking pulls down") renamed
      every bat hang to a stance: a toe hooked overhead with the climber inverted holds
      the body weight *up*, which the sign test declared impossible for a hook.
    * **The cloud normal at the nearest scene point** votes "top of toe" for ordinary
      stances on small holds: the nearest cloud points are the wall plane behind the
      hold, and a standing toe-top naturally faces the wall. It is only trustworthy on
      large volumes the cloud actually resolved — where it agrees with the load anyway.

    Runs shorter than ``min_frames`` are dropped at the end. Sides are resolved by NAME
    (:func:`toe_side_slots`): the stored frame order is not the obvious one.
    """
    from .contacts import toe_side_slots

    out = np.array(labels, dtype=bool, copy=True)
    sides = toe_side_slots(frame_names)
    for sole, top in sides:
        sole, top = list(sole), list(top)
        for start, stop in _runs(out[:, sole + top].any(axis=1)):
            standing = magnitude[start:stop, sole].sum()
            hooking = magnitude[start:stop, top].sum()
            win, lose = (sole, top) if standing >= hooking else (top, sole)
            out[start:stop, lose] = False
            gap = ~out[start:stop, win].any(axis=1)
            if gap.any():
                dominant = win[int(np.argmax(magnitude[start:stop, win].sum(axis=0)))]
                out[start:stop, dominant] |= gap
    toe_columns = sorted({c for sole, top in sides for c in (*sole, *top)})
    out[:, toe_columns] = min_span(out[:, toe_columns], min_frames)
    return out


def force_labels(magnitude: np.ndarray, *, frame_names, valid: np.ndarray,
                 threshold_n: float, hold_n: float | None = None,
                 min_frames: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Contact labels from solved force magnitudes ``(T, F)`` → ``(labels, toe_split)``.

    A frame enters contact when it carries more than ``threshold_n`` newtons and holds it
    while it carries more than ``hold_n`` (default ``None`` = no hysteresis, i.e. one
    threshold). Runs shorter than ``min_frames`` are dropped, then each foot gets one toe
    patch per contact (:func:`resolve_toe_conflicts` on the same magnitudes) — the same
    rule the stage applies to the union, so there is only ever one toe rule in the
    pipeline.

    ``toe_split`` marks the steps where both sides of a foot's toe were labelled and the
    rule had to choose; it is a pure diagnostic.
    """
    from .contacts import toe_side_slots

    fires = hysteresis(magnitude, enter_above=threshold_n,
                       hold_above=threshold_n if hold_n is None else hold_n)
    fires &= np.asarray(valid, bool)[:, None]
    fires = min_span(fires, min_frames)
    toe_split = np.zeros_like(fires)
    for sole, top in toe_side_slots(frame_names):
        cols = [*sole, *top]
        both = fires[:, list(sole)].any(axis=1) & fires[:, list(top)].any(axis=1)
        toe_split[:, cols] |= (both[:, None] & fires[:, cols])
    resolved = resolve_toe_conflicts(
        fires, magnitude, frame_names=frame_names, min_frames=min_frames)
    return resolved, toe_split
