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

and the solve minimizes ``r`` over the unknown forces. Writing it with Jacobians
(instead of RNEA's per-joint ``fext``) keeps everything in one coordinate system:
``J_iᵀ f`` already lives in the same tangent space as ``tau_nc``, so no moment
arms are computed by hand and the two toe frames that share a parent joint cannot
collide.

**Friction cones are what make the answer mean something.** With free forces, any
one limb can absorb almost any residual, so the solve says nothing about *which*
limb touched — that is the recorded failure of the earlier attempt. Here every
candidate frame gets a cone: its force is a nonnegative combination of the four
edges of a pyramid inscribed in a 45° friction cone about that frame's own
contact axis, so it can only push roughly the way that patch of skin faces. A
frame can then only explain the part of the residual it is physically oriented to
explain. The two toe frames of one foot carry *opposite* axes (their skin faces
opposite ways), so the solve — not a normal-vs-scene tiebreak — decides which of
them is holding.

Two prices keep the rest honest. The **stillness price** makes force on a moving
frame expensive (a sliding limb carries little load), and the **gap price** makes force
far from the scene expensive. The gap
price is measured **relative** to the nearest contact frame of the same step, so a
body that is entirely misregistered — every frame far from the wall — can still be
assigned force. That is exactly the case the distance gate cannot see.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch

from .friction import cone_generators, cone_normals
from .qmath import normalize_q

_GRAVITY_MAG = 9.81  # m/s²

#: ``frame_scene_signals`` reports ``finfo.max`` when a step's contact frames have no
#: scene point anywhere near them. That is **no evidence**, not "far from the wall", and
#: the two must never be confused — see :func:`solve_frame_forces`. Anything above this
#: is the sentinel.
NO_SCENE_M = 1e3


@dataclass(frozen=True)
class ForceKnobs:
    """Every switch of the force solve, in one place (defaults = the shipped recipe)."""

    #: Friction half-angle in degrees; μ = tan(half_angle_deg).
    half_angle_deg: float = 45.0
    #: Where a frame's contact axis comes from — see :func:`contact_axes`.
    cone_axis: str = "frame"
    #: kNN used to average scene normals (``"scene"`` axes only).
    normal_k: int = 16
    #: A scene-derived axis is unusable beyond this distance (``"scene"`` axes only).
    axis_max_dist_m: float = 1.0
    #: Frames further than this from the scene are not candidates at all (inf = no cap).
    candidate_max_dist_m: float = float("inf")
    #: Contact frames that keep a free (unconstrained) force instead of a cone.
    free_frames: tuple[str, ...] = ()
    #: Feed RNEA the finite-difference velocity/acceleration (False = static gravity only).
    use_motion: bool = True
    #: Characteristic lever arm (m) that converts the torque-carrying residual rows into
    #: force units, so :attr:`lambda_tau` is dimensionless. 1.0 = leave them in N·m.
    torque_length_m: float = 1.0
    #: Weight of the non-root ("effort") residual rows relative to the root rows.
    lambda_tau: float = 0.0
    #: Price the gap relative to the step's nearest frame (False = absolute distance).
    relative_gap: bool = True
    #: Stillness price: force on a frame drifting faster than this is expensive (m/s).
    stillness_vstar_mps: float = 0.2
    #: Gap price: force further than this from the scene is expensive (m).
    gap_dstar_m: float = 0.05
    #: Plain magnitude price (inside the square, like the dynamics stage).
    mag_weight: float = 0.01
    #: Temporal force-smoothness weight (0 = off).
    smooth_weight: float = 0.0
    #: Projected-Adam iterations for the whole-sequence solve.
    iters: int = 2000


@dataclass(frozen=True)
class ForceSolution:
    """One person's solved contact forces and the residual they explain."""

    #: ``(T, 12, 3)`` world contact forces, newtons; zero off the candidate set.
    forces: np.ndarray
    #: ``(T, 12)`` force magnitudes, newtons.
    magnitude: np.ndarray
    #: ``(T, 12)`` bool — frames the solve was allowed to load.
    candidate: np.ndarray
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

    Everything here depends only on ``q`` and the body, so it is computed once and
    reused across solver settings. ``tau_motion`` uses finite-difference velocity
    and acceleration; ``tau_static`` pins both to zero (pure gravity).

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
        self.body_weight_n = float(model.values.body_inertias[..., 1:, 0].sum()) * _GRAVITY_MAG

        with torch.no_grad():
            velocity, acceleration = velocity_acceleration_from_trajectory(
                model, self.q_model, 1.0 / float(fps))
            zero = torch.zeros_like(velocity)
            #: ``(T, nv)`` no-contact generalized force, with and without motion.
            self.tau_motion = rnea(model, self.q_model, velocity, acceleration)
            self.tau_static = rnea(model, self.q_model, zero, zero)

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

    def tau_no_contact(self, use_motion: bool) -> torch.Tensor:
        """The ``(T, nv)`` no-contact generalized force under the chosen motion model."""
        return self.tau_motion if use_motion else self.tau_static


def contact_axes(points: torch.Tensor, normals: torch.Tensor, scene_pts: torch.Tensor,
                 scene_normals: torch.Tensor, *, mode: str, k: int, max_dist_m: float
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit contact axis per ``(step, contact frame)`` — the way a contact pushes the body.

    Three sources, the ablation axis of this module:

      * ``"frame"`` — the negated outward skin normal. The skin of a touching frame
        faces into the surface, so the surface pushes back the other way. Needs no
        scene at all, which is why it survives a misregistered body, and it is the
        only source that separates the two toe frames (their skin faces opposite
        ways, so their cones are opposite).
      * ``"scene"`` — the fused cloud's own normal, averaged over the ``k`` nearest
        scene points (:func:`~tools.human_optim.friction.cone_normals`). Physically
        grounded but blind to the toe pair: the two frames sit millimetres apart, so
        they read the same scene points and get the same axis.
      * ``"scene_signed"`` — the scene normal flipped into the hemisphere the skin
        faces. Grounded in the cloud *and* toe-specific.

    Returns ``(axes (T, 12, 3), valid (T, 12))``. ``"frame"`` axes are always valid;
    scene axes are invalid where no coherent scene normal is within ``max_dist_m``.
    """
    if mode == "frame":
        return -normals, torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
    scene_axis, valid = cone_normals(points, scene_pts, scene_normals, k, max_dist_m)
    if mode == "scene":
        return scene_axis, valid
    if mode == "scene_signed":
        agreement = (scene_axis * -normals).sum(-1, keepdim=True)
        return scene_axis * torch.where(agreement < 0, -1.0, 1.0), valid
    raise ValueError(f"unknown cone axis mode {mode!r}")


def solve_frame_forces(dyn: FrameDynamics, *, valid: np.ndarray, axes: torch.Tensor,
                       axes_valid: torch.Tensor, drift_m: torch.Tensor,
                       drift_window_s: float, dist_m: torch.Tensor,
                       knobs: ForceKnobs) -> ForceSolution:
    """Solve the whole sequence's contact forces at once → :class:`ForceSolution`.

    :param valid: ``(T,)`` bool — steps where this person is tracked.
    :param axes: ``(T, 12, 3)`` contact axes from :func:`contact_axes`.
    :param axes_valid: ``(T, 12)`` bool — where that axis is usable.
    :param drift_m: ``(T, 12)`` net window drift of each contact frame, metres. This is
        the *pre-enforcement* drift (``drift_pre``): motion measured before any contact
        term acted, so the stillness price cannot be manufactured by the optimizer.
    :param drift_window_s: the window ``drift_m`` was measured over — drift/window is
        the frame's speed for the stillness price.
    :param dist_m: ``(T, 12)`` distance to the nearest scene point, metres. Values above
        :data:`NO_SCENE_M` are the "no scene point anywhere near" sentinel.
    """
    device = dyn.points.device
    num_steps, num_frames = dyn.points.shape[:2]
    # Everything is solved in units of body weight: forces are fractions of it and the
    # whole objective is divided by its square. That is a pure rescaling — it moves no
    # optimum, every term scales the same way — but it turns a problem whose numbers run
    # to 10⁵ N² into an O(1) one, which is the difference between L-BFGS converging and
    # its line search giving up on the first step (measured: it did, silently).
    weight = dyn.body_weight_n
    tau_nc = dyn.tau_no_contact(knobs.use_motion) / weight
    valid_t = torch.as_tensor(np.asarray(valid, bool), device=device)

    # "No scene point near this frame" is missing evidence, not distance. Treating the
    # sentinel as a number is exactly the bug that made the relative and absolute gap
    # ablations measure the encoding instead of the pricing: relative mode subtracted
    # sentinel−sentinel = 0 and handed those steps FREE force, absolute mode clamped them
    # to the 1 m cap and blocked them. One policy, applied before any pricing: a frame with
    # no scene evidence is not a candidate. Everywhere else in this pipeline a hole in the
    # cloud reads as *unknown*, and unknown never manufactures a contact.
    known = torch.isfinite(dist_m) & (dist_m < NO_SCENE_M)
    candidate = axes_valid & valid_t[:, None] & known
    if math.isfinite(knobs.candidate_max_dist_m):
        candidate = candidate & (dist_m < knobs.candidate_max_dist_m)
    free = torch.tensor([name in knobs.free_frames for name in dyn.frame_names], device=device)

    # Cone forces: four pyramid edges per frame, coefficients constrained nonnegative, so
    # the force cannot leave the cone and there is nothing to tune. The coefficients ARE
    # the variables — no reparameterization. The objective is a convex quadratic in them
    # over the nonnegative orthant, so projected gradient reaches the global optimum and
    # exactly zero force is attainable. (An earlier softplus reparameterization made zero
    # unreachable at finite parameters and the optimized problem nonconvex, for nothing.)
    generators = cone_generators(axes, math.tan(math.radians(knobs.half_angle_deg))).detach()
    coefficients = torch.zeros((num_steps, num_frames, 4), device=device, requires_grad=True)
    free_force = torch.zeros((num_steps, num_frames, 3), device=device, requires_grad=True)
    active = (candidate & ~free).to(torch.float32)[..., None]
    active_free = (candidate & free).to(torch.float32)[..., None]

    speed = drift_m / float(drift_window_s)
    gap = torch.where(known, dist_m, torch.zeros_like(dist_m))
    if knobs.relative_gap:
        # Relative pricing is the floater rescue: with every frame far from the wall,
        # absolute pricing makes all force so expensive that the solve leaves gravity
        # unbalanced and nothing crosses the threshold — on exactly the bodies the force
        # witness exists to rescue. Relative pricing keeps the price's real job (prefer
        # the nearer frame) while the weight still has to land somewhere. The datum is the
        # nearest CANDIDATE, so a frame with no scene evidence cannot set it.
        near = torch.where(candidate, gap, torch.full_like(gap, float("inf"))).amin(dim=1)
        gap = gap - torch.where(torch.isfinite(near), near, torch.zeros_like(near))[:, None]
    gap = gap.clamp(0.0, 1.0)

    # Torque-carrying residual rows divided by a characteristic lever arm, so every
    # residual row is in force units and lambda_tau is a dimensionless ratio.
    row_scale = torch.ones(dyn.jacobians.shape[-1], device=device)
    row_scale[3:] = 1.0 / float(knobs.torque_length_m)

    def compose() -> torch.Tensor:
        """Contact force per ``(step, frame)``, in units of body weight."""
        cone_force = torch.einsum("tfg,tfgd->tfd", coefficients, generators)
        return active * cone_force + active_free * free_force

    def loss_of(force: torch.Tensor) -> torch.Tensor:
        residual = (tau_nc - torch.einsum("tfkn,tfk->tn", dyn.jacobians, force)
                    ) * valid_t[:, None] * row_scale
        loss = 0.5 * residual[:, :6].pow(2).sum()
        if knobs.lambda_tau:
            loss = loss + 0.5 * knobs.lambda_tau * residual[:, 6:].pow(2).sum()
        squared = force.pow(2).sum(-1)                     # |f|², exactly quadratic
        loss = loss + 0.5 * knobs.mag_weight**2 * squared.sum()
        loss = loss + 0.5 * ((speed / knobs.stillness_vstar_mps).pow(2) * squared).sum()
        loss = loss + 0.5 * ((gap / knobs.gap_dstar_m).pow(2) * squared).sum()
        if knobs.smooth_weight:
            loss = loss + 0.5 * knobs.smooth_weight * (force[1:] - force[:-1]).pow(2).sum()
        return loss

    # Projected Adam with a decaying step. The decay is what makes the KKT check
    # meaningful: at a fixed step size Adam orbits the optimum and the gradient never
    # settles, so "converged" could not be distinguished from "still moving".
    parameters = [coefficients, free_force]
    optimizer = torch.optim.Adam(parameters, lr=0.05)
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
        before = dyn.tau_no_contact(knobs.use_motion)
        after = before - torch.einsum("tfkn,tfk->tn", dyn.jacobians, force)
    to_numpy = lambda t: t.detach().cpu().numpy().astype(np.float32)  # noqa: E731
    if not np.isfinite(to_numpy(force)).all():
        raise FloatingPointError("force solve produced non-finite forces")
    return ForceSolution(
        forces=to_numpy(force),
        magnitude=to_numpy(force.norm(dim=-1)),
        candidate=candidate.cpu().numpy(),
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
    _, dist_m, _, _ = frame_scene_signals(
        dynamics.points, dynamics.normals, scene_pts, scene_normals, scene_conf,
        prune_margin_m=SEARCH_MARGIN_M)
    axes, axes_valid = contact_axes(
        dynamics.points, dynamics.normals, scene_pts, scene_normals,
        mode=knobs.cone_axis, k=knobs.normal_k, max_dist_m=knobs.axis_max_dist_m)
    return solve_frame_forces(
        dynamics, valid=valid, axes=axes, axes_valid=axes_valid, drift_m=drift_pre,
        drift_window_s=drift_window_s, dist_m=dist_m, knobs=knobs)


def min_span(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Drop ``True`` runs shorter than ``min_frames`` in each column of an ``(T, C)`` bool array."""
    out = np.array(mask, dtype=bool, copy=True)
    for column in range(out.shape[1]):
        run = out[:, column]
        edges = np.flatnonzero(np.diff(np.concatenate([[0], run.view(np.int8), [0]])))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start < min_frames:
                run[start:stop] = False
    return out


def resolve_toe_conflicts(labels: np.ndarray, magnitude: np.ndarray, *, frame_names,
                          min_frames: int = 3) -> np.ndarray:
    """Force at most one toe frame per foot per step onto ``labels (T, 12)``.

    The geometric detector already keeps a single toe, and so does :func:`force_labels` —
    but they can pick *different* ones, and unioning the two then puts both toes of a foot
    in contact at the same step, which is exactly what the physics forbids. The union is
    resolved here, and **the force decides**: the toe carrying the larger solved magnitude
    at that step survives (exact ties keep ``toe_under``, so the result never depends on
    evaluation order).

    This is the one place a geometric label can be *removed*. Exclusivity outranks the
    add-only rule: "at most one toe per foot" is a statement about the world, while
    add-only is only a policy about how two readouts combine.

    Dropping a step can split a surviving toe's span in two, so the four toe columns get a
    final minimum-duration pass. Pairs are resolved by NAME (:func:`toe_pair_slots`) — the
    stored frame order is not the obvious one.
    """
    from .contacts import toe_pair_slots

    out = np.array(labels, dtype=bool, copy=True)
    pairs = toe_pair_slots(frame_names)
    for under, above in pairs:
        both = out[:, under] & out[:, above]
        keep_under = magnitude[:, under] >= magnitude[:, above]
        out[:, above] &= ~(both & keep_under)
        out[:, under] &= ~(both & ~keep_under)
    toe_columns = sorted({column for pair in pairs for column in pair})
    out[:, toe_columns] = min_span(out[:, toe_columns], min_frames)
    return out


def force_labels(magnitude: np.ndarray, *, frame_names, valid: np.ndarray,
                 threshold_n: float, min_frames: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Contact labels from solved force magnitudes ``(T, 12)`` → ``(labels, toe_split)``.

    A frame is in contact when it carries more than ``threshold_n`` newtons. Then the
    one physical assumption: **at most one toe frame per foot holds at a time**, so
    where both fire the larger force wins (exact ties keep ``toe_under``, so the result
    never depends on evaluation order). Runs shorter than ``min_frames`` are dropped
    last, after the loser is already out — the same order the geometric detector uses.

    ``toe_split`` marks the steps where both toes of a foot crossed the threshold and
    the force had to choose; it is a pure diagnostic.
    """
    from .contacts import toe_pair_slots

    fires = (magnitude > threshold_n) & np.asarray(valid, bool)[:, None]
    toe_split = np.zeros_like(fires)
    for under, above in toe_pair_slots(frame_names):
        both = fires[:, under] & fires[:, above]
        toe_split[:, under] |= both
        toe_split[:, above] |= both
        keep_under = magnitude[:, under] >= magnitude[:, above]
        fires[:, above] &= ~(both & keep_under)
        fires[:, under] &= ~(both & ~keep_under)
    return min_span(fires, min_frames), toe_split
