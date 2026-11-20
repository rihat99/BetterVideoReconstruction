"""Dynamics: the forces a motion needs from its contacts, and the terms that make pose and forces agree.

Inverse dynamics gives the generalised force a body needs to move the way it does::

    torques = M(q) a + b(q, v) + g(q) - sum over contact frames of J^T f

The first six rows belong to the root, which has no motor: what is left there once the contact
forces ``f`` are counted is the root residual, zero for a motion its contacts explain. The other
rows are the joint torques. Velocity and acceleration are central differences over the stride of
the frame-rate rule (:mod:`bvr.terms.motion`): row ``t`` reads frames ``t - h``, ``t`` and
``t + h`` and counts only where those frames, and the frames between them, are tracked.

A contact frame carries force only where its label says it touches, and only the way its skin
faces: the force is a non-negative combination of the four edges of a friction pyramid about the
negated skin normal (:func:`cone_edges`). Skin pushes; it does not pull. That is what stops one
limb from explaining another's load.

Forces are world vectors in newtons, ``(P, T, F, 3)``. Every term is divided by the body weight
squared and by the number of rows that count, so a weight means the same for every body and
every clip length. Tensors carry people first and time second; a term returns one loss per
person, ``(P,)``.
"""
from __future__ import annotations

import dataclasses
import math
from functools import partial

import better_human as bh
import torch
from better_robot import Model, kinematics, optim
from better_robot.dynamics import inverse_dynamics
from torch.nn.functional import normalize, pad, softplus

from bvr.terms import Term, motion

#: Standard gravity (m/s^2).
GRAVITY = 9.81

#: The lever arm (m) that puts a root torque beside a root force: the torque over it is a force.
LEVER = 0.2

#: The terms :func:`register` adds. A phase that solves the forces of a fixed pose weights only these.
TERMS = (
    "root_force", "root_torque", "force_magnitude", "force_rate_magnitude", "force_rate_direction",
    "force_acceleration_magnitude", "force_acceleration_direction", "force_jerk_magnitude", "force_jerk_direction",
    "torque_magnitude", "torque_rate", "torque_acceleration")


def under_gravity(identity: bh.Identity, down: torch.Tensor) -> bh.Identity:
    """``identity`` in a world where gravity pulls along ``down (3,)``, a direction of any length."""
    gravity = torch.cat((GRAVITY * normalize(down, dim=0), down.new_zeros(3)))
    return dataclasses.replace(identity, model=identity.model.with_values(gravity=gravity))


def weight(identity: bh.Identity) -> torch.Tensor:
    """The body weight ``(P,)`` in newtons of each person of ``identity`` (:func:`bvr.body.bake`)."""
    return GRAVITY * identity.model.body_inertias[:, 0, :, 0].sum(dim=-1)


def joint_weights(model: Model, multipliers: dict[str, float]) -> torch.Tensor:
    """What a unit of torque costs at each joint coordinate, ``(nv - 6,)``, from ``[torque_multipliers]``.

    A joint reads the entry of its group, its name without side and number: ``left_hip`` is ``hip``, ``spine2`` is
    ``spine``, ``right_index3`` is ``index``. A weak joint costs more, so a load that an arm or a leg could carry
    lands on the leg.
    """
    groups = [name.removeprefix("left_").removeprefix("right_").rstrip("0123456789") for name in model.joint_names]
    costs = [multipliers[group] for group, width in zip(groups[2:], model.nvs[2:]) for _ in range(width)]
    return torch.tensor(costs, dtype=model.dtype, device=model.device)


def cone_edges(normals: torch.Tensor, half_angle_deg: float) -> torch.Tensor:
    """Unit edges ``(..., 4, 3)`` of the friction pyramid of skin whose outward normals are ``normals (..., 3)``.

    The pyramid opens about the negated normal, the way a surface pushes back on the skin that faces it, with its
    edges ``half_angle_deg`` off that axis. A non-negative combination of the edges stays inside the friction cone.
    """
    axis = -normals
    seed = torch.where(axis[..., :1].abs() > 0.9, axis.new_tensor([0.0, 1.0, 0.0]), axis.new_tensor([1.0, 0.0, 0.0]))
    first = normalize(torch.linalg.cross(axis, seed), dim=-1)
    second = torch.linalg.cross(axis, first)
    slope = math.tan(math.radians(half_angle_deg))
    edges = torch.stack((axis + slope * first, axis - slope * first, axis + slope * second, axis - slope * second), -2)
    return normalize(edges, dim=-1)


def cone_forces(activation: torch.Tensor, edges: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Cached value: world forces ``(P, T, F, 3)`` in newtons from ``activation (P, T, F, 4)`` in body weights.

    A force is ``softplus(weight * activation)`` newtons along each of the four ``edges (P, T, F, 4, 3)`` of its
    pyramid, with ``weight (P,)`` the body weights: it cannot leave the pyramid, it is the activation itself above
    a newton or so, and it has a gradient everywhere, which a coefficient clamped at zero would not. Where a
    contact frame may carry no force its edges are zero. The activations are in body weights so that they are of
    the size of the pose's coordinates: one line search then steps both, and its first step is not lost in
    rounding.
    """
    return torch.einsum("ptfg,ptfgd->ptfd", softplus(weight[:, None, None, None] * activation), edges)


def stencil_rows(time: motion.Time, tracked: torch.Tensor) -> torch.Tensor:
    """The frames whose inverse dynamics count, ``(P, T)`` bool: ``t - h`` to ``t + h`` are all ``tracked (P, T)``.

    A frame beyond an end of the clip counts as tracked. The first and last ``h`` frames have no stencil, and
    :func:`rates` takes the body at rest there: the balance of a body at rest still says which contacts hold it.
    A frame beside a tracking gap does not count: what it would be differenced against is a held pose.
    """
    return time.rows(pad(tracked, (time.stride, time.stride), value=True), 2)


def rates(time: motion.Time, velocity: torch.Tensor, acceleration: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Velocity (per s) and acceleration (per s^2) of the pose at every frame, each ``(P, T, nv)``.

    ``velocity (P, T - h, nv)`` and ``acceleration (P, T - 2h, nv)`` are the strided derivatives of the pose per
    reference step (:func:`bvr.terms.motion.velocity` and its difference). Frame ``t`` gets the central
    differences over ``t - h``, ``t``, ``t + h``; the first and last ``h`` frames have no such stencil and get
    zeros, a body at rest.
    """
    stride, rate = time.stride, motion.REFERENCE_RATE
    ends = (0, 0, stride, stride)
    return pad(0.5 * rate * (velocity[:, :-stride] + velocity[:, stride:]), ends), pad(rate**2 * acceleration, ends)


def torques(body: bh.HumanState, velocity: torch.Tensor, acceleration: torch.Tensor, forces: torch.Tensor, *,
            time: motion.Time) -> torch.Tensor:
    """Cached value: inverse dynamics ``(P, T, nv)`` of a moving body under contact ``forces (P, T, F, 3)``.

    Columns 0 to 2 are the root residual force (N) and 3 to 5 the root residual torque (N m), both in the root's
    own frame; the rest are the joint torques. The gradient reaches the pose through the body, its derivatives
    and the lever arms of the forces.
    """
    v, a = rates(time, velocity, acceleration)
    wrenches = bh.terms.contact_wrenches(
        body.model, body.kinematics, forces=forces, frames=body.identity.human.contact_frames.names)
    return inverse_dynamics(body.model, body.kinematics, v=v, a=a, external_forces=wrenches)


@torch.no_grad()
def root_balance(state: bh.HumanState, time: motion.Time) -> tuple[torch.Tensor, torch.Tensor]:
    """The root rows of a fixed motion, split: without contact, ``(P, T, 6)``, and what each contact frame adds.

    At a fixed pose inverse dynamics is affine in the contact forces: with forces ``f (P, T, F, 3)`` the root rows
    are ``free - einsum("ptfdr,ptfd->ptr", jacobians, f)``, where ``jacobians (P, T, F, 3, 6)`` are the root
    columns of the contact frames' translation Jacobians. One product per evaluation instead of an inverse
    dynamics pass: the form a solve of forces alone reads.
    """
    model, names = state.model, state.identity.human.contact_frames.names
    velocity = motion.velocity(model, time)(state.q)
    v, a = rates(time, velocity, time.difference(velocity))
    free = inverse_dynamics(model, state.kinematics, v=v, a=a)[..., :6]
    jacobians = kinematics.point_jacobian(model, state.kinematics, frames=names, points=free.new_zeros(len(names), 3))
    return free, jacobians[..., :6]


def register(problem: optim.Problem, identity: bh.Identity, *, edges: torch.Tensor, held: torch.Tensor,
             tracked: torch.Tensor, time: motion.Time, multipliers: dict[str, float]) -> None:
    """Put the physics on a kinematic problem whose variables include ``activation (P, T, F, 4)``.

    :param identity: the bodies, under the clip's gravity (:func:`under_gravity`).
    :param held: ``(P, T, F)`` bool, where a contact frame may carry force: labelled, on a tracked frame.
    :param edges: ``(P, T, F, 4, 3)`` the friction pyramids (:func:`cone_edges`), zero where not ``held``. They are
        fixed for the solve: a pyramid that turned with the pose would be a target the pose could rotate to make
        force cheap.
    :param tracked: ``(P, T)`` bool.
    :param multipliers: ``[torque_multipliers]``.

    Cached values: ``forces`` (:func:`cone_forces`) and ``torques`` (:func:`torques`). The terms are
    :data:`TERMS`, each off until a phase weights it: the root residual, then force and joint torque and their
    strided derivatives, the force's split along and across its mean direction over the window. A person without
    a contact has no force to hold the body up with, and the root rows alone would pull the fit into free fall:
    no row counts for such a person, so every term here is zero and the fit stays kinematic.
    """
    rows = stencil_rows(time, tracked) & held.any(dim=(1, 2))[:, None]
    newtons = weight(identity)
    scale = 0.5 / (rows.sum(dim=1).clamp_min(1) * newtons.square())
    costs = joint_weights(identity.model, multipliers)
    problem.cached(partial(cone_forces, edges=edges, weight=newtons), name="forces", reads=("activation",))
    problem.cached(partial(torques, time=time), name="torques", reads=("body", "velocity", "acceleration", "forces"))

    term = partial(problem.scalar, weight=0.0)
    term("root_force", _squares(scale, rows[..., None], slice(0, 3)), reads=("torques",))
    term("root_torque", _squares(scale, rows[..., None], slice(3, 6)), reads=("torques",))
    term("force_magnitude", _squares(scale, 1.0), reads=("forces",))
    for order, name in enumerate(("rate", "acceleration", "jerk"), start=1):
        problem.cached(partial(_split, time=time, order=order), name=f"force_{name}", reads=("forces",))
        steady = time.rows(held, order)[..., None]
        term(f"force_{name}_magnitude", _squares(scale, steady, 0), reads=(f"force_{name}",))
        term(f"force_{name}_direction", _squares(scale, steady, 1), reads=(f"force_{name}",))
    term("torque_magnitude", _squares(scale, rows[..., None] * costs, slice(6, None)), reads=("torques",))
    for order, (lower, name) in enumerate((("torques", "torque_rate"), ("torque_rate", "torque_acceleration")), 1):
        problem.cached(time.difference, name=name, reads=(lower,))
        term(name, _squares(scale, time.rows(rows, order)[..., None] * costs, slice(6, None)), reads=(name,))


def _split(forces: torch.Tensor, *, time: motion.Time, order: int) -> torch.Tensor:
    """The strided derivative of ``forces (P, T, F, 3)`` of ``order``, along and across the force's mean direction.

    Returns ``(P, T - order h, F, 3, 2)``: the part of the derivative that changes the force's size, and the part
    that turns it, each a vector. The mean is over the frames the derivative reads.
    """
    change, stride, frames = forces, time.stride, forces.shape[1] - order * time.stride
    for _ in range(order):
        change = time.difference(change)
    direction = normalize(sum(forces[:, step * stride:step * stride + frames] for step in range(order + 1)), dim=-1)
    along = (change * direction).sum(dim=-1, keepdim=True) * direction
    return torch.stack((along, change - along), dim=-1)


def _squares(scale: torch.Tensor, counted: torch.Tensor | float, columns: slice | int = slice(None)) -> Term:
    """Term: ``scale (P,)`` times the sum of squares of the last axis's ``columns`` of a value ``(P, ...)``, over
    the entries where ``counted`` (which broadcasts against them) holds."""
    def loss(value: torch.Tensor) -> torch.Tensor:
        return scale * (value[..., columns].square() * counted).flatten(1).sum(dim=1)
    return loss
