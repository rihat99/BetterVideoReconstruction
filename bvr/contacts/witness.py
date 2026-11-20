"""The force witness of contact: which contact frames must carry load for the motion to be possible.

Inverse dynamics of the fixed trajectory with nothing touching the body leaves, on the six rows of
the root, the wrench that contact must have supplied (:func:`bvr.terms.dynamics.root_balance`). The
witness asks which forces at the contact frames explain it, for all frames of the clip at once.

Friction cones are what give the answer a meaning. A free force at any one limb can absorb almost
any residual, which says nothing about which limb touched. Here a frame's force is a non-negative
combination of the edges of its friction pyramid, so it can explain only the part of the residual
its skin is oriented to explain; the two sides of a foot's toes face opposite ways, and the solve
decides between them. Two prices keep the rest honest: force on a moving frame is expensive (a
sliding limb carries little), and so is force far from the scene. The distance is priced relative
to the nearest candidate frame of the same instant, so a body that is misplaced as a whole, with
every frame far from the wall, can still be given force. That is the case geometry cannot see.

The problem is convex: a quadratic in coefficients that are bounded below by zero, people as the
batch. Forces are solved in body weights, which keeps the numbers near one.
"""
from __future__ import annotations

from typing import NamedTuple

import better_human as bh
import torch
from better_robot import optim

from bvr import body
from bvr.terms import dynamics, motion


class Witness(NamedTuple):
    """The solved forces of the people of a clip and the root residual they explain."""

    forces: torch.Tensor      # (P, T, F, 3) world, newtons; zero where a frame is no candidate
    before: torch.Tensor      # (P, T) root residual force with no contact (N)
    after: torch.Tensor       # (P, T) and with the solved forces (N)
    heard: torch.Tensor       # (P, T) bool: the frames whose inverse dynamics count; elsewhere nothing was solved


def solve(state: bh.HumanState, tracked: torch.Tensor, drift: torch.Tensor, distance: torch.Tensor,
          time: motion.Time, *, window_s: float, half_angle_deg: float, gap_dstar_m: float,
          stillness_vstar_mps: float, magnitude_weight: float, iterations: int, learning_rate: float,
          learning_rate_final: float) -> Witness:
    """Solve the contact forces of a fixed trajectory, every person and frame at once.

    :param state: the posed bodies ``(P, T)``, under the clip's gravity (:func:`bvr.terms.dynamics.under_gravity`).
    :param tracked: ``(P, T)`` bool.
    :param drift: ``(P, T, F)`` metres across a window of ``window_s``, measured before any contact term acted.
    :param distance: ``(P, T, F)`` metres to the nearest scene point; infinite where the scene is nowhere near.
    :param gap_dstar_m: force this much farther from the scene than the nearest candidate doubles in price,
    :param stillness_vstar_mps: as does force on a frame drifting this fast;
    :param magnitude_weight: the plain price of force, inside the square.

    A contact frame is a candidate where its inverse dynamics count and both its distance and its drift were
    measured: missing evidence reads as unknown, and unknown never makes a contact. The solve is Adam with the
    bound enforced after every step, its learning rate decaying from ``learning_rate`` to ``learning_rate_final``.
    """
    weight = dynamics.weight(state.identity)[:, None, None]
    free, jacobians = dynamics.root_balance(state, time)
    heard = dynamics.stencil_rows(time, tracked)
    candidate = heard[..., None] & distance.isfinite() & drift.isfinite()
    edges = dynamics.cone_edges(body.frame_normals(body.contact_frames(state)), half_angle_deg)
    edges = edges * candidate[..., None, None]
    gap = distance - distance.masked_fill(~candidate, torch.inf).amin(dim=2, keepdim=True)
    gap = torch.where(candidate, gap, 0.0).clamp(0.0, 1.0)
    price = (magnitude_weight**2 + (drift.nan_to_num() / (window_s * stillness_vstar_mps)).square()
             + (gap / gap_dstar_m).square())
    # Torque rows over a lever arm are forces, so the six root rows add up as one wrench in one unit.
    lever = free.new_tensor([1.0, 1.0, 1.0, *3 * [1.0 / dynamics.LEVER]])
    needed, pull = free / weight * heard[..., None] * lever, jacobians * heard[..., None, None, None] * lever

    def loss(coefficients: torch.Tensor) -> torch.Tensor:
        force = torch.einsum("ptfg,ptfgd->ptfd", coefficients, edges)
        left = needed - torch.einsum("ptfdr,ptfd->ptr", pull, force)
        return 0.5 * left.square().sum(dim=(1, 2)) + 0.5 * (price * force.square().sum(dim=-1)).sum(dim=(1, 2))

    zero = free.new_zeros(())
    problem = optim.Problem({"coefficients": free.new_zeros(*candidate.shape, 4)},
                            bounds={"coefficients": (zero, zero + torch.inf)})
    problem.scalar("witness", loss)
    solver = optim.TorchOptim(
        lambda parameters: torch.optim.Adam(parameters, lr=learning_rate),
        scheduler=lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=(learning_rate_final / learning_rate) ** (1.0 / iterations)))
    solved = optim.run(problem, [optim.Phase(solver, iterations, stop_every=0)]).x["coefficients"]
    forces = torch.einsum("ptfg,ptfgd->ptfd", solved, edges) * weight[..., None]
    left = free - torch.einsum("ptfdr,ptfd->ptr", jacobians, forces)
    return Witness(forces, free[..., :3].norm(dim=-1), left[..., :3].norm(dim=-1), heard)
