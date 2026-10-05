"""The frozen-pose force solve behind ``scripts/label_bedlam2.py --step forces``.

Two things must hold. The affine form of the RNEA that the solve runs on — the
unloaded torques minus each contact force pulled back through its frame's
Jacobian — has to be the same physics as feeding the forces to RNEA as parent-joint
wrenches, or the fast solve answers a different question from ``kindyn_1``. And a
body standing still on its labelled feet must come out carrying its own weight.
"""

from __future__ import annotations

import numpy as np
import torch

from tools.bedlam_forces import person_forces
from tools.body import load_body
from tools.human_optim.config import load_config
from tools.human_optim.kindyn import build_dynamics_spec
from tools.human_optim.qmath import normalize_q

_GRAVITY = np.array([0.0, -1.0, 0.0], np.float32)     # the rest pose stands y-up
_FPS = 30.0
_SOLES = ("heel_left", "heel_right", "ball_inner_left", "ball_outer_left",
          "ball_inner_right", "ball_outer_right", "big_toe_left", "big_toe_right")


def _standing(body, frames: int = 8) -> np.ndarray:
    """A rest-pose trajectory ``(N, nq)`` — the body stands still on its feet."""
    q = np.zeros((frames, body.nq), np.float32)
    q[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    q[:, 7:] = np.tile([0.0, 0.0, 0.0, 1.0], body.num_joints - 1)
    return q


def _sole_labels(body, frames: int) -> np.ndarray:
    contact = np.zeros((frames, len(body.contact_frame_names)), bool)
    for name in _SOLES:
        contact[:, body.contact_frame_names.index(name)] = True
    return contact


def _solve(body, q, contact):
    cfg = load_config("configs/human_optimization.toml")
    return person_forces(
        body, np.zeros(10, np.float32), q, contact, gravity_world=_GRAVITY, fps=_FPS,
        device="cpu", weights={**cfg.weights_for("kindyn_1"), "force_edge": 3000.0},
        dyn_opt=cfg.knobs["dynamics_optimizer"], torque_table=cfg.knobs["torque"]["multipliers"],
        half_angle_deg=45.0, edge_speed_m_s=0.8)


def test_a_standing_body_carries_its_weight_on_its_feet():
    body = load_body(device="cpu")
    q = _standing(body)
    out = _solve(body, q, _sole_labels(body, len(q)))

    weight = float(out["total_mass"]) * 9.81
    total = out["frame_forces"].sum(axis=1)                        # (N, 3) world newtons
    assert np.allclose(total[:, 1], weight, rtol=0.05)             # pushes up by the body weight
    assert np.abs(total[:, [0, 2]]).max() < 0.05 * weight          # nothing sideways
    assert float(out["base_wrench_after"]) < 0.05 * weight         # the residual is explained
    unlabelled = ~_sole_labels(body, len(q))
    assert not out["frame_forces"][unlabelled].any()               # only labelled frames push


def test_without_contact_there_is_no_force_and_the_body_is_unsupported():
    body = load_body(device="cpu")
    q = _standing(body)
    out = _solve(body, q, np.zeros((len(q), len(body.contact_frame_names)), bool))

    assert not out["frame_forces"].any()
    assert float(out["base_wrench_after"]) > 0.9 * float(out["total_mass"]) * 9.81


def test_the_frozen_affine_dynamics_are_the_rnea_with_external_wrenches():
    """``tau_free − Σ J_cᵀ f_c`` must equal RNEA fed the forces as parent-joint wrenches.

    A bent pose, so every lever arm is non-trivial, and forces on every contact frame.
    """
    body = load_body(device="cpu")
    frames = 4
    q = torch.as_tensor(_standing(body, frames))
    generator = torch.Generator().manual_seed(0)
    half_angle = torch.tensor(0.2)                                 # 23° about a random axis
    for joint in range(1, body.num_joints):
        axis = torch.randn(3, generator=generator)
        axis = axis / axis.norm()
        q[:, 7 + 4 * (joint - 1): 7 + 4 * joint] = torch.cat(
            [axis * half_angle.sin(), half_angle.cos()[None]])
    cfg = load_config("configs/human_optimization.toml")
    labels = torch.ones(frames, len(body.contact_frame_names), dtype=torch.int8)
    spec = build_dynamics_spec(
        body, torch.zeros(10), labels, np.ones(frames, bool), _GRAVITY,
        weights=cfg.weights_for("kindyn_1"), dyn_opt=cfg.knobs["dynamics_optimizer"],
        fps=_FPS, device="cpu", q_world=q, cones=False)
    qn = normalize_q(q, body.num_joints)
    force = 300.0 * torch.randn(frames, spec.C, 3, generator=generator)

    tau_rnea, f_eff = spec.forward(qn, force)
    tau_free, jac = spec.frozen_dynamics(qn)
    tau_affine = tau_free - torch.einsum("tcdn,tcd->tn", jac, f_eff)

    assert torch.allclose(tau_affine, tau_rnea, atol=1e-2, rtol=1e-4)
    assert float((tau_rnea - tau_free).abs().max()) > 100.0      # the forces really act
