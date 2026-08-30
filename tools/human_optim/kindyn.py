"""The ``kindyn_1`` stage: joint kinematics + dynamics refinement.

The sequential chain fits the motion first (kinematics), then solves contact
forces on the FROZEN pose. ``kindyn_1`` instead optimizes the pose ``q`` **and**
the contact forces together, so the physics residual — the RNEA base wrench, the
6-DOF force/torque left on the un-actuated free-flyer root — can reshape the
motion instead of only being absorbed by forces.

Contact lives where the rest of the pipeline puts it: on the **12 contact
frames** (toes, heels, knees, palms, fingertips). A frame may carry a force only
on the video frames its labels say it touches, and that force is confined to a
45° friction pyramid about the frame's own **contact normal** — the negated
outward skin normal, the same axis the contact readout's force witness uses
(:mod:`tools.human_optim.forces`). A patch of skin can only push the way it
faces, which is what stops one limb from silently explaining another's load.

A world force at a contact frame is fed to RNEA as an external wrench on that
frame's parent joint: the force itself plus the moment ``r × f`` of its lever arm
from the joint origin. Both are recomputed from the *current* ``q`` each
iteration (with gradient), so moving the body changes the physics — that is the
whole point of solving pose and force together. Two frames sharing a parent joint
(the two toes of a foot) accumulate, they do not overwrite.

The physics half of the objective is packaged as a :class:`DynamicsSpec`, built
once per person. :func:`~tools.human_optim.optimizer.optimize_person` owns the
shared kinematic Problem and simply adds the force leaf, threads the RNEA forward
through its per-iteration state and registers the dynamics terms. The stage
entry point is :func:`run_kindyn`, driven by
``scripts/stages/human_optimization.py --stages kindyn_1``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tools.body import frame_normals
from tools.smplx_robot.dynamics import velocity_acceleration_from_trajectory

from . import paths
from .camera import world_from_cam
from .config import FPS_REF, HumanOptimConfig, fps_scaled_weights
from .force_confidence import force_confidence
from .friction import cone_generators
from .optimizer import PhaseSpec, optimize_person
from .qmath import normalize_q
from .stages import (
    _ALL_DOF, _body_context, _load_scene, _new_out, _append_classic,
    _person_visual_inputs, _save_body, annotation_frame_labels, load_frame_contacts,
    resolve_annotation_path,
)

_GRAVITY_MAG = 9.81  # m/s²

#: Binomial stencil of the k-th finite difference, over k+1 consecutive frames.
_DIFF_COEFFS = {1: (-1.0, 1.0), 2: (1.0, -2.0, 1.0), 3: (-1.0, 3.0, -3.0, 1.0)}


# ─────────────────────────── the physics half ────────────────────────────

@dataclass
class DynamicsSpec:
    """The force part of the ``kindyn_1`` objective for one person.

    Holds the pieces that stay fixed while ``q`` and the forces move: the RNEA
    ``model`` (gravity baked to the scene's world-down), which BetterRobot frame /
    joint rows the 12 contact frames occupy, the per-frame contact ``mask``, and the
    frozen friction pyramids. Its methods run the RNEA forward (recomputing
    ``v``/``a``, the contact points and their lever arms from the *current* ``q`` so
    gradient flows into the pose), define the dynamics loss terms, warm-start
    the forces at the initial pose, and package the result for the output npz.

    All dynamics loss weights are **linear** — the term value is ``0.5·mean_t Σ residual²``
    and the engine multiplies by the (kindyn-column) weight, exactly like every kinematic
    term.

    Every residual is divided by the body weight and averaged over the valid frames
    (:attr:`res_scale`). Without that the physics terms are a *sum* of newtons squared,
    so their pull on ``q`` grows with clip length and body size while every kinematic
    term stays a per-frame mean — the same objective would dominate a 900-frame clip and
    barely register on a 60-frame one. In body-weight units all terms are O(1) and
    one weight means one thing everywhere (the force witness solves in the same units,
    for the same reason).
    """

    model: object                 #: BetterRobot ``Model`` (scene-down gravity baked in)
    frame_ids: torch.Tensor       #: (12,) rows of the model's frame table
    parent_ids: torch.Tensor      #: (12,) model body id of each contact frame's parent joint
    parent_body: list[int]        #: (12,) the same parents as SMPLXMid joint indices
    frame_names: list[str]        #: (12,) contact-frame names, force-point order
    mask: torch.Tensor            #: (T, 12) float 0/1 — a frame carries force only where labelled
    fpair: torch.Tensor           #: (T-1, 12) float — smoothness mask across persisting contacts
    valid_pair: torch.Tensor      #: (T-1,) float — both-ends-valid frame steps
    dt: float                     #: frame interval, seconds
    weights: dict                 #: kindyn dynamics loss weights (linear)
    dyn_opt: dict                 #: L-BFGS knobs for the force warm-start
    gravity_world: np.ndarray     #: (3,) scene world-down (for the output npz)
    valid: np.ndarray             #: (T,) bool per-frame validity
    total_mass: float             #: body mass, kg (baked inertia sum)
    res_scale: float = 1.0        #: 1 / (counted frames × body weight²) — see the class docstring
    spacing: int = 1              #: v/a stencil step (frames) — round(fps/30), keeps the accel noise at the 30 fps reference
    dyn_row: torch.Tensor | None = None  #: (T,) float 0/1 — frames whose ±spacing stencil is fully tracked; only these rows carry the base-wrench residual (defaults to ``valid``)
    cone_axis: torch.Tensor | None = None   #: (T, 12, 3) frozen contact normal per frame
    cone_gen: torch.Tensor | None = None    #: (T, 12, 4, 3) frozen pyramid edge directions
    mu: float = 1.0                         #: tan(cone half-angle) — the friction coefficient
    base_wrench_before: float = float("nan")  #: per-frame RMS ‖root residual‖ with no contact force
    base_wrench_warm: float = float("nan")    #: …after the frozen-pose force warm-start

    def __post_init__(self) -> None:
        # An occlusion gap holds the nearest tracked pose, so a v/a stencil that touches
        # a held frame reads the mid-gap jump as a huge phantom acceleration. Only rows
        # whose whole ±spacing stencil is tracked carry the base-wrench residual.
        if self.dyn_row is None:
            self.dyn_row = torch.as_tensor(np.asarray(self.valid, bool),
                                           dtype=self.mask.dtype, device=self.mask.device)
        self.dyn_pair = self.dyn_row[1:] * self.dyn_row[:-1]   # (T-1,) both-rows-counted steps
        # Persisting-contact masks for the higher-order force differences: an order-k
        # term may act only where the contact holds over its whole (k+1)-frame window.
        self.ftrip = self.fpair[1:] * self.fpair[:-1]          # (T-2, 12)
        self.fquad = self.ftrip[1:] * self.ftrip[:-1]          # (T-3, 12)

    def wrench_rms(self, tau: torch.Tensor) -> float:
        """Per-counted-frame RMS of the 6-row root residual — the stage's headline number.

        A root-mean-square over the ``dyn_row`` frames, not the whole trajectory's
        Frobenius norm: the latter grows with clip length, so two scenes could not be
        compared and the same reduction read differently on a long clip than a short
        one."""
        sq = (self.dyn_row[:, None] * tau[:, :6].pow(2)).sum()
        return float(sq.div(max(int(self.dyn_row.sum()), 1)).sqrt())

    @property
    def C(self) -> int:
        return len(self.frame_names)

    def zero_forces(self) -> torch.Tensor:
        """Initial free-force leaf ``(T, 12, 3)`` — zeros; the warm-start fills it."""
        return torch.zeros(self.mask.shape[0], self.C,
                           3, dtype=self.mask.dtype, device=self.mask.device)

    def zero_generators(self) -> torch.Tensor | None:
        """Initial pyramid-activation leaf ``(T, 12, 4)`` — zeros, or ``None`` with no cones."""
        if self.cone_gen is None:
            return None
        return torch.zeros(self.mask.shape[0], self.C, 4,
                           dtype=self.mask.dtype, device=self.mask.device)

    def compose_forces(self, f_free: torch.Tensor, gen_u: torch.Tensor | None) -> torch.Tensor:
        """World force ``(T, 12, 3)`` — inside the friction pyramid when cones are on.

        ``gen_u`` are raw activations; ``softplus`` makes them nonnegative (and unlike
        relu leaves no dead zero-gradient point), so the force is a nonnegative
        combination of the four frozen pyramid edges and cannot leave the cone. With
        cones off every frame keeps its free bilateral vector instead.
        """
        if gen_u is None or self.cone_gen is None:
            return f_free
        lam = torch.nn.functional.softplus(gen_u)                      # (T, 12, 4) ≥ 0
        return torch.einsum("tcg,tcgd->tcd", lam, self.cone_gen)       # (T, 12, 3) in the pyramid

    def cone_violation(self, f_world: torch.Tensor) -> torch.Tensor:
        """``relu(‖f_t‖ − μ·f_n)`` per (step, frame) against the TRUE cone — a readout.

        ``f_n = f·n̂`` is the signed normal component and ``f_t`` the tangential rest.
        Generator forces live in the pyramid inscribed in the cone, so this must read ~0;
        it is printed as an invariant check, never optimized."""
        f_n = (f_world * self.cone_axis).sum(-1)                       # (T, 12)
        f_t = f_world - f_n[..., None] * self.cone_axis
        return torch.relu(f_t.pow(2).sum(-1).add(1e-12).sqrt() - self.mu * f_n)

    def placements(self, qn: torch.Tensor):
        """``(joint_pose_world, contact-frame poses (T, 12, 7))`` at the current pose."""
        from better_robot.kinematics import forward_kinematics, update_frame_placements

        data = forward_kinematics(self.model, qn)
        update_frame_placements(self.model, data)
        return data.joint_pose_world, data.frame_pose_world[..., self.frame_ids, :]

    def build_fext(self, f_world: torch.Tensor, oMi: torch.Tensor, frames7: torch.Tensor):
        """``(fext (T, njoints, 6), f_eff (T, 12, 3))`` — frame forces as parent-joint wrenches.

        A world force ``f`` applied at a contact frame is, at that frame's parent joint,
        the wrench ``[f, r × f]`` with ``r`` the lever arm from the joint origin to the
        contact point. Both are expressed in the joint's local frame, which is the
        convention RNEA's ``fext`` expects. Contributions are **accumulated** — the two
        toe frames of a foot share a parent joint and both must land.
        """
        from better_robot.lie import so3

        f_eff = f_world * self.mask[..., None]                                 # (T, 12, 3)
        rot_local_world = so3.to_matrix(oMi[:, self.parent_ids, 3:7]).transpose(-1, -2)
        f_local = torch.einsum("tcij,tcj->tci", rot_local_world, f_eff)
        lever = frames7[..., :3] - oMi[:, self.parent_ids, :3]                 # (T, 12, 3) world
        lever_local = torch.einsum("tcij,tcj->tci", rot_local_world, lever)
        wrench = torch.cat([f_local, torch.linalg.cross(lever_local, f_local)], dim=-1)
        fext = torch.zeros((f_world.shape[0], self.model.njoints, 6),
                           dtype=f_world.dtype, device=f_world.device)
        index = self.parent_ids.view(1, self.C, 1).expand(f_world.shape[0], self.C, 6)
        return fext.scatter_add(dim=1, index=index, src=wrench), f_eff

    def forward(self, qn: torch.Tensor, f_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """RNEA base + joint torques from the current pose and forces.

        Recomputes ``v``/``a`` (central differences on the manifold), the contact-frame
        placements and their lever arms from ``qn`` **with gradient**, so the base-wrench
        residual pulls on ``q``. Returns ``(tau (T, nv), f_eff (T, 12, 3))`` where
        ``f_eff`` is the mask-gated world force.
        """
        from better_robot.dynamics import rnea

        v, a = velocity_acceleration_from_trajectory(self.model, qn, self.dt,
                                                     spacing=self.spacing)
        oMi, frames7 = self.placements(qn)
        fext, f_eff = self.build_fext(f_world, oMi, frames7)
        return rnea(self.model, qn, v, a, fext=fext), f_eff

    @staticmethod
    def force_split(f_world: torch.Tensor, order: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the ``order``-th force difference into magnitude vs direction parts.

        The k-th finite difference (binomial stencil over ``k+1`` consecutive
        frames) is decomposed about the mean direction ``u`` of the forces in its
        window: ``d_par`` is the ‖f‖-change part and ``d_perp`` the
        direction-change part. Order 1 is the adjacent-frame decomposition the
        stage always had; orders 2 and 3 restrain the force's acceleration and
        jerk the same way. Each ``(T-k, 12, 3)``.
        """
        num_frames = f_world.shape[0]
        window = [f_world[i:num_frames - order + i] for i in range(order + 1)]
        d = sum(c * w for c, w in zip(_DIFF_COEFFS[order], window))
        u = sum(window)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        d_par = (d * u).sum(-1, keepdim=True) * u
        return d_par, d - d_par

    def populate(self, st, f_free: torch.Tensor, gen_u: torch.Tensor | None) -> None:
        """Add the per-iteration lazy RNEA products to the shared :class:`State`.

        The state's ``"f_world"`` is the COMPOSED force (newtons) — the cone parametrization
        is applied once, here, so every dynamics term (smoothness included) sees forces and
        never raw pyramid activations."""
        st.put_lazy("f_world", lambda: self.compose_forces(f_free, gen_u))
        st.put_lazy("dyn", lambda: self.forward(st["qn"], st["f_world"]))
        st.put_lazy("dyn_split", lambda: self.force_split(st["f_world"]))
        st.put_lazy("dyn_split2", lambda: self.force_split(st["f_world"], 2))
        st.put_lazy("dyn_split3", lambda: self.force_split(st["f_world"], 3))

    def register(self, problem) -> None:
        """Register the dynamics terms (active only where their weight is).

        A person with no contact labels registers nothing: with zero force capacity
        the base-wrench term's only descent direction is accelerating the body at
        gravity — a freefall bias on exactly the person promised a pure kinematic
        re-fit."""
        if not bool(self.mask.any()):
            return
        half = 0.5 * self.res_scale

        def _base(st):
            tau = st["dyn"][0]
            sq = (self.dyn_row[:, None] * tau[:, :3].pow(2)).sum()
            return half * sq, {"bwN": sq.sqrt()}

        def _base_ang(st):
            tau = st["dyn"][0]
            sq = (self.dyn_row[:, None] * tau[:, 3:6].pow(2)).sum()
            return half * sq, {"bwNm": sq.sqrt()}

        def _fmag(st):
            f_eff = st["dyn"][1]
            return half * f_eff.pow(2).sum()

        def _smooth(split_key: str, part: int, mask: torch.Tensor):
            def term(st):
                return half * (mask[..., None] * st[split_key][part].pow(2)).sum()
            return term

        def _tsmooth(st):
            tau = st["dyn"][0]
            dtau = tau[1:, 6:] - tau[:-1, 6:]
            return half * (self.dyn_pair[:, None] * dtau.pow(2)).sum()

        problem.add_term("base_wrench", _base)
        problem.add_term("base_wrench_ang", _base_ang)
        problem.add_term("force_magnitude", _fmag)
        problem.add_term("force_smooth_mag", _smooth("dyn_split", 0, self.fpair))
        problem.add_term("force_smooth_dir", _smooth("dyn_split", 1, self.fpair))
        problem.add_term("force_accel_mag", _smooth("dyn_split2", 0, self.ftrip))
        problem.add_term("force_accel_dir", _smooth("dyn_split2", 1, self.ftrip))
        problem.add_term("force_jerk_mag", _smooth("dyn_split3", 0, self.fquad))
        problem.add_term("force_jerk_dir", _smooth("dyn_split3", 1, self.fquad))
        problem.add_term("torque_smooth", _tsmooth)

    def warmstart(self, qn_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Solve the forces at the fixed initial pose (the sequential dynamics solve).

        Forces start at zero, so the base-wrench gradient on ``q`` is meaningless
        (an unsupported body is in freefall). A short L-BFGS on the forces alone —
        pose detached — seeds them near their sequential optimum, so the joint
        phase starts from a physically-reasonable point for both ``q`` and ``f``.
        Both leaves are solved (free force AND pyramid activations) so the returned
        values seed the joint phase coherently. Returns ``(f_free (T, 12, 3),
        gen_u (T, 12, 4) or None)``, detached; the before/after base wrench lands
        on :attr:`base_wrench_before` / :attr:`base_wrench_warm`.
        """
        from better_robot.dynamics import rnea

        w = self.weights
        with torch.no_grad():
            qn = qn_init.detach()
            v, a = velocity_acceleration_from_trajectory(self.model, qn, self.dt,
                                                         spacing=self.spacing)
            oMi, frames7 = self.placements(qn)
            oMi, frames7 = oMi.detach(), frames7.detach()
            base0 = self.wrench_rms(rnea(self.model, qn, v, a))

        f = self.zero_forces().requires_grad_(True)
        gen_u = self.zero_generators()
        if gen_u is not None:
            gen_u.requires_grad_(True)
        opt = torch.optim.LBFGS(
            [f] if gen_u is None else [f, gen_u],
            lr=float(self.dyn_opt["lr"]), max_iter=int(self.dyn_opt["max_iter"]),
            history_size=int(self.dyn_opt["history_size"]),
            tolerance_grad=float(self.dyn_opt["tolerance_grad"]),
            tolerance_change=float(self.dyn_opt["tolerance_change"]), line_search_fn="strong_wolfe")

        half = 0.5 * self.res_scale

        smooth_orders = ((1, "force_smooth_mag", "force_smooth_dir", self.fpair),
                         (2, "force_accel_mag", "force_accel_dir", self.ftrip),
                         (3, "force_jerk_mag", "force_jerk_dir", self.fquad))

        def closure():
            opt.zero_grad()
            f_world = self.compose_forces(f, gen_u)
            fext, f_eff = self.build_fext(f_world, oMi, frames7)
            tau = rnea(self.model, qn, v, a, fext=fext)
            loss = (w["base_wrench"] * half * (self.dyn_row[:, None] * tau[:, :3].pow(2)).sum()
                    + w.get("base_wrench_ang", 0.0) * half
                    * (self.dyn_row[:, None] * tau[:, 3:6].pow(2)).sum()
                    + w["force_magnitude"] * half * f_eff.pow(2).sum())
            for order, key_mag, key_dir, mask in smooth_orders:
                if f.shape[0] > order and (w.get(key_mag, 0.0) or w.get(key_dir, 0.0)):
                    d_par, d_perp = self.force_split(f_world, order)
                    loss = loss + w.get(key_mag, 0.0) * half * (mask[..., None] * d_par.pow(2)).sum()
                    loss = loss + w.get(key_dir, 0.0) * half * (mask[..., None] * d_perp.pow(2)).sum()
            if f.shape[0] > 1:
                dtau = tau[1:, 6:] - tau[:-1, 6:]
                loss = loss + w["torque_smooth"] * half * (self.dyn_pair[:, None] * dtau.pow(2)).sum()
            loss.backward()
            return loss

        if bool(self.mask.any()):
            opt.step(closure)
        with torch.no_grad():
            f_world = self.compose_forces(f, gen_u)
            fext, _ = self.build_fext(f_world, oMi, frames7)
            base1 = self.wrench_rms(rnea(self.model, qn, v, a, fext=fext))
        self.base_wrench_before, self.base_wrench_warm = base0, base1
        print(f"    [kindyn warm-start] base wrench {base0:.1f}→{base1:.2f} N/frame RMS "
              f"(forces only, q frozen)")
        return f.detach(), None if gen_u is None else gen_u.detach()

    def extract(self, qn: torch.Tensor, f_free: torch.Tensor, gen_u: torch.Tensor | None,
                body) -> dict:
        """Package the solved forces + diagnostics for the output npz.

        Keeps the per-joint fields the viewer and the dynamics stage share — a contact
        frame's force and its labels land on the frame's PARENT joint (accumulated, so a
        foot carrying both toes sums) — and adds the frame-level arrays, which are the
        honest resolution of the solve.
        """
        J = body.num_joints
        with torch.no_grad():
            f_world = self.compose_forces(f_free, gen_u)                    # (T, 12, 3) newtons
            f_final = f_world * self.mask[..., None]
            tau, _ = self.forward(qn, f_world)                              # (T, nv)
            oMi, _ = self.placements(qn)
            base1 = self.wrench_rms(tau)

            idx_vs = [int(x) for x in self.model.idx_vs]
            nvs = [int(x) for x in self.model.nvs]
            N = qn.shape[0]
            torque_mag = np.zeros((N, J), np.float32)
            joints_world = np.zeros((N, J, 3), np.float32)
            force_world = np.zeros((N, J, 3), np.float32)
            for jb in range(J):
                mb = int(self.model.body_name_to_id[body.joint_names[jb]])
                n = nvs[mb]
                if n > 0:
                    torque_mag[:, jb] = tau[:, idx_vs[mb]:idx_vs[mb] + n].norm(dim=-1).cpu().numpy()
                joints_world[:, jb] = oMi[:, mb, :3].cpu().numpy()
            frame_force = f_final.cpu().numpy().astype(np.float32)          # (T, 12, 3)
            for c, jb in enumerate(self.parent_body):
                force_world[:, jb] += frame_force[:, c]                     # two toes share a foot

            inval = ~self.valid.astype(bool)
            # RNEA products are meaningless where the residual was not enforced (untracked
            # frames AND rows whose stencil straddles a gap); forces are real on any
            # tracked frame the labels cover.
            ungated = ~self.dyn_row.bool().cpu().numpy()
            torque_mag[ungated] = 0.0
            force_world[inval] = 0.0
            frame_force[inval] = 0.0
            base_wrench = tau[:, :6].cpu().numpy().astype(np.float32)
            base_wrench[ungated] = 0.0

            fnorm = f_final.norm(dim=-1)                                    # (T, 12)
            act = self.mask.bool().any(dim=1)
            mean_f = float(fnorm.sum(dim=-1)[act].mean()) if bool(act.any()) else 0.0
            fjit_mag = fjit_dir = tjit = 0.0
            if N > 1:
                df = f_final[1:] - f_final[:-1]
                u = f_final[1:] + f_final[:-1]
                u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                dperp = (df - (df * u).sum(-1, keepdim=True) * u).norm(dim=-1)
                dmag = (f_final[1:].norm(dim=-1) - f_final[:-1].norm(dim=-1)).abs()
                fp = self.fpair.bool()
                if bool(fp.any()):
                    fjit_mag, fjit_dir = float(dmag[fp].mean()), float(dperp[fp].mean())
                dtq = (tau[1:, 6:] - tau[:-1, 6:]).norm(dim=-1)
                vp = self.dyn_pair.bool()
                tjit = float(dtq[vp].mean()) if bool(vp.any()) else 0.0
            fjit2_mag = fjit2_dir = 0.0
            if N > 2:
                d2_par, d2_perp = self.force_split(f_final, 2)
                fp2 = self.ftrip.bool()
                if bool(fp2.any()):
                    fjit2_mag = float(d2_par.norm(dim=-1)[fp2].mean())
                    fjit2_dir = float(d2_perp.norm(dim=-1)[fp2].mean())
            cnt = max(int(self.dyn_row.sum()), 1)
            lin_rms = float((self.dyn_row[:, None] * tau[:, :3].pow(2)).sum().div(cnt).sqrt())
            ang_rms = float((self.dyn_row[:, None] * tau[:, 3:6].pow(2)).sum().div(cnt).sqrt())
            # Cone violation of the solved forces vs the TRUE cone (N). Generator forces live
            # in the inscribed pyramid, so this must read ~0; nonzero means a broken invariant.
            cone_str, cone_viol_max = "  cone OFF", float("nan")
            if self.cone_axis is not None:
                live = self.mask.bool()
                viol = self.cone_violation(f_final)[live]
                cone_viol_max = float(viol.max()) if viol.numel() else 0.0
                cone_str = (f"  cone[{int(viol.numel())}] viol μ={float(viol.mean()):.2e} "
                            f"max={cone_viol_max:.2e} N" if viol.numel() else "  cone[0]")

        print(f"    [kindyn dynamics] base wrench {base1:.2f} N/frame RMS "
              f"(lin {lin_rms:.1f} N  ang {ang_rms:.2f} N·m)  mean|f|={mean_f:.0f} N "
              f"(weight≈{self.total_mass * _GRAVITY_MAG:.0f} N)  jitter |Δ|f||={fjit_mag:.1f} "
              f"|Δf⊥|={fjit_dir:.1f} |Δ²f∥|={fjit2_mag:.1f} |Δ²f⊥|={fjit2_dir:.1f} N "
              f"|Δτ|={tjit:.2f}{cone_str}")
        return dict(
            joints_world=joints_world, torque_mag=torque_mag, force_world=force_world,
            joint_torques=tau.cpu().numpy().astype(np.float32),
            base_wrench=base_wrench,
            frame_forces=frame_force,
            total_mass=np.float32(self.total_mass),
            base_wrench_before=np.float32(self.base_wrench_before),
            base_wrench_warm=np.float32(self.base_wrench_warm),
            base_wrench_after=np.float32(base1),
            cone_violation_max_n=np.float32(cone_viol_max),
        )


def build_dynamics_spec(body, betas: torch.Tensor, frame_labels: torch.Tensor,
                        valid: np.ndarray, gravity_world: np.ndarray, *,
                        weights: dict, dyn_opt: dict, fps: float, device: str,
                        q_world: torch.Tensor, cones: bool,
                        half_angle_deg: float = 45.0) -> DynamicsSpec:
    """Assemble one person's :class:`DynamicsSpec` from the frozen-at-init pieces.

    The RNEA model comes baked from BetterHuman at ``betas`` (real per-part inertia,
    native ``q`` order); only gravity is overridden to the scene's world-down. The
    contact mask marks, per contact frame and video frame, whether the labels say that
    frame touches on a valid frame — exactly the steps it may carry a force.

    The friction pyramids are frozen here, once, at the warm-start pose ``q_world``:
    forward kinematics gives every contact frame its outward skin normal, the contact
    axis is that normal **negated** (a touching patch of skin faces into the surface, so
    the surface pushes back the other way — the same axis the force witness uses), and
    :func:`~tools.human_optim.forces.cone_generators` turns it into the four pyramid
    edges. They stay fixed while ``q`` moves: a cone recomputed every iteration would be
    a moving target the pose could rotate to make force cheap, and the pose shifts here
    are small (this stage starts from a converged body). ``cones=False`` gives every
    frame a free bilateral force instead.
    """
    dev = torch.device(device)
    dtype = torch.float32

    model = body.shaped(betas.to(dev)).robot
    g = torch.as_tensor(gravity_world, dtype=dtype, device=dev)
    g = g / (g.norm() + 1e-9) * _GRAVITY_MAG
    g6 = torch.cat([g, torch.zeros(3, dtype=dtype, device=dev)])
    model = dataclasses.replace(model, values=dataclasses.replace(model.values, gravity=g6))
    total_mass = float(model.values.body_inertias[..., 1:, 0].sum())

    frame_names = list(body.contact_frame_names)
    frame_ids = torch.as_tensor(list(body.contact_frames.frame_ids), dtype=torch.long, device=dev)
    parent_body = [int(p) for p in body.contact_frame_parents]
    parent_ids = torch.tensor([int(model.body_name_to_id[body.joint_names[p]]) for p in parent_body],
                              dtype=torch.long, device=dev)

    valid_t = torch.as_tensor(np.asarray(valid, bool), dtype=torch.bool, device=dev)
    mask = ((frame_labels.to(dev) == 1) & valid_t[:, None]).to(dtype)        # (T, 12)
    valid_pair = (valid_t[1:] & valid_t[:-1]).to(dtype)                      # (T-1,)
    fpair = mask[1:] * mask[:-1] * valid_pair[:, None]

    body_weight_n = total_mass * _GRAVITY_MAG
    # High-fps clips: wider v/a stencil (fps-invariant accel noise) + the 30 fps
    # restraint of the force/torque smoothness weights. Both identity at fps <= 30.
    spacing = max(1, round(float(fps) / FPS_REF))
    # Rows whose ±spacing stencil never touches an untracked frame — a gap holds the
    # nearest tracked pose and differencing across the held jump would read as a huge
    # phantom acceleration, so only these rows carry the base-wrench residual.
    vb = np.asarray(valid, bool)
    ok = vb.copy()
    for k in range(1, spacing + 1):
        ok[k:] &= vb[:-k]
        ok[:-k] &= vb[k:]
    spec = DynamicsSpec(
        model=model, frame_ids=frame_ids, parent_ids=parent_ids, parent_body=parent_body,
        frame_names=frame_names, mask=mask, fpair=fpair, valid_pair=valid_pair,
        dt=1.0 / float(fps), weights=fps_scaled_weights(weights, fps), dyn_opt=dyn_opt,
        gravity_world=np.asarray(gravity_world, np.float32), valid=vb,
        total_mass=total_mass, mu=math.tan(math.radians(float(half_angle_deg))),
        res_scale=1.0 / (max(int(ok.sum()), 1) * body_weight_n ** 2),
        spacing=spacing, dyn_row=torch.as_tensor(ok, dtype=dtype, device=dev))
    if cones:
        with torch.no_grad():
            _, frames7 = spec.placements(normalize_q(q_world.to(dev), body.num_joints))
            spec.cone_axis = -frame_normals(frames7)                         # (T, 12, 3)
            spec.cone_gen = cone_generators(spec.cone_axis, spec.mu).detach()
    return spec


# ───────────────────────────── the stage runner ───────────────────────────

def _kindyn_phases(cfg: HumanOptimConfig) -> list[PhaseSpec]:
    """Phases for ``kindyn_1`` — a single scene-on joint L-BFGS polish.

    The pose starts already converged (``kinematic_2``) and the forces are
    warm-started, so there is nothing for an Adam warm-up to do: a fixed-lr Adam
    cannot co-adapt the ~10³ N forces to a moving ``q`` (they stay frozen while
    ``q`` flails against the stale physics residual, wrecking the motion). L-BFGS
    moves pose and forces together under a Wolfe line search, which keeps the step
    from overshooting — the right tool once both variables start near-optimal.
    """
    it = cfg.iters["kindyn_1"]
    w = cfg.weights_for("kindyn_1")
    return [PhaseSpec("lbfgs", w, it.get("lbfgs", 0), _ALL_DOF, True, "lbfgs")]


def run_kindyn(source, pred: Path, cfg: HumanOptimConfig, device: str, *,
               annotations=None, overlay: bool = False, labels_mode: str = "auto",
               bodies: str = "kinematic_2", cones: bool = True) -> Path:
    """Joint kinematics + dynamics refinement of a refined body → ``human_optim/kindyn_1.npz``.

    Starts from a converged body and that body's contact labels, and optimizes the pose
    and the per-contact-frame forces together. The output carries the refined-body fields
    (same schema as ``kinematic_2.npz``) plus the solved force fields, so it works both as
    a body source and as an inspectable force artifact. ``bodies`` picks the starting body
    — ``'kinematic_2'`` (default) or ``'kinematic_1'`` when the chain skips kinematics_2 —
    and with it the matching contact readout (``contacts_2`` / ``contacts_1``).
    ``labels_mode='auto'`` (the pipeline default) reads that readout, measured on the very
    body this stage starts from; ``'manual'`` reads the human annotation ``data/<stem>.npz``
    and spreads its coarse joints onto the contact frames they cover (a metrics-only
    comparison path). ``cones`` confines every contact force to a 45° friction pyramid
    about its frame's contact normal.
    """
    knobs = cfg.knobs
    pred = Path(pred)
    tag = "human_optim/kindyn_1"

    if bodies not in ("kinematic_1", "kinematic_2"):
        raise SystemExit(f"[{tag}] [stages.kindyn_1].bodies must be 'kinematic_1' or "
                         f"'kinematic_2', got {bodies!r}")
    contacts_tag = bodies.removeprefix("kinematic_")
    kin_path = paths.body_path(pred, bodies)
    if not kin_path.exists():
        raise SystemExit(f"missing {kin_path} — run human_optimization --stages "
                         f"{bodies.replace('kinematic', 'kinematics')} first")
    kin = dict(np.load(kin_path, allow_pickle=True))
    pose = dict(np.load(pred / "sapiens" / "pose.npz", allow_pickle=True))
    human = dict(np.load(pred / "geometry" / "human.npz", allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    if not bool(tr.get("metric", True)):
        raise SystemExit(f"[{tag}] geometry under {pred} is up-to-scale — run estimate_scale after fuse")

    N, P = int(kin["num_frames"]), int(kin["num_people"])
    H, W = int(kin["video_height"]), int(kin["video_width"])
    object_ids = np.asarray(kin["object_ids"], dtype=np.int32)
    fps = float(tr.get("fps", kin.get("fps", 30.0)))

    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)
    K = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)
    wfc = world_from_cam(extr)
    cam_centers = wfc[:, :3, 3]
    scene_pts, scene_normals, scene_conf = _load_scene(pred, knobs, wfc, device, tag)

    grav_path = pred / "geocalib" / "gravity.npz"
    if grav_path.exists():
        gd = np.load(grav_path)
        gravity_world = np.asarray(gd["gravity_world"], np.float32)
        gsrc = str(gd["source"]) if "source" in gd.files else "geocalib"
    else:
        gravity_world = np.array([0.0, 1.0, 0.0], np.float32)
        gsrc = "fallback_down(no geocalib)"

    half_angle = float(knobs.get("friction", {}).get("half_angle_deg", 45.0))
    print(f"[{tag}] {pred.name}: P={P} N={N} fps={fps:.1f}  bodies={bodies}  "
          f"gravity={gravity_world.round(2)} ({gsrc})  "
          f"cones={'ON @ %.0f°' % half_angle if cones else 'OFF'}")

    mid, fvids, corr, sub, chamfer_w = _body_context(knobs, device)
    frame_names = list(mid.contact_frame_names)

    ann_contacts = ann_names = ann_oids = cnt_oids = cnt_contact = None
    if labels_mode == "auto":
        cnt_oids, cnt_contact = load_frame_contacts(
            pred, contacts_tag, kin_path, N, frame_names, tag)
    else:
        ann_path = resolve_annotation_path(source, annotations)
        ann = dict(np.load(ann_path, allow_pickle=True))
        ann_contacts = ann["contacts"]
        ann_names = [str(x) for x in ann["joint_names"]]
        ann_oids = [int(x) for x in ann["object_ids"]]
        print(f"[{tag}] annotation {ann_path} — joints {ann_names}, ids {ann_oids}")

    weights = cfg.weights_for("kindyn_1")
    dyn_opt = knobs["dynamics_optimizer"]
    torch.manual_seed(0)
    phases = _kindyn_phases(cfg)

    out = _new_out()
    betas_out = []
    force_world_p, torque_p, base_wrench_p = [], [], []
    joints_world_p, joint_torque_p, frame_force_p = [], [], []
    total_mass_p, joint_contact_p, frame_contact_p, diag_p = [], [], [], []
    fconf_p, fconf_clip_p = [], []
    for p in range(P):
        oid = int(object_ids[p])
        vmask = np.asarray(kin["valid_mask"][p], dtype=bool)
        betas_mean = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        q_world = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device).clone()
        q_world[:, 3:] += 1e-6 * torch.randn_like(q_world[:, 3:])
        betas_T = betas_mean[None].expand(N, -1)

        target_kp, scores, depth, frame_valid = _person_visual_inputs(
            pose, human, oid, N, vmask, knobs, device,
            mid=mid, betas_T=betas_T, q_world=q_world, extr=extr, K=K)

        # Labels on the 12 contact frames. Positives-only (+1 / -1, never 0) on purpose:
        # arming the clearance repulsion with negatives was tried (2026-07-26) and measured
        # to ADD jitter — the ann terms then fight the force terms in the joint L-BFGS.
        if labels_mode == "auto" and cnt_oids is not None and oid in cnt_oids:
            fc = cnt_contact[cnt_oids.index(oid)]                          # (N, 12)
            lab = np.where(fc, np.int8(1), np.int8(-1)).astype(np.int8)
        elif labels_mode != "auto" and ann_oids is not None and oid in ann_oids:
            lab = annotation_frame_labels(ann_contacts[ann_oids.index(oid)], ann_names,
                                          frame_names, N)
            lab[lab == 0] = -1                                             # positives-only
        else:
            lab = np.full((N, len(frame_names)), np.int8(-1), np.int8)
            print(f"[{tag}] person obj_id={oid}: NO contact labels (kinematic re-fit, no forces)")
        lab[~vmask] = -1
        contact_labels = torch.as_tensor(lab, dtype=torch.int8, device=device)
        per_frame = " ".join(f"{n}={int((lab[:, i] == 1).sum())}" for i, n in enumerate(frame_names))
        print(f"[{tag}] obj_id={oid}: contacts on {int((lab == 1).any(1).sum())} frames  ({per_frame})")
        frame_contact_p.append((lab == 1))

        spec = build_dynamics_spec(
            mid, betas_mean, contact_labels, vmask, gravity_world,
            weights=weights, dyn_opt=dyn_opt, fps=fps, device=device,
            q_world=q_world, cones=cones, half_angle_deg=half_angle)

        print(f"[{tag}] person obj_id={oid} ({int(vmask.sum())}/{N} valid) …")
        # Fingers stay UNfrozen (like kinematics_2): a fingertip contact frame must be
        # able to move with the physics residual.
        q_opt, before, after, aux = optimize_person(
            mid, betas_mean, q_world, extrinsics=extr, K=K, depth_points=depth,
            target_kp=target_kp, scores=scores, corr=corr, frame_vertex_ids=fvids,
            sub=sub, knobs=knobs, phases=phases,
            scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, fps=fps,
            contact_labels=contact_labels, contact_axis="frames", image_hw=(H, W),
            chamfer_weights=chamfer_w, frame_valid=frame_valid,
            cam_centers=cam_centers, dynamics=spec)
        print(f"    before: {before}\n    after:  {after}")

        _append_classic(out, mid, betas_T, q_opt)
        betas_out.append(betas_mean.cpu().numpy().astype(np.float32))
        dyn = aux["dynamics"]
        force_world_p.append(dyn["force_world"])
        torque_p.append(dyn["torque_mag"])
        base_wrench_p.append(dyn["base_wrench"])
        joints_world_p.append(dyn["joints_world"])
        joint_torque_p.append(dyn["joint_torques"])
        frame_force_p.append(dyn["frame_forces"])
        total_mass_p.append(dyn["total_mass"])
        diag_p.append([dyn[k] for k in ("base_wrench_before", "base_wrench_warm",
                                        "base_wrench_after", "cone_violation_max_n")])
        # Whole-body force confidence: the trustworthiness of THIS person's solved forces,
        # per frame, from the physics alone (residuals + overload). See force_confidence.py.
        fc = force_confidence(dyn["base_wrench"], dyn["frame_forces"], vmask,
                              float(dyn["total_mass"]) * _GRAVITY_MAG)
        fconf_p.append(fc["confidence"])
        fconf_clip_p.append(fc["clip"])
        print(f"    [kindyn dynamics] force confidence {fc['clip']:.2f} "
              f"(lin res med {np.nanmedian(fc['lin_residual_bw'][vmask]):.2f} BW, "
              f"load med {np.nanmedian(fc['load_bw'][vmask]):.2f} BW)")
        # Per-joint contact for the viewer: a joint is in contact where any of its frames is.
        jc = np.zeros((N, mid.num_joints), bool)
        for c, jb in enumerate(spec.parent_body):
            jc[:, jb] |= lab[:, c] == 1
        joint_contact_p.append(jc)

    # Body fields (same schema as kinematic_2.npz) + the solved force fields.
    out_path = paths.body_path(pred, "kindyn_1")
    extra = dict(
        contact_forces_world=np.stack(force_world_p, 0),
        torque_mag=np.stack(torque_p, 0),
        base_wrench=np.stack(base_wrench_p, 0),               # (P, N, 6) ROOT-LOCAL frame
        joints_world=np.stack(joints_world_p, 0),
        joint_torques=np.stack(joint_torque_p, 0),
        frame_forces=np.stack(frame_force_p, 0),              # (P, N, 12, 3) world newtons
        frame_contact=np.stack(frame_contact_p, 0),           # (P, N, 12) the labels the solve used
        contact_frame_names=np.array(frame_names),
        contact_frame_parents=np.array([int(x) for x in mid.contact_frame_parents], np.int64),
        joint_contact=np.stack(joint_contact_p, 0),           # (P, N, 52) the same, folded to joints
        total_mass=np.stack(total_mass_p, 0),
        gravity_world=gravity_world.astype(np.float32),
        base_wrench_diagnostics=np.asarray(diag_p, np.float32),   # (P, 4) before/warm/after/coneviol
        force_confidence=np.stack(fconf_p, 0),                    # (P, N) whole-body, per frame
        force_confidence_clip=np.asarray(fconf_clip_p, np.float32),  # (P,) mean over valid frames
        cones=np.bool_(cones), cone_half_angle_deg=np.float32(half_angle),
        source_bodies=np.array(bodies, dtype="U16"),
        source_labels=np.array(labels_mode, dtype="U8"),
    )
    # Carry kinematics_1's pre-enforcement drift through, for the same reason kinematics_2
    # does: any contact readout taken on THIS body must gate stillness on motion measured
    # before a contact term ever acted, never on the damping this stage just applied.
    extra.update({k: kin[k] for k in ("drift_pre", "auto_labels") if k in kin})
    save = _save_body(out_path, mid, out, betas_out, object_ids, kin["valid_mask"],
                      N, P, H, W, fps, extra=extra)
    print(f"[{tag}] wrote {out_path}")

    if overlay:
        from .overlay import render_keypoint_overlay
        ov = render_keypoint_overlay(
            source, paths.overlay_path(pred, "kindyn_1"), mid, save, corr, fvids,
            extr, K, target_scheme=pose, fps=fps, device=device)
        print(f"[{tag}] wrote {ov}")
    return out_path
