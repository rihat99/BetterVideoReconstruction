"""The ``kindyn_1`` stage: joint kinematics + dynamics refinement.

The sequential chain fits the motion first (kinematics), then solves contact
forces on the FROZEN pose (dynamics). ``kindyn_1`` instead optimizes the pose
``q`` **and** the contact forces together, so the physics residual — the RNEA
base wrench, the 6-DOF force/torque left on the un-actuated free-flyer root — can
reshape the motion instead of only being absorbed by forces.

It starts from a converged body (``kinematic_2`` by default, ``kinematic_1`` when
the chain skips kinematics_2) and that body's contact labels. The
objective is the ``kinematics_2`` losses (reprojection, chamfer, priors, motion
smoothness, penetration, the annotation terms) PLUS the ``dynamics`` terms
(base wrench, force magnitude/smoothness, torque smoothness), with the forces as
extra optimization variables: one world-frame force vector per contact group per
frame, masked to the frames the labels say that group touches. Feet and ankles
push through **friction pyramids** built like the ``dynamics`` stage's (a force
that is a nonnegative combination of the four pyramid edges cannot leave the
cone), except the cone frames are frozen once at the warm-start pose — ``q`` moves
during the solve, and re-reading the scene normals every iteration would make the
cone itself a moving target. Hands keep a free bilateral force (they grasp).

The physics half of the objective is packaged as a :class:`DynamicsSpec`
(built once per person from the RNEA model, the contact-group force points and
the per-frame contact mask). :func:`~tools.human_optim.optimizer.optimize_person`
owns the shared kinematic Problem and simply adds the force leaf, threads the
RNEA forward through its per-iteration state (with gradient, so ``v``/``a``/
``oMi`` are live — not detached as in the frozen-pose dynamics stage) and
registers the five dynamics terms. The stage entry point is :func:`run_kindyn`,
driven by ``scripts/stages/human_optimization.py --stages kindyn_1``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tools.geometry.normals import load_scene_cloud
from tools.smplx_robot import load_smplx_mid
from tools.smplx_robot.dynamics import velocity_acceleration_from_trajectory

from . import paths
from .camera import world_from_cam
from .config import HumanOptimConfig
from .dynamics import _GRAVITY_MAG, _build_fext, _contact_catalog, _world_to_local
from .friction import cone_generators, cone_normals
from .optimizer import PhaseSpec, optimize_person
from .qmath import normalize_q
from .stages import (
    _ALL_DOF, _DEFAULT_SMPLX_MODEL, _body_context, _contact_labels, _load_scene,
    _new_out, _append_classic, _person_visual_inputs, _save_body, load_auto_contacts,
    resolve_annotation_path,
)


# ─────────────────────────── the physics half ────────────────────────────

@dataclass
class DynamicsSpec:
    """The force part of the ``kindyn_1`` objective for one person.

    Holds the pieces that stay fixed while ``q`` and the forces move: the RNEA
    ``model`` (gravity baked to the scene's world-down), the contact-group force
    points, the per-frame contact ``mask``, and the frozen friction cones. Its methods
    run the RNEA forward
    (recomputing ``v``/``a``/``oMi`` from the *current* ``q`` so gradient flows
    into the pose), define the five dynamics loss terms, warm-start the forces at
    the initial pose, and package the solved forces for the output npz.

    All five loss weights are **linear** — the term value is ``0.5·Σ residual²``
    and the engine multiplies by the (kindyn-column) weight, exactly like every
    kinematic term. This differs from the standalone ``dynamics`` stage, which
    folds the weight inside the square; a linear weight ``w`` here plays the role
    of ``w_inside²`` there.
    """

    model: object                 #: BetterRobot ``Model`` (scene-down gravity baked in)
    contact_ids: torch.Tensor     #: (C,) model body ids of the force points
    rep_bh: list[int]             #: (C,) SMPLXMid joint index of each force point
    group_names: list[str]        #: (C,) contact-group names, force-point order
    mask: torch.Tensor            #: (T, C) float 0/1 — a group carries force only where labelled
    fpair: torch.Tensor           #: (T-1, C) float — smoothness mask across persisting contacts
    valid_pair: torch.Tensor      #: (T-1,) float — both-ends-valid frame steps
    dt: float                     #: frame interval, seconds
    weights: dict                 #: kindyn dynamics loss weights (linear)
    dyn_opt: dict                 #: L-BFGS knobs for the force warm-start
    gravity_world: np.ndarray     #: (3,) scene world-down (for the output npz)
    valid: np.ndarray             #: (T,) bool per-frame validity
    total_mass: float             #: body mass, kg (baked inertia sum)
    cone_cols: tuple[int, ...] = ()      #: force-point columns carrying a friction cone (non-hand groups)
    cone_normal: torch.Tensor | None = None  #: (T, Cc, 3) frozen surface normal per cone entry
    cone_gen: torch.Tensor | None = None     #: (T, Cc, 4, 3) frozen pyramid edge directions
    cone_mask: torch.Tensor | None = None    #: (T, Cc) float 0/1 — cone active (labelled + usable normal)
    cone_w: torch.Tensor | None = None       #: (T, C, 1) float 0/1 — the cone force replaces the free one
    mu: float = 1.0                          #: tan(cone half-angle) — the friction coefficient

    @property
    def C(self) -> int:
        return len(self.group_names)

    def zero_forces(self) -> torch.Tensor:
        """Initial free-force leaf ``(T, C, 3)`` — zeros; the warm-start fills it."""
        T = self.mask.shape[0]
        return torch.zeros(T, self.C, 3, dtype=self.mask.dtype, device=self.mask.device)

    def zero_generators(self) -> torch.Tensor | None:
        """Initial pyramid-activation leaf ``(T, Cc, 4)`` — zeros, or ``None`` with no cones."""
        if self.cone_gen is None:
            return None
        T = self.mask.shape[0]
        return torch.zeros(T, len(self.cone_cols), 4, dtype=self.mask.dtype, device=self.mask.device)

    def compose_forces(self, f_free: torch.Tensor, gen_u: torch.Tensor | None) -> torch.Tensor:
        """World force ``(T, C, 3)``: pyramid-generator force on cone entries, free elsewhere.

        ``gen_u`` are raw activations; ``softplus`` makes them nonnegative (and unlike
        relu leaves no dead zero-gradient point), so the cone force is a nonnegative
        combination of the four frozen pyramid edges and cannot leave the cone. Cone
        entries are selected by ``cone_w`` — a group with no usable normal, or a hand,
        keeps its free bilateral vector.
        """
        if gen_u is None or self.cone_gen is None:
            return f_free
        lam = torch.nn.functional.softplus(gen_u)                          # (T, Cc, 4) ≥ 0
        f_cone = torch.einsum("tcg,tcgd->tcd", lam, self.cone_gen)         # (T, Cc, 3) inside the pyramid
        full = torch.zeros_like(f_free)
        full[:, self.cone_cols] = f_cone
        return self.cone_w * full + (1.0 - self.cone_w) * f_free

    def cone_violation(self, f_world: torch.Tensor) -> torch.Tensor:
        """``relu(‖f_t‖ − μ·f_n)`` per (frame, cone entry) against the TRUE cone — a readout.

        ``f_n = f·n̂`` is the signed normal component and ``f_t`` the tangential rest.
        Generator forces live in the pyramid inscribed in the cone, so this must read ~0;
        it is printed as an invariant check, never optimized."""
        fc = f_world[:, self.cone_cols]                                    # (T, Cc, 3)
        f_n = (fc * self.cone_normal).sum(-1)                              # (T, Cc)
        f_t = fc - f_n[..., None] * self.cone_normal
        return torch.relu(f_t.pow(2).sum(-1).add(1e-12).sqrt() - self.mu * f_n)

    def forward(self, qn: torch.Tensor, f_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """RNEA base + joint torques from the current pose and forces.

        Recomputes ``v``/``a`` (central differences on the manifold) and the
        contact-joint world→local rotations from ``qn`` **with gradient**, so the
        base-wrench residual pulls on ``q``. Returns ``(tau (T, nv), f_eff (T, C, 3))``
        where ``f_eff`` is the mask-gated world force.
        """
        from better_robot.dynamics import rnea
        from better_robot.kinematics import forward_kinematics

        v, a = velocity_acceleration_from_trajectory(self.model, qn, self.dt)
        oMi = forward_kinematics(self.model, qn).joint_pose_world
        R_w2l = _world_to_local(oMi, self.contact_ids)              # live (q varies) — NOT detached
        fext, f_eff = _build_fext(f_world, self.mask, R_w2l, self.contact_ids, self.model.njoints)
        tau = rnea(self.model, qn, v, a, fext=fext)
        return tau, f_eff

    @staticmethod
    def force_split(f_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split adjacent-frame force change into magnitude vs direction parts.

        About the mean direction ``u`` of the two forces, ``df_par`` is the
        ‖f‖-change part and ``df_perp`` the direction-change part — the same
        orthogonal decomposition the dynamics stage uses. Each ``(T-1, C, 3)``.
        """
        df = f_world[1:] - f_world[:-1]
        u = f_world[1:] + f_world[:-1]
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        df_par = (df * u).sum(-1, keepdim=True) * u
        return df_par, df - df_par

    def populate(self, st, f_free: torch.Tensor, gen_u: torch.Tensor | None) -> None:
        """Add the per-iteration lazy RNEA products to the shared :class:`State`.

        The state's ``"f_world"`` is the COMPOSED force (newtons) — the cone parametrization
        is applied once, here, so every dynamics term (smoothness included) sees forces and
        never raw pyramid activations."""
        st.put_lazy("f_world", lambda: self.compose_forces(f_free, gen_u))
        st.put_lazy("dyn", lambda: self.forward(st["qn"], st["f_world"]))
        st.put_lazy("dyn_split", lambda: self.force_split(st["f_world"]))

    def register(self, problem) -> None:
        """Register the five dynamics terms (active only where their weight is)."""
        half = 0.5

        def _base(st):
            tau = st["dyn"][0]
            return half * tau[:, :6].pow(2).sum(), {"bwN": tau[:, :6].pow(2).sum().sqrt()}

        def _fmag(st):
            f_eff = st["dyn"][1]
            return half * f_eff.pow(2).sum()

        def _fs_mag(st):
            df_par = st["dyn_split"][0]
            return half * (self.fpair[..., None] * df_par.pow(2)).sum()

        def _fs_dir(st):
            df_perp = st["dyn_split"][1]
            return half * (self.fpair[..., None] * df_perp.pow(2)).sum()

        def _tsmooth(st):
            tau = st["dyn"][0]
            dtau = tau[1:, 6:] - tau[:-1, 6:]
            return half * (self.valid_pair[:, None] * dtau.pow(2)).sum()

        problem.add_term("base_wrench", _base)
        problem.add_term("force_magnitude", _fmag)
        problem.add_term("force_smooth_mag", _fs_mag)
        problem.add_term("force_smooth_dir", _fs_dir)
        problem.add_term("torque_smooth", _tsmooth)

    def warmstart(self, qn_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Solve the forces at the fixed initial pose (the sequential dynamics solve).

        Forces start at zero, so the base-wrench gradient on ``q`` is meaningless
        (an unsupported body is in freefall). A short L-BFGS on the forces alone —
        pose detached — seeds them near their sequential optimum, so the joint
        phases start from a physically-reasonable point for both ``q`` and ``f``.
        Both leaves are solved (free force AND pyramid activations) so the returned
        values seed the joint phases coherently. Returns ``(f_free (T, C, 3),
        gen_u (T, Cc, 4) or None)``, detached.
        """
        from better_robot.dynamics import rnea
        from better_robot.kinematics import forward_kinematics

        w = self.weights
        with torch.no_grad():
            qn = qn_init.detach()
            v, a = velocity_acceleration_from_trajectory(self.model, qn, self.dt)
            oMi = forward_kinematics(self.model, qn).joint_pose_world
            R_w2l = _world_to_local(oMi, self.contact_ids).detach()
            base0 = float(rnea(self.model, qn, v, a)[..., :6].pow(2).sum().sqrt())

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

        def closure():
            opt.zero_grad()
            f_world = self.compose_forces(f, gen_u)
            fext, f_eff = _build_fext(f_world, self.mask, R_w2l, self.contact_ids, self.model.njoints)
            tau = rnea(self.model, qn, v, a, fext=fext)
            loss = (w["base_wrench"] * 0.5 * tau[:, :6].pow(2).sum()
                    + w["force_magnitude"] * 0.5 * f_eff.pow(2).sum())
            if f.shape[0] > 1:
                df_par, df_perp = self.force_split(f_world)
                loss = loss + w["force_smooth_mag"] * 0.5 * (self.fpair[..., None] * df_par.pow(2)).sum()
                loss = loss + w["force_smooth_dir"] * 0.5 * (self.fpair[..., None] * df_perp.pow(2)).sum()
                dtau = tau[1:, 6:] - tau[:-1, 6:]
                loss = loss + w["torque_smooth"] * 0.5 * (self.valid_pair[:, None] * dtau.pow(2)).sum()
            loss.backward()
            return loss

        if bool(self.mask.any()):
            opt.step(closure)
        with torch.no_grad():
            f_world = self.compose_forces(f, gen_u)
            base1 = float(self.forward(qn, f_world)[0][..., :6].pow(2).sum().sqrt())
        print(f"    [kindyn warm-start] base wrench {base0:.1f}→{base1:.2f} N (forces only, q frozen)")
        return f.detach(), None if gen_u is None else gen_u.detach()

    def extract(self, qn: torch.Tensor, f_free: torch.Tensor, gen_u: torch.Tensor | None,
                mid) -> dict:
        """Package the solved forces + diagnostics for the output npz.

        Mirrors the fields the ``dynamics`` stage saves so the forces are
        inspectable the same way. Returns a dict of numpy arrays plus scalar
        diagnostics (base wrench, force / torque jitter) used in the stage log.
        """
        from better_robot.kinematics import forward_kinematics

        J = mid.num_joints
        with torch.no_grad():
            f_world = self.compose_forces(f_free, gen_u)                    # (T, C, 3) newtons
            f_final = (f_world * self.mask[..., None])                      # (T, C, 3)
            tau, _ = self.forward(qn, f_world)                              # (T, nv)
            oMi = forward_kinematics(self.model, qn).joint_pose_world       # (T, njoints, 7)
            base1 = float(tau[:, :6].pow(2).sum().sqrt())

            idx_vs = [int(x) for x in self.model.idx_vs]
            nvs = [int(x) for x in self.model.nvs]
            N = qn.shape[0]
            torque_mag = np.zeros((N, J), np.float32)
            joints_world = np.zeros((N, J, 3), np.float32)
            force_world = np.zeros((N, J, 3), np.float32)
            for jb in range(J):
                mb = int(self.model.body_name_to_id[mid.joint_names[jb]])
                n = nvs[mb]
                if n > 0:
                    torque_mag[:, jb] = tau[:, idx_vs[mb]:idx_vs[mb] + n].norm(dim=-1).cpu().numpy()
                joints_world[:, jb] = oMi[:, mb, :3].cpu().numpy()
            for c, jb in enumerate(self.rep_bh):
                force_world[:, jb] = f_final[:, c].cpu().numpy()

            inval = ~self.valid.astype(bool)
            torque_mag[inval] = 0.0
            force_world[inval] = 0.0
            base_wrench = tau[:, :6].cpu().numpy().astype(np.float32)
            base_wrench[inval] = 0.0

            fnorm = f_final.norm(dim=-1)                                    # (T, C)
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
                vp = self.valid_pair.bool()
                tjit = float(dtq[vp].mean()) if bool(vp.any()) else 0.0
            # Cone violation of the solved forces vs the TRUE cone (N). Generator forces live in
            # the inscribed pyramid, so this must read ~0; nonzero means a broken invariant.
            cone_str = ""
            if self.cone_mask is not None:
                viol = self.cone_violation(f_final)[self.cone_mask.bool()]
                cone_str = (f"  cone[{int(viol.numel())}] viol μ={float(viol.mean()):.1f} "
                            f"max={float(viol.max()):.1f} N" if viol.numel() else "  cone[0]")

        print(f"    [kindyn dynamics] base wrench {base1:.2f} N  mean|f|={mean_f:.0f} N "
              f"(weight≈{self.total_mass * _GRAVITY_MAG:.0f} N)  jitter |Δ|f||={fjit_mag:.1f} "
              f"|Δf⊥|={fjit_dir:.1f} N |Δτ|={tjit:.2f}{cone_str}")
        return dict(
            joints_world=joints_world, torque_mag=torque_mag, force_world=force_world,
            joint_torques=tau.cpu().numpy().astype(np.float32),
            base_wrench=base_wrench,
            contact_forces=f_final.cpu().numpy().astype(np.float32),
            rep_joint_names=[mid.joint_names[j] for j in self.rep_bh],
            total_mass=np.float32(self.total_mass),
            base_wrench_after=np.float32(base1), mean_force=np.float32(mean_f),
            jitter_force_mag=np.float32(fjit_mag), jitter_force_dir=np.float32(fjit_dir),
            jitter_torque=np.float32(tjit),
        )


def build_dynamics_spec(mid, betas: torch.Tensor, contact_labels: torch.Tensor,
                        valid: np.ndarray, gravity_world: np.ndarray, *, groups: list[str],
                        weights: dict, dyn_opt: dict, fps: float, device: str,
                        q_world: torch.Tensor, scene_pts: torch.Tensor | None = None,
                        scene_normals: torch.Tensor | None = None,
                        friction: dict | None = None) -> DynamicsSpec:
    """Assemble one person's :class:`DynamicsSpec` from the frozen-at-init pieces.

    The RNEA model comes baked from BetterHuman at ``betas`` (real per-part
    inertia, native ``q`` order); only gravity is overridden to the scene's
    world-down. The contact mask marks, per contact group and frame, whether the
    annotation labels any member joint of that group in contact on a valid frame —
    exactly the frames that group may carry a force.

    The friction cones for the non-hand groups (feet + ankles) are frozen here, once,
    at the warm-start pose ``q_world``: forward kinematics gives each foot/ankle its
    world position, :func:`~tools.human_optim.friction.cone_normals` reads the wall
    normal there off the scene cloud, and
    :func:`~tools.human_optim.friction.cone_generators` turns it into the four pyramid
    edges. They stay fixed while ``q`` moves — a cone recomputed every iteration would
    be a moving target, and the pose shifts here are small (this stage starts from a
    converged body). A cone entry with no usable normal (surface too far, neighbouring
    normals disagreeing) keeps a free bilateral force, like a hand; without
    ``friction`` or a scene cloud there are no cones at all.
    """
    dev = torch.device(device)
    dtype = torch.float32
    J = mid.num_joints

    model = mid.shaped(betas.to(dev)).robot
    g = torch.as_tensor(gravity_world, dtype=dtype, device=dev)
    g = g / (g.norm() + 1e-9) * _GRAVITY_MAG
    g6 = torch.cat([g, torch.zeros(3, dtype=dtype, device=dev)])
    model = dataclasses.replace(model, values=dataclasses.replace(model.values, gravity=g6))
    total_mass = float(model.values.body_inertias[..., 1:, 0].sum())

    catalog = _contact_catalog(list(mid.joint_names))
    name_to_bh = {n: i for i, n in enumerate(mid.joint_names)}
    group_names = [gg for gg in groups if gg in catalog]
    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=dev)
    labels = contact_labels.to(dev)
    rep_bh, rep_model, masks = [], [], []
    for gname in group_names:
        rep, members = catalog[gname]
        member_idx = [name_to_bh[m] for m in members if m in name_to_bh]
        active = (labels[:, member_idx] == 1).any(dim=1) & valid_t          # (T,)
        rep_bh.append(name_to_bh[rep])
        rep_model.append(int(model.body_name_to_id[rep]))
        masks.append(active)
    C = len(group_names)
    mask = (torch.stack(masks, dim=1).to(dtype) if C
            else torch.zeros((valid_t.shape[0], 0), dtype=dtype, device=dev))
    valid_pair = (valid_t[1:] & valid_t[:-1]).to(dtype)                     # (T-1,)
    fpair = (mask[1:] * mask[:-1] * valid_pair[:, None]) if C else mask[1:]
    contact_ids = torch.tensor(rep_model, dtype=torch.long, device=dev)

    # Friction pyramids on the non-hand groups, frozen at the warm-start pose.
    cone_cols = tuple(c for c, gn in enumerate(group_names) if "hand" not in gn)
    cone_n = cone_gen = cone_mask = cone_w = None
    mu = math.tan(math.radians(float(friction["half_angle_deg"]))) if friction else 1.0
    if cone_cols and friction is not None and scene_pts is not None and scene_pts.shape[0]:
        from better_robot.kinematics import forward_kinematics
        with torch.no_grad():
            oMi = forward_kinematics(model, normalize_q(q_world.to(dev), J)).joint_pose_world
            joint_xyz = oMi[:, contact_ids[list(cone_cols)], :3]            # (T, Cc, 3)
        cone_n, cone_valid = cone_normals(joint_xyz, scene_pts, scene_normals,
                                          int(friction["normal_k"]), float(friction["max_dist_m"]))
        cone_mask = (mask[:, cone_cols].bool() & cone_valid).to(dtype)      # (T, Cc)
        if bool(cone_mask.any()):
            cone_gen = cone_generators(cone_n, mu).detach()                 # (T, Cc, 4, 3)
            cone_w = torch.zeros((mask.shape[0], C, 1), dtype=dtype, device=dev)
            cone_w[:, cone_cols, 0] = cone_mask
        else:
            cone_n = cone_mask = None

    return DynamicsSpec(
        model=model, contact_ids=contact_ids,
        rep_bh=rep_bh, group_names=group_names, mask=mask, fpair=fpair, valid_pair=valid_pair,
        dt=1.0 / float(fps), weights=weights, dyn_opt=dyn_opt,
        gravity_world=np.asarray(gravity_world, np.float32), valid=np.asarray(valid, bool),
        total_mass=total_mass, cone_cols=cone_cols, cone_normal=cone_n, cone_gen=cone_gen,
        cone_mask=cone_mask, cone_w=cone_w, mu=mu)


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
               annotations=None, overlay: bool = False, labels_mode: str = "manual",
               bodies: str = "kinematic_2") -> Path:
    """Joint kinematics + dynamics refinement of a refined body → ``human_optim/kindyn_1.npz``.

    Starts from a converged body and its contact labels, and
    optimizes the pose and per-contact-group forces together. The output carries
    the refined-body fields (same schema as ``kinematic_2.npz``) plus the solved
    force fields, so it works both as a body source and as an inspectable force
    artifact. ``bodies`` picks the starting body — ``'kinematic_2'`` (default) or
    ``'kinematic_1'`` when the chain skips kinematics_2 — and with it the matching
    contact readout (``contacts_2`` / ``contacts_1``). ``labels_mode='manual'``
    (default) reads the human annotation ``data/<stem>.npz`` instead; ``'auto'``
    reads that readout — measured on the very body this stage starts from.
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

    # Friction-cone scene cloud: UNGATED (conf_drop_q = 0) so a low-confidence voxel under a
    # foot still carries a normal — the cone only needs a direction, and dropping voxels there
    # would silently turn cones off. Separate from the gated cloud the kinematic scene terms use.
    friction = knobs.get("friction")
    cone_pts = cone_nrm = None
    if friction is not None and (pred / "geometry" / "scene.npz").exists():
        sc = dict(np.load(pred / "geometry" / "scene.npz", allow_pickle=True))
        cone_pts, cone_nrm, _ = load_scene_cloud(
            sc, cam_centers, conf_drop_q=0.0, k=int(friction["normal_k"]), device=device)
        print(f"[{tag}] friction cones ON ({cone_pts.shape[0]} scene pts, "
              f"half-angle {float(friction['half_angle_deg']):.0f}°)")
    else:
        print(f"[{tag}] no scene/[friction] — friction cones OFF")

    print(f"[{tag}] {pred.name}: P={P} N={N} fps={fps:.1f}  bodies={bodies}  "
          f"gravity={gravity_world.round(2)} ({gsrc})")
    ann_contacts = ann_names = ann_oids = cnt_oids = cnt_contact = None
    if labels_mode == "auto":
        kin_names = [str(x) for x in np.asarray(kin["joint_names"]).tolist()]
        cnt_oids, cnt_contact, _ = load_auto_contacts(
            pred, contacts_tag, kin_path, N, kin_names, tag)
    else:
        ann_path = resolve_annotation_path(source, annotations)
        ann = dict(np.load(ann_path, allow_pickle=True))
        ann_contacts = ann["contacts"]
        ann_names = [str(x) for x in ann["joint_names"]]
        ann_oids = [int(x) for x in ann["object_ids"]]
        print(f"[{tag}] annotation {ann_path} — joints {ann_names}, ids {ann_oids}")

    weights = cfg.weights_for("kindyn_1")
    groups = knobs["contact"]["groups"]
    dyn_opt = knobs["dynamics_optimizer"]

    mid, fvids, corr, sub, chamfer_w = _body_context(knobs, device)
    mid_names = list(mid.joint_names)
    torch.manual_seed(0)
    phases = _kindyn_phases(cfg)

    out = _new_out()
    betas_out = []
    force_world_p, torque_p, base_wrench_p = [], [], []
    joints_world_p, joint_torque_p, raw_forces_p, rep_names_p = [], [], [], []
    total_mass_p, joint_contact_p = [], []
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

        if labels_mode == "auto" and oid in cnt_oids:
            jc = cnt_contact[cnt_oids.index(oid)]                          # (N, 52)
            lab = np.where(jc, np.int8(1), np.int8(-1)).astype(np.int8)    # +1 / -1, NEVER 0
            lab[~vmask] = -1
            contact_labels = torch.as_tensor(lab, dtype=torch.int8, device=device)
            nc = int((lab == 1).any(1).sum())
            print(f"[{tag}] obj_id={oid}: AUTO contacts on {nc} frames")
        elif labels_mode != "auto" and oid in ann_oids:
            lab = _contact_labels(ann_contacts[ann_oids.index(oid)], ann_names, mid_names, N)
            lab[~vmask] = -1
            contact_labels = torch.as_tensor(lab, dtype=torch.int8, device=device)
            nc = int((lab == 1).any(1).sum()); nn = int((lab == 0).any(1).sum())
            print(f"[{tag}] person obj_id={oid}: contacts on {nc} frames, no-contact on {nn} frames")
        else:
            contact_labels = torch.full((N, mid.num_joints), -1, dtype=torch.int8, device=device)
            contact_labels[~vmask] = -1
            print(f"[{tag}] person obj_id={oid}: NO contact labels (kinematic re-fit, no forces)")
        joint_contact_p.append((contact_labels == 1).cpu().numpy())

        spec = build_dynamics_spec(
            mid, betas_mean, contact_labels, vmask, gravity_world,
            groups=groups, weights=weights, dyn_opt=dyn_opt, fps=fps, device=device,
            q_world=q_world, scene_pts=cone_pts, scene_normals=cone_nrm, friction=friction)

        print(f"[{tag}] person obj_id={oid} ({int(vmask.sum())}/{N} valid) …")
        q_opt, before, after, aux = optimize_person(
            mid, betas_mean, q_world, extrinsics=extr, K=K, depth_points=depth,
            target_kp=target_kp, scores=scores, corr=corr, frame_vertex_ids=fvids,
            sub=sub, knobs=knobs, phases=phases,
            scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, fps=fps,
            contact_labels=contact_labels, chamfer_weights=chamfer_w, frame_valid=frame_valid,
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
        raw_forces_p.append(dyn["contact_forces"])
        rep_names_p.append(dyn["rep_joint_names"])
        total_mass_p.append(dyn["total_mass"])

    # Body fields (same schema as kinematic_2.npz) + the solved force fields.
    out_path = paths.body_path(pred, "kindyn_1")
    extra = dict(
        contact_forces_world=np.stack(force_world_p, 0),
        torque_mag=np.stack(torque_p, 0),
        base_wrench=np.stack(base_wrench_p, 0),
        joints_world=np.stack(joints_world_p, 0),
        joint_torques=np.stack(joint_torque_p, 0),
        contact_forces=np.stack(raw_forces_p, 0),
        contact_force_joints=np.array(rep_names_p, dtype="<U16"),
        total_mass=np.stack(total_mass_p, 0),
        gravity_world=gravity_world.astype(np.float32),
        joint_contact=np.stack(joint_contact_p, 0),          # (P, N, 52) the labels the solve used
        source_bodies=np.array(bodies, dtype="U16"),
    )
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
