"""Contact completion: recover the contacts the detector misses, union them on.

The scene-contact detector is precise but low recall — it only labels a joint
that is within a few cm of the wall AND still. This module adds the contacts it
misses with two looser signals and unions them onto the detector labels
(add-only: completion never removes a detector label). Per joint *group* (two
hands, two feet, two ankles) it fires:

  * **gates**  — the group is within ``gate_dist_cm`` of the scene surface AND
    its net window drift (the pre-enforcement stillness measure) is below
    ``gate_drift_m``. This is just the detector's own two signals, loosened.
  * **force**  — a *static support solve* (below) allocates the body's weight
    among the six candidate groups every frame; a group that must carry more
    than ``force_theta_bw`` of body weight, while drifting less than
    ``force_drift_m``, is holding a contact the gates could not see (typically a
    misregistered limb, where distance lies and global drift corrupts stillness).

"Static" means no motion. Per frame the solve balances only the gravity wrench
(RNEA at zero velocity and acceleration) across the groups, shaped by a work
penalty (force on a fast-moving limb is expensive — it uses the raw,
finite-difference limb speed), a gap penalty (force far from the wall is
expensive), friction pyramids on feet/ankles, and force smoothness. A long
diagnostic round found that zeroing the body's velocity and acceleration leaves
the labels unchanged, so the production solve drops motion entirely and keeps
only the raw limb speed the work penalty needs.

Each component mask is 3-frame min-span filtered (short flickers dropped) after
invalid frames are cleared, then unioned:

    label = detector ∪ span(gates) ∪ span(force)

Completion additions land on the group's own joint only (wrist / foot / ankle);
finger and toe joints stay detector-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .qmath import normalize_q

# Defaults for the ``[contacts.completion]`` TOML section (the validated recipe:
# gates dist < 30 cm ∧ drift < 0.08 m, force |f| > 0.05·BW ∧ drift < 0.15 m,
# 3-frame min-span, work v* 0.2 m/s, gap d* 0.05 m).
COMPLETION_DEFAULTS: dict[str, float] = {
    "gate_dist_cm": 30.0,     # gate enter: group within this distance of the scene (cm)
    "gate_drift_m": 0.08,     # gate enter: group net window drift below this (m)
    "force_theta_bw": 0.05,   # force enter: solved |f| above this fraction of body weight
    "force_drift_m": 0.15,    # force enter: group drift below this (m) — looser than the gate
    "min_span": 3,            # drop completion runs shorter than this many frames
    "work_vstar_mps": 0.2,    # work penalty soft velocity: force on a limb faster than this is expensive
    "gap_dstar_m": 0.05,      # gap penalty soft distance: force further than this from the wall is expensive
}


@dataclass(frozen=True)
class DynamicsContext:
    """The dynamics-stage knobs the static solve borrows (loaded once per run)."""

    groups: tuple[str, ...]           # the six contact-group names, force order
    weights: dict[str, float]         # dynamics loss weights (base_wrench, force_magnitude, …)
    dyn_opt: dict                     # L-BFGS knobs
    friction: dict | None             # friction-cone knobs, or None (no cones)


def load_dynamics_context(config_path) -> DynamicsContext:
    """Read the dynamics weights, optimizer, friction and contact groups from the TOML."""
    from .config import load_config

    cfg = load_config(config_path)
    return DynamicsContext(
        groups=tuple(cfg.knobs["contact"]["groups"]),
        weights=cfg.weights_for("dynamics"),
        dyn_opt=cfg.knobs["dynamics_optimizer"],
        friction=cfg.knobs.get("friction"),
    )


def _min_span(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Drop ``True`` runs shorter than ``min_frames`` in each column of ``(N, C)`` bool."""
    out = np.array(mask, dtype=bool, copy=True)
    for c in range(out.shape[1]):
        col = out[:, c]
        edges = np.flatnonzero(np.diff(np.concatenate([[0], col.view(np.int8), [0]])))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start < min_frames:
                col[start:stop] = False
    return out


def solve_static_support(mid, betas: torch.Tensor, q: torch.Tensor, valid: np.ndarray,
                         gravity_world: np.ndarray, fps: float, *,
                         group_dist_cm: np.ndarray, groups, weights: dict, dyn_opt: dict,
                         friction: dict | None, cone_pts, cone_normals_t,
                         vstar: float, dstar: float, device: str) -> tuple:
    """One person's static support solve → ``(forces (N, C, 3) world, total_mass, base0, base1)``.

    Balances the per-frame gravity wrench (RNEA with zero velocity/acceleration)
    among the ``groups`` contact points with the dynamics stage's terms plus the
    work and gap penalties. Feet/ankles push through friction pyramids; hands
    keep a free force. ``base0``/``base1`` are the root-wrench norms before/after
    the solve, printed for provenance.
    """
    import dataclasses
    import math

    from better_robot.dynamics import rnea
    from better_robot.kinematics import forward_kinematics

    from .dynamics import _GRAVITY_MAG, _build_fext, _contact_catalog, _world_to_local
    from .friction import cone_generators, cone_normals

    dev = torch.device(device)
    dtype = torch.float32
    J = mid.num_joints

    # BetterHuman bakes real per-part inertia into the robot at these betas; we only
    # override gravity to the scene's world-down direction, like the dynamics stage.
    model = mid.shaped(betas.to(dev)).robot
    g = torch.as_tensor(gravity_world, dtype=dtype, device=dev)
    g = g / (g.norm() + 1e-9) * _GRAVITY_MAG
    g6 = torch.cat([g, torch.zeros(3, dtype=dtype, device=dev)])
    model = dataclasses.replace(model, values=dataclasses.replace(model.values, gravity=g6))
    total_mass = float(model.values.body_inertias[..., 1:, 0].sum())

    q_model = normalize_q(q.to(dev), J)
    N = q_model.shape[0]
    C = len(groups)
    v = torch.zeros((N, model.nv), dtype=dtype, device=dev)   # static: no velocity …
    a = torch.zeros((N, model.nv), dtype=dtype, device=dev)   # … no acceleration

    with torch.no_grad():
        oMi = forward_kinematics(model, q_model).joint_pose_world     # (N, njoints, 7)
    catalog = _contact_catalog(list(mid.joint_names))
    rep_model = [int(model.body_name_to_id[catalog[gn][0]]) for gn in groups]
    contact_ids = torch.tensor(rep_model, dtype=torch.long, device=dev)
    R_w2l = _world_to_local(oMi, contact_ids).detach()

    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=dev)
    mask = valid_t[:, None].expand(N, C).to(dtype)                     # every group allowed each valid frame
    valid_pair = (valid_t[1:] & valid_t[:-1]).to(dtype)
    fpair = mask[1:] * mask[:-1] * valid_pair[:, None]

    # Friction pyramids on the non-hand groups (feet + ankles); a joint far from a
    # coherent surface falls back to a free force, exactly like the dynamics stage.
    cone_cols = [c for c, gn in enumerate(groups) if "hand" not in gn]
    cone_w = gen = None
    if friction is not None and cone_pts is not None and cone_pts.shape[0]:
        mu = math.tan(math.radians(float(friction["half_angle_deg"])))
        joint_xyz = oMi[:, contact_ids[cone_cols], :3]
        cone_n, cone_valid = cone_normals(joint_xyz, cone_pts, cone_normals_t,
                                          int(friction["normal_k"]), float(friction["max_dist_m"]))
        cone_mask = (mask[:, cone_cols].bool() & cone_valid).to(dtype)
        if bool(cone_mask.any()):
            gen = cone_generators(cone_n, mu).detach()
            cone_w = torch.zeros((N, C, 1), dtype=dtype, device=dev)
            cone_w[:, cone_cols, 0] = cone_mask

    # Raw (unsmoothed) limb speed for the work penalty: central-difference of the
    # group joint positions, zeroed on invalid frames (their force is masked anyway).
    group_pos = oMi[:, contact_ids, :3].detach().cpu().numpy()         # (N, C, 3)
    # fps is assumed native and per-frame (the stage-wide timebase assumption): a clip
    # longer than VGGT's max_frames subsamples the frame axis but keeps native fps, so
    # this speed — and every temporal window in the stage — would be wrong there.
    vel = np.gradient(group_pos, axis=0) * float(fps)
    vel[~np.asarray(valid, bool)] = 0.0
    vmag = torch.as_tensor(np.linalg.norm(vel, axis=-1), dtype=dtype, device=dev)   # (N, C)
    dist_m = torch.as_tensor(np.minimum(np.asarray(group_dist_cm, np.float64) / 100.0, 1.0),
                             dtype=dtype, device=dev)                  # (N, C) m, capped at 1

    f_free = torch.zeros((N, C, 3), dtype=dtype, device=dev, requires_grad=True)
    gen_u = None
    if gen is not None:
        gen_u = torch.zeros((N, len(cone_cols), 4), dtype=dtype, device=dev, requires_grad=True)

    w_base = float(weights["base_wrench"])
    w_mag = float(weights["force_magnitude"])
    w_fs_mag = float(weights.get("force_smooth_mag", 0.0))
    w_fs_dir = float(weights.get("force_smooth_dir", 0.0))
    w_ts = float(weights["torque_smooth"])

    def compose_forces() -> torch.Tensor:
        if gen_u is None:
            return f_free
        lam = torch.nn.functional.softplus(gen_u)
        f_cone = torch.einsum("ncg,ncgd->ncd", lam, gen)
        full = torch.zeros((N, C, 3), dtype=dtype, device=dev)
        full[:, cone_cols] = f_cone
        return cone_w * full + (1.0 - cone_w) * f_free

    def closure():
        opt.zero_grad()
        f_world = compose_forces()
        fext, f_eff = _build_fext(f_world, mask, R_w2l, contact_ids, model.njoints)
        tau = rnea(model, q_model, v, a, fext=fext)
        loss = 0.5 * ((w_base * tau[..., :6]).pow(2).sum() + (w_mag * f_eff).pow(2).sum())
        fmag = f_eff.pow(2).sum(-1).add(1e-12).sqrt()                  # (N, C)
        if math.isfinite(vstar):
            loss = loss + 0.5 * ((fmag * vmag) / vstar).pow(2).sum()
        if math.isfinite(dstar):
            loss = loss + 0.5 * (fmag * dist_m / dstar).pow(2).sum()
        df = f_world[1:] - f_world[:-1]
        u = f_world[1:] + f_world[:-1]
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        df_par = (df * u).sum(-1, keepdim=True) * u
        loss = loss + 0.5 * w_fs_mag * (fpair[..., None] * df_par.pow(2)).sum()
        loss = loss + 0.5 * w_fs_dir * (fpair[..., None] * (df - df_par).pow(2)).sum()
        dtau = tau[1:, 6:] - tau[:-1, 6:]
        loss = loss + 0.5 * w_ts * (valid_pair[:, None] * dtau.pow(2)).sum()
        loss.backward()
        return loss

    params = [p for p in (f_free, gen_u) if p is not None]
    opt = torch.optim.LBFGS(
        params, lr=float(dyn_opt["lr"]), max_iter=int(dyn_opt["max_iter"]),
        history_size=int(dyn_opt["history_size"]), tolerance_grad=float(dyn_opt["tolerance_grad"]),
        tolerance_change=float(dyn_opt["tolerance_change"]), line_search_fn="strong_wolfe")
    with torch.no_grad():
        base0 = float(rnea(model, q_model, v, a)[..., :6].pow(2).sum().sqrt())
    opt.step(closure)
    with torch.no_grad():
        f_final = (compose_forces() * mask[..., None]).detach()
        fext, _ = _build_fext(f_final, mask, R_w2l, contact_ids, model.njoints)
        base1 = float(rnea(model, q_model, v, a, fext=fext)[..., :6].pow(2).sum().sqrt())
    forces = f_final.cpu().numpy().astype(np.float32)
    # A diverged L-BFGS line search can return NaN/Inf forces; raising here lets
    # complete_person's per-person except fall back to gates-only for this person.
    if not (np.isfinite(forces).all() and math.isfinite(total_mass) and total_mass > 0.0):
        raise FloatingPointError("static support solve produced non-finite forces or mass")
    return forces, total_mass, base0, base1


def complete_person(mid, betas: torch.Tensor, q: torch.Tensor, valid: np.ndarray,
                    joint_names: list[str], *, detector_contact: np.ndarray,
                    joint_distance_cm: np.ndarray, gate_drift, gravity_world: np.ndarray,
                    fps: float, cone_pts, cone_normals_t, ctx: DynamicsContext,
                    knobs: dict, device: str) -> dict:
    """Static solve + gate/force labels for one person → the completion result dict.

    ``detector_contact`` and ``joint_distance_cm`` are the stage's per-joint
    ``(N, J)`` detector labels and scene distances; ``gate_drift`` is its per-joint
    drift (prefers the persisted ``drift_pre``). Returns per-group ``(N, C)`` masks
    ``gate_span``/``force_span`` (already min-span filtered, invalid frames cleared
    first), ``limb_force_n`` (solved |f| in N, zero on invalid frames), ``rep_idx``
    (each group's own joint index), ``body_weight_n`` (total baked mass × g, or
    ``inf`` when the solve failed), and the base-wrench provenance numbers.
    """
    groups = list(ctx.groups)
    C = len(groups)
    N = detector_contact.shape[0]

    from .dynamics import _GRAVITY_MAG, _contact_catalog

    catalog = _contact_catalog(joint_names)
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    rep_idx = [name_to_idx[catalog[gn][0]] for gn in groups]
    member_idx = [[name_to_idx[m] for m in catalog[gn][1] if m in name_to_idx] for gn in groups]

    gate_drift_np = (gate_drift.detach().cpu().numpy()
                     if torch.is_tensor(gate_drift) else np.asarray(gate_drift))
    dist = np.asarray(joint_distance_cm, np.float64)
    group_dist = np.empty((N, C), np.float64)
    group_drift = np.empty((N, C), np.float64)
    for c, members in enumerate(member_idx):
        # NaN distance = no nearby scene point → treat as far so one pruned member
        # cannot poison the group minimum. Drift keeps NaN (an invalid window fails
        # the gate anyway).
        d = np.where(np.isnan(dist[:, members]), np.inf, dist[:, members])
        group_dist[:, c] = d.min(axis=1)
        group_drift[:, c] = gate_drift_np[:, members].min(axis=1)

    valid_b = np.asarray(valid, bool)
    try:
        forces, total_mass, base0, base1 = solve_static_support(
            mid, betas, q, valid_b, gravity_world, fps,
            group_dist_cm=group_dist, groups=groups, weights=ctx.weights, dyn_opt=ctx.dyn_opt,
            friction=ctx.friction, cone_pts=cone_pts, cone_normals_t=cone_normals_t,
            vstar=float(knobs["work_vstar_mps"]), dstar=float(knobs["gap_dstar_m"]), device=device)
        body_weight_n = total_mass * _GRAVITY_MAG
    except Exception as exc:  # noqa: BLE001 — one failed solve must not kill the stage
        print(f"[estimate_contacts] completion solve failed ({exc}); gates-only for this person")
        forces = np.zeros((N, C, 3), np.float32)
        body_weight_n = np.inf         # force term can never fire without a solve
        base0 = base1 = float("nan")

    fmag = np.linalg.norm(forces, axis=-1)                            # (N, C) newtons

    # Component masks. Invalid frames are cleared BEFORE the min-span filter so a
    # stray invalid frame can never bridge a short run into surviving it.
    gate = (group_dist < float(knobs["gate_dist_cm"])) & (group_drift < float(knobs["gate_drift_m"]))
    force = (fmag > float(knobs["force_theta_bw"]) * body_weight_n) & (group_drift < float(knobs["force_drift_m"]))
    gate &= valid_b[:, None]
    force &= valid_b[:, None]
    ms = int(knobs["min_span"])
    return dict(
        rep_idx=rep_idx,
        gate_span=_min_span(gate, ms),
        force_span=_min_span(force, ms),
        limb_force_n=(fmag * valid_b[:, None]).astype(np.float32),
        body_weight_n=float(body_weight_n),
        base_before=base0, base_after=base1,
    )
