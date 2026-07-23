"""The dynamics stage: force-only RNEA on a refined body.

With the kinematic trajectory ``q`` frozen, one L-BFGS solve per person finds
world-frame contact forces at the ``[contact].groups`` joints (hands, feet,
ankles) such that the RNEA base wrench — the residual 6-DOF force/torque on the
un-actuated free-flyer root — vanishes. Feet and ankles push through friction
pyramids built from scene normals (:mod:`tools.human_optim.friction`); hands
keep a free bilateral force (they grasp). Entry point: :func:`run_dynamics`,
driven by ``scripts/stages/human_optimization.py --stages dynamics``.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import torch

from tools.geometry.normals import load_scene_cloud
from tools.smplx_robot import load_smplx_mid
from tools.smplx_robot.dynamics import velocity_acceleration_from_trajectory

from . import paths
from .camera import world_from_cam
from .config import HumanOptimConfig
from .friction import cone_generators, cone_normals
from .qmath import normalize_q
from .stages import _DEFAULT_SMPLX_MODEL, _FINGER

_GRAVITY_MAG = 9.81  # m/s²


def _contact_catalog(joint_names: list[str]) -> dict[str, tuple[str, list[str]]]:
    """``group_name -> (representative_joint, member_joints)`` for SMPLXMid.

    A hand folds its wrist + finger joints onto the wrist (fingers are too small to
    carry their own force point); every other group is a single joint. Foot and
    ankle stay **separate** so a planted foot can push through two contact points —
    the six default groups are the two hands, two feet and two ankles.
    """
    have = set(joint_names)
    cat: dict[str, tuple[str, list[str]]] = {}
    for side in ("left", "right"):
        hand = [f"{side}_wrist"] + [
            n for n in joint_names
            if n.startswith(f"{side}_") and any(f in n for f in _FINGER)
        ]
        cat[f"{side}_hand"] = (f"{side}_wrist", [n for n in hand if n in have])
        for jn in ("foot", "ankle", "knee", "elbow", "hip"):
            n = f"{side}_{jn}"
            if n in have:
                cat[f"{side}_{jn}"] = (n, [n])
    return cat


def _world_to_local(oMi: torch.Tensor, contact_ids: torch.Tensor) -> torch.Tensor:
    """``R_world_to_local (T, C, 3, 3)`` at the contact joints (q frozen ⇒ static)."""
    from better_robot.lie import so3
    R_lw = so3.to_matrix(oMi[:, contact_ids, 3:7])     # local→world
    return R_lw.transpose(-1, -2).contiguous()


def _build_fext(f_world, mask, R_w2l, contact_ids, njoints):
    """``(fext (T, njoints, 6), f_eff (T, C, 3))`` — masked world force → body wrench."""
    T, C, _ = f_world.shape
    f_eff = f_world * mask[..., None]
    f_local = torch.einsum("tcij,tcj->tci", R_w2l, f_eff)
    fext_lin = torch.zeros((T, njoints, 3), dtype=f_world.dtype, device=f_world.device)
    index = contact_ids.view(1, C, 1).expand(T, C, 3)
    fext_lin = fext_lin.scatter(dim=1, index=index, src=f_local)
    fext = torch.cat([fext_lin, torch.zeros_like(fext_lin)], dim=-1)
    return fext, f_eff


def _solve_person(mid, betas, q_bh, contact_bool, valid, gravity_world, *,
                  weights, groups, dyn_opt, fps, device, label="",
                  scene_pts=None, scene_normals=None, friction=None):
    """Solve world-frame contact forces for one person. Returns a result dict."""
    from better_robot.dynamics import rnea
    from better_robot.kinematics import forward_kinematics

    dev = torch.device(device)
    dtype = torch.float32
    J = mid.num_joints
    N = q_bh.shape[0]
    dt = 1.0 / float(fps)

    # BetterHuman already bakes the BetterRobot model with real per-part inertia at
    # these betas (closed part meshes — not convex hulls; native q order → no remap).
    # We only override gravity to the scene's world-down direction.
    model = mid.shaped(betas.to(dev)).robot
    g = torch.as_tensor(gravity_world, dtype=dtype, device=dev)
    g = g / (g.norm() + 1e-9) * _GRAVITY_MAG
    g6 = torch.cat([g, torch.zeros(3, dtype=dtype, device=dev)])
    model = dataclasses.replace(model, values=dataclasses.replace(model.values, gravity=g6))
    total_mass = float(model.values.body_inertias[..., 1:, 0].sum())

    q_model = normalize_q(q_bh.to(dev), J)
    with torch.no_grad():
        v, a = velocity_acceleration_from_trajectory(model, q_model, dt)
        oMi = forward_kinematics(model, q_model).joint_pose_world          # (N, njoints, 7)

    catalog = _contact_catalog(list(mid.joint_names))
    name_to_bh = {n: i for i, n in enumerate(mid.joint_names)}
    group_names = [g for g in groups if g in catalog]
    rep_bh, rep_model, masks = [], [], []
    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=dev)
    for gname in group_names:
        rep, members = catalog[gname]
        member_idx = [name_to_bh[m] for m in members if m in name_to_bh]
        active = contact_bool[:, member_idx].any(dim=1) & valid_t              # (N,)
        rep_bh.append(name_to_bh[rep])
        rep_model.append(int(model.body_name_to_id[rep]))
        masks.append(active)
    C = len(group_names)
    mask = torch.stack(masks, dim=1).to(dtype) if C else torch.zeros((N, 0), dtype=dtype, device=dev)
    contact_ids = torch.tensor(rep_model, dtype=torch.long, device=dev)
    R_w2l = _world_to_local(oMi, contact_ids).detach()

    valid_pair = (valid_t[1:] & valid_t[:-1]).to(dtype)                    # (N-1,)
    fpair = (mask[1:] * mask[:-1] * valid_pair[:, None]) if C else None    # (N-1, C)

    # Friction cones on the non-hand contact groups (feet + ankles): a wall/floor contact can
    # only push, and its tangential force is capped at μ·(normal force); hands are bilateral
    # (grasp + pull) and keep a free force vector. The cone is enforced BY CONSTRUCTION: on
    # cone entries the force is a softplus-nonnegative combination of the four pyramid edge
    # directions, so it cannot leave the pyramid inscribed in the cone — no penalty term, no
    # weight. Normals come from the scene cloud, detached (q frozen ⇒ joint positions static);
    # a cone entry without a usable normal (far surface, incoherent patch) falls back to a
    # free vector, like a hand.
    cone_cols = [c for c, gn in enumerate(group_names) if "hand" not in gn]
    cone_n = cone_mask = cone_w = gen = None
    mu = math.tan(math.radians(float(friction["half_angle_deg"]))) if friction else 1.0
    if cone_cols and friction is not None and scene_pts is not None and scene_pts.shape[0]:
        cone_ids = contact_ids[cone_cols]
        joint_xyz = oMi[:, cone_ids, :3]                                   # (N, Cc, 3)
        cone_n, cone_valid = cone_normals(
            joint_xyz, scene_pts, scene_normals,
            int(friction["normal_k"]), float(friction["max_dist_m"]))
        cone_mask = (mask[:, cone_cols].bool() & cone_valid).to(dtype)     # (N, Cc)
        if bool(cone_mask.any()):
            gen = cone_generators(cone_n, mu).detach()                    # (N, Cc, 4, 3)
            cone_w = torch.zeros((N, C, 1), dtype=dtype, device=dev)
            cone_w[:, cone_cols, 0] = cone_mask

    def cone_violation(f: torch.Tensor) -> torch.Tensor:
        """``relu(‖f_t‖ − μ·f_n)`` per (frame, cone joint) vs the TRUE cone — a readout.

        ``f_n = f·n̂`` (signed normal component), ``f_t = f − f_n n̂``. Generator forces
        lie in the pyramid inscribed in the cone, so this should be ~0; it is printed
        as an invariant check, not optimized."""
        fc = f[:, cone_cols]                                               # (N, Cc, 3)
        f_n = (fc * cone_n).sum(-1)                                        # (N, Cc)
        f_t = fc - f_n[..., None] * cone_n
        ft = f_t.pow(2).sum(-1).add(1e-12).sqrt()
        return torch.relu(ft - mu * f_n)

    # Leaves: a free world force per (frame, group), and — on cone entries — four
    # generator activations whose softplus replaces it. softplus (not relu/square) so no
    # activation has a dead zero-gradient point and the map is smooth for L-BFGS.
    f_free = torch.zeros((N, C, 3), dtype=dtype, device=dev, requires_grad=True)
    gen_u = None
    if gen is not None:
        gen_u = torch.zeros((N, len(cone_cols), 4), dtype=dtype, device=dev, requires_grad=True)

    def compose_forces() -> torch.Tensor:
        """World force ``(N, C, 3)``: pyramid-generator force on cone entries, free elsewhere."""
        if gen_u is None:
            return f_free
        lam = torch.nn.functional.softplus(gen_u)                          # (N, Cc, 4) ≥ 0
        f_cone = torch.einsum("ncg,ncgd->ncd", lam, gen)                   # (N, Cc, 3) in the pyramid
        full = torch.zeros((N, C, 3), dtype=dtype, device=dev)
        full[:, cone_cols] = f_cone
        return cone_w * full + (1.0 - cone_w) * f_free

    w_base = float(weights["base_wrench"])
    w_mag = float(weights["force_magnitude"])
    w_fs_mag = float(weights.get("force_smooth_mag", 0.0))
    w_fs_dir = float(weights.get("force_smooth_dir", 0.0))
    w_ts = float(weights["torque_smooth"])

    def closure():
        opt.zero_grad()
        f_world = compose_forces()
        fext, f_eff = _build_fext(f_world, mask, R_w2l, contact_ids, model.njoints)
        tau = rnea(model, q_model, v, a, fext=fext)
        loss = 0.5 * ((w_base * tau[..., :6]).pow(2).sum() + (w_mag * f_eff).pow(2).sum())
        if N > 1:
            # Split force smoothness into magnitude vs direction via the orthogonal decomposition
            # about the mean direction u: df_par is the |f|-change part, df_perp the direction part.
            df = f_world[1:] - f_world[:-1]                                # (N-1, C, 3)
            u = f_world[1:] + f_world[:-1]
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            df_par = (df * u).sum(-1, keepdim=True) * u
            df_perp = df - df_par
            loss = loss + 0.5 * w_fs_mag * (fpair[..., None] * df_par.pow(2)).sum()
            loss = loss + 0.5 * w_fs_dir * (fpair[..., None] * df_perp.pow(2)).sum()
            dtau = tau[1:, 6:] - tau[:-1, 6:]                             # (N-1, nv-6)
            loss = loss + 0.5 * w_ts * (valid_pair[:, None] * dtau.pow(2)).sum()
        loss.backward()
        return loss

    if C > 0 and bool(mask.any()):
        opt = torch.optim.LBFGS(
            [f_free] if gen_u is None else [f_free, gen_u],
            lr=float(dyn_opt["lr"]), max_iter=int(dyn_opt["max_iter"]),
            history_size=int(dyn_opt["history_size"]), tolerance_grad=float(dyn_opt["tolerance_grad"]),
            tolerance_change=float(dyn_opt["tolerance_change"]), line_search_fn="strong_wolfe")
        with torch.no_grad():
            base0 = float((rnea(model, q_model, v, a)[..., :6]).pow(2).sum().sqrt())
        opt.step(closure)
    else:
        base0 = float("nan")

    with torch.no_grad():
        f_final = (compose_forces() * mask[..., None]).detach()               # (N, C, 3)
        fext, _ = _build_fext(f_final, mask, R_w2l, contact_ids, model.njoints)
        tau = rnea(model, q_model, v, a, fext=fext)                           # (N, nv)
        base_wrench = tau[..., :6].cpu().numpy().astype(np.float32)           # (N, 6): force | torque
        base1 = float(tau[..., :6].pow(2).sum().sqrt())

        idx_vs = [int(x) for x in model.idx_vs]
        nvs = [int(x) for x in model.nvs]
        torque_mag = np.zeros((N, J), np.float32)
        joints_world = np.zeros((N, J, 3), np.float32)
        force_world = np.zeros((N, J, 3), np.float32)
        for jb in range(J):
            mb = int(model.body_name_to_id[mid.joint_names[jb]])
            n = nvs[mb]
            if n > 0:
                torque_mag[:, jb] = tau[:, idx_vs[mb]:idx_vs[mb] + n].norm(dim=-1).cpu().numpy()
            joints_world[:, jb] = oMi[:, mb, :3].cpu().numpy()
        for c, jb in enumerate(rep_bh):
            force_world[:, jb] = f_final[:, c].cpu().numpy()

    inval = ~np.asarray(valid, bool)
    torque_mag[inval] = 0.0
    force_world[inval] = 0.0
    base_wrench[inval] = 0.0
    fnorm = f_final.norm(dim=-1)                                              # (N, C)
    act = mask.bool().any(dim=1).cpu().numpy()
    mean_f = float(fnorm.sum(dim=-1)[mask.bool().any(dim=1)].mean()) if act.any() else 0.0

    fjit_mag = fjit_dir = tjit = 0.0
    if N > 1:
        with torch.no_grad():
            df = f_final[1:] - f_final[:-1]                                   # (N-1, C, 3)
            u = f_final[1:] + f_final[:-1]
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            dperp = (df - (df * u).sum(-1, keepdim=True) * u).norm(dim=-1)    # (N-1, C) direction jitter
            dmag = (f_final[1:].norm(dim=-1) - f_final[:-1].norm(dim=-1)).abs()  # (N-1, C) magnitude jitter
            fp = (mask[1:] * mask[:-1]).bool()
            if bool(fp.any()):
                fjit_mag = float(dmag[fp].mean())
                fjit_dir = float(dperp[fp].mean())
            dtq = (tau[1:, 6:] - tau[:-1, 6:]).norm(dim=-1)                   # (N-1,)
            vp = valid_pair.bool()
            tjit = float(dtq[vp].mean()) if bool(vp.any()) else 0.0

    # Cone violation of the solved forces vs the TRUE cone (Newtons) — generator forces live
    # in the inscribed pyramid, so this must read ~0; a nonzero value means a broken invariant.
    cone_pairs, cviol_mean, cviol_max = 0, 0.0, 0.0
    if cone_mask is not None:
        with torch.no_grad():
            viol = cone_violation(f_final)[cone_mask.bool()]
            cone_pairs = int(viol.numel())
            if cone_pairs:
                cviol_mean, cviol_max = float(viol.mean()), float(viol.max())

    cone_str = f"  cone[{cone_pairs}] viol μ={cviol_mean:.1f} max={cviol_max:.1f} N" if cone_mask is not None else ""
    print(f"  [{label}] mass={total_mass:.1f}kg  contact-frames={int(act.sum())}/{N}  "
          f"base wrench {base0:.1f}→{base1:.2f}  mean|f|={mean_f:.0f} N "
          f"(weight≈{total_mass * _GRAVITY_MAG:.0f} N)  jitter |Δ|f||={fjit_mag:.1f} |Δf⊥|={fjit_dir:.1f} N "
          f"|Δτ|={tjit:.2f}{cone_str}")

    return dict(
        joints_world=joints_world, torque_mag=torque_mag, force_world=force_world,
        base_wrench=base_wrench,
        contact_forces=f_final.cpu().numpy().astype(np.float32),
        joint_torques=tau.cpu().numpy().astype(np.float32),
        total_mass=np.float32(total_mass),
        rep_joint_names=[mid.joint_names[j] for j in rep_bh],
        base_wrench_before=np.float32(base0), base_wrench_after=np.float32(base1),
    )


def run_dynamics(pred: Path, cfg: HumanOptimConfig, device: str, *, bodies="kinematic_1") -> Path:
    """Force-only RNEA on a refined body → ``human_optim/dynamics.npz``."""
    knobs = cfg.knobs
    weights = cfg.weights_for("dynamics")
    pred = Path(pred)

    body = "kinematic_2" if bodies in ("kinematics_2", "kinematic_2") else "kinematic_1"
    kin_path = paths.body_path(pred, body)
    cnt_path = paths.contacts_path(pred, "2" if body == "kinematic_2" else "1")
    grav_path = pred / "geocalib" / "gravity.npz"
    for pth in (kin_path, cnt_path):
        if not pth.exists():
            raise SystemExit(f"missing {pth} — run human_optimization {body} + estimate_contacts first")

    kin = dict(np.load(kin_path, allow_pickle=True))
    cnt = dict(np.load(cnt_path, allow_pickle=True))
    fps = float(kin["fps"])

    N, P = int(kin["num_frames"]), int(kin["num_people"])
    object_ids = np.asarray(kin["object_ids"], np.int32)
    cnt_oids = list(np.asarray(cnt["object_ids"], np.int32))
    joint_contact = np.asarray(cnt["joint_contact"], bool)                    # (Pc, N, 52)

    if grav_path.exists():
        gd = np.load(grav_path)
        gravity_world = np.asarray(gd["gravity_world"], np.float32)
        gsrc = str(gd["source"]) if "source" in gd.files else "geocalib"
    else:
        gravity_world = np.array([0.0, 1.0, 0.0], np.float32)                 # VGGT world-down fallback
        gsrc = "fallback_down(no geocalib)"
    print(f"[human_optim/dynamics] {pred.name}: P={P} N={N} fps={fps:.1f}  bodies={bodies}  "
          f"gravity={gravity_world.round(2)} ({gsrc})")

    # Scene cloud for the friction cones (feet/ankles). A missing scene or [friction] section
    # ⇒ no cone (old artifacts still run). Camera centers orient the fuse-stored normals.
    friction = knobs.get("friction")
    scene_pts = scene_normals = None
    scene_path = pred / "geometry" / "scene.npz"
    transform_path = pred / "geometry" / "transform.npz"
    if friction is not None and scene_path.exists() and transform_path.exists():
        sc = dict(np.load(scene_path, allow_pickle=True))
        tr = dict(np.load(transform_path, allow_pickle=True))
        cam_centers = world_from_cam(
            torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device))[:, :3, 3]
        scene_pts, scene_normals, _ = load_scene_cloud(
            sc, cam_centers, conf_drop_q=0.0, k=int(friction["normal_k"]), device=device)
        print(f"[human_optim/dynamics] friction cones ON ({scene_pts.shape[0]} scene pts)")
    else:
        print("[human_optim/dynamics] no scene/[friction] — friction cones OFF")

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    J = mid.num_joints
    joints_world = np.zeros((P, N, J, 3), np.float32)
    torque_mag = np.zeros((P, N, J), np.float32)
    contact_forces_world = np.zeros((P, N, J, 3), np.float32)
    joint_torques = np.zeros((P, N, 6 + 3 * (J - 1)), np.float32)
    base_wrench = np.zeros((P, N, 6), np.float32)                              # root residual force | torque
    total_mass = np.zeros(P, np.float32)
    rep_names_p: list = []
    raw_forces_p: list = []

    for p in range(P):
        oid = int(object_ids[p])
        cp = cnt_oids.index(oid) if oid in cnt_oids else p
        res = _solve_person(
            mid, torch.as_tensor(kin["betas"][p], dtype=torch.float32),
            torch.as_tensor(kin["q"][p], dtype=torch.float32),
            torch.as_tensor(joint_contact[cp], dtype=torch.bool, device=device),
            np.asarray(kin["valid_mask"][p], bool),
            gravity_world, weights=weights,
            groups=knobs["contact"]["groups"], dyn_opt=knobs["dynamics_optimizer"],
            fps=fps, device=device, label=f"obj {oid}",
            scene_pts=scene_pts, scene_normals=scene_normals, friction=friction,
        )
        joints_world[p] = res["joints_world"]
        torque_mag[p] = res["torque_mag"]
        contact_forces_world[p] = res["force_world"]
        joint_torques[p] = res["joint_torques"]
        base_wrench[p] = res["base_wrench"]
        total_mass[p] = res["total_mass"]
        rep_names_p.append(res["rep_joint_names"])
        raw_forces_p.append(res["contact_forces"])

    out_path = paths.dynamics_path(pred)
    np.savez(
        out_path,
        joints_world=joints_world, torque_mag=torque_mag,
        contact_forces_world=contact_forces_world, joint_torques=joint_torques,
        base_wrench=base_wrench,
        contact_forces=np.stack(raw_forces_p, 0),
        contact_force_joints=np.array(rep_names_p, dtype="<U16"),
        total_mass=total_mass, gravity_world=gravity_world.astype(np.float32),
        joint_contact=joint_contact[:P] if joint_contact.shape[0] >= P else joint_contact,
        object_ids=object_ids, joint_names=kin["joint_names"],
        parents=np.asarray(kin["parents"], np.int32), valid_mask=kin["valid_mask"],
        num_people=np.int32(P), num_frames=np.int32(N),
        video_height=np.int32(kin["video_height"]), video_width=np.int32(kin["video_width"]),
        fps=np.float32(fps), frame="world", source_bodies=str(bodies),
    )
    print(f"[human_optim/dynamics] wrote {out_path}")
    return out_path
