"""Contact completion by residual-driven force solve with physical penalties.

The follow-up to ``missing_contacts.py``: instead of *scoring* candidates by
the residual, this makes the residual the **objective**. One solve per person
with ALL six contact groups allowed on every valid frame; besides the dynamics
stage's own terms (base wrench, magnitude prior, smoothness, friction pyramids)
two penalties encode plausibility without thresholds:

* **work**  — ``(|f|·|v| / v*)²`` per (frame, group): a real contact does not
  slide, so force at a moving limb pays its mechanical work. The scale ``v*``
  is a soft velocity: force is worth carrying roughly iff the limb moves
  slower than ``v*`` (form ``dot`` uses the signed power ``f·v`` instead).
* **gap**   — ``(|f|·d / d*)²``: force cannot act across a gap; ``d*`` is a
  soft distance. Distances are broken on floater scenes (whole body off the
  wall), so ``d* = inf`` disables the term — the work penalty survives there.

The completion *score* is the solved force magnitude per limb; evaluation
mirrors ``missing_contacts.py`` (pooled candidate AUROC/AP + GT from
``data/<stem>.npz``). Reads finished ``out/<stem>/`` trees, writes per-run
scores to ``scratch/missing_contacts/completion/``, modifies nothing.

Usage::

    uv run python scripts/diagnostics/completion_solve.py \\
        --scenes boulder_1,2nwG1Qa5k_c_0115,competition_2 \\
        --vstar 0.1,0.2,0.4 --dstar 0.05,0.1,0.2,inf      # grid mode
    uv run python scripts/diagnostics/completion_solve.py --vstar 0.2 --dstar 0.1
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "diagnostics")):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import itertools  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from missing_contacts import (  # noqa: E402
    PersonDynamics, evaluate_scene, fold_limbs, load_scene)
from tools.human_optim.config import load_config  # noqa: E402
from tools.human_optim.dynamics import _build_fext, _contact_catalog, _world_to_local  # noqa: E402
from tools.human_optim.friction import cone_generators, cone_normals  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"
_OUT_DIR = _REPO_ROOT / "scratch" / "missing_contacts" / "completion"


def _limb_velocity(group_pos: np.ndarray, valid: np.ndarray, fps: float,
                   sigma: float = 0.0) -> np.ndarray:
    """Central-difference world velocity of the group representative joints.

    ``group_pos``: (N, C, 3) -> (N, C, 3) m/s; invalid frames get zero velocity
    (no work penalty there — their force is masked out anyway). ``sigma`` > 0
    applies a temporal Gaussian to the velocity (same de-noising as the wrench).
    """
    v = np.gradient(group_pos, axis=0) * fps
    if sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        v = gaussian_filter1d(v, sigma, axis=0)
    v[~valid] = 0.0
    return v


def solve_completion(scene: dict, person: dict, pd: PersonDynamics, cfg,
                     device: str, *, vstar: float, dstar: float,
                     form: str = "mag", vel_sigma: float = 0.0,
                     solve_gravity: bool = False, rel_gap: bool = False) -> dict:
    """One all-groups-allowed force solve with work + gap penalties.

    Returns ``dict(forces (N, C, 3) world, base_before, base_after)``. The
    dynamics-stage terms use the stage's own weights; ``vstar``/``dstar`` are
    the soft velocity/distance scales (``inf`` disables a penalty).
    ``solve_gravity`` co-optimizes the gravity direction (2 DOF, magnitude
    pinned) using the fact that RNEA is linear in the gravity vector.
    """
    import dataclasses

    from better_robot.dynamics import rnea
    from better_robot.kinematics import forward_kinematics
    from tools.human_optim.dynamics import _GRAVITY_MAG

    dev = torch.device(device)
    dtype = torch.float32
    model, q_model, v, a = pd.model, pd.q_model, pd.v, pd.a
    N = q_model.shape[0]
    weights = cfg.weights_for("dynamics")
    dyn_opt = cfg.knobs["dynamics_optimizer"]
    groups = scene["groups"]
    C = len(groups)

    with torch.no_grad():
        oMi = forward_kinematics(model, q_model).joint_pose_world

    catalog = _contact_catalog(scene["joint_names"])
    rep_model = [int(model.body_name_to_id[catalog[g][0]]) for g in groups]
    contact_ids = torch.tensor(rep_model, dtype=torch.long, device=dev)
    R_w2l = _world_to_local(oMi, contact_ids).detach()

    valid_t = torch.as_tensor(person["valid"], dtype=torch.bool, device=dev)
    mask = valid_t[:, None].expand(N, C).to(dtype)                        # all groups allowed
    valid_pair = (valid_t[1:] & valid_t[:-1]).to(dtype)
    fpair = mask[1:] * mask[:-1] * valid_pair[:, None]

    # Friction pyramids, exactly as the stage builds them (free fallback far away).
    friction = cfg.knobs.get("friction")
    cone_cols = [c for c, gn in enumerate(groups) if "hand" not in gn]
    cone_w = gen = None
    if friction is not None and scene.get("scene_pts") is not None:
        mu = math.tan(math.radians(float(friction["half_angle_deg"])))
        joint_xyz = oMi[:, contact_ids[cone_cols], :3]
        cone_n, cone_valid = cone_normals(joint_xyz, scene["scene_pts"], scene["scene_normals"],
                                          int(friction["normal_k"]), float(friction["max_dist_m"]))
        cone_mask = (mask[:, cone_cols].bool() & cone_valid).to(dtype)
        if bool(cone_mask.any()):
            gen = cone_generators(cone_n, mu).detach()
            cone_w = torch.zeros((N, C, 1), dtype=dtype, device=dev)
            cone_w[:, cone_cols, 0] = cone_mask

    # Plausibility scales, precomputed per (frame, group), detached.
    group_pos = person["group_pos"]
    if group_pos is None:                              # no dynamics.npz (DB scenes): use FK
        group_pos = oMi[:, contact_ids, :3].cpu().numpy()
    vel = _limb_velocity(group_pos, person["valid"], scene["fps"], vel_sigma)
    vmag = torch.as_tensor(np.linalg.norm(vel, axis=-1), dtype=dtype, device=dev)  # (N, C)
    vvec = torch.as_tensor(vel, dtype=dtype, device=dev)                  # (N, C, 3)
    dist_np = np.minimum(person["group_dist"] / 100.0, 1.0)                  # (N, C) m
    if rel_gap:
        # Charge the gap RELATIVE to the frame's closest limb: immune to a
        # whole-body offset (floaters) while still ranking limbs within a frame.
        dist_np = dist_np - dist_np.min(axis=1, keepdims=True)
    dist_m = torch.as_tensor(dist_np, dtype=dtype, device=dev)

    f_free = torch.zeros((N, C, 3), dtype=dtype, device=dev, requires_grad=True)
    gen_u = None
    if gen is not None:
        gen_u = torch.zeros((N, len(cone_cols), 4), dtype=dtype, device=dev, requires_grad=True)

    # Gravity as a 2-DOF variable: RNEA is linear in g, so tau(g, fext) =
    # rnea(g=0, fext) + T_g @ g with T_g precomputed from three basis solves.
    g_leaf = model_g0 = T_g = None
    lin_err = 0.0
    if solve_gravity:
        g_now = model.values.gravity[:3]
        zero6 = torch.zeros(6, dtype=dtype, device=dev)
        model_g0 = dataclasses.replace(
            model, values=dataclasses.replace(model.values, gravity=zero6))
        with torch.no_grad():
            tau_zero = rnea(model_g0, q_model, v, a)
            cols = []
            for k in range(3):
                gk = torch.zeros(6, dtype=dtype, device=dev)
                gk[k] = _GRAVITY_MAG
                mk = dataclasses.replace(
                    model, values=dataclasses.replace(model.values, gravity=gk))
                cols.append((rnea(mk, q_model, v, a) - tau_zero) / _GRAVITY_MAG)
            T_g = torch.stack(cols, dim=-1)                               # (N, dof, 3)
            lin_err = float((tau_zero + torch.einsum("njk,k->nj", T_g, g_now)
                             - rnea(model, q_model, v, a)).abs().max())
        g_leaf = (g_now / g_now.norm().clamp_min(1e-9)).clone().requires_grad_(True)

    def compose_forces() -> torch.Tensor:
        if gen_u is None:
            return f_free
        lam = torch.nn.functional.softplus(gen_u)
        f_cone = torch.einsum("ncg,ncgd->ncd", lam, gen)
        full = torch.zeros((N, C, 3), dtype=dtype, device=dev)
        full[:, cone_cols] = f_cone
        return cone_w * full + (1.0 - cone_w) * f_free

    w_base = float(weights["base_wrench"])
    w_mag = float(weights["force_magnitude"])
    w_fs_mag = float(weights.get("force_smooth_mag", 0.0))
    w_fs_dir = float(weights.get("force_smooth_dir", 0.0))
    w_ts = float(weights["torque_smooth"])

    def current_tau(fext: torch.Tensor) -> torch.Tensor:
        if not solve_gravity:
            return rnea(model, q_model, v, a, fext=fext)
        gvec = g_leaf / g_leaf.norm().clamp_min(1e-9) * _GRAVITY_MAG
        return rnea(model_g0, q_model, v, a, fext=fext) + torch.einsum("njk,k->nj", T_g, gvec)

    def closure():
        opt.zero_grad()
        f_world = compose_forces()
        fext, f_eff = _build_fext(f_world, mask, R_w2l, contact_ids, model.njoints)
        tau = current_tau(fext)
        loss = 0.5 * ((w_base * tau[..., :6]).pow(2).sum() + (w_mag * f_eff).pow(2).sum())
        fmag = f_eff.pow(2).sum(-1).add(1e-12).sqrt()                     # (N, C)
        if math.isfinite(vstar):
            work = (f_eff * vvec).sum(-1) if form == "dot" else fmag * vmag
            loss = loss + 0.5 * (work / vstar).pow(2).sum()
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

    params = [p for p in (f_free, gen_u, g_leaf) if p is not None]
    opt = torch.optim.LBFGS(
        params, lr=float(dyn_opt["lr"]), max_iter=int(dyn_opt["max_iter"]),
        history_size=int(dyn_opt["history_size"]),
        tolerance_grad=float(dyn_opt["tolerance_grad"]),
        tolerance_change=float(dyn_opt["tolerance_change"]), line_search_fn="strong_wolfe")
    with torch.no_grad():
        base0 = float(rnea(model, q_model, v, a)[..., :6].pow(2).sum().sqrt())
    opt.step(closure)
    with torch.no_grad():
        f_final = (compose_forces() * mask[..., None]).detach()
        fext, _ = _build_fext(f_final, mask, R_w2l, contact_ids, model.njoints)
        base1 = float(current_tau(fext)[..., :6].pow(2).sum().sqrt())
        g_out = (g_leaf / g_leaf.norm().clamp_min(1e-9)).cpu().numpy() \
            if solve_gravity else None
    return dict(forces=f_final.cpu().numpy().astype(np.float32),
                base_before=base0, base_after=base1,
                gravity_dir=g_out, lin_err=lin_err)


def _load_cones(scene: dict, cfg, device: str) -> None:
    """Attach the scene cloud + normals for the friction pyramids (stage-identical)."""
    friction = cfg.knobs.get("friction")
    sc_path = scene["pred"] / "geometry" / "scene.npz"
    tr_path = scene["pred"] / "geometry" / "transform.npz"
    scene["scene_pts"] = scene["scene_normals"] = None
    if friction is None or not (sc_path.exists() and tr_path.exists()):
        return
    from tools.geometry.normals import load_scene_cloud
    from tools.human_optim.camera import world_from_cam
    tr = dict(np.load(tr_path, allow_pickle=True))
    cam_centers = world_from_cam(
        torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device))[:, :3, 3]
    scene["scene_pts"], scene["scene_normals"], _ = load_scene_cloud(
        dict(np.load(sc_path, allow_pickle=True)), cam_centers,
        conf_drop_q=0.0, k=int(cfg.knobs["friction"]["normal_k"]), device=device)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", default=None, help="comma-separated stems "
                                                   "(default: all data/ scenes except sport_climb)")
    ap.add_argument("--vstar", default="0.2", help="soft velocity scale(s) m/s, csv, inf disables")
    ap.add_argument("--dstar", default="0.1", help="soft distance scale(s) m, csv, inf disables")
    ap.add_argument("--form", default="mag", choices=("mag", "dot"),
                    help="work penalty: |f|·|v| (mag) or signed power f·v (dot)")
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="temporal Gaussian (frames) on v/a and limb velocity; 0 = off")
    ap.add_argument("--raw-vel", action="store_true",
                    help="keep the work penalty's limb velocity UNsmoothed (fast limbs stay "
                         "expensive even when sigma > 0)")
    ap.add_argument("--rel-gap", action="store_true",
                    help="gap penalty on distance relative to the frame's closest limb")
    ap.add_argument("--null-static", action="store_true",
                    help="NULL experiment: zero v/a so the residual is the pure static "
                         "gravity wrench (no motion information)")
    ap.add_argument("--gravity", default="fallback", choices=("fallback", "geocalib", "solve"),
                    help="gravity: stage value / GeoCalib raw estimate / co-optimized 2-DOF")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.scenes:
        stems = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        stems = sorted(p.stem for p in (_REPO_ROOT / "data").glob("*.npz")
                       if p.stem != "sport_climb")
    vstars = [float(x) for x in args.vstar.split(",")]
    dstars = [float(x) for x in args.dstar.split(",")]

    from tools.human_optim.stages import _DEFAULT_SMPLX_MODEL
    from tools.smplx_robot import load_smplx_mid
    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, args.device)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[tuple, list] = {}
    suffix = (f"_s{args.sigma:g}" if args.sigma > 0 else "") \
        + ("_rv" if args.raw_vel else "") + ("_rg" if args.rel_gap else "") \
        + (f"_g{args.gravity}" if args.gravity != "fallback" else "") \
        + ("_null" if args.null_static else "")
    for stem in stems:
        scene = load_scene(stem, cfg, args.device)
        if not scene["people"]:
            continue
        if args.gravity == "geocalib":
            gp = scene["pred"] / "geocalib" / "gravity.npz"
            if gp.exists():
                scene["gravity"] = np.asarray(
                    np.load(gp)["gravity_world_geocalib"], np.float32)
        _load_cones(scene, cfg, args.device)
        pds = [PersonDynamics(mid, p["betas"], p["q"], p["valid"], scene["gravity"],
                              scene["fps"], args.device, smooth_sigma=args.sigma)
               for p in scene["people"]]
        if args.null_static:
            for pd in pds:
                pd.v = torch.zeros_like(pd.v)
                pd.a = torch.zeros_like(pd.a)
        for vstar, dstar in itertools.product(vstars, dstars):
            scores, save_f = [], []
            drop, g_dirs = [], []
            for person, pd in zip(scene["people"], pds):
                res = solve_completion(scene, person, pd, cfg, args.device,
                                       vstar=vstar, dstar=dstar, form=args.form,
                                       vel_sigma=0.0 if args.raw_vel else args.sigma,
                                       solve_gravity=args.gravity == "solve",
                                       rel_gap=args.rel_gap)
                scores.append(fold_limbs(np.linalg.norm(res["forces"], axis=-1),
                                         scene["groups"]))
                save_f.append(res["forces"])
                drop.append((res["base_before"], res["base_after"]))
                if res["gravity_dir"] is not None:
                    g_dirs.append(res["gravity_dir"])
            row = evaluate_scene(scene, {"completion": scores})
            summary.setdefault((vstar, dstar), []).append((stem, row["completion"]))
            tag = f"v{vstar:g}_d{dstar:g}_{args.form}{suffix}"
            np.savez(_OUT_DIR / f"{stem}__{tag}.npz",
                     forces=np.stack(save_f), vstar=vstar, dstar=dstar, form=args.form,
                     sigma=args.sigma, gravity=args.gravity,
                     base_drop=np.array(drop, np.float32),
                     gravity_dir=(np.stack(g_dirs) if g_dirs else np.zeros((0, 3))),
                     groups=np.array(scene["groups"]))
            au, ap_ = row["completion"]
            g_txt = ""
            if g_dirs:
                ang = np.degrees(np.arccos(np.clip(g_dirs[0] @ np.array([0.0, 1.0, 0.0]),
                                                   -1, 1)))
                g_txt = f"  g_solved {ang:.1f}deg from +y"
            print(f"{stem:22s} v*={vstar:<4g} d*={dstar:<4g} AUROC/AP {au:.3f}/{ap_:.3f}  "
                  f"wrench {drop[0][0]:.0f}->{drop[0][1]:.0f}{g_txt}", flush=True)

    print("\nmacro AUROC | AP per (v*, d*):")
    for (vstar, dstar), rows in sorted(summary.items()):
        au = np.mean([r[1][0] for r in rows])
        ap_ = np.mean([r[1][1] for r in rows])
        print(f"  v*={vstar:<4g} d*={dstar:<4g}  {au:.3f} | {ap_:.3f}   ({len(rows)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
