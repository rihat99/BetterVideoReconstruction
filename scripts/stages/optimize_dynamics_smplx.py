"""SMPL-X dynamics: per-contact-joint force inference via RNEA (force-only).

The first dynamics pass on the SMPL-X path. It reads the refined world-frame
SMPL-X trajectory (``sam3d/smplx_kinematic_2.npz`` by default — the
annotation-guided fit), the scene contacts computed on that same body
(``sam3d/cnt_kin_2.npz``), and the world-frame gravity (``geocalib/gravity.npz``),
and for every tracked person, with the kinematic ``q`` **frozen**, optimises
per-contact-joint world-frame linear forces ``f(t, c)`` so that the RNEA base
wrench ``tau[..., :6]`` vanishes (no free "rocket" at the pelvis). A small
force-magnitude regulariser picks the unique minimum-norm solution. The joint
**torques** are whatever RNEA returns at the converged forces.

Why these inputs:
  * **kinematics_2** is the most-refined body (scaled, world-frame, on the
    52-joint ``SMPLXMid``); the per-person ``scale`` and shaped ``betas`` are
    folded into a real-inertia BetterRobot model (:func:`make_smplx_inertial_model`)
    so RNEA is metrically consistent with the posed body.
  * **cnt_kin_2** contacts (per-joint SDF contact on all 52 joints) decide *when*
    each force-bearing joint may push/pull — NOT the human annotation, whose 10
    coarse joints are too few to gate the finger-resolved SMPL-X contacts.
  * **gravity** is the GeoCalib world-frame down vector; magnitude 9.81 is applied
    here and written onto the model (RNEA folds it into the base acceleration).

Force-bearing joints default to the hands (applied at the wrist) + feet — the
contact-bearing extremities — each *active* on a frame when **any** of the
sub-joints it covers (a hand's wrist+fingers, a foot's ankle+toe) is in contact.
Forces are world-frame for inspection and rotated to the joint's local body frame
just before RNEA. Unlike the legacy SMPL stage there is **no upward-only term** —
a climber's hand pulls *down/in* on a hold, so a unilateral "ground pushes up"
prior is wrong here.

Run from CLI::

    .venv/bin/python scripts/stages/optimize_dynamics_smplx.py \\
        --source video.mp4 --predictions out/video/ [--bodies smplx_kinematic_2]

Reads:  sam3d/<bodies>.npz, sam3d/cnt<tag>.npz, geocalib/gravity.npz
Writes: sam3d/smplx_dynamics.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import tomllib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.smplx_robot.dynamics import velocity_acceleration_from_trajectory  # noqa: E402
from tools.smplx_optim.qmath import normalize_q  # noqa: E402
from tools.smplx_robot import (  # noqa: E402
    load_smplx_mid, make_smplx_inertial_model, remap_q_bh_to_model,
)

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "smplx_dynamics.toml"
_DEFAULT_SMPLX_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
)
_GRAVITY_MAG = 9.81  # m/s²

#: Finger-joint name fragments (SMPLXMid hand joints), for the hand contact group.
_FINGER = ("index", "middle", "pinky", "ring", "thumb")


# ─────────────────────────────── config ──────────────────────────────────

def _load_config(path: Path) -> dict:
    cfg = tomllib.load(path.open("rb")) if path.exists() else {}

    w = cfg.setdefault("weights", {})
    w.setdefault("base_wrench", 1.0)
    w.setdefault("force_magnitude", 0.01)
    w.setdefault("force_smooth", 0.0)    # temporal ‖Δf‖² (within sustained contact)
    w.setdefault("torque_smooth", 0.0)   # temporal ‖Δτ‖² (joint torques)

    c = cfg.setdefault("contact", {})
    # Which force-bearing joint groups are active (names resolved in _contact_catalog).
    c.setdefault("groups", ["left_hand", "right_hand", "left_foot", "right_foot"])

    it = cfg.setdefault("inertia", {})
    it.setdefault("total_mass_kg", 70.0)
    it.setdefault("density", 1000.0)

    o = cfg.setdefault("optimizer", {})
    o.setdefault("max_iter", 200)
    o.setdefault("history_size", 20)
    o.setdefault("lr", 1.0)
    o.setdefault("tolerance_grad", 1e-9)
    o.setdefault("tolerance_change", 1e-12)

    cfg.setdefault("runtime", {}).setdefault("device", "cpu")
    return cfg


# ─────────────────────────── contact joint groups ────────────────────────

def _contact_catalog(joint_names: list[str]) -> dict[str, tuple[str, list[str]]]:
    """``group_name -> (representative_joint, member_joints)`` for SMPLXMid.

    A group is *in contact* on a frame when any member joint is; the resultant
    force is applied at the representative joint (a hand's wrist, a foot's toe).
    """
    have = set(joint_names)
    cat: dict[str, tuple[str, list[str]]] = {}
    for side in ("left", "right"):
        hand = [f"{side}_wrist"] + [
            n for n in joint_names
            if n.startswith(f"{side}_") and any(f in n for f in _FINGER)
        ]
        cat[f"{side}_hand"] = (f"{side}_wrist", [n for n in hand if n in have])
        foot = [n for n in (f"{side}_ankle", f"{side}_foot") if n in have]
        rep = f"{side}_foot" if f"{side}_foot" in have else f"{side}_ankle"
        cat[f"{side}_foot"] = (rep, foot)
        for jn in ("knee", "elbow", "hip"):  # available for opt-in via config
            n = f"{side}_{jn}"
            if n in have:
                cat[f"{side}_{jn}"] = (n, [n])
    return cat


# ─────────────────────────────── fext / rnea ─────────────────────────────

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


# ─────────────────────────────── per person ──────────────────────────────

def _solve_person(mid, betas, scale, q_bh, contact_bool, valid, gravity_world,
                  cfg, device, label=""):
    """Solve world-frame contact forces for one person. Returns a result dict."""
    import better_robot as br
    from better_robot.kinematics import forward_kinematics

    dev = torch.device(device)
    dtype = torch.float32
    J = mid.num_joints
    N = q_bh.shape[0]
    dt = 1.0 / float(cfg["_fps"])

    # ── Real-inertia, scale-folded model + GeoCalib gravity ──────────────
    model = make_smplx_inertial_model(
        mid, betas, float(scale), total_mass_kg=float(cfg["inertia"]["total_mass_kg"]),
        density=float(cfg["inertia"]["density"]), device=dev, dtype=dtype,
    )
    g = torch.as_tensor(gravity_world, dtype=dtype, device=dev)
    g = g / (g.norm() + 1e-9) * _GRAVITY_MAG
    g6 = torch.cat([g, torch.zeros(3, dtype=dtype, device=dev)])
    model = dataclasses.replace(model, gravity=g6)
    total_mass = float(model.body_inertias[1:, 0].sum())

    # ── Frozen kinematics: q (DFS), v, a, FK joint poses ─────────────────
    q_model = remap_q_bh_to_model(normalize_q(q_bh.to(dev), J), mid, model)
    with torch.no_grad():
        v, a = velocity_acceleration_from_trajectory(model, q_model, dt)
        oMi = forward_kinematics(model, q_model).joint_pose_world          # (N, njoints, 7)

    # ── Contact groups → force-bearing joints + per-frame mask ───────────
    catalog = _contact_catalog(list(mid.joint_names))
    name_to_bh = {n: i for i, n in enumerate(mid.joint_names)}
    group_names = [g for g in cfg["contact"]["groups"] if g in catalog]
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

    # Temporal pair masks for the smoothness terms: force smoothness applies only
    # within a *sustained* grip (both frames in contact ⇒ smooth the planted force,
    # don't fight the legitimate 0→f step at a grip's start/end); torque smoothness
    # applies on every valid consecutive pair.
    valid_pair = (valid_t[1:] & valid_t[:-1]).to(dtype)                    # (N-1,)
    fpair = (mask[1:] * mask[:-1] * valid_pair[:, None]) if C else None    # (N-1, C)

    # ── Optimise world-frame forces so the base wrench vanishes ──────────
    f_world = torch.zeros((N, C, 3), dtype=dtype, device=dev, requires_grad=True)
    data = model.create_data(batch_shape=(N,), device=dev, dtype=dtype)
    w_base = float(cfg["weights"]["base_wrench"])
    w_mag = float(cfg["weights"]["force_magnitude"])
    w_fs = float(cfg["weights"]["force_smooth"])
    w_ts = float(cfg["weights"]["torque_smooth"])
    oc = cfg["optimizer"]

    def closure():
        opt.zero_grad()
        fext, f_eff = _build_fext(f_world, mask, R_w2l, contact_ids, model.njoints)
        tau = br.rnea(model, data, q_model, v, a, fext=fext)
        loss = 0.5 * ((w_base * tau[..., :6]).pow(2).sum() + (w_mag * f_eff).pow(2).sum())
        if N > 1:  # temporal smoothness (the per-frame solve is otherwise independent → flicker)
            df = f_world[1:] - f_world[:-1]                                # (N-1, C, 3)
            loss = loss + 0.5 * w_fs * (fpair[..., None] * df.pow(2)).sum()
            dtau = tau[1:, 6:] - tau[:-1, 6:]                             # (N-1, nv-6) joint torques
            loss = loss + 0.5 * w_ts * (valid_pair[:, None] * dtau.pow(2)).sum()
        loss.backward()
        return loss

    if C > 0 and bool(mask.any()):
        opt = torch.optim.LBFGS(
            [f_world], lr=float(oc["lr"]), max_iter=int(oc["max_iter"]),
            history_size=int(oc["history_size"]), tolerance_grad=float(oc["tolerance_grad"]),
            tolerance_change=float(oc["tolerance_change"]), line_search_fn="strong_wolfe")
        with torch.no_grad():
            base0 = float((br.rnea(model, data, q_model, v, a).clone()[..., :6]).pow(2).sum().sqrt())
        opt.step(closure)
    else:
        base0 = float("nan")

    # ── Final forces, torques, per-joint diagnostics (bh order) ──────────
    with torch.no_grad():
        f_final = (f_world * mask[..., None]).detach()                        # (N, C, 3)
        fext, _ = _build_fext(f_final, mask, R_w2l, contact_ids, model.njoints)
        tau = br.rnea(model, data, q_model, v, a, fext=fext)                  # (N, nv)
        base1 = float(tau[..., :6].pow(2).sum().sqrt())

        # per-joint torque magnitude, world joint centres, per-joint world force
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
    fnorm = f_final.norm(dim=-1)                                              # (N, C)
    act = mask.bool().any(dim=1).cpu().numpy()
    mean_f = float(fnorm.sum(dim=-1)[mask.bool().any(dim=1)].mean()) if act.any() else 0.0

    # Flicker diagnostics: mean frame-to-frame change of the contact force (within
    # sustained grips) and of the joint torques (the quantities the viewer animates).
    fjit = tjit = 0.0
    if N > 1:
        with torch.no_grad():
            df = (f_final[1:] - f_final[:-1]).norm(dim=-1)                    # (N-1, C)
            fp = (mask[1:] * mask[:-1]).bool()
            fjit = float(df[fp].mean()) if bool(fp.any()) else 0.0
            dtq = (tau[1:, 6:] - tau[:-1, 6:]).norm(dim=-1)                   # (N-1,)
            vp = valid_pair.bool()
            tjit = float(dtq[vp].mean()) if bool(vp.any()) else 0.0
    print(f"  [{label}] mass={total_mass:.1f}kg  contact-frames={int(act.sum())}/{N}  "
          f"base wrench {base0:.1f}→{base1:.2f} N·…  mean|f|={mean_f:.0f} N "
          f"(weight≈{total_mass * _GRAVITY_MAG:.0f} N)  jitter |Δf|={fjit:.1f} N |Δτ|={tjit:.2f}")

    return dict(
        joints_world=joints_world, torque_mag=torque_mag, force_world=force_world,
        contact_forces=f_final.cpu().numpy().astype(np.float32),               # (N, C, 3)
        joint_torques=tau.cpu().numpy().astype(np.float32),                    # (N, nv)
        total_mass=np.float32(total_mass),
        rep_joint_names=[mid.joint_names[j] for j in rep_bh],
        base_wrench_before=np.float32(base0), base_wrench_after=np.float32(base1),
    )


# ─────────────────────────────── run ─────────────────────────────────────

def run(source, predictions_dir, config_path, *, device_override=None,
        bodies="smplx_kinematic_2") -> Path:
    cfg = _load_config(Path(config_path))
    if device_override:
        cfg["runtime"]["device"] = device_override
    device = cfg["runtime"]["device"]
    pred = Path(predictions_dir)
    sam3d = pred / "sam3d"

    kin_path = sam3d / f"{bodies}.npz"
    cnt_tag = "kin" + bodies.removeprefix("smplx_kinematic")
    cnt_path = sam3d / f"cnt_{cnt_tag}.npz"
    grav_path = pred / "geocalib" / "gravity.npz"
    for p in (kin_path, cnt_path):
        if not p.exists():
            raise SystemExit(f"missing {p} — run the optimize_smplx_2 / estimate_contacts stages first")

    kin = dict(np.load(kin_path, allow_pickle=True))
    cnt = dict(np.load(cnt_path, allow_pickle=True))
    cfg["_fps"] = float(kin["fps"])

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
    print(f"[optimize_dynamics_smplx] {pred.name}: P={P} N={N} fps={cfg['_fps']:.1f}  "
          f"bodies={bodies}  gravity={gravity_world.round(2)} ({gsrc})")

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    J = mid.num_joints

    joints_world = np.zeros((P, N, J, 3), np.float32)
    torque_mag = np.zeros((P, N, J), np.float32)
    contact_forces_world = np.zeros((P, N, J, 3), np.float32)
    joint_torques = np.zeros((P, N, 6 + 3 * (J - 1)), np.float32)
    total_mass = np.zeros(P, np.float32)
    rep_names_p: list = []
    raw_forces_p: list = []

    for p in range(P):
        oid = int(object_ids[p])
        cp = cnt_oids.index(oid) if oid in cnt_oids else p
        res = _solve_person(
            mid,
            torch.as_tensor(kin["betas"][p], dtype=torch.float32),
            float(kin["scale"][p]) if "scale" in kin else 1.0,
            torch.as_tensor(kin["q"][p], dtype=torch.float32),
            torch.as_tensor(joint_contact[cp], dtype=torch.bool, device=device),
            np.asarray(kin["valid_mask"][p], bool),
            gravity_world, cfg, device, label=f"obj {oid}",
        )
        joints_world[p] = res["joints_world"]
        torque_mag[p] = res["torque_mag"]
        contact_forces_world[p] = res["force_world"]
        joint_torques[p] = res["joint_torques"]
        total_mass[p] = res["total_mass"]
        rep_names_p.append(res["rep_joint_names"])
        raw_forces_p.append(res["contact_forces"])

    out_path = sam3d / "smplx_dynamics.npz"
    np.savez(
        out_path,
        joints_world=joints_world, torque_mag=torque_mag,
        contact_forces_world=contact_forces_world, joint_torques=joint_torques,
        contact_forces=np.stack(raw_forces_p, 0),
        contact_force_joints=np.array(rep_names_p, dtype="<U16"),
        total_mass=total_mass, gravity_world=gravity_world.astype(np.float32),
        joint_contact=joint_contact[:P] if joint_contact.shape[0] >= P else joint_contact,
        object_ids=object_ids, joint_names=kin["joint_names"],
        parents=np.asarray(kin["parents"], np.int32), valid_mask=kin["valid_mask"],
        num_people=np.int32(P), num_frames=np.int32(N),
        video_height=np.int32(kin["video_height"]), video_width=np.int32(kin["video_width"]),
        fps=np.float32(cfg["_fps"]), frame="world", source_bodies=str(bodies),
    )
    print(f"[optimize_dynamics_smplx] wrote {out_path}")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=False, default=None)  # unused (3D); kept for CLI parity
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--bodies", default="smplx_kinematic_2",
                    help="optimized body npz under sam3d/ to read (default smplx_kinematic_2)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    run(args.source, args.predictions, args.config,
        device_override=args.device, bodies=args.bodies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
