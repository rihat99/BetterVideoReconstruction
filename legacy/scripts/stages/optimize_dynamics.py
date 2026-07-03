"""Human-motion Stage 2: contact-force inference via RNEA.

Reads ``<clip>/results/<input_index>.npz`` (the kinematic solution from
Stage 1) and, with ``q`` frozen, optimises per-contact-joint world-frame
linear forces ``f(t, n)`` so that the RNEA base wrench ``tau[..., :6]``
vanishes ("no rocket at the butt"). A small force-magnitude regulariser
picks the unique minimum-norm solution. Output is written to
``<clip>/results/<output_index>.npz`` with the contact-force trajectory
appended to the existing schema.

Usage::

    uv run python scripts/stages/optimize_dynamics.py <clip_dir> \\
        --config configs/dynamics.toml

Only linear forces are optimised (user spec); torque entries of ``fext``
stay zero. Forces are applied at the joint origin of each contact joint
in its local body frame; the optimisation variable is held in the world
frame (for easier inspection) and rotated just before the RNEA call.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
if _BETTER_HUMAN_SRC.exists() and str(_BETTER_HUMAN_SRC) not in sys.path:
    sys.path.insert(0, str(_BETTER_HUMAN_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "dynamics.toml"


# ─────────────────────────────── config ──────────────────────────────────

def _load_config(config_path: Path) -> dict:
    """Read a TOML config with inline defaults for unspecified fields."""
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    stage = cfg.get("stage", {})
    stage.setdefault("input_index", 1)
    stage.setdefault("output_index", 2)

    opt = cfg.get("optimizer", {})
    opt.setdefault("max_iter", 200)
    opt.setdefault("history_size", 20)
    opt.setdefault("lr", 1.0)
    opt.setdefault("tolerance_grad", 1e-8)
    opt.setdefault("tolerance_change", 1e-11)

    w = cfg.get("weights", {})
    w.setdefault("base_wrench", 1.0)
    w.setdefault("force_magnitude", 0.01)
    w.setdefault("upward_only", 10.0)

    c = cfg.get("contact", {})
    c.setdefault("joints", ("left_foot", "right_foot", "left_hand", "right_hand"))
    c.setdefault("threshold", 0.5)

    rt = cfg.get("runtime", {})
    rt.setdefault("device", "cpu")

    cfg["stage"] = stage
    cfg["optimizer"] = opt
    cfg["weights"] = w
    cfg["contact"] = c
    cfg["runtime"] = rt
    return cfg


# ─────────────────────────────── io ──────────────────────────────────────

def _resolve_input(clip_dir: Path, input_index: int) -> Path:
    candidate = clip_dir / "results" / f"{input_index}.npz"
    if not candidate.exists():
        raise FileNotFoundError(
            f"expected input at {candidate} — run optimize_motion.py first"
        )
    return candidate


def _resolve_output(clip_dir: Path, output_index: int) -> Path:
    results = clip_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    return results / f"{output_index}.npz"


# ─────────────────────────────── core ────────────────────────────────────

def _build_fext(
    f_world: torch.Tensor,               # (T, N, 3) optimisation variable
    mask: torch.Tensor,                  # (T, N) float {0,1}
    R_world_to_local: torch.Tensor,      # (T, N, 3, 3) precomputed, detached
    contact_joint_ids: torch.Tensor,     # (N,) long
    njoints: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct ``fext`` for RNEA and return ``(fext, f_effective)``.

    ``f_effective`` is the masked world-frame force actually used (for the
    regulariser); ``fext`` is the ``(T, njoints, 6)`` body-frame wrench
    tensor expected by RNEA.
    """
    T, N, _ = f_world.shape
    f_eff = f_world * mask[..., None]
    # World → local rotation; einsum preserves autograd through f_world.
    f_local = torch.einsum("tnij,tnj->tni", R_world_to_local, f_eff)

    # Scatter linear forces into a (T, njoints, 3) tensor; torque part zero.
    fext_lin = torch.zeros((T, njoints, 3), dtype=f_world.dtype, device=f_world.device)
    index = contact_joint_ids.view(1, N, 1).expand(T, N, 3)
    fext_lin = fext_lin.scatter(dim=1, index=index, src=f_local)

    fext = torch.cat(
        [fext_lin, torch.zeros_like(fext_lin)],
        dim=-1,
    )  # (T, njoints, 6)
    return fext, f_eff


def _precompute_world_to_local(oMi: torch.Tensor, contact_joint_ids: torch.Tensor) -> torch.Tensor:
    """Return ``R_world_to_local (T, N, 3, 3)`` for the given contact joints."""
    from better_robot.lie import so3

    quats = oMi[:, contact_joint_ids, 3:7]          # (T, N, 4)
    R_lw = so3.to_matrix(quats)                     # local→world
    return R_lw.transpose(-1, -2).contiguous()      # world→local


def _joint_contact_columns(model, contact_names) -> list[int]:
    """Column in the saved ``joint_contact (T, 24)`` for each contact joint.

    The saved array is DFS with column ``i`` matching BR body ``i + 1``
    (see docs/motion_npz_schema.md).
    """
    return [int(model.body_name_to_id[n]) - 1 for n in contact_names]


def optimize_dynamics(
    clip_dir: Path,
    config_path: Path,
    *,
    device_override: str | None = None,
) -> Path:
    cfg = _load_config(config_path)
    if device_override is not None:
        cfg["runtime"]["device"] = device_override

    import better_robot as br
    from tools.robot_motion import (
        rebuild_inertial_model,
        velocity_acceleration_from_trajectory,
    )

    dtype = torch.float32
    dev = torch.device(cfg["runtime"]["device"])

    in_npz = _resolve_input(clip_dir, cfg["stage"]["input_index"])
    out_npz = _resolve_output(clip_dir, cfg["stage"]["output_index"])
    if in_npz.resolve() == out_npz.resolve():
        raise ValueError(
            f"input and output resolve to the same file: {in_npz}. "
            f"Bump stage.output_index."
        )

    with np.load(in_npz) as npz:
        npz_data = {k: np.asarray(npz[k]) for k in npz.files}

    fps = float(npz_data["fps"])
    dt = 1.0 / fps
    q_np = npz_data["q"]
    joint_contact_np = npz_data["joint_contact"]
    T = q_np.shape[0]
    print(
        f"[optimize_dynamics] in: {in_npz}\n"
        f"                    out: {out_npz}\n"
        f"                    T={T}, fps={fps:.2f}, dt={dt * 1000:.1f} ms"
    )

    # ── Rebuild model with real per-body inertias ────────────────────────
    model = rebuild_inertial_model(npz_data, dtype=dtype, device=dev)
    total_mass = float(model.body_inertias[1:, 0].sum())
    print(
        f"[optimize_dynamics] model: nq={model.nq}, nv={model.nv}, "
        f"njoints={model.njoints}, total_mass={total_mass:.2f} kg"
    )

    # ── Frozen trajectory + kinematics ───────────────────────────────────
    q = torch.from_numpy(q_np).to(dtype=dtype, device=dev)
    joint_contact = torch.from_numpy(joint_contact_np).to(dtype=dtype, device=dev)

    with torch.no_grad():
        v, a = velocity_acceleration_from_trajectory(model, q, dt)
        fk = br.forward_kinematics(model, q)
        oMi = fk.joint_pose_world                          # (T, njoints, 7)

    contact_names = tuple(cfg["contact"]["joints"])
    threshold = float(cfg["contact"]["threshold"])
    contact_joint_ids = torch.tensor(
        [int(model.body_name_to_id[n]) for n in contact_names],
        dtype=torch.long, device=dev,
    )
    contact_cols = _joint_contact_columns(model, contact_names)
    mask_bool = joint_contact[:, contact_cols] > threshold
    mask = mask_bool.to(dtype=dtype)                       # (T, N) 0/1
    N = mask.shape[1]

    n_active = int(mask_bool.any(dim=1).sum().item())
    print(
        f"[optimize_dynamics] contacts: {list(contact_names)} "
        f"(threshold={threshold:.2f})  active frames: {n_active}/{T}"
    )

    # Precompute rotation matrices; q is frozen so these never change.
    R_world_to_local = _precompute_world_to_local(oMi, contact_joint_ids).detach()

    # ── Optimisation variable ────────────────────────────────────────────
    f_world = torch.zeros((T, N, 3), dtype=dtype, device=dev, requires_grad=True)

    # Shared RNEA workspace.
    data = model.create_data(batch_shape=(T,), device=dev, dtype=dtype)

    w = cfg["weights"]
    w_base = float(w["base_wrench"])
    w_mag  = float(w["force_magnitude"])
    w_up   = float(w["upward_only"])

    def _loss_breakdown(fw: torch.Tensor) -> tuple[float, dict[str, float]]:
        with torch.no_grad():
            fext, f_eff = _build_fext(
                fw, mask, R_world_to_local, contact_joint_ids, model.njoints,
            )
            tau = br.rnea(model, data, q, v, a, fext=fext)
            base_sq = float(((w_base * tau[..., :6]) ** 2).sum())
            mag_sq  = float(((w_mag * f_eff) ** 2).sum())
            up_sq   = float(((w_up * torch.relu(-f_eff[..., 2])) ** 2).sum())
            total = 0.5 * (base_sq + mag_sq + up_sq)
            return total, {
                "base_wrench": 0.5 * base_sq,
                "force_magnitude": 0.5 * mag_sq,
                "upward_only": 0.5 * up_sq,
            }

    with torch.no_grad():
        init_loss, init_breakdown = _loss_breakdown(f_world)
    print("\n[optimize_dynamics] before:")
    print("  loss breakdown: "
          + ", ".join(f"{k}={v:.3e}" for k, v in init_breakdown.items())
          + f"  total={init_loss:.3e}")

    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.LBFGS(
        [f_world],
        lr=float(opt_cfg["lr"]),
        max_iter=int(opt_cfg["max_iter"]),
        history_size=int(opt_cfg["history_size"]),
        tolerance_grad=float(opt_cfg["tolerance_grad"]),
        tolerance_change=float(opt_cfg["tolerance_change"]),
        line_search_fn="strong_wolfe",
    )

    call_counter = {"n": 0}
    best = {"loss": float("inf")}

    def closure():
        optimizer.zero_grad()
        fext, f_eff = _build_fext(
            f_world, mask, R_world_to_local, contact_joint_ids, model.njoints,
        )
        tau = br.rnea(model, data, q, v, a, fext=fext)
        base_sq = (w_base * tau[..., :6]).pow(2).sum()
        mag_sq  = (w_mag * f_eff).pow(2).sum()
        up_sq   = (w_up * torch.relu(-f_eff[..., 2])).pow(2).sum()
        loss = 0.5 * (base_sq + mag_sq + up_sq)
        loss.backward()
        call_counter["n"] += 1
        lf = float(loss.detach())
        if lf < best["loss"]:
            best["loss"] = lf
            gn = float(f_world.grad.norm()) if f_world.grad is not None else float("nan")
            print(
                f"    call {call_counter['n']:4d}  "
                f"loss* = {lf:.6e}  "
                f"‖grad‖ = {gn:.3e}"
            )
        return loss

    print(
        f"\n[optimize_dynamics] running L-BFGS (max_iter={opt_cfg['max_iter']}, "
        f"line_search=strong_wolfe) …"
    )
    optimizer.step(closure)
    print(
        f"[optimize_dynamics] {call_counter['n']} closure calls, "
        f"best loss = {best['loss']:.6e}"
    )

    # Hard-zero inactive entries before saving (they already have zero
    # grad, but make it explicit in case LBFGS left numerical dust). Also
    # recompute τ with the final forces so we can persist joint torques
    # alongside the motion for downstream inspection / visualisation.
    with torch.no_grad():
        f_final = (f_world * mask[..., None]).detach()
        final_loss, final_breakdown = _loss_breakdown(f_final)
        final_fext, _ = _build_fext(
            f_final, mask, R_world_to_local, contact_joint_ids, model.njoints,
        )
        final_tau = br.rnea(model, data, q, v, a, fext=final_fext).detach()

    print("\n[optimize_dynamics] after:")
    print("  loss breakdown: "
          + ", ".join(f"{k}={v:.3e}" for k, v in final_breakdown.items())
          + f"  total={final_loss:.3e}")
    force_norms = f_final.norm(dim=-1)  # (T, N)
    if n_active > 0:
        active_frames = mask_bool.any(dim=1)
        avg_total = float(force_norms[active_frames].sum(dim=-1).mean())
        print(
            f"  mean total |f| over active frames = {avg_total:.1f} N "
            f"(body weight ≈ {total_mass * 9.81:.1f} N)"
        )

    # ── Save results/<output_index>.npz ──────────────────────────────────
    out = dict(npz_data)
    out["contact_forces"] = f_final.cpu().numpy().astype(np.float32)
    out["contact_force_joints"] = np.array(contact_names, dtype="<U20")
    out["contact_force_threshold"] = np.float32(threshold)
    out["joint_torques"] = final_tau.cpu().numpy().astype(np.float32)
    out["stage_index"] = np.int32(cfg["stage"]["output_index"])
    out["source_index"] = np.int32(cfg["stage"]["input_index"])
    out["config_path"] = str(config_path)
    np.savez(out_npz, **out)
    print(f"\n[optimize_dynamics] wrote {out_npz}")
    return out_npz


# ─────────────────────────────── cli ─────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip_dir", type=Path,
                    help="directory containing results/<input_index>.npz")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                    help=f"TOML config (default: {_DEFAULT_CONFIG.relative_to(_REPO_ROOT)})")
    ap.add_argument("--device", type=str, default=None,
                    help="override runtime.device from the config")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    optimize_dynamics(args.clip_dir, args.config, device_override=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
