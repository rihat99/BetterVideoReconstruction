"""Human-motion kinematic trajectory optimisation.

Reads ``<clip>/results/<input_index>.npz`` (default: ``0.npz`` — the raw
motion produced by ``build_robot_motion.py``), runs PyTorch's
autograd-based L-BFGS against a ``CostStack`` of acceleration smoothness,
reference anchor, and contact consistency, and writes the optimised
trajectory to ``<clip>/results/<output_index>.npz``.

Weights, iteration counts, and contact-joint selection live in a TOML
config — see ``configs/kinematic.toml``. The CLI only takes the clip
directory and the config path (plus optional --device overrides). This
keeps every experiment reproducible from a checked-in config rather than
a long argv.

Forward expansion: once RNEA lands in BetterRobot, the ``[weights]
dynamics`` field will unlock an inverse-dynamics residual (stage 2 of
the results/ series). Passing ``dynamics > 0`` currently raises
``NotImplementedError`` to prevent silent no-ops.

Usage::

    uv run python scripts/stages/optimize_motion.py <clip_dir> \\
        --config configs/kinematic.toml
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


_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "kinematic.toml"


# ─────────────────────────────── config ──────────────────────────────────

def _load_config(config_path: Path) -> dict:
    """Read a TOML config with inline defaults for unspecified fields."""
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    # Fill in defaults so callers can treat the returned dict as flat.
    stage = cfg.get("stage", {})
    stage.setdefault("input_index", 0)
    stage.setdefault("output_index", 1)

    opt = cfg.get("optimizer", {})
    opt.setdefault("max_iter", 200)
    opt.setdefault("history_size", 20)
    opt.setdefault("lr", 1.0)
    opt.setdefault("tolerance_grad", 1e-6)
    opt.setdefault("tolerance_change", 1e-9)

    w = cfg.get("weights", {})
    w.setdefault("acceleration", 1.0)
    w.setdefault("reference", 0.3)
    w.setdefault("contact", 5.0)
    w.setdefault("dynamics", 0.0)

    c = cfg.get("contact", {})
    c.setdefault("joints", ("left_foot", "right_foot", "left_hand", "right_hand"))

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
    results = clip_dir / "results"
    candidate = results / f"{input_index}.npz"
    if candidate.exists():
        return candidate
    # One-shot migration: move pre-results/ robot_motion.npz → results/0.npz.
    legacy = clip_dir / "robot_motion.npz"
    if input_index == 0 and legacy.exists():
        results.mkdir(parents=True, exist_ok=True)
        legacy.rename(candidate)
        print(f"[optimize_motion] migrated {legacy} → {candidate}")
        return candidate
    raise FileNotFoundError(
        f"expected input at {candidate} (or legacy {legacy} for index 0)"
    )


def _resolve_output(clip_dir: Path, output_index: int) -> Path:
    results = clip_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    return results / f"{output_index}.npz"


def _rebuild_model(npz_data: dict, *, dtype: torch.dtype, device: torch.device):
    """Rebuild the BetterRobot model from ``.npz`` metadata (DFS-ordered).

    See ``docs/motion_npz_schema.md`` for the stored-array layout.
    """
    from better_robot.io.builders import JOINT_NAMES, build_kinematic_tree_model

    saved_names = tuple(str(x) for x in npz_data["joint_names"])
    saved_parents = tuple(int(p) for p in np.asarray(npz_data["parents"]))
    if set(saved_names) != set(JOINT_NAMES):
        raise ValueError(
            f"saved joint_names do not form the SMPL-24 body set "
            f"(expected {sorted(set(JOINT_NAMES))}, got {sorted(set(saved_names))})"
        )
    offsets = torch.from_numpy(npz_data["joint_offsets"]).to(dtype=dtype, device=device)
    return build_kinematic_tree_model(
        name="smpl_motion_body",
        joint_names=saved_names,
        parents=saved_parents,
        translations=offsets,
        root_kind="free_flyer",
        child_kind="spherical",
        mass_per_body=1.0,
        device=device,
        dtype=dtype,
    )


def _contact_frame_ids(model, contact_joint_names) -> tuple[int, ...]:
    return tuple(model.frame_id(f"body_{n}") for n in contact_joint_names)


def _contact_column_indices(model, contact_joint_names) -> list[int]:
    """Column in the DFS-ordered ``joint_contact`` array for each name.

    ``joint_contact`` is ``(T, 24)`` with column ``i`` matching BR body
    ``i + 1`` (DFS). The column for body ``n`` is ``body_name_to_id[n] - 1``.
    """
    return [int(model.body_name_to_id[n]) - 1 for n in contact_joint_names]


# ─────────────────────────────── metrics ─────────────────────────────────

def _report_metrics(
    label: str,
    q_traj: torch.Tensor,
    q_ref: torch.Tensor,
    model,
    *,
    dt: float,
    contact_frame_ids: tuple[int, ...],
    contact_weights: torch.Tensor,
) -> None:
    import better_robot as br

    with torch.no_grad():
        diff_fwd = model.difference(q_traj[1:-1], q_traj[2:])
        diff_back = model.difference(q_traj[:-2], q_traj[1:-1])
        a = (diff_fwd - diff_back) / (dt ** 2)
        accel_norm = float(a.norm())

        ref_norm = float(model.difference(q_ref, q_traj).norm())

        data = br.forward_kinematics(model, q_traj, compute_frames=True)
        idx = torch.as_tensor(contact_frame_ids)
        p = data.frame_pose_world[..., :, :3][..., idx, :]      # (T, K, 3)
        dp = (p[1:] - p[:-1]) / dt                                # (T-1, K, 3)
        w_pair = 0.5 * (contact_weights[:-1] + contact_weights[1:]).unsqueeze(-1)
        contact_norm = float((w_pair * dp).norm())

    print(
        f"  [{label}] ‖accel‖ = {accel_norm:.4f}  "
        f"‖q − q_ref‖ = {ref_norm:.4f}  "
        f"‖w·Δp/dt‖ = {contact_norm:.4f}"
    )


# ─────────────────────────────── optimiser ───────────────────────────────

def optimize_motion(
    clip_dir: Path,
    config_path: Path,
    *,
    device_override: str | None = None,
) -> Path:
    """Optimise one clip per ``config_path``. Returns the output npz path."""
    cfg = _load_config(config_path)
    if device_override is not None:
        cfg["runtime"]["device"] = device_override

    if cfg["weights"]["dynamics"] > 0.0:
        raise NotImplementedError(
            "dynamics residual (RNEA) is a future milestone — set weights.dynamics = 0 in v1"
        )

    import better_robot as br
    from better_robot.costs.stack import CostStack
    from better_robot.residuals import (
        AccelerationResidual,
        ContactConsistencyResidual,
        ReferenceTrajectoryResidual,
    )
    from better_robot.residuals.base import ResidualState

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
        f"[optimize_motion] in: {in_npz}\n"
        f"                 out: {out_npz}\n"
        f"                 T={T}, fps={fps:.2f}, dt={dt * 1000:.1f} ms"
    )

    model = _rebuild_model(npz_data, dtype=dtype, device=dev)
    print(f"[optimize_motion] model: nq={model.nq}, nv={model.nv}, njoints={model.njoints}")

    # .npz layout is BR DFS end-to-end (see docs/motion_npz_schema.md),
    # so q and joint_contact are consumed without permutation.
    q_init = torch.from_numpy(q_np).to(dtype=dtype, device=dev)
    q_ref = q_init.clone()
    joint_contact = torch.from_numpy(joint_contact_np).to(dtype=dtype, device=dev)

    contact_names = tuple(cfg["contact"]["joints"])
    contact_frame_ids = _contact_frame_ids(model, contact_names)
    contact_weights = joint_contact[:, _contact_column_indices(model, contact_names)]

    w = cfg["weights"]
    stack = CostStack()
    if w["acceleration"] > 0:
        stack.add("accel", AccelerationResidual(model, dt=dt), weight=float(w["acceleration"]))
    if w["reference"] > 0:
        stack.add("reference", ReferenceTrajectoryResidual(model, q_ref), weight=float(w["reference"]))
    if w["contact"] > 0:
        stack.add(
            "contact",
            ContactConsistencyResidual(
                model,
                frame_ids=contact_frame_ids,
                contact_weights=contact_weights,
                dt=dt,
            ),
            weight=float(w["contact"]),
        )
    print(
        f"[optimize_motion] cost stack dim: {stack.total_dim()}  "
        f"vars: {T * model.nv} (tangent)"
    )

    # Tangent-space parameterisation: ``delta_v`` is the step from q_init.
    # We bypass autograd entirely — pypose's ``SE3.Log().backward()`` has a
    # factor-of-2 error on the quaternion components (see
    # ``BetterRobot/src/better_robot/kinematics/CLAUDE.md``) which polluted
    # the root-rotation gradient and pushed the optimizer into local minima
    # where contacts drifted. ``stack.gradient`` uses per-residual
    # ``apply_jac_transpose`` for O(T·nv) sparse ``J^T r`` evaluation.
    delta_v = torch.zeros(T * model.nv, dtype=dtype, device=dev)

    def _q_from(dv_flat: torch.Tensor) -> torch.Tensor:
        return model.integrate(q_init, dv_flat.reshape(T, model.nv))

    def _state_of(q: torch.Tensor) -> ResidualState:
        data_fk = br.forward_kinematics(model, q, compute_frames=True)
        return ResidualState(model=model, data=data_fk, variables=q)

    def _loss_breakdown(state: ResidualState) -> dict[str, float]:
        """Per-term ``0.5 · ‖weighted_r‖²`` so the user can see trade-offs."""
        out: dict[str, float] = {}
        for name, item in stack.items.items():
            if not item.active:
                continue
            r_i = item.residual(state)
            out[name] = 0.5 * float(((item.weight * r_i) ** 2).sum())
        return out

    with torch.no_grad():
        init_state = _state_of(q_init)
        breakdown = _loss_breakdown(init_state)
        init_loss = sum(breakdown.values())
    print("\n[optimize_motion] before:")
    _report_metrics(
        "init", q_init, q_ref, model, dt=dt,
        contact_frame_ids=contact_frame_ids, contact_weights=contact_weights,
    )
    print("  loss breakdown: " + ", ".join(f"{k}={v:.3e}" for k, v in breakdown.items())
          + f"  total={init_loss:.3e}")

    opt_cfg = cfg["optimizer"]
    # Use a dummy leaf with requires_grad=True so torch.optim.LBFGS picks it
    # up; we bypass autograd by setting ``.grad`` explicitly in the closure.
    delta_v = delta_v.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [delta_v],
        lr=float(opt_cfg["lr"]),
        max_iter=int(opt_cfg["max_iter"]),
        history_size=int(opt_cfg["history_size"]),
        tolerance_grad=float(opt_cfg["tolerance_grad"]),
        tolerance_change=float(opt_cfg["tolerance_change"]),
        line_search_fn="strong_wolfe",
    )

    # ``closure`` is invoked multiple times per L-BFGS iteration (once for
    # the gradient, additional times as strong-Wolfe line search probes). To
    # avoid misleading output we track best-accepted-loss-so-far and only
    # print when it improves.
    call_counter = {"n": 0}
    best = {"loss": float("inf")}

    def closure():
        optimizer.zero_grad()
        with torch.no_grad():
            q = _q_from(delta_v.detach())
            state = _state_of(q)
            r = stack.residual(state)
            loss_val = 0.5 * float((r * r).sum())
            g = stack.gradient(state)
        # Wire the hand-computed gradient into the parameter so LBFGS reads
        # it via ``_gather_flat_grad``.
        delta_v.grad = g.detach().clone()
        call_counter["n"] += 1
        if loss_val < best["loss"]:
            best["loss"] = loss_val
            print(
                f"    call {call_counter['n']:4d}  "
                f"loss* = {loss_val:.6e}  "
                f"‖grad‖ = {float(g.norm()):.3e}"
            )
        return torch.tensor(loss_val, dtype=delta_v.dtype, device=delta_v.device)

    print(f"\n[optimize_motion] running L-BFGS (max_iter={opt_cfg['max_iter']}, "
          f"line_search=strong_wolfe) …")
    optimizer.step(closure)
    print(f"[optimize_motion] {call_counter['n']} closure calls, "
          f"best loss = {best['loss']:.6e}")

    with torch.no_grad():
        q_opt = _q_from(delta_v.detach()).detach()

    print("\n[optimize_motion] after:")
    _report_metrics(
        "opt", q_opt, q_ref, model, dt=dt,
        contact_frame_ids=contact_frame_ids, contact_weights=contact_weights,
    )
    with torch.no_grad():
        opt_state = _state_of(q_opt)
        breakdown_opt = _loss_breakdown(opt_state)
        final_loss = sum(breakdown_opt.values())
    print("  loss breakdown: " + ", ".join(f"{k}={v:.3e}" for k, v in breakdown_opt.items())
          + f"  total={final_loss:.3e}")

    # ── Save full motion file; downstream tooling reuses the schema. ────
    # q is already BR DFS, matching the schema (see docs/motion_npz_schema.md).
    out = dict(npz_data)
    out["q"] = q_opt.cpu().numpy().astype(np.float32)
    out["stage_index"] = np.int32(cfg["stage"]["output_index"])
    out["source_index"] = np.int32(cfg["stage"]["input_index"])
    out["config_path"] = str(config_path)
    np.savez(out_npz, **out)
    print(f"\n[optimize_motion] wrote {out_npz}")
    return out_npz


# ─────────────────────────────── cli ─────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip_dir", type=Path,
                    help="directory containing results/<input_index>.npz (or legacy robot_motion.npz)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                    help=f"TOML config with weights/optimizer settings (default: {_DEFAULT_CONFIG.relative_to(_REPO_ROOT)})")
    ap.add_argument("--device", type=str, default=None,
                    help="override runtime.device from the config")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    optimize_motion(args.clip_dir, args.config, device_override=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
