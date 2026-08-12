"""Measure how smooth a refined SMPL-X trajectory is (the kinematics_1 jitter metric).

Four numbers per body, all over frame steps whose both ends are valid:

* ``ang_jitter_deg`` — mean per-frame change of the 21 **body** joint quaternions in
  degrees (fingers excluded: they are frozen + pre-smoothed, so they would flatter the
  number). This is the "mean |Δangle|/frame" jitter used in earlier rounds.
* ``root_trans_accel_mm`` — mean ``‖t_{t+1} − 2 t_t + t_{t−1}‖`` of the root translation,
  in mm/frame² (the global-trajectory wobble the joint number cannot see).
* ``cart_accel_mm`` — the same second difference on the 22 posed **world** body joints,
  mm/frame². Cartesian, so a small shoulder rotation counts as much as it moves the wrist.
* ``cart_speed_mm`` — mean world joint speed, mm/frame (context: a scene where the climber
  moves fast has more of everything).

Per-joint breakdowns of the angular and Cartesian numbers come out too, so a change can be
attributed to particular limbs (e.g. only the occluded ones).

Run::

    uv run --no-sync python scripts/diagnostics/motion_smoothness.py \\
        --predictions out --scenes boulder_1 sport_climb --json scratch/jitter.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.human_optim.qmath import normalize_q  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402

NUM_BODY_JOINTS = 22   # pelvis + the 21 fitted body joints; the remaining 30 are fingers


def _quat_step_deg(quats: np.ndarray) -> np.ndarray:
    """Per-step rotation angle in degrees of a ``(T, J, 4)`` unit-quaternion track → ``(T-1, J)``."""
    dot = np.abs((quats[1:] * quats[:-1]).sum(-1)).clip(0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


@torch.no_grad()
def _world_joints(mid, betas: np.ndarray, q: np.ndarray, device: str, chunk: int = 128) -> np.ndarray:
    """Pose one person's whole ``q (T, nq)`` trajectory → world joint positions ``(T, J, 3)``."""
    betas_t = torch.as_tensor(betas, dtype=torch.float32, device=device)
    q_t = torch.as_tensor(q, dtype=torch.float32, device=device)
    out = []
    for start in range(0, q_t.shape[0], chunk):
        end = min(start + chunk, q_t.shape[0])
        posed = mid.forward(betas_t[None].expand(end - start, -1),
                            normalize_q(q_t[start:end], mid.num_joints))
        out.append(posed.joints_world.tensor()[..., :3].cpu().numpy())
    return np.concatenate(out, 0)


def measure_body(body: dict, mid, device: str) -> dict:
    """Smoothness numbers for one refined-body npz, per person and pooled over people."""
    q_all = np.asarray(body["q"], np.float32)                  # (P, N, nq)
    valid = np.asarray(body["valid_mask"], bool)               # (P, N)
    joint_names = [str(x) for x in body["joint_names"]][:NUM_BODY_JOINTS]
    people = []
    for p in range(q_all.shape[0]):
        q = q_all[p]
        step = valid[p][1:] & valid[p][:-1]                    # (N-1,) both ends valid
        stencil = valid[p][2:] & valid[p][1:-1] & valid[p][:-2]
        joint_quats = q[:, 7:].reshape(q.shape[0], -1, 4)[:, :NUM_BODY_JOINTS - 1]
        joint_quats = joint_quats / np.linalg.norm(joint_quats, axis=-1, keepdims=True).clip(1e-9)
        ang = _quat_step_deg(joint_quats)                      # (N-1, 21)
        root_quat = q[:, 3:7] / np.linalg.norm(q[:, 3:7], axis=-1, keepdims=True).clip(1e-9)
        root_ang = _quat_step_deg(root_quat[:, None, :])[:, 0]  # (N-1,)
        transl = q[:, :3]
        root_acc = np.linalg.norm(transl[2:] - 2.0 * transl[1:-1] + transl[:-2], axis=-1)

        world = _world_joints(mid, body["betas"][p], q, device)[:, :NUM_BODY_JOINTS]
        cart_acc = np.linalg.norm(world[2:] - 2.0 * world[1:-1] + world[:-2], axis=-1)  # (N-2, 22)
        cart_spd = np.linalg.norm(world[1:] - world[:-1], axis=-1)                      # (N-1, 22)

        people.append({
            "n_steps": int(step.sum()),
            "ang_jitter_deg": float(ang[step].mean()) if step.any() else float("nan"),
            "root_ang_jitter_deg": float(root_ang[step].mean()) if step.any() else float("nan"),
            "root_trans_accel_mm": (float(root_acc[stencil].mean()) * 1e3
                                    if stencil.any() else float("nan")),
            "cart_accel_mm": (float(cart_acc[stencil].mean()) * 1e3
                              if stencil.any() else float("nan")),
            "cart_speed_mm": float(cart_spd[step].mean()) * 1e3 if step.any() else float("nan"),
            "per_joint_ang_deg": {joint_names[j + 1]: float(ang[step, j].mean())
                                  for j in range(ang.shape[1])} if step.any() else {},
            "per_joint_cart_accel_mm": {joint_names[j]: float(cart_acc[stencil, j].mean()) * 1e3
                                        for j in range(cart_acc.shape[1])} if stencil.any() else {},
        })
    weights = np.array([person["n_steps"] for person in people], np.float64)
    pooled = {}
    if weights.sum() > 0:
        for key in ("ang_jitter_deg", "root_ang_jitter_deg", "root_trans_accel_mm",
                    "cart_accel_mm", "cart_speed_mm"):
            vals = np.array([person[key] for person in people], np.float64)
            keep = np.isfinite(vals)
            pooled[key] = float((vals[keep] * weights[keep]).sum() / weights[keep].sum())
    return {"people": people, "pooled": pooled,
            "object_ids": [int(x) for x in body["object_ids"]]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", required=True, help="root holding <scene>/human_optim/")
    ap.add_argument("--scenes", nargs="+", default=None, help="default: every scene with the body")
    ap.add_argument("--body", default="kinematic_1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    root = Path(args.predictions)
    names = args.scenes
    if names is None:
        names = sorted(d.name for d in root.iterdir()
                       if (d / "human_optim" / f"{args.body}.npz").exists())
    mid = load_smplx_mid(None, args.device)

    report: dict[str, dict] = {}
    header = (f"{'scene':22s} {'ang°/f':>8s} {'root°/f':>8s} {'rootAcc':>8s} "
              f"{'cartAcc':>8s} {'speed':>8s}")
    print(header)
    print("-" * len(header))
    for name in names:
        path = root / name / "human_optim" / f"{args.body}.npz"
        if not path.exists():
            print(f"{name:22s} MISSING {path}")
            continue
        result = measure_body(dict(np.load(path, allow_pickle=True)), mid, args.device)
        report[name] = result
        pooled = result["pooled"]
        print(f"{name:22s} {pooled['ang_jitter_deg']:8.4f} {pooled['root_ang_jitter_deg']:8.4f} "
              f"{pooled['root_trans_accel_mm']:8.3f} {pooled['cart_accel_mm']:8.3f} "
              f"{pooled['cart_speed_mm']:8.3f}")

    if report:
        weights = np.array([sum(p["n_steps"] for p in r["people"]) for r in report.values()], float)
        print("-" * len(header))
        overall = {}
        for key in ("ang_jitter_deg", "root_ang_jitter_deg", "root_trans_accel_mm",
                    "cart_accel_mm", "cart_speed_mm"):
            vals = np.array([r["pooled"][key] for r in report.values()], float)
            overall[key] = float((vals * weights).sum() / weights.sum())
        print(f"{'POOLED':22s} {overall['ang_jitter_deg']:8.4f} "
              f"{overall['root_ang_jitter_deg']:8.4f} {overall['root_trans_accel_mm']:8.3f} "
              f"{overall['cart_accel_mm']:8.3f} {overall['cart_speed_mm']:8.3f}")
        report["POOLED"] = {"pooled": overall}

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
