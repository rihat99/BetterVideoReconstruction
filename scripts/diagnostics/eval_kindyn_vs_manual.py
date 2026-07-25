"""Compare the automatic kindyn_1 solve against the manual-label etalon, per out/ clip.

``human_optim/kindyn_1.npz`` is the joint pose+force solve driven by the pipeline's own
contact labels; ``human_optim/kindyn_manual.npz`` is the same solve driven by the manual
annotations — the etalon. This script says how far apart the two land, on three axes:

* **motion** — per-joint world distance between the two bodies over the valid frames:
  RMSE and median, in centimetres.
* **contact forces** — for each of the six contact groups (two hands, two feet, two
  ankles), the mean force magnitude of each solve over the frames where *both* pushed
  (a group only has a force where its labels said contact, so comparing elsewhere would
  compare a force against a structural zero).
* **root residual** — the mean norm of the force half of ``base_wrench``, the newtons of
  gravity the contact forces failed to explain. Lower is better, per solve.

Usage::

    uv run --no-sync python scripts/diagnostics/eval_kindyn_vs_manual.py
    ... --scenes boulder_1,sport_climb --json scratch/kindyn_vs_manual.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_ROOT = _REPO_ROOT / "out"


def _load(path: Path) -> dict:
    """The kindyn fields this comparison needs, with the group names flattened to ``(6,)``."""
    with np.load(path, allow_pickle=True) as npz:
        groups = np.asarray(npz["contact_force_joints"])
        return dict(
            joints_world=np.asarray(npz["joints_world"], np.float64),      # (P,N,52,3) m
            contact_forces=np.asarray(npz["contact_forces"], np.float64),  # (P,N,6,3) N
            base_wrench=np.asarray(npz["base_wrench"], np.float64),        # (P,N,6) force|torque
            valid_mask=np.asarray(npz["valid_mask"], bool),                # (P,N)
            total_mass=np.asarray(npz["total_mass"], np.float64).reshape(-1),
            object_ids=[int(v) for v in np.asarray(npz["object_ids"])],
            groups=[str(v) for v in (groups if groups.ndim == 1 else groups[0])],
        )


def compare(auto: dict, manual: dict) -> dict:
    """Motion, per-group force and root-residual comparison over the shared people/frames."""
    shared = [oid for oid in auto["object_ids"] if oid in manual["object_ids"]]
    if auto["groups"] != manual["groups"]:
        raise ValueError(f"contact group mismatch: {auto['groups']} vs {manual['groups']}")
    dists, base_auto, base_manual = [], [], []
    force = {g: [[], []] for g in auto["groups"]}
    for oid in shared:
        a, m = auto["object_ids"].index(oid), manual["object_ids"].index(oid)
        valid = auto["valid_mask"][a] & manual["valid_mask"][m]
        if not valid.any():
            continue
        delta = auto["joints_world"][a, valid] - manual["joints_world"][m, valid]
        dists.append(np.linalg.norm(delta, axis=-1).ravel() * 100.0)       # cm
        base_auto.append(np.linalg.norm(auto["base_wrench"][a, valid, :3], axis=-1))
        base_manual.append(np.linalg.norm(manual["base_wrench"][m, valid, :3], axis=-1))
        fa = np.linalg.norm(auto["contact_forces"][a], axis=-1)            # (N,6) newtons
        fm = np.linalg.norm(manual["contact_forces"][m], axis=-1)
        for c, group in enumerate(auto["groups"]):
            both = valid & (fa[:, c] > 0.0) & (fm[:, c] > 0.0)
            force[group][0].append(fa[both, c])
            force[group][1].append(fm[both, c])

    if not dists:
        raise ValueError("no shared valid frames between the two solves")
    d = np.concatenate(dists)
    out = dict(
        people=shared,
        n_joint_frames=int(d.size),
        joint_rmse_cm=float(np.sqrt(np.mean(d ** 2))),
        joint_median_cm=float(np.median(d)),
        base_force_auto_n=float(np.mean(np.concatenate(base_auto))),
        base_force_manual_n=float(np.mean(np.concatenate(base_manual))),
        total_mass_auto=float(auto["total_mass"].mean()),
        total_mass_manual=float(manual["total_mass"].mean()),
        forces={},
    )
    for group, (fa, fm) in force.items():
        a, m = np.concatenate(fa) if fa else np.zeros(0), np.concatenate(fm) if fm else np.zeros(0)
        out["forces"][group] = dict(
            n_both=int(a.size),
            auto_n=float(a.mean()) if a.size else float("nan"),
            manual_n=float(m.mean()) if m.size else float("nan"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--scenes", default=None, help="comma-separated scene names (default: all)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    scenes = ([s.strip() for s in args.scenes.split(",") if s.strip()] if args.scenes
              else sorted(d.name for d in args.root.iterdir()
                          if (d / "human_optim" / "kindyn_manual.npz").exists()
                          and (d / "human_optim" / "kindyn_1.npz").exists()))
    rows: dict[str, dict] = {}
    for scene in scenes:
        try:
            rows[scene] = compare(_load(args.root / scene / "human_optim" / "kindyn_1.npz"),
                                  _load(args.root / scene / "human_optim" / "kindyn_manual.npz"))
        except Exception as exc:  # noqa: BLE001 — one bad scene must not kill the sweep
            print(f"[eval_kindyn_vs_manual] {scene}: FAILED {type(exc).__name__}: {exc}")

    print("\n== kindyn_1 vs kindyn_manual — joint distance (cm) and root residual (N)")
    print(f"{'scene':24s}{'P':>3s}{'RMSE':>8s}{'median':>8s}{'base_auto':>11s}{'base_man':>10s}")
    for scene, row in rows.items():
        print(f"{scene:24s}{len(row['people']):3d}{row['joint_rmse_cm']:8.1f}"
              f"{row['joint_median_cm']:8.1f}{row['base_force_auto_n']:11.1f}"
              f"{row['base_force_manual_n']:10.1f}")

    groups = list(next(iter(rows.values()))["forces"]) if rows else []
    print("\n== mean contact-force magnitude (N) where BOTH solves push  [auto | manual | frames]")
    print(f"{'scene':24s}" + "".join(f"{g:>26s}" for g in groups))
    for scene, row in rows.items():
        cells = "".join(f"{row['forces'][g]['auto_n']:10.1f}"
                        f"{row['forces'][g]['manual_n']:10.1f}"
                        f"{row['forces'][g]['n_both']:6d}" for g in groups)
        print(f"{scene:24s}{cells}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\n[eval_kindyn_vs_manual] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
