"""Human-optimization stage — four-stage refinement of the tracked SMPL-X bodies.

Dispatches ``root`` → ``kinematics_1`` → ``kinematics_2`` → ``dynamics`` (any subset
via ``--stages``) onto the runners in :mod:`tools.human_optim.stages`, reading one
config (``configs/human_optimization.toml``) and writing to ``out/<stem>/human_optim/``.

The four stages: ``root`` places each body rigidly (root SE3, joints frozen);
``kinematics_1`` runs the visually-guided kinematic fit (full → refine → L-BFGS);
``kinematics_2`` refines it against manual per-joint contact annotation;
``dynamics`` solves per-contact force via RNEA on the frozen ``q``. ``root`` +
``kinematics_1`` share one converter-anchored optimization; the default chain is
``root, kinematics_1`` (``kinematics_2`` needs annotation, ``dynamics`` needs contacts).

Run from CLI::

    .venv/bin/python scripts/stages/human_optimization.py \\
        --source video.mp4 --predictions out/video/ [--stages root,kinematics_1] [--overlay]

Reads:  sam3d/smplx_params.npz, sapiens/pose.npz, geometry/{human,transform,scene}.npz;
        (kinematics_2) human_optim/kinematic_1.npz + data/<stem>.npz;
        (dynamics) human_optim/kinematic_2.npz + human_optim/contacts_2.npz + geocalib/gravity.npz
Writes: human_optim/{root,kinematic_1,kinematic_2,dynamics}.npz (+ *_overlay.mp4)
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

from tools.human_optim.config import STAGE_ORDER, load_config  # noqa: E402
from tools.human_optim.stages import (  # noqa: E402
    run_dynamics, run_kinematic, run_kinematics_2,
)

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"


def run(source, predictions_dir, config_path, *, stages=None, annotations=None,
        device_override=None, overlay=False, bodies="kinematic_2", labels="manual") -> None:
    """Run the requested human-optimization stages in dependency order."""
    cfg = load_config(config_path)
    requested = stages if stages else list(cfg.run)
    bad = [s for s in requested if s not in STAGE_ORDER]
    if bad:
        raise SystemExit(f"[human_optimization] unknown stage(s) {bad}; known: {list(STAGE_ORDER)}")
    pred = Path(predictions_dir)

    kin_stages = [s for s in ("root", "kinematics_1") if s in requested]
    kin_device = device_override or cfg.device
    if kin_stages:
        run_kinematic(source, pred, cfg, kin_stages, kin_device, overlay=overlay)
    if "kinematics_2" in requested:
        run_kinematics_2(source, pred, cfg, kin_device, annotations=annotations, overlay=overlay,
                         labels_mode=labels)
    if "dynamics" in requested:
        dyn_device = device_override or cfg.stage_opts.get("dynamics", {}).get("device", "cpu")
        run_dynamics(pred, cfg, dyn_device, bodies=bodies)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=None,
                    help="original video / frames — needed for overlays and kinematics_2 "
                         "annotation auto-resolution; not needed for dynamics")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--stages", default=None,
                    help="comma-separated subset of root,kinematics_1,kinematics_2,dynamics "
                         "(default: config [stages].run)")
    ap.add_argument("--annotations", default=None,
                    help="kinematics_2 manual-contact npz (default: <source>.npz or data/<stem>.npz)")
    ap.add_argument("--labels", choices=("manual", "auto"), default="manual",
                    help="kinematics_2 contact labels: manual annotation (default) or our own "
                         "contacts_1.npz, weighted by schema-4 per-joint confidence")
    ap.add_argument("--bodies", default="kinematic_2",
                    help="dynamics: which refined body to solve on (kinematic_1|kinematic_2)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()] if args.stages else None
    run(args.source, args.predictions, args.config, stages=stages, annotations=args.annotations,
        device_override=args.device, overlay=args.overlay, bodies=args.bodies, labels=args.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
