"""Human-optimization stage — staged refinement of the tracked SMPL-X bodies.

Dispatches ``root`` → ``kinematics_1`` → ``kinematics_2`` → ``kindyn_1`` (any subset
via ``--stages``) onto the runners in :mod:`bvr.human_optim.stages` / ``.kindyn``,
reading one config (``configs/human_optimization.toml``) and writing to
``out/<stem>/human_optim/``.

The stages: ``root`` places each body rigidly (root SE3, joints frozen);
``kinematics_1`` runs the visually-guided kinematic fit (full → refine → L-BFGS);
``kinematics_2`` repairs it against the contact labels of ``contacts_1``; ``kindyn_1``
re-optimizes the ``kinematics_2`` pose jointly with the contact forces.
``root`` + ``kinematics_1`` share one optimization anchored to the starting pose —
the scale stage's pre-fit (``geometry/prefit.npz``) when present, else the
converter; the default chain is ``root, kinematics_1`` (``kinematics_2`` needs
``contacts_1``, ``kindyn_1`` needs its own body's contacts).

Run from CLI::

    python -m bvr.stages.human_optimization \\
        --source video.mp4 --predictions out/video/ [--stages root,kinematics_1]

Reads:  sam3d/smplx_params.npz, sapiens/pose.npz, geometry/{human,transform,scene}.npz;
        (kinematics_2) human_optim/kinematic_1.npz + its contact labels;
        (kindyn_1)     the [stages.kindyn_1].bodies body + its contacts_<tag>.npz
                       (+ geocalib/gravity.npz when present)
Writes: human_optim/{root,kinematic_1,kinematic_2,kindyn_1}.npz (+ *_overlay.mp4)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bvr import determinism
from bvr.paths import REPO_ROOT
from bvr.config import STAGE_ORDER, load_human_config
from bvr.human_optim.kindyn import run_kindyn
from bvr.human_optim.stages import run_kinematic, run_kinematics_2

_DEFAULT_CONFIG = REPO_ROOT / "configs" / "human_optimization.toml"


def run(source, predictions_dir, config_path, *, stages=None) -> None:
    """Run the requested human-optimization stages in dependency order.

    Per-stage options (overlay, starting body, device) come from the config's
    ``[stages.<name>]`` tables.
    """
    cfg = load_human_config(config_path)
    requested = stages if stages else list(cfg.run)
    bad = [s for s in requested if s not in STAGE_ORDER]
    if bad:
        raise SystemExit(f"[human_optimization] unknown stage(s) {bad}; known: {list(STAGE_ORDER)}")
    pred = Path(predictions_dir)
    opts = cfg.stage_opts

    kin_stages = [s for s in ("root", "kinematics_1") if s in requested]
    if kin_stages:
        k1 = opts.get("kinematics_1", {})
        run_kinematic(source, pred, cfg, kin_stages, cfg.device,
                      overlay=bool(k1.get("overlay", False)),
                      freeze_fingers=bool(k1.get("freeze_fingers", True)))
    if "kinematics_2" in requested:
        k2 = opts.get("kinematics_2", {})
        run_kinematics_2(source, pred, cfg, cfg.device, overlay=bool(k2.get("overlay", False)))
    if "kindyn_1" in requested:
        kd = opts.get("kindyn_1", {})
        run_kindyn(source, pred, cfg, cfg.device,
                   overlay=bool(kd.get("overlay", False)),
                   bodies=str(kd.get("bodies", "kinematic_2")),
                   cones=bool(kd.get("cones", True)),
                   mass_from_meta=bool(kd.get("mass_from_meta", False)),
                   edge_speed_scale=float(kd.get("edge_speed_m_s", 0.0)))


def main(argv=None) -> int:
    determinism.apply()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=None,
                    help="original video / frames — needed for overlays")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--stages", default=None,
                    help="comma-separated subset of root,kinematics_1,kinematics_2,kindyn_1 "
                         "(default: config [stages].run)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                    help="human-optimization TOML (default: configs/human_optimization.toml)")
    args = ap.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()] if args.stages else None
    run(args.source, args.predictions, args.config, stages=stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
