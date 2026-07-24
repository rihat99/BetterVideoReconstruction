"""Export completion-solve labels as annotation npz files kinematics_2 can read.

Takes the solved forces of one completion config, runs the standard readout
(theta x body-weight + min-span), unions with the auto labels, and writes one
annotation-format npz per scene: the legacy 10-joint schema (kinematics_2
rejects 14-joint files), the four limb rows filled with the completed labels,
every other row and every frame we know nothing about set to -1 (ignore) — so
the refinement never repels a joint we have no evidence about. Known GT defects
are masked to -1 as well.

Usage::

    uv run python scripts/diagnostics/export_completed_labels.py \\
        --tag v0.2_d0.05_mag_s4_rv_rg --theta 0.01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "diagnostics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_missing_contacts import load  # noqa: E402
from label_readout import _GRAVITY, limb_force, readout  # noqa: E402

_JOINTS_10 = ["left_hand", "right_hand", "left_foot", "right_foot",
              "left_knee", "right_knee", "left_hip", "right_hip",
              "left_elbow", "right_elbow"]
_LIMB_ROWS = (0, 1, 2, 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="completion config tag, e.g. v0.2_d0.05_mag_s4")
    ap.add_argument("--theta", type=float, default=0.01)
    ap.add_argument("--scenes", default=None, help="comma-separated stems (default: all with forces)")
    ap.add_argument("--out-dir", type=Path,
                    default=_REPO_ROOT / "scratch" / "missing_contacts" / "labels")
    args = ap.parse_args(argv)

    comp_dir = _REPO_ROOT / "scratch" / "missing_contacts" / "completion"
    stems = ([s.strip() for s in args.scenes.split(",") if s.strip()] if args.scenes
             else sorted(f.name.split("__")[0] for f in comp_dir.glob(f"*__{args.tag}.npz")))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        d = load(stem)
        comp = np.load(comp_dir / f"{stem}__{args.tag}.npz")
        dyn = np.load(_REPO_ROOT / "out" / stem / "human_optim" / "dynamics.npz")
        mass_of = dict(zip(np.asarray(dyn["object_ids"], int), np.asarray(dyn["total_mass"])))
        weight_n = np.array([mass_of[int(o)] * _GRAVITY for o in d["object_ids"]])

        limb_auto = np.stack([d["group_auto"][..., 0], d["group_auto"][..., 1],
                              d["group_auto"][..., [2, 4]].any(-1),
                              d["group_auto"][..., [3, 5]].any(-1)], -1)
        completed = limb_auto | readout(limb_force(comp["forces"]), weight_n, args.theta)

        P, N = completed.shape[:2]
        contacts = np.full((P, len(_JOINTS_10), N), -1, np.int8)
        for j, row in enumerate(_LIMB_ROWS):
            contacts[:, row] = completed[..., j].astype(np.int8)
        for p in range(P):
            for row in _LIMB_ROWS:
                contacts[p, row, ~d["valid"][p]] = -1

        src = dict(np.load(_REPO_ROOT / "data" / f"{stem}.npz", allow_pickle=True))
        out = dict(
            contacts=contacts,
            joint_names=np.array(_JOINTS_10),
            object_ids=np.asarray(d["object_ids"], np.int32),
            ignored=np.zeros(P, bool),
            num_frames=np.int64(N),
            fps=src["fps"], width=src["width"], height=src["height"],
            video_basename=src["video_basename"], video_fp=src["video_fp"],
            annotation_version=np.int32(2),
            completion_tag=np.str_(args.tag), completion_theta=np.float64(args.theta),
        )
        path = args.out_dir / f"{stem}__{args.tag}_t{args.theta:g}.npz"
        np.savez(path, **out)
        on = (contacts == 1).sum()
        print(f"{stem:22s} -> {path.name}  ({on} contact limb-frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
