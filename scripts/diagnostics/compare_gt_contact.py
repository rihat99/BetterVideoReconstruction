"""What measured contacts buy: the board-labelled chain against the estimated one.

Both trees of a peter rig run the same stages with the same weights and differ in one
thing — where the contact labels came from. ``human_optim/`` estimated them from the
video; ``human_optim_gt_contact/`` measured them off the force boards
(``docs/board_contacts.md``). Every clip's boards are also the force ground truth, and
neither run is supervised on them, so the comparison is honest for both.

Two tables, pooled over every clip of the tree and its four limbs:

* **contacts** — how well the estimated labels agree with the boards (the error the
  second table prices), per limb and pooled.
* **forces** — each run's solved limb force against the measured one: mean absolute
  error of the size in newtons (and its root-mean-square, which charges a few large
  misses more than many small ones), the angle between the vectors where both read above
  50 N, the correlation of the sizes over time, the load-share error in percentage
  points (where "the total is right but the limbs are wrong" shows), the hands-to-feet
  split against the boards' own, the root residual the solve could not explain, and the
  per-frame force step (jitter), with the boards' own step for scale.

The force table is printed three times: over every clip, then over the clips whose
reconstruction put the loaded limbs on the wall and over those where it did not
(``frame_rescued``, see ``docs/board_contacts.md``). Measured contacts can only help a
body that is in roughly the right place, and pooling the two hides that.

Each run's jitter is measured over the frames *it* labelled in contact — how shaky its
own forces are — not over a shared mask, so read it as a per-run property rather than a
head-to-head difference.

Two things a rig can fail to give, both handled by skipping rather than guessing. A limb
the rig did not measure at all (a Parkour foot landing off the plates) is written NaN by
the importer and drops out of every number here — it is not a released limb. And on a rig
whose vectors live in its own world (Parkour's MoCap yaw is not ours), only SIZES compare,
so the angle column is blank; ``sensor_forces.npz`` says which case it is
(``world_aligned``).

Usage::

    uv run --no-sync python scripts/diagnostics/compare_gt_contact.py
    uv run --no-sync python scripts/diagnostics/compare_gt_contact.py --tree peter/out_climb_wall_2_single
    uv run --no-sync python scripts/diagnostics/compare_gt_contact.py \\
        --tree /home/rikhat.akizhanov/better/data/Parkour-dataset/processed/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIAGNOSTICS = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_DIAGNOSTICS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from score_pipeline_contacts import _limb_members  # noqa: E402
from tools.human_optim.boards import LIMBS, limb_of_frame  # noqa: E402

#: A board above this is loaded — the same rule the climb_wall_2 comparison uses.
CONTACT_N = 50.0

RUNS = {"Estimated contacts (video)": "human_optim",
        "Measured contacts (boards)": "human_optim_gt_contact"}


def limb_forces(kindyn_path: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """``(force (N, 4, 3), root_residual_bw (N,))`` for one clip's solved forces."""
    kindyn = np.load(kindyn_path, allow_pickle=True)
    members = _limb_members([str(x) for x in kindyn["joint_names"]])
    forces = np.asarray(kindyn["contact_forces_world"][0], float)[:num_frames]
    body_weight = float(np.asarray(kindyn["total_mass"], float)[0]) * 9.81
    residual = np.linalg.norm(
        np.asarray(kindyn["base_wrench"][0], float)[:num_frames, :3], axis=-1) / body_weight
    return np.stack([forces[:, m].sum(1) for m in members], 1), residual


def limb_labels(contacts_path: Path, num_frames: int) -> np.ndarray:
    """``(N, 4)`` bool — did this run's labels claim each limb was in contact."""
    contacts = np.load(contacts_path, allow_pickle=True)
    names = [str(x) for x in contacts["contact_frame_names"]]
    frame_contact = np.asarray(contacts["frame_contact"][0], bool)[:num_frames]
    return np.stack([frame_contact[:, [c for c, n in enumerate(names)
                                       if limb_of_frame(n) == limb]].any(1) for limb in LIMBS], 1)


def rescued_share(clip: Path) -> float:
    """Fraction of this clip's measured contact labels that landed off the wall.

    The boards cannot be wrong about a loaded limb, so a ``rescued`` label means the
    *reconstruction* put that limb nowhere near the reconstructed wall — the body is in
    the wrong place, and a true contact label is then being pinned onto a bad pose.
    """
    contacts = np.load(clip / "human_optim_gt_contact" / "contacts_1.npz", allow_pickle=True)
    labelled = np.asarray(contacts["frame_contact"][0], bool)
    return float(np.asarray(contacts["frame_rescued"][0], bool).sum()) / max(labelled.sum(), 1)


def finished_clips(tree: Path) -> list[Path]:
    """Every clip of the tree that has BOTH runs finished."""
    return [c for c in sorted(tree.iterdir())
            if all((c / d / "kindyn_1.npz").exists() for d in RUNS.values())]


def gather(clips: list[Path]) -> dict[str, np.ndarray]:
    """Stack the given clips into pooled per-frame arrays."""
    columns: dict[str, list[np.ndarray]] = {key: [] for key in
                                            ("boards", "board_on", "measured", "estimated_on",
                                             *(f"{d}_force" for d in RUNS.values()),
                                             *(f"{d}_residual" for d in RUNS.values()))}
    aligned = True
    for clip in clips:
        sensor = np.load(clip / "human_optim" / "sensor_forces.npz")
        boards = np.asarray(sensor["reaction_world"], float)
        aligned &= bool(sensor["world_aligned"]) if "world_aligned" in sensor else True
        n = min(len(boards), len(np.load(clip / "human_optim" / "kindyn_1.npz")["base_wrench"][0]))
        columns["boards"].append(boards[:n])
        columns["board_on"].append(np.linalg.norm(boards[:n], axis=-1) > CONTACT_N)
        columns["measured"].append(np.isfinite(boards[:n]).all(-1))
        columns["estimated_on"].append(limb_labels(clip / "human_optim" / "contacts_2.npz", n))
        for directory in RUNS.values():
            force, residual = limb_forces(clip / directory / "kindyn_1.npz", n)
            columns[f"{directory}_force"].append(force)
            columns[f"{directory}_residual"].append(residual)
    data = {key: np.concatenate(parts) for key, parts in columns.items()}
    data["world_aligned"] = np.bool_(aligned)
    return data


def hands_over_feet(size: np.ndarray, full: np.ndarray) -> float:
    """Hand load divided by foot load, over the frames where all four limbs were measured."""
    hands, feet = size[full][:, :2].sum(), size[full][:, 2:].sum()
    return float(hands / max(feet, 1e-6))


def force_metrics(boards: np.ndarray, ours: np.ndarray, residual: np.ndarray,
                  contact: np.ndarray, world_aligned: bool = True) -> dict[str, float]:
    """One run's solved limb forces against the measured ones.

    A limb-frame the rig did not measure is NaN in ``boards``; blanking our own vector
    there makes every number below skip it instead of scoring against nothing.
    """
    ours = np.where(np.isfinite(boards), ours, np.nan)
    measured, solved = np.linalg.norm(boards, axis=-1), np.linalg.norm(ours, axis=-1)
    full = np.isfinite(measured).all(1)                   # frames with all four limbs measured
    both = (measured > CONTACT_N) & (solved > CONTACT_N)  # NaN compares False, so it drops out
    cos = (boards * ours).sum(-1)[both] / (measured[both] * solved[both])
    share = lambda m: m / np.maximum(m.sum(1, keepdims=True), 1e-6)  # noqa: E731
    step = np.linalg.norm(np.diff(ours, axis=0), axis=-1)[contact[1:] & contact[:-1]]
    pair = np.isfinite(solved) & np.isfinite(measured)
    return {
        "limb MAE [N]": np.nanmean(np.abs(solved - measured)),
        "limb RMSE [N]": np.sqrt(np.nanmean((solved - measured) ** 2)),
        "total MAE [N]": np.nanmean(np.abs(np.linalg.norm(ours.sum(1), axis=-1)
                                           - np.linalg.norm(boards.sum(1), axis=-1))),
        "corr": np.corrcoef(solved[pair], measured[pair])[0, 1],
        "angle [deg]": (np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()
                        if world_aligned else np.nan),
        "share err [pp]": 100 * np.nanmean(np.abs(share(solved) - share(measured))),
        "hands/feet": hands_over_feet(solved, full),
        "root resid [BW]": np.median(residual),
        "|df| [N]": np.nanmean(step),
    }


def force_table(title: str, data: dict[str, np.ndarray]) -> None:
    """Print both runs' solved limb forces against the boards, for one set of clips."""
    truth, claimed = data["board_on"], data["estimated_on"]
    aligned = bool(data["world_aligned"])
    rows = {name: force_metrics(data["boards"], data[f"{d}_force"], data[f"{d}_residual"],
                                claimed if d == "human_optim" else truth, aligned)
            for name, d in RUNS.items()}
    keys = list(next(iter(rows.values())))
    cell = lambda v: f"{'—':>16s}" if np.isnan(v) else f"{v:16.3f}"  # noqa: E731
    print(f"\n  Solved limb forces vs the boards — {title}")
    print(f"  {'':28s}" + "".join(f"{k:>16s}" for k in keys))
    for name, metrics in rows.items():
        print(f"  {name:28s}" + "".join(cell(metrics[k]) for k in keys))
    measured = np.linalg.norm(data["boards"], axis=-1)
    step = np.linalg.norm(np.diff(data["boards"], axis=0), axis=-1)[truth[1:] & truth[:-1]]
    print(f"  {'boards themselves':28s}" + " " * 80
          + f"{hands_over_feet(measured, np.isfinite(measured).all(1)):16.3f}"
          + f"{'':16s}{np.nanmean(step):16.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", type=Path,
                    default=_REPO_ROOT / "peter" / "out_climb_wall_3_single")
    ap.add_argument("--clean", type=float, default=0.05,
                    help="rescued share below which a clip counts as reconstructed on the wall")
    ap.add_argument("--broken", type=float, default=0.50,
                    help="rescued share above which a clip counts as reconstructed off the wall")
    args = ap.parse_args(argv)
    clips = finished_clips(args.tree)
    print(f"[compare] {len(clips)} clip(s) of {args.tree}")
    data = gather(clips)
    truth, claimed, measured = data["board_on"], data["estimated_on"], data["measured"]
    if not measured.all():
        print(f"[compare] {int((~measured).sum())}/{measured.size} limb-frames were never "
              "measured by the rig; they are left out of every number below")

    print("\n  Estimated labels vs the boards")
    print(f"  {'limb':12s} {'board f':>8s} {'ours f':>8s} {'prec':>6s} {'rec':>6s} {'F1':>6s}")
    for i, limb in enumerate([*LIMBS, "pooled"]):
        keep = measured[:, i] if limb != "pooled" else measured.ravel()
        t = (truth[:, i] if limb != "pooled" else truth.ravel())[keep]
        c = (claimed[:, i] if limb != "pooled" else claimed.ravel())[keep]
        tp = float((t & c).sum())
        precision, recall = tp / max(c.sum(), 1), tp / max(t.sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print(f"  {limb:12s} {int(t.sum()):8d} {int(c.sum()):8d} "
              f"{precision:6.3f} {recall:6.3f} {f1:6.3f}")

    force_table("all clips", data)
    # Measured contacts can only help a body that is in roughly the right place. Split on
    # exactly that, so a win on the good clips is not averaged away by the bad ones.
    for label, keep in ((f"reconstruction ON the wall (rescued < {100 * args.clean:.0f}%)",
                         lambda f: f < args.clean),
                        (f"reconstruction OFF the wall (rescued > {100 * args.broken:.0f}%)",
                         lambda f: f > args.broken)):
        subset = [c for c in clips if keep(rescued_share(c))]
        if subset:
            force_table(f"{label} — {len(subset)} clips", gather(subset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
