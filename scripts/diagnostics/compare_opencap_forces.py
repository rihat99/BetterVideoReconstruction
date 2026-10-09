"""Our solved foot forces against the OpenCap LabValidation force plates.

Two plates under a laboratory floor, filtered and already assigned per foot, and a
calibration chain that reaches our world without a single fitted parameter
(``scripts/opencap_sensor_forces.py``). Nothing in the reconstruction ever sees a plate,
so every number below is an out-of-sample measurement of the force solve.

Per clip and pooled over all of them, both feet, over the frames the plates saw and the
reconstruction called valid:

* **|F| MAE** — mean absolute error of the force size, newtons, and the same for each
  component of the OpenSim GROUND frame (``Fx``/``Fy(up)``/``Fz``); the components say
  whether an error is a size error or a direction error, and ``Fy(up)`` is the vertical
  comparison — the component the plates measure best and the one that carries body weight.
* **Fy mass-sc** — the same vertical MAE after multiplying OUR forces by
  ``mass_gt / mass_ours``. A sensitivity check on the body mass, not a correction: it says
  how much of the vertical error a perfectly-massed body would still have. It is not a
  rerun of the dynamics with corrected inertias, and on some clips it makes things worse.
* **corr** — correlation of the force size over time. Timing and shape, immune to a
  constant scale error.
* **angle** — mean angle between the two vectors on the frames where both read above
  :data:`CONTACT_N`. Meaningful here, unlike Parkour, because the plates are in OUR frame.
* **total err** — the two feet summed as VECTORS, ``mean ‖Σours − Σplates‖``, the number a
  whole-body force estimate lives on.
* **share err** — the load split between the two feet in percentage points, over the
  frames where the plate total clears :data:`CONTACT_N`. Below that there is no measured
  split to compare against: normalising a zero total turns any estimate into a flat 50 pp.
* **phantom** — mean of OUR vertical foot force over the frames where that foot's plate
  was watching and read under :data:`~opencap_plates.LOADED_N`. Support we invented: a
  swing foot the contact labels kept in contact. It never enters ``share err``, so it gets
  its own column.
* **hand |F|** / **hand max** — mean and max size of our two hand forces. The ground truth
  is exactly zero in all four trial types, so this is pure phantom support too, and it
  moves the horizontal components of everything else.
* **cop gap** — distance from the plate's own centre of pressure, carried into our world,
  to our foot joint on the loaded frames. It prices the TRANSLATION of the calibration,
  which no force metric and no gravity check can see (a wrong ``t`` rotates nothing and
  shifts everything): 10-30 cm is the honest foot-joint-to-COP offset plus reconstruction
  error, a metre is a broken transform.
* **lin resid** — the LINEAR part of the root wrench our solve could not explain, in body
  weights; **ang resid** is its angular part in N·m. The two are separate because a solve
  can balance every force and still leave a moment.
* **mass** — our ``total_mass`` (baked off the SMPL-X body, never measured) over the
  subject's true ``mass_kg``.
* **onset Δ** — how many frames after the plates our own total vertical force first
  crosses 50 N, counted from the first frame the plates were watching. This is the SYNC
  evidence: it is a single sharp, unambiguous event, and it only exists on the drop jumps,
  where the box phase leaves the plates unloaded until the landing. Blank where the clip
  is already loaded at its first captured frame, i.e. where there is no onset to find.
* **lag** — the whole-clip shift maximising the correlation of the total vertical force.
  A SOLVER-TIMING diagnostic, not a sync check: it is dominated by the shape of the force
  over the whole trial, and on a periodic trial (squats, sit-to-stand) many shifts
  correlate nearly as well. ``n/r`` means the optimum sat on the ±:data:`MAX_LAG` scan
  boundary, i.e. the scan resolved nothing. Use **onset Δ** for synchronisation.

Limb-frames the plates did not see (a drop jump's box phase, a foot standing off the
plates) are NaN in ``sensor_forces.npz``; those, and frames the reconstruction marked
invalid, drop out of every number rather than scoring as zero.

Two pooled rows, because they answer different questions: **POOLED** pools frames, so the
long trials dominate; **MEAN OF CLIPS** weights every clip equally.

Writes ``processed/out/_compare/<clip>.png`` (ours vs plates per foot: size and the three
components over time) and ``processed/out/_compare/summary.md``. ``--run`` picks the run
directory to score; a variant chain lands in ``_compare_<its suffix>/`` beside it.

Usage::

    uv run --no-sync python scripts/diagnostics/compare_opencap_forces.py
    uv run --no-sync python scripts/diagnostics/compare_opencap_forces.py \
        --run human_optim_floorplane
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIAGNOSTICS = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_DIAGNOSTICS), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opencap_plates import LOADED_N  # noqa: E402
from _common import _limb_members  # noqa: E402
from bvr.human_optim.boards import LIMBS  # noqa: E402
from bvr.paths import LABVALIDATION  # noqa: E402

PROCESSED = LABVALIDATION / "processed"

#: A foot above this is loaded — the threshold the peter and Parkour comparisons use.
CONTACT_N = 50.0

#: Video frames the sync scan searches either way (±10 at 60 Hz is ±0.17 s).
MAX_LAG = 10

#: The two feet and the two hands, as columns of the ``LIMBS``-ordered arrays.
FEET = [LIMBS.index("left_foot"), LIMBS.index("right_foot")]
HANDS = [LIMBS.index("left_hand"), LIMBS.index("right_hand")]

#: Our total vertical force must clear this for the onset check to call it a landing.
ONSET_N = 50.0

#: Okabe-Ito blue / vermillion — a colour-blind-safe pair for the two series.
COLOURS = {"ours": "#0072B2", "plates": "#D55E00"}


def limb_forces(kindyn: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(force (N, 4, 3), linear root residual (N,) in BW, angular residual (N,) in N·m)``.

    The per-limb force is the solve's ``contact_forces_world`` folded onto each limb's
    joints, exactly as ``compare_gt_contact`` does it. The root residual is split: only
    ``base_wrench[:3]`` is a force, and a solve can balance every force and still leave an
    unexplained moment, which ``base_wrench[3:]`` carries.
    """
    members = _limb_members([str(x) for x in kindyn["joint_names"]])
    forces = np.asarray(kindyn["contact_forces_world"][0], float)
    wrench = np.asarray(kindyn["base_wrench"][0], float)
    body_weight = float(np.asarray(kindyn["total_mass"], float)[0]) * 9.81
    return (np.stack([forces[:, m].sum(1) for m in members], 1),
            np.linalg.norm(wrench[:, :3], axis=-1) / body_weight,
            np.linalg.norm(wrench[:, 3:], axis=-1))


def clip_series(run_dir: Path) -> dict[str, np.ndarray]:
    """One clip's plate and solved limb forces, in the OpenSim ground frame, unseen NaN.

    Both sides are carried BACK to the mocap frame (``Rᵀ``) rather than compared in our
    world, for two reasons: the components are then physically named (``y`` is up, so the
    vertical comparison is simply the ``y`` column), and they are the same axes in every
    clip, so the pooled row is not an average over a dozen different camera frames. Every
    other metric here is rotation-invariant and reads the same either way.

    A frame is scored only where the plate saw the limb AND the reconstruction called the
    frame valid; everything else is NaN, which every reduction below skips.
    """
    sensor = np.load(run_dir / "sensor_forces.npz", allow_pickle=True)
    kindyn = np.load(run_dir / "kindyn_1.npz", allow_pickle=True)
    plates_all = np.asarray(sensor["reaction_world"], float)            # (N, 4, 3) our world
    ours_all, linear, angular = limb_forces(kindyn)
    if len(ours_all) != len(plates_all):
        print(f"  {run_dir.parent.name}: the plates cover {len(plates_all)} frames and the body "
              f"{len(ours_all)} — comparing the first {min(len(plates_all), len(ours_all))}")
    n = min(len(plates_all), len(ours_all))
    valid = np.asarray(kindyn["valid_mask"][0], bool)[:n]
    rotation = np.asarray(sensor["R_world_from_mocap"], float)
    joint_names = [str(x) for x in kindyn["joint_names"]]
    feet_joints = [joint_names.index(LIMBS[limb]) for limb in FEET]

    plates = np.where(valid[:, None, None], plates_all[:n, FEET], np.nan)
    seen = np.isfinite(plates)
    ours = np.where(seen, ours_all[:n, FEET], np.nan)
    mass_ours = float(np.asarray(kindyn["total_mass"], float)[0])
    mass_gt = float(np.asarray(sensor["mass_kg"], float))
    return {
        "plates": plates @ rotation, "ours": ours @ rotation,
        "ours_mass_scaled": ours @ rotation * (mass_gt / max(mass_ours, 1e-6)),
        # hands are a measured zero in every one of these trials, so our hand force is
        # phantom support by construction — no plate column needed to score it
        "hands": np.where(valid[:, None, None], ours_all[:n, HANDS], np.nan) @ rotation,
        "linear": np.where(valid & seen.all(-1).any(-1), linear[:n], np.nan),
        "angular": np.where(valid & seen.all(-1).any(-1), angular[:n], np.nan),
        "cop": np.asarray(sensor["cop_world"], float)[:n, FEET],
        "foot_joint": np.asarray(kindyn["joints_world"][0], float)[:n, feet_joints],
        "captured": np.asarray(sensor["captured"], bool)[:n, FEET] & valid[:, None],
        "fps": float(np.asarray(sensor["video_fps"], float)),
        "mass_ours": mass_ours, "mass_gt": mass_gt,
        "gravity_check_deg": float(np.asarray(sensor["gravity_check_deg"], float)),
        "gravity_source": str(sensor["gravity_source"]) if "gravity_source" in sensor else "?",
    }


def best_lag(ours: np.ndarray, plates: np.ndarray) -> tuple[float, float]:
    """``(lag, correlation)`` maximising ``corr(ours[t + lag], plates[t])`` over ±:data:`MAX_LAG`.

    A positive lag means the plates see an event at frame ``t`` that our forces only show
    at ``t + lag``. An optimum sitting ON the scan boundary is not an optimum — the scan
    ran out of room — and comes back as NaN, which the table prints as ``n/r``.
    """
    scores = {}
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        if abs(lag) >= len(ours):                 # no overlap left to correlate
            continue
        left = ours[max(lag, 0):len(ours) + min(lag, 0)]
        right = plates[max(-lag, 0):len(plates) - max(lag, 0)]
        pair = np.isfinite(left) & np.isfinite(right)
        if pair.sum() > 2 and left[pair].std() > 0 and right[pair].std() > 0:
            scores[lag] = float(np.corrcoef(left[pair], right[pair])[0, 1])
    if not scores:
        return np.nan, np.nan
    lag = max(scores, key=scores.__getitem__)
    return (np.nan, np.nan) if abs(lag) == MAX_LAG else (float(lag), scores[lag])


def onset_delay(series: dict[str, np.ndarray]) -> float:
    """Frames between the plates' first loaded frame and ours — the synchronisation check.

    Both onsets are searched from the first frame the plates were watching, so a drop
    jump's unmeasured box phase does not count as "unloaded". NaN where the clip is
    already loaded there: without an unloaded run first, there is no onset to time.
    """
    watching = series["captured"].all(1)
    if not watching.any():
        return np.nan
    start = int(np.argmax(watching))
    plate_up = np.nan_to_num(series["plates"][..., 1]).sum(1) > ONSET_N
    ours_up = np.nan_to_num(series["ours"][..., 1]).sum(1) > ONSET_N
    if plate_up[start] or not plate_up[start:].any() or not ours_up[start:].any():
        return np.nan
    return float(np.argmax(ours_up[start:]) - np.argmax(plate_up[start:]))


def metrics(series: dict[str, np.ndarray]) -> dict[str, float]:
    """Every scalar of the table for one clip (or one pooled stack of clips)."""
    plates, ours, hands = series["plates"], series["ours"], series["hands"]
    measured, solved = np.linalg.norm(plates, axis=-1), np.linalg.norm(ours, axis=-1)
    both = (measured > CONTACT_N) & (solved > CONTACT_N)             # NaN compares False
    cos = (plates * ours).sum(-1)[both] / (measured[both] * solved[both])
    pair = np.isfinite(measured) & np.isfinite(solved)
    full = np.isfinite(measured).all(1)                              # both feet measured
    # A share is only defined where there is a measured total to share out; below that,
    # normalising zero would score any estimate at a flat 50 pp.
    split = full & (np.linalg.norm(np.nan_to_num(plates).sum(1), axis=-1) > CONTACT_N)
    share = lambda f: f / np.maximum(f.sum(1, keepdims=True), 1e-6)  # noqa: E731
    # Support we invented: our vertical force on a foot the plates WERE watching and found
    # unloaded. Scored on the plate's own vertical reading, not on |F|.
    quiet = series["captured"] & (np.nan_to_num(plates[..., 1]) < LOADED_N)
    hand_size = np.linalg.norm(hands, axis=-1)
    lag, lag_corr = best_lag(np.nan_to_num(ours[..., 1], nan=np.nan).sum(1),
                             plates[..., 1].sum(1))
    return {
        "frames": float(np.isfinite(measured).any(1).sum()),
        "|F| MAE [N]": float(np.nanmean(np.abs(solved - measured))),
        "Fx MAE [N]": float(np.nanmean(np.abs(ours[..., 0] - plates[..., 0]))),
        "Fy(up) MAE [N]": float(np.nanmean(np.abs(ours[..., 1] - plates[..., 1]))),
        "Fz MAE [N]": float(np.nanmean(np.abs(ours[..., 2] - plates[..., 2]))),
        "Fy mass-sc [N]": float(np.nanmean(np.abs(
            series["ours_mass_scaled"][..., 1] - plates[..., 1]))),
        "corr |F|": (float(np.corrcoef(solved[pair], measured[pair])[0, 1])
                     if pair.sum() > 2 else np.nan),
        "angle [deg]": (float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())
                        if cos.size else np.nan),
        "total err [N]": float(np.nanmean(np.linalg.norm(
            ours[full].sum(1) - plates[full].sum(1), axis=-1))),
        "share err [pp]": (float(100 * np.nanmean(np.abs(share(solved[split])
                                                         - share(measured[split]))))
                           if split.any() else np.nan),
        "phantom [N]": float(np.nanmean(ours[..., 1][quiet])) if quiet.any() else np.nan,
        "hand |F| [N]": float(np.nanmean(hand_size)),
        "hand max [N]": float(np.nanmax(hand_size)) if np.isfinite(hand_size).any() else np.nan,
        "cop gap [m]": float(np.nanmedian(np.where(
            measured > CONTACT_N,
            np.linalg.norm(series["cop"] - series["foot_joint"], axis=-1), np.nan))),
        "lin resid [BW]": float(np.nanmedian(series["linear"])),
        "ang resid [N·m]": float(np.nanmedian(series["angular"])),
        "mass ours/gt": series["mass_ours"] / max(series["mass_gt"], 1e-6),
        "onset Δ [fr]": onset_delay(series),
        "lag [fr]": lag,
        "lag corr": lag_corr,
    }


#: Metrics that mean something for ONE foot — the rest describe the pair or the body.
FOOT_KEYS = ["frames", "|F| MAE [N]", "Fx MAE [N]", "Fy(up) MAE [N]", "Fz MAE [N]",
             "corr |F|", "angle [deg]", "phantom [N]", "cop gap [m]"]

#: Quantities that are per-clip by nature and cannot be formed by concatenating clips:
#: a mass ratio is per subject, and a lag or an onset needs one unbroken time axis.
PER_CLIP_ONLY = ["mass ours/gt", "onset Δ [fr]", "lag [fr]", "lag corr"]


def per_foot(series: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """The single-foot metrics of :data:`FOOT_KEYS`, computed on each foot alone."""
    columns = ("plates", "ours", "ours_mass_scaled", "cop", "foot_joint", "captured")
    return {LIMBS[limb]: {k: v for k, v in
                          metrics({**series,
                                   **{c: series[c][:, i:i + 1] for c in columns}}).items()
                          if k in FOOT_KEYS}
            for i, limb in enumerate(FEET)}


def plot(clip: str, series: dict[str, np.ndarray], path: Path) -> None:
    """Ours vs the plates, per foot: force size and the three ground components over time."""
    plates, ours = series["plates"], series["ours"]
    time = np.arange(len(plates)) / series["fps"]
    panels = [("|F|", np.linalg.norm(plates, axis=-1), np.linalg.norm(ours, axis=-1)),
              *((f"F{a}", plates[..., k], ours[..., k])
                for k, a in enumerate(("x", "y (up)", "z")))]
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 5.4), sharex=True)
    for row, limb in enumerate(FEET):
        for col, (label, measured, solved) in enumerate(panels):
            axis = axes[row, col]
            axis.axhline(0.0, color="0.85", lw=0.8)
            # the plates run thick underneath so a perfect match still shows both series
            axis.plot(time, measured[:, row], color=COLOURS["plates"], lw=2.6, label="plates")
            axis.plot(time, solved[:, row], color=COLOURS["ours"], lw=1.3, label="ours")
            axis.grid(True, color="0.92", lw=0.6)
            axis.set_axisbelow(True)
            if row == 0:
                axis.set_title(f"{label} [N]", fontsize=10)
            if row == 1:
                axis.set_xlabel("time [s]", fontsize=9)
            if col == 0:
                axis.set_ylabel(LIMBS[limb].replace("_", " "), fontsize=10)
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(f"{clip} — solved foot forces vs the plates (OpenSim ground frame)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def cell(key: str, value: float) -> str:
    """One table cell: an unresolved lag reads ``n/r``, a quantity with no value reads ``—``."""
    if key == "lag [fr]" and not np.isfinite(value):
        return "n/r"
    if not np.isfinite(value):
        return "—"
    if key == "frames":
        return f"{value:.0f}"
    return f"{value:+.0f}" if key in ("lag [fr]", "onset Δ [fr]") else f"{value:.3f}"


def table(rows: dict[str, dict[str, float]]) -> list[str]:
    """Markdown rendering of ``{row name: metrics}``, one column per metric."""
    keys = list(next(iter(rows.values())))
    out = ["| clip | " + " | ".join(k.replace("|", "\\|") for k in keys) + " |",
           "|" + "---|" * (len(keys) + 1)]
    for name, values in rows.items():
        out.append(f"| {name} | " + " | ".join(cell(k, values[k]) for k in keys) + " |")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=PROCESSED, help="the processed clip root")
    ap.add_argument("--run", default="human_optim", help="run directory under processed/out/<clip>/")
    args = ap.parse_args(argv)

    out_root = args.source / "out"
    clips = [c for c in sorted(out_root.iterdir()) if c.is_dir() and not c.name.startswith("_")
             and (c / args.run / "sensor_forces.npz").exists()
             and (c / args.run / "kindyn_1.npz").exists()]
    if not clips:
        print(f"[compare_opencap] no clip under {out_root} has both sensor_forces and kindyn_1"
              f" under {args.run}/")
        return 1
    # the baseline run keeps _compare/; a variant gets _compare_<its suffix>/
    compare_dir = out_root / ("_compare" + args.run.removeprefix("human_optim"))
    compare_dir.mkdir(parents=True, exist_ok=True)

    rows, pooled = {}, []
    for clip in clips:
        series = clip_series(clip / args.run)
        pooled.append(series)
        rows[clip.name] = metrics(series)
        plot(clip.name, series, compare_dir / f"{clip.name}.png")
        print(f"\n{clip.name}  ({len(series['plates'])} frames, "
              f"mass {series['mass_ours']:.1f} kg ours vs {series['mass_gt']:.1f} kg dataset, "
              f"gravity check {series['gravity_check_deg']:.1f} deg vs "
              f"{series['gravity_source']}, "
              f"{int((~series['captured']).sum())} unseen foot-frames)")
        for name, values in {"both feet": rows[clip.name], **per_foot(series)}.items():
            print(f"  {name:12s}" + "".join(f" {k}={v:.3f}" for k, v in values.items()))

    stacked = {key: np.concatenate([s[key] for s in pooled]) for key in
               ("plates", "ours", "ours_mass_scaled", "hands", "linear", "angular",
                "cop", "foot_joint", "captured")}
    rows["POOLED"] = metrics({**stacked, "mass_ours": 1.0, "mass_gt": 1.0})
    for key in PER_CLIP_ONLY:                 # see the constant: not formable by pooling
        rows["POOLED"][key] = np.nan
    rows["MEAN OF CLIPS"] = {k: float(np.nanmean([rows[c.name][k] for c in clips]))
                             if np.isfinite([rows[c.name][k] for c in clips]).any() else np.nan
                             for k in rows["POOLED"]}
    for name in ("POOLED", "MEAN OF CLIPS"):
        print(f"\n{name}" + "".join(f" {k}={v:.3f}" for k, v in rows[name].items()))

    lines = [f"# OpenCap LabValidation — solved foot forces vs the plates "
             f"({len(clips)} clips, run `{args.run}`)", "",
             *table(rows), "",
             f"Both feet pooled, vectors in the OpenSim ground frame (y up), over the frames "
             f"the plates saw and the reconstruction called valid. Angle over frames where "
             f"both read above {CONTACT_N:.0f} N; `share err` over frames where the plate "
             f"total does. `phantom` is our own vertical force on a foot the plates were "
             f"watching and found unloaded, and `hand |F|`/`hand max` are our hand forces, "
             f"whose ground truth is zero in all four trials — both are support we invented. "
             f"`cop gap` is the plate COP to our foot joint, which prices the calibration's "
             f"translation (nothing else here can see it). `Fy mass-sc` is the vertical MAE "
             f"after scaling our forces by `mass_gt / mass_ours` — a sensitivity check on the "
             f"body mass, not a rerun of the dynamics. **`onset Δ` is the synchronisation "
             f"evidence** — frames between the plates' first loaded frame and ours, counted "
             f"from the first frame the plates were watching, and blank where the clip starts "
             f"already loaded. `lag` is a whole-clip solver-timing diagnostic, NOT a sync "
             f"check: `n/r` means its optimum sat on the ±{MAX_LAG} scan boundary and resolved "
             f"nothing. POOLED pools frames, so the long trials dominate; MEAN OF CLIPS "
             f"weights every clip equally.", ""]
    (compare_dir / "summary.md").write_text("\n".join(lines))
    print(f"\n[compare_opencap] -> {compare_dir}/summary.md  and {len(clips)} PNG(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
