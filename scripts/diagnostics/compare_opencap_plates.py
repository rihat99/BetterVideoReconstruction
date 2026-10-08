"""Our forces against the OpenCap LabValidation plates, in the plates' own format and metrics.

The plates state a foot's load as one *triple* per frame — a force ``F``, a centre of
pressure ``p`` on the floor, a vertical free moment ``T_y`` — and the papers that
evaluate on this dataset (Uhlrich 2023, Miller 2025, Gilon 2026, Tan 2026) all score
ground-reaction-force (GRF) estimates on that triple with the same small metric set.
This script follows their protocol. Our per-frame contact forces are first folded into
the same triple per foot by ``scripts/export_opencap_mot.py``
(``human_optim*/estimated_forces.npz``; the algebra is in ``scripts/opencap_plates.py``)
and compared in the OpenSim ground frame: ``x`` anterior–posterior (AP), ``y`` vertical
(V, up), ``z`` medio-lateral (ML).

**Protocol** (the choices that make a number comparable to the published ones):

* Both sides are low-pass filtered identically first — 4th-order zero-lag Butterworth at
  OpenCap's own per-activity cutoffs (:data:`CUTOFF_HZ`: 6 Hz walking, 4 Hz squat and
  sit-to-stand, 30 Hz drop jump). At our 60 Hz sample rate the 30 Hz cutoff is the
  Nyquist limit, so the drop jumps are compared unfiltered.
* Forces are in **% body weight** of the subject's measured mass.
* Per component (V / AP / ML), **per foot**, over **stance frames**: the frames where
  that foot's plate reads more than :data:`STANCE_BW` of body weight (Miller 2025 and
  Gilon 2026 score stance only; Uhlrich 2023 scores whole trials — that row is given
  too, as ``MAE V all``).
* Aggregation as in Uhlrich 2023: per clip → mean per subject → mean ± SD across
  subjects, per activity; the ALL row averages the activities.

**Metrics**:

* **MAE** per component, %BW — the headline number of every OpenCap paper.
* **MAE V own-BW** — the same vertical error with OUR force in % of OUR body weight (the
  mass baked off the SMPL-X shape) and the plate in % of the measured one. A force
  estimate is a fraction of body weight, so a wrong mass scales our newtons but not this;
  it is the mass-independent form of the headline number. **fit gain** is the per-clip
  least-squares scalar that best maps our vertical force onto the plate's: equal to
  ``mass_gt / mass_ours`` when mass is the only scale error, larger when the solve also
  under-reads the load.
* **RMSE V**, %BW (Uhlrich 2023 S3 table also lists RMSE).
* **rRMSE V** — Ren et al. 2008: RMSE over the mean of the two signals' peak-to-peak
  ranges, in %. Undefined for a flat signal (the static trial).
* **r V** — Pearson correlation of the vertical force over stance (shape and timing).
* **peak V err** — error of the clip's peak vertical force, %BW (Tan 2026).
* **CoP** — mean and median Euclidean distance on the floor, mm, on frames where BOTH
  sides carry more than :data:`~opencap_plates.LOADED_N` (a CoP does not exist
  otherwise). Three flavours, because our world is registered to the mocap frame through
  the dataset's calibration, whose translation carries a 0.1–0.3 m residual, on top of
  our own body-placement error: ``raw``; ``hip-aligned`` (one translation per clip,
  the median offset between our hip joints and the mocap hip-joint-centre markers, no
  rotation, no scale); and ``in-foot`` (each CoP relative to its own foot's heel–toe
  midpoint, plate vs mocap markers, ours vs our frames — where under the foot the
  pressure sits, independent of where the foot is). Miller 2025 reports 1.5 cm on this
  dataset with a per-trial registration, which is closest to ``hip-aligned``.
* **free moment** — RMSE of ``T_y`` in N·m on the same frames (no published baseline).
* **stance P / R / F1** — the plates' loaded state as the label and ours as the
  prediction, per foot. Isolates the contact decision from the force solve; a false
  positive here is exactly the old "phantom" load.

Clips whose video-to-mocap sync could not be pinned down (``sync.json`` ``uncertain``)
are scored on their own row and left out of everything else. Frames the plates could not
see (a drop jump's box phase) and frames the reconstruction marked invalid drop out of
every number.

**External methods.** A run directory whose ``estimated_forces.npz`` carries ``hips_mocap``
(``(N, 2, 3)`` left / right hip centres), ``foot_points_mocap`` (``(N, 2, 2, 3)`` heel and
toe per foot), ``valid_mask`` (``(N,)``) and ``total_mass`` — all already in the mocap frame
— is scored without a ``kindyn_1.npz``; that is how the learned model, PhysPT and Li et al.
are put on this protocol (``scripts/methods/opencap_*_export.py``).

Usage::

    uv run --no-sync python scripts/diagnostics/compare_opencap_plates.py \
        --run human_optim_floorplane
    uv run --no-sync python scripts/diagnostics/compare_opencap_plates.py \
        --run method_physpt --clips out/_methods/clips_natural.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_opencap_mot import frame_positions  # noqa: E402
from opencap_plates import LOADED_N  # noqa: E402
from tools.human_optim.boards import LIMBS  # noqa: E402
from tools.paths import LABVALIDATION  # noqa: E402

PROCESSED = LABVALIDATION / "processed"

#: The two feet as columns of the importer's ``LIMBS``-ordered arrays.
FEET = {"left_foot": LIMBS.index("left_foot"), "right_foot": LIMBS.index("right_foot")}

#: OpenSim ground axes, in the order the plates store them.
AXES = ("AP", "V", "ML")

#: Uhlrich et al. 2023: low-pass cutoff per activity, applied to plates and estimate alike.
CUTOFF_HZ = {"walking": 6.0, "walkingTS": 6.0, "squats": 4.0, "squatsAsym": 4.0,
             "STS": 4.0, "STSweakLegs": 4.0, "DJ": 30.0, "DJAsym": 30.0, "static": 4.0}

#: A foot is in stance where its plate carries this fraction of body weight.
STANCE_BW = 0.05

#: Mocap markers whose midpoint stands in for the foot on the floor, per foot.
FOOT_MARKERS = {"left_foot": ("L_calc", "L_toe"), "right_foot": ("r_calc", "r_toe")}

#: Our heel and toe contact frames, per foot — the same midpoint on our body.
FOOT_FRAMES = {"left_foot": ("heel_left", "big_toe_left"),
               "right_foot": ("heel_right", "big_toe_right")}

#: Mocap hip-joint-centre markers against our hip joints: the same anatomical point on
#: both sides, used for the per-clip translation alignment.
HIP_PAIRS = (("left_hip", "L_HJC"), ("right_hip", "R_HJC"))

GRAVITY = 9.81


def trial_type(clip: str) -> str:
    """``subject3_STSweakLegs1p2_cam1`` → ``STSweakLegs``."""
    return re.sub(r"\d+(p\d+)?$", "", clip.split("_")[1])


def lowpass(x: np.ndarray, cutoff_hz: float, fps: float) -> np.ndarray:
    """Zero-lag 4th-order Butterworth along axis 0; identity at or above Nyquist."""
    if cutoff_hz >= 0.5 * fps:
        return x
    b, a = butter(2, cutoff_hz / (0.5 * fps))         # filtfilt doubles the order
    return filtfilt(b, a, x, axis=0)


def hold_edges(force: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """``(N, 2, 3)`` forces with every row NOT in ``keep (N, 2)`` replaced by the nearest kept row.

    The importer writes zero where the plates recorded nothing and the exporters write zero
    where the method produced nothing; a zero-lag filter would smear those invented zeros
    into the neighbouring measured rows, so the unmeasured rows hold their neighbour's value
    instead (the same constant extension ``filtfilt`` pads the ends with) and are still
    masked out of every number afterwards.
    """
    out = force.copy()
    for foot in range(force.shape[1]):
        kept = np.flatnonzero(keep[:, foot])
        if kept.size == 0:
            continue
        nearest = kept[np.clip(np.searchsorted(kept, np.arange(len(force))), 0, kept.size - 1)]
        before = np.clip(np.searchsorted(kept, np.arange(len(force)), side="right") - 1, 0, kept.size - 1)
        prev, nxt = kept[before], nearest
        pick = np.where(np.abs(np.arange(len(force)) - prev) <= np.abs(nxt - np.arange(len(force))), prev, nxt)
        out[:, foot] = force[pick, foot]
    return out


def clip_series(clip: Path, run_dir: Path) -> dict:
    """One clip: plate and our triples per foot in the OpenSim ground frame, plus masks.

    Force arrays are ``(N, 2, 3)`` in :data:`FEET` order and already filtered.
    ``watched`` is the frame mask a number may use: the plates saw that foot AND the
    reconstruction called the frame valid.
    """
    gt = np.load(clip / "forces.npz", allow_pickle=True)
    ours = np.load(run_dir / "estimated_forces.npz", allow_pickle=True)
    external = "hips_mocap" in ours.files
    kindyn = {} if external else dict(np.load(run_dir / "kindyn_1.npz", allow_pickle=True))
    markers = np.load(clip / "markers.npz", allow_pickle=True)
    mass_ours = float(np.asarray(ours["total_mass"] if external else kindyn["total_mass"], float).ravel()[0])
    n = min(len(gt["force"]), len(ours["force"]), len(markers["markers"]))
    if len(gt["force"]) != len(ours["force"]):
        print(f"  {clip.name}: plates cover {len(gt['force'])} frames, ours {len(ours['force'])}"
              f" — comparing the first {n}")
    fps = float(gt["fps"])
    kind = trial_type(clip.name)

    feet = list(FEET.values())
    our_cols = [list(map(str, ours["limbs"])).index(name) for name in FEET]
    valid = np.asarray(ours["valid_mask"] if external else kindyn["valid_mask"][0], bool)[:n]
    watched = np.asarray(gt["captured"], bool)[:n, feet] & valid[:, None]

    marker_names = list(map(str, markers["names"]))
    mk = np.asarray(markers["markers"], float)[:n]
    if external:
        hips = np.asarray(ours["hips_mocap"], float)[:n]
        our_foot = np.asarray(ours["foot_points_mocap"], float)[:n].mean(2)
    else:
        rotation = np.asarray(ours["R_world_from_mocap"], float)
        translation = np.asarray(ours["t_world_from_mocap"], float)
        to_mocap = lambda p: (p - translation) @ rotation  # noqa: E731  Rᵀ (r − t)
        joints = to_mocap(np.asarray(kindyn["joints_world"][0], float)[:n])
        joint_names = list(map(str, kindyn["joint_names"]))
        hips = np.stack([joints[:, joint_names.index(j)] for j, _ in HIP_PAIRS], 1)
        frames = to_mocap(frame_positions(kindyn)[:n])
        frame_names = list(map(str, kindyn["contact_frame_names"]))
        our_foot = np.stack([frames[:, [frame_names.index(a), frame_names.index(b)]].mean(1)
                             for a, b in FOOT_FRAMES.values()], 1)
    offsets = np.concatenate([(hips[:, i] - mk[:, marker_names.index(m)])[valid]
                              for i, (_, m) in enumerate(HIP_PAIRS)])
    sync = clip / "sync.json"
    return {
        "clip": clip.name, "subject": clip.name.split("_")[0], "type": kind,
        "fps": fps, "mass_kg": float(gt["mass_kg"]), "mass_ours": mass_ours,
        "sync_uncertain": bool(json.loads(sync.read_text())["uncertain"]) if sync.exists() else False,
        "watched": watched,
        "gt_force": lowpass(hold_edges(np.asarray(gt["force"], float)[:n, feet],
                                       np.asarray(gt["captured"], bool)[:n, feet]),
                            CUTOFF_HZ[kind], fps),
        "our_force": lowpass(hold_edges(np.asarray(ours["force"], float)[:n, our_cols],
                                        np.repeat(valid[:, None], 2, 1)),
                             CUTOFF_HZ[kind], fps),
        # unfiltered copies: a row counts as loaded for the CoP only if BOTH raw forces say so
        # (the exporters write a zero or fallback point where their own force is below LOADED_N)
        "gt_force_raw": np.asarray(gt["force"], float)[:n, feet],
        "our_force_raw": np.asarray(ours["force"], float)[:n, our_cols],
        "gt_cop": np.asarray(gt["cop"], float)[:n, feet],
        "our_cop": np.asarray(ours["cop"], float)[:n, our_cols],
        "gt_torque": np.asarray(gt["free_torque"], float)[:n, feet],
        "our_torque": np.asarray(ours["free_torque"], float)[:n, our_cols],
        "hip_offset": np.nanmedian(offsets, 0),
        "our_foot": our_foot,
        "gt_foot": np.stack([mk[:, [marker_names.index(a), marker_names.index(b)]].mean(1)
                             for a, b in FOOT_MARKERS.values()], 1),
    }


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d))) if d.size else np.nan


def relative_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Ren et al. 2008: RMSE over the mean peak-to-peak range of the two signals, in %."""
    span = 0.5 * (np.ptp(a) + np.ptp(b)) if a.size else 0.0
    return 100.0 * rmse(a, b) / span if span > 0 else np.nan


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def foot_metrics(gt: np.ndarray, ours: np.ndarray, watched: np.ndarray,
                 body_weight: float, body_weight_ours: float) -> dict[str, float]:
    """The GRF block for one foot: ``(N, 3)`` forces in N, the watched mask, BWs in N."""
    stance = watched & (gt[:, 1] > STANCE_BW * body_weight)
    g, o = gt[stance] / body_weight * 100.0, ours[stance] / body_weight * 100.0   # %BW
    out = {"stance frames": float(stance.sum())}
    for k, axis in enumerate(AXES):
        out[f"MAE {axis} [%BW]"] = float(np.mean(np.abs(o[:, k] - g[:, k]))) if stance.any() else np.nan
    own = ours[stance] / body_weight_ours * 100.0
    for k, axis in enumerate(AXES):          # our force in OUR body weights = the measured-mass reading
        out[f"MAE {axis} own-BW [%BW]"] = (float(np.mean(np.abs(own[:, k] - g[:, k])))
                                           if stance.any() else np.nan)
    out["fit gain"] = (float((o[:, 1] * g[:, 1]).sum() / (o[:, 1] * o[:, 1]).sum())
                       if stance.any() and (o[:, 1] * o[:, 1]).sum() > 0 else np.nan)
    for k, axis in enumerate(AXES):          # Uhlrich 2023 scores whole trials
        out[f"MAE {axis} all [%BW]"] = (float(np.mean(np.abs(ours[watched, k] - gt[watched, k])))
                                        / body_weight * 100.0 if watched.any() else np.nan)
        out[f"MAE {axis} all own-BW [%BW]"] = (float(np.mean(np.abs(
            ours[watched, k] / body_weight_ours - gt[watched, k] / body_weight))) * 100.0
            if watched.any() else np.nan)
    out["RMSE V [%BW]"] = rmse(o[:, 1], g[:, 1])
    out["rRMSE V [%]"] = relative_rmse(o[:, 1], g[:, 1])
    out["r V"] = pearson(o[:, 1], g[:, 1])
    out["peak V err [%BW]"] = (float(abs(ours[watched, 1].max() - gt[watched, 1].max()))
                               / body_weight * 100.0 if watched.any() else np.nan)
    return out


def cop_errors(series: dict) -> dict[str, np.ndarray]:
    """Per-frame CoP and free-moment errors where both sides carry a load (pool by concat)."""
    loaded = (series["watched"] & (series["gt_force"][..., 1] > LOADED_N)
              & (series["our_force"][..., 1] > LOADED_N)
              & (series["gt_force_raw"][..., 1] > LOADED_N) & (series["our_force_raw"][..., 1] > LOADED_N)
              # the plate's own CoP is NaN on a few rows whose FILTERED force crosses the threshold
              & np.isfinite(series["gt_cop"]).all(-1) & np.isfinite(series["our_cop"]).all(-1))
    gt_cop, our_cop = series["gt_cop"][loaded], series["our_cop"][loaded]
    floor = lambda v: v[..., [0, 2]]  # noqa: E731  AP, ML on the floor
    in_foot = floor((our_cop - series["our_foot"][loaded]) - (gt_cop - series["gt_foot"][loaded]))
    return {
        "raw": np.linalg.norm(floor(our_cop - gt_cop), axis=-1),
        "aligned": np.linalg.norm(floor(our_cop - series["hip_offset"] - gt_cop), axis=-1),
        "in_foot": np.linalg.norm(in_foot, axis=-1),
        "in_foot_ap": np.abs(in_foot[:, 0]), "in_foot_ml": np.abs(in_foot[:, 1]),
        "torque": series["our_torque"][loaded][:, 1] - series["gt_torque"][loaded][:, 1],
    }


def cop_metrics(errors: dict[str, np.ndarray]) -> dict[str, float]:
    mm = lambda k, f: 1000.0 * float(f(errors[k])) if len(errors[k]) else np.nan  # noqa: E731
    return {
        "CoP frames": float(len(errors["raw"])),
        "CoP raw [mm]": mm("raw", np.mean),
        "CoP hip-aligned [mm]": mm("aligned", np.mean),
        "CoP hip-aligned median [mm]": mm("aligned", np.median),
        "CoP in-foot [mm]": mm("in_foot", np.mean),
        "CoP in-foot AP [mm]": mm("in_foot_ap", np.mean),
        "CoP in-foot ML [mm]": mm("in_foot_ml", np.mean),
        "free moment RMSE [N·m]": (float(np.sqrt(np.mean(errors["torque"] ** 2)))
                                   if len(errors["torque"]) else np.nan),
    }


def stance_metrics(series: dict) -> dict[str, float]:
    """Precision / recall / F1 of our loaded state against the plates', both feet."""
    w = series["watched"]
    label = (series["gt_force"][..., 1] > LOADED_N)[w]
    pred = (series["our_force"][..., 1] > LOADED_N)[w]
    tp, fp, fn = (label & pred).sum(), (~label & pred).sum(), (label & ~pred).sum()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"stance precision": float(precision), "stance recall": float(recall),
            "stance F1": float(2 * precision * recall / max(precision + recall, 1e-9))}


def clip_metrics(series: dict) -> dict[str, float]:
    """One clip's row: the per-foot GRF block averaged over the two feet, plus CoP/stance."""
    body_weight = series["mass_kg"] * GRAVITY
    per_foot = [foot_metrics(series["gt_force"][:, i], series["our_force"][:, i],
                             series["watched"][:, i], body_weight, series["mass_ours"] * GRAVITY)
                for i in range(2)]
    out = {k: float(np.nanmean([f[k] for f in per_foot])) if np.isfinite([f[k] for f in per_foot]).any()
           else np.nan for k in per_foot[0]}
    out["stance frames"] = float(sum(f["stance frames"] for f in per_foot))
    for i, foot in enumerate(FEET):
        out[f"MAE V {foot[0].upper()} [%BW]"] = per_foot[i]["MAE V [%BW]"]
    out["mass gt/ours"] = series["mass_kg"] / series["mass_ours"]
    out.update(cop_metrics(cop_errors(series)))
    out.update(stance_metrics(series))
    return out


def mean_over_subjects(rows: list[dict[str, float]], subjects: list[str]) -> dict[str, float]:
    """Mean per subject, then mean across subjects; ``… SD`` columns carry the spread."""
    keys = list(rows[0])
    per_subject = {}
    for s in sorted(set(subjects)):
        mine = [r for r, subj in zip(rows, subjects) if subj == s]
        per_subject[s] = {k: float(np.nanmean([r[k] for r in mine]))
                          if np.isfinite([r[k] for r in mine]).any() else np.nan for k in keys}
    out = {}
    for k in keys:
        values = np.array([per_subject[s][k] for s in per_subject])
        if k.endswith("frames"):                      # counts are summed over clips, not averaged
            out[k] = float(np.nansum([r[k] for r in rows]))
            continue
        out[k] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
        if k in ("MAE V [%BW]", "CoP hip-aligned [mm]"):
            out[f"{k} SD"] = float(np.nanstd(values)) if np.isfinite(values).sum() > 1 else np.nan
    out["subjects"] = float(len(per_subject))
    out["clips"] = float(len(rows))
    return out


def cell(value: float, key: str) -> str:
    if not np.isfinite(value):
        return "—"
    if key.endswith("frames") or key in ("subjects", "clips"):
        return f"{value:.0f}"
    if key.startswith("r ") or key.startswith("stance") or key in ("fit gain", "mass gt/ours"):
        return f"{value:.2f}"
    return f"{value:.1f}"


def table(rows: dict[str, dict[str, float]], keys: list[str], first: str) -> list[str]:
    out = [f"| {first} | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for name, values in rows.items():
        out.append(f"| {name} | " + " | ".join(cell(values.get(k, np.nan), k) for k in keys) + " |")
    return out


GRF_KEYS = ["clips", "subjects", "stance frames", "MAE V [%BW]", "MAE V [%BW] SD",
            "MAE V own-BW [%BW]", "fit gain", "mass gt/ours", "MAE AP [%BW]",
            "MAE ML [%BW]", "MAE AP own-BW [%BW]", "MAE ML own-BW [%BW]", "MAE V all [%BW]",
            "MAE AP all [%BW]", "MAE ML all [%BW]", "MAE V all own-BW [%BW]",
            "MAE AP all own-BW [%BW]", "MAE ML all own-BW [%BW]", "RMSE V [%BW]", "rRMSE V [%]", "r V",
            "peak V err [%BW]", "MAE V L [%BW]", "MAE V R [%BW]"]
COP_KEYS = ["CoP frames", "CoP raw [mm]", "CoP hip-aligned [mm]", "CoP hip-aligned [mm] SD",
            "CoP hip-aligned median [mm]", "CoP in-foot [mm]", "CoP in-foot AP [mm]",
            "CoP in-foot ML [mm]", "free moment RMSE [N·m]", "stance precision",
            "stance recall", "stance F1"]
CLIP_KEYS = ["stance frames", "MAE V [%BW]", "MAE V own-BW [%BW]", "fit gain", "MAE AP [%BW]", "MAE ML [%BW]", "MAE V all [%BW]",
             "RMSE V [%BW]", "rRMSE V [%]", "r V", "peak V err [%BW]", "CoP hip-aligned [mm]",
             "CoP in-foot [mm]", "free moment RMSE [N·m]", "stance F1"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=PROCESSED)
    ap.add_argument("--run", default="human_optim_floorplane")
    ap.add_argument("--clips", type=Path, default=None,
                    help="score only the clips listed in this file (one name per line); the "
                         "summary goes to a directory suffixed with the file's stem")
    args = ap.parse_args(argv)

    out_root = args.source / "out"
    wanted = set(args.clips.read_text().split()) if args.clips else None
    clips = [c for c in sorted(out_root.iterdir()) if c.is_dir() and not c.name.startswith("_")
             and (wanted is None or c.name in wanted)
             and (c / args.run / "estimated_forces.npz").exists()]
    if not clips:
        print(f"[compare_plates] no clip under {out_root} has {args.run}/estimated_forces.npz")
        return 1
    if wanted is not None and len(clips) < len(wanted):
        print(f"[compare_plates] {len(wanted) - len(clips)} listed clips have no "
              f"{args.run}/estimated_forces.npz: {sorted(wanted - {c.name for c in clips})}")
    suffix = args.run.removeprefix("human_optim") if args.run.startswith("human_optim") else "_" + args.run
    compare_dir = out_root / ("_plates" + suffix + (f"_{args.clips.stem}" if args.clips else ""))
    compare_dir.mkdir(parents=True, exist_ok=True)

    all_series = []
    for clip in clips:
        s = clip_series(args.source / clip.name, clip / args.run)
        s["metrics"] = clip_metrics(s)
        all_series.append(s)
        m = s["metrics"]
        print(f"{clip.name:32s} MAE V {m['MAE V [%BW]']:6.1f} %BW  r_V {m['r V']:5.2f}  "
              f"CoP {m['CoP hip-aligned [mm]']:6.1f} mm  stance F1 {m['stance F1']:.2f}"
              f"{'  (sync uncertain)' if s['sync_uncertain'] else ''}")

    synced = [s for s in all_series if not s["sync_uncertain"]]
    unsure = [s for s in all_series if s["sync_uncertain"]]
    by_type: dict[str, list[dict]] = {}
    for s in synced:
        by_type.setdefault(s["type"], []).append(s)
    aggregate = lambda group: mean_over_subjects([s["metrics"] for s in group],  # noqa: E731
                                                 [s["subject"] for s in group])
    type_rows = {t: aggregate(v) for t, v in sorted(by_type.items())}
    rows = {"ALL (mean of activities)": {k: float((np.nansum if k.endswith("frames") else np.nanmean)
                                                  ([r[k] for r in type_rows.values()]))
                                         if np.isfinite([r[k] for r in type_rows.values()]).any()
                                         else np.nan for k in next(iter(type_rows.values()))},
            "ALL (mean of clips)": {k: float(np.nanmean([s["metrics"][k] for s in synced]))
                                    for k in synced[0]["metrics"]},
            **type_rows}
    rows["ALL (mean of activities)"]["clips"] = float(len(synced))
    rows["ALL (mean of activities)"]["subjects"] = float(len({s["subject"] for s in synced}))
    rows["ALL (mean of clips)"]["clips"] = float(len(synced))
    if unsure:
        rows["sync uncertain, excluded above"] = aggregate(unsure)

    clip_rows = {(f"{s['clip']} *" if s["sync_uncertain"] else s["clip"]): s["metrics"]
                 for s in all_series}
    lines = [f"# OpenCap LabValidation — plate-format comparison ({len(clips)} clips, run `{args.run}`)",
             "", "## Ground reaction force per foot, stance frames, %BW", "",
             *table(rows, GRF_KEYS, "activity"), "",
             "## Centre of pressure, free moment, stance detection", "",
             *table(rows, COP_KEYS, "activity"), "",
             "## Per clip", "",
             *table(clip_rows, CLIP_KEYS, "clip"), "",
             "Protocol: both sides low-pass filtered with OpenCap's per-activity zero-lag "
             "Butterworth (6 Hz walking, 4 Hz squat / sit-to-stand / static, drop jump unfiltered at "
             "60 Hz); forces in % body weight of the measured mass; per foot over the frames where "
             f"that foot's plate carries more than {STANCE_BW:.0%} BW (`MAE V all` over every frame "
             "the plates watched); the two feet averaged per clip (`MAE V L/R` separately); clip → "
             "subject → mean ± SD across subjects per activity; ALL averages the activities. CoP on "
             f"frames where both sides read above {LOADED_N:.0f} N vertically. Clips marked `*` "
             f"({len(unsure)}) have an uncertain video-to-mocap sync (`sync.json`) and only appear "
             "in their own row.", ""]
    (compare_dir / "summary.md").write_text("\n".join(lines))
    np.savez(compare_dir / "metrics.npz", rows=rows, clips=clip_rows, allow_pickle=True)
    print(f"\n[compare_plates] -> {compare_dir}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
