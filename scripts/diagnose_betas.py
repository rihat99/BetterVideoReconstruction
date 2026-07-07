"""Betas noise diagnostic for round-5 T1 (stop-or-go).

Per clip / person, ask three things about the per-frame SAM-3D-Body ``betas``:

1. how noisy are they over the clip (per-dim std + MAD)?
2. do they track how large / well-seen the person is (limb length vs bbox
   height and vs mean Sapiens keypoint score)?
3. how much would a quality-weighted robust aggregate move the **implied limb
   lengths** versus the plain mean the optimizer uses today
   (``stages.py`` ``betas_mean``)?

The mechanism the round-5 plan (``docs/UPDATE_AGENT.md`` §2) is chasing: when the
climber is small / truncated in the image, per-frame betas regress toward the
average human — short limbs for a tall climber — and dilute the good frames. If
that is real, small-bbox frames imply shorter limbs and a weighted median moves
the arm/leg length by a visible number of millimetres. If betas are already
tight (implied-limb spread and the robust-vs-mean delta both ≲ 2 mm), the fix is
a no-op and should be skipped.

Read-only; writes one plot per clip under ``out/<clip>/human_optim/diagnostics/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.geometry.human_scale import weighted_median  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402

_DEFAULT_SMPLX_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
)
_ARM = (("left_shoulder", "left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow", "right_wrist"))
_LEG = (("left_hip", "left_knee", "left_ankle"),
        ("right_hip", "right_knee", "right_ankle"))


def _rest_joints(mid, betas: np.ndarray, device: str) -> np.ndarray:
    """Rest-pose (shape-only) joint positions for each betas row → ``(F, J, 3)``."""
    with torch.no_grad():
        _, joints = mid.forward_shape(torch.as_tensor(betas, dtype=torch.float32, device=device))
    return joints.cpu().numpy()


def _limb_lengths(joints: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """L/R-averaged arm and leg length (m) per frame from rest-pose ``joints (F,J,3)``.

    Limb length = summed bone chords (shoulder→elbow→wrist, hip→knee→ankle) — a
    pose-independent, purely shape-driven measure since the joints are at rest.
    """
    idx = {n: i for i, n in enumerate(names)}

    def chain(a: str, b: str, c: str) -> np.ndarray:
        ab = np.linalg.norm(joints[:, idx[a]] - joints[:, idx[b]], axis=-1)
        bc = np.linalg.norm(joints[:, idx[b]] - joints[:, idx[c]], axis=-1)
        return ab + bc

    arm = 0.5 * (chain(*_ARM[0]) + chain(*_ARM[1]))
    leg = 0.5 * (chain(*_LEG[0]) + chain(*_LEG[1]))
    return arm, leg


def _robust_betas(betas: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Per-dim weighted median of ``betas (F,10)`` with per-frame ``weight (F,)`` → ``(10,)``."""
    return np.array([weighted_median(betas[:, d], weight) for d in range(betas.shape[1])],
                    dtype=np.float64)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, ``nan`` if either side is (near-)constant."""
    if a.size < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _match_row(object_ids: np.ndarray, oid: int) -> int | None:
    ids = [int(x) for x in object_ids]
    return ids.index(oid) if oid in ids else None


def _plot(path: Path, oid: int, arm: np.ndarray, leg: np.ndarray,
          bbh: np.ndarray, sap: np.ndarray, frames: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_q) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax_l.plot(frames, arm * 100, lw=1.2, label="arm (shoulder→wrist)")
    ax_l.plot(frames, leg * 100, lw=1.2, label="leg (hip→ankle)")
    ax_l.set_ylabel("implied limb length (cm)")
    ax_l.set_title(f"person {oid}: implied limb length & frame quality over time")
    ax_l.legend(fontsize=8)
    ax_l.grid(alpha=0.3)
    ax_q.plot(frames, bbh, lw=1.0, color="tab:green", label="bbox height (norm)")
    ax_q.plot(frames, sap, lw=1.0, color="tab:red", label="mean Sapiens score")
    ax_q.set_ylabel("frame quality")
    ax_q.set_xlabel("frame index")
    ax_q.legend(fontsize=8)
    ax_q.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def diagnose_clip(clip_dir: Path, mid, names: list[str], device: str) -> None:
    sp = dict(np.load(clip_dir / "sam3d" / "smplx_params.npz", allow_pickle=True))
    bb = dict(np.load(clip_dir / "sam3" / "bboxes.npz", allow_pickle=True))
    po = dict(np.load(clip_dir / "sapiens" / "pose.npz", allow_pickle=True))

    P = int(sp["num_people"])
    object_ids = np.asarray(sp["object_ids"], dtype=np.int32)
    betas_all = np.asarray(sp["betas"], dtype=np.float64)          # (P, N, 10)
    valid_all = np.asarray(sp["valid_mask"], dtype=bool)           # (P, N)
    bboxes = np.asarray(bb["bboxes_per_obj"], dtype=np.float64)    # (Pb, N, 4) x1y1x2y2
    scores = np.asarray(po["keypoint_scores"], dtype=np.float64)  # (Ps, N, 308)

    print(f"\n=== {clip_dir.name}  (P={P}) ===")
    for p in range(P):
        oid = int(object_ids[p])
        vmask = valid_all[p]
        fv = np.flatnonzero(vmask)
        if fv.size < 5:
            print(f"  person {oid}: <5 valid frames — skip")
            continue
        betas = betas_all[p][fv]                                   # (Fv, 10)

        pb = _match_row(bb["object_ids"], oid)
        h = bboxes[pb][fv, 3] - bboxes[pb][fv, 1] if pb is not None else np.full(fv.size, np.nan)
        bbh = h / h.max() if np.isfinite(h).any() and h.max() > 0 else np.zeros_like(h)
        ps = _match_row(po["object_ids"], oid)
        sap = scores[ps][fv].mean(axis=1) if ps is not None else np.zeros(fv.size)  # mean over 308 kp
        weight = bbh * sap

        joints = _rest_joints(mid, betas, device)
        arm, leg = _limb_lengths(joints, names)

        b_mean = betas.mean(axis=0)
        b_rob = _robust_betas(betas, weight)
        jl = _rest_joints(mid, np.stack([b_mean, b_rob]), device)   # (2, J, 3): [mean, robust]
        arm_agg, leg_agg = _limb_lengths(jl, names)                 # each (2,)
        arm_m, arm_r = float(arm_agg[0]), float(arm_agg[1])
        leg_m, leg_r = float(leg_agg[0]), float(leg_agg[1])

        per_dim_std = betas.std(axis=0)
        per_dim_mad = np.median(np.abs(betas - np.median(betas, axis=0)), axis=0)
        print(f"  person {oid}: {fv.size} valid frames")
        print(f"    betas std   (per-dim max / mean): {per_dim_std.max():.3f} / {per_dim_std.mean():.3f}")
        print(f"    betas MAD   (per-dim max / mean): {per_dim_mad.max():.3f} / {per_dim_mad.mean():.3f}")
        print(f"    arm length  mean={arm_m*100:.2f}cm  robust={arm_r*100:.2f}cm  "
              f"Δ(rob−mean)={(arm_r-arm_m)*1000:+.1f}mm  spread(std)={arm.std()*1000:.1f}mm")
        print(f"    leg length  mean={leg_m*100:.2f}cm  robust={leg_r*100:.2f}cm  "
              f"Δ(rob−mean)={(leg_r-leg_m)*1000:+.1f}mm  spread(std)={leg.std()*1000:.1f}mm")
        print(f"    corr(arm, bbox_h)={_corr(arm, bbh):+.2f}  corr(arm, sapiens)={_corr(arm, sap):+.2f}"
              f"   corr(leg, bbox_h)={_corr(leg, bbh):+.2f}  corr(leg, sapiens)={_corr(leg, sap):+.2f}")

        _plot(clip_dir / "human_optim" / "diagnostics" / f"betas_person{oid}.png",
              oid, arm, leg, bbh, sap, fv)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "out", help="out/ root")
    ap.add_argument("--clips", nargs="+",
                    default=["boulder_1", "boulder_3", "competition_1", "competition_2", "sport_climb"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    mid = load_smplx_mid(str(_DEFAULT_SMPLX_MODEL), args.device)
    names = list(mid.joint_names)
    for clip in args.clips:
        d = args.out / clip
        if not (d / "sam3d" / "smplx_params.npz").exists():
            print(f"\n=== {clip}: no sam3d/smplx_params.npz — skip ===")
            continue
        diagnose_clip(d, mid, names, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
