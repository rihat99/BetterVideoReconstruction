"""Turn completion-solve forces into contact labels and score them against GT.

The readout (per person, limb, frame): contact iff the solved force magnitude
exceeds ``theta`` x body weight, then spans shorter than ``MIN_FRAMES`` are
dropped (the contact detector's own min-span rule). Labels are **add-only**:
``completed = auto ∪ readout`` — completion never removes an auto label.

Two scores per threshold, macro over scenes:

* **new-contact F1** — readout vs GT restricted to candidates (auto off), the
  detection view;
* **final label F1** — completed vs GT over all valid limb-frames, next to the
  auto-only baseline: the number that says how much closer to GT the labels get.

Reads ``scratch/missing_contacts/`` (sweep bundles + completion forces) and the
scenes' ``dynamics.npz`` for body mass; writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "diagnostics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_missing_contacts import _FLOATERS, load  # noqa: E402

_COMPLETION_TAG = "v0.2_d0.05_mag"
_MIN_FRAMES = 3          # same min-span rule as the contact detector
_GRAVITY = 9.81


def limb_force(f6: np.ndarray) -> np.ndarray:
    """(P, N, 6, 3) group forces -> (P, N, 4) limb magnitude (foot = max(foot, ankle))."""
    m = np.linalg.norm(f6, axis=-1)
    return np.stack([m[..., 0], m[..., 1], np.maximum(m[..., 2], m[..., 4]),
                     np.maximum(m[..., 3], m[..., 5])], -1)


def drop_short_spans(on: np.ndarray, min_frames: int) -> np.ndarray:
    """Zero out runs of True shorter than ``min_frames`` along a 1-D bool array."""
    out = on.copy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], on.view(np.int8), [0]])))
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start < min_frames:
            out[start:stop] = False
    return out


def readout(fmag: np.ndarray, weight_n: np.ndarray, theta: float) -> np.ndarray:
    """(P, N, 4) force magnitude -> bool labels at ``theta`` x body weight + min-span."""
    on = fmag > theta * weight_n[:, None, None]
    for p in range(on.shape[0]):
        for j in range(on.shape[2]):
            on[p, :, j] = drop_short_spans(on[p, :, j], _MIN_FRAMES)
    return on


def prf(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = float((pred & gt).sum())
    if tp == 0.0:
        return 0.0, 0.0, 0.0
    p = tp / max(pred.sum(), 1)
    r = tp / max(gt.sum(), 1)
    return p, r, 2 * p * r / (p + r)


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else _COMPLETION_TAG
    thetas = [0.005, 0.01, 0.02, 0.03, 0.05]
    per_theta: dict[float, list] = {t: [] for t in thetas}
    base_rows = []
    stems = sorted(f.name.split("__")[0] for f in
                   (_REPO_ROOT / "scratch" / "missing_contacts" / "completion")
                   .glob(f"*__{tag}.npz"))
    print(f"tag: {tag}  ({len(stems)} scenes)")
    for stem in stems:
        d = load(stem)
        comp = np.load(_REPO_ROOT / "scratch" / "missing_contacts" / "completion" /
                       f"{stem}__{tag}.npz")
        dyn = np.load(_REPO_ROOT / "out" / stem / "human_optim" / "dynamics.npz")
        mass_of = dict(zip(np.asarray(dyn["object_ids"], int), np.asarray(dyn["total_mass"])))
        weight_n = np.array([mass_of[int(o)] * _GRAVITY for o in d["object_ids"]])

        fmag = limb_force(comp["forces"])
        limb_auto = np.stack([d["group_auto"][..., 0], d["group_auto"][..., 1],
                              d["group_auto"][..., [2, 4]].any(-1),
                              d["group_auto"][..., [3, 5]].any(-1)], -1)
        scored = d["valid"][..., None] & (d["gt"] >= 0)
        gt_on = d["gt"] == 1
        base_rows.append(prf(limb_auto[scored], gt_on[scored]))
        for t in thetas:
            new = readout(fmag, weight_n, t)
            cand = scored & ~limb_auto
            f1_new = prf(new[cand], gt_on[cand])
            completed = limb_auto | new                                   # add-only
            f1_fin = prf(completed[scored], gt_on[scored])
            per_theta[t].append((stem, f1_new, f1_fin))

    print(f"auto-only baseline  P/R/F1 macro: "
          + "/".join(f"{np.mean([r[i] for r in base_rows]):.3f}" for i in range(3)))
    print(f"\n{'theta':>6s} {'new-contact P/R/F1':>22s} {'final-label P/R/F1':>22s}")
    for t in thetas:
        rows = per_theta[t]
        n = [np.mean([r[1][i] for r in rows]) for i in range(3)]
        f = [np.mean([r[2][i] for r in rows]) for i in range(3)]
        print(f"{t:6.2f} {n[0]:8.3f}/{n[1]:.3f}/{n[2]:.3f} {f[0]:12.3f}/{f[1]:.3f}/{f[2]:.3f}")

    best = max(thetas, key=lambda t: np.mean([r[2][2] for r in per_theta[t]]))
    print(f"\nbest theta by final-label F1: {best}")
    print(f"\nper-scene final-label F1 at theta={best} (auto-only -> completed):")
    for (stem, _, fin), base in zip(per_theta[best], base_rows):
        star = "*" if stem in _FLOATERS else " "
        print(f"  {stem:21s}{star} {base[2]:.3f} -> {fin[2]:.3f}   (P {base[0]:.3f}->{fin[0]:.3f}, "
              f"R {base[1]:.3f}->{fin[1]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
