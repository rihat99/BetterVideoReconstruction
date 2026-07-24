"""Ablate the contact-completion signals as LABELERS: precision/recall, not AUROC.

Each signal (solved force, distance to scene, stillness) is turned into labels
the same way the readout does — threshold, 3-frame min-span, add-only union
with the auto labels — and scored as final labels against GT (macro over
scenes). This answers: what does each signal contribute, and which combination
reaches the target of precision > 0.9 with recall > 0.8?

Families::

    force        |f| > theta * body_weight        (the current readout)
    dist         distance < tau_d cm
    drift        drift_pre < tau_v m
    dist&drift   both kinematic gates             (= the auto detector, loosened)
    force&dist   force fires AND limb within tau_d
    force&drift  force fires AND limb stiller than tau_v

Also prints a false-positive autopsy of the current readout (theta = 0.005):
where the wrong labels are — their distance, stillness, limb, scene — and the
"air label" rate (labels further than 20 cm from any scene point).

Reads ``scratch/missing_contacts/`` bundles + completion forces; writes nothing.
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
from label_readout import _COMPLETION_TAG, _GRAVITY, _MIN_FRAMES, drop_short_spans, limb_force, prf  # noqa: E402

_LIMB_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")


def _min_span(on: np.ndarray) -> np.ndarray:
    """(P, N, 4) bool -> same, spans shorter than ``_MIN_FRAMES`` dropped."""
    out = on.copy()
    for p in range(on.shape[0]):
        for j in range(on.shape[2]):
            out[p, :, j] = drop_short_spans(out[p, :, j], _MIN_FRAMES)
    return out


def load_all(tag: str = _COMPLETION_TAG) -> list[dict]:
    scenes = []
    comp_dir = _REPO_ROOT / "scratch" / "missing_contacts" / "completion"
    for f in sorted(comp_dir.glob(f"*__{tag}.npz")):
        stem = f.name.split("__")[0]
        d = load(stem)
        comp = np.load(f)
        dyn = np.load(_REPO_ROOT / "out" / stem / "human_optim" / "dynamics.npz")
        mass_of = dict(zip(np.asarray(dyn["object_ids"], int), np.asarray(dyn["total_mass"])))
        weight_n = np.array([mass_of[int(o)] * _GRAVITY for o in d["object_ids"]])
        limb_auto = np.stack([d["group_auto"][..., 0], d["group_auto"][..., 1],
                              d["group_auto"][..., [2, 4]].any(-1),
                              d["group_auto"][..., [3, 5]].any(-1)], -1)
        scenes.append(dict(
            stem=stem,
            fmag=limb_force(comp["forces"]),                    # (P, N, 4) N
            dist=-d["dist"],                                    # (P, N, 4) cm (inf = no point)
            drift=-d["drift"],                                  # (P, N, 4) m
            auto=limb_auto,
            scored=d["valid"][..., None] & (d["gt"] >= 0),
            gt_on=d["gt"] == 1,
            weight_n=weight_n,
        ))
    return scenes


def score(scenes: list[dict], labeler) -> tuple[float, float, float]:
    """Macro final-label P/R/F1 of ``completed = auto | min_span(labeler(scene))``."""
    rows = []
    for s in scenes:
        completed = s["auto"] | _min_span(labeler(s))
        rows.append(prf(completed[s["scored"]], s["gt_on"][s["scored"]]))
    return tuple(np.mean([r[i] for r in rows]) for i in range(3))


def sweep(scenes: list[dict]) -> None:
    thetas = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]
    taud = [2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0]
    tauv = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]

    families: dict[str, list[tuple[str, tuple[float, float, float]]]] = {}

    def add(family: str, tag: str, labeler) -> None:
        families.setdefault(family, []).append((tag, score(scenes, labeler)))

    for t in thetas:
        add("force", f"th={t:g}", lambda s, t=t: s["fmag"] > t * s["weight_n"][:, None, None])
    for td in taud:
        add("dist", f"d<{td:g}cm", lambda s, td=td: s["dist"] < td)
    for tv in tauv:
        add("drift", f"v<{tv:g}m", lambda s, tv=tv: s["drift"] < tv)
    for td in taud:
        for tv in tauv:
            add("dist&drift", f"d<{td:g},v<{tv:g}",
                lambda s, td=td, tv=tv: (s["dist"] < td) & (s["drift"] < tv))
    for t in thetas:
        for td in taud:
            add("force&dist", f"th={t:g},d<{td:g}",
                lambda s, t=t, td=td: (s["fmag"] > t * s["weight_n"][:, None, None])
                                      & (s["dist"] < td))
    for t in thetas:
        for tv in tauv:
            add("force&drift", f"th={t:g},v<{tv:g}",
                lambda s, t=t, tv=tv: (s["fmag"] > t * s["weight_n"][:, None, None])
                                      & (s["drift"] < tv))
    for t in thetas:
        for td in (20.0, 50.0):
            for tv in (0.08, 0.12):
                add("f&d&v", f"th={t:g},d<{td:g},v<{tv:g}",
                    lambda s, t=t, td=td, tv=tv:
                    (s["fmag"] > t * s["weight_n"][:, None, None])
                    & (s["dist"] < td) & (s["drift"] < tv))

    base = np.mean([prf(s["auto"][s["scored"]], s["gt_on"][s["scored"]]) for s in scenes], 0)
    print(f"auto-only baseline: P {base[0]:.3f} R {base[1]:.3f} F1 {base[2]:.3f}\n")
    print(f"{'family':12s} {'best-F1 config':>22s} {'P':>6s} {'R':>6s} {'F1':>6s}"
          f"   {'best config with P>=0.9':>26s} {'P':>6s} {'R':>6s}")
    for fam, rows in families.items():
        best = max(rows, key=lambda r: r[1][2])
        hi = [r for r in rows if r[1][0] >= 0.90]
        hi_best = max(hi, key=lambda r: r[1][1]) if hi else None
        hi_txt = (f"{hi_best[0]:>26s} {hi_best[1][0]:6.3f} {hi_best[1][1]:6.3f}"
                  if hi_best else f"{'—':>26s}      —      —")
        print(f"{fam:12s} {best[0]:>22s} {best[1][0]:6.3f} {best[1][1]:6.3f} {best[1][2]:6.3f}"
              f"   {hi_txt}")

    print("\nconfigs reaching the target (P >= 0.9 AND R >= 0.8):")
    any_hit = False
    for fam, rows in families.items():
        for tag, (p, r, f1) in rows:
            if p >= 0.90 and r >= 0.80:
                print(f"  {fam:12s} {tag:>22s}  P {p:.3f} R {r:.3f} F1 {f1:.3f}")
                any_hit = True
    if not any_hit:
        print("  none")


def per_scene(scenes: list[dict]) -> None:
    """Per-scene final-label P/R of the head-to-head operating points."""
    configs = [
        ("drift v<0.12", lambda s: s["drift"] < 0.12),
        ("dist&drift 50/0.12", lambda s: (s["dist"] < 50) & (s["drift"] < 0.12)),
        ("force th=0.005", lambda s: s["fmag"] > 0.005 * s["weight_n"][:, None, None]),
        ("force&drift 0.0025/0.12",
         lambda s: (s["fmag"] > 0.0025 * s["weight_n"][:, None, None]) & (s["drift"] < 0.12)),
        ("f&d&v 0.0025/50/0.12",
         lambda s: (s["fmag"] > 0.0025 * s["weight_n"][:, None, None])
                   & (s["dist"] < 50) & (s["drift"] < 0.12)),
    ]
    print("\n== per-scene final-label P/R at the head-to-head operating points")
    print(f"{'scene':22s} " + " ".join(f"{name:>24s}" for name, _ in configs))
    for s in scenes:
        cells = []
        for _, lab in configs:
            completed = s["auto"] | _min_span(lab(s))
            p, r, _ = prf(completed[s["scored"]], s["gt_on"][s["scored"]])
            cells.append(f"{p:.3f}/{r:.3f}")
        star = "*" if s["stem"] in _FLOATERS else " "
        print(f"{s['stem']:21s}{star} " + " ".join(f"{c:>24s}" for c in cells))


def fp_autopsy(scenes: list[dict], theta: float = 0.005) -> None:
    print(f"\n== false-positive autopsy of the readout at theta={theta:g}")
    rows = []
    for s in scenes:
        new = _min_span(s["fmag"] > theta * s["weight_n"][:, None, None]) & ~s["auto"]
        m = s["scored"] & new
        tp, fp = m & s["gt_on"], m & ~s["gt_on"]
        rows.append((s["stem"], tp, fp, s))
    print(f"{'scene':22s} {'newTP':>6s} {'newFP':>6s} {'FP med d(cm)':>12s} {'TP med d':>9s}"
          f" {'FP med v(m)':>11s} {'TP med v':>9s} {'FP air>20cm':>11s}")
    for stem, tp, fp, s in rows:
        star = "*" if stem in _FLOATERS else " "
        def med(mask, sig):
            v = s[sig][mask]
            v = v[np.isfinite(v)]
            return float(np.median(v)) if len(v) else float("nan")
        air = float((s["dist"][fp] > 20).mean()) if fp.sum() else float("nan")
        print(f"{stem:21s}{star} {tp.sum():6d} {fp.sum():6d} {med(fp, 'dist'):12.1f}"
              f" {med(tp, 'dist'):9.1f} {med(fp, 'drift'):11.3f} {med(tp, 'drift'):9.3f}"
              f" {air:11.2f}")
    tp_all = np.concatenate([tp[s["scored"]] for _, tp, _, s in rows])
    fp_all = np.concatenate([fp[s["scored"]] for _, _, fp, s in rows])
    print(f"pooled: newTP {int(tp_all.sum())}  newFP {int(fp_all.sum())} "
          f"(new-label precision {tp_all.sum() / max(tp_all.sum() + fp_all.sum(), 1):.3f})")
    per_limb_tp = np.zeros(4, int)
    per_limb_fp = np.zeros(4, int)
    for _, tp, fp, s in rows:
        per_limb_tp += tp.sum((0, 1))
        per_limb_fp += fp.sum((0, 1))
    for j, name in enumerate(_LIMB_NAMES):
        print(f"  {name:11s} TP {per_limb_tp[j]:5d}  FP {per_limb_fp[j]:5d}"
              f"  (precision {per_limb_tp[j] / max(per_limb_tp[j] + per_limb_fp[j], 1):.3f})")


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else _COMPLETION_TAG
    scenes = load_all(tag)
    print(f"tag: {tag}  {len(scenes)} scenes\n")
    sweep(scenes)
    per_scene(scenes)
    fp_autopsy(scenes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
