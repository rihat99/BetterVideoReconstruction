"""Precision-first sweep of contact labelers. Target: P >= 0.95, R >= 0.90.

Label structure swept::

    labels = (auto MINUS veto) ∪ span(gates) ∪ span(force-term)
    gates      = dist < τd  AND  drift < τv
    force-term = |f| > θ·BW  [AND dist < d2]  [AND drift < v2]
    veto       = auto AND |f| < θv·BW        (force is the only signal
                 orthogonal to the detector's own near+still false positives)

Spans are applied per component (3 or 5 frames), so components are
precomputed once per scene and configs are pure boolean unions — the full
grid costs seconds.

Selection is honest: leave-one-source-out over the 30 annotated test scenes
(pick the best feasible config on the 12 tuning scenes + the other sources,
score the left-out source), reported next to the in-sample frontiers. Reads
``scratch/missing_contacts/sweep_cache.pkl`` (built by the probe run);
writes nothing.
"""

from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "diagnostics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ablation_labelers import _min_span  # noqa: E402
from heldout_completion import _TUNED  # noqa: E402
from label_readout import prf  # noqa: E402

_CACHE = _REPO_ROOT / "scratch" / "missing_contacts" / "sweep_cache.pkl"

TAUD = [10.0, 20.0, 30.0]
TAUV = [0.06, 0.08, 0.10]
THETA = [0.01, 0.015, 0.02, 0.03, 0.05]
D2 = [30.0, 50.0, np.inf]
V2 = [0.15, 0.20, np.inf]
SPAN = [3, 5]
VETO = [0.0, 0.002, 0.005]


def _span(mask: np.ndarray, n: int) -> np.ndarray:
    import label_readout
    old = label_readout._MIN_FRAMES
    try:
        label_readout._MIN_FRAMES = n
        return _min_span(mask)
    finally:
        label_readout._MIN_FRAMES = old


def precompute(scene: dict) -> dict:
    """Span-filtered gate and force-term component masks for one scene."""
    f_rel = scene["fmag"] / scene["weight_n"][:, None, None]
    comp = {"gate": {}, "force": {}, "veto": {}}
    for td, tv, sp in itertools.product(TAUD, TAUV, SPAN):
        comp["gate"][(td, tv, sp)] = _span((scene["dist"] < td) & (scene["drift"] < tv), sp)
    for th, d2, v2, sp in itertools.product(THETA, D2, V2, SPAN):
        m = f_rel > th
        if np.isfinite(d2):
            m = m & (scene["dist"] < d2)
        if np.isfinite(v2):
            m = m & (scene["drift"] < v2)
        comp["force"][(th, d2, v2, sp)] = _span(m, sp)
    for tv_ in VETO:
        comp["veto"][tv_] = scene["auto"] & (f_rel < tv_) if tv_ > 0 else np.zeros_like(scene["auto"])
    return comp


def config_labels(scene: dict, comp: dict, cfgk: tuple) -> np.ndarray:
    (td, tv, gsp), (th, d2, v2, fsp), veto = cfgk
    lab = (scene["auto"] & ~comp["veto"][veto]) | comp["gate"][(td, tv, gsp)]
    if th is not None:
        lab = lab | comp["force"][(th, d2, v2, fsp)]
    return lab


def macro(scenes, comps, cfgk) -> tuple:
    rows = [prf(config_labels(s, c, cfgk)[s["scored"]], s["gt_on"][s["scored"]])
            for s, c in zip(scenes, comps)]
    return tuple(np.mean([r[i] for r in rows]) for i in range(3))


def all_configs():
    gates = list(itertools.product(TAUD, TAUV, SPAN))
    force = [None] + list(itertools.product(THETA, D2, V2, SPAN))
    for g, f, v in itertools.product(gates, force, VETO):
        yield (g, (None, None, None, None) if f is None else f, v)


def feasible_best(scenes, comps, p_t=0.95, r_t=0.90):
    """Best config by: feasibility (P>=p_t and R>=r_t), then max F1; fallback
    max min(P/p_t, R/r_t)."""
    best, best_key, feas = None, None, False
    for cfgk in all_configs():
        p, r, f1 = macro(scenes, comps, cfgk)
        ok = p >= p_t and r >= r_t
        key = (1, f1) if ok else (0, min(p / p_t, r / r_t))
        if best_key is None or key > best_key:
            best, best_key, feas = (cfgk, (p, r, f1)), key, ok
    return best, feas


def main() -> int:
    data = pickle.loads(_CACHE.read_bytes())
    comps = [precompute(s) for s in data]
    tune = [(s, c) for s, c in zip(data, comps) if s["dset"] == "tune"]
    test28 = [(s, c) for s, c in zip(data, comps)
              if s["dset"] == "test" and s["stem"] not in _TUNED]

    def unzip(sc):
        return [x[0] for x in sc], [x[1] for x in sc]

    # In-sample frontier on the 28 held-out scenes: feasible configs.
    print("== feasible configs on the 28 held-out scenes (P>=0.95, R>=0.90), top by F1")
    rows = []
    s28, c28 = unzip(test28)
    for cfgk in all_configs():
        p, r, f1 = macro(s28, c28, cfgk)
        if p >= 0.95 and r >= 0.90:
            rows.append((f1, p, r, cfgk))
    rows.sort(reverse=True)
    for f1, p, r, cfgk in rows[:12]:
        (td, tv, gsp), (th, d2, v2, fsp), veto = cfgk
        ftxt = "-" if th is None else f"f>{th:g} d2<{d2:g} v2<{v2:g} sp{fsp}"
        print(f"  P {p:.3f} R {r:.3f} F1 {f1:.3f}  gates d<{td:g} v<{tv:g} sp{gsp}"
              f"  [{ftxt}]  veto<{veto:g}")
    print(f"  ({len(rows)} feasible configs)")

    # LOSO by source over ALL 30 test scenes; selection set = 12 tuning + other sources.
    print("\n== leave-one-source-out (honest held-out estimate)")
    test_all = [(s, c) for s, c in zip(data, comps) if s["dset"] == "test"]
    sources = sorted({s["source"] for s, _ in test_all})
    loso_rows = []
    for src in sources:
        sel = tune + [(s, c) for s, c in test_all if s["source"] != src]
        hold = [(s, c) for s, c in test_all if s["source"] == src]
        (cfgk, _), feas = feasible_best(*unzip(sel))
        p, r, f1 = macro(*unzip(hold), cfgk)
        loso_rows.append((len(hold), p, r))
        (td, tv, gsp), (th, d2, v2, fsp), veto = cfgk
        ftxt = "-" if th is None else f"f>{th:g}/{d2:g}/{v2:g}/sp{fsp}"
        print(f"  {src:14s} ({len(hold):2d} scenes) P {p:.3f} R {r:.3f}"
              f"   picked: d<{td:g} v<{tv:g} sp{gsp} [{ftxt}] veto<{veto:g}"
              f" {'(feasible)' if feas else '(fallback)'}")
    w = np.array([r[0] for r in loso_rows], float)
    ps = np.array([r[1] for r in loso_rows])
    rs = np.array([r[2] for r in loso_rows])
    print(f"  source-macro : P {ps.mean():.3f} R {rs.mean():.3f}")
    print(f"  scene-weighted: P {(ps*w).sum()/w.sum():.3f} R {(rs*w).sum()/w.sum():.3f}")

    # The overall pick (tuning + all test used for selection — in-sample, for the record)
    print("\n== overall pick selected on tune12 only, evaluated everywhere")
    (cfgk, sel_num), feas = feasible_best(*unzip(tune))
    for name, part in [("tune12", tune), ("held-out 28", test28)]:
        p, r, f1 = macro(*unzip(part), cfgk)
        print(f"  {name:12s} P {p:.3f} R {r:.3f} F1 {f1:.3f}")
    (td, tv, gsp), (th, d2, v2, fsp), veto = cfgk
    print(f"  config: gates d<{td:g} v<{tv:g} sp{gsp}; force "
          + ("-" if th is None else f">{th:g} d2<{d2:g} v2<{v2:g} sp{fsp}")
          + f"; veto<{veto:g}  ({'feasible on tune12' if feas else 'fallback on tune12'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
