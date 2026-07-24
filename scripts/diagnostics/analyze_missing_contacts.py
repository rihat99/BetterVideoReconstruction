"""Decompose the missing-contact scores saved by ``missing_contacts.py``.

The pooled (person, frame, limb) AUROC in the sweep mixes two different
questions. This script separates them, per scene and pooled:

1. **THAT** — frame level: does the residual magnitude detect frames where at
   least one candidate limb is missing a GT contact? (AUROC of ``|w|``, its
   force part, and the counterfactual total force, over frames with >=1
   candidate limb.)
2. **WHICH** — within a miss frame: does each score's top-ranked candidate limb
   hit a GT-missed one? (top-1 hit rate over frames with >=1 missed and >=1
   clean candidate limb, vs the random-pick baseline.)
3. **Combination** — pooled candidate AUROC of simple rank-averages (no learned
   weights): each method's candidate ranks are averaged per scene.

Reads ``scratch/missing_contacts/<stem>.npz`` only; prints tables, writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.diagnostics.missing_contacts import _GT_DEFECTS, auroc  # noqa: E402

_DIR = _REPO_ROOT / "scratch" / "missing_contacts"
_FLOATERS = {"2nwG1Qa5k_c_0115", "9RvrpYoWDoo_0001", "gtIGgs_y2nY_0024", "33ODIivxhF8_0025"}
_METHODS = ("probe_abs", "probe_tq", "wnorm", "dist", "drift", "resolve")


def _clean(scores: np.ndarray) -> np.ndarray:
    """NaN scores (e.g. distance where no scene point was found) rank last."""
    return np.where(np.isnan(scores), -np.inf, scores)


def load(stem: str) -> dict:
    d = dict(np.load(_DIR / f"{stem}.npz", allow_pickle=True))
    if stem in _GT_DEFECTS:                        # saved bundles predate the GT patch
        lo, hi = _GT_DEFECTS[stem]
        d["gt"][:, lo:hi] = -1
    limb_auto = np.stack([d["group_auto"][..., [0]].any(-1), d["group_auto"][..., [1]].any(-1),
                          d["group_auto"][..., [2, 4]].any(-1), d["group_auto"][..., [3, 5]].any(-1)], -1)
    d["cand"] = d["valid"][..., None] & ~limb_auto & (d["gt"] >= 0)         # (P, N, 4)
    d["missed"] = d["cand"] & (d["gt"] == 1)
    return d


def frame_level(d: dict) -> dict[str, float]:
    """AUROC of frame-level scores for 'this frame misses >=1 contact'."""
    has_cand = d["cand"].any(-1)                                            # (P, N)
    label = d["missed"].any(-1)[has_cand]
    w = d["w_world"]
    out = {}
    for name, s in [("|w|", np.linalg.norm(w, axis=-1)),
                    ("|w_f|", np.linalg.norm(w[..., :3], axis=-1)),
                    ("|w_tq|", np.linalg.norm(w[..., 3:], axis=-1)),
                    ("resolve_sum", np.linalg.norm(d["resolve_f6"], axis=-1).sum(-1)
                     if "resolve_f6" in d else None)]:
        out[name] = auroc(s[has_cand], label) if s is not None else float("nan")
    out["miss_rate"] = float(label.mean())
    return out


def which_limb(d: dict) -> dict[str, float]:
    """Top-1 hit rate: among frames with >=1 missed and >=1 clean candidate limb."""
    sel = (d["missed"].sum(-1) >= 1) & ((d["cand"] & ~d["missed"]).sum(-1) >= 1)  # (P, N)
    out = {"frames": int(sel.sum())}
    if not sel.any():
        return out
    rnd = []
    for p, t in zip(*np.nonzero(sel)):
        rnd.append(d["missed"][p, t].sum() / d["cand"][p, t].sum())
    out["random"] = float(np.mean(rnd))
    for m in _METHODS:
        s = _clean(d[m].astype(np.float64))
        hits = []
        for p, t in zip(*np.nonzero(sel)):
            c = d["cand"][p, t]
            best = np.argmax(np.where(c, s[p, t], -np.inf))
            hits.append(bool(d["missed"][p, t, best]))
        out[m] = float(np.mean(hits))
    return out


def combos(d: dict) -> dict[str, float]:
    """Pooled candidate AUROC of rank-averaged score pairs/triples."""
    from scipy.stats import rankdata
    cand = d["cand"]
    labels = d["missed"][cand]
    ranks = {m: rankdata(_clean(d[m].astype(np.float64))[cand]) for m in _METHODS}
    out = {}
    for name, ms in [("drift", ["drift"]), ("drift+dist", ["drift", "dist"]),
                     ("drift+wnorm", ["drift", "wnorm"]),
                     ("drift+resolve", ["drift", "resolve"]),
                     ("drift+dist+wnorm", ["drift", "dist", "wnorm"]),
                     ("drift+dist+resolve", ["drift", "dist", "resolve"]),
                     ("dist+resolve", ["dist", "resolve"])]:
        out[name] = auroc(np.mean([ranks[m] for m in ms], axis=0), labels)
    return out


def main() -> int:
    stems = sorted(p.stem for p in _DIR.glob("*.npz"))
    tables = {fn.__name__: {} for fn in (frame_level, which_limb, combos)}
    for stem in stems:
        d = load(stem)
        for fn in (frame_level, which_limb, combos):
            tables[fn.__name__][stem] = fn(d)

    for name, table in tables.items():
        keys = [k for k in next(iter(table.values())) if k not in ("frames",)]
        print(f"\n== {name}")
        print(f"{'scene':22s} " + " ".join(f"{k:>12s}" for k in keys))
        for stem, row in table.items():
            tag = "*" if stem in _FLOATERS else " "
            print(f"{stem:21s}{tag} " + " ".join(
                f"{row.get(k, float('nan')):12.3f}" for k in keys))
        core = [r for s, r in table.items() if s not in _FLOATERS]
        flt = [r for s, r in table.items() if s in _FLOATERS]
        for lbl, rows in [("mean core", core), ("mean floaters*", flt)]:
            vals = [np.nanmean([r.get(k, float("nan")) for r in rows]) for k in keys]
            print(f"{lbl:22s} " + " ".join(f"{v:12.3f}" for v in vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
