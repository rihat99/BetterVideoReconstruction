#!/usr/bin/env python3
"""Render the SMPL-X template mesh colored by candidate per-vertex weight fields.

Builds a menu of per-vertex weight candidates from
:mod:`tools.smplx_robot.vertex_weights` (see that module for the algorithms) and
renders each colored onto the SMPL-X-mid template mesh, as a montage of rows
(candidates) × columns (views) with one shared colorbar. A per-anatomical-group
mean-weight table is printed and saved so the distribution can be read as numbers,
not only colors. Nothing here touches optimization — this is the inspection tool
for choosing which field (and which direction) to wire into the scale ruler /
chamfer later.

    uv run python scripts/render_vertex_weights.py --out out/vertex_weights

The candidate list is the editable block in :func:`build_candidates`; the profiles
and generators it calls live in :mod:`tools.smplx_robot.vertex_weights`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from tools.smplx_robot import vertex_weights as vw  # noqa: E402

_DEFAULT_MODEL = _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"

# (name, elevation°, azimuth°) — calibrated to the SMPL-X template (native +y up,
# faces +z); the mesh is plotted with body-height on the vertical axis.
_VIEWS = (("front", 12, 90), ("side", 12, 0), ("3/4", 20, 45))

# Groups in extremity→core order for the printed table.
_GROUP_ORDER = ("hand", "foot", "forearm", "shin", "upper_arm", "thigh", "chest", "torso", "head")


# Geodesic weight range: the core lands near ``_GEO_LO`` (÷10), an extremity near
# ``_GEO_HI`` (×10), the neutral ``1.0`` at the log-midpoint.
_GEO_LO, _GEO_HI = 0.1, 10.0


def build_candidates(verts, faces, lbs, names, seg, which: str = "geodesic") -> dict[str, np.ndarray]:
    """The weight fields to compare.

    ``which='geodesic'`` sweeps the geodesic-to-endeffector falloff (log-mapped to
    ``[0.1, 10]`` so a wide multiplicative range separates tips from core, with **no**
    mean-normalization so the core reaches the floor). ``which='all'`` is the full
    method menu, mean-normalized. Returns ``{name: (V,) weights}``.
    """
    geo = vw.geodesic_endeffector_field(verts, faces, lbs, names)

    if which == "geodesic":
        def field(kind: str, radius: float, gamma: float = 2.0) -> np.ndarray:
            x = vw.falloff(geo, kind=kind, radius=radius, gamma=gamma)
            return vw.scale_to_range(x, lo=_GEO_LO, hi=_GEO_HI, mode="log")
        # A larger radius spreads the high→low transition over more of the limb
        # (smoother); compact-support falloffs (linear/power/smoothstep) drop the
        # core to the floor, the gaussian keeps a soft tail on it.
        return {
            "linear · r1.0": field("linear", 1.0),
            "quadratic · r1.0": field("power", 1.0),
            "quadratic · r1.3": field("power", 1.3),
            "smoothstep · r1.1": field("smoothstep", 1.1),
            "gaussian · r0.7": field("gaussian", 0.7),
        }

    up, down = vw.PROFILES["extremity_up"], vw.PROFILES["extremity_down"]
    raw = {
        # semantic, extremity-up profile — smooth LBS blend / hard argmax / mesh parts
        "lbs_soft · up": vw.group_weights(lbs, names, up, soft=True),
        "lbs_hard · up": vw.group_weights(lbs, names, up, soft=False),
        "segmentation · up": vw.segmentation_weights(seg, names, lbs, up),
        # the inverse direction (trust the torso) + the uniform reference
        "lbs_soft · down": vw.group_weights(lbs, names, down, soft=True),
        "uniform": vw.group_weights(lbs, names, vw.PROFILES["uniform"], soft=True),
    }
    return {name: vw.normalize(w) for name, w in raw.items()}


def group_table(cands: dict[str, np.ndarray], vlab: np.ndarray) -> str:
    """A text table of each candidate's mean weight per anatomical group."""
    head = f"{'candidate':22s} " + " ".join(f"{g[:5]:>6s}" for g in _GROUP_ORDER) + "    min  mean   max"
    lines = [head, "-" * len(head)]
    for name, w in cands.items():
        per = " ".join(f"{w[vlab == g].mean():6.2f}" for g in _GROUP_ORDER)
        lines.append(f"{name:22s} {per}   {w.min():5.2f} {w.mean():5.2f} {w.max():5.2f}")
    return "\n".join(lines)


def render_montage(verts, faces, cands: dict[str, np.ndarray], out_png: Path, *,
                   dpi: int, vmin: float, vmax: float, log: bool, cmap: str, title: str) -> None:
    """Trisurf each candidate × view into one montage PNG with a shared colorbar.

    ``log`` uses a :class:`~matplotlib.colors.LogNorm` (right for a wide multiplicative
    range like ``[0.1, 10]``, where a diverging ``cmap`` then centers the neutral ``1.0``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, Normalize

    x, y, z = verts[:, 0], verts[:, 2], verts[:, 1]   # body-height (native y) → vertical
    box = (np.ptp(x), np.ptp(y), np.ptp(z))
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)

    n_rows, n_cols = len(cands), len(_VIEWS)
    fig = plt.figure(figsize=(3.4 * n_cols + 1.4, 3.3 * n_rows))
    axes = []
    for r, (name, w) in enumerate(cands.items()):
        face_w = w[faces].mean(axis=1)
        for c, (vname, elev, azim) in enumerate(_VIEWS):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            axes.append(ax)
            tri = ax.plot_trisurf(x, y, z, triangles=faces, linewidth=0,
                                  antialiased=False, shade=False)
            tri.set_array(face_w)
            tri.set_cmap(cmap)
            tri.set_norm(norm)
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect(box)
            ax.set_axis_off()
            if r == 0:
                ax.set_title(vname, fontsize=12)
            if c == 0:
                ax.text2D(-0.08, 0.5, name, rotation=90, va="center", ha="center",
                          fontsize=12, weight="bold", transform=ax.transAxes)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.45, pad=0.02, aspect=40)
    cbar.set_label("per-vertex weight", fontsize=11)
    if log:
        ticks = [t for t in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0) if vmin <= t <= vmax]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:g}" for t in ticks])
    fig.suptitle(title, fontsize=15, y=0.998)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("out/vertex_weights"),
                   help="output dir for the montage PNG + weights npz + table (default: out/vertex_weights).")
    p.add_argument("--model", type=Path, default=_DEFAULT_MODEL,
                   help=f"SMPL-X-mid .npz (default: {_DEFAULT_MODEL}).")
    p.add_argument("--set", dest="cand_set", choices=("geodesic", "all"), default="geodesic",
                   help="candidate set: 'geodesic' falloff sweep (default) or the full 'all' menu.")
    p.add_argument("--dpi", type=int, default=160, help="montage PNG dpi (default: 160).")
    p.add_argument("--no-npz", action="store_true", help="skip saving the candidate weights npz.")
    args = p.parse_args(argv)

    from better_human.smpl import SMPLXMid
    mid = SMPLXMid(model_path=str(args.model)).eval()
    verts = mid.vertices_template.detach().cpu().numpy()
    faces = mid.faces.detach().cpu().numpy()
    names = list(mid.joint_names)
    seg = mid.vertices_segmentation
    vlab = vw.lbs_group_labels(mid.lbs_weights, names)

    cands = build_candidates(verts, faces, mid.lbs_weights, names, seg, which=args.cand_set)

    table = group_table(cands, vlab)
    print(table)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "group_means.txt").write_text(table + "\n")

    if not args.no_npz:
        np.savez(args.out / "weights.npz", verts=verts, faces=faces, vlab=vlab.astype(str),
                 **{name.replace(" · ", "__").replace(" ", "_"): w for name, w in cands.items()})

    if args.cand_set == "geodesic":
        vmin, vmax, log, cmap = _GEO_LO, _GEO_HI, True, "RdBu_r"
        title = "SMPL-X geodesic weight — falloff sweep  (log scale · core ÷10 … tip ×10)"
    else:
        allw = np.concatenate(list(cands.values()))
        vmin, vmax, log, cmap = float(allw.min()), float(allw.max()), False, "turbo"
        title = "SMPL-X per-vertex weight candidates"

    out_png = args.out / "montage.png"
    render_montage(verts, faces, cands, out_png, dpi=args.dpi,
                   vmin=vmin, vmax=vmax, log=log, cmap=cmap, title=title)
    print(f"\nwrote {out_png}\nwrote {args.out / 'group_means.txt'}"
          + ("" if args.no_npz else f"\nwrote {args.out / 'weights.npz'}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
