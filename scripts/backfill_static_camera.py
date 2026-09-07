"""Stamp the static-camera flag onto results that were fused before the flag existed.

Re-fusing everything to pick up one boolean would take days, and it is not
needed: the flag only wants the cameras and something to project through them,
and both are already on disk. For a fused clip that is ``geometry/transform.npz``
with the scene cloud beside it. For a scene the fuse stage has not reached yet
(the AthleticsVideos corpus) it is ``vggt/camera.npz`` with one frame's depth map
backprojected — the same measurement, a coarser set of probe points.

The flag is written back into the camera file it was measured from, beside
``metric`` — no new npz. ``camera_motion`` goes in with it, so the threshold can
be moved later without measuring everything again. For a corpus the boolean also
lands in the scene DB as ``scenes.static_camera``.

Targets are directories: either one clip tree or a directory of them, following
the ``out/<stem>/`` layout (deeper nestings are left alone — the diagnostic
sub-trees under ``out/`` hold their own copies of the same clips). A corpus run
walks the feature tree of a data root instead::

    python scripts/backfill_static_camera.py out/ out_2/
    python scripts/backfill_static_camera.py --corpus ../data/ClimbingVideos
    python scripts/backfill_static_camera.py --corpus ../data/AthleticsVideos --dry-run
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from tools.config import load_toml, section
from tools.geometry.camera_motion import image_motion, is_static
from tools.npz_io import atomic_savez


def view0(cam: dict) -> np.ndarray:
    """Rows of the camera file belonging to view 0 — the track every stage reads."""
    ids = cam.get("view_ids")
    return np.arange(len(cam["extrinsics"])) if ids is None else np.flatnonzero(ids == 0)


def cameras(cam: dict, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """View-0 extrinsics + pixel intrinsics, and the pixel grid they are written on.

    ``geometry/transform.npz`` keeps its intrinsics at the original frame
    resolution and is already view 0 only; ``vggt/camera.npz`` keeps VGGT's own
    reduced grid and every view. Either way the answer is a fraction of the
    width, so the two grids give the same number.
    """
    if "intrinsics_px_orig" in cam:
        return (cam["extrinsics"], cam["intrinsics_px_orig"],
                int(cam["image_width"]), int(cam["image_height"]))
    return (cam["extrinsics"][rows], cam["intrinsics_px"][rows],
            int(cam["vggt_width"]), int(cam["vggt_height"]))


def depth_probe(vggt_dir: Path, cam: dict, rows: np.ndarray) -> np.ndarray:
    """One view-0 depth map from the middle of the clip, backprojected to world points."""
    row = int(rows[len(rows) // 2])
    names = cam.get("view_names")
    view = "" if names is None else str(names[int(cam["view_ids"][row])])
    path = vggt_dir / "depth" / view / f"frame_{int(cam['frame_indices'][row]):06d}.npz"
    depth = np.load(path)["depth"].astype(np.float64)
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 0))
    keep = slice(None, None, max(1, len(ys) // 8000))
    ys, xs, z = ys[keep], xs[keep], depth[ys[keep], xs[keep]]
    K, extr = cam["intrinsics_px"][row].astype(np.float64), cam["extrinsics"][row].astype(np.float64)
    local = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], -1)
    return (local - extr[:3, 3]) @ extr[:3, :3]          # Rᵀ (X − t), as in tools.geometry.backproject


def probe_points(npz: Path, cam: dict, rows: np.ndarray) -> np.ndarray:
    """The world points to watch: the fused scene cloud when there is one, else depth."""
    scene = npz.parent / "scene.npz"
    if scene.exists():
        return np.load(scene)["points"]
    return depth_probe(npz.parent, cam, rows)


def stamp(npz: Path, threshold: float, dry_run: bool) -> tuple[float, bool]:
    """Measure this clip's camera and write the flag back into its own camera file."""
    cam = dict(np.load(npz, allow_pickle=True))
    rows = view0(cam)
    extr, K, width, height = cameras(cam, rows)
    motion = image_motion(extr, K, width, height, probe_points(npz, cam, rows))
    static = is_static(motion, threshold)
    if not dry_run:
        cam.update(static_camera=np.bool_(static), camera_motion=np.float32(motion))
        atomic_savez(npz, **cam)
    return motion, static


def corpus_targets(root: Path) -> list[tuple[str, Path]]:
    """``(scene_id, camera npz)`` for every scene of a data root that has a camera.

    The fused cameras win when the geometry stage has run, because their scene
    cloud is a better probe than a single depth map; scenes it has not reached
    fall back to the VGGT cameras.
    """
    targets = {p.parent.name: p for p in sorted((root / "features" / "vggt").rglob("camera.npz"))}
    targets |= {p.parent.name: p for p in sorted((root / "features" / "geometry").rglob("transform.npz"))}
    return sorted(targets.items())


def sync_db(db: Path, flags: dict[str, bool]) -> int:
    """Write the boolean into ``scenes.static_camera``, adding the column if it is new."""
    conn = sqlite3.connect(db)
    try:
        if "static_camera" not in {r[1] for r in conn.execute("PRAGMA table_info(scenes)")}:
            conn.execute("ALTER TABLE scenes ADD COLUMN static_camera INTEGER")
        conn.executemany("UPDATE scenes SET static_camera = ? WHERE scene_id = ?",
                         [(int(v), k) for k, v in flags.items()])
        conn.commit()
        return int(conn.execute("SELECT count(*) FROM scenes WHERE static_camera IS NOT NULL")
                   .fetchone()[0])
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trees", nargs="*", type=Path,
                   help="a clip tree, or a directory of them, e.g. out/ out_2/")
    p.add_argument("--corpus", type=Path, help="scene-DB data root, e.g. ../data/ClimbingVideos")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "pipeline.toml",
                   help="pipeline TOML the threshold comes from (default: configs/pipeline.toml)")
    p.add_argument("--dry-run", action="store_true", help="measure and report, write nothing")
    args = p.parse_args(argv)
    threshold = float(section(load_toml(args.config), "fuse", {"static_motion_max": 0.02})
                      .static_motion_max)

    targets: list[tuple[str, Path]] = []
    for tree in args.trees:
        found = ([tree / "geometry" / "transform.npz"] if (tree / "geometry").is_dir()
                 else sorted(tree.glob("*/geometry/transform.npz")))
        targets += [(npz.parent.parent.name, npz) for npz in found]
    if args.corpus:
        targets += corpus_targets(args.corpus)

    flags: dict[str, bool] = {}
    failed = 0
    for name, npz in targets:
        try:
            motion, static = stamp(npz, threshold, args.dry_run)
        except Exception as exc:                        # a corrupt clip must not kill the pass
            print(f"  {name:26s} FAILED {type(exc).__name__}: {exc}", flush=True)
            failed += 1
            continue
        flags[name] = static
        print(f"  {name:26s} {'STATIC' if static else 'moving':6s} {motion * 100:7.2f}% of width"
              f"{'  (from depth)' if not (npz.parent / 'scene.npz').exists() else ''}", flush=True)

    n_static = sum(flags.values())
    print(f"\n{len(flags)} clip(s): {n_static} static, {len(flags) - n_static} moving, "
          f"{failed} unreadable  (threshold {threshold * 100:g}% of width)")
    if args.corpus and not args.dry_run:
        total = sync_db(args.corpus / "scenes" / "scenes.db", flags)
        print(f"scenes.db: {len(flags)} row(s) written, {total} now carry static_camera")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
