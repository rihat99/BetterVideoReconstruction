"""Estimate the world-frame **gravity direction** with GeoCalib.

Full-body dynamics needs gravity in the same world frame as the reconstructed
motion. The camera moves and tilts, so gravity can't be read off it — we need an
external per-image estimator. GeoCalib (cvg, ECCV'24) predicts the gravity (up)
direction from a single image. This stage:

  1. samples ``--num-frames`` frames uniformly across the clip;
  2. runs GeoCalib on each → a camera-frame **up** unit vector + a confidence;
  3. transforms each into the shared world frame with the VGGT extrinsics
     (``geometry/transform.npz``; rotation only) — ``g_world = -Rᵀ · up_cam``,
     where ``R`` is that frame's cam-from-world rotation and the minus turns the
     up vector into physical gravity (down);
  4. robustly combines them: gravity is **constant** in the world, so the
     per-frame world vectors should agree. We **gate on GeoCalib's confidence**
     (its absolute value is small on horizon-free scenes, but its ranking cleanly
     picks the frames it can calibrate), then do one world-frame inlier round and
     take a confidence-weighted spherical mean. The agreement of the surviving
     frames — recovered from *different* camera poses — validates the estimate.

On top of GeoCalib, the stage tries to measure gravity **directly from the
fused scene geometry**: it fits a plane to the lowest scene points
(:func:`bvr.geometry.ground.fit_ground_plane`) and, when the fit passes three
acceptance gates, uses the floor's upward normal as world-down. The gates guard
against scenes where the ground simply is not visible (a wall close-up): the fit
must find a planar band at all (not a wall's bottom edge), keep at least
``ground_min_points`` points, and land within ``ground_max_tilt_deg`` of the
prior down direction (GeoCalib's estimate when reliable, else the fallback
camera-down). Priority for ``gravity_world``: reliable GeoCalib → accepted
ground plane → fallback down (outdoors — the only place GeoCalib validates —
the visible ground is terrain that can genuinely slope, so the learned horizon
wins; indoors the level floor replaces the camera-down fallback). ``source``
records which one won.

The output is a **unit** direction (magnitude 9.81 m/s² is applied by the
dynamics solves).

Per-clip I/O (``--source`` frames + ``--predictions out/<stem>/``)::

    reads:  --source video / folder of frames
            geometry/transform.npz   (VGGT cam-from-world extrinsics + frame_indices;
                                       falls back to vggt/camera.npz — rotation is the same)
            geometry/scene.npz       (fused scene points, for the ground-plane fit;
                                       skipped with a note when missing)
    writes: geocalib/gravity.npz
              ground_accepted    ()    bool whether the ground-plane fit won
              ground_reject      ()    str  why not ("" when accepted)
              ground_centroid/ground_normal (3,) f32, ground_extent/ground_angle_deg ()
              ground_points      ()    i32  inlier count   [all 5 only when a fit exists]
              gravity_world      (3,)   f32  unit gravity (down), world frame  [PRIMARY]
              up_world           (3,)   f32  = -gravity_world (zenith)
              gravity_world_frames (S,3) f32 per-sampled-frame world gravity
              up_cam_frames      (S,3)  f32  raw GeoCalib camera-frame up
              confidence         (S,)   f32  mean up-field confidence per frame
              roll_deg/pitch_deg (S,)   f32  per-frame gravity roll / pitch
              focal_px           (S,)   f32  GeoCalib focal (diagnostic vs VGGT)
              rows/frame_indices (S,)   i32  transform rows / original frame numbers
              selected           (S,)   bool frames used in the final estimate
              angle_deg          (S,)   f32  each frame's world-gravity angle to final
              num_frames/num_selected, inlier_spread_{median,max}_deg,
              angle_to_world_y_deg, conf_rel/inlier_deg/conf_threshold, weights

Standalone CLI::

    python bvr/stages/estimate_gravity_geocalib.py \\
        --source video.mp4 --predictions out/sport_climb/
"""
from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from bvr import determinism
from bvr.paths import REPO_ROOT
from bvr.adapters.geocalib import DEFAULT_WEIGHTS, infer_gravity, load_geocalib
from bvr.config import load_config, section
from bvr.geometry.ground import fit_ground_plane
from bvr.source import resolve_source


def _load_extrinsics(pred: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """``(extrinsics (T,4,4) cam-from-world, frame_indices (T,), source_label)``.

    Prefers ``geometry/transform.npz`` (the canonical fused cameras); falls back
    to ``vggt/camera.npz`` — only the rotation matters for gravity and it is the
    same in both (fuse rescales translation, not rotation).
    """
    for rel in ("geometry/transform.npz", "vggt/camera.npz"):
        f = pred / rel
        if f.exists():
            d = np.load(f, allow_pickle=True)
            extr = d["extrinsics"].astype(np.float64)
            fidx = (d["frame_indices"] if "frame_indices" in d.files
                    else np.arange(len(extr))).astype(int)
            return extr, fidx, rel
    raise FileNotFoundError(
        f"no camera extrinsics under {pred} — run fuse_geometry (or estimate_camera_vggt) first")


def _spherical_wmean(G: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weight-normalized mean of unit vectors, renormalized to the sphere."""
    if w.sum() <= 0:
        w = np.ones_like(w)
    m = (G * w[:, None]).sum(0)
    n = np.linalg.norm(m)
    return m / n if n > 0 else G.mean(0)


def combine_gravity(G: np.ndarray, conf: np.ndarray, *, conf_rel: float,
                    inlier_deg: float, min_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confidence-gate → inlier round → confidence-weighted spherical mean.

    Returns ``(gravity_world (3,), selected_mask (S,), angle_deg (S,))``.
    """
    n = len(G)
    floor = min(min_frames, n)
    keep = conf >= conf_rel * conf.max() if conf.max() > 0 else np.ones(n, bool)
    if keep.sum() < floor:  # too few above the relative threshold → take the best `floor`
        keep = np.zeros(n, bool)
        keep[np.argsort(conf)[::-1][:floor]] = True

    m = _spherical_wmean(G[keep], conf[keep])
    ang = np.degrees(np.arccos(np.clip(G @ m, -1.0, 1.0)))
    inl = keep & (ang < inlier_deg)
    if inl.sum() >= max(1, floor // 2):  # tighten only if the inlier set is substantial
        m = _spherical_wmean(G[inl], conf[inl])
        ang = np.degrees(np.arccos(np.clip(G @ m, -1.0, 1.0)))
    else:
        inl = keep
    return m, inl, ang


def _camera_diversity_deg(rotations: list[np.ndarray]) -> float:
    """Max pairwise geodesic angle (deg) between a set of cam-from-world rotations.

    A gravity estimate is only *validated* if the frames it averages span a range
    of camera orientations — otherwise the per-frame world vectors agree trivially
    (the same pose mapped through the same rotation). Near-static cameras give a
    small value here even when the spread looks tight.
    """
    n = len(rotations)
    best = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            R = rotations[i] @ rotations[j].T
            best = max(best, float(np.degrees(np.arccos(
                np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))))
    return best


def _parse_dir(s: str) -> np.ndarray:
    """Parse a ``'x,y,z'`` string into a unit vector (the fallback gravity)."""
    v = np.array([float(x) for x in s.split(",")], np.float64)
    n = float(np.linalg.norm(v))
    if v.shape != (3,) or n == 0.0:
        raise ValueError(f"--fallback-gravity must be 'x,y,z' nonzero, got {s!r}")
    return v / n


def estimate_gravity(args: argparse.Namespace) -> None:
    pred = args.predictions
    out_npz = pred / "geocalib" / "gravity.npz"
    if args.skip_existing and out_npz.exists():
        print(f"  gravity already done: {out_npz} — skipping")
        return

    fallback = _parse_dir(args.fallback_gravity)  # validate before the (slow) inference
    extr, fidx, extr_src = _load_extrinsics(pred)
    T = len(extr)
    rows = np.unique(np.linspace(0, T - 1, args.num_frames).round().astype(int))

    model = load_geocalib(device=args.device, weights=args.weights)
    try:
        with tempfile.TemporaryDirectory(prefix="geocalib_frames_") as tmp:
            frame_paths, _ = resolve_source(Path(args.source), Path(tmp))
            recs = []
            for row in rows:
                fr = int(fidx[row])
                if fr >= len(frame_paths):
                    continue
                bgr = cv2.imread(str(frame_paths[fr]))
                if bgr is None:
                    print(f"  WARN cannot read frame {fr}; skipping")
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                g = infer_gravity(model, rgb, device=args.device)
                gw = -(extr[row][:3, :3].T @ g["up_cam"])  # world-frame gravity (down)
                if not np.all(np.isfinite(gw)):
                    print(f"  WARN non-finite gravity at frame {fr}; skipping")
                    continue
                recs.append((int(row), fr, gw, g))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not recs:
        raise RuntimeError("GeoCalib produced no usable frames")

    rows_used = np.array([r[0] for r in recs], np.int32)
    frames_used = np.array([r[1] for r in recs], np.int32)
    G = np.stack([r[2] for r in recs]).astype(np.float64)
    conf = np.array([r[3]["confidence"] for r in recs], np.float64)
    up_cam = np.stack([r[3]["up_cam"] for r in recs]).astype(np.float32)
    roll_deg = np.degrees([r[3]["roll"] for r in recs]).astype(np.float32)
    pitch_deg = np.degrees([r[3]["pitch"] for r in recs]).astype(np.float32)
    focal_px = np.array([r[3]["focal_px"] for r in recs], np.float32)

    grav, selected, angle_deg = combine_gravity(
        G, conf, conf_rel=args.conf_rel, inlier_deg=args.inlier_deg, min_frames=args.min_frames)

    sel_ang = angle_deg[selected]
    spread_med = float(np.median(sel_ang)) if sel_ang.size else float("nan")
    spread_max = float(sel_ang.max()) if sel_ang.size else float("nan")
    angle_to_y = float(np.degrees(np.arccos(np.clip(grav @ np.array([0.0, 1.0, 0.0]), -1, 1))))

    # Reliability: gravity is only trustworthy if GeoCalib was actually confident
    # (absolute confidence floor — horizon-free scenes collapse to ~0) AND the
    # selected frames both agree (small spread) AND span a range of camera
    # orientations (else the agreement is trivial — same pose, same rotation).
    max_conf = float(conf.max())
    cam_div = _camera_diversity_deg([extr[r][:3, :3] for r in rows_used[selected]])
    fail = []
    if max_conf < args.conf_floor:
        fail.append(f"max-confidence {max_conf:.4f} < {args.conf_floor:g}")
    if not (spread_med <= args.max_spread_deg):
        fail.append(f"spread_median {spread_med:.1f}° > {args.max_spread_deg:g}°")
    if cam_div < args.min_diversity_deg:
        fail.append(f"camera-rotation-diversity {cam_div:.1f}° < {args.min_diversity_deg:g}°")
    reliable = not fail

    # Ground plane: when the floor shows up in the fused scene, its upward normal
    # is direct geometric evidence of world-down — better than any camera-based
    # estimate. The prior (GeoCalib when reliable, else the fallback camera-down)
    # only seeds which points count as "lowest" and anchors the tilt gate; the
    # gates reject scenes where the ground is not actually visible.
    grav_gc = grav.copy()
    prior = grav_gc if reliable else fallback
    ground, fit, ground_reject = None, None, "disabled"
    if args.ground:
        scene_f = pred / "geometry" / "scene.npz"
        if not scene_f.exists():
            ground_reject = "no geometry/scene.npz (run fuse_geometry first)"
        else:
            fit = fit_ground_plane(np.load(scene_f)["points"], prior,
                                   max_tilt_deg=args.ground_max_tilt_deg)
            if fit is None:
                ground_reject = "no planar floor in the lowest band"
            else:
                centroid, up_n, extent, n_in = fit
                tilt = float(np.degrees(np.arccos(np.clip(up_n @ (-prior), -1.0, 1.0))))
                if n_in < args.ground_min_points:
                    ground_reject = f"{n_in} inlier points < {args.ground_min_points}"
                elif tilt > args.ground_max_tilt_deg:
                    ground_reject = (f"normal {tilt:.1f}° from expected down "
                                     f"(> {args.ground_max_tilt_deg:g}°)")
                else:
                    ground, ground_reject = {"centroid": centroid, "normal": up_n,
                                             "extent": extent, "points": n_in,
                                             "angle_deg": tilt}, ""

    # Priority: reliable GeoCalib → accepted ground plane → fallback world-down
    # (+y — VGGT anchors its world to camera 0, whose opening shot is ~upright).
    # GeoCalib outranks the ground fit because it only survives its gates on
    # outdoor scenes — exactly where the visible ground is terrain that can
    # genuinely slope (sport_climb's crag base sits 25° off its validated
    # gravity). Indoors GeoCalib essentially never validates and the level gym
    # floor is the best evidence, so the ground plane replaces the camera-down
    # fallback there. `source` records which one `gravity_world` carries.
    if reliable:
        grav, source = grav_gc, "geocalib"
    elif ground is not None:
        grav, source = -ground["normal"], "ground"
    else:
        grav, source = fallback.copy(), "fallback_down"

    ground_fields = {"ground_accepted": np.bool_(ground is not None),
                     "ground_reject": np.str_(ground_reject)}
    if fit is not None:
        ground_fields.update(
            ground_centroid=centroid.astype(np.float32),
            ground_normal=up_n.astype(np.float32),
            ground_extent=np.float32(extent),
            ground_points=np.int32(n_in),
            ground_angle_deg=np.float32(tilt))

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        **ground_fields,
        gravity_world=grav.astype(np.float32),
        up_world=(-grav).astype(np.float32),
        source=source,
        gravity_world_geocalib=grav_gc.astype(np.float32),
        gravity_world_frames=G.astype(np.float32),
        up_cam_frames=up_cam,
        confidence=conf.astype(np.float32),
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        focal_px=focal_px,
        rows=rows_used,
        frame_indices=frames_used,
        selected=selected,
        angle_deg=angle_deg.astype(np.float32),
        num_frames=np.int32(len(recs)),
        num_selected=np.int32(int(selected.sum())),
        inlier_spread_median_deg=np.float32(spread_med),
        inlier_spread_max_deg=np.float32(spread_max),
        angle_to_world_y_deg=np.float32(angle_to_y),
        max_confidence=np.float32(max_conf),
        camera_diversity_deg=np.float32(cam_div),
        reliable=np.bool_(reliable),
        conf_rel=np.float32(args.conf_rel),
        inlier_deg=np.float32(args.inlier_deg),
        conf_threshold=np.float32(args.conf_rel * conf.max()),
        conf_floor=np.float32(args.conf_floor),
        min_diversity_deg=np.float32(args.min_diversity_deg),
        fallback_gravity=fallback.astype(np.float32),
        weights=args.weights,
        extrinsics_source=extr_src,
    )
    if source == "ground":
        print(f"  gravity_world = [{grav[0]:+.4f} {grav[1]:+.4f} {grav[2]:+.4f}]  [GROUND PLANE]  "
              f"({ground['points']} floor pts, {ground['angle_deg']:.1f}° from prior down) "
              f"-> {out_npz}")
    elif source == "geocalib":
        print(f"  gravity_world = [{grav[0]:+.4f} {grav[1]:+.4f} {grav[2]:+.4f}]  [GeoCalib, RELIABLE]  "
              f"({int(selected.sum())}/{len(recs)} frames, spread med={spread_med:.2f}°, "
              f"cam-rot-div={cam_div:.1f}°, max-conf={max_conf:.4f}, {angle_to_y:.1f}° from world +y) "
              f"-> {out_npz}")
        if ground is not None:
            print(f"  ground plane: fitted ({ground['points']} pts, {ground['angle_deg']:.1f}° "
                  f"from GeoCalib down) — kept for display; reliable GeoCalib outranks it")
    else:
        print(f"  gravity_world = [{grav[0]:+.4f} {grav[1]:+.4f} {grav[2]:+.4f}]  [FALLBACK world-down]  "
              f"-> {out_npz}")
        print(f"  GeoCalib UNRELIABLE here ({'; '.join(fail)}) — it wasn't trained on climbing "
              f"walls; using the standard down direction. GeoCalib's raw guess was "
              f"[{grav_gc[0]:+.3f} {grav_gc[1]:+.3f} {grav_gc[2]:+.3f}] ({angle_to_y:.0f}° off, "
              f"kept as gravity_world_geocalib).")
    if ground is None and args.ground:
        print(f"  ground plane: rejected — {ground_reject}")


def main(argv: list[str] | None = None) -> int:
    determinism.apply()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True,
                   help="video / folder of frames (same source used for the rest of the clip)")
    p.add_argument("--predictions", required=True, type=Path, help="clip dir, e.g. out/<stem>/")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.toml",
                   help="pipeline TOML (default: configs/pipeline.toml)")
    cli = p.parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(cli.config)
    args = section(cfg, "gravity", {
        "num_frames": 48,
        "conf_rel": 0.5,
        "inlier_deg": 8.0,
        "min_frames": 5,
        "conf_floor": 0.01,
        "min_diversity_deg": 8.0,
        "max_spread_deg": 5.0,
        "fallback_gravity": "0,1,0",
        "ground": True,
        "ground_min_points": 500,
        "ground_max_tilt_deg": 30.0,
        "weights": DEFAULT_WEIGHTS,
        "skip_existing": False,
    })
    args.device = section(cfg, "runtime", {"device": "cuda"}).device
    args.source, args.predictions, args.config = cli.source, cli.predictions, cli.config
    estimate_gravity(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
