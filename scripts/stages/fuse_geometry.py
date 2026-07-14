"""Fuse VGGT geometry into metric scene + human point clouds.

Runs after ``estimate_scale``, which anchors VGGT's arbitrary per-clip scale to
the SAM-3D-Body human ruler (``geometry/scale.npz``). VGGT-Omega gives
cross-frame-consistent cameras and depth but **up to one arbitrary scale** and at
a **reduced resolution**. This stage:

  1. reads the metric scale ``s`` from ``geometry/scale.npz`` (the human-anchored
     factor — depth and camera translations all multiply by it);
  2. denormalizes VGGT's intrinsics to the original frame resolution (FOV is
     VGGT's and is preserved — scaling only touches depth + camera translation);
  3. backprojects every cleaned VGGT depth map to metric world points, split into
     a static **scene** cloud and per-frame **human** points via the SAM3 masks.
     The scene subtracts a *dilated* mask of **every** tracked person, so no one —
     climber or bystander — leaks into it; the **human** cloud keeps only the
     reconstructed climbers (``sam3d/smplx_params.npz`` ``object_ids``), tagged by
     ``obj_id`` (edge filter only — no confidence gate);
  4. merges near-duplicate points by voxel hashing — the scene globally (it is
     static); the human is kept per-frame, unmerged (the person moves);
  5. drops small disconnected voxel clusters from the scene — floating depth
     fragments and off-surface blobs (e.g. an untracked bystander VGGT placed off
     the wall).

Per-clip I/O (``--source`` frames + ``--predictions out/<stem>/``)::

    reads:  --source video / folder of frames          (RGB, for per-point colour)
            vggt/camera.npz, vggt/depth/frame_*.npz     (needs [vggt] save_depth)
            geometry/scale.npz                          (metric scale, from estimate_scale)
            sam3/<obj>/frame_*.png
    writes: geometry/transform.npz   scale + metric extrinsics + original-res K
            geometry/scene.npz        merged metric background points (+ colour)
            geometry/human.npz        per-frame metric human points (frame_id, obj_id, colour)

Standalone CLI::

    python scripts/stages/fuse_geometry.py --source video.mp4 --predictions out/sport_climb/
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch

from tools.geometry.backproject import backproject
from tools.geometry.clean import clean_depth
from tools.geometry.intrinsics import denormalize_K
from tools.geometry.voxel import group_reduce, large_component_mask, voxel_downsample
from tools.mask_io import read_binary_mask
from tools.source import resolve_source


def _discover_objs(sam3_dir: Path, requested: list[int] | None) -> list[int]:
    found = sorted(int(d.name) for d in sam3_dir.iterdir() if d.is_dir() and d.name.isdigit())
    return [o for o in found if o in requested] if requested else found


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean mask by ``radius`` px (elliptical); ``radius<=0`` is a no-op."""
    if radius <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def _human_masks(sam3_dir: Path, idx: int, objs: list[int], hv: int, wv: int):
    """Union of person masks at VGGT resolution: ``(is_human bool, obj_id int16)``."""
    is_human = np.zeros((hv, wv), bool)
    obj_id = np.full((hv, wv), -1, np.int16)
    for o in objs:
        m = read_binary_mask(sam3_dir / f"{o:02d}" / f"frame_{idx:06d}.png")
        if m is None:
            continue
        m = cv2.resize(m.astype(np.uint8), (wv, hv), interpolation=cv2.INTER_NEAREST) > 0
        obj_id[m & ~is_human] = o
        is_human |= m
    return is_human, obj_id


def _save_cloud(path: Path, points: torch.Tensor, **arrays) -> None:
    np.savez(path, points=points.detach().cpu().numpy().astype(np.float32),
             **{k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in arrays.items()})


def _record_anchor_veto(anchor: dict, touch_m: float, burial_m: float) -> None:
    """Make anchor diagnostics describe the baseline world that actually ships."""
    anchor["rescale_r"] = 1.0
    anchor["depth_refuted"][:] = False
    anchor["anchor_pred_gap_q"][:] = np.nan
    anchor["anchor_true_touch"] = float(touch_m)
    anchor["anchor_true_burial"] = float(burial_m)


def _build_geometry(args, pred, cam, frame_indices, extr, K_px, objs, climber_ids,
                    scale, *, base_scale, fixed_human_ids):
    """Run the standard backproject/merge/CC build at ``scale``.

    ``fixed_human_ids`` is empty for the ordinary build. On a fired second pass,
    those people's points retain their ``base_scale`` camera coordinates while
    the camera translations and every other point follow ``scale``.
    """
    hv, wv = int(cam["vggt_height"]), int(cam["vggt_width"])
    dev = args.device
    scene_pts, scene_conf, scene_col = [], [], []
    human_pts, human_conf, human_obj, human_fid, human_col = [], [], [], [], []

    with tempfile.TemporaryDirectory(prefix="fuse_frames_") as tmp:
        frame_paths, src_fps = resolve_source(Path(args.source), Path(tmp))
        for k, idx in enumerate(frame_indices):
            idx = int(idx)
            vd = np.load(pred / "vggt" / "depth" / f"frame_{idx:06d}.npz")
            depth, conf = vd["depth"].astype(np.float32), vd["depth_conf"].astype(np.float32)
            # Scene gets the full clean (confidence-gated); the human keeps every
            # in-mask point — edge filter only, no confidence drop.
            valid_scene = clean_depth(depth, conf, conf_drop_q=args.conf_drop_q, edge_rtol=args.edge_rtol)
            valid_human = clean_depth(depth, edge_rtol=args.edge_rtol)

            rgb = cv2.cvtColor(cv2.imread(str(frame_paths[idx])), cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (wv, hv), interpolation=cv2.INTER_AREA)

            is_human, obj_id = _human_masks(pred / "sam3", idx, objs, hv, wv)
            # Imperfect masks leak human-edge pixels into the background, so subtract
            # a *dilated* band of every tracked person from the scene; collect the
            # human from the tight mask, restricted to the reconstructed climbers.
            scene_mask = valid_scene & ~_dilate(is_human, args.human_dilate)
            human_mask = valid_human & is_human & np.isin(obj_id, climber_ids)

            depth_t, K_t, E_t = (torch.from_numpy(depth).to(dev),
                                 torch.from_numpy(K_px[k]).to(dev), torch.from_numpy(extr[k]).to(dev))
            conf_t = torch.from_numpy(conf).to(dev)
            col_t = torch.from_numpy(np.ascontiguousarray(rgb)).to(dev)

            ss = torch.from_numpy(scene_mask).to(dev)
            scene_pts.append(backproject(depth_t, K_t, E_t, mask=ss))
            scene_conf.append(conf_t[ss])
            scene_col.append(col_t[ss])
            if human_mask.any():
                sh = torch.from_numpy(human_mask).to(dev)
                human_pts.append(backproject(depth_t, K_t, E_t, mask=sh))
                human_conf.append(conf_t[sh])
                human_obj.append(torch.from_numpy(obj_id).to(dev)[sh].to(torch.int32))
                human_fid.append(torch.full((int(sh.sum()),), k, dtype=torch.int32, device=dev))
                human_col.append(col_t[sh])

    centroids, inverse, counts, coords = voxel_downsample(torch.cat(scene_pts) * scale,
                                                          args.voxel_scene)
    scene_confidence = group_reduce(torch.cat(scene_conf), inverse, counts.numel(), "amax")
    scene_color = torch.zeros((counts.numel(), 3), device=dev)
    scene_color.index_add_(0, inverse, torch.cat(scene_col).float())
    scene_color = (scene_color / counts.unsqueeze(1)).round().to(torch.uint8)

    # Drop small disconnected clusters — floating depth fragments and off-surface
    # blobs (e.g. an untracked bystander VGGT placed off the wall).
    keep = large_component_mask(coords, args.min_scene_voxels)
    n_dropped = int((~keep).sum())
    centroids, counts = centroids[keep], counts[keep]
    scene_confidence, scene_color = scene_confidence[keep], scene_color[keep]

    empty = torch.zeros((0, 3), device=dev)
    if human_pts:
        raw_human = torch.cat(human_pts)
        all_obj = torch.cat(human_obj)
        all_fid = torch.cat(human_fid)
        human_points = raw_human * scale
        if fixed_human_ids:
            extr_t = torch.as_tensor(extr, dtype=raw_human.dtype, device=dev)
            cam_offset = torch.einsum("tji,tj->ti", extr_t[:, :3, :3], extr_t[:, :3, 3])
            cam_offset *= base_scale - scale
            for oid in fixed_human_ids:
                fixed = all_obj == int(oid)
                human_points[fixed] = raw_human[fixed] * base_scale + cam_offset[all_fid[fixed].long()]
    else:
        human_points, all_obj, all_fid = empty, empty[:, 0].to(torch.int32), empty[:, 0].to(torch.int32)
    return dict(scene_points=centroids, scene_count=counts, scene_confidence=scene_confidence,
                scene_color=scene_color, human_points=human_points,
                human_color=torch.cat(human_col) if human_col else empty.to(torch.uint8),
                human_confidence=torch.cat(human_conf) if human_conf else empty[:, 0],
                human_frame_id=all_fid, human_obj_id=all_obj, n_dropped=n_dropped,
                n_human=sum(int(t.shape[0]) for t in human_pts), src_fps=src_fps)


def fuse(args: argparse.Namespace) -> None:
    pred = Path(args.predictions)
    if not (pred / "vggt" / "depth").is_dir():
        raise SystemExit(f"[fuse] missing vggt/depth/ under {pred} — run the vggt stage with save_depth=true")
    scale_path = pred / "geometry" / "scale.npz"
    if not scale_path.exists():
        raise SystemExit(f"[fuse] missing {scale_path} — run the scale stage (estimate_scale) first")
    out_dir = pred / "geometry"
    if args.skip_existing and (out_dir / "transform.npz").exists():
        print(f"  skip (exists): {out_dir / 'transform.npz'}")
        return
    s = float(np.load(scale_path)["scale"])

    cam = np.load(pred / "vggt" / "camera.npz")
    frame_indices = cam["frame_indices"]
    extr, K_px, K_norm = cam["extrinsics"], cam["intrinsics_px"], cam["intrinsics"]
    hv, wv = int(cam["vggt_height"]), int(cam["vggt_width"])
    ho, wo = int(cam["image_height"]), int(cam["image_width"])
    requested = [int(x) for x in args.person_obj_ids.split(",")] if args.person_obj_ids else None
    objs = _discover_objs(pred / "sam3", requested)
    # Every tracked person is carved out of the scene, but only the reconstructed
    # climbers (smplx_params object_ids) go into the human cloud — a tracked
    # bystander is thus excluded from both. Falls back to all objs if smplx params
    # are absent (standalone fuse without the smplx stage).
    smplx_path = pred / "sam3d" / "smplx_params.npz"
    climber_ids = objs
    if smplx_path.exists():
        ids = np.load(smplx_path, allow_pickle=True)["object_ids"].ravel()
        climber_ids = [int(o) for o in ids if int(o) in objs]
    built = _build_geometry(args, pred, cam, frame_indices, extr, K_px, objs, climber_ids,
                            s, base_scale=s, fixed_human_ids=set())
    s_eff = s
    anchor = None
    if args.body_anchor == "on":
        if not smplx_path.exists():
            print("  [body-anchor] note: no sam3d/smplx_params.npz; gate skipped")
        elif int(built["scene_points"].shape[0]) == 0:
            print("  [body-anchor] note: empty scene cloud; gate skipped")
        else:
            from tools.human_optim.depth_anchor import evaluate_body_anchor

            extr_base = extr.copy()
            extr_base[:, :3, 3] *= s
            anchor = evaluate_body_anchor(
                smplx_path, built["scene_points"], built["scene_confidence"], extr_base,
                frame_indices, climber_ids, device=args.device, float_gate_m=args.float_gate_m)
            for person in anchor["people"]:
                verdict = "FIRED" if person["fired"] else ("healthy" if person["eligible"] else "SKIP")
                print(f"  [body-anchor] person obj_id={person['object_id']}: "
                      f"gap_med={person['gap_m']:+.3f} m burial_med={person['burial_m']:+.3f} m "
                      f"support={person['support']:.1%} ({person['n_finite']}/{person['n_valid']}) "
                      f"-> {verdict}")
            if not anchor["consistent"]:
                print(f"  [body-anchor] WARNING: rescale skipped — {anchor['reason']}")
            elif bool(anchor["depth_refuted"].any()):
                s_eff = s * anchor["rescale_r"]
                if anchor["burial_conflict"]:
                    print("  [body-anchor] WARNING: touch and burial constraints conflict; burial wins")
                for i, person in enumerate(anchor["people"]):
                    if person["fired"]:
                        q05, q25, q50 = anchor["anchor_pred_gap_q"][i]
                        print(f"  [body-anchor] chosen r={anchor['rescale_r']:.6f} for "
                              f"obj_id={person['object_id']}: predicted gaps "
                              f"p05={q05:+.3f} p25={q25:+.3f} p50={q50:+.3f} m")
                print(f"  [body-anchor] verified true gaps at r={anchor['rescale_r']:.6f}: "
                      f"touch_q25={anchor['anchor_true_touch']:+.3f} m "
                      f"burial_q05={anchor['anchor_true_burial']:+.3f} m")
                fixed_ids = {p["object_id"] for p in anchor["people"] if p["fired"]}
                built = _build_geometry(args, pred, cam, frame_indices, extr, K_px, objs,
                                        climber_ids, s_eff, base_scale=s,
                                        fixed_human_ids=fixed_ids)
                # Final acceptance on the ACTUAL candidate scene: re-voxelization at the
                # new metric can move surfaces by up to a voxel vs the scaled-centroid
                # verifier, which is material against the burial budget. Violation ⇒
                # veto the whole rescale and rebuild the baseline world.
                from tools.human_optim.depth_anchor import BURIAL_CAP_M, verify_built_world

                extr_eff = extr.copy()
                extr_eff[:, :3, 3] *= s_eff
                bt, bb = verify_built_world(
                    smplx_path, built["scene_points"], built["scene_confidence"], extr_eff,
                    frame_indices, sorted(fixed_ids), device=args.device)
                print(f"  [body-anchor] built-world gaps: touch_q25={bt:+.3f} m "
                      f"burial_q05={bb:+.3f} m")
                if not np.isfinite(bb) or bb < -BURIAL_CAP_M - 1e-6:
                    print("  [body-anchor] WARNING: built-world burial check FAILED "
                          f"({bb:+.3f} m) — rescale vetoed, rebuilding baseline world")
                    s_eff = s
                    built = _build_geometry(args, pred, cam, frame_indices, extr, K_px,
                                            objs, climber_ids, s, base_scale=s,
                                            fixed_human_ids=set())
                    baseline_touch, baseline_burial = verify_built_world(
                        smplx_path, built["scene_points"], built["scene_confidence"],
                        extr_base, frame_indices, sorted(fixed_ids), device=args.device)
                    _record_anchor_veto(anchor, baseline_touch, baseline_burial)
                    print(f"  [body-anchor] baseline built-world gaps: "
                          f"touch_q25={baseline_touch:+.3f} m "
                          f"burial_q05={baseline_burial:+.3f} m")
                else:
                    anchor["anchor_true_touch"], anchor["anchor_true_burial"] = bt, bb
                    print("  [body-anchor] WARNING: kinematic_1/contacts/uncertainty/"
                          "kinematic_2/dynamics for this clip are now stale")

    out_dir.mkdir(parents=True, exist_ok=True)
    extr_metric = extr.copy()
    extr_metric[:, :3, 3] *= s_eff
    K_orig = np.stack([denormalize_K(K_norm[k], ho, wo) for k in range(len(frame_indices))])
    fps = float(built["src_fps"]) if built["src_fps"] else 30.0
    transform_args = dict(
        scale=np.float32(s_eff), extrinsics=extr_metric.astype(np.float32),
        intrinsics_px_orig=K_orig.astype(np.float32), intrinsics=K_norm,
        fov_x_deg=cam["fov_x_deg"], fov_y_deg=cam["fov_y_deg"],
        frame_indices=frame_indices.astype(np.int32), image_height=np.int32(ho),
        image_width=np.int32(wo), vggt_height=np.int32(hv), vggt_width=np.int32(wv),
        fps=np.float32(fps), scale_method=np.str_("human-anchored (estimate_scale)"))
    if anchor is not None:
        transform_args.update(
            scale_base=np.float32(s), rescale_r=np.float32(anchor["rescale_r"]),
            depth_refuted=anchor["depth_refuted"], anchor_gap_m=anchor["anchor_gap_m"],
            anchor_burial_m=anchor["anchor_burial_m"], anchor_support=anchor["anchor_support"],
            anchor_pred_gap_q=anchor["anchor_pred_gap_q"],
            anchor_true_touch=np.float32(anchor["anchor_true_touch"]),
            anchor_true_burial=np.float32(anchor["anchor_true_burial"]))
    np.savez(out_dir / "transform.npz", **transform_args)
    _save_cloud(out_dir / "scene.npz", built["scene_points"],
                colors=built["scene_color"], confidence=built["scene_confidence"],
                count=built["scene_count"].to(torch.int32),
                voxel_size=np.float32(args.voxel_scene))
    _save_cloud(
        out_dir / "human.npz",
        built["human_points"], colors=built["human_color"],
        confidence=built["human_confidence"], frame_id=built["human_frame_id"],
        obj_id=built["human_obj_id"],
        frame_indices=frame_indices.astype(np.int32),
    )
    print(f"  scale={s_eff:.4f} | scene {built['scene_points'].shape[0]} pts "
          f"({built['n_dropped']} dropped: small clusters) | human {built['n_human']} pts "
          f"(climbers {climber_ids}) -> {out_dir}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True,
                   help="clip video / folder of frames (RGB, for per-point colour)")
    p.add_argument("--predictions", required=True, type=Path, help="clip dir, e.g. out/<stem>/")
    p.add_argument("--voxel-scene", type=float, default=0.05,
                   help="scene merge voxel size in metres (default: 0.05)")
    p.add_argument("--conf-drop-q", type=float, default=0.3,
                   help="drop this lowest-confidence fraction of VGGT depth (default: 0.3)")
    p.add_argument("--edge-rtol", type=float, default=0.04,
                   help="depth-edge (flying-pixel) filter tolerance; 0 disables (default: 0.04)")
    p.add_argument("--human-dilate", type=int, default=8,
                   help="grow the human mask by N px before subtracting it from the scene, so "
                        "imperfect mask edges don't leak human points into it (default: 8; 0 disables)")
    p.add_argument("--min-scene-voxels", type=int, default=25,
                   help="drop scene clusters smaller than this many connected voxels "
                        "(floating fragments / off-surface blobs); 1 disables (default: 25)")
    p.add_argument("--person-obj-ids", default=None,
                   help="comma-separated SAM3 object ids to treat as people (default: all)")
    p.add_argument("--body-anchor", choices=("on", "off"), default="on",
                   help="body-anchored world rescale gate (default: on)")
    p.add_argument("--float-gate-m", type=float, default=0.28,
                   help="median extremity/scene distance that fires the gate (default: 0.28 — "
                        "calibrated on the 8-scene hard set: healthy rows top out at 0.225, "
                        "floaters start at 0.346)")
    p.add_argument("--device", default="cuda", help="cuda | cpu (default: cuda)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip if geometry/transform.npz already exists")
    fuse(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
