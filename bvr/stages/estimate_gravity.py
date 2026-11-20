"""Find the direction of gravity in the clip's world frame, from GeoCalib and from the floor.

The camera moves and tilts, so gravity cannot be read off it. GeoCalib predicts the up direction
of a single image in the camera frame, with a confidence. On ``[gravity].num_frames`` frames spread
over the clip, each up direction is turned into a world-frame gravity with the frame's camera
rotation (``g_world = -R^T up_camera``). Gravity is the same on every frame, so the frames must
agree: the stage keeps the frames GeoCalib is most confident about, drops the ones far from their
mean, and takes the confidence-weighted spherical mean of the rest. The estimate is reliable only
when GeoCalib's best confidence is above a floor, the kept frames agree, and they were seen from
camera orientations different enough that their agreement means something.

The fused scene is the second witness: a plane fitted to its lowest points, accepted when it has
enough points and lies within ``[gravity].ground_max_tilt_deg`` of the expected down direction.
Its upward normal is world-up.

Which one wins: reliable GeoCalib, else an accepted floor, else ``[gravity].fallback_gravity``.
GeoCalib outranks the floor because it validates only outdoors, where visible ground can slope;
indoors it rarely validates and the level floor is the best evidence.

Writes, under the clip's directory::

    geocalib/gravity.npz   bvr-gravity: gravity_world (3,) unit, world frame, the chosen one; up_world;
                           source ("geocalib", "ground" or "fallback_down"); reliable; ground_accepted,
                           ground_reject, and the fitted plane (ground_centroid, ground_normal,
                           ground_extent, ground_points, ground_angle_deg) when one was found;
                           GeoCalib's own estimate gravity_world_geocalib and, per sampled frame,
                           gravity_world_frames, up_cam_frames, confidence, roll_deg, pitch_deg,
                           focal_px, rows, frame_indices, selected, angle_deg; the gate values
"""
from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from bvr import files, paths, run
from bvr.geometry.ground import fit_ground_plane

log = logging.getLogger(__name__)


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    """Load GeoCalib once; return the per-clip gravity estimate."""
    from geocalib import GeoCalib

    settings = stage.config["gravity"]
    model = GeoCalib(weights=settings["weights"]).to(stage.device).eval()
    fallback = np.array([float(value) for value in settings["fallback_gravity"].split(",")])
    if fallback.shape != (3,) or not np.linalg.norm(fallback) > 0:
        raise ValueError(f"[gravity].fallback_gravity must be 'x,y,z', not zero; got {settings['fallback_gravity']!r}")
    fallback = fallback / np.linalg.norm(fallback)

    @torch.no_grad()
    def calibrate(image: np.ndarray) -> dict:
        """GeoCalib on one BGR frame: camera-frame up (unit), mean up-field confidence, roll, pitch, focal."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = model.calibrate(torch.from_numpy(rgb).permute(2, 0, 1).float().to(stage.device) / 255.0,
                                 camera_model="pinhole")
        gravity = result["gravity"]
        return {"up": gravity.vec3d[0].cpu().numpy().astype(np.float64),
                "confidence": float(result["up_confidence"][0].mean()), "roll": float(gravity.roll[0]),
                "pitch": float(gravity.pitch[0]), "focal": float(result["camera"].f[0].mean())}

    def process(clip: paths.Clip) -> None:
        transform = files.load_npz(clip.find(paths.TRANSFORM), "bvr-transform")
        extrinsics = transform["extrinsics"].astype(np.float64)  # (T, 4, 4) camera-from-world
        frame_indices = transform["frame_indices"].astype(int)
        rows = np.unique(np.linspace(0, len(extrinsics) - 1, settings["num_frames"]).round().astype(int))
        found = []
        with tempfile.TemporaryDirectory(prefix="gravity_") as scratch:
            frame_paths = files.extract_frames(clip.video, Path(scratch) / "frames")
            for row in rows:
                if frame_indices[row] >= len(frame_paths):
                    continue
                estimate = calibrate(cv2.imread(str(frame_paths[frame_indices[row]])))
                world = -(extrinsics[row][:3, :3].T @ estimate["up"])
                if np.all(np.isfinite(world)):
                    found.append((row, world, estimate))
        if not found:
            raise RuntimeError("GeoCalib gave no usable frame")
        used = np.array([row for row, _, _ in found])
        worlds = np.stack([world for _, world, _ in found])
        confidence = np.array([estimate["confidence"] for _, _, estimate in found])
        geocalib, selected, angle = _combine(worlds, confidence, conf_rel=settings["conf_rel"],
                                             inlier_deg=settings["inlier_deg"], min_frames=settings["min_frames"])

        spread = angle[selected]
        spread_median = float(np.median(spread)) if spread.size else float("nan")
        diversity = _rotation_spread_deg([extrinsics[row][:3, :3] for row in used[selected]])
        failed = [reason for reason, failing in (
            (f"max confidence {confidence.max():.4f} < {settings['conf_floor']:g}", confidence.max() < settings["conf_floor"]),
            (f"spread median {spread_median:.1f} deg > {settings['max_spread_deg']:g}",
             not spread_median <= settings["max_spread_deg"]),
            (f"camera rotation diversity {diversity:.1f} deg < {settings['min_diversity_deg']:g}",
             diversity < settings["min_diversity_deg"])) if failing]
        reliable = not failed

        # The floor: the prior down (GeoCalib when reliable, else the fallback) seeds the lowest band and the tilt gate.
        prior = geocalib if reliable else fallback
        ground, plane = None, fit_ground_plane(files.load_npz(clip.find(paths.SCENE), "bvr-scene")["points"], prior,
                                               max_tilt_deg=settings["ground_max_tilt_deg"])
        if plane is None:
            ground_reject = "no planar floor in the lowest band"
        else:
            centroid, up, extent, inliers = plane
            tilt = float(np.degrees(np.arccos(np.clip(up @ -prior, -1.0, 1.0))))
            if inliers < settings["ground_min_points"]:
                ground_reject = f"{inliers} inlier points < {settings['ground_min_points']}"
            elif tilt > settings["ground_max_tilt_deg"]:
                ground_reject = f"normal {tilt:.1f} deg from expected down (> {settings['ground_max_tilt_deg']:g} deg)"
            else:
                ground, ground_reject = -up, ""
        gravity, source = ((geocalib, "geocalib") if reliable else (ground, "ground") if ground is not None
                           else (fallback, "fallback_down"))

        plane_fields = {} if plane is None else dict(
            ground_centroid=centroid.astype(np.float32), ground_normal=up.astype(np.float32),
            ground_extent=np.float32(extent), ground_points=np.int32(inliers), ground_angle_deg=np.float32(tilt))
        estimates = [estimate for _, _, estimate in found]
        files.save_npz(
            clip.path(paths.GRAVITY), "bvr-gravity", **plane_fields,
            ground_accepted=np.bool_(ground is not None), ground_reject=np.str_(ground_reject),
            gravity_world=gravity.astype(np.float32), up_world=(-gravity).astype(np.float32), source=np.str_(source),
            gravity_world_geocalib=geocalib.astype(np.float32), gravity_world_frames=worlds.astype(np.float32),
            up_cam_frames=np.stack([estimate["up"] for estimate in estimates]).astype(np.float32),
            confidence=confidence.astype(np.float32),
            roll_deg=np.degrees([estimate["roll"] for estimate in estimates]).astype(np.float32),
            pitch_deg=np.degrees([estimate["pitch"] for estimate in estimates]).astype(np.float32),
            focal_px=np.array([estimate["focal"] for estimate in estimates], np.float32),
            rows=used.astype(np.int32), frame_indices=frame_indices[used].astype(np.int32), selected=selected,
            angle_deg=angle.astype(np.float32), num_frames=np.int32(len(found)),
            num_selected=np.int32(selected.sum()), inlier_spread_median_deg=np.float32(spread_median),
            inlier_spread_max_deg=np.float32(spread.max() if spread.size else np.nan),
            angle_to_world_y_deg=np.float32(np.degrees(np.arccos(np.clip(geocalib[1], -1.0, 1.0)))),
            max_confidence=np.float32(confidence.max()), camera_diversity_deg=np.float32(diversity),
            reliable=np.bool_(reliable), conf_rel=np.float32(settings["conf_rel"]),
            inlier_deg=np.float32(settings["inlier_deg"]),
            conf_threshold=np.float32(settings["conf_rel"] * confidence.max()),
            conf_floor=np.float32(settings["conf_floor"]), min_diversity_deg=np.float32(settings["min_diversity_deg"]),
            fallback_gravity=fallback.astype(np.float32), weights=np.str_(settings["weights"]))
        log.info("gravity [%+.4f %+.4f %+.4f] from %s; GeoCalib %s (%d/%d frames)%s; floor %s", *gravity, source,
                 "reliable" if reliable else "unreliable", int(selected.sum()), len(found),
                 "" if reliable else f": {'; '.join(failed)}", "accepted" if ground is not None else f"rejected: {ground_reject}")

    return process


def _combine(worlds: np.ndarray, confidence: np.ndarray, *, conf_rel: float, inlier_deg: float,
             min_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gravity ``(3,)`` from per-frame world gravities ``(S, 3)``, the frames used ``(S,)`` and each frame's angle to it.

    Frames under ``conf_rel`` of the best confidence are dropped (at least ``min_frames`` stay),
    then one round drops the frames more than ``inlier_deg`` from the mean if enough remain.
    """
    count = len(worlds)
    floor = min(min_frames, count)
    keep = confidence >= conf_rel * confidence.max() if confidence.max() > 0 else np.ones(count, bool)
    if keep.sum() < floor:
        keep = np.zeros(count, bool)
        keep[np.argsort(confidence)[::-1][:floor]] = True
    mean = _spherical_mean(worlds[keep], confidence[keep])
    angle = np.degrees(np.arccos(np.clip(worlds @ mean, -1.0, 1.0)))
    inliers = keep & (angle < inlier_deg)
    if inliers.sum() >= max(1, floor // 2):
        mean = _spherical_mean(worlds[inliers], confidence[inliers])
        angle = np.degrees(np.arccos(np.clip(worlds @ mean, -1.0, 1.0)))
    else:
        inliers = keep
    return mean, inliers, angle


def _spherical_mean(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The weighted mean of unit vectors, back on the sphere (equal weights when all are zero)."""
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    mean = (vectors * weights[:, None]).sum(0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else vectors.mean(0)


def _rotation_spread_deg(rotations: list[np.ndarray]) -> float:
    """The largest angle between two of ``rotations``: near zero means one viewpoint, whose frames agree trivially."""
    return max((float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1.0) / 2.0, -1.0, 1.0))))
                for i, a in enumerate(rotations) for b in rotations[i + 1:]), default=0.0)


if __name__ == "__main__":
    run.main(load)
