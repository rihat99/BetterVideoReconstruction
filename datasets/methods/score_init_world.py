"""World-frame metrics for the SAM-3D-Body initialisation, on both benchmarks.

SAM-3D-Body regresses a body per frame in the *camera* frame and has no world of
its own, which is why the pose tables leave its world columns blank. Carrying it
through the camera puts it on the same footing as the models that do predict a
world trajectory: every per-frame camera estimate is lifted by the inverse of that
frame's extrinsics, so the camera's motion becomes part of the body's path.

The lift uses the camera that *defines each reference's world*, so the numbers are
comparable with the rows already in the table:

- **ClimbingVideos** — our VGGT camera (``features/geometry/<scene>/transform.npz``),
  which moves, and whose own error therefore enters the result. That is honest:
  this is the world trajectory the initialisation actually implies.
- **climb_wall_3** — the calibrated left camera, which is the GT's world frame and
  is static, so the lift is the identity and the numbers isolate SAM-3D-Body's own
  per-frame instability.

Adds ``WA-MPJPE``, ``W-MPJPE``, ``RTE`` and ``Jitter`` to the ``sam3d/init`` row of
both metrics JSONs, in place. Every other row is left untouched.

Usage:
    uv run --no-sync python datasets/methods/score_init_world.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


import datasets.methods.score_published_pose as published
from datasets.climbingvideos.paths import CLIMBING
from datasets.peter.paths import PETER

CV_RESULTS = CLIMBING / "results"
CW_RESULTS = PETER / "climb_wall_3_gt" / "others"
BODY_KEYS = ("global_orient", "body_pose", "transl", "betas", "left_hand_pose",
             "right_hand_pose", "jaw_pose", "leye_pose", "reye_pose", "expression")


def to_world(pts: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Camera-frame points back into the world; the inverse of ``to_camera``."""
    return np.einsum("nkj,nji->nki", pts - w2c[:, None, :3, 3], w2c[:, :3, :3])


def accumulate(rows: list[dict]) -> dict:
    """Pool exactly as the two scorers do: segments for W-/WA-, scenes for RTE and Jitter."""
    seg = sum(r["segments"] for r in rows)
    return {"WA-MPJPE": sum(r["WA-MPJPE"] for r in rows) / seg,
            "W-MPJPE": sum(r["W-MPJPE"] for r in rows) / seg,
            "RTE": float(np.mean([r["RTE"] for r in rows])),
            "Jitter": float(np.mean([r["Jitter"] for r in rows]))}


def climbing_videos() -> dict:
    """The 108 scenes every model was scored on — the result directories, not the
    current scene list, which has drifted since those runs."""
    scenes = sorted(p.name for p in (CV_RESULTS / "gvhmr").iterdir() if p.is_dir())
    fps = {line.split("\t")[0]: float(line.split("\t")[2])
           for line in (CV_RESULTS / "test_scenes.txt").read_text().splitlines() if line}
    rows = []
    for scene in scenes:
        gj_w, _gv_w, w2c, valid, _check = published.reference("climbingvideos", scene)
        d = np.load(published.scene_dir("sam3d", scene) / "smplx_params.npz")
        pj_c, _ = published.pose({k: np.nan_to_num(d[k][0]) for k in BODY_KEYS})
        pj_w = to_world(pj_c, w2c)
        ok = valid & np.asarray(d["valid_mask"][0], bool)
        rate = fps.get(scene, 25.0)
        rows.append({**published.world_metrics(pj_w, gj_w, ok), "Jitter": published.jitter(pj_w, rate, ok)})
        print(f"[init-world] ClimbingVideos {scene} done", flush=True)
    return accumulate(rows)


def climb_wall_3() -> dict:
    rows = []
    for clip, fps in published.items("climb_wall_3"):
        gj_w, _gv_w, w2c, valid, _check = published.reference("climb_wall_3", clip)
        n = len(gj_w)
        d = np.load(published.OURS_ROOT / clip / "sam3d" / "smplx_params.npz")
        pj_c, _ = published.pose({k: np.nan_to_num(d[k][0][:n]) for k in BODY_KEYS})
        pj_w = to_world(pj_c, w2c)
        ok = valid & np.asarray(d["valid_mask"][0], bool)[:n]
        rows.append({**published.world_metrics(pj_w, gj_w, ok), "Jitter": published.jitter(pj_w, fps, ok)})
        print(f"[init-world] climb_wall_3 {clip} done", flush=True)
    return accumulate(rows)


def patch(path: Path, values: dict) -> None:
    metrics = json.loads(path.read_text())
    metrics["rows"]["sam3d/init"].update(values)
    metrics["rows"]["sam3d/init"]["world_note"] = (
        "world columns obtained by lifting the per-frame camera-frame body through the "
        "reference world's own camera extrinsics (score_init_world.py)")
    path.write_text(json.dumps(metrics, indent=2))
    print(f"[init-world] patched {path}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()   # --help only; no knobs
    # A round trip through the two transforms must be the identity.
    _r = np.linalg.qr(np.random.default_rng(0).normal(size=(4, 3, 3)))[0]
    _w2c = np.concatenate([np.concatenate([_r, np.random.default_rng(1).normal(size=(4, 3, 1))], 2),
                           np.tile(np.array([[[0., 0., 0., 1.]]]), (4, 1, 1))], 1)
    _p = np.random.default_rng(2).normal(size=(4, 22, 3))
    assert np.allclose(to_world(published.to_camera(_p, _w2c), _w2c), _p, atol=1e-9), "lift is not the inverse"

    cw_vals = climb_wall_3()
    cv_vals = climbing_videos()
    patch(CW_RESULTS / "gt_pose_metrics.json", cw_vals)
    patch(CV_RESULTS / "baseline_pose_metrics.json", cv_vals)
    for name, vals in (("climb_wall_3", cw_vals), ("ClimbingVideos", cv_vals)):
        print(f"{name:16s} " + "  ".join(f"{k} {v:.1f}" for k, v in vals.items()))
