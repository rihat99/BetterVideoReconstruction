"""Reprojection overlay for the SMPL-X kinematic stage (debug / QA).

Draws the optimized body's projected keypoints (green) against the Sapiens
target keypoints (red) on each frame, per person. A tight green-on-red overlay
means the reprojection term is doing its job.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .camera import project, transform_points


@torch.no_grad()
def _project_person(mid, q, betas, fvids, corr, extr, K, device, scale=1.0, chunk=64):
    """Return predicted keypoint pixels ``(N, Kpts, 2)`` for one person.

    ``scale`` is the per-person global body scale (about the pelvis) the kinematic
    stage fit; it must be re-applied here so the overlay matches the stored body.
    """
    N = q.shape[0]
    px = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        out = mid.forward(betas[None].expand(e - s, -1), q[s:e])
        jw = out.joints_world.tensor()[..., :3]
        root = jw[:, :1, :]
        jw = root + scale * (jw - root)
        verts = root + scale * (out.vertices - root)
        body = jw[:, corr.joint_model, :]
        lm = verts[:, fvids[corr.lm_frame], :]
        pred = torch.cat([body, lm], dim=1)
        px.append(project(transform_points(pred, extr[s:e]), K[s:e]))
    return torch.cat(px, 0).cpu().numpy()


def render_keypoint_overlay(source, out_path, mid, save, corr, fvids, extr, K,
                            *, target_scheme, fps, device, score_thresh=0.3):
    """Write an mp4 overlaying predicted (green) vs Sapiens (red) keypoints."""
    import tempfile

    from tools.source import resolve_source

    out_path = Path(out_path)
    P, N = save["q"].shape[0], save["q"].shape[1]
    H, W = int(save["video_height"]), int(save["video_width"])
    ids = [int(x) for x in target_scheme["object_ids"]]

    scales = save["scale"] if "scale" in save else np.ones(P, np.float32)
    pred_px = []
    for p in range(P):
        q = torch.as_tensor(save["q"][p], dtype=torch.float32, device=device)
        betas = torch.as_tensor(save["betas"][p], dtype=torch.float32, device=device)
        pred_px.append(_project_person(mid, q, betas, fvids, corr, extr, K, device,
                                       scale=float(scales[p])))

    with tempfile.TemporaryDirectory(prefix="smplx_kin_overlay_") as tmp:
        frame_paths, src_fps = resolve_source(Path(source), Path(tmp))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(fps or src_fps or 30.0), (W, H))
        try:
            for i in tqdm(range(min(N, len(frame_paths))), desc="reprojection overlay"):
                bgr = cv2.imread(str(frame_paths[i]))
                if bgr is None:
                    continue
                for p in range(P):
                    if not save["valid_mask"][p, i]:
                        continue
                    sc = target_scheme["keypoint_scores"][ids.index(int(save["object_ids"][p]))][i]
                    tgt = target_scheme["keypoints"][ids.index(int(save["object_ids"][p]))][i]
                    for k, sap in enumerate(corr.sap.tolist()):
                        u, v = pred_px[p][i, k]
                        if np.isfinite(u) and np.isfinite(v):
                            cv2.circle(bgr, (int(u), int(v)), 3, (0, 255, 0), -1)
                        if sc[sap] >= score_thresh:
                            tu, tv = tgt[sap]
                            cv2.drawMarker(bgr, (int(tu), int(tv)), (0, 0, 255),
                                           cv2.MARKER_CROSS, 8, 1)
                writer.write(bgr)
        finally:
            writer.release()
    return out_path
