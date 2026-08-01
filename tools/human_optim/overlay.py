"""Reprojection overlay for the SMPL-X kinematic stage (debug / QA).

Draws the optimized body's projected Sapiens keypoints (green) against the
observed 2D target (red) on each frame, per person. A tight green-on-red overlay
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


def render_contact_overlay(source, out_path, points_2d, contact, parents,
                           valid_mask, H, W, fps, label_confidence=None):
    """Write an mp4 of projected body points coloured by scene contact.

    One point set per tracked person, each point drawn as a dot — **red** when it
    is in contact with the scene, **green** otherwise. Everything is precomputed
    by the stage; this only rasterizes.

    ``points_2d (P, N, K, 2)`` are the projected pixels and ``contact (P, N, K)``
    their contact flags — the 12 contact frames for the contact stage. ``parents
    (K,)`` optionally names a tree parent per point (-1 for a root), drawn as grey
    bones; pass ``None`` for a point set with no tree, e.g. the contact frames.
    ``valid_mask (P, N)`` gates frames where a person isn't tracked. When
    ``label_confidence (P, N, K)`` is given, a **contact** dot's radius grows with
    the confidence in its label.
    """
    import tempfile

    from tools.source import resolve_source

    out_path = Path(out_path)
    P, N, J = contact.shape

    def ok(uv):  # finite + not a behind-camera blow-up
        return np.isfinite(uv).all() and abs(uv[0]) < 1e4 and abs(uv[1]) < 1e4

    with tempfile.TemporaryDirectory(prefix="contacts_overlay_") as tmp:
        frame_paths, src_fps = resolve_source(Path(source), Path(tmp))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(fps or src_fps or 30.0), (W, H))
        try:
            for i in tqdm(range(min(N, len(frame_paths))), desc="contact overlay"):
                bgr = cv2.imread(str(frame_paths[i]))
                if bgr is None:
                    continue
                for p in range(P):
                    if not valid_mask[p, i]:
                        continue
                    pts = points_2d[p, i]                       # (K, 2)
                    if parents is not None:
                        for j in range(1, J):                   # bones (grey)
                            a, b = pts[int(parents[j])], pts[j]
                            if ok(a) and ok(b):
                                cv2.line(bgr, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                                         (180, 180, 180), 1, cv2.LINE_AA)
                    for j in range(J):                          # points (red=contact / green=free)
                        uv = pts[j]
                        if not ok(uv):
                            continue
                        touching = bool(contact[p, i, j])
                        col = (0, 0, 255) if touching else (0, 255, 0)
                        r = 3
                        if touching and label_confidence is not None:  # confident contact ⇒ bigger dot
                            r = 2 + int(round(4.0 * float(np.clip(label_confidence[p, i, j], 0.0, 1.0))))
                        cv2.circle(bgr, (int(uv[0]), int(uv[1])), r, col, -1, cv2.LINE_AA)
                writer.write(bgr)
        finally:
            writer.release()
    return out_path


def render_keypoint_overlay(source, out_path, mid, save, corr, fvids, extr, K,
                            *, target_scheme, fps, device, score_thresh=0.3):
    """Write an mp4 overlaying predicted (green) vs observed (red) Sapiens keypoints.

    A tight green-on-red overlay means the reprojection term is doing its job.
    """
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

    desc = "reprojection overlay"
    with tempfile.TemporaryDirectory(prefix="smplx_kin_overlay_") as tmp:
        frame_paths, src_fps = resolve_source(Path(source), Path(tmp))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(fps or src_fps or 30.0), (W, H))
        try:
            for i in tqdm(range(min(N, len(frame_paths))), desc=desc):
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
