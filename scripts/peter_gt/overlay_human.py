"""Side-by-side overlay video of a fitted body in the three calibrated cameras.

For every frame the body (``kinematic_1`` / ``kinematic_2`` / ``kindyn_1`` / a single
view's own SAM-3D-Body estimate) is projected into each undistorted view as a vertex
point cloud (green), with the skeleton in white, the Sapiens keypoints the fit saw as
yellow dots and the boards' loaded contact frames as red dots; the three views are
scaled to one height and tiled left to right. This is the quickest way to see whether
the multi-view body lands on the person in EVERY camera at once.

    uv run --no-sync python scripts/peter_gt/overlay_human.py --clip a_00_01_part3 [--body kinematic_1]
    → peter/climb_wall_3_gt/human/<clip>/human_optim/overlay_<body>.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from bvr.body import load_body  # noqa: E402
from bvr.human_optim.qmath import normalize_q  # noqa: E402
from bvr.paths import PETER  # noqa: E402

HUMAN_ROOT = PETER / "climb_wall_3_gt/human"
CONFIG = Path(__file__).resolve().parent / "human_gt.toml"
VIEW_NAMES = ["cam_left", "cam_right", "cam_3"]
TILE_H = 720
FFMPEG = next((_REPO / ".venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries").glob("ffmpeg-*"))


def project(points: np.ndarray, extr: np.ndarray, K: np.ndarray) -> np.ndarray:
    cam = points @ extr[:3, :3].T + extr[:3, 3]
    uv = cam[:, :2] / np.clip(cam[:, 2:3], 1e-6, None)
    return np.stack([K[0, 0] * uv[:, 0] + K[0, 2], K[1, 1] * uv[:, 1] + K[1, 2]], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--body", default="kinematic_1",
                    help="kinematic_1 | kinematic_2 | kindyn_1 | sam3d:<view> (that view's own estimate)")
    ap.add_argument("--stride", type=int, default=12, help="draw every Nth mesh vertex")
    args = ap.parse_args()
    clip_dir = HUMAN_ROOT / args.clip
    cams = np.load(clip_dir / "cameras.npz", allow_pickle=True)
    kin = np.load(clip_dir / "human_optim" / "kinematic_1.npz", allow_pickle=True)
    if args.body.startswith("sam3d:"):
        k = VIEW_NAMES.index(args.body.split(":", 1)[1])
        q, betas, valid = kin["view_q_world"][k], kin["view_betas"][k], kin["view_valid"][k]
    else:
        z = np.load(clip_dir / "human_optim" / f"{args.body}.npz", allow_pickle=True)
        q, betas, valid = z["q"][0], z["betas"][0], z["valid_mask"][0]
    cnt_path = clip_dir / "human_optim" / "contacts_1.npz"
    labels = np.load(cnt_path)["frame_contact"][0] if cnt_path.exists() else None
    n = len(q)                                   # the body's frames (cam_left's clock)
    offsets = np.asarray(cams["frame_offset"], int) if "frame_offset" in cams.files else np.zeros(3, int)

    with CONFIG.open("rb") as f:
        frames = tomllib.load(f)["contacts"]["frames"]
    body = load_body(device="cuda" if torch.cuda.is_available() else "cpu", frames_allow=frames)
    parents = [int(body.parent_tree[0, j]) for j in range(body.num_joints)]
    with torch.no_grad():
        dev = body.vertices_template.device
        b = torch.as_tensor(betas, dtype=torch.float32, device=dev)[None].expand(n, -1)
        out = body.forward(b, normalize_q(torch.as_tensor(q, dtype=torch.float32, device=dev), body.num_joints))
        verts = out.vertices[:, ::args.stride].cpu().numpy()
        joints = out.joints.cpu().numpy()
        cframes = out.frames_world[..., :3].cpu().numpy()

    sapiens = []
    for v, off in zip(VIEW_NAMES, offsets):
        pz = clip_dir / "views" / v / "sapiens" / "pose.npz"
        if pz.exists():
            z = np.load(pz, allow_pickle=True)
            sapiens.append((z["keypoints"][0][off:off + n], z["keypoint_scores"][0][off:off + n]))
        else:
            sapiens.append(None)
    caps = [cv2.VideoCapture(str(clip_dir / "views" / v / "video.mp4")) for v in VIEW_NAMES]
    for cap, off in zip(caps, offsets):          # a lagging view shows cam_left's frame t at its t + offset
        for _ in range(int(off)):
            cap.read()
    scales = [TILE_H / int(cams["image_size"][k][1]) for k in range(3)]
    tile_w = [2 * int(round(int(cams["image_size"][k][0]) * scales[k] / 2)) for k in range(3)]   # even: yuv420p
    out_path = clip_dir / "human_optim" / f"overlay_{args.body.replace(':', '_')}.mp4"
    W, H = sum(tile_w), TILE_H
    cmd = [str(FFMPEG), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(float(cams["fps"])), "-i", "-",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(out_path)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as enc:
        for t in range(n):
            tiles = []
            for k, cap in enumerate(caps):
                ok, img = cap.read()
                if not ok:
                    img = np.zeros((int(cams["image_size"][k][1]), int(cams["image_size"][k][0]), 3), np.uint8)
                extr, K, s = cams["extrinsics"][k].astype(np.float64), cams["K"][k].astype(np.float64), scales[k]
                img = cv2.resize(img, (tile_w[k], TILE_H))
                if sapiens[k] is not None:
                    kp, sc = sapiens[k]
                    for u, v in (kp[t][sc[t] >= 0.3] * s).astype(int):
                        cv2.circle(img, (u, v), 2, (0, 220, 255), -1)
                for u, v in (project(verts[t], extr, K) * s).astype(int):
                    cv2.circle(img, (u, v), 1, (60, 220, 60), -1)
                jpx = project(joints[t], extr, K) * s
                for j, p in enumerate(parents):
                    if p >= 0:
                        cv2.line(img, tuple(jpx[j].astype(int)), tuple(jpx[p].astype(int)), (255, 255, 255), 1)
                if labels is not None:
                    for c, (u, v) in enumerate((project(cframes[t], extr, K) * s).astype(int)):
                        if labels[t, c]:
                            cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
                if not valid[t]:
                    cv2.putText(img, "untracked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                tiles.append(img)
            frame = np.concatenate(tiles, 1)
            cv2.putText(frame, f"{args.clip} {args.body} t={t}", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            enc.stdin.write(frame.tobytes())
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
