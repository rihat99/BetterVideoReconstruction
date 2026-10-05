"""Run Li et al.'s own contact recogniser (CVPR'19) over the estimator's input clips.

The released recogniser (``~/better/other/contact-recognizer``) is python 2 / torch 0.3
code split over three scripts: ``create_data/resize_images.py`` scales every frame of a
video by ONE factor read off the person's size, ``create_data/create_data.py`` crops a
square patch around each joint into an HDF5 file, and ``test_model.py`` pushes those
patches through five ResNet-18s (neck, hands, knees, soles, toes). This is the same
computation in one pass, reading the frames straight from the video and the joints from
the ``data/<dataset>/ours/<clip>.npz`` exports the estimator already runs on.

Two things differ from the original by necessity. The 2D joints are Sapiens Goliath
folded to OpenPose's 18 COCO joints, not OpenPose, so the "undetected" test cannot be
"confidence exactly 0": ``--conf`` (0.1 by default) stands in for it, and Sapiens is
above it on every joint of these three datasets. And the frames come out of the mp4
rather than a directory of JPEGs, which is what the estimator itself reads too.

Output, one ``.npz`` per clip under ``data/<dataset>/recognizer/``:
``contact_states (nf, 9, 4)`` uint8 one-hot in the recogniser's joint order
``[neck, l_hand, r_hand, l_knee, r_knee, l_sole, r_sole, l_toes, r_toes]`` with classes
``0 contact, 1 not in contact, 2 occluded, 3 undetected``, and ``scores (nf, 9, 4)``, the
network's logits in the same layout (undetected rows carry the initial ``[0,0,0,1]``).

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run --no-sync python scripts/methods/contact_recognizer.py \
        --dataset Parkour-dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms

ESTMF = (Path(__file__).resolve().parents[2].parent / "data"
         / "Estimating-3D-Motion-Forces").resolve()
RECOGNIZER = (Path(__file__).resolve().parents[2].parent / "other" / "contact-recognizer").resolve()

#: ``<clip>.mp4`` directory per dataset, relative to ``ESTMF/data/<dataset>``.
VIDEO_DIRS = {"Parkour-dataset": "videos_all", "climb_wall_3": "videos", "OpenCap": "videos"}

#: The five models: name, patch size, checkpoint, the OpenPose joint each output crops,
#: and the row of ``contact_states`` it writes. Soles and toes crop the SAME ankle pixel.
MODELS = (
    ("neck", 180, "v1-lr-0.005_7172a45d_neck_180.pth.tar", (1,), (0,)),
    ("hands", 120, "v1-lr-0.001_df61e384_hands_120.pth.tar", (7, 4), (1, 2)),
    ("knees", 120, "v1-lr-0.001_127696d5_knees_120.pth.tar", (12, 9), (3, 4)),
    ("soles", 120, "v1-lr-0.0012_0ba883bc_soles_120.pth.tar", (13, 10), (5, 6)),
    ("toes", 120, "v1-lr-0.0012_db801a67_toes_120.pth.tar", (13, 10), (7, 8)),
)
#: ``create_data/resize_images.py``: the longest detected torso / leg link is scaled to
#: this many pixels, and the resized frame width is held inside ``WIDTH_LIMITS``.
LINKS_TORSO = ((5, 11), (2, 8), (5, 8), (2, 11))
LINKS_LEGS = ((11, 12), (8, 9), (12, 13), (9, 10))
TARGET_TORSO, TARGET_LEG = 140.0, 120.0
WIDTH_LIMITS = (300.0, 1500.0)
TRANSFORM = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def resize_scale(width: int, joints: np.ndarray, conf: float) -> float:
    """The one scale factor the recogniser resizes a whole video by."""
    scales = np.ones(len(joints))
    for i, j2d in enumerate(joints):
        lengths = []
        for links, target in ((LINKS_TORSO, TARGET_TORSO), (LINKS_LEGS, TARGET_LEG)):
            longest = 0.0
            for a, b in links:
                if j2d[a, 2] > conf and j2d[b, 2] > conf:
                    longest = max(longest, float(np.linalg.norm(j2d[a, :2] - j2d[b, :2])))
            lengths.append((longest, target))
        (torso, t_torso), (leg, t_leg) = lengths
        if torso >= leg and torso > 0.0:
            scales[i] = t_torso / torso
        elif leg > torso and leg > 0.0:
            scales[i] = t_leg / leg
    scale = float(scales.mean())
    if width * scale < WIDTH_LIMITS[0]:
        scale = WIDTH_LIMITS[0] / width
    elif width * scale > WIDTH_LIMITS[1]:
        scale = WIDTH_LIMITS[1] / width
    return scale


def load_models(device: torch.device) -> list[tuple]:
    """``(name, patch, joints, rows, net, nclasses)`` for each of the five recognisers."""
    loaded = []
    for name, patch, checkpoint, joints, rows in MODELS:
        state = torch.load(RECOGNIZER / "models" / f"{name}_{patch}" / "checkpoints" / checkpoint,
                           map_location="cpu", encoding="latin1", weights_only=False)
        net = torchvision.models.resnet18()
        net.fc = nn.Linear(net.fc.in_features, state["nclasses"])
        net.load_state_dict(state["model_state"])
        loaded.append((name, state["patch_size"], joints, rows,
                       net.to(device).eval(), state["nclasses"]))
    return loaded


def run_clip(video: Path, joints: np.ndarray, models: list[tuple], conf: float,
             device: torch.device, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Recognise the contact states of one clip; ``joints`` is ``(nf, 18, 3)`` in video pixels."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = resize_scale(width, joints, conf)
    scaled = joints.copy()
    scaled[:, :, :2] *= scale

    num_frames = len(joints)
    scores = np.zeros((num_frames, 9, 4), np.float32)
    scores[:, :, 3] = 1.0
    states = np.zeros((num_frames, 9, 4), np.uint8)
    states[:, :, 3] = 1

    frame = 0
    while frame < num_frames:
        images = []
        for _ in range(min(chunk, num_frames - frame)):
            ok, bgr = capture.read()
            if not ok:
                break
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            images.append(image.resize((int(image.width * scale), int(image.height * scale)),
                                       Image.LANCZOS))
        if not images:
            break
        for _, patch, model_joints, rows, net, nclasses in models:
            patches, index = [], []
            for k, image in enumerate(images):
                for joint, row in zip(model_joints, rows):
                    if scaled[frame + k, joint, 2] <= conf:
                        continue
                    x, y = scaled[frame + k, joint, :2]
                    half = patch // 2
                    patches.append(TRANSFORM(image.crop(
                        (x - half, y - half, x - half + patch, y - half + patch))))
                    index.append((frame + k, row))
            if not patches:
                continue
            with torch.no_grad():
                logits = net(torch.stack(patches).to(device)).cpu().numpy()
            for (f, row), logit in zip(index, logits):
                scores[f, row] = 0.0
                scores[f, row, :nclasses] = logit
                states[f, row] = 0
                states[f, row, int(np.argmax(logit))] = 1
        frame += len(images)
    capture.release()
    if frame < num_frames:
        raise RuntimeError(f"{video}: video ended at frame {frame}, joints have {num_frames}")
    return states, scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(VIDEO_DIRS), required=True)
    parser.add_argument("--clips", default=None, help="comma-separated names (default: all)")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="2D confidence below which a joint counts as undetected")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = ESTMF / "data" / args.dataset
    videos = root / VIDEO_DIRS[args.dataset]
    out_dir = root / "recognizer"
    out_dir.mkdir(exist_ok=True)
    names = sorted(p.stem for p in (root / "ours").glob("*.npz")
                   if not p.name.startswith("ours_"))
    if args.clips:
        names = [n for n in names if n in set(args.clips.split(","))]

    device = torch.device(args.device)
    models = load_models(device)
    for name in names:
        joints = np.asarray(np.load(root / "ours" / f"{name}.npz")["coco18"], np.float64)
        states, scores = run_clip(videos / f"{name}.mp4", joints, models, args.conf, device)
        np.savez_compressed(out_dir / f"{name}.npz", contact_states=states, scores=scores)
        rate = (states[:, :, 0] == 1).mean(0)
        print(f"{name}: {len(joints):4d} frames, contact rate " +
              " ".join(f"{r:.2f}" for r in rate), flush=True)
    print(f"{len(names)} clip(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
