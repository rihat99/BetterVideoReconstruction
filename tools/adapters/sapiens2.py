"""Sapiens2 inference adapter — wraps ``init_model`` for {pose, pointmap, normal}.

Three public entry points:

* :func:`load_sapiens2` — build a 5B model for one task. Auto-downloads the
  safetensors checkpoint from HuggingFace on first call (cached in
  ``HF_HOME``); a ``checkpoint_root`` override is accepted for the
  ``$SAPIENS_CHECKPOINT_ROOT/{pose,pointmap,normal}/`` layout the upstream
  repo uses. Casts the model to ``bf16`` / ``fp16`` if requested.

* :func:`forward_pose` — top-down 308-keypoint pose inference. Takes a full
  image + ``(K, 4)`` xyxy bboxes; returns ``(K, 308, 2)`` keypoints in
  *original-image* pixel coords plus ``(K, 308)`` per-joint scores.

* :func:`forward_dense` — pointmap or normal forward on a single full image.
  Returns ``(orig_H, orig_W, 3)`` — channels are ``(x, y, z)`` metric camera
  coords for pointmap, unit normals for normal.

Sapiens2's config system uses paths relative to ``sapiens/<pose|dense>/`` —
:func:`load_sapiens2` chdir's into the right subdir while building, then
restores the original CWD. Models for different tasks share the ViT backbone
*architecture* but not the weights (heads diverge), so loading two heads
costs 2× VRAM. Release one before loading the next when memory is tight.
"""
from __future__ import annotations

import contextlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


_HERE = Path(__file__).resolve()
# tools/adapters/sapiens2.py → tools/adapters → tools → repo root
_REPO_ROOT = _HERE.parents[2]
_SAPIENS_ROOT = _REPO_ROOT / "third_party" / "sapiens2"

# Map task → ``sapiens/<pose|dense>/`` working directory.
_TASK_DIRS: dict[str, Path] = {
    "pose": _SAPIENS_ROOT / "sapiens" / "pose",
    "pointmap": _SAPIENS_ROOT / "sapiens" / "dense",
    "normal": _SAPIENS_ROOT / "sapiens" / "dense",
}

# 5B config relative to each task dir. Sapiens2 ships configs at every tier
# (0.4b / 0.8b / 1b / 5b); we hardcode 5B per the project's chosen scale.
_TASK_CONFIG_RELS: dict[str, str] = {
    "pose": (
        "configs/keypoints308/shutterstock_goliath_3po/"
        "sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py"
    ),
    "pointmap": (
        "configs/pointmap/render_people/"
        "sapiens2_5b_pointmap_render_people-1024x768.py"
    ),
    "normal": (
        "configs/normal/metasim_render_people/"
        "sapiens2_5b_normal_metasim_render_people-1024x768.py"
    ),
}

# HuggingFace repos for the 5B checkpoints.
_HF_REPOS: dict[str, tuple[str, str]] = {
    "pose": ("facebook/sapiens2-pose-5b", "sapiens2_5b_pose.safetensors"),
    "pointmap": ("facebook/sapiens2-pointmap-5b", "sapiens2_5b_pointmap.safetensors"),
    "normal": ("facebook/sapiens2-normal-5b", "sapiens2_5b_normal.safetensors"),
}

NUM_KEYPOINTS = 308

Task = Literal["pose", "pointmap", "normal"]


@contextlib.contextmanager
def _chdir(target: Path):
    cwd = os.getcwd()
    os.chdir(str(target))
    try:
        yield
    finally:
        os.chdir(cwd)


def resolve_checkpoint(task: str, checkpoint_root: Path | None = None) -> str:
    """Return a local path to the 5B safetensors for ``task``.

    Probe order:

    1. ``<checkpoint_root>/<task>/sapiens2_5b_<task>.safetensors`` if that
       file exists (matches the upstream ``$SAPIENS_CHECKPOINT_ROOT`` layout).
    2. HF cache via :func:`huggingface_hub.hf_hub_download`. Downloads on
       first call; subsequent calls hit the cache.
    """
    if task not in _HF_REPOS:
        raise ValueError(f"unknown task: {task!r}; choices: {list(_HF_REPOS)}")
    repo_id, filename = _HF_REPOS[task]
    if checkpoint_root is not None:
        local = Path(checkpoint_root) / task / filename
        if local.exists():
            return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_sapiens2(
    task: str,
    *,
    checkpoint_root: Path | None = None,
    device: str = "cuda",
    dtype: str = "bf16",
):
    """Build and return a Sapiens2 model for {pose, pointmap, normal}.

    The returned model is in eval mode, on ``device``, and cast to
    ``dtype`` (``bf16`` / ``fp16`` / ``fp32``). For pose, ``model.codec``
    (UDP heatmap decoder) and ``model.pose_metainfo`` (skeleton + flip
    indices) are populated as side-effects so the demo-style decoding in
    :func:`forward_pose` works directly.
    """
    if task not in _TASK_DIRS:
        raise ValueError(f"unknown task: {task!r}; choices: {list(_TASK_DIRS)}")
    task_dir = _TASK_DIRS[task]
    cfg_rel = _TASK_CONFIG_RELS[task]
    ckpt_path = resolve_checkpoint(task, checkpoint_root)

    # Build + cast on CPU first: init_model would move the fp32 model to the GPU
    # before we can down-cast, a ~20 GB transient that OOMs a half-occupied card.
    if task == "pose":
        with _chdir(task_dir):
            from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
            from sapiens.pose.models import init_model
            model = init_model(cfg_rel, ckpt_path, device="cpu")
            model.pose_metainfo = parse_pose_metainfo(
                dict(from_file="configs/_base_/keypoints308.py")
            )
            codec_cfg = dict(model.cfg.codec)
            codec_type = codec_cfg.pop("type")
            if codec_type != "UDPHeatmap":
                raise NotImplementedError(f"unsupported codec: {codec_type}")
            model.codec = UDPHeatmap(**codec_cfg)
    else:
        with _chdir(task_dir):
            from sapiens.dense.models import init_model
            model = init_model(cfg_rel, ckpt_path, device="cpu")

    if dtype == "bf16":
        model = model.to(dtype=torch.bfloat16)
    elif dtype == "fp16":
        model = model.to(dtype=torch.float16)
    elif dtype != "fp32":
        raise ValueError(f"dtype must be one of bf16/fp16/fp32, got {dtype!r}")

    return model.to(device)


def _model_dtype(model) -> torch.dtype:
    """Return the dtype of the first parameter (or fp32 if no params)."""
    for p in model.parameters():
        return p.dtype
    return torch.float32


def forward_pose(
    model,
    image_bgr: np.ndarray,
    bboxes_xyxy: np.ndarray,
    *,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run top-down pose inference for ``K`` bboxes on one full image.

    Args:
        model: built by :func:`load_sapiens2` with ``task="pose"``.
        image_bgr: ``(H, W, 3)`` ``uint8`` BGR (cv2 layout).
        bboxes_xyxy: ``(K, 4)`` xyxy float bboxes in original-image pixels.
        batch_size: how many bbox-crops to forward in one call. 5B at
            1024×768 is ~20 GB / forward in bf16, so default ``1``.

    Returns:
        ``keypoints``: ``(K, 308, 2)`` ``float32`` in original-image pixels.
        ``scores``: ``(K, 308)`` ``float32`` in ``[0, 1]``.
    """
    bboxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
    K = len(bboxes)
    if K == 0:
        return (
            np.zeros((0, NUM_KEYPOINTS, 2), dtype=np.float32),
            np.zeros((0, NUM_KEYPOINTS), dtype=np.float32),
        )

    # Build per-bbox top-down crops via the model's pipeline + preprocessor.
    inputs_list: list[torch.Tensor] = []
    samples_list: list[dict] = []
    for bbox in bboxes:
        info = dict(img=image_bgr)
        info["bbox"] = bbox[None]
        info["bbox_score"] = np.ones(1, dtype=np.float32)
        data = model.pipeline(info)
        data = model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        samples_list.append(data["data_samples"])
    inputs_full = torch.cat(inputs_list, dim=0)
    inputs_full = inputs_full.to(dtype=_model_dtype(model))

    val_cfg = getattr(model.cfg, "val_cfg", None)
    flip_test = (
        val_cfg is not None
        and (val_cfg.get("flip_test", False) if hasattr(val_cfg, "get") else getattr(val_cfg, "flip_test", False))
    )
    flip_indices = (
        model.pose_metainfo["flip_indices"] if flip_test else None
    )

    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for s in range(0, K, batch_size):
            chunk = inputs_full[s:s + batch_size]
            p = model(chunk)
            if flip_test:
                pf = model(chunk.flip(-1))
                pf = pf.flip(-1)[:, flip_indices]
                p = (p + pf) / 2.0
            preds.append(p.float())
    pred_arr = torch.cat(preds, dim=0).cpu().numpy()  # (K, num_kpts, h, w)

    keypoints = np.zeros((K, NUM_KEYPOINTS, 2), dtype=np.float32)
    scores = np.zeros((K, NUM_KEYPOINTS), dtype=np.float32)
    for i, ds in enumerate(samples_list):
        kpts, scs = model.codec.decode(pred_arr[i])  # (1, K, 2), (1, K)
        input_size = np.asarray(ds["meta"]["input_size"], dtype=np.float32)
        bbox_center = np.asarray(ds["meta"]["bbox_center"], dtype=np.float32)
        bbox_scale = np.asarray(ds["meta"]["bbox_scale"], dtype=np.float32)
        kpts = kpts / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
        keypoints[i] = kpts[0]
        scores[i] = scs[0]
    return keypoints, scores


def forward_pose_batched(
    model,
    image_bgr_list: list[np.ndarray],
    bboxes_list: list[np.ndarray],
    *,
    batch_size: int = 8,
    workers: int = 8,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-image batched top-down pose inference.

    For dataset-scale runs where each image only contributes 1-2 climber
    bboxes, the legacy :func:`forward_pose` (one image per call) leaves
    the 5B backbone idle most of the time. This variant accepts a *list*
    of images + per-image bbox arrays, builds all top-down crops on CPU,
    stacks them into one big tensor, and forwards in chunks of
    ``batch_size``. The backbone sees ``batch_size`` crops per forward
    instead of one.

    Args:
        model: built by :func:`load_sapiens2` with ``task="pose"``.
        image_bgr_list: ``N`` BGR ``uint8`` images (heterogeneous shapes
            OK — each bbox is decoded back into its own image's pixel
            space after the forward).
        bboxes_list: ``N`` arrays of shape ``(K_i, 4)`` xyxy float bboxes
            (per-image). An empty array is allowed (the image
            contributes nothing to the batch and gets an empty result).
        batch_size: crops per forward pass. 5B at 1024×768 bf16 is
            ~10-15 GB / forward including activations — 8 fits on a
            48-49 GB card comfortably.
        workers: threads for the CPU crop building and heatmap decoding.

    Returns:
        List of ``N`` ``(keypoints, scores)`` tuples in input order.
        ``keypoints``: ``(K_i, 308, 2)`` float32 in original-image pixels.
        ``scores``: ``(K_i, 308)`` float32 in ``[0, 1]``.
    """
    if len(image_bgr_list) != len(bboxes_list):
        raise ValueError(
            f"image/bbox list length mismatch: "
            f"{len(image_bgr_list)} vs {len(bboxes_list)}"
        )

    # Crop building (affine warp of the full image) and heatmap decoding
    # (308 CPU Gaussian blurs per crop) are the CPU halves of the call; both
    # release the GIL, so they run on a thread pool.
    def _crop(img_idx: int, bbox: np.ndarray) -> tuple[torch.Tensor, dict]:
        info = dict(img=image_bgr_list[img_idx])
        info["bbox"] = bbox[None]
        info["bbox_score"] = np.ones(1, dtype=np.float32)
        data = model.data_preprocessor(model.pipeline(info))
        return data["inputs"], data["data_samples"]

    jobs: list[tuple[int, np.ndarray]] = []
    for img_idx, bbs in enumerate(bboxes_list):
        for bbox in np.asarray(bbs, dtype=np.float32).reshape(-1, 4):
            jobs.append((img_idx, bbox))
    image_index = [img_idx for img_idx, _ in jobs]
    with ThreadPoolExecutor(workers) as pool:
        crops = list(pool.map(lambda job: _crop(*job), jobs))
    inputs_list = [inp for inp, _ in crops]
    samples_list = [ds for _, ds in crops]

    empty = (
        np.zeros((0, NUM_KEYPOINTS, 2), dtype=np.float32),
        np.zeros((0, NUM_KEYPOINTS), dtype=np.float32),
    )
    if not inputs_list:
        return [empty for _ in image_bgr_list]

    inputs_full = torch.cat(inputs_list, dim=0).to(dtype=_model_dtype(model))
    K = inputs_full.shape[0]

    val_cfg = getattr(model.cfg, "val_cfg", None)
    flip_test = (
        val_cfg is not None
        and (
            val_cfg.get("flip_test", False)
            if hasattr(val_cfg, "get")
            else getattr(val_cfg, "flip_test", False)
        )
    )
    flip_indices = (
        model.pose_metainfo["flip_indices"] if flip_test else None
    )

    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for s in range(0, K, batch_size):
            chunk = inputs_full[s : s + batch_size]
            p = model(chunk)
            if flip_test:
                pf = model(chunk.flip(-1))
                pf = pf.flip(-1)[:, flip_indices]
                p = (p + pf) / 2.0
            preds.append(p.float())
    pred_arr = torch.cat(preds, dim=0).cpu().numpy()

    def _decode(i: int) -> tuple[np.ndarray, np.ndarray]:
        ds = samples_list[i]
        kpts, scs = model.codec.decode(pred_arr[i])
        input_size = np.asarray(ds["meta"]["input_size"], dtype=np.float32)
        bbox_center = np.asarray(ds["meta"]["bbox_center"], dtype=np.float32)
        bbox_scale = np.asarray(ds["meta"]["bbox_scale"], dtype=np.float32)
        kpts = kpts / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
        return kpts[0], scs[0]

    buckets: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in image_bgr_list
    ]
    with ThreadPoolExecutor(workers) as pool:
        decoded = list(pool.map(_decode, range(K)))
    for i, item in enumerate(decoded):
        buckets[image_index[i]].append(item)

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for items in buckets:
        if not items:
            out.append(empty)
        else:
            ks = np.stack([k for k, _ in items], axis=0).astype(np.float32)
            ss = np.stack([s for _, s in items], axis=0).astype(np.float32)
            out.append((ks, ss))
    return out


def forward_dense(
    model,
    image_bgr: np.ndarray,
    *,
    head: str,
) -> np.ndarray:
    """Run pointmap or normal inference on one full image.

    Args:
        model: built by :func:`load_sapiens2` with the matching ``task``.
        image_bgr: ``(H, W, 3)`` ``uint8`` BGR.
        head: ``"pointmap"`` or ``"normal"``.

    Returns:
        ``(H, W, 3)`` ``float32`` array in the input's original resolution.
        Pointmap channels are ``(x, y, z)`` in metric camera coords; normal
        channels are unit-norm ``(nx, ny, nz)``.
    """
    if head not in ("pointmap", "normal"):
        raise ValueError(f"head must be 'pointmap' or 'normal', got {head!r}")
    H, W = image_bgr.shape[:2]
    data = model.pipeline(dict(img=image_bgr))
    data = model.data_preprocessor(data)
    inputs = data["inputs"].to(dtype=_model_dtype(model))
    samples = data["data_samples"]
    with torch.no_grad():
        if head == "pointmap":
            pointmap, scale = model(inputs)
            out = pointmap / scale  # convert to metric
        else:
            n = model(inputs)
            out = n / torch.norm(n, dim=1, keepdim=True).clamp(min=1e-8)
    pad_l, pad_r, pad_t, pad_b = samples["meta"]["padding_size"]
    out = out[:, :, pad_t : inputs.shape[2] - pad_b, pad_l : inputs.shape[3] - pad_r]
    out = F.interpolate(out.float(), size=(H, W), mode="bilinear", align_corners=False)
    return out.squeeze(0).cpu().numpy().transpose(1, 2, 0)


def forward_dense_batched(
    model,
    image_bgr_list: list[np.ndarray],
    *,
    head: str,
) -> list[np.ndarray]:
    """Cross-image batched pointmap / normal inference.

    Every image is preprocessed independently (the pipeline resizes +
    pads to the model's fixed input resolution), then all preprocessed
    tensors are stacked into one batch and forwarded in a single call.
    Unpadding + upsampling back to the original resolution is done
    per-sample after the forward.

    Args:
        model: built by :func:`load_sapiens2` with the matching ``task``.
        image_bgr_list: ``N`` BGR ``uint8`` images.
        head: ``"pointmap"`` or ``"normal"``.

    Returns:
        List of ``N`` ``(H_i, W_i, 3)`` float32 arrays at each image's
        original resolution. Pointmap channels are ``(x, y, z)`` in
        metric camera coords; normal channels are unit-norm.
    """
    if head not in ("pointmap", "normal"):
        raise ValueError(f"head must be 'pointmap' or 'normal', got {head!r}")
    if not image_bgr_list:
        return []

    inputs_list: list[torch.Tensor] = []
    samples_list: list[dict] = []
    original_sizes: list[tuple[int, int]] = []
    for img in image_bgr_list:
        H, W = img.shape[:2]
        data = model.pipeline(dict(img=img))
        data = model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        samples_list.append(data["data_samples"])
        original_sizes.append((H, W))

    inputs_full = torch.cat(inputs_list, dim=0).to(dtype=_model_dtype(model))
    with torch.no_grad():
        if head == "pointmap":
            pointmap, scale = model(inputs_full)
            # scale is (B, 1); reshape so it broadcasts per-sample
            # against pointmap's (B, 3, H, W).
            out = pointmap / scale.view(-1, 1, 1, 1)
        else:
            n = model(inputs_full)
            out = n / torch.norm(n, dim=1, keepdim=True).clamp(min=1e-8)

    results: list[np.ndarray] = []
    in_H = inputs_full.shape[2]
    in_W = inputs_full.shape[3]
    for i, (ds, (H, W)) in enumerate(zip(samples_list, original_sizes)):
        pad_l, pad_r, pad_t, pad_b = ds["meta"]["padding_size"]
        o = out[i : i + 1, :, pad_t : in_H - pad_b, pad_l : in_W - pad_r]
        o = F.interpolate(
            o.float(), size=(H, W), mode="bilinear", align_corners=False,
        )
        results.append(o.squeeze(0).cpu().numpy().transpose(1, 2, 0))
    return results


__all__ = [
    "NUM_KEYPOINTS",
    "Task",
    "forward_dense",
    "forward_dense_batched",
    "forward_pose",
    "forward_pose_batched",
    "load_sapiens2",
    "resolve_checkpoint",
]
