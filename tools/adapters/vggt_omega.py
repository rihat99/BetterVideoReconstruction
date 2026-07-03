"""VGGT-Omega adapter — multi-view camera + depth inference.

VGGT-Omega (``facebookresearch/vggt-omega``) is a *multi-view* model: given a
sequence of frames it predicts, in a single forward (cross-frame attention),
per-frame camera extrinsics + intrinsics and per-frame depth + confidence. It
backs ``scripts/stages/estimate_camera_vggt.py``, which runs right after the
tracker and writes a ``vggt/camera.npz`` contract.

Two things to know about the geometry it returns:

* **Intrinsics are pixel-unit at the model's *reduced* resolution.**
  ``encoding_to_camera`` builds ``K`` with ``cx = W/2``, ``cy = H/2`` and ``fx,
  fy`` in pixels of the resized image VGGT actually saw (e.g. 624x416 / 512x512),
  *not* the original frame. :func:`infer_sequence` therefore also returns the
  ``(fx/W, fy/H)`` **normalized** intrinsics (denormalize with
  ``tools.geometry.intrinsics.denormalize_K``) so the result is
  resolution-independent, plus the VGGT resolution itself so depth / point maps
  can be mapped back to full-res frames later.
* **Extrinsics are camera-from-world, OpenCV convention** (``[R | t]``),
  returned here padded to ``(N, 4, 4)``.
* **Geometry is up-to-scale, not metric.** Depth (``exp`` of the head logits)
  and camera translations share one coherent but arbitrary per-clip scale; there
  is no metric anchor in the package. Metricize downstream with the human-anchored
  ``estimate_scale`` stage.
* **Cameras are per-frame.** ``pose_enc`` is ``(B, N, 9)`` — each frame gets its
  own extrinsics *and* intrinsics (FOV).

Checkpoints are HF **access-gated** (``vggt_omega_1b_512.pt`` /
``vggt_omega_1b_256_text.pt``); :func:`load_vggt_omega` accepts a local ``.pt``
path or an HF repo id. See ``INSTALL.md``.

Upstream reference (project README)::

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    images = load_and_preprocess_images(paths, image_resolution=512).to("cuda")
    pred = model(images)
    extrinsics, intrinsics = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tools.geometry.intrinsics import normalize_K

DEFAULT_CHECKPOINT_REPO = "facebook/VGGT-Omega"
DEFAULT_CHECKPOINT_FILE = "vggt_omega_1b_512.pt"
DEFAULT_IMAGE_RESOLUTION = 512
DEFAULT_MODE = "balanced"
DEFAULT_MAX_FRAMES = 500

# Back-compat alias (the previous stub exported this name).
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_REPO


def uniform_subsample(n: int, max_frames: int | None) -> np.ndarray:
    """Indices of ``min(n, max_frames)`` uniformly-spaced frames (endpoints kept).

    ``max_frames`` falsy or ``<= 0`` means no cap. VGGT-Omega ingests the whole
    clip in one forward, so long clips must be subsampled to fit GPU memory.
    """
    if not max_frames or max_frames <= 0 or n <= max_frames:
        return np.arange(n, dtype=np.int64)
    idx = np.round(np.linspace(0, n - 1, num=max_frames)).astype(np.int64)
    return np.unique(idx)


def _resolve_checkpoint(checkpoint: str | None, *, filename: str = DEFAULT_CHECKPOINT_FILE) -> str:
    """Return a local ``.pt`` path. Accepts a local file, an HF repo id, or None."""
    if checkpoint and Path(checkpoint).is_file():
        return str(checkpoint)
    repo_id = checkpoint or DEFAULT_CHECKPOINT_REPO
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:  # noqa: BLE001 - re-raise with actionable guidance
        raise RuntimeError(
            f"Could not obtain VGGT-Omega checkpoint '{filename}' from '{repo_id}'. "
            "Pass --checkpoint <local .pt path>, or request access at "
            "https://huggingface.co/facebook/VGGT-Omega and authenticate "
            "(`huggingface-cli login`). "
            f"Underlying error: {type(e).__name__}: {e}"
        ) from e


def _unwrap_state_dict(obj):
    """Return a flat ``name -> tensor`` mapping from a possibly-wrapped checkpoint."""
    if isinstance(obj, dict) and not any(torch.is_tensor(v) for v in obj.values()):
        for key in ("model", "state_dict", "model_state_dict"):
            inner = obj.get(key)
            if isinstance(inner, dict):
                return inner
    return obj


def load_vggt_omega(
    checkpoint: str | None = None,
    *,
    device: str = "cuda",
    enable_alignment: bool = False,
):
    """Build :class:`VGGTOmega` and load weights.

    ``checkpoint`` may be a local ``.pt`` path or an HF repo id (``None`` →
    ``facebook/VGGT-Omega``). Use ``enable_alignment=True`` only with the
    256-resolution text-aligned checkpoint.
    """
    from vggt_omega.models import VGGTOmega

    ckpt_path = _resolve_checkpoint(checkpoint)
    model = VGGTOmega(enable_alignment=enable_alignment).eval()
    state = _unwrap_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.load_state_dict(state)
    return model.to(device)


def _to_homogeneous(extr_34: np.ndarray) -> np.ndarray:
    """``(N, 3, 4)`` ``[R | t]`` → ``(N, 4, 4)`` with a ``[0, 0, 0, 1]`` row."""
    n = extr_34.shape[0]
    out = np.tile(np.eye(4, dtype=extr_34.dtype), (n, 1, 1))
    out[:, :3, :4] = extr_34
    return out


def _fov_deg(focal_px: np.ndarray, size_px: int) -> np.ndarray:
    return np.degrees(2.0 * np.arctan((size_px / 2.0) / focal_px))


def infer_sequence(
    model,
    frame_paths,
    *,
    image_resolution: int = DEFAULT_IMAGE_RESOLUTION,
    mode: str = DEFAULT_MODE,
    device: str = "cuda",
    compute_depth: bool = True,
) -> dict:
    """Multi-view forward over a clip's ``frame_paths`` (already subsampled).

    Returns a dict with per-frame:

    * ``extrinsics``    ``(N, 4, 4)`` f32 — camera-from-world, OpenCV
    * ``intrinsics``    ``(N, 3, 3)`` f32 — *normalized* (``fx/W, fy/H``)
    * ``intrinsics_px`` ``(N, 3, 3)`` f32 — pixel units at the VGGT resolution
    * ``fov_x_deg`` / ``fov_y_deg`` ``(N,)`` f32
    * ``vggt_height`` / ``vggt_width`` — int, the resolution VGGT actually saw
    * ``depth`` ``(N, H, W)`` f32 and ``depth_conf`` ``(N, H, W)`` f32 — only
      when ``compute_depth`` and the model has a depth head.

    Geometry is up-to-scale — see the module docstring.
    """
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    paths = [str(p) for p in frame_paths]
    images = load_and_preprocess_images(paths, mode=mode, image_resolution=image_resolution).to(device)

    with torch.inference_mode():
        pred = model(images)

    vh, vw = int(pred["images"].shape[-2]), int(pred["images"].shape[-1])
    extr_34, intr_px = encoding_to_camera(pred["pose_enc"], (vh, vw))
    extr_34 = extr_34[0].float().cpu().numpy()  # (N, 3, 4)
    intr_px = intr_px[0].float().cpu().numpy()  # (N, 3, 3) pixel @ (vh, vw)

    out: dict = {
        "extrinsics": _to_homogeneous(extr_34).astype(np.float32),
        "intrinsics": normalize_K(intr_px, vh, vw).astype(np.float32),
        "intrinsics_px": intr_px.astype(np.float32),
        "fov_x_deg": _fov_deg(intr_px[:, 0, 0], vw).astype(np.float32),
        "fov_y_deg": _fov_deg(intr_px[:, 1, 1], vh).astype(np.float32),
        "vggt_height": vh,
        "vggt_width": vw,
    }
    if compute_depth and "depth" in pred:
        out["depth"] = pred["depth"][0, ..., 0].float().cpu().numpy()  # (N, H, W)
        if "depth_conf" in pred:
            out["depth_conf"] = pred["depth_conf"][0].float().cpu().numpy()  # (N, H, W)
    return out


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_CHECKPOINT_REPO",
    "DEFAULT_CHECKPOINT_FILE",
    "DEFAULT_IMAGE_RESOLUTION",
    "DEFAULT_MODE",
    "DEFAULT_MAX_FRAMES",
    "uniform_subsample",
    "load_vggt_omega",
    "infer_sequence",
]
