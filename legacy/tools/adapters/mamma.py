"""MammaNet (MAMMA ``ma_2d``) inference adapter — dense 2D SMPL-X surface landmarks.

Wraps the landmark detector from the ``cuevhv/mamma`` submodule
(``third_party/mamma``). MammaNet is a ViT-Base backbone + CNN mask encoder +
DETR-style transformer decoder that regresses **512 dense SMPL-X surface
landmarks** per person, each with a per-point **uncertainty** (sigma) and
**visibility** probability (the ``contact`` / ``floor_contact`` heads are
ignored here). We use only the *masked* variant
(``config_mammanet_mask_512.yaml`` + ``mamma_mask_full_cvpr.ckpt``): the bbox +
person mask come from our own SAM 3 tracker, so MAMMA's own detector /
segmenter are never built — this adapter has **no detectron2 dependency**.

Two public entry points:

* :func:`load_mammanet` — build ``DenseLdmks2DViT``, load the checkpoint, eval
  on ``device``. Adds ``third_party/mamma/landmarks`` to ``sys.path`` and
  registers the OmegaConf resolvers the config needs. Returns ``(model, cfg)``.

* :func:`forward_landmarks` — top-down inference for ``K`` person crops on one
  full image, sharing one full-frame mask. Returns landmarks in
  *original-image* pixels plus per-point visibility + uncertainty, unprojected
  exactly like ``landmarks/run_ma_2d.py``.

The heavy ``run_ma_2d`` entry point pulls in detectron2 + MAMMA's ``capture``
package at import time, so its tiny Hydra-style config loader is reimplemented
here (:func:`_load_hydra_style_config`) to keep the import surface minimal.
"""
from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
# tools/adapters/mamma.py → tools/adapters → tools → repo root
_REPO_ROOT = _HERE.parents[2]
_MAMMA_ROOT = _REPO_ROOT / "third_party" / "mamma"
_MAMMA_LANDMARKS = _MAMMA_ROOT / "landmarks"

#: Masked 512-landmark MammaNet config (the only variant we run).
DEFAULT_CONFIG = (
    _MAMMA_LANDMARKS / "configs/train/models_2d/config_mammanet_mask_512.yaml"
)
#: Trained checkpoint location (mirrors ``data/download_mamma_weights.sh``).
DEFAULT_CHECKPOINT = _MAMMA_ROOT / "data/weights/ma_2d/mamma_mask_full_cvpr.ckpt"

NUM_LANDMARKS = 512


# -----------------------------------------------------------------------------
# Import / config plumbing
# -----------------------------------------------------------------------------

def _ensure_mamma_on_path() -> None:
    """Put ``third_party/mamma/landmarks`` on ``sys.path`` (for ``lib`` / ``configs``)."""
    p = str(_MAMMA_LANDMARKS)
    if p not in sys.path:
        sys.path.insert(0, p)


@contextlib.contextmanager
def _chdir(target: Path):
    cwd = os.getcwd()
    os.chdir(str(target))
    try:
        yield
    finally:
        os.chdir(cwd)


def _register_resolvers() -> None:
    """Register the OmegaConf resolvers the MammaNet configs interpolate."""
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("mult", lambda x, y: x * y, replace=True)
    OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z, replace=True)
    OmegaConf.register_new_resolver("div", lambda x, y: x // y, replace=True)
    OmegaConf.register_new_resolver("concat", lambda x: np.concatenate(x), replace=True)
    OmegaConf.register_new_resolver("sorted", lambda x: np.argsort(x), replace=True)


def _sanitize_omegaconf_inplace(cfg) -> None:
    """Convert any embedded ``np.ndarray`` leaves to lists (so resolution works)."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(cfg, DictConfig):
        for k in list(cfg.keys()):
            v = cfg[k]
            if isinstance(v, (DictConfig, ListConfig)):
                _sanitize_omegaconf_inplace(v)
            elif isinstance(v, np.ndarray):
                cfg[k] = v.tolist()
    elif isinstance(cfg, ListConfig):
        for i in range(len(cfg)):
            v = cfg[i]
            if isinstance(v, (DictConfig, ListConfig)):
                _sanitize_omegaconf_inplace(v)
            elif isinstance(v, np.ndarray):
                cfg[i] = v.tolist()


def _load_hydra_style_config(cfg_file: str):
    """Load a MammaNet train config with its ``defaults:`` groups merged in.

    Reimplements ``landmarks/run_ma_2d.py::load_hydra_style_config`` (which we
    can't import without pulling detectron2 + ``capture``). Resolves the
    ``defaults`` list (``model`` / ``data`` / ``optimizer`` groups) relative to
    the config file's directory, merges, and fully resolves all interpolations.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cfg_file)
    base = OmegaConf.create()
    if "defaults" in cfg:
        for entry in cfg.defaults:
            if entry == "_self_":
                continue
            if isinstance(entry, Mapping):
                for group, name in entry.items():
                    path = os.path.join(os.path.dirname(cfg_file), group, f"{name}.yaml")
                    base[group] = OmegaConf.load(path)
            elif isinstance(entry, str):
                path = os.path.join(os.path.dirname(cfg_file), f"{entry}.yaml")
                base = OmegaConf.merge(base, OmegaConf.load(path))
    del cfg["defaults"]
    merged = OmegaConf.merge(base, cfg)
    _sanitize_omegaconf_inplace(merged)
    return OmegaConf.create(
        OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    )


def _model_dtype(model) -> torch.dtype:
    for p in model.parameters():
        return p.dtype
    return torch.float32


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_mammanet(
    *,
    config_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    device: str = "cuda",
):
    """Build a masked MammaNet (``DenseLdmks2DViT``) and load its checkpoint.

    Args:
        config_path: MammaNet train YAML; defaults to the masked 512-landmark
            config (:data:`DEFAULT_CONFIG`).
        checkpoint: trained ``.ckpt``; defaults to :data:`DEFAULT_CHECKPOINT`.
        device: torch device string.

    Returns:
        ``(model, cfg)`` — ``model`` is in eval mode on ``device`` (fp32);
        ``cfg`` is the fully-resolved OmegaConf config (carries
        ``num_joints``, crop ``image_size``, ``normalize_plus_min_one``).
    """
    _ensure_mamma_on_path()
    _register_resolvers()
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG
    ckpt = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    if not cfg_path.exists():
        raise FileNotFoundError(f"MammaNet config not found: {cfg_path}")
    if not ckpt.exists():
        raise FileNotFoundError(
            f"MammaNet checkpoint not found: {ckpt}\n"
            "Download it (MPI registration required) via "
            "`bash third_party/mamma/data/download_mamma_weights.sh --ckpt`, "
            "or pass --checkpoint."
        )

    from lib.models import build_model

    with _chdir(_MAMMA_LANDMARKS):
        cfg = _load_hydra_style_config(str(cfg_path))
        model = build_model(cfg).to(device)
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state)
        model.eval()
    return model, cfg


def bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Tight xyxy bbox of the nonzero region of a single-person mask, or ``None``.

    Mirrors ``run_ma_2d``'s masked path: the box that drives the top-down crop
    is the tight bound of the mask foreground.
    """
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def forward_landmarks(
    model,
    cfg,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bboxes_xyxy: np.ndarray,
    *,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MammaNet on ``K`` person crops of one image (sharing one mask).

    Args:
        model, cfg: from :func:`load_mammanet`.
        image_bgr: ``(H, W, 3)`` uint8 BGR full image.
        mask: ``(H, W)`` uint8 full-frame person mask (the masked variant feeds
            this to the model's mask encoder). May be ``None`` for the no-mask
            variant.
        bboxes_xyxy: ``(K, 4)`` xyxy float boxes in original-image pixels.
        device: torch device string.

    Returns:
        ``landmarks``: ``(K, 512, 2)`` float32 in original-image pixels.
        ``visibility``: ``(K, 512)`` float32 in ``[0, 1]`` (sigmoid of the head).
        ``uncertainty``: ``(K, 512)`` float32 in ``[0, 1]`` — the predicted
        per-point sigma, scaled like ``run_ma_2d``'s uncertainty visualization
        (0 = confident, 1 = uncertain).
    """
    _ensure_mamma_on_path()
    from lib.datasets.vitdet_dataset import ViTDetDataset

    boxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
    K = len(boxes)
    if K == 0:
        z2 = np.zeros((0, NUM_LANDMARKS, 2), np.float32)
        z1 = np.zeros((0, NUM_LANDMARKS), np.float32)
        return z2, z1, z1.copy()

    dataset = ViTDetDataset(cfg, image_bgr, mask, boxes=boxes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=K, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    dtype = _model_dtype(model)
    img = batch["img"].to(device=device, dtype=dtype)
    m = batch["mask"]
    m = m.to(device=device, dtype=dtype) if isinstance(m, torch.Tensor) else None

    with torch.no_grad():
        out = model(img, m)

    joints = out["joints2d"].float().cpu().numpy()                      # (K, 512, 3): x, y, sigma_raw
    vis = torch.sigmoid(out["visibility"].squeeze(-1)).float().cpu().numpy()  # (K, 512)
    box_center = batch["box_center"].float().cpu().numpy()              # (K, 2)
    box_size = batch["box_size"].float().cpu().numpy()                  # (K, 2): w, h

    # Map predicted xy from the model's output space to [0, 1] in the crop.
    normalize_scale = 2.0 if bool(cfg.data["train"]["normalize_plus_min_one"]) else 1.0
    xy = joints[:, :, :2].copy()
    if normalize_scale != 1.0:
        xy = (xy + 1.0) / normalize_scale

    # Unproject [0, 1] crop coords back to original-image pixels.
    bw, bh = box_size[:, 0:1], box_size[:, 1:2]
    cx, cy = box_center[:, 0:1], box_center[:, 1:2]
    landmarks = np.empty_like(xy)
    landmarks[:, :, 0] = xy[:, :, 0] * bw - bw / 2.0 + cx
    landmarks[:, :, 1] = xy[:, :, 1] * bh - bh / 2.0 + cy

    # Per-point uncertainty → [0, 1], matching run_ma_2d's sigma viz scaling:
    #   sigma_px = sqrt(exp(sigma_raw)) / normalize_scale * max(crop_h, crop_w)
    # then sigma_px / 50, clipped. (crop is image_size = [w, h].)
    crop_w, crop_h = int(cfg.data_cfg["image_size"][0]), int(cfg.data_cfg["image_size"][1])
    crop_max = max(crop_h, crop_w)
    sigma_raw = joints[:, :, 2]
    unc = np.sqrt(np.exp(sigma_raw)) / normalize_scale * crop_max
    unc = np.clip(unc / 50.0, 0.0, 1.0)

    return (
        landmarks.astype(np.float32),
        vis.astype(np.float32),
        unc.astype(np.float32),
    )


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_CONFIG",
    "NUM_LANDMARKS",
    "bbox_from_mask",
    "forward_landmarks",
    "load_mammanet",
]
