"""Binary mask I/O helpers shared by SAM3 and downstream consumers.

Three small utilities lifted out of the legacy Phy-SIC helper module so the
active pipeline doesn't depend on it:

* ``read_binary_mask`` — load a 0/255 PNG as a bool array.
* ``write_mask_png`` — save a bool/uint8 mask as a 0/255 PNG.
* ``union_obj_masks`` — OR per-object masks across the two layouts SAM3 emits
  (flat ``sam3/00.png`` for single-image runs, nested ``sam3/00/<frame>.png``
  for video).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_binary_mask(path: Path) -> np.ndarray | None:
    """Read a binary mask PNG; returns ``None`` if absent. Multi-channel inputs
    are reduced to channel 0."""
    if not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    return (m[..., 0] if m.ndim == 3 else m) > 0


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    """Save a bool/0-255 mask as 0/255 uint8 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    m = mask.astype(np.uint8)
    if m.max() <= 1:
        m = m * 255
    cv2.imwrite(str(path), m)


def union_obj_masks(
    masks_root: Path, frame_filename: str | None = None
) -> np.ndarray | None:
    """Union per-object masks under ``masks_root``.

    Two layouts are supported (matching :mod:`bvr.stages.track_sam3`):

    * Single-image (``frame_filename`` is ``None``): looks at every
      ``masks_root/<obj_id:02d>.png`` directly.
    * Multi-frame (``frame_filename`` given): looks at every
      ``masks_root/<obj_id>/<frame_filename>``.

    Returns ``None`` when nothing is found.
    """
    if not masks_root.is_dir():
        return None
    union: np.ndarray | None = None
    if frame_filename is None:
        for f in sorted(masks_root.glob("[0-9][0-9].png")):
            m = read_binary_mask(f)
            if m is not None:
                union = m if union is None else (union | m)
    else:
        for od in sorted(masks_root.iterdir()):
            if not od.is_dir():
                continue
            m = read_binary_mask(od / frame_filename)
            if m is not None:
                union = m if union is None else (union | m)
    return union


__all__ = ["read_binary_mask", "write_mask_png", "union_obj_masks"]
