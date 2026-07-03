"""MoGe2 metric-depth + camera-intrinsics adapter.

Thin wrapper around ``moge.model.v2.MoGeModel``: load once, run per-frame
inference, recover camera intrinsics. MoGe2 returns *normalized* intrinsics
(``intrinsics[0, 0] = fx / W`` and ``intrinsics[1, 1] = fy / H``); the helpers
below convert between that, a horizontal field of view, and a pixel-unit ``K``.

This is the camera-parameter source for the video pipeline (see
``scripts/stages/estimate_camera_moge2.py``). A future VGGT-Omega stage is
expected to produce the same intrinsics contract from a multi-view solve.
"""
from __future__ import annotations

import numpy as np
import torch

DEFAULT_MODEL = "Ruicheng/moge-2-vitl-normal"


def load_moge2(model_id: str = DEFAULT_MODEL, *, device: str = "cuda"):
    """Load a MoGe2 model in eval mode on ``device``."""
    from moge.model.v2 import MoGeModel

    return MoGeModel.from_pretrained(model_id).to(device).eval()


def fov_x_deg_from_intrinsics(intrinsics: np.ndarray) -> float:
    """Horizontal FOV (degrees) from MoGe2 normalized intrinsics.

    ``intrinsics[0, 0] == fx / W``, so ``tan(fov_x / 2) = 1 / (2 * fx_n)``.
    """
    fx_n = float(intrinsics[0, 0])
    return float(np.degrees(2.0 * np.arctan(1.0 / (2.0 * fx_n))))


def denormalize_K(intrinsics: np.ndarray, height: int, width: int) -> np.ndarray:
    """MoGe2 normalized intrinsics → pixel-unit ``K`` (3, 3) float32.

    ``fx, cx`` scale by image width; ``fy, cy`` by image height. This matches
    how the body stage consumes camera params (``cam_int`` in pixels).
    """
    K = np.asarray(intrinsics, dtype=np.float32).copy()
    K[0, 0] *= width
    K[0, 2] *= width
    K[1, 1] *= height
    K[1, 2] *= height
    return K


def infer_frame(
    model,
    rgb_uint8: np.ndarray,
    *,
    fov_x_deg: float | None = None,
    use_fp16: bool = True,
    edge_rtol: float = 0.04,
    edge_normal_tol: float = 0.0,
    device: str = "cuda",
) -> dict:
    """Run MoGe2 on one ``(H, W, 3)`` uint8 **RGB** frame.

    Pass ``fov_x_deg=None`` to let MoGe2's FOV head estimate intrinsics
    (the camera-params use case); pass a value to pin the horizontal FOV.

    Returns a dict with ``points (H, W, 3)``, ``depth (H, W)``,
    ``intrinsics (3, 3)`` (MoGe2 normalized convention, float32), and
    ``mask`` / ``normals`` when the model returns them. The depth-edge filter
    (MoGe's ``infer.py`` default) drops flying boundary pixels from ``mask``.
    """
    img = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(2, 0, 1).float() / 255.0
    img = img.to(device)

    kwargs: dict = {"use_fp16": use_fp16}
    if fov_x_deg is not None:
        kwargs["fov_x"] = fov_x_deg
    out = model.infer(img, **kwargs)

    points = out["points"].detach().cpu().numpy()
    depth = out["depth"].detach().cpu().numpy()
    intrinsics = out["intrinsics"].detach().cpu().numpy().astype(np.float32)
    mask = out["mask"].detach().cpu().numpy().astype(bool) if "mask" in out else None
    normal = out.get("normal")
    normal_np = normal.detach().cpu().numpy() if normal is not None else None

    if edge_rtol > 0:
        import utils3d

        edge = utils3d.np.depth_map_edge(depth, rtol=edge_rtol, mask=mask)
        if edge_normal_tol > 0 and normal_np is not None:
            edge = edge & utils3d.np.normal_map_edge(normal_np, tol=edge_normal_tol, mask=mask)
        mask = (~edge) if mask is None else (mask & ~edge)

    result: dict = {"points": points, "depth": depth, "intrinsics": intrinsics}
    if mask is not None:
        result["mask"] = mask
    if normal_np is not None:
        result["normals"] = normal_np
    return result


__all__ = [
    "DEFAULT_MODEL",
    "load_moge2",
    "fov_x_deg_from_intrinsics",
    "denormalize_K",
    "infer_frame",
]
