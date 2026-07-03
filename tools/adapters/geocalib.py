"""GeoCalib single-image calibration adapter — per-frame gravity (up) direction.

Stateless wrapper around cvg/GeoCalib (ECCV'24, ``third_party/geocalib``).
GeoCalib estimates the camera intrinsics **and** the gravity direction from a
single image. We use only the gravity: ``result["gravity"].vec3d`` is the **up**
(zenith) unit vector in the **camera** frame, in the OpenCV convention (+x right,
+y down, +z forward — the same frame VGGT uses), so an upright camera yields
``[0, -1, 0]`` and the physical gravity (down) direction is its negative. The
gravity stage transforms these per-frame camera-up vectors into the shared world
frame (via the VGGT extrinsics) and robustly combines them.

GeoCalib's per-pixel ``up_confidence`` (averaged over the frame) is the quality
signal that matters: on scenes without a clear horizon / architecture its
absolute value is small, but its **ranking** across frames cleanly separates the
frames GeoCalib can calibrate from the ones it can't — so the stage gates on it.

Used by ``scripts/stages/estimate_gravity_geocalib.py``. Mirrors the other
stateless adapters (``sapiens2``/``vggt_omega``): ``load_*`` once,
``infer_*`` per frame.
"""
from __future__ import annotations

import numpy as np
import torch

DEFAULT_WEIGHTS = "pinhole"  # GeoCalib pinhole checkpoint (no lens distortion)


def load_geocalib(device: str = "cuda", weights: str = DEFAULT_WEIGHTS):
    """Load the GeoCalib model (downloads the checkpoint via ``torch.hub`` once)."""
    from geocalib import GeoCalib  # third_party/geocalib (installed editable)

    return GeoCalib(weights=weights).to(device).eval()


@torch.no_grad()
def infer_gravity(model, rgb: np.ndarray, device: str = "cuda") -> dict:
    """Run GeoCalib on one RGB frame and return the camera-frame gravity (up).

    Args:
        model: a model from :func:`load_geocalib`.
        rgb: ``(H, W, 3)`` ``uint8`` RGB frame (GeoCalib resizes internally).

    Returns a dict:
        ``up_cam``     ``(3,)`` f64  camera-frame **up** (zenith) unit vector.
        ``confidence`` float          mean per-pixel up-field confidence (quality).
        ``roll``/``pitch`` float      gravity roll / pitch, radians.
        ``focal_px``   float          GeoCalib focal estimate, original-res px.
    """
    img = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().to(device) / 255.0
    res = model.calibrate(img, camera_model="pinhole")
    g = res["gravity"]
    return {
        "up_cam": g.vec3d[0].detach().cpu().numpy().astype(np.float64),
        "confidence": float(res["up_confidence"][0].mean().cpu()),
        "roll": float(g.roll[0].cpu()),
        "pitch": float(g.pitch[0].cpu()),
        "focal_px": float(res["camera"].f[0].mean().cpu()),
    }
