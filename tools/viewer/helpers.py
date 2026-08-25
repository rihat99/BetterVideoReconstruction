"""Small pure geometry + colour helpers shared across the viewer package.

Numpy-only (``viser.transforms`` is imported lazily inside the one helper that
needs it), so this is the viewer's dependency-free base layer, imported by both
:mod:`tools.viewer.loading` and :mod:`tools.viewer.scene`.
"""
from __future__ import annotations

import numpy as np

# Per-person body colours (RGB 0-255), matching the stage overlay palette so a
# person keeps one colour across the body/SMPL-X overlays and this viewer.
_BODY_COLORS = (
    (166, 189, 219),  # light blue
    (242, 153, 102),  # salmon
    (153, 217, 140),  # light green
    (204, 166, 224),  # lilac
    (242, 217, 115),  # gold
)

# Viridis control points (RGB) for colouring the camera trajectory by time.
_VIRIDIS = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140),
                     (94, 201, 98), (253, 231, 37)], np.float64)


def _camera_world_pose(extrinsic: np.ndarray):
    """cam-from-world OpenCV ``[R|t]`` → ``(world position, world-from-cam wxyz)``."""
    import viser.transforms as vt

    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    return -R.T @ t, vt.SO3.from_matrix(R.T).wxyz


def _camera_follow_target(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OpenCV cam-from-world ``[R|t]`` → viewer camera ``(position, look_at, up)`` in world.

    For an "onboard" view that matches the reconstructed camera: the eye is the
    camera centre ``C = -Rᵀt``; the optical axis (OpenCV +z) and up (OpenCV −y,
    since image y points down) map to the world frame via ``Rᵀ``. ``look_at`` is one
    unit ahead of ``C`` along the optical axis, so feeding these to
    :attr:`viser.CameraHandle.position` / ``look_at`` / ``up_direction`` orients the
    viewer exactly like the source camera (match its vertical FOV separately).
    """
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    C = -R.T @ t
    forward = R.T @ np.array([0.0, 0.0, 1.0])
    up = R.T @ np.array([0.0, -1.0, 0.0])
    return C, C + forward, up


def _unit(v) -> np.ndarray:
    """Normalize a vector; return it unchanged if its norm is ~0."""
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _time_color(t: float) -> tuple[int, int, int]:
    """Sample the viridis ramp at ``t`` ∈ [0, 1] → an ``(r, g, b)`` uint tuple."""
    x = float(np.clip(t, 0.0, 1.0)) * (len(_VIRIDIS) - 1)
    i = min(int(x), len(_VIRIDIS) - 2)
    c = _VIRIDIS[i] * (1.0 - (x - i)) + _VIRIDIS[i + 1] * (x - i)
    return int(c[0]), int(c[1]), int(c[2])


def _wxyz_shortest_arc_z(d: np.ndarray) -> tuple[float, float, float, float]:
    """Shortest-arc quaternion (wxyz) rotating the unit cylinder's +z axis onto ``d``."""
    d = np.asarray(d, np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    d = d / n
    c = float(d[2])  # cos angle = z·d
    if c > 0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    if c < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)  # antiparallel → 180° about x
    v = np.array([-d[1], d[0], 0.0])  # z × d
    s = np.sqrt((1.0 + c) * 2.0)
    q = np.array([s * 0.5, v[0] / s, v[1] / s, v[2] / s])
    q /= np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _top4_skin_weights(lbs_w: np.ndarray) -> np.ndarray:
    """Top-4 per-vertex skin weights renormalized to sum 1 (``V, J``).

    viser keeps only each vertex's top-4 bone weights and does **not**
    renormalize; SMPL-X spreads weight over ~8 bones, so the dropped tail would
    leave blend-region vertices summing to <1 and — because we bake metre-scale
    world translation into the bones — fling them out as spikes. Pre-restricting
    to the top 4 and renormalizing matches what viser uses.
    """
    rows = np.arange(lbs_w.shape[0])[:, None]
    keep = np.argsort(lbs_w, axis=1)[:, -4:]
    w = np.zeros_like(lbs_w)
    w[rows, keep] = lbs_w[rows, keep]
    return (w / w.sum(1, keepdims=True)).astype(np.float32)


def _background_image() -> np.ndarray:
    """A solid white backdrop, set full-canvas via
    :meth:`viser.SceneApi.set_background_image`."""
    return np.full((2, 2, 3), 255, np.uint8)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB→linear on ``[0, 1]`` floats (standard IEC 61966-2-1 transfer)."""
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _unlit_glb(vertices: np.ndarray, faces: np.ndarray, colors_srgb: np.ndarray) -> bytes:
    """Export an **unlit**, vertex-coloured GLB so a surface mesh matches the point cloud.

    viser's point cloud is drawn by a custom shader that writes the stored colours
    straight to the framebuffer — unlit and un-tone-mapped. A mesh instead goes
    through three.js's default lit ``MeshStandardMaterial`` *and* the renderer's
    ACES tone-map + sRGB output-encode, which is what makes it look dim and washed
    out next to the cloud. Two steps close most of that gap from the Python side:

    * flag the material ``KHR_materials_unlit`` → three.js renders an unlit
      ``MeshBasicMaterial`` (no lighting term), and
    * pre-**linearize** the colours into ``COLOR_0``: three.js reads glTF vertex
      colours as linear and re-encodes to sRGB on output, so storing
      ``sRGB→linear`` values makes that encode restore the original stored colour.

    (A small residual remains: the renderer's ACES tone-map still applies to the
    mesh but not to the point shader, and can't be disabled through glTF — so the
    surface sits a touch darker than the cloud, far closer than the lit default.)

    :param vertices: ``(V, 3)`` mesh vertices.
    :param faces: ``(F, 3)`` triangle indices.
    :param colors_srgb: ``(V, 3)`` ``uint8`` sRGB per-vertex colours.
    :returns: GLB bytes for :meth:`viser.SceneApi.add_glb`.
    """
    import io
    import json
    import struct

    import trimesh
    from trimesh.visual.material import PBRMaterial

    lin = np.clip(_srgb_to_linear(np.asarray(colors_srgb, np.float64) / 255.0), 0.0, 1.0)
    vc = np.empty((len(lin), 4), np.uint8)
    vc[:, :3] = np.round(lin * 255.0).astype(np.uint8)
    vc[:, 3] = 255
    mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vc, process=False)
    mesh.visual.material = PBRMaterial(doubleSided=True, metallicFactor=0.0, roughnessFactor=1.0)
    with io.BytesIO() as buf:
        mesh.export(buf, file_type="glb")
        glb = buf.getvalue()

    # A GLB is a 12-byte header then length-prefixed chunks; the first is the JSON
    # document. Patch it to declare KHR_materials_unlit on every material, re-pad the
    # chunk to a 4-byte boundary, and fix the total-length field. The BIN chunk (the
    # tail) is untouched, so accessor byte-offsets stay valid.
    json_len = struct.unpack("<I", glb[12:16])[0]
    doc = json.loads(glb[20:20 + json_len])
    tail = glb[20 + json_len:]
    used = doc.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in used:
        used.append("KHR_materials_unlit")
    for mat in doc.get("materials", []):
        mat.setdefault("extensions", {})["KHR_materials_unlit"] = {}
    new_json = json.dumps(doc, separators=(",", ":")).encode()
    new_json += b" " * ((4 - len(new_json) % 4) % 4)
    out = bytearray(glb[:12] + struct.pack("<II", len(new_json), 0x4E4F534A) + new_json + tail)
    struct.pack_into("<I", out, 8, len(out))
    return bytes(out)


