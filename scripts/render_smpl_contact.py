"""Render SMPL mesh or skeleton overlay with contact regions highlighted in red.

Loads `smpl_params.npz` (output of `convert_mhr_to_smpl.py`) and a contact-
prediction `.npz` (e.g. `interactvlm_hcontact_vertices.npz`), then composites
onto the source video either:

* ``--render mesh`` (default): the full SMPL mesh with per-vertex contact
  colouring — vertices whose confidence exceeds ``--contact-threshold`` turn
  red instead of the default light-blue mesh colour.
* ``--render skeleton``: a 2D stick figure — a circle per SMPL joint with
  lines between each joint and its parent. Per-vertex contacts are aggregated
  to per-joint (max over each joint's vertices via SMPL LBS argmax), and
  contact joints are drawn red.

Run from CLI:

    uv run python scripts/render_smpl_contact.py \\
        --source path/to/video.mp4 \\
        --predictions out/video/ \\
        --contacts out/video/interactvlm_hcontact_vertices.npz \\
        [--render mesh|skeleton] \\
        [--contact-key pred_contact_3d_smplh] [--contact-threshold 0.5] \\
        [--smpl-model /home/rikhat/better/better_human/models/smpl/SMPL_NEUTRAL.npz] \\
        [--device cuda]

Inputs:
    source             video file or folder of frames
    predictions_dir    folder containing smpl_params.npz
    contacts           .npz with per-vertex contact confidence

Outputs (into <predictions_dir>/):
    <source-stem>_smpl_contact.mp4            mesh overlay (mesh mode)
    <source-stem>_smpl_contact_skeleton.mp4   skeleton overlay (skeleton mode)

The contact .npz must contain a (M, 6890) float array under `contact_key`
(default `pred_contact_3d_smplh`); if it also contains `frame_indices`
(M,), those map each row to a video frame, otherwise rows are taken as
sequential frames 0..M-1. Frames absent from the mapping render with the
default mesh color (no red). Duplicate frame_indices entries: the last
occurrence wins.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This script lives in scripts/; add it to sys.path so `import _pym_preload` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pym_preload  # noqa: E402, F401  (must be first — re-execs to fix libstdc++ ABI)

import argparse
import os
from dataclasses import dataclass

# Match sam_3d_body.visualization.renderer: pick EGL for headless rendering
# unless the caller already chose a platform.
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import numpy as np
import pyrender
import torch
import trimesh
from tqdm import tqdm


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SMPL_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smpl" / "SMPL_NEUTRAL.npz"
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".MP4", ".MOV"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

# Same default mesh color the SAM-3D-Body / convert_mhr_to_smpl scripts use.
_MESH_COLOR = (0.65098039, 0.74117647, 0.85882353)
_CONTACT_COLOR = (1.0, 0.0, 0.0)

_SMPL_NUM_VERTS = 6890
_SMPL_NUM_JOINTS = 24
# SMPL parent indices (matches better_robot.io.builders.PARENTS).
_SMPL_PARENTS: tuple[int, ...] = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12,
    13, 14, 16, 17, 18, 19, 20, 21,
)
# Bone color for the 2D skeleton overlay (BGR, cv2 convention).
_BONE_COLOR_BGR: tuple[int, int, int] = (200, 200, 200)


def _rgb_to_bgr_255(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    """(R, G, B) in [0, 1] → (B, G, R) in [0, 255] for cv2 drawing."""
    r, g, b = rgb
    return int(b * 255), int(g * 255), int(r * 255)


def _list_frame_paths(folder: Path) -> list[Path]:
    paths = [p for p in folder.iterdir() if p.suffix in IMG_EXTS]
    try:
        paths.sort(key=lambda p: int(p.stem))
    except ValueError:
        paths.sort()
    if not paths:
        raise RuntimeError(f"no image frames found in {folder}")
    return paths


def _open_video(video_path: Path) -> tuple[cv2.VideoCapture, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, n


def _add_sys_paths() -> None:
    """Ensure better_human (editable src layout) and tools/ are importable."""
    better_human_src = _REPO_ROOT.parent / "better_human" / "src"
    if better_human_src.exists():
        sys.path.insert(0, str(better_human_src))
    sys.path.insert(0, str(_REPO_ROOT))


def _create_raymond_lights(intensity: float = 0.3) -> list[pyrender.Node]:
    """Three Raymond directional lights — copied from sam_3d_body renderer.

    Default intensity is dialled down from the sam_3d_body default of 1.0 so
    per-vertex chroma is not washed out to near-white.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes: list[pyrender.Node] = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                matrix=matrix,
            )
        )
    return nodes


def _render_with_vertex_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors_rgba: np.ndarray,
    bgr_image: np.ndarray,
    focal_length: float,
    scene_bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Composite a per-vertex-colored SMPL mesh over a BGR frame.

    Mirrors `sam_3d_body.visualization.renderer.Renderer.__call__` but uses
    vertex colors baked into the trimesh instead of a uniform material.
    Vertices are expected to already be in the camera frame.
    """
    image = bgr_image.astype(np.float32) / 255.0
    h, w = image.shape[:2]

    renderer = pyrender.OffscreenRenderer(viewport_height=h, viewport_width=w)
    try:
        mesh_tm = trimesh.Trimesh(
            vertices.copy(),
            faces.copy(),
            vertex_colors=vertex_colors_rgba,
            process=False,
        )
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh_tm.apply_transform(rot)

        # IMPORTANT: do NOT pass `material=...` here. pyrender's
        # `Mesh._get_trimesh_props` only extracts vertex_colors when
        # `material is None`, so providing an explicit material silently
        # discards the per-vertex coloring. Instead, let pyrender build the
        # default vertex-color material, then tighten metallic / alpha to
        # match the matte opaque look used by Renderer.__call__.
        mesh = pyrender.Mesh.from_trimesh(mesh_tm)
        prim_material = mesh.primitives[0].material
        prim_material.metallicFactor = 0.0
        prim_material.roughnessFactor = 1.0
        prim_material.alphaMode = "OPAQUE"

        scene = pyrender.Scene(
            bg_color=[*scene_bg_color, 0.0], ambient_light=(0.6, 0.6, 0.6)
        )
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4)
        camera_center = [w / 2.0, h / 2.0]
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length,
            fy=focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12,
        )
        scene.add(camera, pose=camera_pose)

        for node in _create_raymond_lights():
            scene.add_node(node)

        color, _depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    color = color.astype(np.float32) / 255.0
    valid_mask = color[:, :, -1:]
    output = color[:, :, :3] * valid_mask + (1.0 - valid_mask) * image
    return (output * 255.0).astype(np.uint8)


def _render_skeleton_overlay(
    joints_cam: np.ndarray,
    parents: tuple[int, ...],
    joint_in_contact: np.ndarray,
    bgr_image: np.ndarray,
    focal_length: float,
    *,
    joint_radius: int = 8,
    line_thickness: int = 3,
) -> np.ndarray:
    """Draw a 2D skeleton overlay (circles at joints, lines between parents)
    on a BGR frame.

    ``joints_cam`` holds camera-frame 3D joint positions in OpenCV convention
    (+X right, +Y down, +Z forward). Projection uses a pinhole camera with
    ``fx=fy=focal_length`` and principal point at the image center — matching
    what ``_render_with_vertex_colors`` does for the pyrender camera.

    Joints where ``joint_in_contact[j]`` is True are filled with
    ``_CONTACT_COLOR`` (red); all others use ``_MESH_COLOR``. Bone lines use
    ``_BONE_COLOR_BGR``.
    """
    h, w = bgr_image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    zs = joints_cam[:, 2]
    zs_safe = np.where(np.abs(zs) < 1e-6, 1e-6, zs)
    us = (focal_length * joints_cam[:, 0] / zs_safe + cx).astype(np.int32)
    vs = (focal_length * joints_cam[:, 1] / zs_safe + cy).astype(np.int32)

    out = bgr_image.copy()

    joint_color_bgr = _rgb_to_bgr_255(_MESH_COLOR)
    contact_color_bgr = _rgb_to_bgr_255(_CONTACT_COLOR)

    # Bones first so joint circles sit on top.
    for j in range(len(parents)):
        p = parents[j]
        if p < 0:
            continue
        pt_c = (int(us[j]), int(vs[j]))
        pt_p = (int(us[p]), int(vs[p]))
        cv2.line(out, pt_c, pt_p, _BONE_COLOR_BGR, line_thickness, lineType=cv2.LINE_AA)

    # Circles with a thin black outline for legibility against busy backgrounds.
    for j in range(len(parents)):
        pt = (int(us[j]), int(vs[j]))
        color = contact_color_bgr if bool(joint_in_contact[j]) else joint_color_bgr
        cv2.circle(out, pt, joint_radius, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(out, pt, joint_radius, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    return out


@dataclass
class RenderResult:
    num_frames: int
    num_rendered: int
    num_with_contact: int
    output_path: Path


class RenderSmplContactScript:
    def __init__(
        self,
        device: str = "cuda",
        smpl_model_path: str | Path | None = None,
        batch_size: int = 64,
        contact_threshold: float = 0.5,
        render_mode: str = "mesh",
    ) -> None:
        if render_mode not in ("mesh", "skeleton"):
            raise ValueError(
                f"render_mode must be 'mesh' or 'skeleton', got {render_mode!r}"
            )
        self.device = device
        self.smpl_model_path = Path(smpl_model_path or _DEFAULT_SMPL_MODEL)
        self.batch_size = batch_size
        self.contact_threshold = contact_threshold
        self.render_mode = render_mode
        self._smpl = None

    def _build_smpl(self):
        _add_sys_paths()
        from better_human.smpl import SMPL

        if self._smpl is None:
            self._smpl = SMPL(
                str(self.smpl_model_path), gender="neutral", num_betas=10
            ).to(self.device)
        return self._smpl

    # -- main entry point ----------------------------------------------------

    def run(
        self,
        source: str | Path,
        predictions_dir: str | Path,
        contact_npz: str | Path,
        contact_key: str = "pred_contact_3d_smplh",
        video_fps: float | None = None,
    ) -> RenderResult:
        source = Path(source)
        predictions_dir = Path(predictions_dir)
        contact_npz = Path(contact_npz)

        smpl_path = predictions_dir / "smpl_params.npz"
        if not smpl_path.exists():
            raise FileNotFoundError(smpl_path)
        sp = np.load(smpl_path)

        betas = sp["betas"]  # (N, 10)
        body_pose = sp["body_pose"]  # (N, 69)
        global_orient = sp["global_orient"]  # (N, 3)
        transl = sp["transl"]  # (N, 3)
        cam_intrinsics = sp["cam_intrinsics"]  # (3, 3)
        valid_mask = sp["valid_mask"].astype(bool)  # (N,)
        N = int(sp["num_frames"])
        H = int(sp["video_height"])
        W = int(sp["video_width"])

        if not contact_npz.exists():
            raise FileNotFoundError(contact_npz)
        cd = np.load(contact_npz)
        if contact_key not in cd.files:
            raise KeyError(
                f"key {contact_key!r} not in {contact_npz} "
                f"(available: {list(cd.files)})"
            )
        conf = cd[contact_key]  # (M, V)
        if conf.ndim != 2 or conf.shape[1] != _SMPL_NUM_VERTS:
            raise ValueError(
                f"{contact_key} has shape {conf.shape}; expected "
                f"(M, {_SMPL_NUM_VERTS}). For SMPL-X (10475) contacts, convert "
                f"to SMPL first or pass --contact-key pred_contact_3d_smplh"
            )
        if "frame_indices" in cd.files:
            fidx = cd["frame_indices"].astype(np.int64)
        else:
            fidx = np.arange(conf.shape[0], dtype=np.int64)
        if fidx.shape[0] != conf.shape[0]:
            raise ValueError(
                f"frame_indices length {fidx.shape[0]} != "
                f"{contact_key} rows {conf.shape[0]}"
            )
        contact_by_frame: dict[int, np.ndarray] = {
            int(f): conf[k] for k, f in enumerate(fidx)
        }

        smpl = self._build_smpl()
        faces = smpl.faces.detach().cpu().numpy().astype(np.int32)

        # Source: video file or folder of frames
        cap = None
        frame_paths: list[Path] | None = None
        src_fps: float | None = None
        if source.suffix in VIDEO_EXTS:
            cap, src_fps, _ = _open_video(source)
        elif source.is_dir():
            frame_paths = _list_frame_paths(source)
        else:
            raise ValueError(f"source must be a video file or folder: {source}")

        suffix = "_smpl_contact" if self.render_mode == "mesh" else "_smpl_contact_skeleton"
        out_path = predictions_dir / f"{source.stem}{suffix}.mp4"
        fps = float(video_fps or src_fps or 30.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        if not writer.isOpened():
            if cap is not None:
                cap.release()
            raise RuntimeError(f"cannot open video writer at {out_path}")

        # Pre-compute SMPL vertices and joint positions in camera frame for
        # all valid frames. Joint positions come from SMPLOutputs.joints_world
        # (a (B, J, 7) SE3 tensor); we keep just the translation block.
        valid_idx = np.flatnonzero(valid_mask)
        smpl_verts_cam = np.full((N, _SMPL_NUM_VERTS, 3), np.nan, dtype=np.float32)
        smpl_joints_cam = np.full((N, _SMPL_NUM_JOINTS, 3), np.nan, dtype=np.float32)
        with torch.no_grad():
            dev = smpl.vertices_template.device
            betas_t = torch.from_numpy(betas[valid_idx].astype(np.float32)).to(dev)
            body_pose_t = torch.from_numpy(
                body_pose[valid_idx].astype(np.float32)
            ).to(dev)
            global_orient_t = torch.from_numpy(
                global_orient[valid_idx].astype(np.float32)
            ).to(dev)
            transl_t = torch.from_numpy(transl[valid_idx].astype(np.float32)).to(dev)
            bs = self.batch_size
            for s in range(0, valid_idx.size, bs):
                e = min(s + bs, valid_idx.size)
                q = smpl.from_classic(
                    betas_t[s:e],
                    body_pose_t[s:e],
                    transl_t[s:e],
                    global_orient_t[s:e],
                )
                out = smpl(betas=betas_t[s:e], q=q)
                smpl_verts_cam[valid_idx[s:e]] = out.vertices.cpu().numpy()
                smpl_joints_cam[valid_idx[s:e]] = (
                    out.joints_world.tensor()[..., :3].cpu().numpy()
                )

        # Per-joint contact preprocessing (only needed in skeleton mode).
        # Reuses tools.robot_motion.contact so the vertex→joint logic is
        # shared with build_robot_motion.py.
        joint_contact_by_frame: dict[int, np.ndarray] = {}
        if self.render_mode == "skeleton":
            from tools.robot_motion.contact import (
                aggregate_vertex_contacts_to_joints,
                smpl_vertex_to_joint,
            )

            vert_to_joint = smpl_vertex_to_joint(smpl)
            per_frame_joint_conf = aggregate_vertex_contacts_to_joints(
                conf, vert_to_joint, num_joints=_SMPL_NUM_JOINTS, reduce="max",
            )  # (M, 24)
            joint_contact_by_frame = {
                int(f): per_frame_joint_conf[k] for k, f in enumerate(fidx)
            }

        fx = float(cam_intrinsics[0, 0])

        def _read_frame(i: int) -> np.ndarray | None:
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, bgr = cap.read()
                return bgr if ok else None
            return cv2.imread(str(frame_paths[i]))

        num_rendered = 0
        num_with_contact = 0
        desc = (
            "Rendering SMPL contact overlay"
            if self.render_mode == "mesh"
            else "Rendering SMPL skeleton+contact overlay"
        )
        try:
            for i in tqdm(range(N), desc=desc):
                bgr = _read_frame(i)
                if bgr is None:
                    continue
                if not valid_mask[i] or np.isnan(smpl_verts_cam[i]).any():
                    writer.write(bgr)
                    continue

                if self.render_mode == "mesh":
                    colors_rgb = np.tile(_MESH_COLOR, (_SMPL_NUM_VERTS, 1)).astype(
                        np.float32
                    )
                    if i in contact_by_frame:
                        mask = contact_by_frame[i] > self.contact_threshold
                        if mask.any():
                            colors_rgb[mask] = _CONTACT_COLOR
                            num_with_contact += 1

                    # R/B swap to match Renderer.__call__'s BGR-against-BGR convention.
                    colors_bgr = colors_rgb[:, ::-1]
                    vertex_colors_rgba = np.concatenate(
                        [colors_bgr, np.ones((_SMPL_NUM_VERTS, 1), dtype=np.float32)],
                        axis=1,
                    )

                    overlay = _render_with_vertex_colors(
                        vertices=smpl_verts_cam[i].astype(np.float32),
                        faces=faces,
                        vertex_colors_rgba=vertex_colors_rgba,
                        bgr_image=bgr,
                        focal_length=fx,
                    )
                else:
                    joint_in_contact = np.zeros(_SMPL_NUM_JOINTS, dtype=bool)
                    if i in joint_contact_by_frame:
                        joint_in_contact = (
                            joint_contact_by_frame[i] > self.contact_threshold
                        )
                        if joint_in_contact.any():
                            num_with_contact += 1

                    overlay = _render_skeleton_overlay(
                        joints_cam=smpl_joints_cam[i].astype(np.float32),
                        parents=_SMPL_PARENTS,
                        joint_in_contact=joint_in_contact,
                        bgr_image=bgr,
                        focal_length=fx,
                    )

                writer.write(overlay)
                num_rendered += 1
        finally:
            writer.release()
            if cap is not None:
                cap.release()

        return RenderResult(
            num_frames=N,
            num_rendered=num_rendered,
            num_with_contact=num_with_contact,
            output_path=out_path,
        )


# ------------------------------ CLI -------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render SMPL mesh overlay with contact vertices in red.",
    )
    p.add_argument(
        "--source", required=True,
        help="video file or folder of frames (same source used upstream)",
    )
    p.add_argument(
        "--predictions", required=True,
        help="folder containing smpl_params.npz (from convert_mhr_to_smpl.py)",
    )
    p.add_argument(
        "--contacts", required=True,
        help="path to contacts .npz (e.g. interactvlm_hcontact_vertices.npz)",
    )
    p.add_argument(
        "--contact-key", default="pred_contact_3d_smplh",
        help="key in contacts .npz holding per-vertex confidence (default: %(default)s)",
    )
    p.add_argument(
        "--contact-threshold", type=float, default=0.5,
        help="confidence threshold for marking a vertex as in contact (default: %(default)s)",
    )
    p.add_argument(
        "--render", choices=["mesh", "skeleton"], default="mesh",
        help="output style: 'mesh' renders the full SMPL mesh with per-vertex "
             "contact colouring; 'skeleton' draws circles at joint positions "
             "connected by bone lines, with joint-aggregated contact colouring "
             "(default: %(default)s)",
    )
    p.add_argument(
        "--smpl-model", default=None,
        help=f"path to better_human SMPL .npz (default: {_DEFAULT_SMPL_MODEL})",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--fps", type=float, default=None,
        help="output video fps (default: source video fps, or 30 for folders)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    script = RenderSmplContactScript(
        device=args.device,
        smpl_model_path=args.smpl_model,
        batch_size=args.batch_size,
        contact_threshold=args.contact_threshold,
        render_mode=args.render,
    )
    result = script.run(
        source=args.source,
        predictions_dir=args.predictions,
        contact_npz=args.contacts,
        contact_key=args.contact_key,
        video_fps=args.fps,
    )
    print(
        f"rendered {result.num_rendered}/{result.num_frames} frames "
        f"({result.num_with_contact} with contact above "
        f"{args.contact_threshold:.2f}) -> {result.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
