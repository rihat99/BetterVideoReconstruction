"""One clip as BetterRobot / BetterHuman viewer layers: the BVR adapter.

:func:`clip_layers` turns a :class:`~bvr.viewer.loading.ClipData` into layers from
``better_robot.viewer`` and ``better_human.viz``; this module only picks arrays, colours and sizes.
Each layer comes back as an :class:`Item` naming its sidebar checkbox (``kind``, one of
:data:`KINDS`) and the body source it belongs to, so the app shows one source at a time. Bodies are
posed once per clip with the BetterHuman SMPL-X body (forward kinematics on the CPU) and every
per-frame array is a whole ``(T, ...)`` clip; ``valid`` hides a person on the frames they are not
tracked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from better_human.viz import skinned_layer
from better_robot.lie import so3
from better_robot.viewer import (
    ArrowsLayer,
    CameraLayer,
    GroundLayer,
    Layer,
    PointsLayer,
    SkeletonLayer,
    SpheresLayer,
)

from bvr.body import Body, frame_normals

from .loading import FINGER_TOKENS, BodySource, ClipData, camera_poses

#: Sidebar checkbox key -> (label, on by default), in sidebar order.
KINDS = {
    "body": ("SMPL-X body", True),
    "skeleton": ("body skeleton", True),
    "contact_frames": ("contact frames", True),
    "frame_normals": ("frame normal arrows", False),
    "joint_forces": ("contact forces (per joint)", True),
    "frame_forces": ("contact forces (per frame)", False),
    "root_wrench": ("root force / torque", True),
    "human": ("human points", False),
    "scene": ("scene points", True),
    "normals": ("scene normals", False),
    "cameras": ("cameras", True),
    "gravity": ("gravity", True),
    "ground": ("ground plane", True),
}

# Per-person body colours, matching the stage overlay palette.
_BODY_RGB = ((166, 189, 219), (242, 153, 102), (153, 217, 140), (204, 166, 224), (242, 217, 115))
_MESH_OPACITY = 0.7                       # see-through, so the skeleton and contact frames read through it
_JOINT_R, _BONE_R, _FINGER_SCALE = 0.028, 0.012, 0.30
_JOINT_RGB, _BONE_RGB, _TORQUE_RGB = (120, 128, 140), (235, 235, 235), (245, 200, 40)
_CONTACT_RGB, _FREE_RGB, _LINK_RGB = (235, 45, 45), (60, 200, 90), (150, 190, 160)
_FRAME_R, _FRAME_NORMAL_LEN = 0.012, 0.06
_FORCE_RGB, _ROOT_FORCE_RGB, _ROOT_TORQUE_RGB = (235, 45, 45), (60, 220, 90), (220, 60, 220)
_FORCE_SCALE, _TORQUE_SCALE = 1.5e-3, 4.8e-3      # metres per newton / per newton-metre
# Solid arrows, sizes in metres: shaft radius, head radius, head length (the head keeps its size).
_FORCE_ARROW = dict(shaft_radius=0.012, head_radius=0.028, head_size=0.05)
_ROOT_ARROW = dict(shaft_radius=0.014, head_radius=0.032, head_size=0.06)
_FORCE_FLOOR_N = 1e-3                             # a solved force below this is a numerical zero: no arrow
_CAMERA_RGB, _PATH_RGB, _RIG_RGB = (235, 40, 40), (255, 140, 0), (0, 200, 255)
# Ground grid: ten times the fitted floor's data extent, half-metre cells (coarser past 80 cells,
# finer below 8 on geometry that was never metricized), a translucent grey fill under grey lines.
_GROUND_GRID_MULT = 10.0
_GROUND_CELL_M, _GROUND_CELLS_MAX, _GROUND_CELLS_MIN = 0.5, 80, 8


def rgb(color) -> np.ndarray:
    """0-255 colours ``(..., 3)`` -> the ``[0, 1]`` floats BetterRobot layers take."""
    return np.asarray(color, np.float32) / 255.0


@dataclass
class Item:
    """One layer and where it belongs: its viewer group, its sidebar checkbox and its body source."""

    layer: Layer
    group: str
    kind: str
    source: str | None = None             #: ``None``: shown whatever body source is active
    valid: np.ndarray | None = None       #: ``Viewer.add(valid=)``: (T,) or (T, N) bool


def clip_layers(data: ClipData, body: Body | None, *, point_size: float, human_point_size: float,
                camera_scale: float) -> list[Item]:
    """Every layer of one clip: the scene first, then one group per body source, then a dataset's extra arrows."""
    items = _scene_items(data, point_size, human_point_size, camera_scale)
    for name, src in data.bodies.items():
        items += _body_items(name, src, body)
    for extra in data.extras:             # measured forces, on the solved forces' scale
        group = "clip" if extra.source is None else extra.source.replace(" ", "_")
        layer = ArrowsLayer(extra.name, extra.origins, extra.vectors, scale=_FORCE_SCALE, colors=extra.color,
                            **_FORCE_ARROW)
        items.append(Item(layer, group, extra.kind, extra.source, extra.valid))
    return items


def _scene_items(data: ClipData, point_size: float, human_point_size: float,
                 camera_scale: float) -> list[Item]:
    items = []
    add = lambda layer, kind, valid=None: items.append(Item(layer, "clip", kind, None, valid))  # noqa: E731
    if len(data.scene_points):
        add(PointsLayer("scene_points", data.scene_points, colors=rgb(data.scene_colors),
                        point_size=point_size), "scene")
        if data.scene_normals is not None:
            # One arrow per point, coloured by direction ((n + 1) / 2 -> RGB) so flipped normals stand out.
            length = 2.0 * data.scene_voxel if data.scene_voxel > 0 else 0.05
            normals = np.asarray(data.scene_normals, np.float32)
            add(ArrowsLayer("scene_normals", data.scene_points, normals * length, colors=(normals + 1.0) * 0.5,
                            head_size=0.3 * length, line_width=1.0), "normals")
    human = _human_points(data, human_point_size)
    if human is not None:
        add(*human)

    poses = camera_poses(data.extr)
    if data.views is not None:            # a calibrated rig: fixed, measured cameras
        rig = camera_poses(data.views["extr"])
        for k, view in enumerate(data.views["names"]):
            add(CameraLayer(f"camera_{view}", rig[k], fov=float(data.views["fov_y"][k]),
                            aspect=float(data.views["aspect"][k]), scale=camera_scale, color=rgb(_RIG_RGB)),
                "cameras")
    else:                                 # the reconstructed camera, moving along its path
        add(CameraLayer("camera", poses, fov=float(np.median(data.fov_y)), aspect=data.aspect,
                        scale=camera_scale, color=rgb(_CAMERA_RGB)), "cameras")
        if len(poses) > 1:
            add(ArrowsLayer("camera_path", poses[:-1, :3], np.diff(poses[:, :3], axis=0), colors=rgb(_PATH_RGB),
                            head_size=0.0, line_width=2.0), "cameras")

    gravity = data.gravity
    if gravity is not None:
        color = (245, 170, 40) if gravity["is_fallback"] else (235, 40, 40)
        add(ArrowsLayer("gravity", poses[:, :3].mean(0)[None], gravity["g"][None] * camera_scale * 5.0,
                        colors=rgb(color), head_size=camera_scale * 0.4, shaft_radius=camera_scale * 0.12,
                        head_radius=camera_scale * 0.25), "gravity")
        plane = gravity.get("ground")
        if plane is not None:
            size = plane["extent"] * _GROUND_GRID_MULT
            cell = min(max(_GROUND_CELL_M, size / _GROUND_CELLS_MAX), size / _GROUND_CELLS_MIN)
            add(GroundLayer("ground", size=size, cell_size=cell, section_size=cell * 5.0,
                            origin=plane["centroid"], normal=plane["normal"], fill=0.25,
                            cell_color=rgb((110, 110, 110)), section_color=rgb((70, 70, 70)),
                            fill_color=rgb((150, 150, 150))), "ground")
    return items


def _human_points(data: ClipData, point_size: float) -> tuple[PointsLayer, str, np.ndarray] | None:
    """Per-frame human depth points as one ``(T, N, 3)`` layer, ``N`` the busiest frame's count.

    A ``PointsLayer`` draws a fixed number of points, so each frame is padded with copies of its
    own first point (they draw nothing new); a frame with no points at all is hidden.
    """
    T = data.n_frames
    keep = data.human_frame < T
    frame = data.human_frame[keep]
    if not frame.size:
        return None
    order = np.argsort(frame, kind="stable")
    frame = frame[order]
    counts = np.bincount(frame, minlength=T)
    rank = np.arange(frame.size) - np.concatenate([[0], np.cumsum(counts)[:-1]])[frame]
    n = int(counts.max())
    points = np.zeros((T, n, 3), np.float32)
    colors = np.zeros((T, n, 3), np.float32)
    points[frame, rank] = data.human_points[keep][order]
    colors[frame, rank] = rgb(data.human_colors[keep][order])
    pad = np.arange(n)[None] >= counts[:, None]
    points[pad] = np.broadcast_to(points[:, :1], points.shape)[pad]
    colors[pad] = np.broadcast_to(colors[:, :1], colors.shape)[pad]
    return PointsLayer("human_points", points, colors=colors, point_size=point_size), "human", counts > 0


def _rotate(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return so3.act(torch.as_tensor(quat_xyzw), torch.as_tensor(vec)).numpy()


def _body_items(name: str, src: BodySource, body: Body) -> list[Item]:
    """One body source's layers, per person: mesh, skeleton, contact frames and, for a dynamics body, forces."""
    items: list[Item] = []
    group = name.replace(" ", "_")
    finger = np.array([any(t in n for t in FINGER_TOKENS) for n in body.joint_names])
    radius = np.where(finger, _JOINT_R * _FINGER_SCALE, _JOINT_R)
    parents = body.parent_tree[0].cpu().numpy()
    frame_ids = list(body.contact_frame_ids)
    for p, oid in enumerate(src.object_ids):
        valid = src.valid[p]
        if not valid.any():
            continue
        tag = f"p{int(oid):02d}"

        def add(layer: Layer, kind: str, mask: np.ndarray = valid) -> None:
            items.append(Item(layer, group, kind, name, mask))

        with torch.no_grad():
            identity = body.shaped(torch.as_tensor(src.betas[p]))
            state = identity.forward(torch.as_tensor(src.q[p]))
            joints = state.joints.cpu().numpy()                                     # (T, J, 3)
            frames = state.frame_poses[:, frame_ids]                                # (T, F, 7)
            normals = frame_normals(frames).cpu().numpy()
            frames = frames.cpu().numpy()
        add(skinned_layer(identity, state, name=f"{tag}_mesh", color=tuple(rgb(_BODY_RGB[p % len(_BODY_RGB)])),
                          opacity=_MESH_OPACITY), "body")

        forces = src.forces
        joint_rgb = rgb(_JOINT_RGB)
        if forces is not None:            # joints coloured by torque, grey -> amber, the root stays grey
            heat = np.clip(forces["torque"][p] / forces["torque_ref"], 0.0, 1.0)[..., None]
            heat[:, 0] = 0.0
            joint_rgb = joint_rgb + heat * (rgb(_TORQUE_RGB) - joint_rgb)
        add(SkeletonLayer(f"{tag}_skeleton", joints, parents, radius=radius, bone_radius=_BONE_R,
                          colors=joint_rgb, bone_color=tuple(rgb(_BONE_RGB))), "skeleton")

        if src.contact is not None:       # red touching the scene, green free
            hot = rgb(np.where(src.contact[p][..., None], _CONTACT_RGB, _FREE_RGB))
            positions = frames[..., :3]
            add(SpheresLayer(f"{tag}_contact_frames", positions, np.full(positions.shape[1], _FRAME_R),
                             colors=hot, opacity=1.0), "contact_frames")
            anchor = joints[:, list(body.contact_frame_parents)]
            add(ArrowsLayer(f"{tag}_frame_links", anchor, positions - anchor, colors=rgb(_LINK_RGB),
                            head_size=0.0, line_width=1.5), "contact_frames")
            add(ArrowsLayer(f"{tag}_frame_normals", positions, normals * _FRAME_NORMAL_LEN, colors=hot,
                            head_size=0.3 * _FRAME_NORMAL_LEN, shaft_radius=0.05 * _FRAME_NORMAL_LEN,
                            head_radius=0.12 * _FRAME_NORMAL_LEN), "frame_normals")

        if forces is None:
            continue
        loaded = lambda f: valid[:, None] & (np.linalg.norm(f, axis=-1) > _FORCE_FLOOR_N)  # noqa: E731
        add(ArrowsLayer(f"{tag}_joint_forces", joints, forces["joint_forces"][p], scale=_FORCE_SCALE,
                        colors=rgb(_FORCE_RGB), **_FORCE_ARROW), "joint_forces", loaded(forces["joint_forces"][p]))
        if forces["frame_forces"] is not None:
            add(ArrowsLayer(f"{tag}_frame_forces", frames[..., :3], forces["frame_forces"][p],
                            scale=_FORCE_SCALE, colors=rgb(_FORCE_RGB), **_FORCE_ARROW),
                "frame_forces", loaded(forces["frame_forces"][p]))
        if forces["base_wrench"] is not None:
            # RNEA gives the root's residual wrench in the root-local frame; rotate it to the world.
            # A consistent solve drives it to zero: short arrows are good.
            root_rot = src.q[p, :, None, 3:7].repeat(2, axis=1)
            wrench = forces["base_wrench"][p].reshape(-1, 2, 3) * np.float32([_FORCE_SCALE, _TORQUE_SCALE])[:, None]
            add(ArrowsLayer(f"{tag}_root_wrench", joints[:, :1].repeat(2, axis=1), _rotate(root_rot, wrench),
                            colors=rgb([_ROOT_FORCE_RGB, _ROOT_TORQUE_RGB]), **_ROOT_ARROW),
                "root_wrench")
    return items
