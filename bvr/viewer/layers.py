"""A clip as BetterRobot and BetterHuman viewer layers, each under the tag that switches it in the sidebar.

:func:`scene_layers` draws what every body source shares: the scene and human depth points, the camera on its path
with the video frame on its frustum, gravity and the fitted floor. :func:`body_layers` draws the people of one source:
the exact posed mesh (BetterHuman's ``mesh_layer``), the skeleton (a ``RobotVisual`` without a skin, its joints
coloured by torque on a dynamics body), the contact frames red in contact and green free with their normals, and on a
dynamics body the contact forces and the root's residual force and torque. Each comes as ``(tag, layer, tracked)``,
``tracked (T,)`` the frames a person is drawn on. A skinned mesh is not used: viser 1.1.1 turns the page white when
one is removed with a browser open, and the app removes bodies on every switch of clip or source.
"""
from __future__ import annotations

import better_human as bh
import numpy as np
import torch
from better_robot.lie import se3, so3
from better_robot.viewer import (
    ArrowsLayer,
    CameraLayer,
    GroundLayer,
    Layer,
    LinesLayer,
    PointsLayer,
    RobotVisual,
    SpheresLayer,
    Theme,
)

from bvr import body

from .clip import Bodies, ClipData

#: The tags of the body layers, which a switch of source replaces, and the tags hidden until switched on.
BODY_TAGS = ("body", "skeleton", "contact_frames", "forces", "root_wrench")
HIDDEN = ("human_points",)

BODY_OPACITY = 0.7                       # see-through, so the skeleton and the contact frames show
FORCE_SCALE, TORQUE_SCALE = 1.5e-3, 4.8e-3   # metres per newton and per newton-metre
FRAME_RADIUS, NORMAL_LENGTH = 0.012, 0.06
CONTACT, FREE, UNLABELLED = (0.92, 0.18, 0.18), (0.24, 0.78, 0.35), (0.55, 0.58, 0.62)
FORCE, ROOT_FORCE, ROOT_TORQUE = (0.92, 0.18, 0.18), (0.24, 0.86, 0.35), (0.86, 0.24, 0.86)
CAMERA, PATH, DOWN, FALLBACK_DOWN = (0.92, 0.16, 0.16), (1.0, 0.55, 0.0), (0.92, 0.16, 0.16), (0.96, 0.67, 0.16)
FLOOR = (0.43, 0.43, 0.43), (0.27, 0.27, 0.27), (0.59, 0.59, 0.59)   # cell lines, section lines, fill
FORCE_ARROW = {"shaft_radius": 0.012, "head_radius": 0.028, "head_length": 0.05}

Placed = tuple[str, Layer, np.ndarray | None]


def scene_layers(data: ClipData, *, point_size: float, human_point_size: float, camera_scale: float) -> list[Placed]:
    """The layers of ``data`` that every body source shares."""
    poses = camera_poses(data.extrinsics)
    placed: list[Placed] = [
        ("scene", PointsLayer("points", data.scene[0], colors=data.scene[1] / 255.0, point_size=point_size), None),
        ("human_points", PointsLayer("points", data.human[0], colors=[colors / 255.0 for colors in data.human[1]],
                                     point_size=human_point_size), None),
        ("cameras", CameraLayer("camera", poses, intrinsics=np.median(data.intrinsics, axis=0), size=data.size,
                                scale=camera_scale, color=CAMERA, image=data.video), None),
        ("cameras", LinesLayer("path", np.stack([poses[:-1, :3], poses[1:, :3]], axis=1), colors=PATH), None),
    ]
    if data.gravity is not None:
        down = ArrowsLayer("down", poses[:, :3].mean(axis=0)[None], vectors=data.gravity[None] * 5.0 * camera_scale,
                           colors=FALLBACK_DOWN if data.gravity_source == "fallback_down" else DOWN,
                           shaft_radius=0.12 * camera_scale, head_radius=0.25 * camera_scale,
                           head_length=0.4 * camera_scale)
        placed.append(("gravity", down, None))
    if data.ground is not None:
        centroid, normal, extent = data.ground
        floor = GroundLayer("floor", size=10.0 * extent, cell_size=0.5, section_size=2.5, origin=centroid,
                            normal=normal, fill=0.25, cell_color=FLOOR[0], section_color=FLOOR[1], fill_color=FLOOR[2])
        placed.append(("ground_plane", floor, None))
    return placed


def body_layers(bodies: Bodies, human: bh.HumanModel, theme: Theme) -> list[Placed]:
    """The layers of the people of one body source, in the theme's hue of each person."""
    placed: list[Placed] = []
    torques = None if bodies.torques is None else np.linalg.norm(
        bodies.torques[..., 6:].reshape(*bodies.torques.shape[:2], -1, 3), axis=-1)       # (P, T, 51) per joint
    top = 1.0 if torques is None else float(np.percentile(torques[bodies.tracked], 95)) or 1.0
    for person, object_id in enumerate(bodies.object_ids):
        tracked, q, hue, name = bodies.tracked[person], bodies.q[person], theme.hue(person), f"person_{object_id}"
        if not tracked.any():
            continue
        identity = human.bake(betas=torch.as_tensor(bodies.betas[person]))
        with torch.no_grad():
            state = identity.forward(torch.as_tensor(q))
            frames = body.contact_frames(state)
            normals = body.frame_normals(frames).numpy()
        positions = frames[..., :3].numpy()
        # The universe and the free root keep the skeleton's colour (NaN); the root's wrench is drawn as arrows.
        values = None if torques is None else np.pad(torques[person], ((0, 0), (2, 0)), constant_values=np.nan)
        colors = UNLABELLED if bodies.contact is None else np.where(bodies.contact[person][..., None], CONTACT, FREE)
        placed += [
            ("body", bh.viz.mesh_layer(state, name=name, color=hue, opacity=BODY_OPACITY), tracked),
            ("skeleton", RobotVisual(identity.model, q, name=name, style="medium", hue=hue, joint_values=values,
                                     value_range=(0.0, top)), tracked),
            ("contact_frames", SpheresLayer(name, positions, radii=np.full(positions.shape[1], FRAME_RADIUS),
                                            colors=colors, opacity=1.0), tracked),
            ("contact_frames", ArrowsLayer(f"{name}_normals", positions, vectors=normals * NORMAL_LENGTH, colors=colors,
                                           shaft_radius=0.003, head_radius=0.007, head_length=0.018), tracked),
        ]
        if bodies.forces is not None:
            placed.append(("forces", ArrowsLayer(name, positions, vectors=bodies.forces[person], scale=FORCE_SCALE,
                                                 colors=FORCE, **FORCE_ARROW), tracked))
        if bodies.torques is not None:
            # The root's residual force and torque, stored in the root's frame: a consistent solve makes them short.
            wrench = so3.act(torch.as_tensor(q[:, None, 3:7]), torch.as_tensor(
                bodies.torques[person, :, :6].reshape(-1, 2, 3))).numpy() * [[FORCE_SCALE], [TORQUE_SCALE]]
            placed.append(("root_wrench", ArrowsLayer(name, q[:, None, :3].repeat(2, axis=1), vectors=wrench,
                                                      colors=[ROOT_FORCE, ROOT_TORQUE], **FORCE_ARROW), tracked))
    return placed


def camera_poses(extrinsics: np.ndarray) -> np.ndarray:
    """Camera-to-world poses ``(T, 7)`` of world-to-camera matrices ``(T, 4, 4)``."""
    return se3.inverse(se3.from_matrix(torch.as_tensor(extrinsics, dtype=torch.float64))).numpy()
