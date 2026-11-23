"""A clip as BetterRobot and BetterHuman viewer layers, each under the tag that switches it in the sidebar.

:func:`scene_layers` draws what every body source shares: the scene and human depth points (read when first shown),
the camera's frustum on its path, gravity and the fitted floor. :func:`body_layers` draws the people of one source:
the exact posed mesh (BetterHuman's ``mesh_layer``), the skeleton (a ``RobotVisual`` without a skin: the joint tree,
its spheres of one size and smaller on the fingers, coloured by torque on a dynamics body, or orange and sized by it),
the contact frames red in contact and green free with their normals and a link to the joint each hangs on, and on a
dynamics body the contact forces and the root's residual force and torque. Each comes as ``(tag, layer, tracked)``,
``tracked (T,)`` the frames a person is drawn on. A skinned mesh is not used: viser 1.1.1 turns the page white when
one is removed with a browser open, and the app removes bodies on every switch of clip or source.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import better_human as bh
import numpy as np
import torch
from better_robot.lie import se3, so3
from better_robot.viewer import (
    MINIMAL,
    ArrowsLayer,
    CameraLayer,
    GroundLayer,
    Layer,
    LinesLayer,
    PointsLayer,
    RobotVisual,
    SkeletonRecipe,
    SpheresLayer,
    Theme,
)

from bvr import body

from .clip import Bodies, ClipData

#: The tags of the body layers, which a switch of source replaces, and the tags hidden until switched on.
BODY_TAGS = ("body", "skeleton", "contact_frames", "forces", "root_wrench")
HIDDEN = ("human_points",)

#: The people's colours in order: a mid blue, then the salmon, green, lilac and gold of the viewer before the rebuild.
HUES = tuple((red / 255, green / 255, blue / 255) for red, green, blue in
             ((74, 118, 181), (242, 153, 102), (153, 217, 140), (204, 166, 224), (242, 217, 115)))
ACCENT = (31 / 255, 58 / 255, 104 / 255)  # the sidebar's navy (#1f3a68), as before the rebuild
BODY_OPACITY = 0.7                       # see-through, so the skeleton and the contact frames show
FORCE_SCALE, TORQUE_SCALE = 1.5e-3, 4.8e-3   # metres per newton and per newton-metre
FRAME_RADIUS, NORMAL_LENGTH = 0.012, 0.06
LINK_RADIUS, LINK = 0.0036, (0.59, 0.75, 0.63)   # a contact frame's link to its joint: thin, yet no hairline
FINGERS = ("index", "middle", "pinky", "ring", "thumb")   # a joint whose name holds one is a finger joint
CONTACT, FREE, UNLABELLED = (0.92, 0.18, 0.18), (0.24, 0.78, 0.35), (0.55, 0.58, 0.62)
FORCE, ROOT_FORCE, ROOT_TORQUE = (0.92, 0.18, 0.18), (0.24, 0.86, 0.35), (0.86, 0.24, 0.86)
CAMERA, PATH, DOWN, FALLBACK_DOWN = (0.92, 0.16, 0.16), (1.0, 0.55, 0.0), (0.92, 0.16, 0.16), (0.96, 0.67, 0.16)
FLOOR = (0.43, 0.43, 0.43), (0.27, 0.27, 0.27), (0.59, 0.59, 0.59)   # cell lines, section lines, fill
FLOOR_FILL = 0.25                        # the floor's grey fill, see-through
FORCE_ARROW = {"shaft_radius": 0.012, "head_radius": 0.028, "head_length": 0.05}

Placed = tuple[str, Layer, np.ndarray | None]


def scene_layers(data: ClipData, *, point_size: float, point_size_spacings: float, human_point_size: float,
                 camera_scale: float) -> list[Placed]:
    """The layers of ``data`` that every body source shares; the scene's points are ``point_size_spacings`` of its
    sample spacing in radius, so a surface reads closed at any spacing, or ``point_size`` where it is not recorded."""
    poses = camera_poses(data.extrinsics)
    placed: list[Placed] = []
    if data.scene is not None:
        size = point_size if data.spacing is None else point_size_spacings * data.spacing
        placed.append(("scene", PointsLayer("points", data.scene[0], colors=data.scene[1] / 255.0,
                                            point_size=size), None))
    if data.human is not None:
        placed.append(("human_points", HumanPoints(data.human, len(poses), point_size=human_point_size), None))
    intrinsics = np.median(data.intrinsics, axis=0)
    # The frustum is cut where its volume is 4 camera_scale^3, as viser sizes one, at any field of view.
    halves = np.asarray(data.size) / (2.0 * intrinsics[[0, 1], [0, 1]])   # tan of the half fields of view, x and y
    placed += [
        ("cameras", CameraLayer("camera", poses, intrinsics=intrinsics, size=data.size,
                                scale=camera_scale / float(np.cbrt(halves.prod() / 3.0)), color=CAMERA), None),
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
        # Ten times the floor's extent in the scene, in half-metre cells, coarser past 80 cells and finer below 8.
        size = 10.0 * extent
        cell = min(max(0.5, size / 80.0), size / 8.0)
        floor = GroundLayer("floor", size=size, cell_size=cell, section_size=5.0 * cell, origin=centroid,
                            normal=normal, fill=FLOOR_FILL, cell_color=FLOOR[0], section_color=FLOOR[1],
                            fill_color=FLOOR[2])
        placed.append(("ground_plane", floor, None))
    return placed


def body_layers(bodies: Bodies, human: bh.HumanModel, theme: Theme, *, opacity: float, joint_radius: float,
                finger_joint_radius: float, torque_size: bool, torque_growth: float) -> list[Placed]:
    """The layers of the people of one body source, in the theme's hue of each person, the meshes at ``opacity``;
    the skeleton's joints are spheres of ``joint_radius``, ``finger_joint_radius`` on the fingers, and with
    ``torque_size`` a dynamics body's torque sizes them, up to ``torque_growth`` times, instead of colouring them."""
    placed: list[Placed] = []
    radii = {name: finger_joint_radius if any(finger in name for finger in FINGERS) else joint_radius
             for name in human.model.joint_names[1:]}
    # A joint with a torque is drawn in the top colour of the torque ramp, orange: torque as colour paints over it,
    # torque as size keeps it, and it stands out of the person's hue.
    torqued = {} if bodies.torques is None else dict.fromkeys(human.model.joint_names[2:], theme.ramp[1])
    medium = SkeletonRecipe(joint_radius=radii, bodies=False, color=torqued, growth=torque_growth)
    minimal = replace(MINIMAL, bodies=False, color=torqued, growth=torque_growth)
    # The joint each contact frame hangs on, in joint_poses, which leaves out the universe (joint 0 of the model).
    hung = [human.model.frame_parent_joint_ids[frame] - 1 for frame in human.contact_frames.frame_ids]
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
            anchors = state.joint_poses[..., hung, :3].numpy()
        positions = frames[..., :3].numpy()
        # The universe and the free root keep the skeleton's colour (NaN); the root's wrench is drawn as arrows.
        values = None if torques is None else np.pad(torques[person], ((0, 0), (2, 0)), constant_values=np.nan)
        colors = UNLABELLED if bodies.contact is None else np.where(bodies.contact[person][..., None], CONTACT, FREE)
        placed += [
            ("body", bh.viz.mesh_layer(state, name=name, color=hue, opacity=opacity), tracked),
            ("skeleton", RobotVisual(identity.model, q, name=name, style="medium", medium=medium, minimal=minimal,
                                     hue=hue, joint_values=values, value_range=(0.0, top),
                                     values_as="size" if torque_size else "color"), tracked),
            ("contact_frames", SpheresLayer(name, positions, radii=np.full(positions.shape[1], FRAME_RADIUS),
                                            colors=colors, opacity=1.0), tracked),
            ("contact_frames", ArrowsLayer(f"{name}_normals", positions, vectors=normals * NORMAL_LENGTH, colors=colors,
                                           shaft_radius=0.003, head_radius=0.007, head_length=0.018), tracked),
            ("contact_frames", ArrowsLayer(f"{name}_links", anchors, vectors=positions - anchors, colors=LINK,
                                           shaft_radius=LINK_RADIUS, head_radius=LINK_RADIUS, head_length=0.0), tracked),
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


class HumanPoints(Layer):
    """The human depth points, one cloud per frame, which ``read()`` returns as points ``(N_t, 3)`` and colours
    ``(N_t, 3)`` uint8 the first time the layer is shown: their file is large (310 MB on boulder_1) and the layer
    starts hidden, so a switch of clip does not read it."""

    def __init__(self, read: Callable[[], tuple[list[np.ndarray], list[np.ndarray]]], frames: int, *,
                 point_size: float) -> None:
        super().__init__("points", frames)
        self.read, self.point_size = read, point_size
        self.cloud: PointsLayer | None = None
        self._root: tuple[Any, str] | None = None

    def build(self, scene: Any, root: str) -> None:
        self._root = scene, root

    def set_step(self, step: int) -> None:
        if not self.visible:
            return
        if self.cloud is None:
            points, colors = self.read()
            self.cloud = PointsLayer("points", points, colors=[rgb / 255.0 for rgb in colors],
                                     point_size=self.point_size)
            self.cloud.build(*self._root)
        self.cloud.set_step(step)

    def set_visible(self, visible: bool) -> None:
        super().set_visible(visible)
        if self.cloud is not None:
            self.cloud.set_visible(visible)

    def dispose(self) -> None:
        if self.cloud is not None:
            self.cloud.dispose()
        super().dispose()
