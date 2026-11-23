"""The viewer of a run: a BetterRobot ``Viewer`` that shows the clips of a run one at a time.

The sidebar holds, in the order of the viewer before the rebuild, the clip (a pager over the run's clips), the
Timeline (a Play/Pause button, step, speed), the Images (the video, in step with the timeline), the body source
(converter, pre-fit, the two kinematic fits and the dynamics fit, each when the clip holds it, whether a dynamics
body's joint torque sizes the joint spheres instead of colouring them, and whether the contact forces are drawn per
joint or per contact frame), the Layers (one checkbox per tag of
:mod:`bvr.viewer.layers`), the Display (body opacity and scene point size), then BetterRobot's skeleton Style and a
line of information. A switch of clip takes the old clip off the screen at once, then reads and draws the new one; a
switch of source replaces only the bodies and keeps the frame. A layer switched off stays off across switches. The
view opens on the people over the whole clip, so a person who travels stays in it, seen from above the side of the
filming camera (BetterRobot's ``Viewer.aim``).
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from better_robot.exceptions import UnknownNameError
from better_robot.viewer import LIGHT, ImagePanel, Layer, Viewer

from bvr import body, paths

from . import clip as clips_module
from . import layers

log = logging.getLogger(__name__)


class App:
    """The clips of a run in one viewer; ``settings`` is the config's ``[viewer]`` table."""

    def __init__(self, clips: list[paths.Clip], settings: dict) -> None:
        self.clips, self.settings = {clip.name: clip for clip in clips}, settings
        self.human = body.load("cpu")
        theme = replace(LIGHT, up=settings["up"], hues=layers.HUES, accent=layers.ACCENT)
        self.viewer = Viewer(host="0.0.0.0", port=settings["port"], theme=theme, ground=False)
        self.viewer.aim()           # stop BetterRobot's framing after each add: the app frames the people itself
        self.viewer.timeline.play_button = True
        self.hidden = set(layers.HIDDEN)
        self.data: clips_module.ClipData | None = None
        self.source: str | None = None
        self.view: tuple[np.ndarray, np.ndarray] | None = None
        self.torque_size, self.body_opacity = settings["torque_size"], layers.BODY_OPACITY
        self.force_display = settings["force_display"]
        sidebar, names = self.viewer.sidebar, list(self.clips)
        sidebar.order(("Clip", "Timeline", "Images", "Body", "Layers", "Display"))
        if len(names) > 1:
            menu = sidebar.dropdown("Clip", "Clip", options=names, value=names[0],
                                    on_change=lambda name: self.show(self.clips[name]))
            for label, step in (("Previous", -1), ("Next", 1)):
                sidebar.button("Clip", label, on_click=lambda step=step: setattr(
                    menu, "value", names[(names.index(menu.value) + step) % len(names)]))
        sidebar.dropdown("Body", "Source", options=list(clips_module.SOURCES),
                         value=list(clips_module.SOURCES)[-1], on_change=self.show_source)
        sidebar.checkbox("Body", "Torque as size", value=self.torque_size, on_change=self.show_torque_size)
        sidebar.dropdown("Body", "Forces", options=list(layers.FORCE_DISPLAYS), value=self.force_display,
                         on_change=self.show_force_display)
        sidebar.slider("Display", "Body opacity", low=0.1, high=1.0, step=0.05, value=self.body_opacity,
                       on_change=self.set_body_opacity)
        sidebar.slider("Display", "Scene point size", low=0.002, high=0.1, step=0.002, value=settings["point_size"],
                       on_change=self.set_point_size)
        with sidebar.section("Info"):
            self.info = sidebar.gui.add_markdown("")
        self.show(clips[0])

    def show(self, clip: paths.Clip) -> None:
        """Show ``clip``, with the body source on screen if the clip holds it, else the last source its own tree
        holds, else its last source. The clip on screen goes first, so a switch shows at once."""
        with self.viewer.lock:
            self._remove(self.viewer.tags)
            self.info.content = f"loading **{clip.name}** …"
            data = clips_module.read(clip, human_max_points=self.settings["human_max_points"])
            source = self.source if self.source in data.sources else (data.written or data.sources or (None,))[-1]
            bodies = self._bodies(clip, data, source)
            self.data, self.source = data, source
            settings = {key: self.settings[key] for key in
                        ("point_size", "point_size_spacings", "human_point_size", "camera_scale")}
            placed = layers.scene_layers(data, **settings) + bodies
            for tag, layer, _ in placed:
                if tag == "scene":
                    self.viewer.sidebar.set_value("Display", "Scene point size", value=layer.point_size)
            # The view frames the people alone, else everything: the scene and the cameras would widen it, so they
            # are switched on once it is aimed.
            later = {tag for tag, _, _ in placed if tag not in layers.BODY_TAGS} - self.hidden if bodies else set()
            self._add(placed, self.hidden | later)
            self.viewer.add(ImagePanel("Video", data.video, nsteps=len(data.extrinsics)), tags="video")
            self.viewer.set_fps(data.fps)
            if source is not None:
                self.viewer.sidebar.set_value("Body", "Source", value=source)
            self.view = self.viewer.aim(
                azimuth=opening_azimuth(data, bodies, np.asarray(self.viewer.theme.up_vector)))
            for tag in later:
                self.viewer.set_visible(tag, True)
            people = sum(tag == "body" for tag, _, _ in bodies)
            upstream = [name for name in data.sources if name not in data.written]
            self.info.content = (f"**{data.name}** · {len(data.extrinsics)} frames at {data.fps:.2f} fps · "
                                 f"{people} people · gravity: {data.gravity_source or 'none'}"
                                 + (f" · from upstream: {', '.join(upstream)}" if upstream else ""))
        log.info("showing %s (%s)", data.name, source)

    def show_source(self, source: str) -> None:
        """Replace the bodies on screen by those of ``source``; the frame stays."""
        assert self.data is not None
        if source not in self.data.sources:
            raise UnknownNameError("body source of this clip", source, valid=list(self.data.sources))
        bodies = self._bodies(self.clips[self.data.name], self.data, source)
        self._remove([tag for tag in self.viewer.tags if tag in layers.BODY_TAGS])
        self._add(bodies, self.hidden)
        self.source = source

    def show_torque_size(self, on: bool) -> None:
        """Let a dynamics body's joint torque size the joint spheres (``on``) or colour them."""
        self.torque_size = on
        for layer in self._tagged("skeleton"):
            layer.set_joint_values(layer.joint_values, values_as="size" if on else "color")

    def show_force_display(self, display: str) -> None:
        """Draw the contact forces as ``display``, one of ``layers.FORCE_DISPLAYS``: the bodies are drawn again."""
        self.force_display = display
        if self.source is not None:
            self.show_source(self.source)

    def set_body_opacity(self, opacity: float) -> None:
        """Draw the bodies at ``opacity``, now and after a switch."""
        self.body_opacity = opacity
        for layer in self._tagged("body"):
            layer.set_opacity(opacity)

    def set_point_size(self, size: float) -> None:
        """Draw the scene's points at radius ``size`` in metres, until the next clip."""
        for layer in self._tagged("scene"):
            layer.set_point_size(size)

    def record(self, path: Path) -> Path:
        """Write the clip on screen to ``path`` (``.mp4``, ``.webm`` or a ``.png`` still) as the browsers see it."""
        return self.viewer.record(path, camera=self.view)

    def _bodies(self, clip: paths.Clip, data: clips_module.ClipData, source: str | None) -> list[layers.Placed]:
        if source is None:
            return []
        bodies = clips_module.read_bodies(clip, source, self.human, data.extrinsics)
        return layers.body_layers(bodies, self.human, self.viewer.theme, opacity=self.body_opacity,
                                  joint_radius=self.settings["joint_radius"],
                                  finger_joint_radius=self.settings["finger_joint_radius"], torque_size=self.torque_size,
                                  torque_growth=self.settings["torque_growth"],
                                  force_display=self.force_display)

    def _tagged(self, tag: str) -> list[Layer]:
        return [layer for layer in self.viewer.layers if tag in layer.tags]

    def _remove(self, tags: list[str]) -> None:
        """Remove the layers of ``tags``, remembering which of them were switched off."""
        for tag in tags:
            shown = any(layer.visible for layer in self.viewer.layers if tag in layer.tags)
            (self.hidden.discard if shown else self.hidden.add)(tag)
            self.viewer.remove(tag)

    def _add(self, placed: list[layers.Placed], hidden: set[str]) -> None:
        """Add ``placed``, the layers of the tags in ``hidden`` switched off."""
        for tag, layer, tracked in placed:
            if tag in hidden:
                layer.set_visible(False)
            self.viewer.add(layer, tags=tag, valid=tracked)


def opening_azimuth(data: clips_module.ClipData, bodies: list[layers.Placed], up: np.ndarray) -> float:
    """The azimuth of ``Viewer.aim`` about ``up (3,)`` that looks at the people (else the scene, else the cameras)
    from the side of the mean camera centre."""
    cameras = layers.camera_poses(data.extrinsics)[:, :3].mean(axis=0)
    roots = [layer.q[:, :3].numpy()[tracked] for tag, layer, tracked in bodies if tag == "skeleton"]
    centre = (np.concatenate(roots).mean(axis=0) if roots
              else cameras if data.scene is None else data.scene[0].mean(axis=0))
    away = cameras - centre
    # Viewer.aim looks from `side` at azimuth 0 and from `up x side` at 90 degrees (its code; documented for +z up).
    side = np.cross([0.0, 0.0, 1.0] if abs(up[2]) < 0.9 else [1.0, 0.0, 0.0], up)
    return math.atan2(float(away @ np.cross(up, side)), float(away @ side))


def serve(clips: list[paths.Clip], settings: dict, *, record: Path | None = None) -> None:
    """Serve ``clips`` until interrupted, or write the first one to ``record`` and return."""
    app = App(clips, settings)
    if record is not None:
        log.info("wrote %s", app.record(record))
        return
    log.info("serving %d clip(s) at %s", len(clips), app.viewer.url)
    app.viewer.serve()
