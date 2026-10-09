"""The BVR viewer app: a BetterRobot ``Viewer`` plus a small sidebar for browsing ``out/<stem>/`` clips.

The sidebar holds a clip pager (when there is more than one clip), BetterRobot's Timeline folder
(play, step, speed, loop), the video pane, the body-source dropdown, one checkbox per layer kind and
an info panel. Switching clips reads the new clip first (a clip that fails to read leaves the old one
on screen), then removes the old clip's groups, so the timeline fits the new clip, and plays at the
new clip's fps. Every drawing primitive comes from :mod:`bvr.viewer.layers`; this module only adds
``server.gui`` widgets and frames the browser camera on the person. Clips are read with
``load_clip_data`` unless the caller passes a dataset-aware ``read`` (``scripts/view_3d.py`` does);
a layer kind no core checkbox knows gets its own checkbox, off by default.
"""
from __future__ import annotations

import random
import threading
import traceback
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
from better_robot.viewer import LIGHT, ImagePanel, Viewer

from bvr.body import Body, load_body

from .layers import KINDS, Item, clip_layers
from .loading import ClipData, camera_poses, load_clip_data

_LATE = 1e6          # GUI order that puts a folder after BetterRobot's Timeline folder
#: The body source a clip opens on: the first of these it has, else its first source.
_DEFAULT_SOURCES = ("kinematics", "kinematics_2", "prefit", "original")
#: What makes a directory a clip for ``load_clip_data``: a fused ``out/<stem>/`` tree.
CLIP_MARKERS = (Path("geometry") / "scene.npz",)


class _VideoPanel(ImagePanel):
    """A video ``ImagePanel`` whose image sits in the app's video folder.

    Only the image is built inside the folder, not the whole ``Viewer.add``: an add can build
    BetterRobot's Timeline folder (on the first layer with more than one step), which must not land
    in the video folder.
    """

    def __init__(self, name: str, frames: Callable[[int], np.ndarray], nsteps: int, folder) -> None:
        super().__init__(name, frames, nsteps=nsteps)
        self.folder, self.listed = folder, False      # the app's view dropdown toggles it

    def build_gui(self, gui) -> None:
        with self.folder:
            super().build_gui(gui)


class App:
    """Sidebar and clip switching around one BetterRobot ``Viewer``.

    Each clip's layers come from :func:`~bvr.viewer.layers.clip_layers` as :class:`Item` s; a layer
    is shown when its kind's checkbox is on and it belongs to no body source or to the active one.
    ``read(clip, *, body, human_max_points, video, video_dir) -> ClipData`` reads a clip
    (``load_clip_data``, or a dataset-aware reader that builds on it).
    """

    def __init__(self, viewer: Viewer, clips: list[Path], *, body: Body,
                 read: Callable[..., ClipData] = load_clip_data, show_smplx: bool = True,
                 video: bool = True, video_dir: Path | None = None, human_max_points: int = 0,
                 point_size: float = 0.02, human_point_size: float = 0.01, camera_scale: float = 0.1) -> None:
        self.viewer, self.clips, self.body, self.read = viewer, clips, body, read
        self.read_opts = dict(body=body if show_smplx else None, video=video, video_dir=video_dir,
                              human_max_points=human_max_points)
        self.draw_opts = dict(point_size=point_size, human_point_size=human_point_size, camera_scale=camera_scale)
        self.cache: dict[Path, ClipData] = {}
        self.data: ClipData | None = None
        self.items: list[Item] = []
        self.panels: dict[str, ImagePanel] = {}
        self._lock = threading.RLock()     # viser runs GUI callbacks on its own threads

        gui = viewer.server.gui
        gui.add_markdown("### BetterVideoReconstruction · 3D viewer")
        self.clip_menu = self.position = None
        if len(clips) > 1:
            with gui.add_folder("clip"):
                self.clip_menu = gui.add_dropdown("scene", tuple(c.name for c in clips), initial_value=clips[0].name)
                self.clip_menu.on_update(lambda _: self.load(next(c for c in clips if c.name == self.clip_menu.value)))
                for label, delta in (("◀ prev", -1), ("random", 0), ("next ▶", 1)):
                    gui.add_button(label).on_click(lambda _, d=delta: self._step_clip(d))
                self.position = gui.add_markdown("")
        self.video_folder = self.view_menu = None
        if video:
            self.video_folder = gui.add_folder("video", order=_LATE)
            with self.video_folder:     # a calibrated rig picks which camera plays
                self.view_menu = gui.add_dropdown("view", ("-",), initial_value="-", visible=False)
                self.view_menu.on_update(lambda _: self._show_view())
        with gui.add_folder("body", order=_LATE + 1):
            self.source = gui.add_dropdown("source", ("(none)",), initial_value="(none)")
        self.layers_folder = gui.add_folder("layers", order=_LATE + 2)
        with self.layers_folder:
            self.kinds = {key: gui.add_checkbox(label, initial_value=on) for key, (label, on) in KINDS.items()}
        for control in (self.source, *self.kinds.values()):
            control.on_update(lambda _: self._apply())
        self.info = gui.add_markdown("", order=_LATE + 3)
        viewer.server.on_client_connect(self._frame)
        self.load(clips[0])

    def load(self, clip: Path) -> None:
        """Replace the clip on screen with ``clip`` (read once, then cached); if it fails to read, keep the old one."""
        with self._lock:
            if self.data is not None and self.cache.get(clip) is self.data:
                return                                   # already on screen
            print(f"[view_3d] loading {clip.name} …", flush=True)
            try:
                data = self.cache.get(clip) or self.read(clip, **self.read_opts)
            except Exception:
                if self.data is None:                    # nothing on screen to fall back to
                    raise
                traceback.print_exc()
                print(f"[view_3d] could not read {clip.name}; still showing {self.data.name}", flush=True)
                if self.clip_menu is not None:
                    self.clip_menu.value = self.data.name
                return
            for group in {item.group for item in self.items} | ({"video"} if self.panels else set()):
                self.viewer.remove_group(group)
            for frames in self.data.videos.values() if self.data is not None else ():
                frames.close()                           # its panels are gone: free the decoder
            self.cache[clip] = self.data = data
            self.items = clip_layers(data, self.body, **self.draw_opts)
            for kind in dict.fromkeys(item.kind for item in self.items):
                if kind not in self.kinds:           # a dataset's own layer: its kind is the checkbox label
                    with self.layers_folder:
                        self.kinds[kind] = self.viewer.server.gui.add_checkbox(kind, initial_value=False)
                    self.kinds[kind].on_update(lambda _: self._apply())
            for item in self.items:
                item.layer.listed = False            # the app's own checkboxes toggle them
                self.viewer.add(item.layer, item.group, valid=item.valid)
            self.panels = {}
            if self.video_folder is not None:        # reset on every clip: a clip may have no video
                for name, frames in data.videos.items():
                    self.panels[name] = self.viewer.add(_VideoPanel(name, frames, data.n_frames, self.video_folder),
                                                        "video")
                names = tuple(self.panels) or ("-",)
                self.view_menu.options, self.view_menu.visible = names, len(self.panels) > 1
                self.view_menu.value = names[0]
                self.video_folder.visible = bool(self.panels)
                self._show_view()
            self.viewer.set_fps(data.fps or 30.0)
            sources = tuple(data.bodies) or ("(none)",)
            self.source.options = sources
            self.source.value = next((s for s in _DEFAULT_SOURCES if s in data.bodies), sources[0])
            self._apply()
            self.info.content = info_markdown(data)
            if self.position is not None:
                self.position.content = f"scene **{self.clips.index(clip) + 1} / {len(self.clips)}**"
        for client in self.viewer.server.get_clients().values():
            self._frame(client)
        print(f"[view_3d] {data.name}: {data.n_frames} frames · {len(data.scene_points)} scene pts · "
              f"bodies {list(data.bodies)}", flush=True)

    def _apply(self) -> None:
        """Show each layer whose kind is ticked and whose body source is active; hide checkboxes with nothing to toggle."""
        with self._lock:
            active, present = self.source.value, set()
            for item in self.items:
                mine = item.source is None or item.source == active
                if mine:
                    present.add(item.kind)
                on = mine and bool(self.kinds[item.kind].value)
                if item.layer.visible != on:
                    self.viewer.set_visible(item.layer, on)
            for key, checkbox in self.kinds.items():
                checkbox.visible = key in present

    def _show_view(self) -> None:
        with self._lock:
            for name, panel in self.panels.items():
                if panel.visible != (name == self.view_menu.value):
                    self.viewer.set_visible(panel, name == self.view_menu.value)

    def _step_clip(self, delta: int) -> None:
        """Previous (-1) / next (+1) clip with wraparound, or a different random one (0)."""
        n = len(self.clips)
        current = next(i for i, c in enumerate(self.clips) if c.name == self.clip_menu.value)
        self.clip_menu.value = self.clips[(current + (random.randrange(1, n) if delta == 0 else delta)) % n].name

    def _frame(self, client) -> None:
        """Aim a browser at the person, from the filming side, so orbiting pivots on the person."""
        data = self.data
        if data is None:
            return
        focus, position = opening_view(data, self.source.value)
        client.camera.position = tuple(float(x) for x in position)    # position first: it drags look_at along
        client.camera.look_at = tuple(float(x) for x in focus)


def opening_view(data: ClipData, source: str | None) -> tuple[np.ndarray, np.ndarray]:
    """``(look_at, position)``: the mean root of ``source``'s people (else the human / scene points),
    seen from the mean camera centre's side, at least 2 m away."""
    src = data.bodies.get(source)
    if src is not None and src.valid.any():
        focus = src.q[..., :3][src.valid].mean(0)               # the free-flyer root sits at the pelvis
    elif len(data.human_points):
        focus = data.human_points.mean(0)
    elif len(data.scene_points):
        focus = data.scene_points.mean(0)
    else:
        focus = np.zeros(3)
    focus = np.asarray(focus, np.float64)
    away = camera_poses(data.extr)[:, :3].mean(0) - focus
    dist = float(np.linalg.norm(away))
    away, dist = (away / dist, dist) if dist > 1e-3 else (np.array([0.0, 0.0, -1.0]), 1.0)
    return focus, focus + away * max(dist, 2.0)


def info_markdown(data: ClipData) -> str:
    """The info panel: clip size, cameras, people, body sources, gravity and the ground-plane fit."""
    gravity = data.gravity
    gtext, ground = "-", ""
    if gravity is not None:
        g = gravity["g"]
        source = gravity["source"]            # a dataset's own tag (ground-truth gravity) prints as it is
        tag = {"ground": "ground plane", "geocalib": f"GeoCalib, spread {gravity['spread']:.1f}°",
               "fallback_down": "default down"}.get(source, source)
        gtext = f"[{g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}] ({tag})"
        if gravity["ground"] is not None:
            ground = f" · ground ∠prior {gravity['ground']['angle_deg']:.1f}°"
        elif gravity["ground_reject"]:
            ground = f" · ground: {gravity['ground_reject']}"
    people = sorted({int(o) for src in data.bodies.values() for o in src.object_ids})
    cameras = len(data.views["names"]) if data.views is not None else len(data.extr)
    fps = f"{data.fps:.1f} fps" if data.fps else "fps unknown"
    return (f"**{data.name}** · {data.n_frames} frames · {fps} · {len(data.scene_points):,} scene pts · "
            f"{cameras} cameras · {len(data.human_points):,} human pts · people {people}\n\n"
            f"bodies: {', '.join(data.bodies) or '-'} · gravity: {gtext}{ground}")


def _is_clip(d: Path, markers=CLIP_MARKERS) -> bool:
    return any((d / marker).exists() for marker in markers)


def _discover_clips(root: Path, markers=CLIP_MARKERS) -> list[Path]:
    """Clips = ``root`` itself when it is one, else its clip subdirectories (sorted)."""
    root = Path(root)
    if _is_clip(root, markers):
        return [root]
    if not root.is_dir():
        return []
    return [d for d in sorted(root.iterdir()) if d.is_dir() and _is_clip(d, markers)]


def view_3d(root: Path, *, port: int = 8082, up: str = "-y", markers=CLIP_MARKERS, **opts) -> None:
    """Serve every clip under ``root`` (an ``out/`` tree or one ``out/<stem>/``) until interrupted.

    A clip is a directory holding one of ``markers`` (a dataset reader passed as ``read`` adds its
    own). ``opts`` are :class:`App`'s keyword options. The server listens on all interfaces, like
    viser's default.
    """
    clips = _discover_clips(root, markers)
    if not clips:
        raise FileNotFoundError(f"no clips under {root}: a clip holds one of {[str(m) for m in markers]} "
                                f"(an out/<stem>/ tree gets geometry/scene.npz from the 'fuse' stage)")
    viewer = Viewer(host="0.0.0.0", port=port, theme=replace(LIGHT, up=up, control_width="large"))
    App(viewer, clips, body=load_body(device="cpu"), **opts)
    print(f"[view_3d] serving {len(clips)} clip(s) at http://localhost:{viewer.port}", flush=True)
    viewer.server.sleep_forever()
