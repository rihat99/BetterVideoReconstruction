"""Viewer — persistent GUI + example switching, and the ``view_3d`` entry point.

:class:`Viewer` owns the persistent viser sidebar (layers / body source /
playback / sizes) and swaps the active :class:`~tools.viewer.scene.ClipScene` as
the example dropdown changes. :func:`view_3d` discovers the clips under a root,
starts the viser server, and runs the playback loop.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from .helpers import _background_image
from .loading import ClipData, load_clip_data
from .scene import ClipScene

_LAYER_LABELS = (("body", "SMPL-X body"), ("human", "human points"),
                 ("scene", "scene points"), ("cameras", "cameras"), ("gravity", "gravity"))
# scene rendering choices; the label lower-cases to the internal method name
# ("points" / "bpa" / "poisson").
_SURFACE_OPTIONS = ("points", "BPA", "Poisson")


class Viewer:
    """Owns the persistent sidebar and swaps the active :class:`ClipScene`."""

    def __init__(self, server, clips: list[Path], *, point_size: float,
                 human_point_size: float, camera_scale: float, camera_stride: int,
                 show_smplx: bool, device: str, human_max_points: int = 0,
                 scene_surface: str = "points") -> None:
        self.server = server
        self.clips = clips
        self.opts = dict(camera_stride=camera_stride, device=device,
                         show_smplx=show_smplx, human_max_points=human_max_points)
        self.layers = {"body": show_smplx, "human": True, "scene": True,
                       "cameras": True, "gravity": False}
        self._surface = scene_surface    # scene rendering: "points" / "bpa" / "poisson"
        self.cache: dict[Path, ClipData] = {}
        self.scene: ClipScene | None = None
        # viser fires GUI callbacks on its own thread and setting a slider's
        # ``.value`` invokes its callback synchronously — so switching examples
        # mid-playback would let ``load_clip`` dispose a clip while the playback
        # loop is still posing it. This lock serializes every scene mutation
        # (reentrant: the playback loop holds it while the value-set fires the
        # frame callback on the same thread).
        self._lock = threading.RLock()
        # Frame each new browser on the clip's human root (see ``_frame_camera``).
        server.on_client_connect(self._on_client_connect)

        g = server.gui
        g.add_markdown("### BetterVideoReconstruction · 3D viewer")
        if len(clips) > 1:
            self.example = g.add_dropdown("example", tuple(c.name for c in clips),
                                          initial_value=clips[0].name)
            self.example.on_update(lambda _: self._on_example())
        with g.add_folder("layers"):
            self.checkboxes = {}
            for key, label in _LAYER_LABELS:
                c = g.add_checkbox(label, self.layers[key])
                c.on_update(self._make_layer_cb(key))
                self.checkboxes[key] = c
            init_surface = next((o for o in _SURFACE_OPTIONS if o.lower() == self._surface),
                                "points")
            self.surface_dd = g.add_dropdown("scene surface", _SURFACE_OPTIONS,
                                             initial_value=init_surface)
            self.surface_dd.on_update(lambda _: self._on_surface())
        with g.add_folder("body"):
            self.body = g.add_dropdown("source", ("(none)",), initial_value="(none)")
            self.body.on_update(lambda _: self._on_body())
        self.play_folder = g.add_folder("playback")
        with self.play_folder:
            self.play = g.add_checkbox("play", False)
            self.fps = g.add_slider("fps", 1, 60, 1, 30)
        # Sizes sit below playback: independent point + camera-frustum radii (metres).
        sc0 = float(min(max(point_size, 0.002), 0.08))
        hu0 = float(min(max(human_point_size, 0.002), 0.08))
        cam0 = float(min(max(camera_scale, 0.02), 0.6))
        with g.add_folder("sizes"):
            self.scene_ps = g.add_slider("scene points", 0.002, 0.08, 0.002, sc0)
            self.human_ps = g.add_slider("human points", 0.002, 0.08, 0.002, hu0)
            self.cam_size = g.add_slider("camera frustum", 0.02, 0.6, 0.02, cam0)
        self.scene_ps.on_update(lambda _: self._on_scene_ps())
        self.human_ps.on_update(lambda _: self._on_human_ps())
        self.cam_size.on_update(lambda _: self._on_camera_scale())
        self.frame = None
        self.info = g.add_markdown("")

        self.load_clip(clips[0])

    # -- GUI callbacks --
    def _on_example(self) -> None:
        clip = next((c for c in self.clips if c.name == self.example.value), None)
        if clip is not None:
            self.load_clip(clip)

    def _make_layer_cb(self, key: str):
        def cb(_) -> None:
            self.layers[key] = self.checkboxes[key].value
            with self._lock:
                if self.scene is None:
                    return
                if key in ("scene", "cameras", "gravity"):
                    self.scene.set_static_layer(key, self.layers[key])
                else:  # body / human are per-frame → re-apply the current frame
                    self.scene.apply_frame(self._cur())
        return cb

    def _on_body(self) -> None:
        with self._lock:
            if self.scene is not None and self.body.value not in ("", "(none)"):
                self.scene.set_active_body(self.body.value)

    def _on_surface(self) -> None:
        self._surface = self.surface_dd.value.lower()
        with self._lock:
            if self.scene is not None:
                self.scene.set_scene_surface(self._surface)

    def _on_scene_ps(self) -> None:
        with self._lock:
            if self.scene is not None:
                self.scene.set_scene_point_size(float(self.scene_ps.value))

    def _on_human_ps(self) -> None:
        with self._lock:
            if self.scene is not None:
                self.scene.set_human_point_size(float(self.human_ps.value))

    def _on_camera_scale(self) -> None:
        with self._lock:
            if self.scene is not None:
                self.scene.set_camera_scale(float(self.cam_size.value))

    def _on_client_connect(self, client) -> None:
        self._frame_camera(client)

    def _frame_camera(self, client) -> None:
        """Aim ``client``'s camera at the current clip's human root, so orbiting
        pivots on the climber rather than the world origin."""
        with self._lock:
            scene = self.scene
            if scene is None:
                return
            focus, pos = scene.focus_point, scene.view_position()
        # position first (it drags look_at along), then look_at — per viser's API.
        client.camera.position = tuple(float(x) for x in pos)
        client.camera.look_at = tuple(float(x) for x in focus)

    def _apply_current(self) -> None:
        with self._lock:
            if self.scene is not None:
                self.scene.apply_frame(self._cur())

    def _cur(self) -> int:
        return int(self.frame.value) if self.frame is not None else 0

    def load_clip(self, clip: Path) -> None:
        print(f"[view_3d] loading {clip.name} …", flush=True)
        with self._lock:
            if self.scene is not None:
                self.scene.dispose()
            data = self.cache.get(clip)
            if data is None:
                data = load_clip_data(clip, device=self.opts["device"],
                                      show_smplx=self.opts["show_smplx"],
                                      human_max_points=self.opts["human_max_points"])
                self.cache[clip] = data
            self.scene = ClipScene(
                self.server, data, self.layers,
                scene_point_size=float(self.scene_ps.value),
                human_point_size=float(self.human_ps.value),
                camera_scale=float(self.cam_size.value),
                camera_stride=self.opts["camera_stride"], scene_surface=self._surface)

            srcs = self.scene.body_source_names
            self.body.options = tuple(srcs) if srcs else ("(none)",)
            self.body.value = self.scene.default_body or "(none)"
            if srcs:
                self.scene.set_active_body(self.body.value)

            # Recreate the frame slider — its range is per-clip (viser sliders
            # can't be re-ranged in place). Rebuild it in the playback folder.
            if self.frame is not None:
                self.frame.remove()
            with self.play_folder:
                self.frame = self.server.gui.add_slider("frame", 0, max(data.n_frames - 1, 1), 1, 0)
            self.frame.on_update(lambda _: self._apply_current())

            self.info.content = self.scene.info_md
            self.scene.apply_frame(0)
        # Re-aim any already-connected browsers at the new clip's human root.
        for client in self.server.get_clients().values():
            self._frame_camera(client)
        print(f"[view_3d] {data.name}: {data.n_frames} frames · "
              f"{len(data.scene_points)} scene pts · bodies {srcs}", flush=True)

    def run(self) -> None:
        """Playback loop: advance the frame while ``play`` is on."""
        while True:
            with self._lock:
                playing = (self.frame is not None and self.play.value
                           and self.scene is not None and self.scene.n_frames > 0)
                if playing:
                    # Setting ``.value`` fires the frame callback (same thread,
                    # reentrant lock) → apply_frame; no explicit call needed.
                    self.frame.value = (int(self.frame.value) + 1) % self.scene.n_frames
            time.sleep(1.0 / float(self.fps.value) if playing else 0.05)


def _discover_clips(root: Path) -> list[Path]:
    """Clips = ``root`` itself if it holds ``geometry/scene.npz``, else its clip subdirs."""
    root = Path(root)
    if (root / "geometry" / "scene.npz").exists():
        return [root]
    if not root.is_dir():
        return []
    return [d for d in sorted(root.iterdir())
            if d.is_dir() and (d / "geometry" / "scene.npz").exists()]


def view_3d(root: Path, *, port: int = 8082, point_size: float = 0.02,
            human_point_size: float = 0.01, camera_scale: float = 0.1,
            camera_stride: int = 1, up: str = "-y", show_smplx: bool = True,
            device: str = "cuda", human_max_points: int = 6000,
            scene_surface: str = "points") -> None:
    import viser

    clips = _discover_clips(root)
    if not clips:
        raise FileNotFoundError(
            f"no clips with geometry/scene.npz under {root} — run the 'fuse' stage first")

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction(up)  # VGGT world is OpenCV y-down → up ≈ -y
    server.scene.set_background_image(_background_image())    # dark grey backdrop
    # viser's default lights stay on for ambient fill; each clip adds one key light
    # from the camera side (ClipScene._build_lights).
    print(f"[view_3d] serving {len(clips)} example(s) at http://localhost:{port}", flush=True)
    viewer = Viewer(server, clips, point_size=point_size, human_point_size=human_point_size,
                    camera_scale=camera_scale, camera_stride=camera_stride,
                    show_smplx=show_smplx, device=device,
                    human_max_points=human_max_points, scene_surface=scene_surface)
    viewer.run()
