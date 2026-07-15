"""Viewer — persistent GUI + example switching, and the ``view_3d`` entry point.

:class:`Viewer` owns the persistent viser sidebar (layers / body source /
playback / sizes) and swaps the active :class:`~tools.viewer.scene.ClipScene` as
the example dropdown changes. :func:`view_3d` discovers the clips under a root,
starts the viser server, and runs the playback loop.
"""
from __future__ import annotations

import random
import threading
import time
from pathlib import Path

import numpy as np

from .helpers import _background_image, _camera_follow_target
from .loading import ClipData, load_clip_data
from .scene import ClipScene, _SKEL_ALPHA

_LAYER_LABELS = (("body", "SMPL-X body"), ("skeleton", "body skeleton"),
                 ("confidence", "contact confidence"),
                 ("human", "human points"), ("scene", "scene points"),
                 ("normals", "scene normals"),
                 ("cameras", "cameras"), ("gravity", "gravity"))

# World-frame up vector per ``--up`` axis, mirroring ``SceneApi.set_up_direction``.
# ``set_up_direction`` rotates the scene root (it does NOT set the camera's
# ``up_direction``), so the standard view's camera up is exactly this vector — the
# value to restore when leaving onboard mode.
_UP_VECTORS = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
               "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
               "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}


class Viewer:
    """Owns the persistent sidebar and swaps the active :class:`ClipScene`."""

    def __init__(self, server, clips: list[Path], *, point_size: float,
                 human_point_size: float, camera_scale: float, camera_stride: int,
                 show_smplx: bool, device: str, human_max_points: int = 0,
                 video_dir: Path | None = None, up: str = "-y") -> None:
        self.server = server
        self.clips = clips
        self.opts = dict(camera_stride=camera_stride, device=device,
                         show_smplx=show_smplx, human_max_points=human_max_points,
                         video_dir=video_dir)
        self.layers = {"body": show_smplx, "skeleton": False, "confidence": True,
                       "human": True, "scene": True, "normals": False,
                       "cameras": True, "gravity": False}
        self._surface = "points"    # surface-mesh rendering is off (code kept in ClipScene)
        self.cache: dict[Path, ClipData] = {}
        self.scene: ClipScene | None = None
        # Standard-view camera bits to restore when leaving onboard mode: the up is
        # the scene up (from ``--up``; position/look-at come live from
        # ``_frame_camera``), and the FOV is snapshotted from a client the first time
        # onboard is enabled (a fresh handle still holds the three.js default, so
        # capturing at connect would be stale).
        self._up_vec = _UP_VECTORS.get(up, (0.0, -1.0, 0.0))
        self._std_fov: float | None = None
        # viser fires GUI callbacks on its own thread and setting a slider's
        # ``.value`` invokes its callback synchronously — so switching examples
        # mid-playback would let ``load_clip`` dispose a clip while the playback
        # loop is still posing it. This lock serializes every scene mutation
        # (reentrant: the playback loop holds it while the value-set fires the
        # frame callback on the same thread).
        self._lock = threading.RLock()

        g = server.gui
        g.add_markdown("### BetterVideoReconstruction · 3D viewer")
        # Sidebar order (top→bottom): example · video · playback · body · layers · sizes.
        # ``example``/``pos`` stay None for a single clip (no switcher shown).
        self.example = None
        self.pos = None
        if len(clips) > 1:
            with g.add_folder("example"):
                self.example = g.add_dropdown("scene", tuple(c.name for c in clips),
                                              initial_value=clips[0].name)
                self.example.on_update(lambda _: self._on_example())
                # Prev / random / next pager (the BetterParsing scene-viewer pattern):
                # each just moves the dropdown to another clip, whose callback loads it.
                g.add_button("◀ prev").on_click(lambda _: self._on_nav(-1))
                g.add_button("random").on_click(lambda _: self._on_nav(0))
                g.add_button("next ▶").on_click(lambda _: self._on_nav(1))
                self.pos = g.add_markdown(self._pos_md(0))
        # Video pane: the clip's original footage, synced to the frame slider
        # (``_update_video``); hidden when the clip has no source video.
        with g.add_folder("video"):
            self.video_img = g.add_image(_background_image(), format="jpeg", jpeg_quality=80)
        self.video_frames = None
        self.play_folder = g.add_folder("playback")
        with self.play_folder:
            self.play = g.add_checkbox("play", False)
            self.fps = g.add_slider("fps", 1, 60, 1, 30)
            # Onboard camera: ride the reconstructed camera (pose + FOV) each frame
            # and lock the user out of orbiting while it's on. Off by default.
            self.follow = g.add_checkbox("onboard camera", False)
            self.follow.on_update(lambda _: self._on_follow())
        with g.add_folder("body"):
            self.body = g.add_dropdown("source", ("(none)",), initial_value="(none)")
            self.body.on_update(lambda _: self._on_body())
        with g.add_folder("layers"):
            self.checkboxes = {}
            for key, label in _LAYER_LABELS:
                c = g.add_checkbox(label, self.layers[key])
                c.on_update(self._make_layer_cb(key))
                self.checkboxes[key] = c
        # Sizes: independent point + camera-frustum radii (metres).
        sc0 = float(min(max(point_size, 0.002), 0.08))
        hu0 = float(min(max(human_point_size, 0.002), 0.08))
        cam0 = float(min(max(camera_scale, 0.02), 0.6))
        with g.add_folder("sizes"):
            self.scene_ps = g.add_slider("scene points", 0.002, 0.08, 0.002, sc0)
            self.human_ps = g.add_slider("human points", 0.002, 0.08, 0.002, hu0)
            self.cam_size = g.add_slider("camera frustum", 0.02, 0.6, 0.02, cam0)
            # Kinematic mesh opacity while the skeleton overlays it (0.1 see-through → 1.0 opaque).
            self.body_opacity = g.add_slider("body opacity", 0.1, 1.0, 0.05, _SKEL_ALPHA)
        self.scene_ps.on_update(lambda _: self._on_scene_ps())
        self.human_ps.on_update(lambda _: self._on_human_ps())
        self.cam_size.on_update(lambda _: self._on_camera_scale())
        self.body_opacity.on_update(lambda _: self._on_body_opacity())
        self.frame = None
        self.info = g.add_markdown("")

        # Frame each new browser on the clip's human root, and lock its camera to
        # the reconstructed one while onboard mode is on (see ``_on_client_connect``).
        server.on_client_connect(self._on_client_connect)
        self.load_clip(clips[0])

    # -- GUI callbacks --
    def _on_example(self) -> None:
        clip = next((c for c in self.clips if c.name == self.example.value), None)
        if clip is not None:
            self.load_clip(clip)

    def _pos_md(self, idx: int) -> str:
        return f"scene **{idx + 1} / {len(self.clips)}**"

    def _on_nav(self, delta: int) -> None:
        """Prev (``-1``) / next (``+1``) with wraparound, or a different random clip
        (``delta == 0``). Moves the ``example`` dropdown, whose callback loads it."""
        n = len(self.clips)
        if self.example is None or n < 2:
            return
        cur = next((i for i, c in enumerate(self.clips) if c.name == self.example.value), 0)
        nxt = (cur + (random.randrange(1, n) if delta == 0 else delta)) % n
        self.example.value = self.clips[nxt].name

    def _make_layer_cb(self, key: str):
        def cb(_) -> None:
            self.layers[key] = self.checkboxes[key].value
            with self._lock:
                if self.scene is None:
                    return
                if key in ("scene", "normals", "cameras", "gravity"):
                    self.scene.set_static_layer(key, self.layers[key])
                else:  # body / skeleton / confidence / human are per-frame → re-apply the frame
                    self.scene.apply_frame(self._cur())
        return cb

    def _on_body(self) -> None:
        with self._lock:
            if self.scene is not None and self.body.value not in ("", "(none)"):
                self.scene.set_active_body(self.body.value)

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

    def _on_body_opacity(self) -> None:
        with self._lock:
            if self.scene is not None:
                self.scene.set_mesh_opacity(float(self.body_opacity.value))

    def _on_client_connect(self, client) -> None:
        # Snap the camera back whenever the user moves it while onboard mode is on
        # (viser has no "disable controls"); then either lock it onto the
        # reconstructed camera or frame it on the human.
        client.camera.on_update(lambda _cam: self._on_cam_lock(client))
        with self._lock:
            onboard = self.follow.value and self.scene is not None
            if onboard:
                self._drive_camera(client, self._cur())
        if not onboard:
            self._frame_camera(client)

    def _on_follow(self) -> None:
        """Toggle onboard mode: on → snap every client onto the camera; off →
        return them to the standard home view (not wherever onboard left them)."""
        with self._lock:
            if self.scene is None:
                return
            if self.follow.value:
                if self._std_fov is None:            # capture the standard FOV once
                    for client in self.server.get_clients().values():
                        self._std_fov = float(client.camera.fov)
                        break
                self._drive_all(self._cur())
            else:
                for client in self.server.get_clients().values():
                    self._reset_camera(client)

    def _reset_camera(self, client) -> None:
        """Restore the standard view: scene up + captured FOV + home framing."""
        client.camera.up_direction = self._up_vec
        if self._std_fov is not None:
            client.camera.fov = self._std_fov
        self._frame_camera(client)

    def _drive_all(self, f: int) -> None:
        for client in self.server.get_clients().values():
            self._drive_camera(client, f)

    def _drive_camera(self, client, f: int) -> None:
        """Place ``client``'s camera on the reconstructed camera at frame ``f`` —
        pose (position / look-at / up) plus its vertical FOV."""
        if self.scene is None or f >= len(self.scene.data.extr):
            return
        data = self.scene.data
        pos, look_at, up = _camera_follow_target(data.extr[f])
        cam = client.camera
        cam.position = tuple(float(x) for x in pos)
        cam.up_direction = tuple(float(x) for x in up)
        cam.look_at = tuple(float(x) for x in look_at)
        if f < len(data.fov_y):
            cam.fov = float(data.fov_y[f])

    def _on_cam_lock(self, client) -> None:
        """Camera-moved callback: while onboard mode is on, snap back to the
        reconstructed camera unless we're already there (our own echo)."""
        with self._lock:
            if not (self.follow.value and self.scene is not None):
                return
            f = self._cur()
            if f >= len(self.scene.data.extr):
                return
            pos = _camera_follow_target(self.scene.data.extr[f])[0]
            # 1 cm deadzone: imperceptible in these metric scenes, but well above any
            # world→client→world round-trip noise, so our own echo never re-fires the
            # snap-back (an idle message loop) — a real orbit moves the eye far more.
            if float(np.linalg.norm(np.asarray(client.camera.position) - pos)) < 1e-2:
                return
            self._drive_camera(client, f)

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
            self._update_video(self._cur())
            if self.follow.value:
                self._drive_all(self._cur())

    def _update_video(self, f: int) -> None:
        """Point the sidebar video pane at frame ``f`` (no-op if the clip has no video)."""
        vf = self.video_frames
        if vf is not None:
            self.video_img.image = vf[max(0, min(int(f), len(vf) - 1))]

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
                                      human_max_points=self.opts["human_max_points"],
                                      video_dir=self.opts["video_dir"])
                self.cache[clip] = data
            self.scene = ClipScene(
                self.server, data, self.layers,
                scene_point_size=float(self.scene_ps.value),
                human_point_size=float(self.human_ps.value),
                camera_scale=float(self.cam_size.value),
                camera_stride=self.opts["camera_stride"], scene_surface=self._surface,
                mesh_opacity=float(self.body_opacity.value))

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

            self.video_frames = data.video_frames
            self.video_img.visible = self.video_frames is not None
            if self.video_frames is not None:
                self.video_img.image = self.video_frames[0]
            # Default playback speed to this scene's own fps (24/25/30/60 differ).
            if data.fps:
                self.fps.value = int(np.clip(round(data.fps), 1, 60))

            self.info.content = self.scene.info_md
            if self.pos is not None:
                idx = next((i for i, c in enumerate(self.clips) if c.name == clip.name), 0)
                self.pos.content = self._pos_md(idx)
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
            video_dir: Path | None = None) -> None:
    import viser

    clips = _discover_clips(root)
    if not clips:
        raise FileNotFoundError(
            f"no clips with geometry/scene.npz under {root} — run the 'fuse' stage first")

    server = viser.ViserServer(port=port)
    # Wider sidebar for the video pane: "large" is viser's widest built-in
    # (24em ≈ 1.2× the 20em default; the theme API has no finer control).
    server.gui.configure_theme(control_width="large")
    server.scene.set_up_direction(up)  # VGGT world is OpenCV y-down → up ≈ -y
    server.scene.set_background_image(_background_image())    # solid white backdrop
    # Lighting is viser's default (no custom key light).
    print(f"[view_3d] serving {len(clips)} example(s) at http://localhost:{port}", flush=True)
    viewer = Viewer(server, clips, point_size=point_size, human_point_size=human_point_size,
                    camera_scale=camera_scale, camera_stride=camera_stride,
                    show_smplx=show_smplx, device=device,
                    human_max_points=human_max_points, video_dir=video_dir, up=up)
    viewer.run()
