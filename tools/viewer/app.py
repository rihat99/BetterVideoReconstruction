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

from .helpers import _background_image, _camera_pose_at
from .loading import ClipData, load_clip_data
from .scene import ClipScene, _SKEL_ALPHA

_LAYER_LABELS = (("body", "SMPL-X body"), ("skeleton", "body skeleton"),
                 ("contact_frames", "contact frames"),
                 ("frame_normals", "frame normal arrows"),
                 ("human", "human points"), ("scene", "scene points"),
                 ("normals", "scene normals"),
                 ("cameras", "cameras"), ("gravity", "gravity"),
                 ("ground", "ground plane"))

# Force/torque display toggles (their own sidebar folder; only bite while a force
# body source — ``dynamics`` or ``kindyn_1`` — is active). Joint spheres stay the
# neutral skeleton grey; the ``joint torques`` toggle turns them yellow and sizes
# them by torque magnitude.
# The folder also holds the ``force display`` dropdown: "per-joint" accumulates each
# joint's frame forces into one arrow at the joint (hand forces summed at the
# wrist), "per-frame" draws each contact
# frame's own solved force where it acts (sources that store ``frame_forces`` only).
_DYN_LAYER_LABELS = (("dyn_torque", "joint torques (sphere size)"),
                     ("dyn_fingers", "fingers / palm"),
                     ("dyn_forces", "contact forces"),
                     ("dyn_root_force", "root force"),
                     ("dyn_root_torque", "root torque"),
                     ("dyn_sensor", "sensor forces (measured)"))

# Layer checkboxes that only apply to the contact-frame overlay — hidden unless the
# active body source has contact labels (see ``ClipScene.frame_sources``).
_FRAME_LAYER_KEYS = ("contact_frames", "frame_normals")

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
                 video: bool = True, video_dir: Path | None = None,
                 up: str = "-y", groups: dict[str, str] | None = None) -> None:
        self.server = server
        self.clips = clips
        #: Clip name → group label (e.g. the athletics DB's sport); a ``group``
        #: dropdown narrows the scene list to one label. Empty = no grouping.
        self.groups = groups or {}
        self._all_clips = list(clips)
        self.opts = dict(camera_stride=camera_stride, device=device,
                         show_smplx=show_smplx, human_max_points=human_max_points,
                         video=video, video_dir=video_dir)
        self.layers = {"body": show_smplx, "skeleton": True, "contact_frames": True,
                       "frame_normals": False,
                       "human": False, "scene": True, "normals": False,
                       "cameras": True, "gravity": True, "ground": True,
                       # force/torque section (torque scaling + fingers off by default)
                       "dyn_torque": False, "dyn_fingers": False, "dyn_forces": True,
                       "dyn_force_mode": "per-joint",
                       "dyn_root_force": True, "dyn_root_torque": True,
                       # measured wall-sensor arrows (climb_wall clips only) — off until
                       # asked for; the checkbox is hidden when the clip has no sensor npz
                       "dyn_sensor": False}
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
        # Playback clock: (playhead in fractional frames, wall-clock instant it was
        # advanced) as ONE tuple so the lock-free onboard-camera thread reads a
        # consistent pair; ``None`` while paused. That thread extrapolates from it
        # between scene updates, so the camera glides at 60 Hz however long a frame
        # takes to pose.
        self._clock: tuple[float, float] | None = None
        self._play_frame = -1                       # last frame the loop set on the slider
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
                if self.groups:
                    labels = sorted({self.groups[c.name] for c in clips if c.name in self.groups})
                    self.group = g.add_dropdown("sport", ("all", *labels), initial_value="all")
                    self.group.on_update(lambda _: self._on_group())
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
        # (``_update_video``); hidden when the clip has no source video. With
        # ``[viewer].video = false`` the pane is never created and the footage is
        # never decoded — streaming a JPEG per frame lags remote sessions.
        self.video_img = None
        self.video_view = None
        if video:
            with g.add_folder("video"):
                # Multi-view clips (a calibrated rig) pick which camera the pane
                # plays; hidden for the single-camera clips that have only one.
                self.video_view = g.add_dropdown("view", ("—",), initial_value="—",
                                                 visible=False)
                self.video_view.on_update(lambda _: self._on_video_view())
                self.video_img = g.add_image(_background_image(), format="jpeg",
                                             jpeg_quality=80)
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
        # Force/torque gets its own section — toggles that apply while a force body
        # source (dynamics / kindyn_1) is active. Collapsed by default to keep it tidy;
        # the whole folder is hidden while a non-force source is active.
        self.dyn_folder = g.add_folder("dynamics", expand_by_default=False)
        with self.dyn_folder:
            d = g.add_dropdown("force display", ("per-joint", "per-frame"),
                               initial_value=self.layers["dyn_force_mode"])
            d.on_update(self._make_layer_cb("dyn_force_mode"))
            self.checkboxes["dyn_force_mode"] = d
            for key, label in _DYN_LAYER_LABELS:
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

    def _on_group(self) -> None:
        """Narrow the scene list to one group label (``"all"`` restores every clip),
        then jump to its first scene."""
        self.clips = clips_in_group(self._all_clips, self.groups, self.group.value)
        self.example.options = tuple(c.name for c in self.clips)
        self.example.value = self.clips[0].name  # fires _on_example → load_clip

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
                if key in ("scene", "normals", "cameras", "gravity", "ground"):
                    self.scene.set_static_layer(key, self.layers[key])
                else:  # body / skeleton / contact frames / human are per-frame → re-apply
                    self.scene.apply_frame(self._cur())
        return cb

    def _on_body(self) -> None:
        with self._lock:
            if self.scene is not None and self.body.value not in ("", "(none)"):
                self.scene.set_active_body(self.body.value)
                self._apply_ctx_visibility(self.body.value)

    def _apply_ctx_visibility(self, src: str) -> None:
        """Show only the controls that affect the active body source ``src`` (instant, no
        rebuild; hidden controls keep their value). The scene reports which sources each
        control category applies to; global layers (scene / cameras / human / …) never move.
        """
        sc = self.scene
        if sc is None:
            return
        for key in _FRAME_LAYER_KEYS:
            self.checkboxes[key].visible = src in sc.frame_sources
        self.dyn_folder.visible = src in sc.force_sources        # the whole dyn_* section
        self.checkboxes["dyn_sensor"].visible = sc.sensors is not None
        self.body_opacity.visible = src in sc.translucency_sources

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
        # Either lock the camera onto the reconstructed one or frame it on the human.
        # (No camera-moved callback: while onboard mode is on, the 60 Hz camera
        # thread re-drives every client anyway, which is what locks out orbiting —
        # a per-event snap-back would answer each of the browser's ~50 Hz camera
        # echoes with another send, a feedback loop that lags the whole client.)
        with self._lock:
            scene = self.scene
            onboard = self.follow.value and scene is not None
            if onboard:
                self._drive_camera(client, self._playhead(scene), scene.data)
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
                self._drive_all(self._playhead(self.scene))
            else:
                for client in self.server.get_clients().values():
                    self._reset_camera(client)

    def _reset_camera(self, client) -> None:
        """Restore the standard view: scene up + captured FOV + home framing."""
        client.camera.up_direction = self._up_vec
        if self._std_fov is not None:
            client.camera.fov = self._std_fov
        self._frame_camera(client)

    def _drive_all(self, t: float) -> None:
        data = self.scene.data
        for client in self.server.get_clients().values():
            self._drive_camera(client, t, data)

    def _drive_camera(self, client, t: float, data: ClipData) -> None:
        """Place ``client``'s camera on the reconstructed trajectory at playhead
        ``t`` (fractional frames: pose and vertical FOV interpolated between frames).
        The four camera fields go out as one atomic batch, so the browser never
        renders a half-updated pose."""
        pos, look_at, up, fov = _camera_pose_at(data.extr, data.fov_y, t)
        cam = client.camera
        with client.atomic():
            cam.position = tuple(float(x) for x in pos)
            cam.up_direction = tuple(float(x) for x in up)
            cam.look_at = tuple(float(x) for x in look_at)
            cam.fov = float(fov)

    def _playhead(self, scene: ClipScene) -> float:
        """Where playback is, in fractional frames: the slider while paused, else
        the wall clock's extrapolation of the last playback tick (wrapping)."""
        clock = self._clock
        if clock is None:
            return float(self._cur())
        pos, t0 = clock
        return (pos + (time.monotonic() - t0) * float(self.fps.value)) % scene.n_frames

    def _camera_loop(self) -> None:
        """Onboard-camera thread (60 Hz): glide every browser along the reconstructed
        trajectory at the current playhead. Runs lock-free off the scene thread —
        posing a frame can take tens of ms and the camera must not wait for it."""
        while True:
            time.sleep(1.0 / 60.0)
            scene = self.scene
            if not self.follow.value or scene is None:
                continue
            t = self._playhead(scene)
            for client in self.server.get_clients().values():
                self._drive_camera(client, t, scene.data)

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

    def _on_video_view(self) -> None:
        """Play another calibrated view in the sidebar pane (already frame-aligned)."""
        with self._lock:
            views = None if self.scene is None else self.scene.data.video_views
            if views and self.video_view.value in views:
                self.video_frames = views[self.video_view.value]
                self._update_video(self._cur())

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
                                      video=self.opts["video"],
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
            # Reapply here too: an unchanged default source name across clips fires no
            # dropdown callback, but the new clip's source categories may differ.
            self._apply_ctx_visibility(self.body.value)

            # Recreate the frame slider — its range is per-clip (viser sliders
            # can't be re-ranged in place). Rebuild it in the playback folder.
            if self.frame is not None:
                self.frame.remove()
            with self.play_folder:
                self.frame = self.server.gui.add_slider("frame", 0, max(data.n_frames - 1, 1), 1, 0)
            self.frame.on_update(lambda _: self._apply_current())

            self.video_frames = data.video_frames
            if self.video_view is not None:
                names = tuple(data.video_views or ())
                self.video_view.visible = len(names) > 1
                if names:
                    self.video_view.options = names
                    # Assigning an unchanged value fires no callback, so point the pane
                    # at this clip's first view here rather than relying on one.
                    self.video_view.value = names[0]
                    self.video_frames = data.video_views[names[0]]
            if self.video_img is not None:
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
        """Playback loop: advance the playhead by wall-clock time while ``play`` is
        on, so playback runs at the chosen fps however long a frame takes to pose
        (a slow frame is skipped, not stretched)."""
        threading.Thread(target=self._camera_loop, daemon=True).start()
        while True:
            with self._lock:
                playing = (self.frame is not None and self.play.value
                           and self.scene is not None and self.scene.n_frames > 0)
                if playing:
                    now = time.monotonic()
                    fps = float(self.fps.value)
                    cur = int(self.frame.value)
                    if self._clock is None or cur != self._play_frame:   # start, or scrubbed
                        self._clock = (float(cur), now)
                    pos = (self._clock[0] + (now - self._clock[1]) * fps) % self.scene.n_frames
                    self._clock = (pos, now)
                    f = int(pos)
                    if f != cur:
                        # Setting ``.value`` fires the frame callback (same thread,
                        # reentrant lock) → apply_frame; no explicit call needed.
                        self.frame.value = f
                    self._play_frame = f
                    wait = (f + 1 - pos) / fps                  # until the next frame is due
                else:
                    self._clock = None
                    wait = 0.05
            time.sleep(wait)


def clips_in_group(clips: list[Path], groups: dict[str, str], label: str) -> list[Path]:
    """The clips whose group label is ``label``; ``"all"`` keeps every clip."""
    if label == "all":
        return list(clips)
    return [c for c in clips if groups.get(c.name) == label]


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
            video: bool = True, video_dir: Path | None = None,
            groups: dict[str, str] | None = None) -> None:
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
                    human_max_points=human_max_points, video=video, video_dir=video_dir,
                    up=up, groups=groups)
    viewer.run()
