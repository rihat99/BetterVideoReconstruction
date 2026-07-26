"""``ClipScene`` — all viser nodes for one clip, grouped into toggleable layers.

Builds every scene node for a single clip from its :class:`~tools.viewer.loading.ClipData`
— the scene cloud, the per-point normal arrows (debug), the camera frustums +
centre path, the gravity arrow, the per-frame human
points, the skinned SMPL-X bodies, and the force/torque skeletons (dynamics /
kindyn_1) — and
moves them together with the frame slider. One ``/clip`` root frame owns them all, so :meth:`ClipScene.dispose`
tears the whole clip down in a single removal.
"""
from __future__ import annotations

import numpy as np

from .helpers import (
    _BODY_COLORS,
    _camera_world_pose,
    _fit_ground_plane,
    _time_color,
    _unit,
    _unlit_glb,
    _wxyz_shortest_arc_z,
)
from .loading import _FINGER_TOKENS, ClipData, ensure_scene_mesh

# dynamics sphere-floor / bone radius / max torque grow factor / metres-per-newton for
# the contact-force arrows
_DYN_R0, _DYN_RB, _DYN_MAXS, _DYN_FSCALE = 0.022, 0.012, 3.0, 1.2e-3
# root residual-wrench arrows: metres per newton-metre for the torque part (the force
# part reuses _DYN_FSCALE so it is directly comparable to the contact-force arrows)
_DYN_TSCALE = 4.8e-3
# contact-force arrow colour — a saturated blue that reads against the grey scene
# cloud and the red skeleton
_DYN_FORCE_RGB = (50, 120, 255)
# measured wall-sensor arrows (climb_wall GT): orange, one per channel; below the
# floor a channel is not drawn (the sensors never read exactly zero, and the fixed
# arrow head would clutter the wall)
_SENSOR_RGB, _SENSOR_FLOOR_N = (255, 150, 30), 10.0
# body-skeleton overlay: fixed joint-sphere radius / bone radius (m) and the
# default kinematic mesh opacity while the skeleton shows over it (slider-tunable).
_SKEL_R, _SKEL_RB, _SKEL_ALPHA = 0.028, 0.012, 0.7
# joint-sphere colours — red in scene contact / green free (matches estimate_contacts);
# the flat fallback when the "contact confidence" layer is off (or no confidence was written).
_CONTACT_RGB, _FREE_RGB = (235, 45, 45), (60, 200, 90)
# With the confidence layer on, the sphere is ramped by the per-joint label confidence
# (contacts_1.npz ``label_confidence`` ∈ [0, 1], both classes): a neutral grey at ~0 →
# these darker, saturated endpoints at ~1, so a more-certain label reads darker + bolder.
_CONTACT_HI, _FREE_HI, _CONF_GREY = (150, 12, 12), (14, 120, 42), (150, 150, 150)
# Depth trust override: calm green at trust=1 → bright magenta at trust=0;
# missing depth gaps are a distinct neutral slate grey.
_DEPTH_GOOD, _DEPTH_BAD, _DEPTH_NODATA = (90, 170, 120), (255, 0, 110), (105, 110, 125)
# (``_FINGER_TOKENS`` — the hand joints excluded from the body skeleton — is imported
# from ``loading`` so the skeleton build and the finger→wrist contact fold agree.)
# cap on drawn camera frustums, so a long clip isn't a dense wire mess
_CAM_MAX_FRUSTUMS = 24
# draw the ground grid this many times the fitted floor's data extent, so it reads
# as a floor spanning the scene rather than a small patch under the climber
_GROUND_GRID_MULT = 10.0


def _unit_cylinder():
    """A unit +z cylinder (radius 1, height 1) with a light-grey vertex colour,
    scaled per-frame into skeleton / dynamics bones."""
    import trimesh

    cyl = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=10)
    cyl.visual.vertex_colors = np.array([235, 235, 235, 255], np.uint8)
    return cyl


def _conf_color(hot: bool, conf: float) -> tuple[int, int, int]:
    """Joint-sphere RGB ramped by label confidence: neutral grey at 0 → a dark, saturated
    red (contact) / green (free) at 1. ``conf`` clamped to [0, 1] (non-finite → grey)."""
    hi = _CONTACT_HI if hot else _FREE_HI
    c = float(conf)
    t = min(max(c, 0.0), 1.0) if c == c else 0.0     # c == c is False for NaN
    return tuple(int(round(g + (h - g) * t)) for g, h in zip(_CONF_GREY, hi))


def _depth_color(gap_cm: float, trust: float) -> tuple[int, int, int]:
    """Joint-sphere RGB from depth trust; missing depth data is neutral slate grey."""
    if not np.isfinite(gap_cm) or not np.isfinite(trust):
        return _DEPTH_NODATA
    t = 1.0 - float(np.clip(trust, 0.0, 1.0))
    return tuple(int(round(g + (b - g) * t)) for g, b in zip(_DEPTH_GOOD, _DEPTH_BAD))


class ClipScene:
    """Builds and owns every scene node for one clip; ``dispose`` removes them.

    ``layers`` is the viewer's shared ``{name: bool}`` visibility dict, read live
    so a clip built after the user hid a layer starts with that layer hidden.
    """

    def __init__(self, server, data: ClipData, layers: dict, *,
                 scene_point_size: float, human_point_size: float, camera_scale: float,
                 camera_stride: int, scene_surface: str = "points",
                 mesh_opacity: float = _SKEL_ALPHA) -> None:
        self.server = server
        self.data = data
        self.layers = layers
        self.mesh_opacity = float(mesh_opacity)   # kinematic mesh alpha under the skeleton
        self.scene_point_size = scene_point_size
        self.human_point_size = human_point_size
        self.camera_scale = camera_scale
        self.n_frames = data.n_frames
        self._frame = -1                  # last applied frame
        self._human_frame = -1            # currently-shown human frame (-1 = none)
        # How to render the scene cloud: "points" or a surface mesh method
        # ("bpa" / "poisson"), reconstructed on demand.
        self._surface = scene_surface
        ss = server.scene

        # Everything hangs under one "/clip" root frame, so a single removal
        # cascades to every node this clip added (verified multi-level) — no
        # per-handle bookkeeping is needed for teardown.
        self.root = ss.add_frame("/clip", show_axes=False)

        # -- scene cloud (points or reconstructed mesh) --
        self.scene_node = self._build_scene_node()

        # -- cameras: a frustum per (strided) frame + the full centre path --
        centers = np.stack([_camera_world_pose(e)[0] for e in data.extr]).astype(np.float32)
        # Framing anchor: where the human is. The viewer orbits about ``focus_point``.
        self.camera_mean = centers.mean(0).astype(np.float64)
        self.focus_point = self._compute_focus()
        self.camera_nodes: list = []
        # Thin, time-coloured frustums along the path, capped in number: hundreds of
        # identical blue frustums read as a wire mess, so cap the count (the spline
        # still shows the full path) and ramp their colour over the trajectory.
        n_extr = len(data.extr)
        step = max(camera_stride, int(np.ceil(n_extr / _CAM_MAX_FRUSTUMS)))
        for k in range(0, n_extr, step):
            pos, wxyz = _camera_world_pose(data.extr[k])
            self.camera_nodes.append(ss.add_camera_frustum(
                f"/clip/cameras/cam_{k:04d}", fov=float(data.fov_y[k]), aspect=data.aspect,
                scale=camera_scale, color=_time_color(k / max(n_extr - 1, 1)),
                line_width=1.5, wxyz=wxyz, position=pos))
        self.camera_nodes.append(ss.add_spline_catmull_rom(
            "/clip/cameras/path", centers, color=(255, 140, 0), line_width=2.0))
        self.n_cameras = len(self.camera_nodes) - 1
        # A marker that rides the path to show the camera's position at the current
        # frame (moved in ``apply_frame``); toggles with the cameras layer.
        self.camera_centers = centers
        self.camera_cursor = ss.add_icosphere(
            "/clip/cameras/cursor", radius=max(camera_scale * 0.5, 0.02),
            color=(235, 40, 40), position=tuple(centers[0].tolist()),
            visible=self.layers["cameras"])

        # -- gravity arrow (down), anchored at the mean camera centre --
        self.gravity_nodes: list = []
        if data.gravity is not None:
            g, is_fb = data.gravity["g"], data.gravity["is_fallback"]
            anchor = centers.mean(0)
            glen = camera_scale * 5.0
            tip = anchor + g * glen
            self.gravity_nodes = [
                ss.add_arrows("/clip/gravity/arrow",
                              points=np.array([[anchor, tip]], np.float32),
                              colors=(245, 170, 40) if is_fb else (235, 40, 40),
                              shaft_radius=camera_scale * 0.12,
                              head_radius=camera_scale * 0.25, head_length=camera_scale * 0.4),
            ]

        # -- ground plane: estimation turned off (code kept in ``_build_ground``) --
        self.ground_nodes: list = []
        self.ground_angle: float | None = None

        # -- per-point scene normal arrows (debug layer, built lazily on first toggle) --
        self.normal_nodes: list = []

        # -- human points: one hidden node per (frame, climber), preloaded --
        self.human_nodes: list[list] = [[] for _ in range(self.n_frames)]
        pts, col = data.human_points, data.human_colors
        fid, obj = data.human_fid, data.human_obj
        if len(pts):
            obj_masks = {o: (obj == o) for o in data.obj_ids}
            for f in range(self.n_frames):
                on = fid == f
                for o in data.obj_ids:
                    sel = on & obj_masks[o]
                    if sel.any():
                        self.human_nodes[f].append(ss.add_point_cloud(
                            f"/clip/human/climber_{o:02d}/frame_{f:04d}",
                            pts[sel], col[sel], point_size=self.human_point_size,
                            point_shape="circle", visible=False))

        # -- refined skinned bodies (kinematics / kinematics_2), hidden --
        self.skinned: dict[str, list] = {}
        for name, (faces, oids, people) in data.skinned.items():
            hs: list = []
            for p, person in enumerate(people):
                if person is None:
                    hs.append(None)
                    continue
                ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32),
                                (person["j_rest"].shape[0], 1))
                h = ss.add_mesh_skinned(
                    f"/clip/smplx/{name}/person_{int(oids[p]):02d}", person["v_shaped"], faces,
                    bone_wxyzs=ident, bone_positions=person["j_rest"],
                    skin_weights=person["weights"], color=_BODY_COLORS[p % len(_BODY_COLORS)])
                h.visible = False
                hs.append((h, person))
            self.skinned[name] = hs

        # -- body-skeleton overlays (spheres + bones), hidden — one per refined body
        # source that carries a skeleton dict ("kinematics" from contacts_1,
        # "kinematics_2" from contacts_2). Joints come from that body's per-frame world
        # joints; a sphere is red in scene contact / green free, and the mesh goes
        # translucent while the skeleton overlays it. Fingers are excluded.
        self.skels = data.skeletons
        self.skel_nodes: dict[str, dict[int, dict]] = {}     # source -> oid -> nodes
        self._contact: dict[str, dict[int, np.ndarray]] = {}
        self._conf: dict[str, dict[int, np.ndarray]] = {}    # (N, J) label confidence, folded
        self._depth_gap: dict[str, dict[int, np.ndarray]] = {}
        self._depth_trust: dict[str, dict[int, np.ndarray]] = {}
        self._skel_src: str | None = None      # source whose skeleton nodes are visible
        self._translucent: str | None = None   # active source whose mesh is currently translucent
        for source, skel in self.skels.items():
            body = data.skinned.get(source)
            if body is None:
                continue
            _f, body_oids, people = body
            names, parents = skel["joint_names"], skel["parents"]
            joints = [j for j, nm in enumerate(names)
                      if not any(t in nm for t in _FINGER_TOKENS)]
            jointset = set(joints)
            if skel["contact"] is not None:
                c_oids = (skel["contact_oids"] if skel["contact_oids"] is not None
                          else np.arange(skel["contact"].shape[0]))
                self._contact[source] = {int(o): skel["contact"][i] for i, o in enumerate(c_oids)}
                if skel.get("confidence") is not None:
                    self._conf[source] = {int(o): skel["confidence"][i]
                                          for i, o in enumerate(c_oids)}
                if skel.get("joint_depth_gap_cm") is not None:
                    self._depth_gap[source] = {int(o): skel["joint_depth_gap_cm"][i]
                                               for i, o in enumerate(c_oids)}
                if skel.get("joint_depth_trust") is not None:
                    self._depth_trust[source] = {int(o): skel["joint_depth_trust"][i]
                                                 for i, o in enumerate(c_oids)}
            cyl = _unit_cylinder()
            nodes: dict[int, dict] = {}
            for p, person in enumerate(people):
                if person is None:
                    continue
                oid = int(body_oids[p])
                spheres = {j: ss.add_icosphere(
                    f"/clip/skeleton/{source}/p{oid:02d}/joint_{j:02d}", radius=_SKEL_R,
                    color=_FREE_RGB, position=(0.0, 0.0, 0.0), visible=False) for j in joints}
                bones = []
                for j in joints:
                    par = int(parents[j])
                    if par < 0 or par not in jointset:
                        continue
                    h = ss.add_mesh_trimesh(f"/clip/skeleton/{source}/p{oid:02d}/bone_{j:02d}",
                                            cyl, visible=False)
                    bones.append((h, j, par))
                nodes[oid] = {"spheres": spheres, "bones": bones}
            if nodes:
                self.skel_nodes[source] = nodes

        # -- force/torque skeletons (spheres + bones), one per force source
        #    ("dynamics" and/or "kindyn_1"), hidden; forces added per frame. Node paths
        #    are keyed by the source name, so "dynamics" keeps its original /clip/dynamics
        #    paths and each source's nodes stay independent. --
        self.forces = data.forces                          # {source: force dict}
        self.force_nodes: dict[str, list[dict]] = {}
        self._dyn_finger: dict[str, np.ndarray] = {}       # source -> (J,) finger/palm mask
        for source, dyn in self.forces.items():
            unit_cyl = _unit_cylinder()
            Jd = dyn["jw"].shape[2]
            node_list: list[dict] = []
            for p in range(dyn["P"]):
                oid = int(dyn["oids"][p])
                spheres = [ss.add_icosphere(f"/clip/{source}/p{oid:02d}/sphere_{j:02d}",
                                            radius=_DYN_R0, color=(235, 45, 45),
                                            position=(0.0, 0.0, 0.0), visible=False)
                           for j in range(Jd)]
                bones = []
                for j in range(1, Jd):
                    par = int(dyn["parents"][j])
                    if par < 0:
                        continue
                    h = ss.add_mesh_trimesh(f"/clip/{source}/p{oid:02d}/bone_{j:02d}",
                                            unit_cyl, visible=False)
                    bones.append((h, j, par))
                node_list.append({"oid": oid, "spheres": spheres, "bones": bones,
                                  "forces": None, "root_force": None, "root_torque": None,
                                  "sensor": None})
            self.force_nodes[source] = node_list
            # Per-joint finger/palm mask (for the fingers toggle).
            jnames = dyn.get("joint_names")
            self._dyn_finger[source] = (
                np.array([any(t in n for t in _FINGER_TOKENS) for n in jnames], bool)
                if jnames else np.zeros(Jd, bool))

        # Available body sources (display order) + default. A force source that is also a
        # skinned body ("kindyn_1") already appears via ``data.skinned``; "dynamics" (no
        # mesh) is appended. kinematics_2 is selectable but kinematic_1 stays the default.
        self.body_source_names = (
            list(data.skinned) + [s for s in self.forces if s not in data.skinned])
        self.default_body = next(
            (s for s in ("kinematics", "kinematics_2", "prefit", "original", "dynamics")
             if s in self.body_source_names), None)
        self.active = self.default_body

        # Which sources own a drawable skeleton, and which contextual GUI controls
        # apply to which source (read by the viewer to hide inapplicable controls):
        #   force sources     → the ``dyn_*`` overlays + the torque skeleton
        #   overlay sources   → the ``confidence`` / ``depth_trust`` skeleton colouring
        #   translucency ones → the body-opacity slider (a mesh a skeleton can overlay)
        self.force_sources = set(self.forces)
        # Measured wall-sensor forces (climb_wall clips only) — drawn on top of any
        # active force source by the ``sensor forces`` toggle.
        self.sensors = data.sensors
        self.overlay_sources = set(self.skel_nodes)
        self._skeleton_sources = self.force_sources | self.overlay_sources
        self.translucency_sources = {s for s in self.skinned
                                     if s in self._skeleton_sources}

        self._apply_static_layers()
        self.info_md = self._build_info()

    # -- info panel --
    def _build_info(self) -> str:
        d = self.data
        gtxt = "—"
        if d.gravity is not None:
            g = d.gravity["g"]
            tag = ("default↓" if d.gravity["is_fallback"]
                   else f"GeoCalib, spread {d.gravity['spread']:.1f}°")
            gtxt = f"[{g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}] ({tag})"
        srcs = ", ".join(self.body_source_names) or "—"
        gnd = "" if self.ground_angle is None else f" · ground ∠grav {self.ground_angle:.1f}°"
        return (f"**{d.name}** — {d.n_frames} frames · {len(d.scene_points):,} scene pts · "
                f"{self.n_cameras} cameras · {len(d.human_points):,} human pts · "
                f"climbers {d.obj_ids}\n\nbodies: {srcs} · gravity: {gtxt}{gnd}")

    # -- framing + lighting --
    def _compute_focus(self) -> np.ndarray:
        """Average human root for camera framing — body pelvis, else point centroid.

        Prefers the fitted body's pelvis (joint 0) averaged over its valid frames
        and people; falls back to the human depth-point centroid, then the scene.
        """
        for _faces, _oids, people in self.data.skinned.values():
            roots = []
            for person in people:
                if person is None:
                    continue
                pelvis = person["bone_pos"][:, 0, :]         # (N, 3), NaN on invalid frames
                good = np.isfinite(pelvis).all(1)
                if good.any():
                    roots.append(pelvis[good].mean(0))
            if roots:
                return np.mean(roots, axis=0).astype(np.float64)
        if len(self.data.human_points):
            return self.data.human_points.mean(0).astype(np.float64)
        if len(self.data.scene_points):
            return self.data.scene_points.mean(0).astype(np.float64)
        return np.zeros(3, np.float64)

    def view_position(self) -> np.ndarray:
        """A comfortable opening camera position: backed off from the focus toward
        the mean camera centre, so the clip opens framed from the filmed side."""
        d = self.camera_mean - self.focus_point
        dist = float(np.linalg.norm(d))
        if dist < 1e-3:
            d, dist = np.array([0.0, 0.0, -1.0]), 1.0
        return self.focus_point + (d / dist) * max(dist, 2.0)

    def _build_normals(self) -> None:
        """Draw one arrow per scene point along its stored normal (debug layer).

        Needs ``normals`` in ``geometry/scene.npz`` (written by fuse); silently
        stays empty for older artifacts. Arrows are coloured by direction
        (``(n+1)/2`` → RGB) so flipped or noisy normals stand out, and sized by
        the merge voxel so they read at any scene scale.
        """
        d = self.data
        if d.scene_normals is None or not len(d.scene_points):
            return
        n = np.asarray(d.scene_normals, np.float32)
        length = 2.0 * d.scene_voxel if d.scene_voxel > 0 else 0.05
        seg = np.stack([d.scene_points, d.scene_points + n * length], axis=1)
        colors = ((n + 1.0) * 0.5 * 255.0).astype(np.uint8)
        self.normal_nodes = [self.server.scene.add_arrows(
            "/clip/normals", points=seg.astype(np.float32), colors=colors,
            shaft_radius=0.05 * length, head_radius=0.12 * length,
            head_length=0.3 * length)]

    def _build_ground(self) -> None:
        """Fit a plane to the lowest scene points and draw it as a ground grid.

        (Currently not called — ground estimation is turned off in the viewer.)

        Uses :func:`~tools.viewer.helpers._fit_ground_plane`; the recovered normal
        should sit close to the gravity axis on a level floor, and ``ground_angle``
        records that offset (degrees) for the info panel. The grid (natively in its
        local x-y plane, normal ``+z``) is rotated so ``+z`` aligns with the fitted
        up-normal and placed at the plane centroid.
        """
        g_down = (np.asarray(self.data.gravity["g"], np.float64)
                  if self.data.gravity is not None
                  else np.array([0.0, 1.0, 0.0]))             # VGGT world up is -y ⇒ down +y
        fit = _fit_ground_plane(self.data.scene_points, g_down)
        if fit is None:
            return
        centroid, normal, extent = fit
        up = _unit(-g_down)
        self.ground_angle = float(np.degrees(np.arccos(np.clip(normal @ up, -1.0, 1.0))))
        size = extent * _GROUND_GRID_MULT
        self.ground_nodes = [self.server.scene.add_grid(
            "/clip/ground", width=size, height=size,
            cell_size=max(size / 20.0, 0.05), section_size=max(size / 4.0, 0.25),
            cell_color=(170, 170, 170), section_color=(110, 110, 110),
            plane="xy", wxyz=_wxyz_shortest_arc_z(normal),
            position=tuple(float(x) for x in centroid), visible=self.layers["ground"])]

    # -- layer visibility --
    def _apply_static_layers(self) -> None:
        """Scene / normals / cameras / gravity visibility from the shared ``layers`` dict."""
        self.scene_node.visible = self.layers["scene"]
        if self.layers["normals"] and not self.normal_nodes:
            self._build_normals()
        for h in self.normal_nodes:
            h.visible = self.layers["normals"]
        for h in self.camera_nodes:
            h.visible = self.layers["cameras"]
        self.camera_cursor.visible = self.layers["cameras"]
        for h in self.gravity_nodes:
            h.visible = self.layers["gravity"]

    def set_static_layer(self, name: str, on: bool) -> None:
        """Toggle one static layer (``scene`` / ``normals`` / ``cameras`` / ``gravity``)."""
        if name == "scene":
            self.scene_node.visible = on
        elif name == "normals":
            if on and not self.normal_nodes:
                self._build_normals()
            for h in self.normal_nodes:
                h.visible = on
        elif name == "cameras":
            for h in self.camera_nodes:
                h.visible = on
            self.camera_cursor.visible = on
        elif name == "gravity":
            for h in self.gravity_nodes:
                h.visible = on

    def _build_scene_node(self):
        """Create the ``/clip/scene`` node — a surface mesh for the selected method
        (if it reconstructs), else the point cloud."""
        ss = self.server.scene
        mesh_data = (ensure_scene_mesh(self.data, self._surface)
                     if self._surface != "points" else None)
        if mesh_data is not None:
            v, f, vc, _vn = mesh_data       # normals unused: unlit, and viser recomputes them
            # Render the surface UNLIT and colour-matched to the point cloud
            # (helpers._unlit_glb): the default lit + tone-mapped material is what
            # makes a mesh look dim/washed next to the raw-coloured cloud.
            node = ss.add_glb("/clip/scene", _unlit_glb(v, f, vc))
        else:
            node = ss.add_point_cloud(
                "/clip/scene", self.data.scene_points, self.data.scene_colors,
                point_size=self.scene_point_size, point_shape="circle")
        node.visible = self.layers["scene"]
        return node

    def set_scene_surface(self, surface: str) -> None:
        """Switch the scene rendering: ``points`` / ``bpa`` / ``poisson``."""
        if surface == self._surface:
            return
        self._surface = surface
        self.scene_node.remove()
        self.scene_node = self._build_scene_node()

    def set_scene_point_size(self, size: float) -> None:
        """Live-update the scene point radius (a no-op while a surface mesh shows)."""
        self.scene_point_size = size
        if self._surface == "points":
            self.scene_node.point_size = size

    def set_human_point_size(self, size: float) -> None:
        """Live-update every preloaded human point cloud's radius."""
        self.human_point_size = size
        with self.server.atomic():
            for frame_nodes in self.human_nodes:
                for h in frame_nodes:
                    h.point_size = size

    def set_camera_scale(self, scale: float) -> None:
        """Live-update the camera frustum sizes (the centre-path spline is left as is)."""
        self.camera_scale = scale
        with self.server.atomic():
            for h in self.camera_nodes[:-1]:   # last node is the centre-path spline
                h.scale = scale

    def set_mesh_opacity(self, alpha: float) -> None:
        """Live-update the mesh opacity used while a skeleton overlays the active body's
        mesh (the kinematic 22-joint overlay or a force source's own mesh)."""
        self.mesh_opacity = float(alpha)
        if self._translucent is not None:
            with self.server.atomic():
                for hp in self.skinned.get(self._translucent, []):
                    if hp is not None:
                        hp[0].opacity = self.mesh_opacity

    def set_active_body(self, src: str) -> None:
        """Select the body source and re-pose it at the current frame."""
        self.active = src
        if self._frame >= 0:
            with self.server.atomic():
                self._apply_bodies(self._frame)

    def apply_frame(self, f: int) -> None:
        """Show frame ``f``: flip human points, move the camera cursor onto that
        frame's centre, and re-pose the active body (all layer-gated)."""
        f = int(np.clip(f, 0, max(self.n_frames - 1, 0)))
        with self.server.atomic():
            self._apply_human(f)
            self._apply_bodies(f)
            self.camera_cursor.position = tuple(float(x) for x in self.camera_centers[f])
        self._frame = f

    def _apply_human(self, f: int) -> None:
        target = f if self.layers["human"] else -1
        if target == self._human_frame:
            return
        if self._human_frame >= 0:
            for h in self.human_nodes[self._human_frame]:
                h.visible = False
        if target >= 0:
            for h in self.human_nodes[target]:
                h.visible = True
        self._human_frame = target

    def _apply_bodies(self, f: int) -> None:
        """Draw the active source as two independent layers: its mesh (``body`` layer)
        and its skeleton (``skeleton`` layer). Any combination works; the mesh goes
        translucent only when both show, so the skeleton reads through it."""
        src = self.active
        mesh_on = self.layers["body"]
        skel_on = self.layers.get("skeleton", False)
        for name, handles in self.skinned.items():
            on = mesh_on and src == name
            for hp in handles:
                if hp is None:
                    continue
                handle, person = hp
                vis = on and f < len(person["valid"]) and bool(person["valid"][f])
                handle.visible = vis
                if vis:
                    bw, bp = person["bone_wxyz"][f], person["bone_pos"][f]
                    for j, bone in enumerate(handle.bones):
                        bone.wxyz, bone.position = bw[j], bp[j]
        self._apply_skeleton(f)
        for source in self.forces:
            self._apply_dynamics(source, f, src == source, skel_on and src == source)
        self._apply_translucency()

    # -- body-skeleton overlay --
    def _hide_skeleton_nodes(self, source: str) -> None:
        for nd in self.skel_nodes.get(source, {}).values():
            for sp in nd["spheres"].values():
                sp.visible = False
            for h, _j, _par in nd["bones"]:
                h.visible = False

    def _apply_skeleton(self, f: int) -> None:
        """Pose + contact-colour the active body source's 22-joint skeleton — gated on
        the ``skeleton`` layer + a source with skeleton nodes ("kinematics" coloured by
        contacts_1, "kinematics_2" by contacts_2). (The mesh translucency it needs is
        applied centrally in :meth:`_apply_translucency`.)

        The per-joint pose/visibility loop is skipped entirely while the skeleton is
        hidden and already torn down, so idle playback stays cheap.
        """
        src = self.active if (self.layers.get("skeleton", False)
                              and self.active in self.skel_nodes) else None
        # Confidence shading (on by default): grey→saturated red/green by label confidence.
        use_conf = self.layers.get("confidence", True)
        use_depth = self.layers.get("depth_trust", False)
        if src is None and self._skel_src is None:
            return
        if self._skel_src is not None and self._skel_src != src:
            self._hide_skeleton_nodes(self._skel_src)     # switching source / turning off
        if src is not None:
            _f, body_oids, people = self.data.skinned[src]
            nodes = self.skel_nodes[src]
            contact = self._contact.get(src, {})
            conf_d = self._conf.get(src, {})
            gap_d = self._depth_gap.get(src, {})
            trust_d = self._depth_trust.get(src, {})
            for p, person in enumerate(people):
                nd = None if person is None else nodes.get(int(body_oids[p]))
                if nd is None:
                    continue
                live = f < len(person["valid"]) and bool(person["valid"][f])
                bp = person["bone_pos"][f]                    # (J, 3) world joints
                con = contact.get(int(body_oids[p]))          # (N, J) bool or None
                cf = conf_d.get(int(body_oids[p]))            # (N, J) confidence or None
                gap = gap_d.get(int(body_oids[p]))            # (N, J) gap cm or None
                trust = trust_d.get(int(body_oids[p]))        # (N, J) trust or None
                for j, sp in nd["spheres"].items():
                    if live:
                        sp.position = (float(bp[j, 0]), float(bp[j, 1]), float(bp[j, 2]))
                        hot = con is not None and f < len(con) and bool(con[f, j])
                        if use_depth:
                            dg = gap[f, j] if gap is not None and f < len(gap) else np.nan
                            dt = trust[f, j] if trust is not None and f < len(trust) else np.nan
                            sp.color = _depth_color(dg, dt)
                        elif use_conf and cf is not None and f < len(cf):
                            sp.color = _conf_color(hot, cf[f, j])
                        else:
                            sp.color = _CONTACT_RGB if hot else _FREE_RGB
                        sp.visible = True
                    else:
                        sp.visible = False
                for h, j, par in nd["bones"]:
                    a, b = bp[par], bp[j]
                    d = b - a
                    length = float(np.linalg.norm(d))
                    if live and length > 1e-6:
                        h.position = (float((a[0] + b[0]) * 0.5), float((a[1] + b[1]) * 0.5),
                                      float((a[2] + b[2]) * 0.5))
                        h.wxyz = _wxyz_shortest_arc_z(d)
                        h.scale = (_SKEL_RB, _SKEL_RB, length)
                        h.visible = True
                    else:
                        h.visible = False
        self._skel_src = src

    # -- force / torque layers (dynamics, kindyn_1) --
    def _apply_translucency(self) -> None:
        """Make the active source's mesh translucent when its skeleton also shows over
        it, so the skeleton reads through; restore any previously translucent mesh.

        One rule for both skeletons — the kinematic 22-joint overlay and a force
        source's torque skeleton. The force-only ``dynamics`` source has no mesh, so it
        never triggers this. Pushed only on a state change, so idle playback stays cheap.
        """
        src = self.active
        want = (src if (src in self.skinned and self.layers["body"]
                        and self.layers.get("skeleton", False)
                        and src in self._skeleton_sources) else None)
        if want == self._translucent:
            return
        for target, opacity in ((self._translucent, None), (want, self.mesh_opacity)):
            if target is not None:
                for hp in self.skinned.get(target, []):
                    if hp is not None:
                        hp[0].opacity = opacity
        self._translucent = want

    def _apply_dynamics(self, source: str, f: int, active: bool, skel_on: bool) -> None:
        """Draw one force source's overlays. The torque skeleton (spheres + bones)
        follows the ``skeleton`` layer; the contact-force and root-wrench arrows follow
        their own ``dyn_*`` toggles — everything only while ``source`` is the active one.
        """
        dyn = self.forces[source]
        nodes = self.force_nodes[source]
        finger = self._dyn_finger[source]
        ly = self.layers
        scale_torque = ly.get("dyn_torque", False)    # sphere size ∝ joint torque
        show_fingers = ly.get("dyn_fingers", False)   # show finger / palm joints
        for p in range(dyn["P"]):
            nd = nodes[p]
            valid = active and f < dyn["valid"].shape[1] and bool(dyn["valid"][p, f])
            skel_live = valid and skel_on                # torque skeleton follows the skeleton layer
            jw, tm = dyn["jw"][p, f], dyn["tm"][p, f]
            fw = dyn["fw"][p, f]                          # (J, 3) solved contact force
            fmag = np.linalg.norm(fw, axis=-1)           # (J,)
            jc = dyn.get("jc")                           # (P, N, J) contact labels or None
            con = jc[p, f] if jc is not None and f < jc.shape[1] else None
            for j, sp in enumerate(nd["spheres"]):
                if skel_live and (show_fingers or not finger[j]):
                    sp.position = (float(jw[j, 0]), float(jw[j, 1]), float(jw[j, 2]))
                    # scale by joint torque, but not the root (j==0): its wrench is the arrows
                    s = (1.0 + min(float(tm[j]) / dyn["ref"], 1.0) * (_DYN_MAXS - 1.0)
                         if scale_torque and j != 0 else 1.0)
                    sp.scale = (s, s, s)
                    # red only where the joint is in contact: from the contact labels when
                    # present (dynamics), else where a force was solved (kindyn_1 has none).
                    hot = bool(con[j]) if con is not None else bool(fmag[j] > 1e-3)
                    sp.color = _CONTACT_RGB if hot else _FREE_RGB
                    sp.visible = True
                else:
                    sp.visible = False
            for h, j, par in nd["bones"]:
                a, b = jw[par], jw[j]
                d = b - a
                L = float(np.linalg.norm(d))
                if skel_live and L > 1e-6 and (show_fingers or not (finger[j] or finger[par])):
                    h.position = (float((a[0] + b[0]) * 0.5), float((a[1] + b[1]) * 0.5),
                                  float((a[2] + b[2]) * 0.5))
                    h.wxyz = _wxyz_shortest_arc_z(d)
                    h.scale = (_DYN_RB, _DYN_RB, L)
                    h.visible = True
                else:
                    h.visible = False
            # Contact-force arrows at the contact joints — their own toggle, independent
            # of the skeleton layer. Remove the old node before re-adding (reusing the
            # handle would leak a node every frame).
            if nd["forces"] is not None:
                nd["forces"].remove()
                nd["forces"] = None
            if valid and ly.get("dyn_forces", True):
                idx = np.flatnonzero(fmag > 1e-3)
                if idx.size:
                    seg = np.stack([jw[idx], jw[idx] + fw[idx] * _DYN_FSCALE], axis=1)
                    nd["forces"] = self.server.scene.add_arrows(
                        f"/clip/{source}/p{nd['oid']:02d}/forces", points=seg.astype(np.float32),
                        colors=_DYN_FORCE_RGB, shaft_radius=0.012,
                        head_radius=0.028, head_length=0.05)
            # Root residual wrench: force (green) and torque (magenta) at the pelvis,
            # each on its own toggle. A physically-consistent solve drives these toward
            # zero, so short arrows are good — they show what the contact forces failed
            # to explain.
            bw = dyn.get("bw")
            root = jw[0]
            self._root_arrow(source, nd, "root_force",
                             valid and bw is not None and ly.get("dyn_root_force", True),
                             root, None if bw is None else bw[p, f, :3] * _DYN_FSCALE, (60, 220, 90))
            self._root_arrow(source, nd, "root_torque",
                             valid and bw is not None and ly.get("dyn_root_torque", True),
                             root, None if bw is None else bw[p, f, 3:] * _DYN_TSCALE, (220, 60, 220))
            # Measured wall-sensor forces (climb_wall GT), same scale as the solved
            # arrows so lengths compare directly: orange per-channel reactions at the
            # four limb joints. A channel is its support cell's reaction —
            # limb-dominated but mixed across limbs; drawn once, on the first person.
            sens = self.sensors
            sensor_on = (sens is not None and p == 0 and valid
                         and ly.get("dyn_sensor", False))
            if nd["sensor"] is not None:
                nd["sensor"].remove()
                nd["sensor"] = None
            names = dyn.get("joint_names")
            if sensor_on and names is not None and f < len(sens["rw"]):
                joint_of = {"left_hand": "left_wrist", "right_hand": "right_wrist",
                            "left_foot": "left_foot", "right_foot": "right_foot"}
                pts, vecs = [], []
                for c, limb in enumerate(sens["limbs"]):
                    v = sens["rw"][f, c]
                    jn = joint_of.get(limb)
                    if jn in names and float(np.linalg.norm(v)) > _SENSOR_FLOOR_N:
                        pts.append(jw[names.index(jn)])
                        vecs.append(v * _DYN_FSCALE)
                if pts:
                    seg = np.stack([np.asarray(pts), np.asarray(pts) + np.asarray(vecs)],
                                   axis=1)
                    nd["sensor"] = self.server.scene.add_arrows(
                        f"/clip/{source}/p{nd['oid']:02d}/sensor",
                        points=seg.astype(np.float32), colors=_SENSOR_RGB,
                        shaft_radius=0.012, head_radius=0.028, head_length=0.05)

    def _root_arrow(self, source, nd, slot, active, root, vec, color) -> None:
        """Redraw one root arrow (force or torque) — remove-then-add each frame."""
        if nd[slot] is not None:
            nd[slot].remove()
            nd[slot] = None
        if active and vec is not None and float(np.linalg.norm(vec)) > 1e-4:
            seg = np.stack([[root, root + vec]], axis=0).astype(np.float32)   # (1, 2, 3)
            nd[slot] = self.server.scene.add_arrows(
                f"/clip/{source}/p{nd['oid']:02d}/{slot}", points=seg, colors=color,
                shaft_radius=0.014, head_radius=0.032, head_length=0.06)

    def dispose(self) -> None:
        """Remove every node this clip added — one root removal cascades to all."""
        for node_list in self.force_nodes.values():
            for nd in node_list:
                nd["forces"] = None  # lives under /clip → cascade-removed with the root
                nd["root_force"] = None
                nd["root_torque"] = None
                nd["sensor"] = None
        self.root.remove()
