"""``ClipScene`` — all viser nodes for one clip, grouped into toggleable layers.

Builds every scene node for a single clip from its :class:`~tools.viewer.loading.ClipData`
— the scene cloud, the per-point normal arrows (debug), the camera frustums +
centre path, the gravity arrow, the per-frame human
points, the skinned SMPL-X bodies, their joint skeletons and contact frames, and
the force/torque skeletons (kindyn_1 / kindyn_manual) — and
moves them together with the frame slider. One ``/clip`` root frame owns them all, so :meth:`ClipScene.dispose`
tears the whole clip down in a single removal.
"""
from __future__ import annotations

import numpy as np

from .helpers import (
    _BODY_COLORS,
    _camera_world_pose,
    _time_color,
    _unlit_glb,
    _wxyz_shortest_arc_z,
)
from .loading import _FINGER_TOKENS, ClipData, ensure_scene_mesh

# dynamics sphere-floor / bone radius / max torque grow factor / metres-per-newton for
# the contact-force arrows
_DYN_R0, _DYN_RB, _DYN_MAXS, _DYN_FSCALE = 0.022, 0.012, 2.0, 1.5e-3
# joint-sphere colour while the ``joint torques`` toggle is on (torque-sized spheres);
# off, the spheres stay the neutral skeleton grey like the kinematic overlay.
_DYN_TORQUE_RGB = (245, 200, 40)
# root residual-wrench arrows: metres per newton-metre for the torque part (the force
# part reuses _DYN_FSCALE so it is directly comparable to the contact-force arrows)
_DYN_TSCALE = 4.8e-3
# contact-force arrow colour — a saturated red that reads against the grey scene
# cloud and the grey skeleton (the same red as the contact frames the forces act on)
_DYN_FORCE_RGB = (235, 45, 45)
# measured wall-sensor arrows (climb_wall GT): orange, one per channel; below the
# floor a channel is not drawn (the sensors never read exactly zero, and the fixed
# arrow head would clutter the wall)
_SENSOR_RGB, _SENSOR_FLOOR_N = (255, 150, 30), 10.0
# body-skeleton overlay: fixed joint-sphere radius / bone radius (m) and the
# default kinematic mesh opacity while the skeleton shows over it (slider-tunable).
_SKEL_R, _SKEL_RB, _SKEL_ALPHA = 0.028, 0.012, 0.7
# Finger joints (``_FINGER_TOKENS``, imported from ``loading``) draw at 30 % of that
# size, so a hand reads as a hand instead of a clump of body-sized spheres.
_SKEL_FINGER_SCALE = 0.30
# Joint-sphere colour: joints carry no contact state any more (contact lives on the
# contact frames), so every joint is the same neutral slate grey.
_SKEL_RGB = (120, 128, 140)
# contact/free colours — red touching the scene / green free (matches estimate_contacts).
_CONTACT_RGB, _FREE_RGB = (235, 45, 45), (60, 200, 90)
# contact frames: sphere radius (m, readable next to the 0.028 joints), the
# frame→parent-joint connector radius as a fraction of it, the pale connector colour,
# and the length of the outward-normal arrow drawn at each frame.
_FRAME_R, _FRAME_CONN_SCALE, _FRAME_CONN_RGB, _FRAME_NORMAL_LEN = 0.012, 0.3, (150, 190, 160), 0.06
# cap on drawn camera frustums, so a long clip isn't a dense wire mess
_CAM_MAX_FRUSTUMS = 24
# a calibrated rig's fixed cameras (``cameras.npz``): one cyan frustum per view,
# named — they are known positions, not a reconstructed trajectory
_CALIB_CAM_RGB = (0, 200, 255)
# draw the ground grid this many times the fitted floor's data extent, so it reads
# as a floor spanning the scene rather than a small patch under the climber
_GROUND_GRID_MULT = 10.0


def _quat_mat(q: np.ndarray) -> np.ndarray:
    """3×3 rotation from a scalar-last quaternion (normalized defensively)."""
    x, y, z, w = np.asarray(q, np.float64) / max(float(np.linalg.norm(q)), 1e-9)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float32)


def _unit_cylinder(color: tuple[int, int, int] = (235, 235, 235)):
    """A unit +z cylinder (radius 1, height 1) in ``color``, scaled per-frame into
    skeleton / dynamics bones and contact-frame connectors."""
    import trimesh

    cyl = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=10)
    cyl.visual.vertex_colors = np.array([*color, 255], np.uint8)
    return cyl


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
        self.camera_nodes: list = []       # everything the "cameras" layer toggles
        self.camera_frustums: list = []    # just the frustums, which the size slider scales
        self.camera_centers = centers
        self.camera_cursor = None          # only a moving camera gets a position marker
        if data.views is not None:
            # A calibrated multi-camera rig (the climb_wall_3 ground-truth clips): the
            # cameras never move and their poses are measured, so draw one labelled
            # frustum per view instead of a trajectory of reconstructed ones.
            for k, name in enumerate(data.views["names"]):
                pos, wxyz = _camera_world_pose(data.views["extr"][k])
                self.camera_frustums.append(ss.add_camera_frustum(
                    f"/clip/cameras/{name}", fov=float(data.views["fov_y"][k]),
                    aspect=float(data.views["aspect"][k]), scale=camera_scale,
                    color=_CALIB_CAM_RGB, line_width=1.5, wxyz=wxyz, position=pos))
                self.camera_nodes.append(self.camera_frustums[-1])
                self.camera_nodes.append(ss.add_label(     # rides its frustum (child node)
                    f"/clip/cameras/{name}/label", name))
        else:
            # Thin, time-coloured frustums along the path, capped in number: hundreds of
            # identical blue frustums read as a wire mess, so cap the count (the spline
            # still shows the full path) and ramp their colour over the trajectory.
            n_extr = len(data.extr)
            step = max(camera_stride, int(np.ceil(n_extr / _CAM_MAX_FRUSTUMS)))
            for k in range(0, n_extr, step):
                pos, wxyz = _camera_world_pose(data.extr[k])
                self.camera_frustums.append(ss.add_camera_frustum(
                    f"/clip/cameras/cam_{k:04d}", fov=float(data.fov_y[k]), aspect=data.aspect,
                    scale=camera_scale, color=_time_color(k / max(n_extr - 1, 1)),
                    line_width=1.5, wxyz=wxyz, position=pos))
            self.camera_nodes = list(self.camera_frustums)
            self.camera_nodes.append(ss.add_spline_catmull_rom(
                "/clip/cameras/path", centers, color=(255, 140, 0), line_width=2.0))
            # A marker that rides the path to show the camera's position at the current
            # frame (moved in ``apply_frame``); toggles with the cameras layer.
            self.camera_cursor = ss.add_icosphere(
                "/clip/cameras/cursor", radius=max(camera_scale * 0.5, 0.02),
                color=(235, 40, 40), position=tuple(centers[0].tolist()),
                visible=self.layers["cameras"])
        self.n_cameras = len(self.camera_frustums)

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

        # -- ground plane: drawn from the fit the gravity stage saved (if accepted) --
        self.ground_nodes: list = []
        self.ground_angle: float | None = None
        self._build_ground()

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
        # source that carries a skeleton dict ("kinematics" from kinematic_1,
        # "kinematics_2" from kinematic_2). Joints come from that body's per-frame world
        # joints; all 52 are drawn (fingers small, because contact frames hang off the
        # finger joints), and the mesh goes translucent while an overlay covers it.
        self.skels = data.skeletons
        self.skel_nodes: dict[str, dict[int, dict]] = {}     # source -> oid -> nodes
        self._skel_src: str | None = None      # source whose skeleton nodes are visible
        self._translucent: str | None = None   # active source whose mesh is currently translucent
        for source, skel in self.skels.items():
            body = data.skinned.get(source)
            # A force source's skeleton is its torque-scaled dyn skeleton — building the
            # 22-joint overlay too would draw two skeletons on top of each other. Its
            # skel dict still feeds the contact-frames layer below.
            if body is None or source in data.forces:
                continue
            _f, body_oids, people = body
            names, parents = skel["joint_names"], skel["parents"]
            small = [any(t in nm for t in _FINGER_TOKENS) for nm in names]
            cyl = _unit_cylinder()
            nodes: dict[int, dict] = {}
            for p, person in enumerate(people):
                if person is None:
                    continue
                oid = int(body_oids[p])
                spheres = {j: ss.add_icosphere(
                    f"/clip/skeleton/{source}/p{oid:02d}/joint_{j:02d}",
                    radius=_SKEL_R * (_SKEL_FINGER_SCALE if small[j] else 1.0),
                    color=_SKEL_RGB, position=(0.0, 0.0, 0.0), visible=False)
                    for j in range(len(names))}
                bones = []
                for j in range(len(names)):
                    par = int(parents[j])
                    if par < 0:
                        continue
                    h = ss.add_mesh_trimesh(f"/clip/skeleton/{source}/p{oid:02d}/bone_{j:02d}",
                                            cyl, visible=False)
                    bones.append((h, j, par, _SKEL_RB * (_SKEL_FINGER_SCALE if small[j] else 1.0)))
                nodes[oid] = {"spheres": spheres, "bones": bones}
            if nodes:
                self.skel_nodes[source] = nodes

        # -- contact frames (spheres + parent links + normal arrows), hidden — the 12
        # named skin points contact is detected on, for every source whose contact
        # artifact carries per-frame labels. Positions come from the SAME body forward
        # pass that poses the mesh, so a frame sits exactly on the drawn skin. --
        self.frame_nodes: dict[str, dict[int, dict]] = {}    # source -> oid -> nodes
        self._frame_contact: dict[str, dict[int, np.ndarray]] = {}
        self._frames_src: str | None = None
        for source, skel in self.skels.items():
            body = data.skinned.get(source)
            if body is None or skel["frame_contact"] is None:
                continue
            _f, body_oids, people = body
            self._frame_contact[source] = {int(o): skel["frame_contact"][i]
                                           for i, o in enumerate(skel["contact_oids"])}
            parent = np.asarray(skel["frame_parent"], np.int32)     # (F,) joint per frame
            link = _unit_cylinder(_FRAME_CONN_RGB)
            nodes = {}
            for p, person in enumerate(people):
                if person is None:
                    continue
                oid = int(body_oids[p])
                spheres = [ss.add_icosphere(
                    f"/clip/frames/{source}/p{oid:02d}/frame_{i:02d}", radius=_FRAME_R,
                    color=_FREE_RGB, position=(0.0, 0.0, 0.0), visible=False)
                    for i in range(len(parent))]
                links = [ss.add_mesh_trimesh(f"/clip/frames/{source}/p{oid:02d}/link_{i:02d}",
                                             link, visible=False) for i in range(len(parent))]
                nodes[oid] = {"oid": oid, "spheres": spheres, "links": links,
                              "parent": parent, "arrows": None}
            if nodes:
                self.frame_nodes[source] = nodes

        # -- force/torque skeletons (spheres + bones), one per force source
        #    ("kindyn_1" / "kindyn_manual"), hidden; forces added per frame. Node paths
        #    are keyed by the source name, so each source's nodes stay independent. --
        self.forces = data.forces                          # {source: force dict}
        self.force_nodes: dict[str, list[dict]] = {}
        self._dyn_finger: dict[str, np.ndarray] = {}       # source -> (J,) finger/palm mask
        # Per-person contact-frame world positions for the "per-frame" force display —
        # reused from the SAME body forward pass that poses the source's mesh, so a
        # per-frame arrow starts exactly on the drawn skin. Only sources that store
        # ``frame_forces`` and have a skinned body ("kindyn_1" / "kindyn_manual")
        # qualify; anything else falls back to per-joint.
        self._force_frame_pos: dict[str, list[np.ndarray | None]] = {}
        for source, dyn in self.forces.items():
            body = data.skinned.get(source)
            if dyn.get("ff") is not None and body is not None:
                _fcs, _oids, people = body
                # A [contacts].frames restriction (the peter rigs run hands+feet only)
                # sizes the force array to a SUBSET of the body's contact frames —
                # expand it by name so the arrows align with the posed frame positions.
                body_names = next((p["frame_names"] for p in people if p is not None), None)
                names = dyn.get("ff_names")
                if (names is not None and body_names is not None and names != body_names
                        and set(names) <= set(body_names)):
                    full = np.zeros(dyn["ff"].shape[:2] + (len(body_names), 3), np.float32)
                    full[:, :, [body_names.index(n) for n in names]] = dyn["ff"]
                    dyn["ff"], dyn["ff_names"] = full, list(body_names)
                    # (the force dict is the viewer's per-clip cache, so record the new
                    # column order too — revisiting the clip must not expand it twice)
                counts = {person["frame_pos"].shape[1] for person in people if person is not None}
                if len(people) == dyn["P"] and counts == {dyn["ff"].shape[2]}:
                    self._force_frame_pos[source] = [
                        None if person is None else person["frame_pos"] for person in people]
        for source, dyn in self.forces.items():
            unit_cyl = _unit_cylinder()
            Jd = dyn["jw"].shape[2]
            node_list: list[dict] = []
            for p in range(dyn["P"]):
                oid = int(dyn["oids"][p])
                spheres = [ss.add_icosphere(f"/clip/{source}/p{oid:02d}/sphere_{j:02d}",
                                            radius=_DYN_R0, color=_SKEL_RGB,
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
        # skinned body ("kindyn_1") already appears via ``data.skinned``. kinematics_2 is
        # selectable but kinematic_1 stays the default.
        self.body_source_names = (
            list(data.skinned) + [s for s in self.forces if s not in data.skinned])
        self.default_body = next(
            (s for s in ("kinematics", "kinematics_2", "prefit", "original")
             if s in self.body_source_names), None)
        self.active = self.default_body

        # Which sources own a drawable skeleton, and which contextual GUI controls
        # apply to which source (read by the viewer to hide inapplicable controls):
        #   force sources     → the ``dyn_*`` overlays + the torque skeleton
        #   frame sources     → the ``contact frames`` layer (labels exist for them)
        #   translucency ones → the body-opacity slider (a mesh an overlay covers)
        self.force_sources = set(self.forces)
        self.frame_sources = set(self.frame_nodes)
        # Measured wall-sensor forces (climb_wall clips only) — drawn on top of any
        # active force source by the ``sensor forces`` toggle.
        self.sensors = data.sensors
        self._skeleton_sources = self.force_sources | set(self.skel_nodes)
        self.translucency_sources = {s for s in self.skinned
                                     if s in self._skeleton_sources | self.frame_sources}

        self._apply_static_layers()
        self.info_md = self._build_info()

    # -- info panel --
    def _build_info(self) -> str:
        d = self.data
        gtxt = "—"
        if d.gravity is not None:
            g = d.gravity["g"]
            tag = {"ground": "ground plane",
                   "fallback_down": "default↓"}.get(
                d.gravity.get("source", ""), f"GeoCalib, spread {d.gravity['spread']:.1f}°")
            gtxt = f"[{g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}] ({tag})"
        srcs = ", ".join(self.body_source_names) or "—"
        gnd = "" if self.ground_angle is None else f" · ground ∠prior {self.ground_angle:.1f}°"
        if not gnd and d.gravity is not None and d.gravity.get("ground_reject"):
            gnd = f" · ground: {d.gravity['ground_reject']}"
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
        """Draw the accepted ground-plane fit as a grid.

        The plane is fitted (and gated) by the gravity stage and saved in
        ``geocalib/gravity.npz``; the viewer only draws what the pipeline used.
        ``ground_angle`` is the fitted normal's angle to the prior down direction,
        shown in the info panel. The grid (natively in its local x-y plane, normal
        ``+z``) is rotated so ``+z`` aligns with the fitted up-normal and placed
        at the plane centroid.
        """
        plane = None if self.data.gravity is None else self.data.gravity.get("ground")
        if plane is None:
            return
        self.ground_angle = plane["angle_deg"]
        size = plane["extent"] * _GROUND_GRID_MULT
        self.ground_nodes = [self.server.scene.add_grid(
            "/clip/ground", width=size, height=size,
            cell_size=max(size / 20.0, 0.05), section_size=max(size / 4.0, 0.25),
            cell_color=(170, 170, 170), section_color=(110, 110, 110),
            plane="xy", wxyz=_wxyz_shortest_arc_z(plane["normal"]),
            position=tuple(float(x) for x in plane["centroid"]),
            visible=self.layers["ground"])]

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
        if self.camera_cursor is not None:
            self.camera_cursor.visible = self.layers["cameras"]
        for h in self.gravity_nodes:
            h.visible = self.layers["gravity"]
        for h in self.ground_nodes:
            h.visible = self.layers["ground"]

    def set_static_layer(self, name: str, on: bool) -> None:
        """Toggle one static layer (``scene`` / ``normals`` / ``cameras`` / ``gravity`` / ``ground``)."""
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
            if self.camera_cursor is not None:
                self.camera_cursor.visible = on
        elif name == "gravity":
            for h in self.gravity_nodes:
                h.visible = on
        elif name == "ground":
            for h in self.ground_nodes:
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
            for h in self.camera_frustums:
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
            if self.camera_cursor is not None:
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
        self._apply_frames(f)
        for source in self.forces:
            self._apply_dynamics(source, f, src == source, skel_on and src == source)
        self._apply_translucency()

    # -- body-skeleton overlay --
    def _hide_skeleton_nodes(self, source: str) -> None:
        for nd in self.skel_nodes.get(source, {}).values():
            self._hide_person_skeleton(nd)

    @staticmethod
    def _hide_person_skeleton(nd: dict) -> None:
        for sp in nd["spheres"].values():
            sp.visible = False
        for h, *_rest in nd["bones"]:
            h.visible = False

    def _apply_skeleton(self, f: int) -> None:
        """Pose the active body source's 52-joint skeleton — gated on the ``skeleton``
        layer + a source with skeleton nodes ("kinematics" / "kinematics_2").

        Spheres are one neutral colour: contact is a property of the contact frames
        (see :meth:`_apply_frames`), not of joints. Finger joints and their bones draw
        small. (The mesh translucency the overlay needs is applied centrally in
        :meth:`_apply_translucency`.)

        The per-joint pose/visibility loop is skipped entirely while the skeleton is
        hidden and already torn down, so idle playback stays cheap.
        """
        src = self.active if (self.layers.get("skeleton", False)
                              and self.active in self.skel_nodes) else None
        if src is None and self._skel_src is None:
            return
        if self._skel_src is not None and self._skel_src != src:
            self._hide_skeleton_nodes(self._skel_src)     # switching source / turning off
        if src is not None:
            _f, body_oids, people = self.data.skinned[src]
            nodes = self.skel_nodes[src]
            for p, person in enumerate(people):
                nd = None if person is None else nodes.get(int(body_oids[p]))
                if nd is None:
                    continue
                # A body can cover fewer frames than the clip — a multi-view fit is
                # shortened by the slowest camera's lag — so past its end, or on an
                # invalid frame, this person simply disappears.
                if not (f < len(person["valid"]) and bool(person["valid"][f])):
                    self._hide_person_skeleton(nd)
                    continue
                bp = person["bone_pos"][f]                    # (J, 3) world joints
                for j, sp in nd["spheres"].items():
                    sp.position = (float(bp[j, 0]), float(bp[j, 1]), float(bp[j, 2]))
                    sp.visible = True
                for h, j, par, radius in nd["bones"]:
                    a, b = bp[par], bp[j]
                    d = b - a
                    length = float(np.linalg.norm(d))
                    if length > 1e-6:
                        h.position = (float((a[0] + b[0]) * 0.5), float((a[1] + b[1]) * 0.5),
                                      float((a[2] + b[2]) * 0.5))
                        h.wxyz = _wxyz_shortest_arc_z(d)
                        h.scale = (radius, radius, length)
                        h.visible = True
                    else:
                        h.visible = False
        self._skel_src = src

    # -- contact-frame overlay --
    def _hide_frame_nodes(self, source: str) -> None:
        for nd in self.frame_nodes.get(source, {}).values():
            self._hide_person_frames(nd)

    @staticmethod
    def _hide_person_frames(nd: dict) -> None:
        for sp in nd["spheres"]:
            sp.visible = False
        for h in nd["links"]:
            h.visible = False
        if nd["arrows"] is not None:
            nd["arrows"].remove()
            nd["arrows"] = None

    def _apply_frames(self, f: int) -> None:
        """Draw the active body source's 12 contact frames at frame ``f``.

        Per frame: a sphere on the skin point (red where the contact artifact says it
        touches the scene, green where it is free), a thin link back to the joint the
        frame rides on, and a short arrow along its outward normal — the direction the
        detector compares against the scene surface. Gated on the ``contact frames``
        layer; skipped entirely while hidden and already torn down.
        """
        src = self.active if (self.layers.get("contact_frames", True)
                              and self.active in self.frame_nodes) else None
        if src is None and self._frames_src is None:
            return
        if self._frames_src is not None and self._frames_src != src:
            self._hide_frame_nodes(self._frames_src)      # switching source / turning off
        if src is not None:
            _f, body_oids, people = self.data.skinned[src]
            nodes, contact = self.frame_nodes[src], self._frame_contact[src]
            for p, person in enumerate(people):
                nd = None if person is None else nodes.get(int(body_oids[p]))
                if nd is None:
                    continue
                if f < len(person["valid"]) and bool(person["valid"][f]):
                    self._draw_frames(src, nd, person, contact[int(body_oids[p])], f)
                else:
                    self._hide_person_frames(nd)
        self._frames_src = src

    def _draw_frames(self, source: str, nd: dict, person: dict,
                     contact: np.ndarray, f: int) -> None:
        """Place one person's contact-frame spheres, parent links and normal arrows."""
        fp, fn = person["frame_pos"][f], person["frame_normal"][f]     # (F, 3) each
        bp = person["bone_pos"][f]                                     # (J, 3) world joints
        hot = contact[f]                                               # (F,) in scene contact
        for i, sp in enumerate(nd["spheres"]):
            sp.position = (float(fp[i, 0]), float(fp[i, 1]), float(fp[i, 2]))
            sp.color = _CONTACT_RGB if hot[i] else _FREE_RGB
            sp.visible = True
        radius = _FRAME_R * _FRAME_CONN_SCALE
        for i, h in enumerate(nd["links"]):
            a, b = bp[int(nd["parent"][i])], fp[i]
            d = b - a
            length = float(np.linalg.norm(d))
            if length > 1e-6:
                h.position = (float((a[0] + b[0]) * 0.5), float((a[1] + b[1]) * 0.5),
                              float((a[2] + b[2]) * 0.5))
                h.wxyz = _wxyz_shortest_arc_z(d)
                h.scale = (radius, radius, length)
                h.visible = True
            else:
                h.visible = False
        # Outward-normal arrows: one node per person, remove-then-add each frame
        # (reusing the handle would leak a node), coloured like their spheres.
        # Their own ``frame normal arrows`` toggle, independent of the frames layer.
        if nd["arrows"] is not None:
            nd["arrows"].remove()
            nd["arrows"] = None
        if not self.layers.get("frame_normals", False):
            return
        seg = np.stack([fp, fp + fn * _FRAME_NORMAL_LEN], axis=1).astype(np.float32)
        colors = np.where(hot[:, None], np.array(_CONTACT_RGB, np.uint8),
                          np.array(_FREE_RGB, np.uint8)).astype(np.uint8)
        nd["arrows"] = self.server.scene.add_arrows(
            f"/clip/frames/{source}/p{nd['oid']:02d}/normals", points=seg, colors=colors,
            shaft_radius=0.05 * _FRAME_NORMAL_LEN, head_radius=0.12 * _FRAME_NORMAL_LEN,
            head_length=0.3 * _FRAME_NORMAL_LEN)

    # -- force / torque layers (dynamics, kindyn_1) --
    def _apply_translucency(self) -> None:
        """Make the active source's mesh translucent when an overlay also shows over it,
        so the overlay reads through; restore any previously translucent mesh.

        One rule for all three overlays — the kinematic 52-joint skeleton, the contact
        frames, and a force source's torque skeleton. Pushed only on a state change, so
        idle playback stays cheap.
        """
        src = self.active
        covered = ((self.layers.get("skeleton", False) and src in self._skeleton_sources)
                   or (self.layers.get("contact_frames", True) and src in self.frame_sources))
        want = src if (src in self.skinned and self.layers["body"] and covered) else None
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
            # As in ``_apply_skeleton``: a body can be shorter than the clip, so ``valid``
            # is False past its end and ``fd`` keeps every array read in range (nothing
            # is drawn from it — the overlays all hang off ``valid``).
            n_body = dyn["valid"].shape[1]
            valid = active and f < n_body and bool(dyn["valid"][p, f])
            fd = min(f, n_body - 1)
            skel_live = valid and skel_on                # torque skeleton follows the skeleton layer
            jw, tm = dyn["jw"][p, fd], dyn["tm"][p, fd]
            fw = dyn["fw"][p, fd]                         # (J, 3) solved contact force (hands at wrists)
            fmag = np.linalg.norm(fw, axis=-1)           # (J,)
            for j, sp in enumerate(nd["spheres"]):
                if skel_live and (show_fingers or not finger[j]):
                    sp.position = (float(jw[j, 0]), float(jw[j, 1]), float(jw[j, 2]))
                    # scale + yellow by joint torque, but not the root (j==0): its
                    # wrench is the arrows. Off, the spheres match the kinematic
                    # skeleton — neutral grey, uniform size (contact lives on the
                    # contact frames, not joints).
                    torqued = scale_torque and j != 0
                    s = (1.0 + min(float(tm[j]) / dyn["ref"], 1.0) * (_DYN_MAXS - 1.0)
                         if torqued else 1.0)
                    sp.scale = (s, s, s)
                    sp.color = _DYN_TORQUE_RGB if torqued else _SKEL_RGB
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
            # Contact-force arrows — their own toggle, independent of the skeleton layer.
            # Two display regimes (the ``force display`` dropdown): "per-joint" draws one
            # accumulated arrow at each loaded joint; "per-frame" draws every contact
            # frame's own solved force where it acts, for sources that store
            # ``frame_forces``. Remove the old node before re-adding (reusing the handle
            # would leak a node every frame).
            if nd["forces"] is not None:
                nd["forces"].remove()
                nd["forces"] = None
            if valid and ly.get("dyn_forces", True):
                frame_pos = self._force_frame_pos.get(source)
                per_frame = (ly.get("dyn_force_mode", "per-joint") == "per-frame"
                             and frame_pos is not None and frame_pos[p] is not None)
                if per_frame:
                    vec = dyn["ff"][p, fd]                                # (F, 3)
                    anchor = frame_pos[p][fd]                             # (F, 3)
                    keep = (np.linalg.norm(vec, axis=-1) > 1e-3) & np.isfinite(anchor).all(-1)
                    idx = np.flatnonzero(keep)
                else:
                    vec, anchor = fw, jw
                    idx = np.flatnonzero(fmag > 1e-3)
                if idx.size:
                    seg = np.stack([anchor[idx], anchor[idx] + vec[idx] * _DYN_FSCALE], axis=1)
                    nd["forces"] = self.server.scene.add_arrows(
                        f"/clip/{source}/p{nd['oid']:02d}/forces", points=seg.astype(np.float32),
                        colors=_DYN_FORCE_RGB, shaft_radius=0.012,
                        head_radius=0.028, head_length=0.05)
            # Root residual wrench: force (green) and torque (magenta) at the pelvis,
            # each on its own toggle. A physically-consistent solve drives these toward
            # zero, so short arrows are good — they show what the contact forces failed
            # to explain. RNEA expresses the wrench in the free-flyer root's LOCAL
            # frame; rotate it to world before drawing, or the arrows point wrong
            # whenever the pelvis is rotated.
            bw = dyn.get("bw")
            rq = dyn.get("rootq")
            root = jw[0]
            bw_force = bw_torque = None
            if bw is not None:
                R = _quat_mat(rq[p, fd]) if rq is not None else np.eye(3, dtype=np.float32)
                bw_force, bw_torque = R @ bw[p, fd, :3], R @ bw[p, fd, 3:]
            self._root_arrow(source, nd, "root_force",
                             valid and bw_force is not None and ly.get("dyn_root_force", True),
                             root, None if bw_force is None else bw_force * _DYN_FSCALE,
                             (60, 220, 90))
            self._root_arrow(source, nd, "root_torque",
                             valid and bw_torque is not None and ly.get("dyn_root_torque", True),
                             root, None if bw_torque is None else bw_torque * _DYN_TSCALE,
                             (220, 60, 220))
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
        for nodes in self.frame_nodes.values():
            for nd in nodes.values():
                nd["arrows"] = None      # lives under /clip → cascade-removed with the root
        for node_list in self.force_nodes.values():
            for nd in node_list:
                nd["forces"] = None  # lives under /clip → cascade-removed with the root
                nd["root_force"] = None
                nd["root_torque"] = None
                nd["sensor"] = None
        self.root.remove()
