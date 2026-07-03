"""``ClipScene`` — all viser nodes for one clip, grouped into toggleable layers.

Builds every scene node for a single clip from its :class:`~tools.viewer.loading.ClipData`
— the scene cloud (points or a reconstructed surface), the camera frustums +
centre path, the gravity arrow, the per-frame human points, the skinned SMPL-X
bodies, and the force-only dynamics skeleton — and moves them together with the
frame slider. One ``/clip`` root frame owns them all, so :meth:`ClipScene.dispose`
tears the whole clip down in a single removal.
"""
from __future__ import annotations

import numpy as np

from .helpers import (
    _BODY_COLORS,
    _camera_world_pose,
    _time_color,
    _unit,
    _wxyz_shortest_arc_z,
)
from .loading import ClipData, ensure_scene_mesh

# dynamics sphere-floor / bone radius / max torque grow factor / metres-per-newton
_DYN_R0, _DYN_RB, _DYN_MAXS, _DYN_FSCALE = 0.022, 0.012, 3.0, 5e-4
# cap on drawn camera frustums, so a long clip isn't a dense wire mess
_CAM_MAX_FRUSTUMS = 24


class ClipScene:
    """Builds and owns every scene node for one clip; ``dispose`` removes them.

    ``layers`` is the viewer's shared ``{name: bool}`` visibility dict, read live
    so a clip built after the user hid a layer starts with that layer hidden.
    """

    def __init__(self, server, data: ClipData, layers: dict, *,
                 scene_point_size: float, human_point_size: float, camera_scale: float,
                 camera_stride: int, scene_surface: str = "points") -> None:
        self.server = server
        self.data = data
        self.layers = layers
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
        # Framing + lighting anchors: where the human is, and the side the cameras
        # watch from. The viewer orbits about ``focus_point``; the key light comes
        # from the camera side (see ``_build_lights``).
        self.camera_mean = centers.mean(0).astype(np.float64)
        self.focus_point = self._compute_focus()
        self._build_lights()
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
            color=(245, 245, 245), position=tuple(centers[0].tolist()),
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

        # -- dynamics skeleton (spheres + bones), hidden; forces added per frame --
        self.dyn = data.dynamics
        self.dyn_nodes: list[dict] = []
        if self.dyn is not None:
            import trimesh

            unit_cyl = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=10)
            unit_cyl.visual.vertex_colors = np.array([235, 235, 235, 255], np.uint8)
            Jd = self.dyn["jw"].shape[2]
            for p in range(self.dyn["P"]):
                oid = int(self.dyn["oids"][p])
                spheres = []
                for j in range(Jd):
                    h = ss.add_icosphere(f"/clip/dynamics/p{oid:02d}/sphere_{j:02d}",
                                         radius=_DYN_R0, color=(235, 45, 45),
                                         position=(0.0, 0.0, 0.0), visible=False)
                    spheres.append(h)
                bones = []
                for j in range(1, Jd):
                    par = int(self.dyn["parents"][j])
                    if par < 0:
                        continue
                    h = ss.add_mesh_trimesh(f"/clip/dynamics/p{oid:02d}/bone_{j:02d}",
                                            unit_cyl, visible=False)
                    bones.append((h, j, par))
                self.dyn_nodes.append({"oid": oid, "spheres": spheres, "bones": bones,
                                       "forces": None})

        # Available body sources (display order) + default (most refined present).
        self.body_source_names = (
            list(data.skinned) + (["dynamics"] if self.dyn is not None else []))
        self.default_body = next(
            (s for s in ("kinematics_2", "kinematics", "original", "dynamics")
             if s in self.body_source_names), None)
        self.active = self.default_body

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
        return (f"**{d.name}** — {d.n_frames} frames · {len(d.scene_points):,} scene pts · "
                f"{self.n_cameras} cameras · {len(d.human_points):,} human pts · "
                f"climbers {d.obj_ids}\n\nbodies: {srcs} · gravity: {gtxt}")

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

    def _build_lights(self) -> None:
        """Add one key light from the viewing (camera) side; viser's default lights
        provide the ambient fill.

        The VGGT cameras all sit in front of the scene, so ``camera_mean - focus``
        is the direction of the side we look from. A directional light aimed back
        along that axis — tilted up and to one side for shape — lights the
        front-facing surfaces we actually see. viser aims a directional light from
        its position toward the origin, so a position along ``+key_dir`` makes it
        travel ``−key_dir`` into the scene (magnitude irrelevant — parallel rays,
        no shadows). Point clouds are unlit in three.js, so this shapes the meshes:
        the bodies, the reconstructed scene surface, and the dynamics bones.
        """
        ss = self.server.scene
        front = self.camera_mean - self.focus_point
        n = float(np.linalg.norm(front))
        front = front / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
        up = np.array([0.0, -1.0, 0.0])                      # VGGT world up is -y
        side = np.cross(up, front)
        sn = float(np.linalg.norm(side))
        side = side / sn if sn > 1e-6 else np.array([1.0, 0.0, 0.0])
        key_dir = _unit(front + 0.55 * up + 0.65 * side)     # front, a bit above + to a side
        self.light_nodes = [
            ss.add_light_directional("/clip/lights/key", color=(255, 246, 232),
                                     intensity=2.3, position=tuple((key_dir * 10.0).tolist())),
        ]

    # -- layer visibility --
    def _apply_static_layers(self) -> None:
        """Scene / cameras / gravity visibility from the shared ``layers`` dict."""
        self.scene_node.visible = self.layers["scene"]
        for h in self.camera_nodes:
            h.visible = self.layers["cameras"]
        self.camera_cursor.visible = self.layers["cameras"]
        for h in self.gravity_nodes:
            h.visible = self.layers["gravity"]

    def set_static_layer(self, name: str, on: bool) -> None:
        """Toggle one static layer (``scene`` / ``cameras`` / ``gravity``)."""
        if name == "scene":
            self.scene_node.visible = on
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
            import trimesh
            from trimesh.visual.material import PBRMaterial
            v, f, vc, vn = mesh_data
            # Bake the camera-oriented vertex normals in (GLB export carries them)
            # so three.js shades a correctly-facing surface.
            mesh = trimesh.Trimesh(v, f, vertex_colors=vc, vertex_normals=vn, process=False)
            # Matte + double-sided: a climbing wall is diffuse (not metal), and a
            # non-metallic surface reads correctly under the camera-side lights —
            # metallic PBR needs an environment map, which we don't set.
            mesh.visual.material = PBRMaterial(
                doubleSided=True, metallicFactor=0.0, roughnessFactor=0.95)
            node = ss.add_mesh_trimesh("/clip/scene", mesh)
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
        show = self.layers["body"]
        src = self.active
        for name, handles in self.skinned.items():
            on = show and src == name
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
        if self.dyn is not None:
            self._apply_dynamics(f, show and src == "dynamics")

    # -- dynamics --
    def _apply_dynamics(self, f: int, on: bool) -> None:
        dyn = self.dyn
        for p in range(dyn["P"]):
            nd = self.dyn_nodes[p]
            live = on and f < dyn["valid"].shape[1] and bool(dyn["valid"][p, f])
            jw, tm = dyn["jw"][p, f], dyn["tm"][p, f]
            for j, sp in enumerate(nd["spheres"]):
                if live:
                    sp.position = (float(jw[j, 0]), float(jw[j, 1]), float(jw[j, 2]))
                    s = 1.0 + min(float(tm[j]) / dyn["ref"], 1.0) * (_DYN_MAXS - 1.0)
                    sp.scale = (s, s, s)
                    sp.visible = True
                else:
                    sp.visible = False
            for h, j, par in nd["bones"]:
                a, b = jw[par], jw[j]
                d = b - a
                L = float(np.linalg.norm(d))
                if live and L > 1e-6:
                    h.position = (float((a[0] + b[0]) * 0.5), float((a[1] + b[1]) * 0.5),
                                  float((a[2] + b[2]) * 0.5))
                    h.wxyz = _wxyz_shortest_arc_z(d)
                    h.scale = (_DYN_RB, _DYN_RB, L)
                    h.visible = True
                else:
                    h.visible = False
            # Force arrows vary per frame — remove the old node before re-adding
            # (reusing the handle would leak a node every frame).
            if nd["forces"] is not None:
                nd["forces"].remove()
                nd["forces"] = None
            if live:
                fw = dyn["fw"][p, f]
                idx = np.flatnonzero(np.linalg.norm(fw, axis=-1) > 1e-3)
                if idx.size:
                    seg = np.stack([jw[idx], jw[idx] + fw[idx] * _DYN_FSCALE], axis=1)
                    nd["forces"] = self.server.scene.add_arrows(
                        f"/clip/dynamics/p{nd['oid']:02d}/forces", points=seg.astype(np.float32),
                        colors=(45, 200, 255), shaft_radius=0.012,
                        head_radius=0.028, head_length=0.05)

    def dispose(self) -> None:
        """Remove every node this clip added — one root removal cascades to all."""
        for nd in self.dyn_nodes:
            nd["forces"] = None  # lives under /clip → cascade-removed with the root
        self.root.remove()
