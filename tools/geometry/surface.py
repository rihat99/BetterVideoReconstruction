"""Reconstruct a continuous colored surface from the scene point cloud.

The fused ``geometry/scene.npz`` cloud renders as scattered dots; for a nicer
view it can be turned into a surface. Two methods (both via Open3D), selected by
``method`` in :func:`point_cloud_to_mesh`:

- ``"bpa"`` — **ball pivoting** (Bernardini et al. 2001): roll a ball over the
  oriented points and keep the triangles it rests on. Faithful (vertices are the
  original points) but conservative on a noisy monocular cloud, so it leaves many
  holes even after a hole-filling pass.
- ``"poisson"`` — **screened Poisson** (Kazhdan & Hoppe 2013): fit a smooth
  implicit surface. Far more continuous (few holes), but by construction it is
  watertight and balloons "bubbles" of invented geometry into empty regions —
  trimmed away here by point-support density and, crucially, by distance to the
  real points, so only the surface near actual data survives.

Both share a final edge-length cap that removes stray oversized triangles.
"""

from __future__ import annotations

import numpy as np


def point_cloud_to_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    method: str = "bpa",
    normals: np.ndarray | None = None,
    radius_mult: tuple[float, ...] = (1.0, 2.0, 4.0),
    hole_size_mult: float = 30.0,
    poisson_depth: int = 9,
    density_quantile: float = 0.05,
    max_dist_mult: float = 3.0,
    max_edge_mult: float = 10.0,
    knn: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Reconstruct a colored surface mesh from a point cloud.

    With ``method="bpa"`` rolls balls of ``radius_mult`` × the median
    nearest-neighbour spacing over the oriented points, then bridges interior
    holes shorter than ``hole_size_mult`` × the spacing with ``fill_holes``.
    Faithful but hole-prone on noisy clouds.

    With ``method="poisson"`` fits a screened-Poisson implicit surface at
    ``poisson_depth`` — continuous with few holes — then removes its "bubbles"
    (watertight extrapolation into empty space) two ways: drop the lowest
    ``density_quantile`` fraction of vertices by point support, and drop every
    vertex farther than ``max_dist_mult`` × the median spacing from the nearest
    input point. The distance cull is the decisive one: no surface survives more
    than that far from real data, so the result hugs the points instead of
    ballooning.

    Both paths then drop triangles whose longest edge exceeds ``max_edge_mult`` ×
    the median spacing (removes stray giant faces), and colour every mesh vertex
    by its nearest input point.

    :param points: ``(N, 3)`` world-frame points.
    :param colors: ``(N, 3)`` ``uint8`` per-point colours.
    :param method: ``"bpa"`` (faithful, holey) or ``"poisson"`` (continuous,
        de-bubbled).
    :param normals: optional ``(N, 3)`` oriented unit normals — e.g. the
        camera-oriented ones from
        :func:`tools.geometry.normals.estimate_point_normals`, which orient both
        methods well and beat Open3D's own consistent orientation on speed.
        Estimated with Open3D when omitted.
    :param radius_mult: BPA ball radii, in multiples of the median point spacing.
    :param hole_size_mult: BPA — fill interior holes whose boundary is shorter
        than this × the median spacing (``0`` disables).
    :param poisson_depth: Poisson octree depth (resolution).
    :param density_quantile: Poisson — drop this fraction of lowest-support
        vertices (``0`` disables).
    :param max_dist_mult: Poisson — drop vertices farther than this × the median
        spacing from any input point (the anti-bubble cull; ``0`` disables).
    :param max_edge_mult: drop triangles whose longest edge exceeds this × the
        median spacing, removing giant-face artifacts (``0`` disables the cap).
    :param knn: neighbours used for the fallback normal estimate.
    :returns: ``(vertices (V, 3) float32, faces (F, 3) int32,
        vertex_colors (V, 3) uint8, vertex_normals (V, 3) float32 | None)`` in the
        frame of ``points`` — the normals are camera-oriented when ``normals`` is
        given, else ``None`` (letting the renderer compute them). ``None`` overall
        if the cloud is too small or nothing reconstructs.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    pts = np.asarray(points, np.float64)
    if len(pts) < 4:
        return None
    tree = cKDTree(pts)
    median_spacing = float(np.median(tree.query(pts, k=2)[0][:, 1]))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, np.float64))
    else:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        pcd.orient_normals_consistent_tangent_plane(k=knn)

    if method == "bpa":
        radii = o3d.utility.DoubleVector([m * median_spacing for m in radius_mult])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
        mesh.remove_unreferenced_vertices()
        if len(mesh.triangles) == 0:
            return None
        if hole_size_mult > 0:
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh).fill_holes(
                hole_size=hole_size_mult * median_spacing).to_legacy()
    elif method == "poisson":
        # n_threads=4, not the -1 ("all cores") default: on a many-core host
        # (e.g. 224 cores) that default makes the solver ~50× slower on small
        # clouds from thread overhead — a few threads is both fast and polite.
        mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth, n_threads=4)
        density = np.asarray(density)
        if density_quantile > 0:
            mesh.remove_vertices_by_mask(density < np.quantile(density, density_quantile))
        if max_dist_mult > 0:                     # anti-bubble: keep surface near data
            far = tree.query(np.asarray(mesh.vertices))[0] > max_dist_mult * median_spacing
            mesh.remove_vertices_by_mask(far)
    else:
        raise ValueError(f"unknown method {method!r} (expected 'bpa' or 'poisson')")

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if max_edge_mult > 0 and len(faces):
        edges = verts[faces]
        longest = np.linalg.norm(edges[:, [0, 1, 2]] - edges[:, [1, 2, 0]], axis=2).max(1)
        faces = faces[longest <= max_edge_mult * median_spacing]
    if len(faces) == 0:
        return None

    # Keep only the vertices the surviving triangles use (drops any oversized-only
    # or corrupt vertex), then remap the faces onto the compacted vertex array.
    used = np.unique(faces)
    remap = np.full(len(verts), -1, np.int64)
    remap[used] = np.arange(len(used))
    verts_out = verts[used].astype(np.float32)
    faces_out = remap[faces].astype(np.int32)
    # Colour each mesh vertex by its nearest input point (exact for BPA, whose
    # vertices are the originals; a good approximation for Poisson's resampled ones).
    _, nearest = tree.query(verts_out)
    vcolors = np.asarray(colors, np.uint8)[nearest]

    # Per-vertex shading normals. Take the mesh's own smooth (area-weighted) normals
    # but flip any that disagree with the camera-oriented input normal at the nearest
    # point, so the whole surface faces the cameras. The renderer shades from these,
    # so a stray inward normal would read as a dark patch — BPA winding isn't globally
    # consistent, so a few flip; Poisson's already is, so none do.
    vnormals = None
    if normals is not None:
        fm = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts_out.astype(np.float64)),
            o3d.utility.Vector3iVector(faces_out.astype(np.int64)))
        fm.compute_vertex_normals()
        vn = np.asarray(fm.vertex_normals)
        ref = np.asarray(normals, np.float64)[nearest]
        vn[np.einsum("ij,ij->i", vn, ref) < 0.0] *= -1.0
        vnormals = vn.astype(np.float32)
    return verts_out, faces_out, vcolors, vnormals
