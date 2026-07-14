# The fusion stage: building the point clouds

`scripts/stages/fuse_geometry.py` — runs after the SMPL-X conversion and **before**
`estimate_scale`. Needs the VGGT stage run with depth saving on.

This doc explains how the 3D world is built from VGGT's output. One thing to hold
in mind throughout: **the world fuse builds is up-to-scale, not metric.** VGGT
reconstructs shape correctly but its overall size is off by one unknown factor per
clip. Fuse does not know that factor and does not try to find it. It builds the
clouds in VGGT's native units, marks them `metric=False`, and the scale stage
(see [scale.md](scale.md)) measures the factor afterwards and multiplies the saved
files in place.

Why this order? The scale stage needs the person's depth points. Fuse already
extracts and filters exactly those points (mask split, edge filter). Running fuse
first means scale reads one clean file instead of re-doing the extraction from raw
depth maps and masks.

## What fusion produces

Three files under `geometry/`:

- `transform.npz` — the camera poses and intrinsics. `extrinsics (F,4,4)` are
  cam-from-world matrices, `intrinsics_px_orig (F,3,3)` the pixel intrinsics at the
  original video resolution, plus the normalized intrinsics, FOVs, frame indices,
  image sizes and fps. Fields `scale=1.0` and `metric=False` say the world is still
  unscaled; the scale stage flips them. Every later stage reads its cameras from here.
- `scene.npz` — one static, merged point cloud of everything that is not a person:
  the wall, the holds, the ground. Per point: position, colour, VGGT depth
  confidence (max over the merged points), how many raw points merged into it
  (`count`), and the voxel size used.
- `human.npz` — the person's depth points, kept **per frame and unmerged** (the
  person moves; merging frames would smear them). Per point: position, colour,
  confidence, `frame_id` (index into `frame_indices`) and `obj_id` (which person).

## Per frame: clean, split, backproject

1. **Clean the depth.** Two filters:
   - An *edge filter* removes "flying pixels" — points VGGT smears across depth
     discontinuities, e.g. between the climber's arm and the wall behind it. It flags
     pixels whose depth jumps by more than 4% (`edge_rtol = 0.04`) relative to a
     neighbour.
   - A *confidence gate* drops the least confident 30% of pixels
     (`conf_drop_q = 0.3`). The threshold is a per-frame quantile, not a fixed
     value, because VGGT's confidence scale drifts from frame to frame.

   The scene gets both filters. The human cloud gets **only the edge filter** — the
   person is exactly where we cannot afford to throw points away, and the scale
   stage applies its own confidence handling later.

2. **Split person from background.** The SAM3 masks say which pixels belong to
   which tracked person (resized to the VGGT resolution, nearest-neighbour).
   - The scene subtracts a *dilated* union of every tracked person — the mask grown
     by 8 px (`human_dilate`). Mask edges are imperfect, and a person pixel leaking
     into the wall becomes a phantom surface right where contact reasoning happens.
   - The human cloud keeps the *tight* mask, and only for people the pipeline
     actually reconstructed (the `object_ids` in `sam3d/smplx_params.npz`). A
     tracked bystander is therefore carved out of the scene but not added to the
     human cloud — they exist in neither.

3. **Backproject** the remaining pixels through the pinhole camera into world
   coordinates, using VGGT's depth as-is (no scale applied). Colour is sampled from
   the original frame resized to the VGGT resolution.

## Once per clip: merge the scene

The same wall is seen in every frame, so the concatenated scene points contain the
same surface hundreds of times. The merge collapses them.

### The voxel size is relative, not absolute

The old fixed 5 cm voxel is meaningless before the world has a scale, so the size
is now derived from the data:

```
voxel = voxel_px × (median scene depth) / (median focal length in pixels)
```

`depth / fx` is the *pixel footprint* — the width one pixel covers at that depth.
So `voxel_px = 5` means: merge points closer than what ~5 pixels resolve at the
typical scene distance. This has two consequences worth knowing:

- It is scale-free. Whatever factor the world is off by cancels in the formula, and
  the metricization later just multiplies the stored voxel size along with the points.
- It **adapts to how close the camera is**. A close-up bouldering wall gets a finer
  voxel (≈ 1.8 cm metric on `9RvrpYoWDoo_0001`) than the old 5 cm; a far scene gets
  a coarser one. That matches the data: depth noise and detail both scale with
  distance, so a fixed metric size over-resolves far scenes and throws away detail
  in near ones.

### The merge runs twice (the boundary-twin problem)

Voxel merging guarantees at most one point per grid cell — but **not** a minimum
spacing between output points. A flat surface that happens to straddle a cell face
leaves one centroid on each side of the face, arbitrarily close together. Measured
on real clips, up to 17% of merged scene points had a neighbour closer than a
quarter voxel. This is why the merged clouds visibly contained pairs much closer
than the voxel size.

The fix: merge a second time on a grid shifted by half a voxel. The twins land in
one shifted cell and collapse. The second pass weights each first-pass centroid by
how many raw points it came from, so a heavy centroid (a real surface patch seen by
thousands of pixels) is not dragged toward a light one. On the measured clips this
cut the sub-quarter-voxel pairs from 17% to 1.8% (worst clip) and from 6% to
~0.04% (typical), while removing 10–30% of redundant points.

Per merged point: position = count-weighted centroid, confidence = max, colour =
count-weighted mean.

One honest limit: VGGT's depth for the same wall disagrees by a few centimetres
between frames, so the merged wall is a *slab* about 1–4 cm thick (measured local
surface σ 0.3–2.5 cm), not an ideal sheet. The merge collapses duplicates; it does
not flatten this noise. Anything that needs a crisper surface has to average across
the slab (the scene-normal estimation downstream effectively does).

### Drop small clusters

Occupied voxels are grouped by 26-connectivity (touching faces, edges or corners),
and connected clusters of fewer than 25 voxels (`min_scene_voxels`) are removed.
These are floating depth fragments and off-surface blobs — for example an untracked
person VGGT placed in mid-air. Because the voxel size follows the pixel footprint,
"25 voxels" corresponds to roughly the same *image area* across scenes, so the
filter behaves consistently for near and far clips.

## What fuse does not do

- **It does not fix "floaters".** When VGGT places the person's depth at a
  different relative distance than the wall, fuse reproduces that error faithfully —
  the human cloud will hover off the wall. The old *body anchor* that tried to catch
  and repair this by rescaling the world has been removed; a more general treatment
  is planned. Until then a floater clip stays a floater.
- **It does not merge the human cloud**, per-frame fidelity matters more than size.
- **It does not gate the human cloud by confidence** — see above.

## Knobs (`configs/pipeline.toml [fuse]`)

| knob | default | meaning |
|---|---|---|
| `voxel_px` | 5.0 | scene merge voxel, in pixel footprints at the median scene depth |
| `conf_drop_q` | 0.3 | per-frame fraction of lowest-confidence depth dropped (scene only) |
| `edge_rtol` | 0.04 | relative depth jump that marks a flying pixel; 0 disables |
| `human_dilate` | 8 | px the person mask grows before being subtracted from the scene |
| `min_scene_voxels` | 25 | smallest connected voxel cluster the scene keeps |
| `person_obj_ids` | all | restrict which SAM3 ids count as people |
