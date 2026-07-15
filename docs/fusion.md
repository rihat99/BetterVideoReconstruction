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
  (`count`), a **surface normal** (see below), and the voxel size used.
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

   The edge filter runs once per frame (it is the expensive half of the
   cleaning) and both clouds share it. The confidence gate applies **only to the
   scene** — the person is exactly where we cannot afford to throw points away,
   and the scale stage applies its own confidence handling later.

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
between frames, so the merged wall is a *slab* (measured local surface σ 0.3–2.5 cm),
not an ideal sheet. The merge collapses duplicates; it does not flatten this noise.
The next step removes the part of the slab that hides behind the rest.

### Remove the hidden points (the slab's back)

When VGGT's frames disagree by more than a voxel — up to a whole duplicated
"ghost" copy of the wall a few voxels behind the real one — the merge keeps both
layers, because they occupy different voxels. Those back points sit behind the
front surface from *every* camera. The prune makes that the test:

**keep a point only if, in at least one frame, it is within one voxel of the
front-most point in its z-buffer bin.**

The z-buffer bins are `voxel_px` pixels wide — the merge voxel's own image
footprint — so the lateral test matches the cloud's resolution. The one-voxel
depth band lets a surface that slants through a bin keep all of its points. Both
quantities derive from the merge voxel, so the prune adds no knobs.

Why this cannot eat real geometry: a genuinely occluded surface (a wall patch
behind a hold, a corridor seen through a doorway) wins its own bins in the frames
that actually saw it — it can only lose everywhere if something sits in front of
it in *every* frame, and the only thing that does that is a duplicate estimate of
the same surface. One copy always survives; we keep the front one.

Measured on real clips: on clean reconstructions the prune is almost a no-op
(~1% of points); on the clip with the worst multi-frame disagreement it removed
30% of the merged points — coherent sheets with hundreds of raw observations
each, sitting several voxels behind the front wall. Exactly the ghost layer.

One known edge: a bin is several pixels wide, so right at a depth discontinuity
(the rim of a hold) a wall point can share a bin with the hold in front of it
and lose that frame's vote even though its own pixel saw it. It still survives
through any frame that views the rim from a different angle; only under nearly
static cameras could a one-bin-wide rim of real points be lost.

### Drop small clusters

Occupied voxels are grouped by 26-connectivity (touching faces, edges or corners),
and connected clusters of fewer than 25 voxels (`min_scene_voxels`) are removed.
These are floating depth fragments and off-surface blobs — for example an untracked
person VGGT placed in mid-air. Because the voxel size follows the pixel footprint,
"25 voxels" corresponds to roughly the same *image area* across scenes, so the
filter behaves consistently for near and far clips.

### Normals are computed here, once

Every surviving scene point gets a surface normal, saved in `scene.npz`. The
normal is the smallest principal axis of the point's 16 nearest neighbours
(local PCA), flipped to face the nearest camera — so "outward" consistently
means "toward the cameras that saw the wall". These used to be recomputed by
every downstream consumer (the kinematic stage's penetration term, the contact
stages, the viewer); now they are computed once, after the pruning, on the final
cloud. Consumers load them and subset them together with the points; geometry
written before this change falls back to the old on-the-fly estimate. Normals
are unit direction vectors, so the later metricization leaves them untouched.

The viewer can show them: the **scene normals** layer draws one arrow per scene
point, coloured by direction, sized by the merge voxel — a quick way to spot
flipped or noisy normals.

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
| `voxel_px` | 5.5 | scene merge voxel, in pixel footprints at the median scene depth (also sizes the hidden-point z-buffer bins) |
| `conf_drop_q` | 0.3 | per-frame fraction of lowest-confidence depth dropped (scene only) |
| `edge_rtol` | 0.04 | relative depth jump that marks a flying pixel; 0 disables |
| `human_dilate` | 8 | px the person mask grows before being subtracted from the scene |
| `min_scene_voxels` | 25 | smallest connected voxel cluster the scene keeps |
| `person_obj_ids` | all | restrict which SAM3 ids count as people |
