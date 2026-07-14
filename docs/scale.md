# The scale stage: making the world metric

`scripts/stages/estimate_scale.py` — runs **after** `fuse_geometry`, before the
kinematic refinement.

This doc explains how the one number that turns VGGT's world into metres is
measured, what exactly gets rescaled, and where the method can be wrong. The
last part describes the per-frame body overfit, which is currently switched off.

## The problem

VGGT-Omega reconstructs the camera path and a depth map for every frame. The
shape is right, but the size is not: the whole reconstruction is off by one
unknown factor per clip. Depth and camera translations share that factor.
Nothing in the video tells VGGT whether the wall is three metres away or thirty.
Fuse therefore hands over an *up-to-scale* world (`metric=False`), and this
stage finds the factor `s` and applies it.

## The idea: the human body is a ruler

SAM-3D-Body fits a body of real, metric size to each frame, and it does so in
the same camera VGGT uses. So over the pixels of the person we have two depth
readings for the same surface: the body's metric depth, and the fused human
cloud's unscaled depth. Their ratio is exactly the unknown factor `s`.

## How the measurement works, step by step

For each person, for each frame where their body fit is valid:

1. **Pose the body** with that frame's own shape (betas) and pose, at the
   converter's translation — exactly as `sam3d/smplx_params.npz` stores it.
   Nothing is re-fit or nudged first (see *Known biases* below).
2. **Keep only camera-facing vertices.** A depth map sees only the front of the
   person; comparing a back vertex against the cloud would compare the person's
   far side with their near side and bias the ratio by the body's thickness.
   Facing is decided by the vertex normal against the view ray.
3. **Project each kept vertex into the image** and look up the fused human cloud
   at that pixel. The cloud (`geometry/human.npz`) stores world points per frame
   and person; transforming them back to the camera and projecting them recovers
   exactly the pixels fuse backprojected them from, so the lookup is a sparse
   depth map containing only person pixels that survived fuse's edge filter.
   A vertex whose pixel holds no cloud point is dropped — that automatically
   applies the person mask, the validity checks and the flying-pixel filter,
   without this stage re-reading any raw depth or mask files.
4. **Drop the least confident 30%** of the surviving vertices (per-frame
   quantile of the cloud's VGGT confidence, `conf_drop_q = 0.3`).
5. **Reduce the frame to one number**: the confidence-weighted median of
   `log(body depth) − log(cloud depth)` over the vertices. A frame with fewer
   than 50 surviving vertices (`min_inlier_verts`) is skipped — a handful of
   pixels cannot give a trustworthy ratio.

Then the per-frame numbers are pooled into one clip scale:

6. Frames more than 3 MADs (`mad_k`; median absolute deviations — a robust
   version of "standard deviations") from the median log-scale are dropped.
7. The survivors are averaged, weighted by each frame's confidence mass, and
   `s = exp` of the result.

### The deliberate choices

- **Log space.** Scale is multiplicative. In log space a 2× error up and a 2×
  error down are the same distance, so the pooling is not biased toward large
  scales.
- **Medians inside a frame, no means.** Depth has gross outliers (mask edges,
  smeared limbs). A median ignores them; a mean would chase them.
- **Every visible vertex counts the same.** No body-part weighting. VGGT
  distorts hands and feet the most, so weighting extremities up would push their
  error into the scale.
- **One scale per clip.** VGGT has a single scale by construction. The per-frame
  scales are only measurements to be pooled, never applied per frame. The
  diagnostics plot makes a drifting or bimodal clip visible at a glance.

## Metricization: what actually gets rescaled

The measured `s` is applied **in place** to the three files fuse wrote:

| file | what multiplies by `s` | what stays |
|---|---|---|
| `transform.npz` | camera translations (`extrinsics[:, :3, 3]`) | rotations, all intrinsics, FOVs |
| `scene.npz` | points, the stored voxel size | colours, confidence, counts |
| `human.npz` | points | frame/object ids, confidence, colours |

`transform.npz` then records `metric=True`, the cumulative `scale`, and the
method. Downstream stages (kinematics, contacts) refuse to run on a world still
marked `metric=False`, so forgetting this stage fails loudly instead of fitting
a body into a five-times-too-small world.

**Re-running is safe.** The measurement always compares the body against the
world *as currently saved*. On an already-metric world the ratio comes out at
1.0 and the rescale is a no-op (verified: a second run measures exactly
1.0000). `scale.npz` stores both the ratio measured by the latest run (`scale`)
and the cumulative world scale (`scale_total`).

## Known biases and limits (things a future update could attack)

- **The converter's translation is trusted as-is.** The body is posed at
  SAM-3D-Body's estimated depth and never moved before measuring. If that
  estimate is systematically 10% too far, `s` comes out 10% too large — the
  error transfers one-to-one. A likely future refinement is to align the body
  to the cloud first (e.g. a small translation-only fit along the ray) and only
  then read the ratio, so translation error and scale stop being entangled.
  Lateral (in-image) translation error is milder: projected vertices then land
  on neighbouring cloud pixels, and the per-frame median absorbs most of it.
- **The ruler makes the *person* right, not the wall.** The scale compares the
  body with the person's own depth pixels. If VGGT placed the person's depth at
  the wrong distance *relative to the wall* (the "floater" failure), the
  measurement is internally consistent and `s` is still recovered cleanly — but
  the body, placed at its own depth, hovers off the wall in the fused world.
  Detecting and repairing that is deliberately out of scope here (the old fuse
  body anchor was removed; a more general mechanism is planned).
- **Baggy clothes and motion blur** put the cloud surface a few cm outside the
  skin surface the SMPL-X body models. Both are handled statistically (medians,
  MAD gate), not modelled.
- **Per-frame betas** follow the converter output. Frames disagree slightly
  about the body's size; that disagreement lands in the per-frame scatter and is
  averaged out by the pooling.

## Outputs

- `geometry/scale.npz` — `scale` (this run's measured ratio), `scale_total`
  (cumulative world scale), `log_scale`, per-frame scales and weights, the MAD,
  and how many frames survived pooling.
- `geometry/diagnostics/scale_over_time.png` — per-frame scale (per person) and
  frame weight over time, with the pooled global `s`.
- The rescaled `transform.npz`, `scene.npz`, `human.npz` (see above).

## The per-frame body overfit (currently off)

`skip_overfit = true` by default; set it to `false` in `[scale]` to re-enable.
It will return in a modified shape later.

With the world metric, the person's masked depth becomes a per-frame metric
point cloud, and the stage can overfit a fresh SMPL-X body to every frame
independently — free shape (per-frame betas), no temporal coupling, no scene
terms. It answers an audit question the kinematic fit hides: *how well can a
body possibly match this depth if nothing holds it back?* It is a diagnostic
for the viewer (`body source: overfit`), never an input to later stages.

Its four losses (run by the shared engine in `tools/optim.py`):

| term | weight | what it does |
|---|---|---|
| `chamfer` | 1.0 | pulls front-facing vertices onto the depth points and back, Geman-McClure robust kernel — the fit itself |
| `reproj` | 0.5 | robust joint reprojection to the Sapiens keypoints — stops chamfer sliding the body along the ray |
| `pose` | 0.02 | weak pull toward the SAM-3D init pose — keeps occluded limbs from drifting |
| `betas` | 0.001 | tiny pull toward the SMPL-X mean shape — a gauge, stops runaway shapes |

Implementation notes that matter: the front-facing mask is computed once from
the init pose and frozen (recomputing normals every iteration was the single
largest cost), and frames are optimized in batches of 512 purely to bound GPU
memory (frames are independent, so batching does not change the result).
