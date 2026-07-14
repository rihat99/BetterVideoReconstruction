# The scale stage: metric scale + per-frame body overfit

`scripts/stages/estimate_scale.py` — runs after the SMPL-X conversion, before `fuse_geometry`.

This doc explains what the stage does and why. It has two parts: first it finds the one
number that turns VGGT's depth into metres, then it overfits a diagnostic body to that
depth.

## Part 1 — the metric scale

### The problem

VGGT-Omega reconstructs the camera path and a depth map for every frame. The shape is
right, but the size is not: the whole reconstruction is off by one unknown factor per
clip. Depth and camera translations share that factor. Nothing in the video tells VGGT
whether the wall is three metres away or thirty.

### The idea: the human body is a ruler

SAM-3D-Body fits a body of real, metric size to each frame, and it does so in the same
camera VGGT uses. So over the pixels of the person, we have two depth readings for the
same surface: the body's metric depth, and VGGT's unscaled depth. Their ratio is exactly
the unknown factor `s`.

### How it is computed

For each frame where the person is tracked:

1. Pose the SAM-3D body with that frame's own shape and pose.
2. Keep only the vertices facing the camera. A depth map only sees the front of the
   person, so back vertices would compare the person's far side with their near side.
3. Project each kept vertex into the image. Read the VGGT depth under it. Keep the
   vertex only if it lands inside the person's mask and the depth there is valid.
4. Take `log(body depth) − log(VGGT depth)` per vertex, and reduce the frame to one
   number with a confidence-weighted median.

The per-frame numbers are then pooled into one clip scale: frames more than 3 MADs
(median absolute deviations — a robust version of "standard deviations from the mean")
from the median are dropped, and the rest are combined with a confidence-weighted
median. `s = exp` of the result.

### The deliberate choices

- **Log space.** Scale is multiplicative. In log space a 2× error up and a 2× error
  down are the same distance, so the median is not biased toward large scales.
- **Medians everywhere, no means.** Depth maps have gross outliers (edges, mask leaks).
  A median ignores them; a mean would chase them.
- **Every visible vertex counts the same.** No body-part weighting. VGGT distorts hands
  and feet the most, so weighting them up would push their error into the scale.
- **One scale per clip.** VGGT has a single scale by construction, so per-frame scales
  are only used as measurements to be pooled, never applied per frame.
- **Frames need at least 50 inlier vertices**, otherwise they are skipped. A frame with
  a handful of pixels on the person cannot give a trustworthy ratio.

### Output

`geometry/scale.npz`: the scale `s`, its log, the per-frame measurements and weights,
and how many frames survived pooling. `geometry/diagnostics/scale_over_time.png` plots
the per-frame scale so a drifting or bimodal clip is visible at a glance.

## Part 2 — the per-frame body overfit

### Why it exists

The kinematic stage (see [human_optimization.md](human_optimization.md)) fits one body
with fixed shape and smooth motion. That is the product, but it hides a question we
often need answered when auditing a clip: *how well can a body possibly match this
depth, if nothing holds it back?* The overfit answers that. It is a diagnostic body for
the viewer, not an input to any later stage.

### What it does

With `s` known, the person's masked VGGT depth becomes a metric point cloud per frame.
The stage then fits a fresh SMPL-X body to every frame independently:

- **Per-frame shape.** Betas (the 10 SMPL-X shape numbers) are free per frame. A real
  person has one shape; letting it flex per frame absorbs whatever the depth demands,
  which is exactly the point of an overfit.
- **No temporal coupling.** No smoothness, no shared pose. Frames never see each other.
- **No scene terms.** No penetration, no contact. Only the depth and the image.

Four losses, summed over the frames (run by the shared engine in `tools/optim.py`):

| term | what it does | why |
|---|---|---|
| `chamfer` (weight 1.0) | pulls front-facing body vertices onto the depth points, and the depth points onto the body, with a Geman-McClure robust kernel | this is the fit itself; the robust kernel lets gross depth outliers saturate instead of dragging the body |
| `reproj` (0.5) | robust reprojection of the body joints to the Sapiens 2D keypoints | holds the body on the image so chamfer cannot slide it along the ray to chase bad depth |
| `pose` (0.02) | small pull of the joint rotations toward the SAM-3D init pose | keeps joints with no depth evidence (occluded limbs) from drifting; deliberately weak |
| `betas` (0.001) | tiny pull of the shape toward the SMPL-X mean | a gauge only — stops the free shape from running away to a degenerate body |

Two implementation choices worth knowing:

- **The front-facing mask is computed once**, from the init pose, and frozen. The body
  only moves centimetres during the fit, so the front/back split barely changes, and
  recomputing full-mesh normals every iteration was the single largest cost.
- **Frames are optimized in batches** (default 512) purely to bound GPU memory. Since
  frames are independent, batching does not change the result.

### Output

`geometry/overfit.npz` — per-frame classic SMPL-X parameters (camera frame), same
layout as `sam3d/smplx_params.npz`. The viewer's *body source* dropdown shows it as
`overfit`, next to `original` and `kinematics`.
