# The scale stage: making the world metric

`scripts/stages/estimate_scale.py` — runs **after** `fuse_geometry`, before the
kinematic refinement.

This doc explains how the one number that turns VGGT's world into metres is
measured, how the **pre-fit** refines that number together with the body
placement, what exactly gets rescaled, and where the method can be wrong.

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

The measurement happens twice, in two forms. First the **ruler** reads `s` off
the converter's bodies exactly as they were stored — that gives a solid starting
value. Then the **pre-fit** refines it: the converter's body placement is itself
imperfect (translated a little off the depth, pose slightly wrong), and that
error leaks into `s`. The pre-fit alternates *aligning the bodies to the depth*
with *re-reading the scale from the aligned bodies* until the two agree.

## The ruler, step by step

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

## The pre-fit: move the body to the depth, move the scale to the body

`prefit = true` in `[scale]` (the default). The dual refinement alternates two
moves until they agree, starting from the ruler's `s`:

1. **The body move** (`tools/human_optim/prefit.py`). Every person's per-frame
   pose `q` is refined so the body sits on its fused depth points — **and the
   scale moves with it**: `log s` is a variable of the same optimization, so
   the cloud enters the fit as `s ×` its unscaled points. This matters because
   body depth and world scale are one soft direction (the depth match cannot
   tell "body nearer" from "world smaller"); if the round pinned the cloud at a
   fixed scale, the body could approach the image evidence only by a tiny
   offset per round (measured: a constant ~0.4%/round crawl that never
   terminates). Optimized jointly, the tug-of-war resolves inside the round —
   the image evidence decides how deep the body sits, and `s` follows.

   Joint rotations may change but are pulled back toward the converter pose,
   so bigger changes cost more; fingers are frozen (the cloud is far too
   coarse to see them). The root gets *gentle* pulls, not locks: orientation
   (a ~180° flip is nearly free on weak frames once the robust terms saturate)
   and translation (on a far, small person the reprojection barely changes
   over ±5% depth — without an anchor the body and `s` wander that valley
   together). **Shape is frozen** to one canonical betas per person — the
   per-coefficient median over the valid frames. The body's size *is* the
   ruler; a scale optimization that could stretch its own ruler would be
   circular (this, plus per-frame shape freedom, is what made the old
   "overfit" produce strange bodies).

2. **The scale move.** After each body round, `s` is re-read — not with the
   projection ruler. Each depth point matches its nearest *visible* body
   vertex (exactly the pairing the body move minimized) and votes
   `log(vertex depth) − log(cloud depth)`, weighted by the robust kernel's own
   weight for that match, so a point the fit treated as an outlier barely
   votes here either. Per frame: weighted median; across frames: the same
   MAD-gated pooling as the ruler. Reading through the fit's own
   correspondences gives the alternation a true fixed point — a converged
   round measures back the very `s` it fitted (a projection readout also sees
   grazing silhouette vertices the fit never matched, a small bias that would
   compound). With several people this pooled reading is also what reconciles
   their per-person fits into the one world scale.

Rounds repeat (`rounds`, default 4) and stop early once `s` changes by less
than 0.1% — typically after round 2 or 3. The per-frame scale spread (MAD)
after the pre-fit is typically several times tighter than the ruler's — the
frames agree about the scale once the bodies actually sit on their depth.

The body move's loss terms (run by the shared engine in `tools/optim.py`):

| term | weight | what it does |
|---|---|---|
| `chamfer` | 1.0 | every depth point is pulled toward its nearest **visible** vertex (one-sided, Geman-McClure). The reverse direction is deliberately absent: the cloud sees only the person's front, and pulling occluded vertices into the observed patch collapses limbs. Evaluated in *cloud units* (`body / s` against the raw cloud, kernel converted at the current `s`): in metric units every cloud-noise residual is proportional to `s`, so a fixed metric kernel would reward shrinking the world regardless of evidence. |
| `reproj` | 0.5 | robust reprojection to the Sapiens keypoints — holds the body on the image and is the evidence that sets its absolute depth |
| `pose` | 2.0 | pull of the joint rotations toward the converter pose ("bigger changes cost more") |
| `orient` | 1.0 | the gentle root-orientation pull |
| `transl` | 0.5 | the gentle root-translation pull (quadratic, metres) — negligible for cm-scale corrections, firm against metre-scale wander |
| `smooth` | 0.1 | tiny tie of consecutive-frame joint velocities to the converter's — damps jitter without fighting real motion |

The front-facing vertex mask is recomputed at the start of every round from the
current pose, so an earlier round's root rotation cannot leave it stale.

The refined bodies are saved to `geometry/prefit.npz` (classic SMPL-X params in
the camera frame, canonical betas repeated per frame) and show in the viewer as
body source **prefit**. The kinematic stage starts from them: when the file
exists, `human_optimization` reads its starting pose (and its priors' anchor)
from the pre-fit instead of the raw converter output.

## Metricization: what actually gets rescaled

The measured `s` is applied **in place** to the three files fuse wrote:

| file | what multiplies by `s` | what stays |
|---|---|---|
| `transform.npz` | camera translations (`extrinsics[:, :3, 3]`) | rotations, all intrinsics, FOVs |
| `scene.npz` | points, the stored voxel size | colours, confidence, counts, normals (unit directions) |
| `human.npz` | points | frame/object ids, confidence, colours |

`transform.npz` then records `metric=True`, the cumulative `scale`, and the
method. Downstream stages (kinematics, contacts) refuse to run on a world still
marked `metric=False`, so forgetting this stage fails loudly instead of fitting
a body into a five-times-too-small world.

**Re-running is safe.** The measurement always compares the body against the
world *as currently saved*. On an already-metric world the ruler reads back
the (unchanged) converter disagreement, the pre-fit re-runs from the same
deterministic converter init and lands at ~1.001, and the cumulative scale
stays put within 0.1% (verified). `scale.npz` stores the factor the latest run
applied (`scale`), the ruler's reading (`scale_ruler`) and the cumulative
world scale (`scale_total`).

## Known biases and limits (things a future update could attack)

- **The scale ends where the reprojection evidence ends.** The pre-fit lets the
  Sapiens keypoints, not the converter's translation, decide how deep the body
  sits — that was the point. But when the keypoints are weak (blur, occlusion),
  the pull toward the converter placement is what remains, so a systematic
  converter depth bias can still leak into `s` on hard clips. Watch the
  `scale_ruler` vs `scale` gap in `scale.npz`: a large gap means the pre-fit
  moved the world substantially away from the converter's opinion.
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
- **A systematic converter shape bias is invisible.** The canonical betas kill
  the frame-to-frame shape jitter, but if the converter thinks the person is 5%
  taller than they are, every measurement inherits it. Only an external size
  cue could fix that.

## Outputs

- `geometry/scale.npz` — `scale` (the final factor this run applied),
  `scale_ruler` (the converter-body starting value), `scale_total` (cumulative
  world scale), `log_scale`, per-frame scales and weights, the MAD, and how
  many frames survived pooling.
- `geometry/prefit.npz` — the aligned bodies (see the pre-fit section).
- `geometry/diagnostics/scale_over_time.png` — per-frame scale (per person) and
  frame weight over time, with the pooled global `s`.
- The rescaled `transform.npz`, `scene.npz`, `human.npz` (see above).
