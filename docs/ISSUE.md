# Scale, fusion, and human-optimization audit

Audit date: 2026-07-14
Audited commit: `589359d` (`Restore contact dataset export and diagnostics`)

This is a read-only audit of the current scale, geometry-fusion, body-anchor, and
human-optimization implementation. It separates defects that affect current outputs
from conditional input-contract bugs, off-default stage bugs, and diagnostic/test gaps.
No source code or generated outputs were changed as part of the audit.

## Executive summary

The current implementation still has major correctness problems. The most severe
concrete bug scrambles SMPL-X vertex normals, corrupting approximately half of every
front-facing visibility mask used by scale estimation and human optimization. Even if
that arithmetic bug is repaired, the current body-anchor method cannot solve the
spaghetti effect: it preserves the distorted human point cloud and globally changes the
scene/camera scale around it. There is no limb-local compression or correction along
camera rays.

The active scale and chamfer choices also place most of their influence on hands and
feet, even though those are the body regions known to contain the strongest VGGT-Omega
distortion. Human optimization freezes body size, so it cannot repair the remaining
body/depth scale mismatch. Auto-contact cannot pull in a body unless it is already near
the wall, and accepted body-anchor results are allowed to remain several centimetres
away from the wall.

## Critical and active defects

### 1. Vertex-normal accumulation is incorrectly indexed

Location: `tools/human_optim/visibility.py:17-32`

`faces.reshape(-1)` produces indices in per-face order:

```text
f0v0, f0v1, f0v2, f1v0, f1v1, f1v2, ...
```

However, `torch.cat([fn, fn, fn], dim=1)` produces normal contributions in block order:

```text
n0, n1, ..., nF, n0, n1, ..., nF, n0, n1, ..., nF
```

The index and contribution rows therefore do not correspond, and face normals are
added to unrelated vertices.

An actual SMPL-X frame produced:

- Mean angular difference from correctly accumulated normals: approximately 88 degrees.
- Median normal cosine: approximately 0.035.
- Front-facing-mask disagreement: 48.19%, or 5,048 of 10,475 vertices.

Affected call sites include:

- Scale visibility: `scripts/stages/estimate_scale.py:197-219`.
- Optimizer diagnostics: `tools/human_optim/optimizer.py:149-178`.
- Every optimized chamfer forward pass: `tools/human_optim/optimizer.py:443-461`.

This means existing scale artifacts and the latest `kinematic_1` reruns were produced
with corrupted front-facing masks. The bug has existed since commit `5031a873`.

### 2. Body anchor is not de-spaghettization or along-ray squeezing

Locations:

- `scripts/stages/fuse_geometry.py:169-180`
- `scripts/stages/fuse_geometry.py:223-260`
- `tests/test_independent_script_fixes.py:34-80`

When body anchor fires, the scene and camera translations use the new global scale
`s_eff`, while points belonging to a fired person are transformed so that their old
camera-frame coordinates are retained. This preserves the per-frame human-cloud shape,
including stretched limbs. It does not compress limb depth, trim tails, constrain human
proportions, or alter the depth of individual points along their camera rays.

The mechanism changes the wall/camera world relative to an unchanged distorted human
cloud. The regression test explicitly preserves this split-scale behaviour.

Current `out_2` scenes where the anchor was applied are:

| Scene | `rescale_r` |
|---|---:|
| `2nwG1Qa5k_c_0115` | 0.9449 |
| `9RvrpYoWDoo_0001` | 0.8597 |
| `gtIGgs_y2nY_0024` | 0.8642 |

All retained spaghetti geometry in these human clouds remains present by construction.

### 3. Accepted body-anchor results are not required to touch the wall

Locations:

- Prospective 2 cm touch target: `tools/human_optim/depth_anchor.py:48-55`.
- Candidate selection: `tools/human_optim/depth_anchor.py:99-126`.
- Exact verification: `tools/human_optim/depth_anchor.py:380-416`.
- Built-world verification: `scripts/stages/fuse_geometry.py:261-289`.

Candidate selection uses a 2 cm touch target, but both exact verification stages only
enforce the burial limit. Touch is reported but never used as an acceptance condition.
Backing the scale toward one to avoid burial can therefore ship a partially rescued
world that still floats.

Current accepted `out_2` values demonstrate this:

| Scene | accepted true touch q25 |
|---|---:|
| `2nwG1Qa5k_c_0115` | 4.8 cm |
| `9RvrpYoWDoo_0001` | 4.9 cm |
| `gtIGgs_y2nY_0024` | 7.7 cm |

Consequently, `depth_refuted=True` does not imply that the fitted body touches the
scene.

### 4. `depth_refuted` is inferred without examining human depth

Locations:

- `tools/human_optim/depth_anchor.py:323-338`
- `scripts/stages/fuse_geometry.py:229-260`
- `tools/human_optim/depth_anchor.py:295-319`

`evaluate_body_anchor()` receives the SMPL-X initializer, scene points, scene
confidence, and cameras. It does not receive or evaluate `geometry/human.npz`.
Therefore, a disagreement between the body initializer and scene cannot be causally
attributed to bad human depth. Plausible alternative causes include:

- Bad static-scene depth.
- Wall geometry removed by human-mask dilation.
- Missing wall points or disconnected-component filtering.
- Incorrect SAM-3D body translation.
- Incorrect scene normals or nearest-neighbour correspondences.
- Mask leakage or an unrelated foreground surface.

Final verification also checks only SMPL-X against the scene. It never verifies that
the retained human cloud remains outside the shifted wall.

### 5. The anchor gate is structurally blind to per-limb spaghetti

Locations:

- `tools/human_optim/depth_anchor.py:177-183`
- `tools/human_optim/depth_anchor.py:185-227`
- `tools/human_optim/depth_anchor.py:229-243`

The float signal is the minimum scene distance over all four end effectors. One hand or
foot near the wall can hide three badly floating or stretched limbs. The scale
correspondence is then selected from a single minimum-SDF end-effector region. The
result is one clip-wide scalar, even though the observed failure is spatially and
temporally limb-specific.

The burial firing signal also examines only end effectors. A deeply buried torso does
not fire the burial gate unless another firing condition is met; whole-body burial is
checked only after an anchor candidate already exists.

### 6. Known-corrupt extremities dominate scale and chamfer

Locations:

- Scale weighting: `scripts/stages/estimate_scale.py:174-177`.
- Human-cloud limb quota: `tools/human_optim/stages.py:83-175`.
- Chamfer weighting: `tools/human_optim/stages.py:351-360`.
- Active configuration: `configs/human_optimization.toml:60-69`.
- Weight implementation and mesh-density warning:
  `tools/smplx_robot/vertex_weights.py:29-34,288-302`.

The scale estimator and vertex-to-depth chamfer use a geodesic end-effector weight
field ranging from 0.1 at the core to 10 at hands and feet. Human-cloud sampling also
guarantees up to 200 retained points for each hand/foot group.

The resulting field places approximately 85.4% of total raw vertex-weight mass on
hands and feet, despite them accounting for approximately 10.6% of body surface area.
Hands are especially overrepresented because SMPL-X is much more densely tessellated
there. An available vertex-area correction at
`tools/smplx_robot/vertex_weights.py:321-326` is not used.

This is the opposite of a robust response to spaghetti depth: the regions known to be
most distorted are given the greatest leverage over global scale and body placement.

### 7. Human optimization cannot correct body size

Locations:

- Mean betas and fixed body shape: `tools/human_optim/stages.py:385-396`.
- Fixed beta expansion and `q`-only optimization:
  `tools/human_optim/optimizer.py:275-276,383-385`.
- Hardcoded output scale: `tools/human_optim/stages.py:246-262`.
- Scale stage's per-frame betas: `scripts/stages/estimate_scale.py:194-203`.

Human optimization averages betas once, freezes them, and optimizes only the pose/root
trajectory `q`. It writes `scale=1` for every person. Therefore, if the optimized body
is smaller than the human depth cloud, no downstream degree of freedom can repair that
mismatch.

The scale stage uses each frame's individual betas as its metric ruler, while human
optimization uses their mean. The body used to choose world scale is consequently not
the same body fitted in the final optimization.

Every inspected `out` and `out_2` kinematic output has body scale exactly 1.0.

### 8. Corrupt depth is the optimizer's principal absolute-depth anchor

Locations:

- Raw bidirectional chamfer: `tools/human_optim/losses.py:135-187`.
- Root-free pose prior: `tools/human_optim/losses.py:455-470`.
- Orientation-only root prior: `tools/human_optim/losses.py:473-489`.
- Radial-velocity term: `tools/human_optim/losses.py:492-520`.

Chamfer uses untruncated, non-robust, bidirectional Euclidean distance. The
depth-to-vertex half remains uniform, so every retained spaghetti point continues to
pull some body vertex. The root prior does not constrain translation, image
reprojection does not determine absolute depth, and radial velocity only penalizes
changes in depth, not absolute depth.

This leaves the distorted human cloud as the main absolute-depth seat for the body.

### 9. Auto-contact has a circular dead zone

Locations:

- One-time detection and frozen labels: `tools/human_optim/optimizer.py:407-437`.
- Contact settings: `configs/human_optimization.toml:104-115`.
- Attraction application: `tools/human_optim/optimizer.py:482-503`.
- Body-anchor threshold: `scripts/stages/fuse_geometry.py:349-354`.

Auto-contact labels are detected only once from the already-positioned body. Entry
requires a joint to be within 8 cm of the wall before attraction can act. The body
anchor normally fires only when median end-effector distance exceeds 28 cm.

A consistently floating body between roughly 8 and 28 cm can therefore receive
neither contact attraction nor body-anchor correction. Penetration only pushes bodies
outward and cannot close a positive gap.

### 10. Temporal regularization is FPS-dependent

Locations:

- Contact velocity: `tools/human_optim/losses.py:410-452`.
- Radial velocity: `tools/human_optim/losses.py:492-520`.
- Acceleration, jerk, and snap: `tools/human_optim/losses.py:523-564`.

All temporal losses operate on raw frame-to-frame differences. FPS is used for printed
speed diagnostics but not for the optimized velocity losses, and no `dt` normalization
is applied to acceleration, jerk, or snap. Comments say the factors are folded into
weights, but one fixed configuration is used for clips at 23.976, 25, 30, and 60 FPS.

For equivalent physical motion, squared acceleration, jerk, and snap costs scale
approximately as `fps^-4`, `fps^-6`, and `fps^-8`. Relative to 24 FPS, a 60 FPS clip
therefore receives roughly 39x, 244x, and 1,526x weaker penalties for these terms.

## Scale-estimation defects and hazards

### 11. Scalar scale cannot repair non-rigid human-depth distortion

Location: `scripts/stages/estimate_scale.py:1-24`

The scale stage estimates one scalar and fusion applies it uniformly to depth and
camera translations. Uniform scaling preserves all depth ratios, limb proportions,
and ray-wise spread. It can metricize a coherent reconstruction, but it cannot
de-spaghettize a non-rigid human point cloud.

### 12. Inlier counts are vertices, not independent depth observations

Location: `scripts/stages/estimate_scale.py:96-117`

Projected vertices are rounded to pixels without deduplication. Many vertices may read
the same depth pixel, especially in densely tessellated hands. The
`min_inlier_verts=50` gate can therefore be satisfied by repeated copies of one or a
few independent depth samples. A synthetic one-pixel test with 100 duplicate vertices
passed the 50-inlier requirement.

### 13. Scale has no minimum temporal/person coverage

Location: `scripts/stages/estimate_scale.py:179-230`

Missing masks and depth files are silently skipped, and the only final requirement is
that `pooled_log` is non-empty. One usable person-frame can define the scale for an
entire clip. There is no minimum number or fraction of frames, no unique-frame count,
and no per-person agreement check.

For multi-person clips, raw confidence-times-vertex mass is pooled globally, allowing
a long/dense person track to dominate another person rather than treating people as
independent rulers. `n_frames_used` actually counts person-frame observations and can
exceed the number of video frames.

### 14. Zero-MAD aggregation admits arbitrary outliers

Location: `tools/geometry/human_scale.py:68-82`

If the median absolute deviation is exactly zero, every finite observation is marked
as an inlier. For example, log scales `[0, 0, 0, 10]` with equal weights retain all
four observations and produce scale approximately 12.18 instead of rejecting the
outlier. A nonpositive `mad_k` is also not validated and can produce an empty inlier
set or failure.

Current stored scale artifacts have nonzero MAD, so this is latent for the inspected
outputs.

### 15. Backface culling is not visibility

Locations:

- `tools/human_optim/visibility.py:1-9,35-45`
- `scripts/stages/estimate_scale.py:197-219`

Even after correcting the vertex-normal indexing bug, a front-facing test does not
perform z-buffering. Self-occluded body vertices can project onto a different visible
body surface and sample that surface's depth. The resulting log-ratio is then not a
correspondence between the vertex and its observed depth.

### 16. No scale sanity bounds or complete input validation

Locations:

- `scripts/stages/estimate_scale.py:227-246`
- `scripts/stages/estimate_scale.py:253-263`
- `tools/geometry/human_scale.py:62-82`

Any finite aggregated log scale is exponentiated and accepted. The stage does not
validate physical scale range, confidence sign/finiteness, frame-index uniqueness and
bounds, array shapes, or valid CLI ranges for quantiles, MAD multiplier, and minimum
inliers.

### 17. Gender is ignored after conversion

Locations:

- Gender-aware conversion: `scripts/stages/convert_mhr_to_smplx.py:137-175,272-284`.
- Neutral-only scale: `scripts/stages/estimate_scale.py:63-65,174`.
- Neutral-only human stages: `tools/human_optim/stages.py:40-42,348,566,875`.

The converter supports and stores neutral, male, and female bodies, but scale and all
human-optimization stages always load `SMPLX_NEUTRAL.npz`. Male/female parameters
would be interpreted against a different model basis, and optimized outputs discard
gender. Current inspected artifacts are neutral, so this is not active in the current
13 scenes.

### 18. Scale artifacts lack sufficient provenance

Location: `scripts/stages/estimate_scale.py:233-246`

The artifact has no schema/version marker, model gender, checkpoint identity, input
hashes, configuration values, or body/camera/depth generation IDs. The aggregation's
inlier mask is discarded. Plots show only accepted per-frame values and do not make
rejected observations clear.

## Fusion and body-anchor defects and hazards

### 19. Fusion completion is non-atomic and uses the first file as its sentinel

Locations:

- Skip condition: `scripts/stages/fuse_geometry.py:198-200`.
- Write order: `scripts/stages/fuse_geometry.py:293-324`.
- Pipeline completion check: `scripts/pipeline.py:129-143,237-242`.

`transform.npz`, `scene.npz`, and `human.npz` form one logical artifact, but
`transform.npz` is written first and its existence alone marks fusion complete. A crash
after the first write leaves a partial geometry set that all `--skip-existing` paths
accept permanently. No schema, dependency, or provenance check is performed.

Scale output has the same existence-only, non-atomic caching pattern.

### 20. Fusion does not validate scale provenance

Locations:

- Scalar-only scale read: `scripts/stages/fuse_geometry.py:194-205`.
- Scale metadata producer: `scripts/stages/estimate_scale.py:233-246`.

Fusion reads only `scale.npz["scale"]`. It does not compare the scale artifact's
`frame_indices` or `object_ids` against current VGGT and SMPL-X inputs. An old scale can
silently metricize a newly generated camera/depth/body set.

Current `out` and `out_2` scale/camera/object IDs happen to match.

### 21. A full rerun can fit SAM-3D using stale intrinsics

Locations:

- Pipeline order: `scripts/pipeline.py:54-58,230-242`.
- Intrinsics priority: `scripts/stages/estimate_body_sam3d.py:142-184`.

On an existing output tree, the pipeline can regenerate VGGT and then run SAM-3D before
new fusion. SAM-3D prefers an old `geometry/transform.npz` over the newly written
`vggt/camera.npz`, silently combining stale intrinsics with new camera/depth outputs.

### 22. Base scale and effective scale use ambiguous metadata

Location: `scripts/stages/fuse_geometry.py:298-312`

`geometry/scale.npz["scale"]` remains the base human-ruler scale, while
`geometry/transform.npz["scale"]` becomes `base_scale * rescale_r`. Nevertheless,
`scale_method` remains `human-anchored (estimate_scale)` and does not identify body
anchor as an additional transformation. Current fired scenes differ by approximately
5.5% to 14% between the two files.

Stored anchor metadata also lacks object IDs, eligibility/fired flags, consistency,
veto reason, thresholds, or fusion configuration, so `depth_refuted` is difficult to
map or interpret safely.

### 23. Current `33ODIivxhF8_0025` contains old veto diagnostics

The stored `out_2/33ODIivxhF8_0025/geometry/transform.npz` has `rescale_r=1` and
`depth_refuted=False` but retains finite `anchor_pred_gap_q`. It was generated before
the current `_record_anchor_veto()` behaviour in `scripts/stages/fuse_geometry.py:93-99`
and was not updated when the source changed.

This is an artifact-generation mismatch rather than a current source-code path, but it
means existing diagnostics do not uniformly satisfy current invariants.

### 24. Multi-person body-anchor verification is incomplete

Locations:

- Fixed-person handling: `scripts/stages/fuse_geometry.py:172-180,257-260`.
- Healthy-person prediction: `tools/human_optim/depth_anchor.py:353-360`.
- Exact verification: `tools/human_optim/depth_anchor.py:395-412`.
- Built-world verification: `scripts/stages/fuse_geometry.py:269-271`.

Only fired people retain their original camera-frame human depth. An unfired person's
human cloud is globally rescaled away from its initializer. The healthy-person check
uses fixed correspondences, while exact nearest-neighbour verification covers fired
people only. Final touch/burial quantiles are pooled, allowing a long track to hide
another person's violation.

### 25. Body-anchor defaults and controls are not wired through the pipeline

Locations:

- Missing configuration: `configs/pipeline.toml:35-41`.
- Fusion argument builder: `scripts/pipeline.py:154-164`.
- Default-on CLI: `scripts/stages/fuse_geometry.py:349-354`.

The standalone fusion stage defaults body anchor to on, but pipeline configuration
cannot disable it or adjust `float_gate_m`. Database preprocessing also uses the CLI
defaults. The 28 cm threshold is explicitly described as calibrated on the same
eight-scene hard set used to evaluate it, so its generalization is unverified.

### 26. Downstream staleness is not managed

Location: `scripts/stages/fuse_geometry.py:288-291`

Changing scale, masks, scene filtering, FPS, or fusion parameters makes
`kinematic_1`, contacts, `kinematic_2`, and dynamics stale. A warning is
printed only when a body-anchor rescale is successfully applied. Other fusion changes
produce no warning or invalidation. A default full pipeline refreshes only through
`kinematic_1`, leaving older optional outputs available for accidental consumption.

### 27. VGGT preprocessing transforms are discarded

Locations:

- Crop/pad implementation: `third_party/vggt-omega/vggt_omega/utils/load_fn.py:37-55,68-80,108-127`.
- Adapter output: `tools/adapters/vggt_omega.py:163-181`.
- Full-frame resize in fusion: `scripts/stages/fuse_geometry.py:126-129`.
- Intrinsics reconstruction: `scripts/stages/fuse_geometry.py:296`.

VGGT can center-crop extreme aspect ratios and pad mixed-sized frames. The applied
crop/pad mapping is not saved. Fusion and scale resize uncropped masks/images directly
to VGGT resolution, and full-resolution intrinsics are denormalized without undoing the
preprocessing. This misregisters masks, color, depth, and intrinsics on such inputs.

Current clips use fixed supported aspect ratios and do not activate this case.

### 28. Long-clip subsampling is not supported end to end

Locations:

- VGGT subsampling: `scripts/stages/estimate_camera_vggt.py:9-12,107-140`.
- SAM-3D camera coverage requirement: `scripts/stages/estimate_body_sam3d.py:156-190`.
- Fused frame IDs: `scripts/stages/fuse_geometry.py:145-150`.
- Human-stage assumptions: `tools/human_optim/stages.py:337-345`.

VGGT advertises uniform subsampling beyond `max_frames=500`, but SAM-3D requires
intrinsics for every tracked frame and therefore fails. If subset artifacts are
provided manually, fusion writes VGGT row indices as human frame IDs and stores raw
source FPS rather than effective sampled FPS, while human optimization assumes body and
camera trajectory lengths are identical.

Current clips have identity frame indices and do not activate this case.

### 29. Stale and missing SAM masks are silently accepted

Locations:

- Numeric-directory discovery: `scripts/stages/fuse_geometry.py:61-63`.
- Silent missing-mask handling: `scripts/stages/fuse_geometry.py:74-85`.
- Tracking output writes: `scripts/stages/track_sam3.py:735-762,849-862`.

Fusion discovers every numeric directory rather than validating against the canonical
object IDs in `sam3/bboxes.npz`. Rerunning tracking with fewer people does not clear old
object directories, so stale masks can still carve the scene. Missing masks are skipped
without an error, causing a person to leak into the static scene and disappear from the
human cloud.

### 30. Single-image input is incompatible with scale/fusion mask layout

Locations:

- Single-frame tracker layout: `scripts/stages/track_sam3.py:758-760,810-820`.
- Scale mask expectation: `scripts/stages/estimate_scale.py:207-219`.
- Fusion mask expectation: `scripts/stages/fuse_geometry.py:61-85`.

The pipeline accepts image extensions, but the tracker writes single-image masks in a
flat layout while scale/fusion require `<object_id>/frame_XXXXXX.png`. A single-image
pipeline run therefore cannot complete these stages correctly.

### 31. Empty-scene handling occurs after an empty-sensitive operation

Locations:

- Voxelization: `scripts/stages/fuse_geometry.py:153-165`.
- Intended empty-scene branch: `scripts/stages/fuse_geometry.py:223-227`.
- Empty reduction: `tools/geometry/voxel.py:16-18`.

If all scene masks are empty, `voxel_downsample()` calls a reduction on an empty tensor
before the later empty-scene body-anchor check. A read-only synthetic probe produced an
`amin()` empty-dimension error.

### 32. Non-finite confidence can discard a complete frame

Locations:

- Confidence cleaning: `tools/geometry/clean.py:26-29`.
- Fusion call: `scripts/stages/fuse_geometry.py:121-124`.

`np.quantile()` propagates NaN. One non-finite confidence value can therefore make the
threshold NaN and reject every pixel in that scene frame. Current `out` and `out_2`
confidence arrays are finite.

### 33. Overlapping person masks are assigned by numeric object-ID order

Location: `scripts/stages/fuse_geometry.py:74-85`

The first numeric object ID claims every overlap pixel. Read-only checks found 4,650
overlapping full-resolution pixels in `out_2/3DAg...` and 29,694 in
`out/competition_2`; individual frames reached 1,109 and 836 overlap pixels. Overall
fractions are small, but the assignment is arbitrary and can distort the human cloud
near person-person occlusion boundaries.

## Human-optimization defects and conditional problems

### 34. Invalid frames still influence priors and smoothness

Locations:

- Camera-frame nearest fill: `tools/human_optim/stages.py:186-201,390-397`.
- Validity masking of data/scene terms:
  `tools/human_optim/stages.py:417-421` and
  `tools/human_optim/optimizer.py:402-469`.
- Ungated priors/smoothness: `tools/human_optim/optimizer.py:462-467`.

Invalid parameters are filled from the nearest valid frame in camera coordinates and
then folded through each frame's moving camera. A held camera-frame pose is therefore
not a flat world trajectory. Reprojection, depth, scene, and radial terms are masked,
but pose/orientation priors and smoothness include every frame and cross invalid-frame
boundaries. Camera-glued invalid poses can pull adjacent valid frames.

All 13 current local scenes are fully valid, so this defect is latent for current
outputs.

### 35. Severe penetration receives no corrective gradient

Location: `tools/human_optim/losses.py:253-294`

Vertices whose signed distance is more negative than `max_pen_m=0.10` are removed from
the active penetration mask. The code treats them as likely wrong correspondences, but
this also means a body more than 10 cm inside a wall receives no penetration gradient
from those vertices.

### 36. Per-frame confidence quantiles keep bad frames alive

Location: `tools/human_optim/stages.py:120-175`

Each frame retains a fixed top fraction of human depth, even when every point in that
frame is low quality. There is no absolute or cross-frame quality gate, so a globally
bad frame receives similar frame-level influence to a good frame.

### 37. Scene candidates are frozen for an optimization phase

Location: `tools/human_optim/optimizer.py:390-405`

Scene candidates are selected once from the pose at phase start and reused for all
iterations. If the body moves farther than the configured prune margin, relevant scene
points may be absent and stale correspondences remain available.

### 38. Stage dependencies and provenance are not enforced

Locations:

- Pipeline output checks: `scripts/pipeline.py:129-143,237-242`.
- Human stage dispatcher: `scripts/stages/human_optimization.py`.

Arbitrary stage subsets are permitted without dependency/version validation.
`kinematics_2` and dynamics can consume stale files. Running `root` and later
`kinematics_1` as separate invocations reinitializes from converter parameters rather
than consuming the previously written root result. Existence-only caching cannot
distinguish compatible output from an old generation.

### 39. Optimizer metrics do not reproduce the optimized objective

Location: `tools/human_optim/optimizer.py:149-213`

Reported Sapiens error is binary-thresholded rather than score-weighted and robustified
like the actual loss. Reported chamfer omits active vertex weights. Some diagnostics do
not apply `frame_valid`. Consequently, printed before/after values cannot be interpreted
as faithful measurements of the objective being optimized.

`scripts/diagnose_human_optim.py` also omits limb quota, extremity weighting, and parts
of invalid-frame handling, so it cannot reproduce the active optimization path.

### 40. Human-optimization configuration is barely validated

Location: `tools/human_optim/config.py:34-67`

The loader mainly checks phase-weight array length. It does not validate documented
relations such as contact `max_dist > band/exit`, `prune_margin >= max_dist`, threshold
ordering, nonnegative values, required keys, or stage names. Incorrect configurations
can silently disable contact or scene terms.

## Manual `kinematics_2` and dynamics defects

### 41. Default root/kinematics_1 does not use manual annotations

Manual annotations are read only by `kinematics_2` in
`tools/human_optim/stages.py:519-564`. The default `root,kinematics_1` run does not read
`data/` or `data_2/` annotations.

### 42. `data_2` annotations are resolved and then deliberately rejected

Locations:

- Path resolution: `tools/human_optim/stages.py:505-516`.
- Schema rejection: `tools/human_optim/stages.py:481-488`.

The manual annotation resolver searches `data_2`, but `_contact_labels()` rejects its
14-joint schema. All eight checked `data_2/*.npz` files use this schema. Therefore,
default manual `kinematics_2` discovers files that it cannot consume.

### 43. Manual annotation velocity leaks into phases described as annotation-free

Locations:

- Kinematics-2 phases: `tools/human_optim/stages.py:306-316`.
- Label binding and velocity: `tools/human_optim/optimizer.py:328-336,498-503`.

Manual labels are bound before optimization. Attraction and repulsion wait for scene
phases, but annotation velocity is applied whenever a phase has a nonzero velocity
weight. This can affect root/full phases despite the configuration's refine-only
description.

### 44. Dynamics can associate contacts with the wrong person

Location: `tools/human_optim/stages.py:885-915`

If a body object ID is missing from contact output, dynamics falls back to positional
person index `p`, which can silently apply another person's contacts or raise an index
error. The saved `joint_contact` array is sliced rather than reordered to output body
IDs, so saved labels can disagree with the contacts used during solving.

### 45. Dynamics drops contact-force moment arms

Location: `tools/human_optim/stages.py:702-711`

Contact forces are applied at a representative wrist/foot body with a zero angular
wrench. The `r x F` moment from the actual fingertip/toe contact point is omitted,
underestimating external moments.

Additional dynamics inconsistencies include:

- `base_wrench` and `force_magnitude` weights are effectively squared inside residuals,
  unlike smoothness weights.
- Invalid frames participate in trajectory derivatives.
- Some saved raw torque arrays remain nonzero on invalid frames even where summary
  force/torque outputs are cleared.

## Pipeline and operational issues

### 46. Built-in fallback configuration cannot run the default chain

Location: `scripts/pipeline.py:62-75`

The fallback configuration sets `vggt.save_depth=False`, while its default stages
include scale and fusion, both of which require saved depth. If the regular TOML is
missing or the fallback is otherwise used, the default pipeline necessarily fails at
scale/fusion.

### 47. Stage failures can terminate a database shard unexpectedly

The scene-database driver catches `Exception`, while several stages fail through
`SystemExit`, which is not an `Exception`. A single stage failure can terminate the
whole shard rather than recording one failed clip and continuing.

### 48. Current outputs mix generations

- `out` is predominantly pre-body-anchor geometry.
- `out_2` includes body-anchor-era geometry.
- The latest `kinematic_1` outputs were rerun against these different geometry
  generations.
- Optional descendants such as contacts, `kinematic_2`, and dynamics may
  predate the latest fusion or kinematic run.

Comparisons between `out` and `out_2` therefore do not isolate one algorithmic change.

## Test and diagnostic coverage gaps

The focused repository suite passed:

```text
16 passed
```

However, there is no unit or integration test for:

- Vertex-normal accumulation, even on a two-triangle mesh.
- Front-facing visibility correctness.
- Scale estimation or robust aggregation edge cases.
- Unique-pixel evidence and coverage requirements.
- Complete/atomic fusion output.
- Body-anchor firing, selection, exact verification, or multi-person safety.
- Empty scenes, missing masks, overlapping masks, or stale object directories.
- VGGT crop/pad coordinate mapping.
- Long-clip frame-index/FPS handling.
- Chamfer gradients and weighted loss behaviour.
- Temporal loss invariance across FPS.
- Invalid-frame masking across priors and smoothness.
- Auto-contact dead-zone behaviour.
- Optimizer end-to-end numerical behaviour.
- Dynamics object-ID association.

The existing fusion tests cover only veto-diagnostic mutation and preservation of the
split-scale human-point policy. In particular, a six-vertex/two-triangle test would
have immediately caught the critical normal-indexing error.

Running bare `pytest` also discovers legacy/third-party tests and fails during
collection because of package-name collisions and a missing `imageio_ffmpeg`
dependency. The focused `tests/` invocation is currently required.

## Dead or stale implementation residue

- `tools/human_optim/velocity.py` exports joint-local speed functionality with no
  active consumer.
- Compatibility branches for body scale remain although the producer always writes
  ones.
- `tools/smplx_robot/vertex_weights.py` says the weighting is not wired, although it is
  actively used by scale and chamfer.
- README references obsolete `optimize_smplx` paths and does not accurately describe
  the current body/annotation pipeline.
- Viewer documentation overstates `kinematic_2`/dynamics loading; the current viewer
  body selection path uses `kinematic_1`.
- The previous ray-squeeze/de-smear implementation is absent. The active remaining
  experimental mechanisms are limb quota, extremity-up weighting, and body anchor.

## Current artifact checks that passed

The following checks passed for all 13 currently inspected `out`/`out_2` scene sets:

- Geometry triplets `transform.npz`, `scene.npz`, and `human.npz` exist.
- Camera, body, and kinematic frame counts align.
- Transform, scene, human, and kinematic arrays are finite.
- Intrinsics/extrinsics have expected shapes.
- Frame indices are identity mappings and human frame IDs are in range.
- Scale/camera/transform frame indices agree.
- Scale and SMPL-X object IDs agree.
- Human/object IDs align with tracked people.
- Kinematic quaternion norms and world-frame schema are structurally valid.
- Expected SAM masks exist.
- Depth-confidence arrays are finite.
- Current inputs do not require the unsupported extreme-aspect VGGT crop path.
- Current valid masks are fully valid, so the invalid-frame defect is not active here.
- Current body genders are neutral, so the neutral-model mismatch is not active here.
- Current scale MAD values are nonzero, so the zero-MAD aggregation bug is latent here.

The following implementations were also checked and found internally correct for their
stated conventions:

- Camera/world inverse, projection, and composition math.
- Standard depth backprojection.
- Uniform scaling of scene points and camera translations.
- The camera-offset formula used to preserve old camera-frame human points. The formula
  is algebraically correct; preserving those points is the unsuitable strategy.
- Attribute ordering for fused points, colors, confidence, frame IDs, and object IDs.
- Root/kinematic Sapiens and contact lookup by object ID, except for the dynamics
  fallback described above.
- Penetration and attraction gradient signs.
- Quaternion shortest-arc logic and saved quaternion normalization.
- Viewer application of world-frame bodies and scale for the body type it actually
  loads.

These structural successes do not validate the geometry or resolve the critical
algorithmic defects above.

## Worktree note

At audit time, the following tracked documentation files were already deleted in the
working tree and were left untouched:

- `docs/ISSUES.md`
- `docs/UPDATE.md`
- `docs/UPDATE_AGENT.md`
- `docs/confidence_factor_audit.md`
- `docs/geometry_fusion_plan.md`
- `docs/round12_depth_root_cause.md`
