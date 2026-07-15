# Human optimization: the four-stage body refinement

`scripts/stages/human_optimization.py` — one dispatcher, four stages: `root`,
`kinematics_1`, `kinematics_2`, `dynamics`. One config,
`configs/human_optimization.toml`. The default chain runs `root` + `kinematics_1`.

This doc explains the optimization problem each stage solves, every loss term, and the
decisions behind them.

## The engine

All fitting stages share one small engine, `tools/optim.py`. A stage describes its fit
as a **Problem**: the tensors being optimized (for kinematics, the per-frame body
configuration `q`), one `forward` that poses the body once per iteration, and a set of
named **loss terms** that read the posed body. **Phases** then drive the fit: each
phase has its own iteration budget, optimizer (Adam or L-BFGS), weight column, and a
mask saying which degrees of freedom may move.

A term is active in a phase exactly when its weight there is non-zero — so the weight
tables in the TOML *are* the objective, and turning a term off means writing a zero,
not editing code. Terms that share expensive intermediate results (the scene
nearest-neighbour search, the motion derivatives) share them through a per-iteration
cache, computed only when some active term needs them.

## What kinematics_1 starts from

- The scale stage's **pre-fit bodies** (`geometry/prefit.npz`) when they exist — the
  converter bodies already re-seated on their own depth points and re-projected (see
  [scale.md](scale.md)) — else the raw converter bodies (`sam3d/smplx_params.npz`).
  Either way the per-frame bodies are moved into the world frame using the metric
  cameras from fusion, and the priors below anchor to this starting pose. Shape
  (betas) is averaged over the person's valid frames and then **frozen** — a person
  has one body.
- Frames where the person was not tracked hold the nearest valid pose, and are excluded
  from every evidence term (their keypoint scores are zeroed, their depth cloud
  emptied, their frames masked out of the anchors).
- There is **no per-body scale variable**. The metric world was already sized to the
  body by the scale stage, so the body is used at its native size.

## The loss terms

Weights live in `configs/human_optimization.toml` as one column per stage. Term names
below are the config names.

### Image terms

- **`reproj_init` — the init anchor.** Each body joint is pinned to its *own* projected
  pixel position at the starting pose. The starting pose already reprojects well
  (the converter lands at ~5 px, the pre-fit better), and matching a joint to its own
  projection needs no detector and no correspondence table. It is the strongest 2D
  term. Its job: hold the body on the image so the 3D terms can move it in depth
  without sliding it across the picture. It cannot fix a starting-pose error — only
  preserve what is already good.
- **`reproj_sapiens` — the detector cross-check.** The same robust reprojection against
  Sapiens keypoints (Goliath-308 scheme), weighted by detection score. Down-weighted
  relative to the anchor; it is an independent observation that can correct the
  converter where the detector is confident.
- Both use a **Geman-McClure** robust kernel: small errors pull normally, gross
  outliers saturate with vanishing gradient, so one bad keypoint cannot yank the pose.
- The 2D **targets** are smoothed over ~0.35 s before optimization (Savitzky-Golay,
  within tracked runs only). This removes per-frame detector jitter from the objective
  so the smoothness terms stop fighting the targets. The *reported* errors are always
  measured against the raw targets, so smoothing cannot flatter the numbers.

### Depth term

- **`chamfer` — the term that owns depth.** Reprojection pins the image plane but says
  nothing about distance along the camera ray; the chamfer pulls the visible body
  vertices onto the person's metric depth cloud (and the cloud onto the body). Too
  weak and the body floats at whatever depth its apparent size suggests; too strong
  and it drags the body off-image chasing locally-wrong depth — the two reprojection
  terms are the counterweight.
- The per-frame cloud is confidence-gated per frame and capped with **limb quotas**:
  hands and feet keep up to 200 points each before the torso fills the rest of the
  budget. Torso pixels vastly outnumber hand pixels, so a uniform cap would starve the
  extremities of depth evidence.
- On the body side, vertices carry an **extremity emphasis** weight (hands/feet up to
  ×10, torso down to ×0.1) so the same imbalance does not recur among the vertices.
- Visibility is recomputed every iteration (front-facing vertices only): the depth map
  sees only the front surface.

### Priors

- **`pose_prior`** — joint rotations stay near the starting pose's per-frame estimate.
  The root is excluded, so placement stays free. This is what keeps occluded limbs sane.
- **`orient_prior`** — a gentle version of the same for the root *orientation* only.
  The smoothness stencil constrains the clip's end frames least, and without this the
  root orientation can run away at the boundary and break the image fit. Root
  translation stays free — depth placement is chamfer's job.

### Motion terms

- **`accel_*`, `jerk_*`, `snap_*`** — squared 2nd, 3rd and 4th finite differences of
  the trajectory, in the exact Lie tangent space (rotations are differenced as
  rotations, not as quaternion numbers). Root and joints are weighted separately: the
  root carries the whole trajectory, the joints only the articulation. Higher
  derivatives target higher-frequency jitter.
- **`radial_velocity`** — damps joint motion *along the viewing ray* only. Depth along
  the ray is the direction monocular evidence constrains worst, so that is where
  jitter concentrates; image-plane motion is left to the reprojection terms. The ray
  direction is held fixed (detached) so the term can only slow motion along it, never
  rotate the ray to cheat.

### Scene terms (refine phase onward)

The static scene cloud from fusion enters only after the body has settled. All three
terms share one nearest-neighbour pass per iteration: each body vertex finds its
nearest scene point and gets a signed distance (negative = behind the surface).

- **`penetration`** — pushes vertices out of the wall. A 1 cm dead zone tolerates soft
  tissue and noise; a Huber ramp keeps one deeply-buried vertex from dominating;
  matches to far or low-confidence scene points are ignored.
- **`ann_attraction`** — the inverse: pulls vertices *onto* the wall, but only for
  joints labeled as in contact. Attraction owns the region just outside the surface,
  penetration the region behind it — they partition the signed-distance axis and never
  fight. This term exists because nothing else pulls the body toward the *scene*
  (chamfer targets the person's own depth, penetration only pushes).
- **`ann_velocity`** — damps the world velocity of in-contact joints. A gripping hand
  should be still.
- **`ann_repulsion`** — pushes joints labeled *not in contact* out past the contact
  threshold. Only meaningful with manual labels (see kinematics_2); automatic labels
  carry no negatives.

Where do the labels come from in `kinematics_1`? **Auto-contact**: at the start of the
refine phase, the contact readout (same machinery as `estimate_contacts`) runs once on
the current body — a joint that is close to the wall and still gets labeled. The labels
are then **frozen** for the rest of the fit, and only positive labels are ever created.
Detection is one-shot and gradient-free, so the fit cannot talk itself into more
contact. The stillness measured *before* the attraction acted (`drift_pre`) is saved
into the output, because the readout later must not be gamed by the very damping this
stage applied — that circularity was observed and is the reason for the rule.

## The phase plan

| phase | optimizer | DOFs | scene | purpose |
|---|---|---|---|---|
| `root` (150 it) | Adam | root translation + orientation | off | rigid placement; joints frozen |
| `full` (300 it) | Adam | all | off | the visual fit: image + depth + priors + smoothness |
| `refine` (200 it) | Adam | all | on | add penetration + auto-contact terms on the settled body |
| `lbfgs` (30 it) | L-BFGS | all | on | second-order polish of the already-close fit |

Adam first because the gradient scales differ wildly (metres vs quaternions, pixels vs
metres) — Adam's per-coordinate steps tolerate that; L-BFGS cold would stall. A fresh
optimizer per phase so no stale momentum leaks across a DOF unfreeze. `root` and
`kinematics_1` share one optimization call so the priors and the init anchor stay
pinned to the starting pose across both.

The scene candidates (the subset of the scene cloud near the body) are rebuilt at the
start of each scene phase from the current pose, then reused for that phase's
iterations — the cloud is static and the body barely moves within a phase.

## kinematics_2 — annotation-guided refinement

Same problem, restarted from the `kinematics_1` result, with contact labels supplied
from outside: manual per-joint annotations (`data/<stem>.npz`), or the pipeline's own
`contacts_1.npz` in auto mode (positives only, weighted by per-joint label confidence).
Manual labels enable the full label set — attraction on in-contact joints, clearance
repulsion on no-contact joints, velocity damping — in every phase. The honest
`drift_pre` from kinematics_1 is passed through unchanged, so the contact readout on
this body still gates on pre-enforcement stillness.

## dynamics — force-only RNEA

With the kinematic trajectory **frozen**, solve for the world-frame contact forces.
RNEA (recursive Newton-Euler) computes, for a given motion, the forces and torques each
joint must produce. The body's root is a free-flyer — nothing can push on it — so a
physically consistent solution must make the six root components (the *base wrench*)
vanish. The optimizer (L-BFGS) finds per-frame forces at the contact joints (hands
resolved at the wrist, feet) minimizing:

| term | what |
|---|---|
| `base_wrench` | squared base wrench — the physics residual |
| `force_magnitude` | small L2 on the forces — picks the smallest forces that explain the motion |
| `force_smooth` | discourages frame-to-frame force flicker on persistent contacts |
| `torque_smooth` | the same for the joint torques |

There is deliberately **no upward-only constraint**: a climber's hand pulls down and
inward on a hold. Segment inertias come from per-part convex hulls of the body mesh
normalized to a total mass (70 kg default). Gravity direction comes from the GeoCalib
stage (world-frame; falls back to world-down when unreliable). Velocities and
accelerations are central finite differences of the frozen `q` — not denoised, by
decision: the forces should explain the reconstructed motion, not a prettified one.

## What is deliberately absent

- No per-body scale variable (the world is sized to the body upstream).
- No per-frame betas in the kinematic fit (one person, one shape); the scale
  stage's pre-fit likewise freezes shape to one canonical betas per person.
- No hand-tuned per-clip gates in the loss: a clip whose world is grossly wrong
  (e.g. the person's depth off the wall — a "floater") is an upstream problem,
  not something extra terms here should paper over.
