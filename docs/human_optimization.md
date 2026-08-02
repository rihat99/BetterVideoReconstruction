# Human optimization: the body refinement stages

`scripts/stages/human_optimization.py` — one dispatcher over the
body-refinement stages: `root`, `kinematics_1`, `kinematics_2`, `dynamics`, and
`kindyn_1` (joint kinematics + dynamics). One config,
`configs/human_optimization.toml`.

The default pipeline chain runs `root` + `kinematics_1`. The remaining stages
are standalone follow-ups; run in order, with the contact readout
([contacts.md](contacts.md)) between them, they form the full-auto chain:

```
kinematics_1 → contacts_1 → kinematics_2 → contacts_2 → kindyn_1
```

Each contact readout measures the body it is given; each later stage consumes
the labels of the readout before it. No manual annotation is needed anywhere in
that chain — manual labels stay available for evaluation and via the
`labels = "manual"` switches.

This doc explains the optimization problem each stage solves, every loss term,
and the decisions behind them.

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

The static scene cloud from fusion enters only after the body has settled. A point
finds its nearest scene point and gets a signed distance (negative = behind the
surface). In `kinematics_1` that point set is the **12 contact frames**, and one
nearest-neighbour pass per iteration serves both scene terms; the manual
`kinematics_2` terms run the same pass over 1000 subsampled mesh vertices instead.

- **`penetration`** — pushes the 12 contact frames out of the wall. A 1 cm dead zone
  tolerates soft tissue and noise; a Huber ramp keeps one deeply-buried point from
  dominating; matches to far or low-confidence scene points are ignored. It acts on
  the contact frames rather than on mesh vertices because those points are where
  contact is *modelled*: a frame buried 8 cm inside a hold gives the detector a
  meaningless signed distance, and this term keeps the signals honest. Note what it
  does *not* do: the dead zone (1 cm) sits inside the contact band (3 cm), so
  penetration never decides *which* frames read as touching — the distance and
  stillness gates do. With ~12 points instead of ~1000 the penalty (a mean over active
  points) is much more concentrated, so its weight is correspondingly larger.
- **`ann_attraction`** — the inverse: pulls the labelled points *onto* the wall
  (contact frames under the auto labels; near-contact vertices under manual
  `kinematics_2` labels). Attraction owns the region just outside the surface,
  penetration the region behind it — they partition the signed-distance axis and never
  fight. This term exists because nothing else pulls the body toward the *scene*
  (chamfer targets the person's own depth, penetration only pushes).
- **`ann_velocity`** — damps the world velocity of in-contact joints — both how fast
  they *translate* and how fast they *rotate* (the angular part carries a smaller weight,
  `[ann_velocity].angular_weight`). A gripping hand should hold both its place and its
  facing. The joint's world position/orientation differenced over time already *is* its
  LOCAL_WORLD_ALIGNED velocity, so no Jacobian is needed.
- **`ann_acceleration`** — the same idea one derivative up: damps the world *acceleration*
  of in-contact joints (steady, jerk-free holds). Off by default — a joint already held
  still by `ann_velocity` is already low-acceleration, so it only earns its keep if
  `ann_velocity` is lowered to let a contact slide slowly.
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

There is a second label source: `[stages.kinematics_1] labels = "predicted"` swaps the
geometric detector for the external image model's per-limb contact predictions
(`predictions/contacts.npz`, see [contact_fusion.md](contact_fusion.md)). Everything
else is identical — same binding point, same freeze, same `drift_pre` honesty. This is
the peter force-plate configuration (`configs/human_optimization_peter.toml`): static
cameras give VGGT no parallax, so its confidence is uniformly low at the limbs and the
geometric detector has nothing to detect against — but the un-gated clouds (the peter
configs set `conf_drop_q = 0` at fuse/scale/chamfer) do contain the board and the
hands, so chamfer stays on to anchor depth, the predicted labels drive the velocity
damper, and kinematics_2 seats the labeled limbs on the recovered geometry.

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

## kinematics_2 — contact-patch repair

A repair stage. It restarts from the `kinematics_1` result and moves the body —
sometimes tens of centimetres — until every labeled contact actually touches the
wall.

**Where the labels come from.** The pipeline default is auto: our own
`contacts_1.npz` (detector + completion labels, positives only). Manual
annotations (`data/<stem>.npz`, 10- or 14-joint schema) remain available via
`[stages.kinematics_2] labels = "manual"` but are metrics-only in the pipeline.

**Why the labels are not confidence-gated.** On a floating body every witness
honestly reports near-zero confidence, so gating on confidence would veto
exactly the repair this stage exists for. Instead the pull itself tests the
label, and reprojection plus the pose priors referee it — a wrong label simply
fails to win against them. This was confirmed the hard way (2026-07-26):
scaling each pull by its label confidence left the floating-body scene ~30 cm
off the wall, because its true contacts carry confidence .07–.34 while
boulder_1's *false* positives carry .92 and kept pulling. Arming the clearance
repulsion with the labels' negatives was tried the same day — it only added
jitter.

**Why chamfer is off.** The human depth cloud is exactly the spaghetti-prone
evidence being repaired, so the term that matches the body to it is disabled.
Reprojection and the pose priors are what keep the repair from diverging.

**How the pull works.** The unit of repair is a **contact instance** — one
labeled group (a hand, a foot, an ankle…) on one frame. For each instance a
small **target patch** is built once per phase: the scene points visible where
that limb appears in the image (projecting within `radius_px` of the limb,
confidence-gated, the `patch_k` nearest kept). Pulling toward the wall *seen at
the limb's pixel* keeps the pull nearly reprojection-neutral. An instance with
no such wall points is reported unresolved and simply not pulled — a missing
hold cannot be repaired honestly. Two terms act on each instance's single
closest vertex:

- **`ann_acquisition`** — a pseudo-Huber pull on the distance to the patch: a
  bounded, roughly constant force from far away that fades out at contact. Only
  the group's closest vertex is pulled — a contact label means *some* surface
  point touches, not the whole hand.
- **`ann_seating`** — **off by default since 2026-07-26.** It used to drive
  that vertex a few millimetres *inside* the wall, on the theory that its
  symmetric push-back was the anti-burial guard. Measured on boulder_1/3, the
  opposite was true: the −5 mm target was itself the main source of penetration
  (it applies to every instance, including the labels that are wrong), and
  removing it cut mean vertex penetration 2.03 → 0.51 and 0.93 → 0.26 mm while
  the true contacts still ended up at the same wall distance as the
  manual-annotation chain. Acquisition's fading pull plus the ordinary
  penetration term seat the limb on their own. The term stays wired; raise its
  weight to re-enable. Propagated over all 13 `out/` scenes (2026-07-26), the
  change plus the cones-off default moved the mean kindyn_1 body from
  14.2 → 12.5 mm jitter and 2.1 → 1.0 mm mean penetration, improving every
  scene's penetration and 12/13 scenes' jitter, with the contacts_2 labels
  unchanged (pooled F1 .833 before and after).

**Why closest-vertex, not the whole limb.** An earlier version simply
strengthened `ann_attraction`, which pulls *all* of a group's vertices. It
buried the body many centimetres into the wall on every test scene: once the
front vertices cross the surface and drop out of the loss, the back vertices
keep pulling. The per-instance closest-vertex form makes that failure
impossible rather than tuned away.

**Phases.** Scene terms (and the patches) are on from the **root phase**: the
rigid root phase does the bulk push toward the wall, then `full` and `refine`
re-settle the pose. There is no L-BFGS phase here — the per-phase patches form
an active set whose churn breaks its line search. Clearance repulsion on
no-contact joints and the in-contact velocity damper run as before. People
whose annotation is absent or marked `ignored` are copied through from
`kinematic_1` untouched, and the honest `drift_pre` from kinematics_1 is passed
through unchanged, so the contact readout on this body still gates on
pre-enforcement stillness. The acquisition weight is the floater knob: it sets
the force balance against reprojection, and was chosen by sweep (see the config
comments).

## dynamics — force-only RNEA

With the kinematic trajectory **frozen**, solve for the world-frame contact forces.
RNEA (recursive Newton-Euler) computes, for a given motion, the forces and torques each
joint must produce. The body's root is a free-flyer — nothing can push on it — so a
physically consistent solution must make the six root components vanish. Those six are
the *base wrench*: a 3D residual force plus a 3D residual torque left over at the pelvis.
The optimizer (L-BFGS) finds per-frame forces at the **six contact joints** — the two
hands (resolved at the wrist), the two feet, and the two ankles — minimizing:

| term | what |
|---|---|
| `base_wrench` | squared base wrench — the physics residual |
| `force_magnitude` | small L2 on the forces — picks the smallest forces that explain the motion |
| `force_smooth_mag` | discourages frame-to-frame change in force **magnitude** on persistent contacts |
| `force_smooth_dir` | the same for force **direction**, smoothed more gently — the force may re-aim faster than it changes size |
| `torque_smooth` | the same for the joint torques |

A joint carries a force only on the frames it is labelled in contact.

**Friction cone.** A hold or foothold can only *push* on the body, never pull, and it
can only push sideways so much before the foot slips. The forces it may apply form a
cone around the surface normal, opening to a half-angle θ (default 45°, so the
sideways force may be at most `tan θ = 1` times the outward push).

The cone is not a loss term — it is built into the force itself. At each coned contact
the force is a sum of the cone's four *edge directions* (the normal tilted by θ toward
each side), each scaled by a learned non-negative amount. Any such sum lies inside the
pyramid those edges span, so the force can never leave the cone or pull — by
construction, with no weight to tune. (The pyramid is inscribed in the round cone, so
between its edges it is slightly stricter than θ.)

The surface normal at each contact comes from the fused scene cloud: take the nearest
scene points to the joint and average their stored normals, which already point out of
the wall toward the climber. A contact whose joint is too far from any surface, or
whose neighbouring normals disagree (a thin hold or a concave corner averages opposing
normals to nothing), falls back to a free force with no cone — better none than one
aimed arbitrarily. Hands are left out on purpose: a climbing hand grips a hold and
pulls down and inward on it, so its force is genuinely bilateral.

Force smoothness is split in two so the **magnitude** of a contact force can be held
steadier than its **direction**: `force_smooth_mag` damps the along-force change and
`force_smooth_dir` the sideways (re-aiming) change, using an orthogonal split about the
mean direction. Equal weights reproduce the old single `force_smooth` term exactly.

The body's mass and per-segment inertia are **baked by BetterHuman**. The SMPL-X body
is itself a BetterRobot model, and BetterHuman integrates real closed per-part meshes to
get each segment's mass, centre of mass and inertia (about 72 kg at the neutral shape,
and it shifts with the person's `betas`). This stage therefore no longer builds its own
inertia — it reads the body's, which is both simpler and more faithful than the old
convex-hull estimate that had to be rescaled to a fixed 70 kg.

Gravity direction comes from the GeoCalib stage (world-frame; falls back to world-down
when unreliable). Velocities and accelerations are central finite differences of the
frozen `q` — not denoised, by decision: the forces should explain the reconstructed
motion, not a prettified one. The per-frame base wrench is saved so the viewer can draw
the leftover root force and torque as arrows at the pelvis (the **root wrench** layer).

By default the stage refines the `kinematics_1` body (`[stages.dynamics].bodies`, which
can point at `kinematics_2` instead) and runs on the GPU. The reworked BetterRobot RNEA
is fast in plain PyTorch; its Warp kernel, though available, measured slower for this
force-plus-gradient solve, so the torch path is used.

## kindyn_1 — joint kinematics + dynamics

**Why it exists.** The chain above is one-way: `kinematics_1/2` fit the motion
using only what the camera sees, then `dynamics` solves the contact forces on
that finished, **frozen** pose. The physics never gets a say in the motion — if
the pose is a little off, the force solve just absorbs the error into a large
leftover base wrench. `kindyn_1` closes the loop: it optimizes the pose **and**
the contact forces together, so the physics residual can push back on the
motion.

**What it optimizes.** It starts from the converged `kinematics_2` body and its
contact labels — by default `contacts_2.npz`, the readout measured on that very
body. (`[stages.kindyn_1] labels = "manual"` switches back to the annotation;
`bodies = "kinematic_1"` starts from `kinematic_1` + `contacts_1` for chains
that skip `kinematics_2`.) The objective is the full `kinematics_2` objective —
reprojection, priors, motion smoothness, scene penetration, the annotation
terms — **plus** the `dynamics` terms (`base_wrench`, `force_magnitude`,
`force_smooth_mag`/`_dir`, `torque_smooth`), with one world-force vector per
contact group per frame added as extra optimization variables. Chamfer stays
off, for the same reason as in `kinematics_2`: kindyn starts from the repaired
body, and chamfer would pull it back toward the spaghetti-prone depth the
repair just left.

**Friction cones are available but off by default**
(`[stages.kindyn_1] cones = true` re-enables). When on, the cone frames are
read off the scene once, at the warm-start pose, and frozen — the pose moves
during the solve, so re-reading the normals every iteration would make the cone
a moving target. They are off because, measured on boulder_1/3, they stack with
contact-label errors (jitter 6.7 → 8.3 mm) and hurt boulder_1 even with
hand-annotated labels (penetration 1.5 → 4.2 mm): a hooking heel or toe
genuinely pulls on the wall, which a push-only cone forbids. Turn them on for
scenes where every foot contact really pushes, like the campus-board
force-plate clips.

**How it runs.** Two steps. First, a quick frozen-pose force solve (like the
`dynamics` stage's, but with this stage's weights) establishes physically
reasonable forces — with zero forces the body is in freefall, and the base
wrench says nothing useful about the pose. Then a single joint **L-BFGS**
polish moves the pose and forces together.

L-BFGS is the right tool here. The pose and the ~1000 N forces live on wildly
different scales, and a fixed-step optimizer (Adam) cannot co-adapt them — it
leaves the forces frozen and lets the pose flail against a stale physics
residual, which wrecks the motion. L-BFGS's line search keeps every step from
overshooting, so the pose settles gently onto the physically-solved forces.

In practice the forces do not move at all during the joint phase — their
gradient is ~10⁸× smaller than the pose's, below the shared line search's step
resolution. So the stage is effectively *solved forces + pose refinement
against them*, and the whole win comes from the **pose** shifting to be
consistent with the physics. (That also means the saved forces are the ones
solved at the *starting* pose; re-solving them at the final pose would give a
slightly different, self-consistent pair — a possible future alternation step.)

**The weights.** The kinematic columns of `kindyn_1` are copied straight from
`kinematics_2`, so the visual fit is asked for exactly as before. The five
dynamics weights are **linear** — the term is `0.5·Σ residual²` and the weight
multiplies it, like every other engine term — unlike the standalone `dynamics`
stage, which folds the weight inside the square. `base_wrench` is tiny (`1e-5`)
on purpose: the base wrench is on the order of 1000 N while the visual losses
are order 1–100, so at weight 1 it would simply overwrite the visual fit. The
three smoothness terms keep the `dynamics` stage's ratios to `base_wrench`
exactly; `force_magnitude` ends up ~100× heavier relative to `base_wrench` than
in the `dynamics` stage (that stage folds its 0.01 *inside* the square, making
it effectively 1e-4 of base — here the linear 0.01 ratio is kept). This only
shapes the warm-start (the force terms are constant while the forces sit frozen
in the joint phase) and the tuned result met its targets, so the value stays.
Tuned on `boulder_1`: the base wrench drops far below the sequential value
while the chamfer and reprojection grow only a few percent.

## What is deliberately absent

- No per-body scale variable (the world is sized to the body upstream).
- No per-frame betas in the kinematic fit (one person, one shape); the scale
  stage's pre-fit likewise freezes shape to one canonical betas per person.
- No hand-tuned per-clip gates in the loss: a clip whose world is grossly wrong
  (e.g. the person's depth off the wall — a "floater") is an upstream problem,
  not something extra terms here should paper over.
