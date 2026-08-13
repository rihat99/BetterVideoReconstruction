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

### Occlusion weighting (`[occlusion].strength`) — off, and here is why

Wired but **off** (`strength = 0`). The idea was reasonable and the measurement says no.

The idea: the three terms that speak *for a specific joint* — `reproj_sapiens`,
`reproj_init` and `pose_prior` — should go quiet when the camera cannot see that joint,
because behind an occluder both of their inputs are guesses (the detector is placing a
keypoint it cannot see; the converter pose they pin to came from the same hidden pixels).
Turning them down should hand the joint to the terms that still know something —
smoothness and the contact damping.

The visibility signal itself works. It is the joint's Sapiens **detection score**,
smoothed over time and divided by that keypoint's own 90th percentile across the clip;
the division is what makes it mean *visibility* rather than detector calibration (Sapiens
scores hips near 0.65 and wrists near 0.95 even in plain view, so only the *drop* is
informative). Checked against an exact camera-ray occlusion cast on four scenes, the mean
score is **0.905 when the joint is clear and 0.614 when body surface stands in front of
it** — a clean separation. Joints with no keypoint (spine, neck, collars, head, feet,
fingers) inherit their parent's number, so a foot follows its ankle and the fingers follow
their wrist. `strength` scales the drop: `weight = 1 − strength·(1 − ratio)`.

What the measurement showed (6 scenes, occlusion measured by ray cast on the baseline
bodies so every variant is scored on the same joint-frames): turning the visual terms down
makes occluded joints jitter **more**, monotonically — mean Cartesian acceleration on
occluded joint-frames 12.1 → 14.1 → 15.8 mm/frame² at strength 0 → 2 → 5, while contact F1
slipped .578 → .575 and reprojection drifted +1.5 px. So on this data those three terms
are the things *stabilising* an occluded joint, not the things shaking it; removing them
leaves the joint free to wander. The jitter must be coming from somewhere else — the
chamfer is the suspect, since behind an occluder the person's depth cloud belongs to the
occluding limb and the nearest-neighbour match is simply wrong.

Every one of the three terms keeps its *unweighted* denominator, so a down-weighted joint
**loosens** its term rather than pushing its share onto the joints that stayed visible.
Only `kinematics_1` reads the knob.

### Motion terms

- **`accel_*`, `jerk_*`, `snap_*`** — squared 2nd, 3rd and 4th finite differences of
  the trajectory, in the exact Lie tangent space (rotations are differenced as
  rotations, not as quaternion numbers). Root and joints are weighted separately: the
  root carries the whole trajectory, the joints only the articulation. Higher
  derivatives target higher-frequency jitter.
  The `kinematics_1` accel weights are ~17× the other stages'. A sweep over ×1/×4/×16.7/
  ×41.7 moved joint-angle jitter 2.75 → 2.69 → 2.53 → 2.35 °/frame with contact F1 flat
  and reprojection within 0.05 px: monotone the whole way, so the old value was simply far
  below where the term starts to bite. Jerk and snap were *not* raised — they were already
  the dominant smoothness weights, and ×4 on them cost F1 (.578 → .560).
- **`velocity_*`** — the 1st difference, the same family one step down. **Off** (weight
  0): unlike the higher differences it penalizes motion itself, not the roughness of
  motion, so it slows the climber as well as the jitter. Measured at the accel weights it
  bought only −2.3% jitter and cost F1 (.578 → .575).
- A difference is **dropped where its stencil touches an untracked frame**. Those frames
  hold a pose copied from their nearest tracked neighbour, so the step across a gap is
  an artefact of the hold, not motion — without the gate the smoothness terms spend
  their budget flattening it. All 13 annotated clips are fully tracked, so this changes
  nothing there; it matters on corpus clips with tracking gaps.
- **`radial_velocity`** — damps joint motion *along the viewing ray* only. Depth along
  the ray is the direction monocular evidence constrains worst, so that is where
  jitter concentrates; image-plane motion is left to the reprojection terms. The ray
  direction is held fixed (detached) so the term can only slow motion along it, never
  rotate the ray to cheat. This is the **most effective** smoothing weight measured:
  ×4 and ×15 give Cartesian joint acceleration −11% / −19% and root translation
  acceleration −14% / −25%, for +0.02 / +0.12 px of reprojection and no F1 cost. It beats
  the generic smoothness terms because it spends its whole budget on the one axis the
  images cannot see. Shipped at ×4 alongside the raised accel; together, on all 13
  annotated scenes, they cut joint-angle jitter 8.9%, Cartesian joint acceleration 16%
  and root translation acceleration 19%, while contact F1 goes .488 → .497.

### Scene terms (refine phase onward)

The static scene cloud from fusion enters only after the body has settled. A point
finds its nearest scene point and gets a signed distance (negative = behind the
surface). Under the frame-level labels the whole chain now uses, that point set is
the **12 contact frames**, and one nearest-neighbour pass per iteration serves both
scene terms; only the manual per-joint mode runs the same pass over 1000 subsampled
mesh vertices instead.

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
  (contact frames under the frame labels; near-contact vertices under manual
  per-joint labels). Attraction owns the region just outside the surface,
  penetration the region behind it — they partition the signed-distance axis and never
  fight. This term exists because nothing else pulls the body toward the *scene*
  (chamfer targets the person's own depth, penetration only pushes).
- **`ann_velocity`** — damps the world velocity of the in-contact points — both how fast
  they *translate* and how fast they *rotate* (the angular part carries a smaller weight,
  `[ann_velocity].angular_weight`). A gripping hand should hold both its place and its
  facing. A point's world position/orientation differenced over time already *is* its
  LOCAL_WORLD_ALIGNED velocity, so no Jacobian is needed.
- **`ann_acceleration`** — the same idea one derivative up: damps the world *acceleration*
  of the in-contact points (steady, jerk-free holds). Off by default — a point already held
  still by `ann_velocity` is already low-acceleration, so it only earns its keep if
  `ann_velocity` is lowered to let a contact slide slowly.
- **`ann_repulsion`** — pushes points labeled *not in contact* out past the contact
  threshold. Only meaningful with the manual per-joint annotation; every automatic label
  source is positives-only, so this term is inert in the pipeline chain.

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
`contacts_1.npz`, one label per **contact frame** per video frame, positives
only. That is the same axis contact is detected on, forced at and measured on
everywhere else, so nothing has to be translated on the way in. Manual
annotations (`data/<stem>.npz`, 10- or 14-joint schema) remain available via
`[stages.kinematics_2] labels = "manual"` but are metrics-only in the pipeline;
they label coarse joints, so in that mode the labels reach the mesh through each
vertex's nearest joint, exactly as they always did.

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
labeled contact frame on one video frame. For each instance a small **target
patch** is built once per phase: the scene points visible where that point
appears in the image (projecting within `radius_px` of it, confidence-gated, the
`patch_k` nearest kept). Pulling toward the wall *seen at that pixel* keeps the
pull nearly reprojection-neutral. An instance with no such wall points is
reported unresolved and simply not pulled — a missing hold cannot be repaired
honestly. Two terms act on the instance's point:

- **`ann_acquisition`** — a pseudo-Huber pull on the distance to the patch: a
  bounded, roughly constant force from far away that fades out at contact. One
  point per label, and it is the contact frame itself — the labeled toe, heel,
  knee, palm or fingertip. A contact label says *that patch of skin* touches, so
  that is what gets pulled, not the whole hand.
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

**Why one point, not the whole limb.** An earlier version simply strengthened
`ann_attraction`, which pulls *all* of a group's vertices. It buried the body
many centimetres into the wall on every test scene: once the front vertices
cross the surface and drop out of the loss, the back vertices keep pulling. The
one-point-per-instance form makes that failure impossible rather than tuned
away. In the per-joint era that point had to be *found* — the closest vertex of
a 14-joint limb group, re-chosen every phase. With frame-level labels there is
nothing to find: the label names the point.

**Phases.** Scene terms (and the patches) are on from the **root phase**: the
rigid root phase does the bulk push toward the wall, then `full` and `refine`
re-settle the pose. There is no L-BFGS phase here — the per-phase patches form
an active set whose churn breaks its line search. The in-contact velocity damper
runs throughout; the clearance repulsion needs *no-contact* labels, which only
the manual annotation carries, so it is inert in auto mode. People whose
annotation is absent or marked `ignored` are copied through from `kinematic_1`
untouched. The acquisition weight is the floater knob: it sets the force balance
against reprojection, and was chosen by sweep (see the config comments).

**Where stillness comes from, on this body.** `kinematics_1`'s `drift_pre` —
the motion measured before any contact term acted — is passed through unchanged,
and the contact readout on this body (`contacts_2`) gates on it rather than
re-measuring. It has to: this stage's velocity damper holds exactly the points
`contacts_1` called touching still, so a re-measured drift would let
`contacts_1` confirm itself through `contacts_2`. `drift_pre` is the one
stillness measurement no contact enforcement anywhere in the chain can
manufacture, which is why it is the source for every readout downstream of it.

**Measured (2026-08-12, all 13 annotated scenes, the real judge).** Running the
full auto chain `kinematics_1 → contacts_1 → kinematics_2 → contacts_2` moved
pooled contact F1 from **.721 to .747** (P .860 → .854, R .621 → .664): 9 scenes
up, 3 down by ≤ .014, 1 flat. The gains are concentrated exactly where the stage
is meant to act — the two worst-placed bodies, `gtIGgs_y2nY_0024` +.094 and
`2nwG1Qa5k_c_0115` +.113 — and they are paid for in reprojection: on `gtIGgs`
the body is dragged so far onto the wall that Sapiens reprojection goes
9.5 → 26.4 px and its chamfer to the human depth cloud 16.6 → 56.0 cm. That is
the trade this stage exists to make, but it is a real one. Pooled over the 13
scenes the body-joint angular jitter is unchanged (2.279 → 2.278 deg/frame)
while Cartesian joint acceleration rises 5.91 → 6.64 mm/frame² and root
translation acceleration 4.05 → 4.90, essentially all of it in the four scenes
whose bodies actually moved.

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
body. (`[stages.kindyn_1] labels = "manual"` switches back to the annotation,
whose coarse joints are spread onto the contact frames each one covers;
`bodies = "kinematic_1"` starts from `kinematic_1` + `contacts_1` for chains
that skip `kinematics_2`.) The objective is the full `kinematics_2` objective —
reprojection, priors, motion smoothness, scene penetration, the annotation
terms — **plus** the `dynamics` terms (`base_wrench`, `force_magnitude`,
`force_smooth_mag`/`_dir`, `torque_smooth`), with one world-force vector per
**contact frame** per video frame added as extra optimization variables, masked
to the steps the labels say that frame touches. Chamfer stays off, for the same
reason as in `kinematics_2`: kindyn starts from the repaired body, and chamfer
would pull it back toward the spaghetti-prone depth the repair just left.

**Where the forces are applied.** At the 12 contact frames, not at joints — the
same points the labels and the contact readout's force witness use. A world
force `f` at a contact frame reaches RNEA as an external wrench on that frame's
**parent joint**: the force itself plus the moment `r × f` of its lever arm from
the joint origin, both expressed in the joint's local frame. The lever arm is
recomputed from the current pose every iteration, with gradient — that is how
moving the body changes the physics, which is the entire point of the stage. Two
frames that share a parent joint (a foot's two toe frames) accumulate their
wrenches rather than overwriting each other.

**Friction cones, on by default.** Each contact force is confined to a 45°
friction pyramid about that frame's own **contact normal** — its outward skin
normal, negated, because a touching patch of skin faces *into* the surface and
the surface pushes back the other way. It is the same axis and the same
construction the force witness uses (`tools/human_optim/forces.py`): the force
is a nonnegative combination of the pyramid's four edge directions, so it cannot
leave the cone and there is no penalty weight to tune. A patch of skin can then
only push the way it faces. `[stages.kindyn_1] cones = false` gives every frame
a free bilateral force instead.

The cones are frozen once, at the warm-start pose. The pose moves during the
solve, so a cone recomputed every iteration would be a moving target the body
could rotate to make force cheap; the shifts here are small because the stage
starts from a converged body.

Cones cost residual, by construction — measured on `boulder_1` and the repaired
`gtIGgs_y2nY_0024` floater (base_wrench 100), the root residual after the joint
solve is 114.8 / 83.8 N per frame RMS with cones and 42.3 / 60.6 without. That
is the constraint doing its job, not a regression: an unconstrained force can
explain *anything*, including loads no real contact could produce (a toe sucking
on a hold), so a smaller residual bought that way says less, not more. The
earlier per-joint version of this stage shipped with cones off, but its cones
came from the *scene* cloud at foot and ankle joints, where a thin hold or a
concave corner averages neighbouring normals to noise; the body's own skin
normal has no such failure mode and needs no scene lookup at all.

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
dynamics weights are **linear** — the term is `0.5 · mean over valid frames of
Σ residual²` and the weight multiplies it, like every other engine term — unlike
the standalone `dynamics` stage, which folds the weight inside the square.

Every physics residual is first divided by the **body weight** and averaged over
the valid frames. Without that the terms are a *sum* of newtons squared, so the
same weight pulls hard on a 900-frame clip and barely registers on a 60-frame
one, and harder on a heavy climber than a light one — while every kinematic term
stays a per-frame mean. In body-weight units all five are O(1) and one number
means one thing everywhere. (The force witness solves in the same units for the
same reason.) The three smoothness terms keep the `dynamics` stage's ratios to
`base_wrench` exactly.

`base_wrench` is the trade knob — it buys physics with pose — and the ladder
behind the shipped 100 is in the config comments. The headline: on `boulder_1`
the frozen-pose warm-start alone takes the root residual 632 → 140 N per frame
RMS, and the joint phase then buys 140 → 115 for +0.35 px of reprojection at
weight 100, or 140 → 93 for **+5.8 px** at weight 1600 (which is what the old
un-normalised `1e-5` amounted to on that clip). Most of the win is the warm
start; the pose's contribution is real but expensive, so it is bought cheaply.

## What is deliberately absent

- No per-body scale variable (the world is sized to the body upstream).
- No per-frame betas in the kinematic fit (one person, one shape); the scale
  stage's pre-fit likewise freezes shape to one canonical betas per person.
- No hand-tuned per-clip gates in the loss: a clip whose world is grossly wrong
  (e.g. the person's depth off the wall — a "floater") is an upstream problem,
  not something extra terms here should paper over.
