# Human optimization: the body refinement stages

`bvr/stages/human_optimization.py` — one dispatcher over the
body-refinement stages: `root`, `kinematics_1`, `kinematics_2`, and `kindyn_1`
(joint kinematics + dynamics). One config,
`configs/human_optimization.toml`.

The default pipeline chain runs `root` + `kinematics_1`. The remaining stages
are standalone follow-ups; run in order, with the contact readout
([contacts.md](contacts.md)) between them, they form the full-auto chain:

```
kinematics_1 → contacts_1 → kinematics_2 → contacts_2 → kindyn_1
```

Each contact readout measures the body it is given; each later stage consumes
the labels of the readout before it. No manual annotation is needed anywhere in
that chain — manual labels are used only for evaluation
(`datasets/examples/score_frame_contacts.py`).

This doc explains the optimization problem each stage solves, every loss term,
and the decisions behind them.

## The engine

All fitting stages run on BetterRobot's optimizer (`better_robot.optim`), through
the small glue in `bvr/human_optim/solve.py`. A stage describes its fit as a
**Problem**: the tensors being optimized (for kinematics, the per-frame body
configuration `q`), one cached `forward` that poses the body once per iteration into a
lazy `State`, and a set of named scalar **loss terms** that read it. **Phases** then
drive the fit (`better_robot.optim.run`): each phase has its own iteration budget,
optimizer (Adam or L-BFGS), weight column, and a mask saying which degrees of freedom
may move.

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
- Each matched distance is **saturated** at `[chamfer].sat_m` (0.25 m, `s·tanh(d/s)`):
  near the body this is the identity, so nothing about the fit changes, but a far
  outlier — a depth tail smeared along the view ray, a patch that really belongs to
  the background — stops pulling instead of dragging the body with constant force
  forever. The reprojection terms and the prefit's chamfer were always robust this
  way; this closes the same gap for the kinematic chamfer.
- Visibility is recomputed every iteration (front-facing vertices only): the depth map
  sees only the front surface.
- Each frame's contribution is multiplied by its **depth-smear weight**
  (`docs/quality.md`): a frame whose human cloud is streaked along the view ray
  (VGGT's fast-motion failure) pulls the body far less, and the image terms hold
  that frame instead. The weight is exactly 1 on every clean frame measured, so
  this only acts where the depth actually broke.

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
- **These weights are tuned at 30 fps, and `kindyn_1` restores that meaning at any frame
  rate.** A k-th per-frame difference of a smooth motion shrinks like `1/fps^k`, so on a
  60 fps clip the same weight restrains the physical derivative `(60/30)^2k` times less —
  while the per-frame pose noise the terms exist to damp does not shrink at all. Measured
  consequence: on 60 fps scenes the kindyn physics term overpowered the smoothness terms
  and shook the force-bearing limbs. So in the `kindyn_1` solve every per-frame-difference
  weight (the motion terms, `radial_velocity`, `ann_velocity`, and the force/torque
  smoothness) is scaled by `(fps/30)^2k` at runtime (`fps_scaled_weights`), identity at
  ≤ 30 fps; kindyn's RNEA velocity/acceleration likewise use a stencil `round(fps/30)`
  frames wide — the acceleration estimate (and its noise) a 30 fps camera of the same
  motion would give. The pure kinematic stages deliberately keep their raw weights at
  every fps: scaling them too was tried and measured to stiffen `kinematics_2` against
  the contact-patch repair (the body seats less, and 60 fps contact recall collapsed).
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
the **contact frames**, and one nearest-neighbour pass per iteration serves both
scene terms.

- **`penetration`** — pushes the contact frames out of the wall. A 1 cm dead zone
  tolerates soft tissue and noise; a Huber ramp keeps one deeply-buried point from
  dominating; matches to far or low-confidence scene points are ignored. It acts on
  the contact frames rather than on mesh vertices because those points are where
  contact is *modelled*: a frame buried 8 cm inside a hold gives the detector a
  meaningless signed distance, and this term keeps the signals honest. Note what it
  does *not* do: the dead zone (1 cm) sits inside the contact band (3 cm), so
  penetration never decides *which* frames read as touching — the distance and
  stillness gates do. With a few dozen points instead of ~1000 the penalty (a mean over active
  points) is much more concentrated, so its weight is correspondingly larger.
- **`ann_attraction`** — the inverse: pulls the labelled points *onto* the wall
  (the labelled contact frames). Attraction owns the region just outside the surface,
  penetration the region behind it — they partition the signed-distance axis and never
  fight. This term exists because nothing else pulls the body toward the *scene*
  (chamfer targets the person's own depth, penetration only pushes).
- **`ann_velocity`** — damps the world velocity of the in-contact points — both how fast
  they *translate* and how fast they *rotate* (the angular part carries a smaller weight,
  `[ann_velocity].angular_weight`). A gripping hand should hold both its place and its
  facing. A point's world position/orientation differenced over time already *is* its
  LOCAL_WORLD_ALIGNED velocity, so no Jacobian is needed.

Where do the labels come from in `kinematics_1`? **Auto-contact**: at the start of the
refine phase, the contact readout (same machinery as `estimate_contacts`) runs once on
the current body — a joint that is close to the wall and still gets labeled. Close-and-
still is the *whole* rule here: the standalone readout also asks the force solve to
confirm the penetration exception (see `docs/contacts.md`), but no force has been
solved this early, so this stage keeps the plain two-gate detector. The labels
are then **frozen** for the rest of the fit, and only positive labels are ever created.
Detection is one-shot and gradient-free, so the fit cannot talk itself into more
contact. The stillness measured *before* the attraction acted (`drift_pre`) is saved
into the output, because the readout later must not be gamed by the very damping this
stage applied — that circularity was observed and is the reason for the rule.

The geometric detector is the only label source.

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
annotations (`../data/examples/data/<stem>.npz`, 10- or 14-joint schema) are
metrics-only (`datasets/examples/score_frame_contacts.py`).

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
honestly. One term acts on the instance's point:

- **`ann_acquisition`** — a pseudo-Huber pull on the distance to the patch: a
  bounded, roughly constant force from far away that fades out at contact. One
  point per label, and it is the contact frame itself — the labeled toe, heel,
  knee, palm or fingertip. A contact label says *that patch of skin* touches, so
  that is what gets pulled, not the whole hand.

A second term used to drive that point a few millimetres *inside* the wall
as an anti-burial guard. Measured on boulder_1/3 (2026-07-26)
it was itself the main source of penetration (it applied to every instance,
wrong labels included); removing it cut mean penetration 2.03 → 0.51 and
0.93 → 0.26 mm with the true contacts at the same wall distance. Acquisition's
fading pull plus the ordinary penetration term seat the limb on their own, so
the term is gone.

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

## kindyn_1 — joint kinematics + dynamics

**Why it exists.** The chain above is one-way: `kinematics_1/2` fit the motion
using only what the camera sees; a frozen-pose force solve (like the contact
readout's force witness) can then only absorb any pose error into a large
leftover base wrench — the physics never gets a say in the motion. `kindyn_1`
closes the loop: it optimizes the pose **and** the contact forces together, so
the physics residual can push back on the motion.

**What it optimizes.** It starts from the converged `kinematics_2` body and its
contact labels — by default `contacts_2.npz`, the readout measured on that very
body (`bodies = "kinematic_1"` starts from `kinematic_1` + `contacts_1` for
chains that skip `kinematics_2`). The objective is the full `kinematics_2` objective —
reprojection, priors, motion smoothness, scene penetration, the annotation
terms — **plus** the physics terms (`base_wrench` + `base_wrench_ang`,
`force_magnitude`, the force smoothness family at three difference orders
(`force_smooth_*`, `force_accel_*`, `force_jerk_*`, each split into a magnitude
and a direction part), and the joint-torque family `torque_magnitude`,
`torque_smooth`, `torque_accel`), with one world-force vector per
**contact frame** per video frame added as extra optimization variables, masked
to the steps the labels say that frame touches. Chamfer stays off, for the same
reason as in `kinematics_2`: kindyn starts from the repaired body, and chamfer
would pull it back toward the spaghetti-prone depth the repair just left.

**The body's mass.** Nothing is measured or assumed about the person's weight.
The body is the SMPL-X mesh at the person's fitted shape, cut into its 52 skinning
parts, and each part's mass, centre of mass and moment of inertia are integrated
from that closed part mesh (BetterHuman bakes them when the shape is set). Each
part is weighed at its own tissue density from Dempster's segment table
(`body_densities.json` in BetterHuman: 920 kg/m³ for the chest, 1010 for the
pelvis and feet, 1110 for the head, 1160 for hands and fingers, and so on), so
the RNEA model has the right mass distribution and not just the right total.
`total_mass` in `kindyn_1.npz` is the sum. Other tables (Clauser, Chandler) or a
single uniform density are one word away in `bvr/body.py` (`load_body`).

**Where the forces are applied.** At the contact frames, not at joints — the
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
construction the force witness uses (`bvr/human_optim/forces.py`): the force
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

**How it runs.** Two steps. First, a quick frozen-pose force solve establishes
physically reasonable forces — with zero forces the body is in freefall, and the base
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
`kinematics_2`, so the visual fit is asked for exactly as before. The
dynamics weights are **linear** — the term is `0.5 · mean over valid frames of
Σ residual²` and the weight multiplies it, like every other engine term.

Every physics residual is first divided by the **body weight** and averaged over
the counted frames. Only frames whose whole velocity/acceleration stencil is
tracked count: an occlusion gap holds the nearest tracked pose, and differencing
across the held jump would read as a huge phantom acceleration that the solve
would spend its budget bending the pose to explain. Without that the terms are a *sum* of newtons squared, so the
same weight pulls hard on a 900-frame clip and barely registers on a 60-frame
one, and harder on a heavy climber than a light one — while every kinematic term
stays a per-frame mean. In body-weight units all terms are O(1) and one number
means one thing everywhere. (The force witness solves in the same units for the
same reason.)

The root residual is weighted in two halves: `base_wrench` for its linear rows
(a leftover net force, in newtons) and `base_wrench_ang` for its angular rows
(a leftover net moment, in newton-metres). They are split because the friction
cones block moment authority — a force stuck inside its cone often cannot cancel
a moment — so at equal weights the angular rows were the residual that actually
stayed large. The shipped 50/300 tilt deliberately accepts more linear residual
to push the angular residual down: on the 13 `./data` clips it buys the angular
RMS 0.47 → 0.22 (body-weight units on a 0.2 m lever) for a linear rise of
0.11 → 0.17 BW.

Force smoothness runs at three difference orders, each split into a magnitude
part and a direction part about the local mean force direction. The first
difference damps frame-to-frame change; the second and third (the force's
acceleration and jerk) damp the alternating flicker a first difference cannot
remove without also flattening real load changes. Every order is masked to
contacts that persist over its whole window, so a 1-frame contact is untouchable
by construction, and the weights scale with `(fps/30)^2k` like every difference
term. The 2026-08-30 sweep behind the shipped values is in the config comments;
the headline is first/second/third-order jitter down 39 / 70 / 83 % with
reprojection flat and load peaks intact.

**Joint torques, weighted by strength.** The physics only pins the *sum* of the
contact forces; how four contacts share a load is statically indeterminate, and
with only `force_magnitude` to break the tie the solve chose the split with the
least internal squeeze — which on the demonstrator wall put too much on the hands
and starved the feet. Three terms now read the joint torques RNEA computes
(everything past the six root rows, in body-weight units): `torque_magnitude`,
the effort cost, plus `torque_smooth` and `torque_accel`, its first and second
differences over the counted frames. Each DOF's torque is multiplied by its
joint's entry in `[torque.multipliers]`, keyed by joint group (the joint name
without side and index), so a torque costs more the weaker the joint. The shipped
table is 200 N·m divided by each joint's published maximum voluntary torque —
legs and trunk 1, shoulder 2.5, elbow 3.3, wrist 17, fingers 25 — so a load the
body could carry with either an arm or a leg lands on the leg. Set every entry to
1 to get the plain torque terms back.

## What is deliberately absent

- No per-body scale variable (the world is sized to the body upstream).
- No per-frame betas in the kinematic fit (one person, one shape); the scale
  stage's pre-fit likewise freezes shape to one canonical betas per person.
- No hand-tuned per-clip gates in the loss: a clip whose world is grossly wrong
  (e.g. the person's depth off the wall — a "floater") is an upstream problem,
  not something extra terms here should paper over.
