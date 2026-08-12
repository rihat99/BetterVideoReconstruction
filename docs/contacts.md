# The contact stage: which points of the body touch the scene, and when

Contact does not live on joints. It lives on **contact frames**: twelve named
points on the skin, each of which knows which way it faces.

A *contact frame* is one mesh vertex turned into a rigid marker on the body. Its
position is that vertex; its orientation is baked so that its local z axis is the
outward surface normal there — the direction the skin looks. Both are attached to
a parent joint once, at body build time, so posing the body reports each contact
point *and* its outward direction straight out of forward kinematics. No mesh
work per frame, no skinning, no lookup.

The twelve frames, their parent joints and the mesh vertex each sits on:

| frame | parent joint | vertex | what it is |
|---|---|---|---|
| `toe_under_left` / `_right` | `left_foot` / `right_foot` | 5855 / 8549 | the **underside** of the toe |
| `toe_above_left` / `_right` | `left_foot` / `right_foot` | 5895 / 8589 | the **top** of the same toe |
| `heel_left` / `_right` | `left_ankle` / `right_ankle` | 8924 / 8712 | the back of the heel |
| `knee_left` / `_right` | `left_knee` / `right_knee` | 3673 / 6434 | the front of the knee |
| `palm_left` / `_right` | `left_wrist` / `right_wrist` | 4704 / 7440 | the middle of the palm |
| `finger_left` / `_right` | `left_middle3` / `right_middle3` | 5077 / 7789 | the middle fingertip |

The set is authored in
`BetterHuman/studio_data/frames/smplx_neutral/contacts.json` and baked on by
`tools/body.py`. Because the fingertip frames hang off finger joints, those
joints must exist in the optimized body — but their pose is **frozen**: at depth
and keypoint resolution there is nothing to fit a finger with, so the fingers
simply follow the wrist and carry their frame along.

## The two gates

For every person, every video frame and every contact frame, two things must
agree before we call it a touch: the point is **at the wall**, and it is **still**.

**1. Surface.** Find the nearest point of the fused scene cloud. That point has
an outward normal (estimated by local PCA at fuse time, oriented toward the
camera). Projecting the offset from the scene point to the contact frame onto
that normal gives a *signed distance*: positive in front of the wall, negative
behind it. It is the same signed distance the kinematic penetration loss uses, so
"at the wall" means one thing everywhere. Two guards reject matches that are not
evidence: the nearest point must be within 10 cm (a match further away is behind
an occluder, not a surface), and it must be a confident point (a hole in the
cloud must read as *unknown*, never as *no wall*).

**2. Stillness.** Touching means holding. A contact frame's *net drift* over a
centred ~0.35 s window must stay small — under `still_m` by default. Net drift, not
speed: a gripping hand jitters by millimetres that cancel out over the window,
while a hand sweeping slowly past the wall accumulates several centimetres. The
drift used is `drift_pre`, saved by the kinematic stage: the motion measured
*before* any contact attraction or damping acted on the body. That matters,
because the optimizer's own damping holds labelled contacts still — stillness
re-measured afterwards would be evidence we manufactured ourselves.

The gates are applied by pushing the signed distance to `+inf` wherever either of
them fails. A blocked step then simply looks infinitely far from the wall, so
everything downstream — hysteresis, minimum duration, span merging — works
without knowing the gates exist.

## Why the frame's own normal is not a gate

Every contact frame knows which way its skin looks, and we used to require that it
look at the scene point it matched (a `normal_max_angle_deg` window). That gate is
gone. Both of its inputs are too noisy to veto on: the scene normals come from a
reconstructed point cloud, and the frame's normal comes from the fitted pose. It
was throwing away real contacts — most visibly fingertips, which reach a hold
while the estimated hand pose points the fingerpad the wrong way. Being near the
wall and still is the evidence we actually trust.

The normal survives in exactly one place, with no knob attached.

## Telling the two toe frames apart

`toe_under` and `toe_above` sit millimetres apart on the same toe. Every gate
above accepts or rejects them together, so both fire on the same hold — and then
the body appears to be standing on a foothold and hooking under it at the same
time. Only their normals can separate them, because those point opposite ways.

So: assume at most one of a pair is really in contact at any step. (Both at once
is physically possible and deliberately ignored — it is rare, and pretending
otherwise costs more than it buys.) At a step where **both** toes of one foot are
near enough to hold a contact, keep the one whose normal looks more into the wall
and drop the other. Exact ties keep `toe_under`, so the answer never depends on
evaluation order. The drop happens before hysteresis, so the loser can neither
start nor continue a span there — the output can never show both toes of a foot
touching at the same step. Standing on a foothold lights `toe_under`; hooking the
toe *under* a hold — the foot pulling up rather than pressing down — lights
`toe_above`.

The count of steps where the tiebreak had to choose is printed per frame by
`estimate_contacts` (`toe-tiebreak suppressed=`).

## From a signal to a label

**Hysteresis.** A contact starts when the signed distance drops below 3 cm and
survives while it stays below 3.5 cm. The looser exit threshold is what stops
labels blinking on and off at the boundary.

**Minimum duration.** Spans shorter than 3 frames are dropped as flicker.

**Bridging short gaps.** Two contacts of the same frame less than 2 s apart are
merged into one when the point barely moved across the gap (< 3 cm) *and* stayed
near the wall throughout (within 8 cm signed distance). That last condition is
the re-grip guard: a hand that left the wall and came back is two contacts, not
one. Bridged frames are flagged in the output (`merged`) so a consumer can ignore
them.

## The force witness: contacts the distance gate cannot see

Everything above needs the contact frame to be **near** the scene. That works when
the reconstructed depth is good. When it is not — the body floats a hand's width
off the wall, or a limb came out too short — the frame is never near anything, the
gate never fires, and a real contact is simply lost.

So the stage asks physics a second question. Freeze the pose and run inverse
dynamics with nothing touching the body. The answer says what force the body would
need to move the way it does while falling freely. Six of those numbers belong to
the **root** (the pelvis), which has no muscle behind it: whatever is left there
had to come in through a contact. That leftover is the **residual** — with no
contacts at all it is simply the body's weight, about 600 N, entirely unexplained.

The stage then solves for the world forces at the 12 contact frames that explain
the residual. Two things keep the answer meaningful.

**A friction cone on every frame.** A cone is the set of pushes a surface can
actually deliver: straight out along the surface, or up to 45° off it, never a
pull. Each frame's cone points along the direction that patch of skin faces
(negated — the wall pushes back the way the skin looks). Without cones a single
free hand can absorb almost any residual and the answer says nothing about *which*
limb touched; that is exactly how an earlier attempt failed. With cones, a frame
can only explain the part of the residual it is physically oriented to explain.
It also settles the toe pair for free: `toe_under` and `toe_above` face opposite
ways, so their cones are opposite and the solve picks one.

**Two prices.** Force on a frame that is drifting is expensive (a sliding limb
carries little load), and force far from the scene is expensive. The gap price is
measured **relative to the nearest contact frame of that step**, not as an
absolute distance. That is the whole floater rescue: on a body where every frame
is far from the wall, absolute pricing would make all force unaffordable and the
solve would leave the body's weight unexplained — on exactly the bodies this
exists to rescue. Relative pricing still prefers the nearer frame, but the weight
has to land somewhere.

**No scene data is not "far away".** When no scene point is anywhere near a step's
frames, that is missing evidence, and the frame is simply not allowed to carry
force. Confusing the two once cost a whole ablation its meaning.

A frame that has to carry more than `theta_bw` of body weight (default 5%) for at
least 3 frames is called in contact. Knee frames are excluded — measured, their
force labels are almost all false. The result is **unioned** onto the geometric
label: force only ever adds.

**With one exception, and it is the toe pair.** Each readout keeps at most one toe
per foot, but they can keep *different* ones, and the union would then hold both —
which the physics forbids. So after the union, wherever both toes of a foot are
labelled, the force decides: the toe carrying the larger solved magnitude survives
(ties keep `toe_under`), even if that drops a geometric label. Exclusivity outranks
add-only, because "at most one toe per foot" is a statement about the world while
add-only is only a policy about how two readouts combine. Dropping a step can split
a surviving span, so the four toe columns get a final minimum-duration pass. Before
this rule the union held both toes on 838 steps across 12 of the 13 scenes; now it
is 0 everywhere.

The solve is a convex problem — a quadratic cost over nonnegative cone
coefficients — so projected gradient descent reaches the one true optimum, and the
stage reports a KKT number that is near zero when it got there. It takes a few
seconds per scene.

Honesty rule, same as everywhere else here: the stillness used to price force is
the **persisted pre-enforcement drift** (`drift_pre`). A body whose `drift_pre` is
missing is a hard error, never a silent re-measurement — otherwise the optimizer's
own contact damping would be voting for its own labels.

What it is worth, on the 13 annotated scenes (see the round report for the full
tables): pooled F1 rises from **.488 to .721** at the default threshold — precision
.869 → .858, recall .340 → .623, so the gain is almost entirely recall at nearly
unchanged precision. Every scene improves, and the three known floaters go from
.02 / .09 / .26 to **.81 / .79 / .80**.

Three caveats travel with those numbers. On eleven never-tuned held-out scenes the
same frozen recipe gains much less — pooled .789 → .837, per-source macro
.725 → .775, with 2 of 5 source videos slightly worse — because those scenes have
good geometry and the distance gate already works there. The force witness pays
where the reconstruction is bad, and there it pays a lot: the one held-out scene
with poor geometry went .519 → .801. A geometry-only control that thresholds the
same *relative* distance recovers much of one floater on its own, so the physics is
not the whole story. And `sport_climb` has a climber on a rope, a force this model
does not know about, so its numbers are reported separately.

## One detector, two callers

`tools/human_optim/contacts.py::detect_frame_contacts` is the only place these
gates are implemented, and both users call it:

- **`kinematics_1`**, inside the fit. Labels are detected **once**, from the
  starting pose at the first scene phase, and then **frozen** for the rest of the
  optimization. They are positive-only (`+1` on a detected frame, `−1`
  everywhere else, never `0`), so they can drive the contact attraction and the
  velocity damper but can never drive the clearance repulsion. Detecting once and
  freezing is what keeps contact detection from feeding back into the fit. The
  labels land in `kinematic_1.npz` as `auto_labels`, and the drift they were
  gated on as `drift_pre`.
- **`scripts/stages/estimate_contacts.py`**, the standalone readout. It poses a
  finished body, runs the same detector, merges spans and writes
  `contacts_<tag>.npz`. `--bodies kinematic_1` → `contacts_1.npz`.

The two use different *distance* thresholds on purpose — `[auto_contact]` is
looser, because inside the fit a label is a target to pull toward, not a claim.
Stillness is **one** notion shared by both. The *code* is shared too, so the two
can never drift apart.

## Knobs

All in `configs/human_optimization.toml`.

`[contacts]` — the readout: `close_thresh_m` / `exit_thresh_m` (hysteresis),
`min_frames`, `max_dist_m` and `conf_thresh` (the surface guards), `conf_drop_q`
and `normal_k` (scene-cloud loading), `prune_margin_m`, and the three `merge_*`
knobs.

`[contact_gate]` — `window_s` and `still_m`, the stillness gate. `still_m` is the
recall knob: raise it and more marginal contacts survive. It is the *only* knob
that moves both the kin1 detector and the readout at once.

`[auto_contact]` — the detector inside `kinematics_1`: its own looser
`detect_thresh_m` / `detect_exit_m`, plus the attraction `band_m`. It does not set
`still_m` or `window_s`, so it inherits `[contact_gate]`.

## Output

`human_optim/contacts_<tag>.npz`, `contact_label_schema = 2`,
`contact_label_method = "frames-v1"`.

| field | shape | meaning |
|---|---|---|
| `frame_contact` | `(P, N, 12)` bool | **the label** |
| `merged` | `(P, N, 12)` bool | frames the span-merge bridged |
| `frame_sdf_cm` | `(P, N, 12)` f32 | signed distance to the wall, cm (NaN = no scene point) |
| `frame_distance_cm` | `(P, N, 12)` f32 | Euclidean distance to the nearest scene point, cm |
| `frame_normal_dot` | `(P, N, 12)` f32 | how well the two normals face, in [−1, 1] |
| `frame_drift_m` | `(P, N, 12)` f32 | the net window drift the stillness gate read |
| `contact_frame_names` | `(12,)` | frame names, in label order |
| `frame_parent_joint` | `(12,)` i64 | parent joint of each frame (52-joint index) |
| `frame_vertex` | `(12,)` i64 | mesh vertex each frame sits on |
| `joint_names`, `parents` | `(52,)` | the body's joint table, for consumers |
| `object_ids`, `valid_mask` | `(P,)`, `(P, N)` | who, and where they were tracked |

Plus every knob echoed as a scalar, and `source_body` / `source_scene` /
`source_transform` with their sha256, so a stale artifact is detectable.

The overlay `contacts_<tag>_overlay.mp4` draws the 12 projected points, red in
contact, green free.

Cost, measured on `boulder_1` (445 frames, one person, one GPU): 12 s end to end
including the overlay. The frame poses come free with forward kinematics and the
nearest-neighbour search is over 12 points per frame instead of 10 475 vertices,
so the readout is essentially free — the overlay is most of the wall time.

## What this replaced, and why

The previous model labelled all 52 **joints**. Every body vertex was
signed-distanced against the scene, contacts were lifted to whichever joint
centre the vertex sat nearest, and three further layers tried to repair what that
lost: a *completion* pass (looser gates plus a static support-force solve), a
*fusion* step folding in an external image model's per-limb probabilities, and a
per-label *confidence* weight. Detection was the whole of `estimate_contacts`;
those layers existed because a joint centre is not a contact point.

The frame model removes the reason they were needed:

- A joint centre sits *inside* the body, several centimetres from any surface, so
  its distance to the wall was never the contact's distance. A contact frame is
  on the skin.
- A joint has no orientation you can ask "which way does this touch?", so a toe
  hook and standing on a foothold were the same measurement. A contact frame
  carries its normal, and the window separates them.
- Lifting per-vertex contacts to joints smeared them: a whole hand became "wrist",
  and a heel on a hold landed on the ankle or the foot depending on rest-pose
  geometry. Twelve named points need no lifting at all.

Dropped with the old model, on this branch: completion, prediction fusion,
per-label confidence, the ray-cast occlusion witness, and the depth-fit
wiring. `tools/human_optim/{completion,fusion,confidence,depth_fit}.py` still
exist for the stages that have not been reworked yet, but nothing in the contact
path calls them, and `docs/contact_confidence.md` /
`docs/contact_fusion.md` / `docs/depth_fit.md` describe the retired per-joint
era. Consumers pinned to the old schema (the dataset export, the offline scorer,
`kinematics_2`, `dynamics`) fail loudly on `contact_label_schema = 2` rather than
misreading it.

### Rules that keep it honest

- The stillness gate reads a drift measured **before** any contact enforcement.
  An enforced body is still by construction; measuring it would prove nothing.
- Labels inside the fit are detected once and frozen. A detector that re-ran
  every iteration would be optimizing its own evidence.
- The readout stage never touches the bodies, so its knobs can be retuned freely.

### Known open issue

For clips longer than VGGT's frame cap the frame axis is subsampled while `fps`
stays native, which stretches every temporal window in this stage. Currently
unreachable — all corpus scenes are under the cap.
