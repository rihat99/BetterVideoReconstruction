# The contact stage: which points of the body touch the scene, and when

Contact does not live on joints. It lives on **contact frames**: named points on
the skin, each of which knows which way it faces.

A *contact frame* is one mesh vertex turned into a rigid marker on the body. Its
position is that vertex; its orientation is baked so that its local z axis is the
outward surface normal there — the direction the skin looks. Both are attached to
a parent joint once, at body build time, so posing the body reports each contact
point *and* its outward direction straight out of forward kinematics. No mesh
work per frame, no skinning, no lookup.

The current set is **33 frames covering the whole body** — grown from the
original 12-frame climbing set (toes, heels, knees, palms, fingertips) so other
sports and ground locomotion have the points they load. The foot alone carries
five: heel, the two ball-of-foot points (inner and outer edge of the forefoot,
so a flat foot can hold a roll torque), the pad of the big toe, and the top of
the toe (toe hooks). The frames, their parent joints and the mesh vertex each
sits on:

| frame | parent joint | vertex | what it is |
|---|---|---|---|
| `heel_left` / `_right` | `left_ankle` / `right_ankle` | 8924 / 8712 | the back of the heel |
| `ball_inner_left` / `_right` | `left_foot` / `right_foot` | 5906 / 8600 | ball of the foot, big-toe side |
| `ball_outer_left` / `_right` | `left_foot` / `right_foot` | 5916 / 8610 | ball of the foot, little-toe side |
| `big_toe_left` / `_right` | `left_foot` / `right_foot` | 5847 / 8541 | the pad under the big toe |
| `toe_above_left` / `_right` | `left_foot` / `right_foot` | 5895 / 8589 | the **top** of the toe |
| `knee_left` / `_right` | `left_knee` / `right_knee` | 3677 / 6438 | just below the kneecap |
| `palm_left` / `_right` | `left_wrist` / `right_wrist` | 4704 / 7440 | the heel of the palm |
| `finger_left` / `_right` | `left_middle3` / `right_middle3` | 5077 / 7789 | the middle fingertip |
| `elbow_left` / `_right` | `left_elbow` / `right_elbow` | 4374 / 7110 | the tip of the elbow |
| `forearm_left` / `_right` | `left_elbow` / `right_elbow` | 4364 / 7100 | the underside of the mid-forearm |
| `shoulder_left` / `_right` | `left_shoulder` / `right_shoulder` | 4078 / 6822 | the side of the shoulder (deltoid) |
| `back_left` / `back_right` | `left_collar` / `right_collar` | 5566 / 8278 | the shoulder blade |
| `sit_left` / `_right` | `pelvis` | 3463 / 6224 | the sit bones |
| `hip_side_left` / `_right` | `left_hip` / `right_hip` | 3958 / 6706 | the side of the hip |
| `pelvis_front_left` / `_right` | `pelvis` | 4420 / 7156 | the front of the pelvis |
| `sacrum` | `pelvis` | 5613 | the base of the spine |
| `chest` | `spine2` | 3855 | the lower breastbone |
| `head` | `head` | 9003 | the crown of the head |

The set is authored in
`BetterHuman/studio_data/frames/smplx_neutral/better_contacts.json` and baked on
by `tools/body.py` (the legacy 12-frame set lives next to it in
`contacts.json`). Because the fingertip frames hang off finger joints, those
joints must exist in the optimized body — but their pose is **frozen**: at depth
and keypoint resolution there is nothing to fit a finger with, so the fingers
simply follow the wrist and carry their frame along.

A recipe can restrict the set: `[contacts].frames` in the human-optimization TOML
is an allowlist of frame names (empty = all), applied where the body is built —
so detection, labels and forces all see the smaller set, with no per-stage gates.
`configs/human_optimization_peter.toml` uses it to keep hands + feet only:
the peter rigs instrument one board per limb, and contact detected on a
knee or elbow would route force where nothing measures it.

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

**When closeness outranks stillness.** The stillness rule assumes quasi-static
support — a climber holds every contact for a while. Fast sports break it: a
sprinting foot plant lasts ~0.1 s, and the ~0.35 s drift window inevitably
straddles the swing before and after, so the plant can *never* look still — no
`still_m` setting fixes that. But the plant presses into the floor, and that is
evidence stillness should not be allowed to veto. `[contact_gate].still_override_m`
does exactly that: a frame whose guarded signed distance is below it (penetrating,
or within a centimetre) stays contact evidence even while moving fast. Measured on
high_jump: 354 penetrating foot steps, 100 % of them killed by the stillness gate
before the override existed. Short plants also need short spans to survive, so
`[contacts].min_span_s` is 0.06 s. Both started as athletics-only knobs on the
theory that climbing's quasi-static support wanted the stricter gate as a
precision guard; the 13-scene climbing A/B (2026-08-25) refuted that — pooled F1
went .729 → .768, and sport_climb, whose fast rope-climbing steps the stillness
gate had been killing exactly like sprint plants, went .27 → .62. Since then one
value (`0.01` / `0.06 s`) serves every sport; `-1` turns the override off.

The same evidence outranks **duration** too. The detector drops spans shorter
than 3 frames as flicker, and the stage repeats an fps-aware version of that
rule after gap-welding — but a one-frame bounce at 25 fps *is* one frame, and
no duration rule can tell it from jitter by length alone. Penetration can: a
span with at least one step pressing below `still_override_m` is exempt from
every duration rule (the detector's blip killer, the post-weld `min_span`, and
the toe rule's cleanup). Measured before the exception: 1FGkg had 117 one-frame
sub-3 cm foot dips, 79 of them ending up with no contact at all. With the
override off (`-1`) the exemption never fires.

## Why the frame's own normal is not a gate

Every contact frame knows which way its skin looks, and we used to require that it
look at the scene point it matched (a `normal_max_angle_deg` window). That gate is
gone. Both of its inputs are too noisy to veto on: the scene normals come from a
reconstructed point cloud, and the frame's normal comes from the fitted pose. It
was throwing away real contacts — most visibly fingertips, which reach a hold
while the estimated hand pose points the fingerpad the wrong way. Being near the
wall and still is the evidence we actually trust.

The normal survives in exactly one place, with no knob attached.

## Telling the two foot sides apart

A foot's contact frames come in two sides facing opposite ways through the
foot: the underside (`heel` / `ball_inner` / `ball_outer` / `big_toe` —
`toe_under` in the legacy set) and `toe_above`, the top of the toe. Every gate
above accepts or rejects them together, so both sides fire on the same hold —
and then the body appears to be standing on a foothold and hooking under it at
the same time. Their normals point opposite ways, so inside this detector the
side whose best candidate looks more into the wall keeps the step and the other
side's candidates are dropped before hysteresis; ties keep the underside. That
is the only thing a frame normal decides here, and it has no knob.

That tiebreak is all the `kinematics_1` auto-labels have to go on, since no force
has been solved yet when they are detected. In the standalone readout it is
**superseded**: the force witness decides the pair for a whole contact at a time,
and its verdict is the last word — see
[`force_contacts.md`](force_contacts.md#step-8-the-toe-rule--which-side-did-the-load-go-through).
Measured on the 13 annotated scenes, the readout's final labels come out
bit-identical whether this per-step tiebreak runs or not.

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

**One debounce over the union.** After the geometric and force labels are
combined, gaps shorter than the span rule are filled. Each readout already drops
its own too-short spans, but the union puts two of them end to end and can leave
a one- or two-frame hole that neither arm ever saw. The rule is symmetric and
uses the same number: a contact lasts at least `min_span_s`, and so does a
release. The knob is in **seconds** (`min_span_s`, default 0.1 s; frames =
ceil(s·fps)) — a frame count would let a 60 fps clip keep contacts half as long
as a 25 fps clip keeps them, which is exactly the label flicker we measured on
the 60 fps scenes.

## The force witness: contacts the distance gate cannot see

Everything above needs the contact frame to be **near** the scene. That works when
the reconstructed depth is good. When it is not — the body floats a hand's width
off the wall, or a limb came out too short — the frame is never near anything, the
gate never fires, and a real contact is simply lost.

So the stage asks physics a second question: with the pose frozen, how much force
is unexplained, and which points on the body could have supplied it? A frame that
has to carry real load is in contact, whatever the distance says. That label is
**unioned** onto the geometric one — force only ever adds — with one exception,
the toe pair, where the force decides which side of the toe is touching.

On the 13 annotated scenes it takes pooled F1 from **.488 to .751**, almost
entirely by recall, and rescues the three known floater clips from .02 / .09 / .26
to .87 / .85 / .82.

**[`force_contacts.md`](force_contacts.md) is the full explanation** — the
residual, the friction cones, the two prices, the no-evidence policy, the label
rules and the toe rule, with the measured evidence and its caveats.

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
`min_span_s` (the span rule, seconds), `max_dist_m` and `conf_thresh` (the
surface guards), `conf_drop_q` and `normal_k` (scene-cloud loading),
`prune_margin_m`, and the three `merge_*` knobs. `min_span_s` also sets the
union debounce, in both directions.

`[contact_force]` — the force witness: `theta_bw` / `theta_hold_bw` (the load
thresholds a label enters and holds at), `gap_dstar_m` and `stillness_vstar_mps`
(the two prices). Four knobs; everything else about the solve is fixed in code.
See [`force_contacts.md`](force_contacts.md#the-knobs).

`[contact_gate]` — `window_s` and `still_m`, the stillness gate. `still_m` is the
recall knob: raise it and more marginal contacts survive. It is the *only* knob
that moves both the kin1 detector and the readout at once.

`[auto_contact]` — the detector inside `kinematics_1`: its own looser
`detect_thresh_m` / `detect_exit_m`, plus the attraction `band_m`. It does not set
`still_m` or `window_s`, so it inherits `[contact_gate]`.

## Output

`human_optim/contacts_<tag>.npz`, `contact_label_schema = 2`,
`contact_label_method = "frames-v2"` (`v1` = geometry only; `v2` adds the force
witness).

| field | shape | meaning |
|---|---|---|
| `frame_contact` | `(P, N, F)` bool | **the label** |
| `merged` | `(P, N, F)` bool | frames the span-merge bridged |
| `frame_sdf_cm` | `(P, N, F)` f32 | signed distance to the wall, cm (NaN = no scene point) |
| `frame_distance_cm` | `(P, N, F)` f32 | Euclidean distance to the nearest scene point, cm |
| `frame_normal_dot` | `(P, N, F)` f32 | how well the two normals face, in [−1, 1] |
| `frame_drift_m` | `(P, N, F)` f32 | the net window drift the stillness gate read |
| `frame_force` | `(P, N, F, 3)` f32 | the solved world contact force, newtons |
| `frame_force_bw` | `(P, N, F)` f32 | its magnitude as a fraction of body weight |
| `frame_force_label` | `(P, N, F)` bool | the force decision, before the union |
| `frame_wall_dot` | `(P, N, F)` f32 | cone direction vs the wall's normal, at the force search radius |
| `body_weight_n` | `(P,)` f32 | baked body mass × g |
| `residual_force_before_n` / `_after_n` | `(P, N)` f32 | unexplained root force, before and after the solve |
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
wiring. `completion.py` and `fusion.py` are deleted;
`tools/human_optim/{confidence,depth_fit}.py` survive only for the frozen
per-joint scorer (`scripts/score_contacts.py`), nothing in the contact path
calls them, and `docs/contact_confidence.md` /
`docs/contact_fusion.md` / `docs/depth_fit.md` describe the retired per-joint
era. Consumers pinned to a schema (the offline scorer, `kinematics_2`, `kindyn_1`)
fail loudly on a mismatched `contact_label_schema` rather than misreading it.

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
