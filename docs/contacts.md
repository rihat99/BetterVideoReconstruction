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

## The three gates

For every person, every video frame and every contact frame, three things must
agree before we call it a touch.

**1. Surface.** Find the nearest point of the fused scene cloud. That point has
an outward normal (estimated by local PCA at fuse time, oriented toward the
camera). Projecting the offset from the scene point to the contact frame onto
that normal gives a *signed distance*: positive in front of the wall, negative
behind it. It is the same signed distance the kinematic penetration loss uses, so
"at the wall" means one thing everywhere. Two guards reject matches that are not
evidence: the nearest point must be within 10 cm (a match further away is behind
an occluder, not a surface), and it must be a confident point (a hole in the
cloud must read as *unknown*, never as *no wall*).

**2. The normal window.** The contact frame's own outward normal must face the
scene point it matched. Concretely, `dot(frame normal, −scene normal)` has to be
at least `cos(normal_max_angle_deg)`; at the default 90° a frame may touch
anything in the hemisphere it looks into, and nothing behind it.

This one gate is why `toe_under` and `toe_above` exist. They sit millimetres
apart on the same toe, so any distance test alone would fire both on the same
hold. Their normals point opposite ways, so the window picks exactly one:
standing on a foothold lights `toe_under`; hooking the toe *under* a hold — the
foot pulling up rather than pressing down — lights `toe_above`, because that is
the surface actually pressed against the hold. Same for a heel hook versus
standing on the heel.

**3. Stillness.** Touching means holding. A contact frame's *net drift* over a
centred ~0.35 s window must stay small — under 3 cm by default. Net drift, not
speed: a gripping hand jitters by millimetres that cancel out over the window,
while a hand sweeping slowly past the wall accumulates several centimetres. The
drift used is `drift_pre`, saved by the kinematic stage: the motion measured
*before* any contact attraction or damping acted on the body. That matters,
because the optimizer's own damping holds labelled contacts still — stillness
re-measured afterwards would be evidence we manufactured ourselves.

The gates are applied by pushing the signed distance to `+inf` wherever any of
them fails. A blocked step then simply looks infinitely far from the wall, so
everything downstream — hysteresis, minimum duration, span merging — works
without knowing the gates exist.

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

The two use different operating points on purpose — `[auto_contact]` for the
detector inside the fit, `[contacts]` + `[contact_gate]` for the readout — so the
readout's recall knob can move without silently changing the bodies that were
already fitted. The *code* is shared, so the two can never drift apart.

## Knobs

All in `configs/human_optimization.toml`.

`[contacts]` — the readout: `close_thresh_m` / `exit_thresh_m` (hysteresis),
`min_frames`, `max_dist_m` and `conf_thresh` (the surface guards), `conf_drop_q`
and `normal_k` (scene-cloud loading), `normal_max_angle_deg` (the normal window),
`prune_margin_m`, and the three `merge_*` knobs.

`[contact_gate]` — `window_s` and `still_m`, the stillness gate. `still_m` is the
recall knob: raise it and more marginal contacts survive.

`[auto_contact]` — the same operating point for the detector inside
`kinematics_1`, with its own looser `detect_thresh_m` / `detect_exit_m` and its
own `still_m`. Omit `still_m`, `window_s` or `normal_max_angle_deg` there and it
inherits the readout's value.

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
