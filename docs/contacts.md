# The contact stage: which joints touch the scene, and when

`scripts/stages/estimate_contacts.py` — a one-shot readout, off the default
chain. It takes a finished body trajectory and answers one question for every
joint on every frame: **is this joint touching the scene right now?** For every
person it writes a yes/no label per joint per frame, a confidence weight saying
how much to trust each label, and a record of where each label came from.

The `--bodies` flag picks which body to read, and that names the output:

- `--bodies kinematic_1` → `contacts_1.npz` — contacts measured on the first
  refined body. These labels drive `kinematics_2`.
- `--bodies kinematic_2` → `contacts_2.npz` — the same readout on the repaired
  body. These labels drive `kindyn_1`.

Same code both times; only the input body changes.

A label is built in three layers. Each one catches what the previous missed:

1. A precise **detector** finds joints that are clearly on the wall and holding
   still. It rarely lies, but it misses about 30 % of true contacts.
2. A **completion** pass adds the contacts the detector was too strict for. It
   only ever adds labels — it never removes one.
3. A **fusion** step folds in an external image model's per-limb predictions,
   which can see contacts where our 3D scene has no geometry at all. Own doc:
   [contact_fusion.md](contact_fusion.md).

`joint_contact` is the final label; `contact_source` records, per joint and
frame, which layer produced it. Every label also gets a confidence weight —
that half has its own doc: [contact_confidence.md](contact_confidence.md).

## Layer 1: the detector

The detector demands two things before calling a joint a contact: it must be
**at the wall**, and it must be **holding still**.

1. **Distance to the wall.** For every vertex of the posed body mesh, find the
   nearest point of the fused scene cloud and that point's surface normal (the
   direction facing out of the wall, estimated by local PCA and oriented toward
   the camera). Projecting the vertex-to-point offset onto the normal gives a
   *signed distance*: positive in front of the wall, negative behind it. This
   is the same signed distance the kinematic penetration loss uses — one
   definition of "at the wall" everywhere. Matches to far-away or
   low-confidence scene points are discarded: a hole in the cloud must read as
   "no evidence", never as "no wall".
2. **Vertices → joints.** Each vertex belongs to the joint *center* it sits
   nearest in the rest pose, and a joint's distance is the minimum over its own
   vertices. Skinning weights would be the wrong assignment here: fingertip
   vertices are partly skinned to joints further up the arm, which would drag
   the contact up the limb.
3. **The stillness gate.** Touching means holding: the joint's net drift over a
   centered ~0.35 s window must stay under 3 cm. This separates a gripping hand
   (millimetre jitter, near-zero net drift) from a hand slowly sweeping past
   the wall. The drift is `drift_pre`, saved by the kinematic stage — the
   motion measured *before* any contact attraction or damping acted on the
   body. That matters because the optimizer's own enforcement holds contacts
   still; stillness re-measured afterwards would be manufactured evidence (see
   the lesson in [human_optimization.md](human_optimization.md)).
4. **Hysteresis + minimum duration.** A contact starts when the joint comes
   within 3 cm and survives while it stays within 3.5 cm — the looser exit
   threshold kills one-frame boundary flicker. Spans shorter than 3 frames are
   dropped.
5. **Bridging short gaps.** Two contacts of the same joint less than 2 s apart
   are merged into one when the joint barely moved across the gap (< 3 cm)
   *and* stayed near the wall throughout (within 8 cm signed distance). That
   last condition is the re-grip guard: a hand that left the wall and came back
   is two contacts, not one. Bridged frames are flagged in the output
   (`merged`) so a consumer can ignore them.

## Layer 2: completion — adding what the detector missed

"Within 3 cm and still" is a high bar: on the held-out test corpus the detector
alone finds only ~70 % of true contacts. Completion unions two looser signals
onto the detector labels. It is strictly **add-only**.

- **Gates.** A contact group (a hand, a foot, an ankle) counts as touching if
  its distance to the scene is under 30 cm *and* its pre-enforcement drift is
  under 8 cm — the detector's two conditions, relaxed.
- **Force.** The static support solve below. A group carrying more than 5 % of
  body weight while plausibly still (drift < 15 cm) counts as touching,
  whatever the measured wall distance says.

Each signal is filtered to spans of at least 3 valid frames before the union.
Additions land on the group's own joint only (wrist / ankle / foot). The solved
force magnitudes are saved (`limb_force_n`), and `contact_source` is a
per-joint-frame bitmask: 1 = detector, 2 = gate, 4 = force, 8 = prediction
fusion.

### The static support solve

On every frame, something has to hold the climber up. A small per-person
optimization takes the body's weight (about 600 N) and splits it among six
candidate support points — the two wrists, two ankles, two feet. The split is
decided by prices:

- force on a fast-moving limb is expensive (a work penalty, using the raw
  un-smoothed limb speed);
- force far from the wall is expensive — but "far" is priced **relative to the
  frame's nearest group**, not absolutely, so on a mis-registered body where
  every limb reads far, the weight still lands on the least-far limbs instead
  of nowhere (the story behind that choice is two sections down);
- foot and ankle forces must stay inside friction cones built from the scene
  normals.

The solve is **static**: body velocities and accelerations are zeroed. That is
not an approximation we hope is fine — re-running the original dynamic solve
with motion zeroed changed under 3 % of labels and none of the scores, so the
motion terms were dead weight and were dropped.

What force adds over the gates: it recovers contacts where the reconstruction
itself is misplaced, so the measured wall distance lies. A limb hanging
10–30 cm "in the air" that the solve keeps loading with body weight is, on the
annotated data, usually a real grip on a misregistered body. This is also the
honest explanation of force arrows floating in mid-air in the viewer: the label
is right, the body is misplaced. Fixing that needs a translation-level body
correction — soft attraction cannot move a body half a meter, and relaxing the
chamfer term does not either (both measured).

### What completion earns

Macro precision / recall against manual annotations, at four limbs
(hand = wrist∨fingers, foot = foot∨ankle):

| labeler | 12 tuning scenes (hard) | 28 held-out scenes |
|---|---|---|
| detector alone | 0.874 / 0.401 | 0.983 / 0.702 |
| detector ∪ gates | 0.931 / 0.677 | 0.958 / 0.873 |
| **detector ∪ gates ∪ force** | 0.930 / 0.715 | **0.951 / 0.909** |

Caveats, measured and left honest: the macro hides per-scene spread (9 of 28
held-out scenes meet P ≥ 0.95 ∧ R ≥ 0.90 individually); at six-group
granularity precision drops to 0.744, because completion often labels foot
*and* ankle together where the annotation credits only one; and the hard tuning
scenes are capped by body misregistration, not by thresholds. The remaining
false positives are near-and-still hovering limbs 0.5–9 cm from the cloud — at
the ambiguity floor of the annotation and the point cloud itself. The
confidence weight is what marks them.

### Relative gap pricing (2026-07-24) — measured effect

The gap penalty first priced each limb's **absolute** wall distance. On a
floating (mis-registered) body every limb reads far, so the solve preferred
leaving gravity unbalanced over paying for any force — the force witness never
fired on the very bodies it exists to rescue. It now prices the distance
**beyond the frame's nearest group**: prefer nearer limbs, never refuse to
allocate. The same change is what lets `contacts_2` run the full completion on
the `kinematics_2` body (whose stillness witness is the forwarded `drift_pre`).

Scores against manual annotation at the **artifact level** (the saved per-joint
labels folded onto the annotation groups, every valid labeled frame, no span
merge — a stricter protocol than the macro table above), on the 12 annotated
local scenes (sport_climb excluded: its `kinematic_1` body drifts ~60 cm per
stillness window, so no honest stillness witness exists and its labels stay
sparse under any pricing):

| pooled, 12 scenes | P | R | F1 | P/R/F1 with foot∪ankle folded |
|---|---|---|---|---|
| contacts_1, absolute gap (before) | 0.783 | 0.664 | 0.719 | — |
| contacts_1, relative gap | 0.794 | 0.725 | 0.758 | 0.900 / 0.728 / 0.805 |
| contacts_2 (on the `kinematics_2` body) | 0.771 | 0.753 | 0.762 | 0.880 / 0.756 / 0.813 |

Nine scenes are unchanged to the third digit; the recall comes from the
floaters (gtIGgs R 0.17 → 0.57 at P 0.985; contacts_2 reaches 0.64). The
foot∪ankle fold matters because annotators almost never label the ankle
(right_ankle has zero positives in the whole set) while heel-on-surface
predictions land there — about half of the raw false positives are that naming
convention, not geometry.

**Effect on confidence.** On `contacts_1` the confidence stays reliable: label
precision rises monotonically across confidence bins (0.70 / 0.74 / 0.77 /
0.83 / 0.89 for bins of 0.2), and true positives average 0.54 against 0.33 for
false ones. The new force-added floater labels arrive with LOW confidence — the
surface witness honestly sees a floating body — which is exactly right for
consumers that trust confidence; `kinematics_2` ignores confidence for its pull
for the same reason, and tests the label against reprojection instead. On
`contacts_2` the confidence is optimistic and mildly non-monotone in the mid
band (0.67 / 0.83 / 0.71 / 0.58 / 0.86): its surface witness is measured on a
body the labels themselves seated, so a false positive that got pulled onto the
wall inherits surface evidence (false positives average 0.57 there). The ≥ 0.8
band still holds P 0.86 and carries most positives. Treat `contacts_2`
confidence as an upper bound; the source-conditional caps were calibrated on
the `kinematic_1` body.

### Rules that keep it honest

- Completion runs only on a body that carries the pre-enforcement drift
  (`drift_pre`). `kinematic_1` saves it at refine start and `kinematic_2`
  forwards it unchanged, so both are eligible; stillness re-measured on an
  enforced body would be manufactured evidence, and an old npz without
  `drift_pre` stays detector-only.
- A completed label must never be judged by how much it helps a later stage —
  the dynamics solve will happily use any force point it is given. Validation
  is against manual annotations only.
- The solve's domain is bouldering. It knows nothing about ropes; on
  rope-supported climbing it would push rope force into the limbs.

### Dead ends, so they are not retried

- The dynamics residual as a per-frame contact *score*: chance level.
- Reading the gravity direction out of the residual: unidentifiable (the
  residual norm is invariant to it).
- Trajectory smoothing for label quality: no effect.
- A relative (closest-limb) gap penalty was first rejected here — "it makes far
  limbs cheaper exactly on floating scenes" — and then adopted (2026-07-24) for
  that same reason: with absolute pricing the solve on a floating body prefers
  leaving gravity unbalanced (every limb is expensive), so the force witness
  never fires on the bodies it exists to rescue. Measured on the 13 annotated
  local scenes: healthy scenes byte-identical, pooled precision +1 pt, and the
  floater gtIGgs goes R 0.17 → 0.57 at P 0.985. The drift guard still blocks
  scenes whose reconstruction is too unstable to witness stillness
  (sport_climb stays unchanged).
- Known open issue: for clips longer than VGGT's frame cap the frame axis is
  subsampled while `fps` stays native, which stretches every temporal window in
  this stage — currently unreachable (all corpus scenes are under the cap).

## Design decisions

- **A separate stage, not part of the fit.** The readout must be able to change
  (e.g. its recall knob) without changing the bodies. The kinematic stage's
  auto-contact detector deliberately pins its *own* copies of the stillness
  knobs for the same reason, in the other direction: retuning the readout must
  not silently change the frozen labels the fit used.
- **No gates beyond the ones above** — in the detector. There is no
  clearance/floating-body gate, no detector-score gate, no body-health gate, no
  persistence prior. Every one of those was tried and removed: they traded
  recall for precision in ways the confidence weight now expresses continuously
  instead. The completion pass is not another gate on the detector — it only
  ever adds.
- **Knobs** live in `configs/human_optimization.toml` under `[contacts]`
  (detection), `[contacts.completion]` (the union recipe and solve scales), and
  `[contact_confidence]` (weighting).

## Scene-free mode (`use_scene = false`)

Some clips have no usable scene reconstruction — a static camera gives VGGT no
parallax (the peter campus board). Setting `[contacts] use_scene = false` drops
every scene witness. The labels become **the external model's predicted
contacts, gated on stillness**: a limb the model calls in contact (probability
≥ 0.5) gets pseudo-distance 0 at its representative joints and +∞ elsewhere,
and that feeds the *unchanged* drift gate, hysteresis and minimum-duration
filter. The SDF detector, completion, fusion and the depth-fit witness are all
skipped. The file records `contact_label_method = predicted-drift-gate-v1` and
no scene hash, so downstream freshness checks know the scene was never read.

Confidence in this mode is the model's probability × the stillness weight for
contact labels, and its complement for no-contact (`scene_free_pred_still_v1`);
visibility is not assessed and stays 1.

One knob matters: the stillness threshold must match the body's own noise.
Without a depth anchor the peter kin1 body sways 5–10 cm during real holds, so
the campus config raises `[contact_gate] still_m` to 0.12 — the 0.03 default
deleted a third of the model's true contacts there. A limb genuinely in flight
still exceeds the 24 cm-per-window kill threshold.

## Output

`human_optim/contacts_<tag>.npz` — see the field table in
[contact_confidence.md](contact_confidence.md). The optional overlay draws red
for contact, green for free, with dot size proportional to confidence.

Cost: ~35 s wall for a 445-frame single-person clip on one GPU, end to end. The
static support solve is ~7 s per person, the ray-cast occlusion ~0.3 s, the
overlay ~4 s; the rest is model load and the per-vertex scene SDF. Each extra
person adds roughly one solve plus its SDF pass.
