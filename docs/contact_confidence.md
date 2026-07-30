# Per-joint contact-label confidence (schema 8)

Every contact label the stage writes — contact *and* no-contact — gets a
weight `label_confidence` between 0 and 1. It answers one question: **how much
should a learner trust this label on this frame?** Zero means "ignore this
label." It is a training weight, not a calibrated probability, and it never
changes the binary label itself.

## The idea in one paragraph

Three things decide the weight. First, **can we even see the joint?** A joint
that is out of frame or hidden behind a body cannot be trusted either way, so
visibility multiplies everything. Second, **how strong is the evidence for
this label's class?** For a contact label, evidence means the body surface
really sits on the wall. For a no-contact label, evidence means the joint
clearly moved — a still joint proves nothing. Third, **where did the label
come from?** The completion pass creates some labels from "near and still"
(gates) and some from the support solve (force). A label must not be
certified by the same signal that created it, so each source gets a
**ceiling** on its evidence, and only a different, semi-independent signal
can raise that ceiling.

```text
label_confidence = visibility × evidence-capped-by-source
```

One exception sits on top of this algebra: a label that prediction **fusion**
changed — a rescue or a veto (see [contact_fusion.md](contact_fusion.md)) —
carries the fused log-odds confidence instead. Labels fusion left alone keep
the witness confidence described here.

## How to read the numbers

All examples assume the joint is fully visible.

| situation | confidence | why |
|---|---:|---|
| detector contact, on the wall | 0.92 | top tier: the detector is precise (0.983), it earns full surface evidence |
| gate-added contact, on the wall | 0.50 | capped: "near and still" both made the label and would confirm it — this is where the hovering false positives live |
| gate-added contact carrying half body weight | 0.80 | the force witness is semi-independent, so real load lifts the gate cap |
| force-added contact 20 cm off the wall | 0.61 | a misregistration rescue: probably a real grip on a misplaced body, but positionally uncertain — capped at 0.70 |
| no-contact, joint moving 0.2 m/s | 0.91 | clear motion is the one honest no-contact witness |
| no-contact, joint still | ~0.00 | a still joint proves nothing, at any distance from the wall |
| anything, joint hidden behind a body | 0.00 | visibility multiplies everything |

The pattern to remember: **red spheres in the viewer come in tiers.** Bright
red = the detector saw it on the wall. Medium red = the support solve loads
it. Dim red = it was merely near and still. Green fades to grey wherever the
"proof" would just be stillness.

## Visibility

Two factors, multiplied.

**In the image.** The joint's projection must be inside the frame (a soft
10-pixel border) and in front of the camera.

**Not behind a body.** One exact ray is cast from the camera through the
joint center and intersected with the full-resolution triangles of every
tracked person. If the nearest surface *in front of* the joint stands more
than a few centimeters before it, the joint is hidden:

```text
occlusion_gap   = joint_depth − nearest_in-front_surface_depth   (NaN if none)
body_visibility = sigmoid((0.05 m − gap) / 0.01 m)               (1 when NaN)
```

So visibility is ~1 for gaps well under 5 cm, one half at exactly 5 cm, and
~0 beyond. A joint's *own* skin never hides it: triangles whose corners
belong to the joint's own group are skipped. The groups are each whole hand,
each ankle+foot pair, and the whole axial torso (pelvis, spines, hips,
collars, neck, shoulders) — a joint center buried inside its own body block
would otherwise always look "occluded". Other people's triangles are never
own, so another climber in front does hide the joint.

One ray is enough because the measured gaps are strongly bimodal: a surface
is almost always either clearly in front (a real occluder, tens of cm) or at
the joint itself (its own skin) — grazing cases in between are ~0.04 % of
frames.

## Evidence for a contact label

The raw ingredient is **surface evidence**: how plausibly the body surface
sits on the wall. With `d` the guarded signed distance to the scene (meters,
positive outside the wall):

```text
near     = sigmoid((0.05 − d) / 0.01)        fades out past 5 cm outside
not_deep = sigmoid((d + 0.02) / 0.01)        deep burial is suspect geometry
surface  = near × (0.40 + 0.60 × not_deep)   ≈ 0.92 right on the wall
```

Then the source cap. The label's `contact_source` bitmask says who created
it (1 = detector, 2 = gate, 4 = force, 8 = fusion), and `share` is the solved
contact force at the joint as a fraction of body weight, with
`force_ramp = 1 − exp(−share / 0.25)`:

| source | evidence | ceiling |
|---|---|---|
| detector | `surface` | none — top tier |
| gate (no detector) | `min(surface, 0.50 + 0.35·force_ramp)` | 0.50, lifted to 0.85 by force |
| force only | `min(0.70, max(surface, 0.70·force_ramp))` | 0.70 |

Why the caps: a gate label was *created* because the joint was near the wall
and still, so nearness and stillness would endorse it in a circle — and the
known false positives (a hand hovering 3 cm beside a hold) are exactly
near-and-still. The solved force is the one signal that is semi-independent
of that geometry, so it, and only it, lifts the gate ceiling. A force-only
label is the opposite case: an honest rescue on a misregistered body, real
but positionally uncertain, so it tops out at 0.70. Only the six group
joints (wrists, ankles, feet) ever carry gate or force bits; every other
joint is detector-only and unaffected. A fusion-created label (bit 8) skips
this table entirely — it carries the fused log-odds confidence.

## Evidence for a no-contact label

The only honest witness for "not touching" is **motion**: the joint clearly
moved through the window. With `v` the net displacement speed over the
~0.35 s window (from the pre-enforcement drift, so the optimizer's own
damping cannot fake stillness):

```text
scale rises from 0.10 to 0.15 m/s near the wall, response steepens to cubic
motion   = 1 − exp(−(v / scale)^power)
evidence = depth_trust × motion
```

Distance is deliberately **not** no-contact evidence: a whole-body floater
sits far from the reconstructed wall while its limbs truly touch, so "far
away" must never make a no-contact label look trustworthy. `depth_trust`
discounts motion measured on a body whose local depth fit is stretched
(the spaghettization detector), because both position and drift are then
unreliable.

A cap for "the depth witness is missing" exists (`nodata_cap`,
`nodata_force_cap`) but **ships disabled** (both 1.0): on the annotated data
its premise is false — no-contact labels without a depth witness are just as
reliable (test set: 25.8 % vs 25.6 % miss rate) or *more* reliable (tuning
set: 36.3 % vs 48.8 %) than those with one. The mechanism stays for study;
lowering the knobs below 1.0 re-enables it.

## Knobs (`[contact_confidence]`)

| knob | default | meaning |
|---|---:|---|
| `gate_only_cap` | 0.50 | ceiling for a gate-only contact |
| `gate_force_cap` | 0.85 | gate ceiling under full force corroboration |
| `force_only_cap` | 0.70 | ceiling for a force-only contact |
| `force_share_scale` | 0.25 | body-weight fraction where the force ramp saturates |
| `occ_margin_m` / `occ_soft_m` | 0.05 / 0.01 | occlusion-gap threshold and softness |
| `image_margin_px` | 10 | soft image border |
| `nodata_cap` / `nodata_force_cap` | 1.0 / 1.0 | missing-depth no-contact cap — off (premise refuted) |
| surface / motion knobs | — | the band and speed scales shown in the formulas above |

## Saved fields

`estimate_contacts` writes these to `human_optim/contacts_<tag>.npz`
(`contacts_1` from a `kinematic_1` body, `contacts_2` from `kinematic_2` —
both run the full detector + completion + fusion pipeline, since
`kinematic_2` forwards the honest `drift_pre`; note the `contacts_2`
confidence caveat in [contacts.md](contacts.md): its caps were calibrated on
the `kinematic_1` body).

| Field | Shape | Meaning |
|---|---|---|
| `joint_contact` | `(P, N, J)` bool | Final binary contact label |
| `contact_source` | `(P, N, J)` int8 | Provenance bitmask 1=detector, 2=gate, 4=force, 8=fusion |
| `limb_force_n` | `(P, N, 6)` float | Solved support force per contact group, Newtons (order in `completion_groups`) |
| `force_share` | `(P, N, J)` float | Solved force / body weight at the six limb joints; NaN elsewhere |
| `body_weight_n` | `(P,)` float | Total baked mass × g per person |
| `label_confidence` | `(P, N, J)` float | Final per-joint training weight |
| `joint_visibility` | `(P, N, J)` float | Image × body visibility |
| `joint_image_visibility` | `(P, N, J)` float | Image-border and camera-depth term |
| `joint_occlusion_visibility` | `(P, N, J)` float | Smooth body-occlusion term |
| `joint_occlusion_gap_cm` | `(P, N, J)` float | In-front occlusion gap (cm); NaN when nothing is in front |
| `surface_evidence` | `(P, N, J)` float | Raw contact evidence (uncapped) |
| `motion_evidence` | `(P, N, J)` float | Raw depth-trusted no-contact evidence (uncapped) |
| `joint_depth_gap_cm` | `(P, N, J)` float | Signed median camera-ray depth gap; NaN with too few matches |
| `joint_depth_points` | `(P, N, J)` float | Matched human-cloud point count |
| `joint_depth_trust` | `(P, N, J)` float | Local human-depth fit trust |
| `joint_sdf_cm` | `(P, N, J)` float | Unguarded signed-distance diagnostic |
| `joint_distance_cm` | `(P, N, J)` float | Euclidean scene-distance diagnostic; not a confidence input |
| `merged` | `(P, N, J)` bool | Label was filled by span merging |
| `pred_prob_limb` | `(P, N, 4)` float | Model contact probability per limb (order `pred_limbs`); NaN where absent |
| `fusion_changed` | `(P, N, 4)` int8 | +1 = fusion rescue, −1 = veto, 0 = untouched, per limb-frame |
| `fusion_method` | scalar string | `predfuse-logodds-v1`, or `none` when no predictions existed |
| `fusion_pred_clamp` | scalar | The clamp applied to the model probability |
| `confidence_schema` | scalar | `8` |
| `confidence_method` | scalar string | `source-conditional-v1` (`scene_free_pred_still_v1` in scene-free mode) |
| `confidence_motion_source` | scalar string | `drift_pre_with_final_fallback` |

The artifact also stores source hashes and every effective knob. Invalid
tracked frames have every term set to zero. Manual annotations are used only
for evaluation.

## What confidence can and cannot do

Measured on the annotated corpora (selective-risk curves,
`scripts/diagnostics/confidence_risk.py`): keeping the top half of contact
labels by confidence raises their precision from 0.961 to **0.983** on the
frozen test set (the v5 algebra managed 0.972), at essentially unchanged
mean confidence. But no confidence reaches precision 1.0: the curves
saturate at the labels' own agreement with the annotations (~0.98 test,
~0.93 tuning), because the residual errors — near-and-still hovers at the
annotation's own ambiguity floor — look identical to true contacts in every
available signal. Confidence finds the uncertain places; it cannot outrun
the labels.

Limitations, for the record: the weight is not learned or calibrated; net
displacement can miss out-and-back motion inside the window; motion does not
distinguish approaching, leaving, or sliding; wall and hold occlusion are
not modeled; the depth-fit metric does not flag a whole-body floater; force
is only semi-independent of geometry (the solve's gap penalty uses
distance), so it lifts a cap but never by itself proves a label; and
confidence does not repair a wrong binary label.
