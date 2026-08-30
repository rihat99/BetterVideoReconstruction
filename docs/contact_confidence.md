# Contact-label confidence

Every label `estimate_contacts` writes — contact *and* no-contact, for every
contact frame at every video frame — gets a weight `frame_label_confidence`
between 0 and 1. It answers one question: **how much should a downstream
consumer trust this label?** Zero means "ignore this label." It is a training
weight, not a calibrated probability, and it never changes the binary label
itself.

The algebra was built in the per-joint era (schema 8, "source-conditional") and
now runs unchanged on the contact-frame axis; the shapes were always
axis-agnostic. It lives in `tools/human_optim/confidence.py` and is computed in
both contact readouts (`contacts_1` from the `kinematic_1` body, `contacts_2`
from `kinematic_2`).

## The idea in one paragraph

Three things decide the weight. First, **can we even see the point?** A contact
frame that is outside the image or hidden behind a body cannot be trusted
either way, so visibility multiplies everything. Second, **how strong is the
evidence for this label's class?** For a contact label, evidence means the
point really sits on the wall. For a no-contact label, evidence means the point
clearly moved — a still point proves nothing. Third, **where did the label come
from?** A label must not be certified by the same signal that created it, so
each source gets a **ceiling** on its evidence, and only a different,
semi-independent signal can raise that ceiling.

```text
frame_label_confidence = visibility × evidence-capped-by-source
```

## The two sources

The stage's union has two arms, and each is one source bit in `contact_source`:

| bit | source | what created the label |
|---|---|---|
| 2 | geometric arm | "near the wall AND still" — or pressing, with the solve's load confirmation (plus the span merge and the union's gap-welding) |
| 4 | force witness | the support solve had to route real load through this frame |

The per-joint era also had an external image-model detector (bit 1) and a
prediction-fusion path (bit 8); neither exists in the frames era, so those bits
never appear.

Why the caps: a label must not be certified by the signal that created it. A
geometric label was created because the frame was near and still, so nearness and
stillness would endorse it in a circle — and the known false positives (a hand
hovering 3 cm beside a hold) are exactly near-and-still.

The 2026-08-30 rework kept the arms' roles — a one-day experiment that made the
force witness the deciding arm was refuted (see [`contacts.md`](contacts.md),
gate 3) — so the ceilings keep their original ordering: both arms agreeing is
trusted most (0.85), a lone force rescue is positionally uncertain by
construction (0.70), and a lone geometric label is merely unconfirmed (0.50).

With `share` the solved force at the frame as a fraction of body weight and
`force_ramp = 1 − exp(−share / 0.25)`:

| source | evidence | ceiling |
|---|---|---|
| geometric (bit 2) | `min(surface, 0.40 + 0.50·force_ramp)` | 0.40, lifted to 0.90 by real load |
| force only (bit 4, no bit 2) | `min(0.85, max(surface, 0.85·force_ramp))` | 0.85 |

So the red spheres come in tiers: a loaded on-the-wall contact reaches ~0.90, a
force decision off the wall ~0.85, and a geometric label physics never saw stops
at 0.40.

## Evidence for a contact label

The raw ingredient is **surface evidence**: how plausibly the point sits on the
wall. With `d` the guarded signed distance to the scene (metres, positive
outside; `+inf` where the far/confidence guards failed = zero evidence):

```text
near     = sigmoid((0.05 − d) / 0.01)        fades out past 5 cm outside
not_deep = sigmoid((d + 0.02) / 0.01)        deep burial is suspect geometry
surface  = near × (0.40 + 0.60 × not_deep)   ≈ 0.92 right on the wall
```

Then the source cap above.

## Evidence for a no-contact label

The only honest witness for "not touching" is **motion**: the point clearly
moved through the window. With `v` the net displacement speed over the ~0.35 s
window (from the persisted `drift_pre`, so the optimizer's own contact damping
cannot fake stillness):

```text
scale rises from 0.10 to 0.15 m/s near the wall, response steepens to cubic
motion   = 1 − exp(−(v / scale)^power)
evidence = depth_trust × motion
```

Distance is deliberately **not** no-contact evidence: a whole-body floater sits
far from the reconstructed wall while its limbs truly touch, so "far away" must
never make a no-contact label look trustworthy. `depth_trust` discounts motion
measured on a body whose local depth fit is stretched (the spaghettization
detector, `[depth_fit]`); each contact frame reads the trust of the joint that
owns its mesh vertex. A cap for "the depth witness is missing" exists
(`nodata_cap`, `nodata_force_cap`) but ships disabled (both 1.0) — its premise
was refuted on the annotated data in the per-joint era; the mechanism stays for
study.

## Visibility

Two factors, multiplied.

**In the image.** The frame's projection must be inside the image (a soft
10-pixel border) and in front of the camera.

**Not behind a body.** One exact ray is cast from the camera to the contact
frame's world position and intersected with the full-resolution triangles of
every tracked person. If the nearest surface *in front of* the point stands
more than a few centimetres before it, the point is hidden:

```text
occlusion_gap   = point_depth − nearest_in-front_surface_depth   (NaN if none)
body_visibility = sigmoid((0.05 m − gap) / 0.01 m)               (1 when NaN)
```

A frame's *own* local skin never hides it: triangles owned by its own occlusion
group — the whole hand, the ankle+foot pair, the whole axial torso, per the
joint that owns the frame's mesh vertex — are skipped. Another person's
triangles are never own, so a climber in front does hide the point.

## The frame → joint fold

Downstream consumers usually want **joints**, not 35 mesh points. The stage
folds the frame labels onto **report joints** and saves the result:

* Every frame reports at its parent joint — except the hand frames (palms,
  fingertips, thumbs), which all report at their **wrist**, the joint we
  usually call "the hand". The rule is an ancestor walk up the kinematic tree,
  so no hand-frame name list exists.
* `joint_contact` is the OR of the joint's frames.
* `joint_label_confidence` folds with the label's own algebra: a **contact**
  joint is as sure as its surest touching frame (the OR needs one true member);
  a **no-contact** joint is only as sure as its least sure frame (the claim
  needs every member). This is the same convention the per-joint era's export
  used for fingers→wrist.
* A joint that owns no contact frame stays `False` with `NaN` confidence — the
  stage makes no statement there.

## Knobs

`[contact_confidence]` in `configs/human_optimization.toml` (defaults in
`tools/human_optim/confidence.py`): the surface band (`surface_*`,
`penetration_floor`), the motion scales (`motion_*`), the visibility margins
(`image_margin_px`, `occ_*`), the source caps (`gate_only_cap`,
`gate_force_cap`, `force_only_cap`, `force_share_scale`) and the disabled
no-data caps. `[depth_fit]` holds the depth-trust knobs (`cell_px`,
`min_points`, `trust_scale_log`).

## Saved fields

Added to `human_optim/contacts_<tag>.npz` (all per person and video frame; `F`
contact frames, `J` model joints):

| Field | Shape | Meaning |
|---|---|---|
| `frame_label_confidence` | `(P, N, F)` | The per-label training weight |
| `contact_source` | `(P, N, F)` int8 | 2 = geometric arm, 4 = force witness (0 off-contact) |
| `frame_visibility` | `(P, N, F)` | Image × body-occlusion visibility |
| `frame_image_visibility` | `(P, N, F)` | In-image / in-front-of-camera term |
| `frame_occlusion_visibility` | `(P, N, F)` | Smooth body-occlusion term |
| `frame_occlusion_gap_cm` | `(P, N, F)` | In-front body gap (cm); NaN = clear |
| `frame_surface_evidence` | `(P, N, F)` | Raw contact evidence (uncapped) |
| `frame_motion_evidence` | `(P, N, F)` | Raw depth-trusted no-contact evidence |
| `frame_depth_trust` | `(P, N, F)` | Local human-depth fit trust |
| `frame_depth_points` | `(P, N, F)` | Matched human-cloud point count |
| `frame_report_joint` | `(F,)` | Frame → report joint (hands fold to the wrist) |
| `joint_contact` | `(P, N, J)` bool | OR of the joint's frames |
| `joint_label_confidence` | `(P, N, J)` | Folded confidence; NaN where the joint owns no frame |

Plus `confidence_method` / `confidence_motion_source` and the effective
`confidence_knobs` / `depth_fit_knobs` as JSON strings. Terms are zero on
invalid tracked frames. The binary labels are untouched — re-running the stage
with confidence produced byte-identical `frame_contact` on every validated
scene.

## What confidence can and cannot do

The caps and scales were calibrated in the per-joint era (selective-risk curves,
2026-07: keeping the top half of contact labels by confidence raised their
precision 0.961 → 0.983 on the frozen test set). The frames-era port keeps the
algebra and the numbers; the selective-risk measurement has not yet been redone
on the frame labels. Limitations carried over: the weight is not learned or
calibrated; net displacement can miss out-and-back motion inside the window;
wall and hold occlusion are not modeled; force is only semi-independent of
geometry, so it lifts a cap but never by itself proves a label; and confidence
does not repair a wrong binary label.
