# The contact stage: per-joint scene contact

`scripts/stages/estimate_contacts.py` — a one-shot readout, off the default chain.
Reads a refined body (`--bodies kinematic_1` → `contacts_1.npz`, `kinematic_2` →
`contacts_2.npz`) and the fused scene cloud; writes, for every person, a boolean
per-joint contact label per frame plus a confidence weight for each label. The
label is produced in two steps: a precise **detector** (below) and a
**completion** pass that adds the contacts the detector missed (its own
section further down). `joint_contact` is the
completed label; `contact_source` records where each label came from. The
confidence half has its own doc: [contact_confidence.md](contact_confidence.md).

## How a contact is detected

1. **Signed distance per vertex.** For each posed body vertex, find the nearest scene
   point and its outward surface normal (estimated by local PCA, oriented toward the
   camera). The dot product of (vertex − point) with the normal is a signed distance:
   positive in front of the wall, negative behind it. This is the same signed distance
   the kinematic penetration loss uses — one definition of "at the wall" everywhere.
   Matches to far-away or low-confidence scene points are discarded (*guarded*): a
   hole in the cloud must read as "no evidence", not as "no wall".
2. **Vertex → joint.** Each vertex belongs to its nearest joint *center* in the rest
   pose. Skinning weights would be wrong here: the vertices of a fingertip are
   partially skinned to the proximal joints, so an LBS assignment displaces contact up
   the limb. A joint's signed distance is the minimum over its own vertices.
3. **Stillness gate.** Touching means holding: a joint's *net drift* over a centered
   ~0.35 s window must be small (3 cm) for a near-wall frame to count. This separates
   a gripping hand (millimetre jitter, near-zero net drift) from a hand slowly passing
   close to the wall. The drift is read from `drift_pre` saved by the kinematic stage —
   the motion measured *before* any contact attraction or damping acted — so the
   optimizer's own enforcement cannot manufacture stillness (see the lesson in
   [human_optimization.md](human_optimization.md)).
4. **Hysteresis + minimum duration.** A contact starts when the joint comes within
   3 cm and survives while it stays within 3.5 cm; spans shorter than 3 frames are
   dropped. The looser exit threshold kills one-frame boundary flicker.
5. **Span merging.** Two contacts of the same joint separated by a short gap (< 2 s)
   are bridged when the joint barely moved across the gap (< 3 cm) *and* stayed near
   the wall throughout (within 8 cm signed distance — the re-grip guard: a hand that
   left the wall and came back is two contacts, not one). Merged frames are flagged in
   the output (`merged`), so a consumer can ignore them.

## Completion: adding the contacts the detector missed

The detector above is precise but strict — on the held-out test corpus it finds
only ~70 % of true contacts (it demands "within 3 cm and still"). The
completion pass unions two looser signals onto the detector labels. It is
**add-only**: it never removes a detector label.

- **Gates:** a contact group (a hand, a foot, an ankle) whose distance to the
  scene is under 30 cm *and* whose pre-enforcement drift is under 8 cm.
- **Force:** the static support solve below. A group carrying more than 5 % of
  body weight while plausibly still (drift < 15 cm) is labeled in contact.

Each component is filtered to spans of at least 3 valid frames before the
union. Additions are written to the group's own joint (wrist / ankle / foot)
only. The solved force magnitudes are saved (`limb_force_n`), and
`contact_source` is a bitmask per joint-frame: 1 = detector, 2 = gate,
4 = force.

### The static support solve

The idea is simple: on every frame, gravity has to be supported by something.
A small per-person optimization takes the body's weight (about 600 N) and
decides how to split it among the six candidate support points — the two
wrists, two ankles, two feet. Force on a fast-moving limb is expensive (the
work penalty, using the raw un-smoothed limb speed), force far from the wall
is expensive (the gap penalty — priced on the distance *beyond the frame's
nearest group*, so on a mis-registered body where every limb reads "far" the
weight still lands on the least-far limbs instead of nowhere), and foot/ankle
forces must stay inside friction cones built from the scene normals. The solve is **static**: body
velocities and accelerations are zeroed. That is not an approximation we hope
is fine — re-running the original dynamic solve with motion zeroed changed
under 3 % of labels and none of the scores, so the motion terms were dead
weight and were dropped.

What force adds over the gates: it recovers contacts where the reconstruction
is misplaced, so the measured wall distance lies. A limb 10–30 cm "in the air"
that the solve keeps loading with body weight is, on the annotated data,
usually a real contact on a misregistered body. This is also the honest
explanation of force arrows appearing in the air in the viewer: the label is
right, the body is misplaced, and fixing it needs a body-registration
(translation-level) correction — soft attraction cannot move a body half a
meter, and relaxing the chamfer term does not either (both measured).

### What it earns

Macro precision / recall against manual annotations, at four limbs
(hand = wrist∨fingers, foot = foot∨ankle):

| labeler | 12 tuning scenes (hard) | 28 held-out scenes |
|---|---|---|
| detector alone | 0.874 / 0.401 | 0.983 / 0.702 |
| detector ∪ gates | 0.931 / 0.677 | 0.958 / 0.873 |
| **detector ∪ gates ∪ force** | 0.930 / 0.715 | **0.951 / 0.909** |

Caveats, measured and left honest: the macro hides per-scene spread (9 of 28
held-out scenes meet P ≥ 0.95 ∧ R ≥ 0.90 individually); at six-group
granularity precision drops to 0.744 because completion often labels foot
*and* ankle together where the annotation credits one; the hard tuning scenes
are capped by body misregistration, not thresholds. The remaining false
positives are near-and-still hovering limbs 0.5–9 cm from the cloud — at the
ambiguity floor of the annotation and the point cloud. The confidence weight
(next doc) is what marks them.

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

The dynamics residual as a per-frame contact *score*: chance level. Reading
the gravity direction out of the residual: unidentifiable (the residual norm
is invariant to it). Trajectory smoothing for label quality: no effect.
A relative (closest-limb) gap penalty was first rejected here — "it makes far
limbs cheaper exactly on floating scenes" — and then adopted (2026-07-24) for
that same reason: with absolute pricing the solve on a floating body prefers
leaving gravity unbalanced (every limb is expensive), so the force witness
never fires on the bodies it exists to rescue. Measured on the 13 annotated
local scenes: healthy scenes byte-identical, pooled precision +1 pt, and the
floater gtIGgs goes R 0.17 → 0.57 at P 0.985. The drift guard still blocks
scenes whose reconstruction is too unstable to witness stillness
(sport_climb stays unchanged). Known open issue: for clips longer than VGGT's
frame cap the frame axis is subsampled while `fps` stays native, which
stretches every temporal window in this stage — currently unreachable (all
corpus scenes are under the cap).

## Design decisions

- **A separate stage, not part of the fit.** The readout must be able to change (e.g.
  its recall knob) without changing the bodies. The kinematic stage's auto-contact
  detector deliberately pins its *own* copies of the stillness knobs for the same
  reason, in the other direction: retuning the readout must not silently change the
  frozen labels the fit used.
- **No gates beyond the ones above** — in the detector. There is no
  clearance/floating-body gate, no detector-score gate, no body-health gate, no
  persistence prior. Every one of those was tried and removed: they traded recall
  for precision in ways the confidence weight now expresses continuously instead.
  The completion pass is not another gate on the detector — it only ever adds.
- **Knobs** live in `configs/human_optimization.toml` under `[contacts]`
  (detection), `[contacts.completion]` (the union recipe and solve scales), and
  `[contact_confidence]` (weighting).

## Output

`human_optim/contacts_<tag>.npz` — see the field table in
[contact_confidence.md](contact_confidence.md). The optional overlay draws red for
contact, green for free, with dot size proportional to confidence.

Cost: ~35 s wall for a 445-frame single-person clip on one GPU, end to end.
The static support solve is ~7 s per person, the ray-cast occlusion ~0.3 s,
the overlay ~4 s; the rest is model load and the per-vertex scene SDF. Each
extra person adds roughly one solve plus its SDF pass.
