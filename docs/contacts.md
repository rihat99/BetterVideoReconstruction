# The contact stage: per-joint scene contact

`scripts/stages/estimate_contacts.py` — a one-shot readout, off the default chain.
Reads a refined body (`--bodies kinematic_1` → `contacts_1.npz`, `kinematic_2` →
`contacts_2.npz`) and the fused scene cloud; writes, for every person, a boolean
per-joint contact label per frame plus a confidence weight for each label. The
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

## Design decisions

- **A separate stage, not part of the fit.** The readout must be able to change (e.g.
  its recall knob) without changing the bodies. The kinematic stage's auto-contact
  detector deliberately pins its *own* copies of the stillness knobs for the same
  reason, in the other direction: retuning the readout must not silently change the
  frozen labels the fit used.
- **No gates beyond the ones above.** There is no clearance/floating-body gate, no
  detector-score gate, no body-health gate, no persistence prior. Every one of those
  was tried and removed: they traded recall for precision in ways the confidence
  weight now expresses continuously instead.
- **Knobs** live in `configs/human_optimization.toml` under `[contacts]` (detection)
  and `[contact_confidence]` (weighting).

## Output

`human_optim/contacts_<tag>.npz` — see the field table in
[contact_confidence.md](contact_confidence.md). The optional overlay draws red for
contact, green for free, with dot size proportional to confidence.
