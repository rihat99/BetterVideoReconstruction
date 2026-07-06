# UPDATE — contacts inside `kinematics_1`, take two

*2026-07-06, rev 4. Covers only `root` + `kinematics_1`. Rewritten after the first
implementation round: the detector worked, the enforcement did not. This version
explains why, and replaces the enforcement design.*

## What the experiment taught us

Round one built a detector (find contacts on the settled body) and an enforcement
phase (pull the detected joints onto the wall, damp their velocity). Results:

- **The detector is good.** Precision 0.83–0.97, false-positive rate far below
  the plain distance readout, and it costs under a second. This part we keep.
- **The enforcement hurt.** Chamfer and penetration got worse on every clip, for
  almost no label improvement, at 2.3× runtime.

Three reasons, all now understood:

1. **It enforced only what was already right.** The detector is precise, so the
   pulled joints were mostly real contacts sitting a few cm off. Pulling them in
   changed almost no labels — but it dragged the body against its depth cloud
   (chamfer up) and wedged vertices into the wall (penetration up).
2. **It could never fix recall.** A missed contact has no label, so it feels no
   pull. All the cost went to joints that didn't need help.
3. **The damping loss poisoned its own well.** We damped the velocity of labeled
   joints — but stillness is exactly the signal detection reads. The optimizer
   learned to make joints *look* still instead of fitting better. Lesson, now a
   rule: **motion is a signal we read, never something we optimize.**

We also understand the low recall (0.19–0.65) now. The detector vetoed any joint
moving faster than 0.2 m/s. But a gripping hand in our fit is not perfectly
still — it jitters a few millimetres per frame, and a few mm × 30 fps is
0.1–0.3 m/s of *instantaneous* speed. Real grips tripped the veto and their
spans broke. Meanwhile a hand slowly passing the wall moves at a similar
instantaneous speed, slipped under the veto, and became exactly the false
contacts you saw visually. **Instantaneous speed cannot tell a jittering grip
from a slow pass.** Net movement over about a third of a second can: a grip
drifts under ~2 cm in that window, a passing hand covers 3–7 cm.

## The new idea

One physical rule, applied everywhere near the wall:

> **A still joint belongs ON the wall. A moving joint belongs OFF it.**

From this rule we get one measurement and one loss.

**The measurement — a stillness weight.** For every joint and frame, compute the
net drift over a short window (~11 frames): how far did the joint actually
travel? Map that to a weight `w` between 1 (still) and 0 (moving). This replaces
the speed veto everywhere. It is robust to jitter (jitter cancels out over a
window), and it catches slow passes (they keep accumulating distance). It is
computed from the current pose and **never optimized** — no gradients flow
through it, per the rule above.

**The loss — two branches blended by `w`.** For vertices near the wall:

- **Still branch** (`w` high): pull the vertex onto the surface — the existing
  attraction math, now gated by stillness instead of by detected labels.
- **Moving branch** (`w` low): push the vertex out to a small clearance
  (~4 cm). This is your "a moving hand should be repelled from the wall"
  insight, and it earns its place twice: it *prevents* the slow-pass false
  contacts instead of merely not rewarding them, and it keeps the attraction
  from ever grabbing a passing hand.

Both branches reuse losses that already exist (`scene_attraction`,
`clearance_repulsion`); the only new machinery is the drift computation and the
blending.

**No labels, no extra phase, no damping.** The weight `w` is recomputed every
iteration directly inside the normal `refine` phase — contacts in the loop, in
the simplest possible form. The separate contact phase is deleted (runtime
returns to normal), the discrete-label plumbing in the optimizer is deleted, the
damping term is deleted, and the old generic attraction ("pull anything near the
wall") is retired — the still branch is its principled replacement.

**Labels are read off once, at the end.** `estimate_contacts` stays the single
readout: distance thresholds with hysteresis, now with the same stillness gate
added. The in-loop losses make the geometry unambiguous — still hands seated on
the wall, passing hands clear of it — and the readout just reports what it sees.
Both precision and recall improve through the same mechanism: true contacts get
pulled inside the readout distance (recall up), passes get pushed outside it
(precision up).

## What changes, term by term

| Term | Was | Now |
|---|---|---|
| `auto_velocity` (damping) | 150, contact phase | **delete** — it optimized the detection signal |
| `auto_attraction` + labels + contact phase | 20, phase off by default | **delete** — replaced by the in-loop still branch |
| `contact_attraction` (generic "near ⇒ pull") | 10 | **retire** — the still branch is the gated replacement |
| `still_attraction` — new | — | ~15, band ~12 cm, inside `refine`, weighted by `w` |
| `motion_clearance` — new | — | ~1000, clearance ~4 cm, inside `refine`, weighted by `1−w` |
| speed veto (detector + readout) | hard 0.2 m/s cut | **replace with the drift/stillness gate** |
| reproj, chamfer, priors, smoothness, penetration | — | unchanged |

## Honest limits

- **Carved-out holds stay invisible.** Where the scene cloud has no points (the
  human carve), nothing can be detected or pulled. That is the recall ceiling;
  the carve-hole probe measures it per clip.
- **A smearing foot** is "moving + touching", which the rule reads as no-contact
  and pushes off. Rare in our clips; accepted.
- **A dyno catch registers a few frames late** — the drift window needs a moment
  to see that the hand has stopped. The readout hysteresis mostly hides this.

## How we judge it (hard gates this time)

Everything compared against the current HEAD run on the local clips:

1. **Fit must stay flat**: reprojection, chamfer, penetration within noise of
   HEAD. Round one failed exactly this — now it is a gate, not a hope.
2. **Runtime** within ~15 % of HEAD (the new work is one blended scene term).
3. **Labels must beat the honest baseline.** Run the new readout on the *old*
   (HEAD) bodies first — that is the baseline. Then on the new bodies: precision
   AND recall against the annotations must improve or hold, and the
   false-positive rate must not rise.
4. **Slow-pass spot check**: the visually-found false contacts (hand moving past
   the wall) should be gone or clearly reduced.

**Decision point, stated up front:** if the fit cannot stay flat at any weight
where the labels improve, we stop enforcing and keep the detector + readout
alone. That is an acceptable outcome — the labels are the product, and the
geometry already fits well without contact losses.

## Order of work

1. **Readout first, optimizer untouched.** Add the drift gate to
   `estimate_contacts`, sweep its two or three knobs on the *current* bodies,
   score precision/recall against the annotations. Gate: an operating point that
   recovers the recall the old veto destroyed, without giving up precision.
2. **The in-loop term.** Delete the label/phase/damping machinery, add the two
   blended branches to `refine`, verify parity with the new weights at zero.
3. **Evaluate** against the gates above; a small weight grid (two values per
   branch); you review and pick.
4. DB re-run stays where it was: once, at the very end, folded into the pending
   Issue-22 re-run.

## Still not doing

- No labels inside the optimizer, no extra phase, no damping of any joint.
- No triangulation cue; smoothness weights stay (standing decisions).
- Frozen anchors, per-clip adaptive distance thresholds, handing chamfer off to
  contacts locally: all deferred until the simple version's numbers ask for them.

## Outcome

Built and measured on four clips. The readout idea — a joint counts as contact only if it is
both near the wall *and* barely moving over about a third of a second — works. On the plain
fit body it gives precision 0.91 with a false-positive rate of 0.04, against 0.83 / 0.12 for
the old distance-only readout: it roughly triples the precision-to-false-alarm margin, and it
needs no change to the body at all. How much recall you keep is one knob (the stillness scale);
the default favours precision because the contacts feed things that *act* on a positive.

The in-loop part — pulling the still joints onto the wall during the fit — was tried again in
this new continuous form and failed the same way round one did. The fit got worse (the body
wants to sit on the human depth points, which are a few centimetres in front of the wall), the
contact labels got worse too, and it ran about a third slower. No weight setting escaped it. So
the in-loop term was removed and only the readout was kept — which is exactly the fallback this
plan allowed for.
