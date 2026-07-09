# UPDATE — round 11: contact confidence, made physical

*2026-07-09, rev 13. Round 10 shipped wall-contact seating; you judged it not
ready to run on everything, so it becomes opt-in (code kept, off by default)
and the experiment leftovers get cleaned. The main work of this round is the
two confidence upgrades you asked for — occlusion awareness and relative
clearance — under one governing rule: **every term in the confidence must
state the concrete physical assumption behind it, and the whole thing must
get simpler, not more complicated.** This reopens the formula round 8 froze,
so the version stamp in the output goes 1 → 2 and downstream consumers can
tell the two apart. Rounds 1–10 live in git history.*

## First, the cleanup

- **Wall-contact seating → default off.** The pull, the rigid seat phase and
  the float gate all stay in the code as an opt-in knob for later; the
  default config no longer enables them. Nothing needs re-running: the float
  gate meant healthy bodies were bit-exact anyway, and the corpus was never
  re-run with seating on.
- Delete the `out/_round10/` experiment trees and any redirected corpus
  scratch. The round-10 numbers and renders survive in git history.

## Your two asks, restated so we can check I understood

**1. Occlusion should humble the confidence.** A contact label is only as
trustworthy as our knowledge of where the joint actually is, and there are
two separate ways to lose that knowledge. First, the camera may simply not
see the joint because the climber's own body is in front of it — that is
knowable *exactly*, from the posed mesh, no detector involved. Second, the
detector itself may be unsure (low Sapiens score) — then that keypoint never
pinned the fit, whatever the geometry says. Either one means the joint's
3D position came mostly from priors and smoothness, so its measured distance
to the wall is a guess — and confidence in **both** labels (contact and
free) should drop. The physical assumption, in one line: *an unseen or
unpinned joint has an unverified position, and an unverified position gives
an unverified contact.*

**2. Clearance should be judged relative to the body, not in absolute
centimetres.** Fifteen centimetres off the wall means two different things.
On a body whose other joints are touching the scene (distance ≈ 0), the
scene sheet next to the body is calibrated — a still joint 15 cm above it is
*confidently free*, and today's formula wrongly stays timid. On a body where
nothing touches, 15 cm is meaningless — the whole body carries an unknown
depth offset. So the "free" decision should be driven by each joint's
clearance **relative to the body's own touching joints**, with absolute
distance only as the fallback when no joint touches. One line: *touching
joints anchor the scene; clearance above that anchor is trustworthy,
clearance without an anchor is not.* (Round 8 added a relative witness as a
third voice; this round promotes relativity to the primary decider.)

## Measure first, change second

The method that worked in rounds 6 and 8: dump the factors, measure on the
labeled clips, only then touch the formula.

- **Occlusion probe.** On the labeled clips, split joint-frames by
  (self-occluded yes/no × Sapiens score) and measure how often the automatic
  label is *wrong* in each bucket. If wrong labels really concentrate on
  occluded / low-score joints, the witness is justified and the buckets
  calibrate its ramp. If they don't, that is an honest null and we do not
  ship a term that only makes confidence timid for no measured reason.
- **Relative-clearance probe.** Among still, in-view joints near the wall,
  does relative clearance separate annotated contact from annotated free
  better than absolute distance does? A simple separation comparison (AUC)
  answers it, and the same data picks the ramp scale.

## Then the changes

- **A visibility witness**: self-occlusion (a ray test against the posed
  body, the joint's own flesh excluded) combined with detector belief (the
  Sapiens score through a calibrated ramp). Applies to both classes.
- **A relative-first free witness**: anchor = the body's touching joints
  this frame; each joint judged by clearance above the anchor; the absolute
  and nothing-within-reach witnesses demoted to fallback for anchor-less
  frames.
- **A simplification pass**: every surviving term gets its one-line physical
  assumption written into `docs/contact_confidence.md` (updated in the same
  commit, as always); any term that cannot state one, or that the ablation
  shows inert, is dropped. The finished formula must have no more terms than
  today's — ideally fewer.
- Version stamp → 2, corpus re-scored (minutes — this is a pure readout
  round, no kinematics re-runs anywhere).

## How we judge

- The contact **labels stay bit-identical** — confidence rides on top.
- The round-8 hard gate, now applied to both classes: the rate of
  *confidently wrong* labels must not rise.
- Reliability — does high confidence actually predict correctness — improves
  or holds, pooled and per clip.
- The two target populations move the way you asked: occluded or weakly
  detected joints that are confidently wrong today get humbler; still, free
  joints with touching neighbours get their confidence back.
- Eyes: the confidence-shaded overlays and viewer, spot-checked before/after.

## Honesty, up front

- Occlusion genuinely destroys information. An occluded foot getting low
  confidence is *correct* even when it is annoying — the fix for that is a
  better fit or another camera, not a braver number.
- A body that touches nothing (a real hover, a fall) has no anchor, and the
  relative witness will correctly refuse to fire; those frames fall back to
  the absolute witness and stay humble. That is the right behaviour, not a
  bug.
- Reopening a frozen schema has a real cost for anything already consuming
  v1 — the stamp exists precisely so consumers can filter, but the cost is
  yours to accept and you have.
