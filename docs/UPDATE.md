# UPDATE — contacts inside `kinematics_1`, round 5

*2026-07-07, rev 7. Covers the betas that shape the body, the `estimate_scale`
stage, the contact readout and its new confidence, a kinematics_2-on-auto-labels
experiment, and a code polish pass. Rounds 1–4 history: revs 4–6 in git.*

## Round 4 worked — where we are now

All of round 4 landed and was measured. Two separate gains: the methods
themselves (at the fixed round-3 readout point), then the operating point you
picked.

| step (readout close 0.02 / still 0.025) | P | R | F1 | FPR | grip speed |
|---|---|---|---|---|---|
| round-3 winner | 0.906 | 0.565 | 0.696 | 0.049 | 11.5 cm/s |
| + span-merge + pre-enforcement gate | 0.905 | 0.608 | 0.727 | 0.053 | 11.5 |
| + target smoothing (0.35 s) | 0.902 | 0.622 | 0.736 | 0.056 | 11.5 |
| + velocity damper (500) | 0.906 | 0.618 | 0.735 | 0.054 | **7.3** |

At your chosen operating point (close 0.03 / still 0.03): **P 0.896 / R 0.678 /
F1 0.772 / FPR 0.066**, leave-one-clip-out stable, and you confirmed the motion
looks visibly better. The damper is back and honest — because the readout now
judges stillness from the body *before* enforcement, damping can no longer fake
a still contact.

Round 4 also closed some questions for good:

- **A global scale cannot fix the hovering.** Still hands/feet already
  *penetrate* the wall by ~3 cm on average while the missed ones hover +4 cm —
  one number can't move both ways at once. The problem is per-limb, i.e.
  **body proportions**, not scale.
- The remaining misses are mostly **moving contacts** (excluded by design) and
  **occluded holds** (never in the scene cloud). Target smoothing recovered a
  little; the camera is not the culprit (3 of 4 cameras are static).
- The optimizer itself is healthy: the two kinematic trees agree on quaternion
  conventions; no hidden graph bug.

## Round 5 — what it is about

You named the two things still standing, and this round attacks exactly them,
plus the groundwork for learning from these labels:

1. **Limbs come out short** (worst on competition_2), so the hand stops a few
   centimetres before the wall. Proportions live in the **betas** — the ten
   shape numbers each frame's body model predicts. Today we take a plain
   *average* of them over all frames, good and bad alike. When the person is
   small in the image or half cut off, the per-frame betas collapse toward the
   average human — which, for a tall climber, means shorter limbs — and those
   bad frames dilute the good ones. So: measure how noisy the per-frame betas
   really are, then replace the average with a robust, quality-weighted
   estimate. The metric scale gets the same noise treatment (details below) —
   scale and betas together decide where the fingertips end up.
2. **Occluded contacts can't be recovered — but they can be honestly labeled.**
   The plan is a full rewrite of contact confidence: every joint in every frame
   gets a confidence in its label, **whether that label is contact or
   no-contact**. A fast-moving joint is a near-certain no-contact. A slow joint
   far from the wall is a certain no-contact. A still joint pressed into the
   wall is a certain contact. A slow joint where the scene cloud has *no
   points* (the occluded case) is a low-confidence unknown — for either label.
   That "unknown" is the honest answer to occlusion, and it is what makes these
   labels usable for training later: a learner can be told which labels to
   trust. The current per-contact confidence was machine-generated and only
   covers positives; it gets deleted and rebuilt from named signals.
3. **Try kinematics_2 with our auto labels.** kinematics_2 with manual
   annotations produces the most natural motion we have. Feed it the automatic
   labels instead (weighted by the new confidence) and see how close it gets —
   the first turn of the self-training loop you described.
4. **Polish.** We are heading toward the final iteration: dead code out, big
   scripts split, war-story comments compressed, no behavior change, runtime
   not a hair slower.

## The plan — five moves

**1. Robust betas (the proportions lever).** First a diagnostic, then the fix.
Diagnostic: per clip, how much do the per-frame betas move, and do they
correlate with how large the person is in the image? Also plot the limb lengths
they imply over time — if competition_2's far-away frames systematically imply
shorter arms, we have the mechanism in hand. Fix: one small function that
aggregates per-frame betas into one per-person vector using a **weighted
median** — frames where the person is big, well-tracked and confidently
detected count more; outlier frames stop mattering at all. The optimizer uses
it (today it uses the plain mean); the scale stage A/Bs it as its ruler.
This is *not* the rejected "optimize betas in the fit" — betas stay frozen in
the optimizer; only how we summarize what the upstream model already predicted
changes.

**2. A noise pass on `estimate_scale`.** Three suspects, each cheap to test:
(a) **Mask edge bleed** — the person's mask includes a rim of pixels whose
VGGT depth is really the *background wall*, farther than the body. Those pixels
drag the scale down, which pushes the whole scene slightly toward the camera —
and matches the measured symptom (still hands *penetrating* by ~3 cm). Fix:
erode the mask a few pixels before use. (b) **Frame quality** — pool per-frame
scale estimates weighted by how well that frame saw the body (mask size, depth
confidence), not just how many pixels it had. (c) **Extremity emphasis** — the
scale stage weights hands/feet ×10, but VGGT depth is at its *worst* exactly on
thin fast limbs; A/B whether that emphasis helps or hurts the scale estimate.
Order matters: fix betas first, then scale, then re-measure the
penetration-vs-hover balance — once proportions are right, one clean global
scale may finally be able to center both.

**3. Contact confidence, rewritten.** Deleted: the old positives-only formula
in the uncertainty stage. Built instead, at the point where labels are made
(the contact readout): per person, frame and joint, three numbers —

- **p_contact** — how contact-like the *geometry and motion* look: rises as the
  joint gets closer to the wall and stiller (using the honest pre-enforcement
  drift, so enforcement can't inflate it).
- **observability** — how much we actually *saw*: are there scene points near
  the joint at all (occlusion check), how confident is VGGT depth there, is the
  joint in frame.
- **label confidence** — confidence in the final binary label. For a *contact*:
  observability × p_contact × persistence. For a *no-contact*: either the joint
  is moving fast (confident from motion alone, even when occluded — you can't
  grip while flying) or we saw the space near it and it wasn't contact-like.
  Slow + occluded ⇒ low confidence: the honest unknown.

Every ingredient is something we already compute (signed distance, drift, VGGT
confidence, Sapiens score, persistence) — one shot, no optimizer cost.
Validation: against the manual annotations, bin the labels by confidence and
check accuracy rises with confidence (a reliability curve), plus the two
monotonicity checks straight from your examples (faster ⇒ more confident
no-contact; closer ⇒ more confident contact).

**4. kinematics_2 on auto labels.** Run the existing kinematics_2 machinery
with `contacts_1` labels instead of the manual file. Three safety rails:
(a) **positives only** — an auto "no contact" is not a real negative (it may be
occlusion), so the repulsion term stays off; only attraction and the velocity
damper act, each scaled by the new label confidence; (b) the honest drift from
kinematics_1 is carried through into the kinematics_2 output, so the readout on
the refined body still judges stillness pre-enforcement (this also fixes a
latent gap for *manual* kinematics_2, which today loses that field); (c) the
final judge stays external — scored against manual annotations, same gates.
If confident negatives prove trustworthy in move 3's reliability curve, a
second variant may cautiously enable repulsion on them.

**5. The polish pass.** Now that the method is settling: remove what brings
nothing (the orphaned velocity helper of a long-removed loss; the old
confidence block; anything a grep proves dead), rename the "prototype" scorer
to what it is (the evaluation harness) and move its reusable parts into tools,
split oversized scripts, and cut the long war-story comments in the config and
code down to one line each — history lives in git and these docs. Hard rule:
behavior-preserving. After cleanup, one clip re-runs bit-identical and no
slower. This lands last, once the round's experiments have settled, so we don't
polish code that changes the next day.

## How we judge it

- **Labels**: P / R / F1 / FPR per clip + pooled (4 clips; sport_climb
  separate), against the round-4 final point (P 0.896 / R 0.678 / F1 0.772 /
  FPR 0.066). Recall up, precision ≥ ~0.88, FPR not above ~0.07.
- **The funnel, re-run** after betas + scale: the hovering category should
  shrink and the penetration/hover imbalance should center; competition_2 is
  the bellwether clip.
- **Reprojection gate** unchanged: vs **raw** targets, within +0.5 px of the
  round-4 winner. Chamfer / penetration reported, not gated.
- **Confidence**: the reliability curve must be monotone-ish per class, and the
  known occluded misses must land in low-confidence no-contact (that is the
  whole point).
- **kinematics_2-auto**: judged against manual annotations *and* your eyes in
  the viewer — naturalness (grip speed, jitter) is the reason to run it.
- **Runtime**: optimizer untouched except the kin2 experiment; everything else
  is one-shot. Cleanup must not cost a millisecond.
- **Leave-one-clip-out** at the final operating point, as before.

## Honest limits

- Occluded contacts stay unrecovered — they become *honestly uncertain* instead
  of confidently wrong. That is the deliverable, not recall on them.
- Moving contacts stay excluded by design; recall 0.9 remains capped by that
  category unless the definition changes (not this round).
- Robust betas can only recover what the per-frame model *sometimes* saw — if
  every frame of a clip under-predicts limb length, no aggregation fixes it.
- kinematics_2-auto learns from labels made on kinematics_1's body — the
  external judge (manual annotations) is what keeps that loop honest.

## Order of work

1. **Commit round 4 first** (12 files, working tree → main) so every round-5
   diff is separable.
2. Betas diagnostic → robust betas → scale noise pass → re-run
   scale→fuse→kinematics_1→contacts on the locals → funnel + scores.
3. Confidence rewrite (readout-only; no bodies change) → reliability curves.
4. kinematics_2-on-auto experiment (uses 2's bodies + 3's confidence).
5. Polish pass, then the final combined tables + leave-one-out.

## Still not doing

- No per-body scale, no betas optimization inside the fit (aggregation only).
- No auto negative *enforcement* by default (repulsion off for auto labels;
  a confidence-gated variant only if the reliability curve earns it).
- No reduction of any smoothness weight; no camera-pose modification.
- No new per-iteration work inside the optimizer loop.
- dynamics, the viewer, and the DB stay untouched (the DB re-run still folds
  into the pending Issue-22 pass at the very end).
