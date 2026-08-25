# The force witness: recovering contacts physics can see and geometry cannot (2026-08-12)

The user's call for this round: the near+still detector works, but it needs the
contact frame to be *near* the scene, and when the reconstructed depth is wrong
(floating bodies, limbs "not long enough") the contact is simply lost. Recover
those with forces — friction cones about the contact normal are the missing
ingredient that made the older residual attempt useless ("any joint in contact
can satisfy the residual minimization") — and let force, not the gates, be the
main signal. Also: let force settle the toe pair, where the normal tiebreak
misbehaves under penetration.

Shipped as `tools/human_optim/forces.py` + a force pass in
`estimate_contacts.py` (`contact_label_method = "frames-v2"`). Judge =
`scripts/score_frame_contacts.py` on the 13 annotated `data/` scenes throughout;
judge parity of every diagnostic number was verified integer-for-integer against
the real CLI.

## The idea, in one paragraph

Run inverse dynamics (RNEA) once over the frozen kinematics_1 trajectory with
nothing touching the body. The six root rows of the result are force and torque
the body cannot produce itself — with no contacts they are simply the body's
weight, ~600 N, unexplained. Solve for world forces at the 12 contact frames
that explain the residual, writing each force's effect as `Jᵀf` (same tangent
space as the residual — no moment arms, no fext bookkeeping). Every frame's
force is confined to a 45° friction cone about the direction that patch of skin
faces, so a frame can only explain what it is physically oriented to explain —
that is what breaks the old "any limb absorbs anything" degeneracy. Force on a
drifting frame is expensive (stillness price, from the persisted pre-enforcement
`drift_pre`), force far from the wall is expensive (gap price, **relative** to
the step's nearest frame — the floater rescue), and "no scene point anywhere
near" is missing evidence, not distance: such frames are not candidates at all.
A frame carrying > 5% of body weight for ≥ 3 steps is in contact; the label is
unioned onto the geometric one, then at-most-one-toe-per-foot is enforced with
the larger force winning. The solve is a convex QP over nonnegative cone
coefficients (projected Adam, KKT readout ~1e-3), ~1 s per scene.

## Headline (shipped defaults, real judge, full 13)

| | P | R | F1 |
|---|---|---|---|
| baseline (near+still readout) | .869 | .340 | .488 |
| **force ∪ gates (shipped)** | **.858** | **.623** | **.721** |

All 13 scenes improve; none regresses. The three floaters, which are the whole
point: gtIGgs .015 → **.807**, vcRBT .090 → **.789**, 9Rvr .257 → **.798**
(force-only precision on them .84–1.00). sport_climb (unmodeled rope force)
improves .143 → .305 but only ~25% of its residual is explainable — reported
separately everywhere. Knee frames are excluded from the force readout (their
force labels are ~all false positives; knee rows stay byte-identical to
baseline, still an open problem). Toe both-fire: 0 by an explicit post-union
exclusivity pass (the union had quietly resurrected 838 both-fire steps;
exclusivity outranks add-only — "at most one toe per foot" is a statement about
the world, add-only was only a policy).

## What the ablations settled (clean re-runs after the sentinel fix)

- **Relative gap pricing** is the single most load-bearing choice: .700 vs .626
  absolute (and it is what unlocks the floaters). **No candidate distance cap**
  second: .700 vs .658 capped at 0.30 m.
- **Cone axis**: the frame's own (negated) skin normal .700 vs sign-corrected
  scene normal .702 — a tie; frame axis ships because it needs no scene cloud
  at all. The *unsigned* scene normal (.617) is the only refuted variant:
  climbers pull on holds, the raw wall normal is the wrong push direction.
- **Motion matters a little**: v=a=0 costs .010. (Phase A had reproduced the
  old "static allocator" null — that turned out to be an artifact of the
  unknown-distance sentinel bug below.)
- **Per-joint effort rows (λτ)**: neutral (.719 off / .717 on) — the
  limb-locality idea did not pay; ships wired at 0. What does matter is scaling
  the root *torque* rows into force units with a 0.2 m lever arm.
- θ (fraction of body weight): clean monotone P↔R trade (.01 → P .71 R .84;
  .05 → P .86 R .62; .10 → P .86 R .48). Shipped 0.05.

## The three controls that make the numbers trustworthy

**Attribution (is it physics or geometry in disguise?).** The gap price uses
distance, so this had to be asked. Best physics-free controls at the same
min-span, same union: loose gates (d<0.30, still<0.12) F1 .671 at P .722;
relative-gap thresholding .665 at P .832. Force dominates the whole frontier
(at matched precision: R .627 vs .554) and beats the very relative-distance
signal it prices against threshold-free (AP on gate-missed cells .707 vs .570).
Honest nuance: on gtIGgs specifically a pure relative-distance gate ties the
force rescue — the *relative-gap idea* unlocks that scene; force's distinctive
contribution is precision at high recall (vcRBT and 9Rvr are force-won).

**Circularity (is the witness an echo of the labels that shaped the body?).**
The kinematics_1 body was itself attracted and damped by geometric auto-labels,
so force run on it could inherit them. On pure-visual bodies (kin1 with
`ann_attraction = 0` — no contact term ever fired; 4 scenes), gates collapse to
F1 .448 while **force-only reaches .819 at precision .934**, floaters
.011/.020 → ~.81. The witness is independent of the geometric labels — and
*stronger* where they never acted. This is the closest thing to "force is the
main metric, gates auxiliary" made measurable.

**Held-out (the recipe was tuned on the 13).** Recipe frozen, then scored on 11
never-tuned annotated DB scenes (5 source videos, per-source macro reported):
gates .789 → force ∪ gates **.837** pooled, macro .725 → .775, 3 of 5 sources
improve. The gain is smaller than in-sample because the held-out geometry is
mostly good — the baseline already scores .789 — and the witness pays where
geometry fails: the one poor-geometry held-out scene goes .519 → **.801**. Two
sources regress on precision (force over-fires on already-solved bodies).
Caveat: both control farms were produced one config revision behind the shipped
stage (λτ=1, pre-exclusivity) — the folded scorer is insensitive to both
deltas, but regenerate before quoting either as a shipped result.

## Toe pair

The solve picks exactly one toe on ~75% of loaded steps and disagrees with the
shipped normal-dot tiebreak on ~44% of them — but the disagreement is *not*
concentrated on penetrating toes (.42 penetrating vs .48 outside), so the
user's hypothesis that the normal tiebreak fails specifically under penetration
is not supported by this data. The annotations fold both toes into "foot", so
which rule is right needs an eyeball in the viewer, not the scorer. In the
shipped stage force wins conflicts by construction (larger |f|, ties keep
toe_under).

## External review

An adversarial codex review (`~/.claude/handoffs/bvr_force_review.codex.out`)
confirmed the math core (residual sign, jacobians, inscribed pyramid, BW
rescale, judge fold) and forced four real fixes before the numbers above:
the unknown-distance sentinel (`finfo.max` slipping past `nan_to_num` made the
relative-vs-absolute ablation partly measure the encoding of "no scene data");
a true projected QP instead of a softplus reparameterization (with a KKT
convergence readout); dimensionless pricing (the "work" price renamed to what
it is — a stillness price); and the held-out split itself.

## Evidence

Trees: `out/_expK_v2/` (shipped config, 13 scenes), `out/_expK_heldout/` (11
farms, zero DB writes), `out/_expK_purevis/` (4 pure-visual farms). Logs +
ablation tables: `scratch/force_recovery/`. Scripts:
`scripts/diagnostics/force_recovery.py` (ablations/controls),
`force_heldout.py`, `force_circularity.py`. Tests: `tests/test_forces.py` (12).

## Open problems after this round

1. Knees: still 0 TP corpus-wide; force fires on them at P .05–.12 and is
   excluded — nothing yet predicts a real knee contact.
2. right_ankle has zero annotated positives corpus-wide yet collects FPs in
   both arms — the heel→ankle fold or the annotations need a look.
3. sport_climb needs a rope force (or exclusion) to mean anything physically.
4. The force witness *detects* contacts on floaters; making the joint still
   (the user's stated end goal) is the next stage's job — nothing here moves
   the body yet.
