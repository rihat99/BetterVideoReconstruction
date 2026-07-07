# UPDATE — round 6: the last modeling lever, then wrap up

*2026-07-07, rev 8. Covers: betas optimization inside `kinematics_1` (the final
experiment), a competition_2 autopsy, confidence calibration, a small weight
ablation, the real cleanup, and corpus readiness. Rounds 1–5 history: revs 4–7
in git.*

## Round 5 — what it left us

Five commits landed (`14f1bac`…`312fa4f`). One move shipped, three returned
honest nulls:

- **Shipped**: the per-joint label confidence, both classes (`86b00a1`) — the
  deliverable for learning from these labels later.
- **Null**: robust betas aggregation + scale erosion (reverted). The short-limb
  effect in the *predicted* betas is only 1–5 mm — far too small to explain a
  ~40 mm hover. Important nuance: aggregation can only pick among what the
  per-frame model already predicted. If every frame under-predicts the limbs,
  no averaging scheme can fix it. That is exactly why round 6 goes one level
  deeper (below).
- **Null**: kinematics_2 on auto labels (kept as an off-by-default capability).
  Auto labels only reinforce contacts we already found — they cannot create the
  missing ones — and kinematics_1 already damps them. It did fix a real latent
  bug (the honest drift now carries into kinematics_2's output).

Numbers are unchanged: **P 0.896 / R 0.678 / F1 0.772 / FPR 0.066** pooled.

## The confidence recheck you asked for

I re-read the shipped code line by line and verified it numerically on the
real outputs. **Verdict: it computes correctly** — no sign, unit, or indexing
bugs; it uses the honest pre-enforcement drift and the final post-merge
labels; and on real data it behaves exactly as designed: no-contact confidence
rises smoothly with joint speed (0.00 → 0.96 across speed deciles), contact
confidence falls with distance to the wall, near-wall slow "hover" joints
correctly get ~0.01 (the honest unknown), fast joints get 0.94–0.99.

Two *calibration* gaps, though — not bugs, but places where the shipped
behavior is narrower than what you asked for:

1. **The "far ⇒ confident no-contact" rule almost never fires.** The distance
   branch is gated by two radii that nearly touch: the signed distance is only
   measured out to 10 cm (beyond that the joint reads "no scene support"), and
   "did we see the space" only counts scene points within 15 cm. So the far
   branch can only trigger in the thin 10–15 cm shell. A slow hand 40 cm from
   the wall — your own example of an obvious no-contact — currently scores ~0,
   indistinguishable from an occluded joint. In practice no-contact confidence
   is motion-and-persistence only. Round 6 fixes this with the *unguarded*
   distance channel we already save, with the "far" threshold set beyond any
   plausible hold protrusion (~25–30 cm) — so a hand touching a hold that was
   carved out of the cloud (nearest reconstructed wall 10–20 cm behind it)
   still reads *uncertain*, not confidently-free.
2. **Bridged and edge contact frames carry near-zero confidence.** Inside a
   merged gap the joint can be up to 8 cm off the wall, which zeroes the
   closeness factor — so the very frames the span-merge deliberately bridged
   get confidence ≈ 0 (9% of contact labels on boulder_1, 29% on
   competition_2). Honest, but a learner can't tell "bridged by strong span
   evidence" from "marginal". Fix: save a small `merged` flag alongside, and
   let the consumer decide.

## Round 6 — the final iteration

Your call: the one unsolved reconstruction problem is **contacts missed
because the limb is too short to reach**. Scale (human and scene) is
exhausted. The last lever is **optimizing the betas themselves inside the
fit** — letting the 3D evidence (depth cloud, wall) adjust the ten shape
numbers, with a strong leash: stay close to the initial betas. Plus: find out
*why* competition_2 specifically is so bad, and then close out — calibrate,
ablate, clean, and get ready for the large corpus.

## The plan — six moves

**1. competition_2 autopsy (first — it sets expectations).** Before the betas
experiment, a read-only investigation of the one clip where we are extremely
bad: is it limb proportions, or is the VGGT depth on the person so poor that
nothing downstream can be right? Concrete checks, all from existing artifacts:
depth confidence on the person vs the other clips, the scale stage's
diagnostics (how many frames actually contributed), mask and tracking quality,
person size in the image, how far the fitted body sits from the human depth
cloud per body part, and the per-limb hover gaps. Deliverable: a short verdict
— "competition_2 fails because of X" — and a prediction of whether betas
optimization can plausibly rescue it. If depth is the culprit, we say so and
stop expecting miracles from move 2 there.

**2. Betas optimization in `kinematics_1` (the experiment).** One shape vector
per person (not per frame), optimized jointly with the pose, starting from the
converter's betas and pulled back to them by a prior — your regularization
choice. Why this can work where round-5 aggregation couldn't: the optimizer is
not limited to what the image model predicted — the depth cloud and the wall
get a vote. And there is a geometric reason it targets exactly the hover: at a
fixed depth, lengthening an arm *along the viewing ray* barely changes the 2D
projection, so the image anchors don't resist the correction we need; sideways
changes do get resisted, which is what keeps the fit honest. Guard rails: the
prior (grid over its weight, pick the gentlest that closes hover), a report of
how much the betas actually moved — in implied limb-length millimetres and
body-height percent — and a consistency flag if the body's overall size drifts
past ~2% (the scene's metric scale was set using the original body as the
ruler; a large size change would undermine it). Off by default until it wins;
judged on the same gates as always. competition_2 and the funnel's *hovering*
category are the bellwethers.

**3. Confidence calibration (the recheck's two findings).** Re-point the far
branch at the unguarded distance with a carve-safe threshold; save the
`merged` flag; fix the misleading knob comment. Small, readout-only, validated
by the same reliability curves — the slow-far bucket must move from ~0 to
clearly-confident while the hover band stays uncertain.

**4. A small weight ablation.** One factor at a time, ×0.5 and ×2, over the
handful of weights that matter (the two reprojections, chamfer, attraction,
the damper, radial, jerk/snap, pose prior, penetration) — scouted on two
clips, winners confirmed on four. Two possible outcomes, both useful: a
clear improvement (adopt it), or a flat table (the config is robust — worth
documenting before pointing it at thousands of videos). No adoption without
the usual gates.

**5. The real cleanup.** Last round's polish was minor by design; this one is
the deliverable. The deferred items: compress the war-story comments in the
config and the optimizer code to one contract line each; rename the
"prototype" scorer to what it is (`score_contacts.py`) and move its reusable
parts into a proper tools module; shrink the uncertainty stage now that its
old confidence block is gone. Plus the experiment residue: delete one-shot
diagnostic scripts once their evidence is recorded, verify the reverted
round-5 experiments left no dead flags, keep the kinematics_2-auto capability
(one small flag, tested, and it's the future self-training hook — flagged for
your veto). Hard rule, unchanged: behavior-preserving, one clip re-runs
bit-identical, nothing gets slower.

**6. Corpus readiness.** The point of all of it: running this on thousands of
videos. A defaults audit (a bare, flagless run of every stage must equal the
tuned behavior); a DB driver for the contacts + confidence stage (same
symlink-farm pattern as the other drivers); a memory check on the one new
full-cloud pass (the confidence neighbourhood search — prune it if a big DB
scene makes it balloon); a per-stage runtime/VRAM table; and a smoke test on a
handful of DB scenes. The full DB re-run itself stays the separately scheduled
finale (Issue-22) — round 6 makes it safe to press that button.

## How we judge it

- **Betas experiment**: pooled F1/recall must *improve* (that's its only job),
  precision ≥ ~0.88, FPR ≤ ~0.07, reprojection vs raw targets within +0.5 px,
  runtime within ~1.05×, naturalness flat, betas movement reported in mm/%.
  Null ⇒ revert and record, like round 5's T1/T2.
- **Confidence calibration**: reliability stays monotone; the slow-far bucket
  gains confidence; hover band stays low; occluded/carve cases stay uncertain.
- **Ablation**: adopt only clear pooled wins; otherwise document sensitivity.
- **Cleanup**: bit-identical labels, float-identical bodies, runtime ≤ 1.0×.
- **Readiness**: the smoke test passes end-to-end with default flags.

## Honest limits

- Betas optimization is leashed to the initial shape on purpose — it can close
  a few centimetres of systematic limb error, not reinvent the body. If
  competition_2's problem is depth (move 1 will tell us), betas won't save it.
- Occluded contacts stay unrecovered-but-honestly-uncertain; moving contacts
  stay excluded by design. Recall's ceiling still stands where round 4 measured
  it — move 2 attacks the *hovering* slice, nothing else.
- The scale-vs-betas consistency flag is a report, not a fix: if betas want to
  change body size materially, the honest response is to revisit scale — a
  bridge we cross only if the flag fires.

## Order of work

1. competition_2 autopsy (read-only; sets expectations for 2).
2. Betas optimization: implement → grid the prior → gates → adopt or revert.
3. Confidence calibration + reliability re-run (independent; readout-only).
4. Weight ablation (on the post-2 config).
5. Cleanup (parity-gated), then corpus readiness + smoke test.
6. Final tables, leave-one-out, and a closing "round 6 outcome" here. After
   that, durable contracts move into CLAUDE.md / module docstrings, and these
   UPDATE docs become history.

## Still not doing

- No per-frame betas, no second body-scale — one shape vector per person,
  prior-leashed.
- No camera-pose changes; no new losses beyond the betas prior.
- No reduction of smoothness weights.
- The full DB re-run is not part of round 6 (Issue-22, you schedule it).

## Round 6 — outcome

One real improvement shipped; the two modelling experiments were honest nulls,
both reverted and recorded. Commits: `26a471d` (T3), `4649ab2` (T5), `240b97e`
(T6). T2 and T4 changed no code (reverted / not adopted).

- **competition_2 autopsy.** It is bad because the climber is the *smallest in
  frame* of any clip (22% of the height vs 55–58% for the boulder clips), which
  gives the weakest keypoints and the sparsest depth — not because its depth is
  broken (its body-to-cloud error and scale are middle-of-the-pack; a good clip,
  boulder_3, actually has worse body-to-cloud error). So the problem is size and
  observability, and it is exactly the regime where the short-limb bias is worst.

- **Betas in the fit — a null, and worse than neutral.** We let the ten shape
  numbers move during the fit, leashed to their starting value. The more they
  move, the *worse* the contacts get (at the loosest leash, recall drops 2 points
  and limbs lengthen only ~5–9 mm against a ~40 mm gap, while body height inflates
  over 1%). The fit simply does not reward lengthening a limb — a short limb
  already sits on the near part of its depth cloud, so nothing pulls it further.
  Reverted to off; the capability is gone, the finding kept.

- **Confidence far-branch — shipped.** The "this joint is clearly far from the
  wall, so we're sure it's not touching" rule almost never fired, because it read
  a distance that was capped at 10 cm. It now reads the true distance. On the four
  clips the no-contact confidence went from barely-correlated to clearly-correlated
  with being right (+0.25 → +0.41), the confident predictions are 92% accurate,
  and the honest misses stay honestly uncertain. We also save a small `merged`
  flag marking the frames the span-merge bridged. Labels are unchanged.

- **Weight ablation — the config is robust.** Nudging each weight up and down by
  2× moved the pooled F1 by at most +0.005 on the four clips — nothing crossed the
  bar to adopt, and most changes hurt. The chamfer weight is the one that really
  matters (either direction costs 2–3 points). A flat table is the point: the
  round-4 weights are well-tuned, which is what we want before running thousands
  of videos.

- **Cleanup + corpus readiness.** Compressed the round-history comments to
  one-line contracts (the config shrank 205→166 lines, no behaviour change),
  renamed the "prototype" contact scorer to `score_contacts.py`, and deleted the
  betas diagnostic. Found and fixed a memory blow-up: the confidence step builds a
  distance matrix against the whole scene cloud, which would hit ~2.5 GB on the
  biggest dataset scenes (measured up to 380k points) — now bounded to ~1 GB with
  no change to the result. Wrote a dataset driver for the contact stage and
  smoke-tested it on three scenes (zero failures, multi-person handled).

*These UPDATE docs are now historical — the durable contracts live in `CLAUDE.md`
and the per-stage module docstrings.*
