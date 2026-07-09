# UPDATE — round 10: seat the flying bodies

*2026-07-09, rev 12. Replaces the round-9 plan entirely. Round 9 closed as a
set of honest nulls; this round first cleans up the debris it left, then goes
after the actual problem — bodies that hang in the air in front of the wall
must end up on the wall. Rounds 1–9 live in git history.*

## What went wrong in round 9 — and what this round is actually about

Round 9 was my mistake twice over, and the executing agent faithfully
executed both mistakes:

1. **Wrong failure model.** I aimed at boundary artifacts — bad pixels at
   the edge of the person mask. Measurement showed the truth: the depth
   error is *region-level*. VGGT places a whole hand, a whole limb,
   sometimes the whole person at the wrong distance from the camera. Edge
   filters and point weighting cannot touch that.
2. **Wrong yardstick.** I judged everything by pooled contact F1 on the
   labeled clips, where this failure barely lives (the measured ceiling was
   +0.013 recall). The round chased crumbs by construction.

What you actually want, restated so we can check I understood: **the flying
body is the problem.** VGGT's human depth is stretched along the viewing
ray. We should trust it *less* — stop letting it dictate where the body sits
in depth — but not throw it away, because it still knows things reprojection
cannot see, like which limb is in front and roughly how far. If this is
still not your meaning, the sentence to correct is this one.

## The reframe in one paragraph

Split what the human depth cloud is used for. Its *relative* structure (the
arrangement of limbs in depth) stays as evidence. Its *absolute* position
(how far the whole thing sits from the camera) gets demoted — that is
exactly the part VGGT gets wrong. Absolute placement instead comes from the
two witnesses we can trust: the scene cloud (walls are big, static and
textured — VGGT does them well; the round-7 audit measured the human cloud
floating *relative to a credible wall*), and the one fact true of every clip
in this project: **a climber on the wall is touching it with something.**
The deliverable is flying bodies visibly seated on the wall, verified on the
scenes that are actually broken — not a decimal on a pooled metric.

## Cleanup first

Everything round 9 built that didn't work gets removed, so the codebase
carries no dead experiment weight:

- the uncommitted T2a corridor code (discarded);
- the three landed-but-off knob sets: the robust-chamfer knobs, the
  cloud-side weight knobs (including the `edge_dist_px` field fuse started
  writing), and the scale-stage weighting/edge flags;
- the `out/_round9/` experiment trees and logs, and any redirected corpus
  scratch.

One thing stays, deliberately: `diagnose_depth_streaks.py`. Its go/no-go
*yardstick* is retired with the rest, but the instrument itself measures the
disease (how stretched the cloud is, per limb, per frame) and round 10's
diagnosis reuses it. Say the word if you want it gone too. Everything
deleted stays recoverable from git history.

## The new yardstick

The round-7 audit already found the target population: ~80 flagged floating
bodies on the corpus, two-thirds of them "body faithfully follows a human
cloud that floats off the wall". The eval set is drawn from there: roughly
15–20 flagged scenes plus healthy controls, plus your 7 showcase scenes,
plus the 5 labeled clips.

- **Primary: seating.** Per frame, how far the body's hands and feet are
  from the scene surface (this number already exists — the contacts stage
  computes it). Headline numbers: the fraction of frames with at least one
  end-effector within touching distance, and the median gap on flagged
  scenes. Success means flagged floaters move from tens of centimetres to
  touching distance, while healthy controls do not move.
- **Guards, not targets: the labeled clips.** Contact precision and FPR must
  not degrade (recall going up is welcome, but it is not the goal).
- **Eyes, mandatory.** Before/after renders on the showcase scenes and a
  sample of flagged scenes go into the results file. You own the verdict.

One honesty note baked into the design: the new wall-contact term pulls
bodies toward the wall, so seating metrics improve partly *by construction*.
That is why adjudication rests on the independent legs — the labeled-clip
precision/FPR guards, the cloud's own limb structure fitting better after
the release, and your eyes.

## The plan

1. **Clean** (above), and close one bookkeeping item: why the re-derived
   baseline recall (0.645) differs from the round-4 record (0.678) — explain
   or record, time-boxed, no rabbit-holing.
2. **Diagnose the disease on the bodies that actually have it.** On ~10
   flagged scenes plus competition_2: measure the along-ray offset between
   body/cloud and wall, frame by frame. Is the offset roughly constant per
   clip, or wandering per frame? Does the cloud's limb arrangement remain
   usable once the offset is removed? Is the wall credible where the body
   should sit? These answers set the knobs of step 3 before any loss is
   written.
3. **Two levers, built to work together:**
   - **Release** — make chamfer offset-tolerant: each frame, estimate the
     whole-cloud depth offset robustly and subtract (a dial from "today's
     behaviour" to "fully relative"). The cloud then still shapes the limbs
     but no longer drags the body to its wrong absolute depth.
   - **Re-anchor** — a wall-contact prior: each frame, the body's closest
     hand/foot should be at touching distance from the scene; a saturating
     penalty on the gap, so airborne moments and hopeless frames cost a
     bounded amount instead of dominating.
   Released depth with no new anchor drifts; a new anchor fighting the old
   chamfer stalls — so the experiment is factorial: baseline, release only,
   anchor only, both, and the dumb-but-honest control arm (just turn chamfer
   down and add the anchor). The fancy version has to beat the dumb one.
4. **Judge on the eval set** (seating + guards + eyes), then wrap: winner
   confirmed, docs synced. Whether to re-run the full 361-scene corpus with
   the winner (~7 h) is your call at the end.

## Risks, stated up front

- **Self-confirmation** is the standing trap: any pull toward the wall
  inflates recall and seating for free. Precision, FPR and your eyes are the
  honest signals, and the plan treats them that way.
- **Airborne frames exist** (dynamic moves, falls). The anchor is
  deliberately saturating and per-frame; it prefers touching but can lose
  the argument on a frame where the image evidence says otherwise.
- **There is no perfect witness for absolute depth here.** On the broken
  frames, VGGT is wrong and SAM-3D's own depth proved unreliable too (round
  9 measured both). "On the wall, limbs plausible, contacts matching the
  annotations where we have them" is the operational definition of correct
  this round — that is a choice, and it is yours to veto.
