# UPDATE — round 7: corpus floaters, confidence v2, the limb-length lever

*2026-07-08, rev 9. Covers: finding every floating body in the corpus with a
metric, understanding why they float, making the label confidence honest about
them (and more generous on solid contacts), moving the contact / uncertainty
outputs into `human_optim`, and the limb-length experiment. Rounds 1–6 live in
git history (revs 4–8).*

## What you reported, and what I measured today

You ran the full corpus (361 scenes) and saw both extremes: scenes where
contacts, motion, and naturalness are close to perfect, and scenes where the
body floats in the air. You named five floaters. I swept the whole corpus
using only artifacts we already have (the contact labels, the per-joint
distance to the wall, and the per-frame body-to-cloud error), and the problem
is both **bigger than five scenes and very detectable**:

- **21 bodies never touch anything at all** — zero contact frames across
  hundreds of frames of climbing. Another handful sit under 2%. In total **68
  of 363 tracked bodies (19%) have contacts in fewer than 40% of their
  frames**, against a corpus median of 81%.
- All five scenes you named land in the worst fifth by this simple measure,
  and their closest joint hovers 2–16 cm off the wall on a typical frame
  (median body has its closest joint slightly *inside* the wall, −4 cm).
- The five you named are the tip: the never-touch scenes (e.g.
  `gtIGgs_y2nY_0095`, hovering ~65 cm) are even worse.

So the detector you asked for essentially already exists: **contact
starvation** (share of frames with at least one joint in contact) plus
**typical distance of the closest joint to the wall**. What's missing is a
script that reports it per scene, a validated flag threshold, and the *why*
behind each flagged scene.

Two more measurements that answer your confidence complaints directly:

- **Contact confidence is low on average — measured: 0.465.** The culprit is
  not visibility (0.96 on the dense corpus clouds) but the **stillness
  factor**: a gripping hand naturally drifts 2–3 cm over the measurement
  window (micro-adjustments), which lands exactly at the half-way point of
  the stillness ramp. Worse, a joint *penetrating* the wall — the strongest
  possible geometric evidence of contact, and 46% of all contact labels —
  still averages only 0.51, because motion keeps throttling it.
- **On the floating scenes, no-contact confidence averages 0.79, with 74% of
  labels above 0.8 — confidently wrong.** Exactly what you predicted: every
  joint of a floating body is far from the wall, so the "clearly far ⇒
  confident no-contact" rule fires everywhere, on the very scenes where
  distances mean nothing.

## Why bodies float — four diseases, one symptom

A floating body can come from four different places, and they need different
responses. The autopsy task below assigns one to every flagged scene:

1. **Body off its own depth cloud** — the optimizer failed; the per-frame
   body-to-cloud error (already computed corpus-wide) is high. Fixable with
   optimization knobs. I expect this to be rare: on the sweep, the floaters'
   body-to-cloud error is mostly unremarkable.
2. **The person's depth cloud is detached from the wall** — VGGT put the
   climber's surface some distance in front of the scene behind them. The
   body then sits correctly *on its cloud* and still touches nothing. Not
   fixable by any body optimization; the honest response is to flag it and
   lower confidence.
3. **Limbs too short** — body on its cloud, cloud on the wall, but arms/legs
   stop a few cm short of the holds. This is the competition_2 disease, and
   the one lever we haven't fully tried (below).
4. **Broken upstream** — near-empty scene cloud, tracking dropouts, garbage
   masks. One named-adjacent scene has *no* usable wall distance at all.
   Flag, don't fix.

## The plan — six moves

**1. Layout quick fix (first, it's small).** `contacts_1.npz` and
`uncertainty_1.npz` move from their own `features/contacts/` and
`features/uncertainty/` trees into `features/human_optim/<scene>/`, next to
the body they describe — the same layout local clips already use
(`out/<stem>/human_optim/`). Migrate the 361 existing files, point both DB
drivers there, and delete the viewer workaround that currently stitches the
two trees together.

**2. The floater detector + corpus audit.** A small script that reads the
existing artifacts and writes one row per (scene, person): contact rate,
typical closest-joint distance, share of frames plausibly seated on the wall,
body-to-cloud error, plus upstream health (cloud size, tracking coverage).
It flags suspects (starting rule: contact rate < 0.4, or typical closest
joint > 5 cm off the wall — thresholds validated against your five and a
visual sample). Deliverable: a ranked CSV + the flagged list you can walk
through in the 3D viewer to confirm.

**3. The autopsy.** For every flagged scene, decide which of the four
diseases it has, using four probes: body-to-cloud error (have it), person
cloud-to-wall gap (new, cheap), limb-length mismatch (below), upstream health
(have it). Deliverable: a cause label per flagged scene and the counts. The
counts decide where fixing effort goes — and which scenes we honestly
declare unfixable-but-flagged.

**4. Confidence v2.** Two calibrated changes, validated on the four
hand-labeled clips and then re-scored over the corpus:
- **Strong geometry beats motion doubt.** A penetrating or firmly-touching
  joint stops being throttled by the stillness factor (the label already
  passed a stillness gate at detection — punishing the same drift twice is
  why solid grips score 0.5). Target: penetrating contacts go from ~0.51
  mean confidence to ≥0.75, without hurting the reliability curve.
- **Floaters lose their false confidence.** The "clearly far ⇒ confident
  no-contact" branch gets multiplied by a per-body health factor computed
  from the same signals as the detector (a body that almost never touches
  anything, or hovers everywhere, doesn't get to be sure about distances).
  Motion-based confidence survives — a fast-moving joint is a confident
  no-contact even on a broken reconstruction. Target: no-contact confidence
  on flagged floaters drops from ~0.79 to ≤0.4; healthy scenes unchanged.

**5. Limb length — measure first, then maybe fix.** The one modeling lever
left, done in a way that respects round 6's null (the fit does not *discover*
longer limbs on its own — anything we do must be driven by a measured
target):
- **Measurement (read-only, corpus-wide).** MHR — the body model SAM-3D
  predicts *before* our SMPL-X conversion — already estimates per-bone
  skeleton scales (28 numbers per frame; I verified they are in the
  conversion target). So we can measure, for every scene: the limb lengths
  MHR believed, the limb lengths that survived the SMPL-X conversion, and —
  as an independent third opinion on a few clips — limb lengths read from
  the VGGT depth at the 2D keypoints. This tells us *where* length is lost:
  conversion (fixable at the converter, helps every scene at once) or MHR
  itself under-predicting (then only scene evidence can help).
- **Fix (only if the measurement says ≥2 cm systematic deficit on flagged
  scenes).** Depending on where the loss is: preserve limb lengths through
  the conversion, or add four symmetric per-person limb-extension parameters
  (upper-arm, forearm, thigh, shin — left/right shared, as you said) to the
  kinematic fit, pulled toward the *measured* target lengths and leashed to
  zero. Judged like every experiment: pooled contact metrics on the tuning
  clips, plus contact rate on your five scenes, plus your visual check.
  Null ⇒ revert and record.

**6. Wrap.** Re-run the two cheap readouts (contacts + uncertainty, ~20 min
total) over the corpus with the final confidence; sync docs; outcome section
here. A kinematics re-run of the corpus only happens if move 5 ships a fit
change — that's your call to schedule, like Issue-22 was.

## On your ideas (you asked me to be skeptical)

- **"MHR can extend limbs as translational joints" — real, and better than I
  expected.** SAM-3D genuinely predicts 28 per-frame skeleton scales, and the
  converter poses MHR *with* them before fitting SMPL-X — so limb-length
  information exists upstream and betas may simply be unable to keep it.
  That's why the measurement comes first: if conversion is where length dies,
  the fix is cheap and benefits every scene without touching the optimizer.
- **"Use VGGT limb lengths as a loss"** — I'd keep VGGT as the *validator*,
  not the primary target. Depth at a 2D keypoint measures the body's front
  surface, not the joint center, is noisy at exactly the small-in-frame
  scenes we care about, and vanishes under occlusion. MHR-vs-SMPLX is clean,
  already on disk, and per-frame. If the two agree limbs are short, that's a
  strong go signal; if they disagree, VGGT noise is the first suspect.
- **"Merge scale estimation with limb scale, maybe on MHR"** — deferred, not
  dismissed. Global scale and limb proportions are related but separable:
  scale is set by the whole-body depth ratio (torso-dominated), limbs are a
  residual on top. Round 5 measured that scale changes can't close per-limb
  hover. If the limb measurement shows the deficit is big enough to bend the
  scale ruler (>~2%), we revisit — same flag as round 6 had.
- **"When body floats, far-from-wall confidence should be lower" — agreed,
  and it's the right fix.** The alternative (trying to *repair* floaters in
  the optimizer with stronger scene attraction) is the round-2 circularity
  trap: it manufactures contacts on bad geometry. Detect, flag, downweight.

## How we judge it

- **Detector**: your five all flagged; a visual sample of flagged scenes
  confirms floating; a sample of unflagged mid-pack scenes confirms health.
- **Confidence v2**: reliability curves on the labeled clips stay monotone
  and at least as accurate in the top decile; penetrating-contact confidence
  ≥0.75; floater no-contact confidence ≤0.4; labels themselves bit-identical
  (confidence is a readout).
- **Limb fix, if attempted**: pooled P ≥ 0.88, FPR ≤ 0.07, reprojection vs
  raw within +0.5 px, runtime ≤1.05×, contact rate up on the five named
  scenes, and your eye on the motion.
- **Everything**: one commit per task, gates before merging, honest nulls
  recorded.

## Honest limits

- Floaters from detached or broken depth (diseases 2 and 4) do not get fixed
  this round — they get found, explained, and their confidence muted. That
  is the correct outcome for a label corpus.
- A low contact rate is suspicious, not damning — a fall, a dyno, walking to
  the wall all lower it legitimately. That's why the flag threshold gets
  validated visually before we trust the list.
- The limb experiment may return a null like betas did. The measurement step
  exists precisely so we can kill it cheaply if the deficit isn't real.

## Order of work

1. Layout move (small, unblocks everything else touching those files).
2. Detector + corpus audit CSV → you eyeball the flagged list.
3. Autopsy → cause counts.
4. Confidence v2 → labeled-clip validation → corpus readout re-run.
5. Limb measurement → go/no-go → experiment if go.
6. Wrap: docs, outcome, and (if 5 shipped) you schedule the corpus
   kinematics re-run.

## Outcome — round 7, 2026-07-08

All six moves landed. Five commits on `main` (T1–T4 plus this wrap); the limb
experiment was measured and dropped.

**Layout (T1, `ecaeb0d`).** The contact and trust readouts now sit next to the
body they describe, in `features/human_optim/<scene>/`, the same as local clips.
722 files moved, the two old trees removed, the viewer's stopgap deleted.

**The floater detector (T2, `487e567`).** `scripts/audit_scenes.py` reads what is
already on disk and scores every body. It flags 80 of 363 bodies (22%) as
floating — all five you named included. The bar had to drop to a 0.50 contact
rate to catch the mildest (`PfwFWSoPF_k_0026`); the plan's claim that the seed
bar already caught all five was wrong. It also settled the "buried behind the
wall" scare: 41 of 51 such bodies are in contact 91% of the time, so the distance
sign merely flipped — they are seated, not behind the wall.

**Why bodies float (T3, `7a6796f`).** `scripts/autopsy_scenes.py` gives each
floater a cause. Two-thirds (65%) sit correctly on their own 3-D reconstruction,
but VGGT placed that reconstruction off the wall — nothing the body fit can
repair. Another 22% are the body sitting off its own cloud (an optimizer slip,
in principle recoverable). 10% are broken upstream. Only 2% are short limbs.

**Limb length (T5 — measured, then closed as a null).** You asked whether short
limbs cause the floating. They do not. The SMPL-X conversion does lose limb
length versus MHR — about 2 cm in the forearm, 5 cm in the thigh — but the loss
is *uniform*: floaters are no shorter than healthy bodies. And the pattern (a
short thigh but a slightly long shin) points to the two skeletons defining the
knee differently, not to a real length loss. So we measured it, recorded it, and
did not build the fix — it would not have moved the floaters.

**Confidence v2 (T4, `596ab76`).** Two honest calibrations, both readout-only
(the contact labels are untouched, bit for bit):
- A firmly *penetrating* joint is the strongest possible sign of contact, so
  motion no longer throttles its confidence. Penetrating contacts went from 0.51
  to 0.85 mean confidence.
- A body that essentially never touches the wall in a climbing video has been
  floated off it, so its "clearly far, so confidently not touching" is muted
  toward zero. Those confidently-wrong floaters dropped from 0.79 to 0.35.
  Healthy bodies are unchanged (293 of 363 are untouched by the gate) and the
  reliability curves on the labeled clips are identical to before.
The whole corpus was re-scored with the new confidence (361/361, 16 min).

**Not done, on purpose.** No kinematics re-run — nothing changed the body fit
(the limb fix was dropped). The unfixable floaters (detached or broken
reconstructions, three-quarters of them) are now *found and their confidence
muted*, which is the correct outcome for a label corpus.
