# Round 12 — fixing the depth at the root (30 new labeled scenes)

*Round-13 note (2026-07-12): the gated de-smear this round shipped was REMOVED in
round 13 — re-measured at +0.006 pooled F1 on the harder test set with its only
growth path (per-limb licensing) proven capped at R ≤ 0.446. The limb-quota
sampling stays. This document remains as the historical record; the round-13 story
and the replacement mechanism (the fuse body anchor) live in `docs/UPDATE.md`.*

*Started 2026-07-10. This is the running log of the round: what we measured,
what we tried, what worked. Written to be read top to bottom in plain language.
The goal, in your words: increase contact accuracy after human optimization by
fixing the root cause — VGGT-Omega's human depth being stretched along camera
rays — not by pushing bodies onto walls. Target: precision ≈ 0.9, recall
0.7–0.8 on the new test set. Manual labels are used only for scoring, never in
optimization.*

## The new test set

You labeled all 30 test-split scenes of the ClimbingVideos corpus (14 joints,
tri-state). I copied everything into the repo so experiments never touch the
canonical corpus:

- `data_2/<scene>.mp4` + `data_2/<scene>.npz` — the clip and your annotation
  (same file format as the old 5-clip `data/` annotations, 4 extra joints:
  ankles and shoulders are now their own labels, and "foot" now means the toe,
  not ankle+toe).
- `out_2/<scene>/` — full per-scene pipeline results (sam3, vggt, sapiens,
  sam3d, geometry, human_optim), copied from the corpus feature trees. 7.3 GB.

All 30 scenes are single-person, annotation frame counts match the bodies
exactly, ~97,000 labeled joint-frames of which ~28,000 are contacts. The
contact scorer is being adapted to the 14-joint format (the old 10-joint
clips in `data/` keep scoring identically).

## First measurements (before changing anything)

**How far from the wall are joints the annotator says are touching?** Using
the per-joint distances already stored in `contacts_1.npz` (current pipeline
output, body = kinematic_1):

- Pooled over all 30 scenes: 70% of annotated-contact hand-frames are already
  within 3 cm of the wall; 67% of foot-frames. But ~18–20% are more than
  10 cm away — those are the depth disease, and no contact-readout tuning can
  ever recover them.
- The disease is **concentrated and one-sided**. Roughly a third of the scenes
  carry almost all of it, and within a scene it usually hits one limb group:
  `CRquJ8H1QLI_0061` has hands touching (−2 cm) while feet hover at +14 cm;
  `33ODIivxhF8_0025` is the reverse (hands +21 cm, feet fine). Fully-diseased
  scenes exist too (`CRquJ8H1QLI_0001`, `_0008`, `_0009`, `s-ArwEzr-2M_0021`,
  `ecx75KwBCV8_0017`). This kills any per-frame whole-body correction — the
  fix has to act per limb.
- **Recall tracks geometry.** Scenes whose contacts sit within 3 cm get good
  recall from the current readout; hovering scenes get almost none. The
  pipeline's problem is where the body ends up, not how contact is read out —
  with one exception below.
- **A suspicious readout failure**: on the `s-ArwEzr-2M` scenes, feet have
  29–45% of annotated contacts within 3 cm of the wall, yet foot recall is
  0.02–0.05. That is not geometry — something in the readout (stillness gate,
  guards, hysteresis) is eating feet that are actually on the wall there.
  Being investigated separately; you warned there might be dormant bugs, and
  this smells like one.
- **The global scale is not the problem.** The per-frame body-vs-cloud scale
  ratio (stored by `estimate_scale`) is stable within each clip (2–4% spread).
  The error lives at limb level, along the ray.

**Who is closest to the wall on diseased contact frames — the final body, the
init body, or the depth cloud itself?** (This decides whether "squeeze the
cloud toward the init body" can work.) On the worst scene (`CRquJ8H1QLI_0001`),
for annotated-contact frames:

- final body hand → wall: 17 cm. Init (SAM-3D) hand → wall: 23 cm — the init
  body floats *more* than the final body. On diseased frames across this
  scene, the init body is within 5 cm of the wall only 1–3% of the time.
  **The init body is not a usable depth anchor**, at least not here — consistent
  with the round-9 finding that the init depth corridor fails exactly on bad
  frames.
- the limb's own depth cloud has a deep tail that reaches much closer to the
  wall (10th percentile 6–12 cm) than either body. The "hand is at the wall"
  information *is present in the cloud* — the stretch dilutes it rather than
  erasing it.

**Full 30-scene version (pooled over diseased contact limb-frames, i.e. final
body > 5 cm off the wall):**

| limb | final body → wall | init body → wall | limb cloud's deep tail → wall | init within 5 cm |
|---|---|---|---|---|
| hands | 12.9 cm | 19.8 cm | **4.9 cm** | 4% |
| feet | 14.3 cm | 18.9 cm | **6.1 cm** | 3% |
| ankles | 11.7 cm | 17.3 cm | 7.7 cm | 6% |

Two conclusions, both corpus-wide now:

1. **The init (SAM-3D) body is not a depth anchor.** On diseased frames it
   floats even further than the final body, essentially always. So the plain
   version of "squeeze the cloud toward the init body" cannot seat contacts —
   it would move the cloud the wrong way. (It could still *stabilize* limbs;
   see the sliding-feet finding below.)
2. **The limb cloud's deep (wall-side) tail is at the wall.** The stretch is a
   smear from a floating front end back to the wall; the body settles
   mid-smear. The contact information survives in the cloud — diluted, not
   erased.

Caveat being tested right now: the deep tail could be an artifact — wall
pixels leaking through the mask edge, or VGGT interpolating human depth toward
the background. The falsification test is to measure the same tail on
annotated-FREE limb-frames: if the tail hugs the wall there too, it is leak
(following it would invent contacts); if free-frame tails hover, it is signal.

**The sliding-feet discovery (a second disease mechanism).** On the
`s-ArwEzr-2M` scenes, feet that ARE within 3 cm of the wall get almost no
recall because the stillness gate vetoes them: the reconstructed foot carries
9–20 cm of drift per 1/3-second window on frames where the annotator says the
foot is planted. Decomposition shows this drift is smooth (millimetre-level
frame-to-frame jitter) and both along-ray AND sideways — the reconstructed
foot *slides* while the true foot is planted. Cause: feet are
evidence-starved — weak/occluded 2D keypoints, and only a handful of the
1,500-point per-frame human-cloud budget lands on a foot — so the foot gets
dragged along by the climbing body's motion instead of being pinned. This is
not a gate bug; the gate is honestly reporting that the reconstructed foot
moves. The fix has to give feet real evidence (more cloud points on limbs, a
firmer chamfer grip), not a looser gate.

So the disease decomposes into three mechanisms, each with its own fix family:

- **A. Limb cloud stretched/offset along the ray** (VGGT wrong) → body hovers.
- **B. Limb evidence starvation** (few cloud points + weak keypoints on feet)
  → limbs slide with the body, stillness gate rightly vetoes → recall lost at
  the wall itself.
- **C. Init body depth wrong** (SAM-3D translation error, up to 20–30 cm) →
  the init anchor + pose prior hold the body at a wrong depth, and nothing in
  the current loss can overrule it when the cloud is also bad.

## The baseline on all 30 scenes

At the shipped operating point (contact if within 3 cm and still, `still=0.03`):

- All 14 joints pooled: **P 0.833, R 0.533, FPR 0.041**.
- FOCUS joints (hands+feet+ankles, the ones that matter): **P 0.885, R 0.520,
  FPR 0.085**. Your target is P ≈ 0.9, R 0.7–0.8.
- Removing the stillness gate entirely only reaches R 0.662 at P 0.732 — so no
  amount of gate loosening gets near the target. The bodies themselves must
  get better.

Where the missing 48% of contacts go (miss funnel, pooled): **18%** hover off
the wall (the depth disease directly, half of them beyond even a 15 cm search),
**12%** are vetoed as "moving" (the sliding-limb mechanism), **13%** sit on
spans the in-loop contact detector never labeled, 2% have no scene points at
all, ~0% flicker.

## Tail test and the de-smearing null

The falsification test came back: on annotated-**free** limb-frames the cloud's
deep tail sits at 9–13 cm from the wall vs 4.9–6.1 cm on contact frames. So the
tail IS informative (free vs contact separates), but noisy — following it
blindly would buy recall with a precision bill.

I then scored six cloud "de-smearing" rules offline on a 6-scene dev split
(trim the streak's camera-side front, keep the densest depth cluster, keep the
deep band, etc. — fixed physics-motivated parameters, no optimizer in the
loop): the best rule only moves the streaked contact core from 16.5 cm to
11.7 cm off the wall, recovering a handful of frames while tripling the free
false-positive proxy. **Conclusion: the limb cloud's central mass is genuinely
at the wrong depth — no reshaping of the same bad data recovers the truth.**
The wall-side information lives in a sparse tail only. This kills pure cloud
squeezing as the primary fix and promotes an *independent depth witness* to
the front of the queue.

## The plan that follows from the evidence

1. **Feed the limbs (running now).** Density-stratified sampling of the human
   cloud (voxel quota instead of uniform stride), so hands and feet keep their
   points instead of being drowned by the torso. Opt-in knob, bit-exact when off.
2. **Stop the wrench (running now).** A robust chamfer kernel — plain
   saturating (PROX-style) and an anisotropic variant that splits each residual
   into along-ray (saturate hard: distrust depth) vs transverse (keep: limb
   location is the part of depth we still trust — your "get some info from it,
   like limbs location"). Opt-in knobs.
3. **Sapiens pointmap as a second depth witness (evaluating now).** A
   human-specialist model reports human depth independently of VGGT; published
   numbers say it beats VGGT on people. If its limb-relative depth is right
   where VGGT streaks, it becomes the anchor the init body failed to be.
4. **Physics feasibility (next wave).** A still, unsupported body is
   impossible; use the dynamics residual to *detect* frames whose depth must be
   wrong, and only there arbitrate corrections. Your idea 3 — kept as the
   arbiter rather than the primary pull, so it can't manufacture contacts.

## The adversarial review (and what it changed)

An external model (codex) attacked the diagnosis and plan. Its confirmed
catches, all adopted:

- **Scoring bug**: the production contact stage gates on the *refine-start*
  drift (so the optimizer's own damping can't fake stillness), but I had been
  scoring with final-body drift. All scoring now uses the production gate.
  The honest baseline moves to **FOCUS P 0.885 / R 0.550** (close=3 cm,
  still=0.03). Widening the readout to its friendliest setting only reaches
  R 0.64 at P 0.85 — knobs still can't reach the target.
- **The first implementation of "feed the limbs" was geometrically wrong**: a
  3D voxel equalizer *preserves* ray streaks (they span many voxels) and
  decimates compact correct limbs. Replaced with screen-space semantic quotas
  (each hand/foot keeps up to N points, assigned by nearest projected init
  joint), and the confidence gate becomes per-group — the old global gate was
  eating depth-edge limb points before any quota could act.
- **The first robust-chamfer kernel changed units** (squared meters vs the
  linear-meter baseline → silently ~10× weaker), and its transverse sigma
  saturated the *trusted* axis first. Replaced with a unit-preserving
  saturating kernel on the along-ray component only; transverse stays raw.
- Honesty limits: the 30 scenes are now *validation*, not test (they've been
  used to find failures and pick designs — 20 of them even share one source
  video); recall gains must be checked against penetration metrics (a body
  pushed inside the wall scores as "contact" today); robust chamfer can only
  stop the wrench, it cannot recover deep hovers — saturating the only depth
  term with no replacement anchor risks bodies reverting toward the (worse)
  init.

## The Sapiens pointmap witness: a real null

The human-specialist depth model was the best candidate for an independent
along-ray anchor. Measured on 7 dev scenes: **its pointmap renders the climber
nearly flat** — hands at torso depth (relief +2 to −11 cm) where the truth on a
healthy scene is +7 to +26 cm (arms reaching to the wall). This holds both
full-frame and on proper person crops, so it is not a resolution artifact; a
climber reaching overhead is simply out of distribution for a model trained on
standing studio scans. The witness idea is dead on this data. (The pilot cost
one hour; wiring it blindly into the optimizer would have cost days.)

With init refuted, cloud reshaping near-null, and Sapiens flat, the only
surviving independent anchors for the ~10-point deep-hover budget are the
scene cloud's sparse limb tail (weak, FP-risky alone) and **physics
feasibility** (a still body must be supported). The wave-2 design combines
them: physics infeasibility *gates* (still + no support within reach = the
depth is provably wrong), the cloud tail *proposes* where support plausibly
is, and healthy bodies never meet the gate. The shallow hovers (4–7 cm gaps,
about half the hover budget) may fall to the limb-grip fixes already in
flight.

## The physics gate works (as a detector)

Your idea 3, prototyped in its simplest form on the current bodies: *a
quasi-still climber must be supported* — flag any still frame whose nearest
end-effector is more than 8 cm from the scene. Result on the 30 scenes:

- Fires on exactly the five whole-body-floating scenes (`CRquJ8H1QLI_0001`
  15% of frames, `_0009` 7%, `_0008` 1%, `_0024` 0.6%, `s-ArwEzr-2M_0021`
  22%), and on **zero** of the other 25 scenes.
- On every flagged frame, the annotator confirms a hand or foot IS in contact
  — so each flag marks a frame where the reconstruction is *provably* wrong.
  100% precision as an error detector.
- Honest scope limits: it cannot adjudicate per-limb hovers (a climber hanging
  by two hands with feet dangling is physically legal even when the annotator
  says the feet touch), and the one torso-inside-wall scene does not flag
  (penetration reads as "support" — the gate needs a symmetric
  impossible-penetration arm).

This is the license wave 2 needs: on gate-flagged stretches, a correction is
justified by physics — and healthy bodies never meet the gate.

**Two tempting wave-2 designs measured and rejected before building anything:**

- *"It's a metric-scale error."* The five flagged scenes each need one very
  consistent clip-wide scale multiplier (0.93–0.98, spread under ±1%) to make
  support possible — suspiciously clean. But two follow-ups killed it: a
  torso-only scale re-estimate moves the scale the *wrong* way (1.00–1.05),
  and rescaling the world moves the human cloud together with the wall, so
  the body (which follows its cloud) would keep almost the same gap. The
  consistent multiplier is an emergent property of bodies settling at the
  front of consistently-shaped smears — not evidence of a scale error. Also
  learned: the human-ruler scale is structurally *blind* to human-vs-scene
  bias (it compares the body to its own human cloud, never to the wall).
- *"Correct the whole body's depth."* Not needed — the data is better than
  that. On the physics-flagged frames, the closest limb's own cloud tail sits
  at median 2.3–4.2 cm from the wall (within 5 cm on ~9 of 10 flagged
  frames). The support location is already IN the VGGT data; it's the smear
  front that hides it.

**So wave 2 becomes exactly your idea 1 + idea 3 combined**: on gate-flagged
stretches only, *squeeze the flagged limb's depth points along their rays*
onto their own deep (wall-side) core — a data correction, not a new loss —
and let the standard optimizer consume the corrected cloud. The
false-positive risk that made tail-following unsafe globally disappears
because the gate has measured 100% precision, and outside the gate not a
single point changes.

## Wave-1 sweep results (30 scenes, production gate, vs same-driver baseline re-run)

The A/A control first: re-running the unchanged pipeline through the driver
reproduces the baseline scores to the third decimal on every scene — the
optimizer is effectively deterministic here, so differences of ±0.005 F1 and
up are real.

| arm | FOCUS P | FOCUS R | FOCUS F1 | verdict |
|---|---|---|---|---|
| baseline re-run | 0.885 | 0.550 | 0.678 | reference |
| **limb-quota sampling** | 0.884 | **0.561** | **0.687** | **adopted candidate** |
| anisotropic chamfer | 0.878 | 0.488 | 0.627 | **rejected** |
| quota + aniso | 0.877 | 0.492 | 0.630 | **rejected** (aniso dominates) |

- **Limb quota (+0.009 F1, +1.1 recall at flat precision)**: gains land
  exactly on the diseased scenes the diagnosis predicted (+0.025…+0.050 on
  `CRquJ8H1QLI_0001/0008/0009/0041/0060/0061`, `ecx…0017`); no healthy scene
  loses more than 0.013; deep-penetration fraction *falls* (0.052 → 0.048) so
  the recall is honestly earned. It also improves the old 5-clip suite
  (pooled +0.007 F1, boulder_3 +0.019, nothing regresses) — adopted as the
  default. The sliding-feet scenes barely move — more points alone don't stop
  the slide.
- **The slide's real cause found (measured after the sweep)**: with 200
  points on the foot, the reconstructed foot still travels 7–18 cm on planted
  frames. The Sapiens 2D keypoints themselves drift there — median 14–19 px
  per window, and on `s-ArwEzr-2M_0050` the right ankle jumps ~170 px at 0.91
  confidence (left/right swaps). The reprojection term then drags the foot
  after a moving target. This is *detector drag*, a fourth mechanism —
  fixable by keypoint-track hygiene (swap detection, temporal-consistency
  gating of keypoint targets), queued as wave 3.
- **Anisotropic chamfer (−0.051 F1)**: recall collapses on 22 of 30 scenes
  (worst −0.30). The external review predicted this failure mode precisely:
  the along-ray chamfer pull — streaks included — is the only absolute-depth
  evidence the optimizer has; saturating it lets bodies drift toward the
  (even worse) init depth. "Trust depth less" fails when there is nothing
  else to trust. The knob stays in the code, off by default; wave 2's
  corrected clouds must be consumed with the raw kernel.

## Wave 2, first iteration: the data fix works, the optimizer ignores it

The physics-gated de-smear was built and verified (opt-in, bit-exact off,
healthy scenes skip at ~zero cost, gate matches the validated prototype
exactly). The trim itself does what it should: on flagged frames it drops the
smear front (~30–43% of the limb's points) and the kept deep band sits at
**0.7–1.2 cm from the wall** — the corrected evidence is right where physics
says the limb is. But the refined body barely moves (closest-limb distance
10.8 → 11.3 cm — a null): the corrected frames are outnumbered — smoothness
couples them to the ~90% neighbouring frames whose clouds still hold the old
depth, and the pose prior + init anchor hold the arm pose, so a ~200-point
band inside a per-frame chamfer mean loses the tug-of-war.

Three iterations later, the seating works:

1. *Stretch-level de-smear* (correct the whole still stretch, not single
   frames) fixed the sign — every stretch moves wall-ward — but only by
   0.5–3 cm against a 10–15 cm gap.
2. *Band boost* (the physics-licensed band gets ×5 weight in chamfer) helped
   marginally.
3. *Anchor relax* was the unlock: on gated stretches the init-projection
   anchor and pose prior are pinned to a converter pose that **physics has
   proven wrong there**, so those two terms (and only those, only there) get
   scaled by 0.1. Result on the pilot scenes: `s-ArwEzr-2M_0021` flagged
   frames went from 12.6 cm median off the wall to **3.1 cm** (25 of 33
   frames under 5 cm), with zero penetration and *better* 2D reprojection on
   those frames — the freed body found a more image-consistent pose.
   `CRquJ8H1QLI_0001` improved 10.8 → 8.0 cm (partial). This is un-pinning
   refuted evidence, not pushing: no new pull exists anywhere.

**Full 30-scene result**: only gated scenes change, all favourably or neutral
(`_0001` +0.015 F1 with precision rising 0.880 → 0.948, `_0009` +0.007,
`s-ArwEzr_0021` +0.009); the 25 ungated scenes are bit-identical; penetration
clean. Pooled FOCUS stays 0.687 — because the gate's perfect precision comes
with tiny coverage (1.5% of frames; ~30 seated frames per diseased scene
against ~1000 annotated positives). The machinery is right; the license is
narrow. Widening the license (per-limb gates, dynamic-phase feasibility) is
the concrete next frontier, and it will cost some of the 100% precision.

An external review of the full diff then confirmed six issues (worst: the
stretch originally extended to the whole still run — three transient flags
could have rewritten 500 frames); all six fixed and re-verified (bounded
stretch binds; validated results unchanged; every new config error fires
loudly; kinematics_2 rejects 14-joint annotations instead of silently
mis-mapping them).

## Where the round lands (honest bottom line)

FOCUS (hands+feet+ankles), production gate, close=3 cm, still=0.03:

| | P | R | F1 |
|---|---|---|---|
| baseline | 0.885 | 0.550 | 0.678 |
| shipped (quota + gated de-smear) | 0.884 | 0.562 | 0.687 |
| your target | ≈0.9 | 0.7–0.8 | — |

+1.2 recall points pooled, honestly earned (penetration down, healthy scenes
guarded, legacy 5-clip suite also +0.007 F1), plus the diseased scenes
individually up +0.015…+0.050. The target remains far, and this round's
biggest value is that the road to it is now *measured*, not guessed:

1. **Deep hover (~10 recall points)**: the physics-gate + de-smear + anchor-
   relax chain is proven end-to-end; it needs a wider license. Extend the
   gate per-limb and to low-acceleration (not just still) phases, accepting a
   measured precision trade.
2. **Detector drag (~part of 12 points)**: Sapiens foot/ankle keypoints drift
   and left/right-swap on planted frames (measured up to 170 px at 0.91
   confidence); the reprojection term drags feet after them, and the honest
   stillness gate then vetoes the contact. Keypoint-track hygiene (swap
   detection, temporal-consistency gating) is untouched territory with a
   clear measurement waiting.
3. **Shallow hover (4–7 cm gaps, ~8 points)**: partially addressed by quota;
   the remainder likely needs the readout to acknowledge that the scene cloud
   itself is 5 cm-voxel data (close_thresh 3 cm sits below the geometry's
   resolution) — a calibration question, not a body question.

## What was refuted this round (do not re-try without new evidence)

- Init-body (SAM-3D) depth as an along-ray anchor — floats 20 cm on exactly
  the diseased frames.
- Sapiens pointmap as a human-depth witness — renders climbers flat, crop or
  no crop.
- Global metric-scale correction — the "consistent multiplier" was an
  artifact; rescaling moves the cloud with the wall; torso-only ruler moves
  the wrong way.
- Global cloud de-smearing / tail-following — free-frame tails hug the wall
  too (9–13 cm); only the physics gate makes tail-trust safe.
- Saturating/anisotropic chamfer — the along-ray pull is load-bearing;
  −0.05 F1 pooled.

## Log

- 2026-07-10: data_2/out_2 built and verified (30/30 clean). Scorer adaptation
  + prior-art research running. No-GPU disease measurements done (tables above).
  Limb-anchor measurement running on all 30 scenes.
- 2026-07-10 (later): scorer landed (old clips bit-identical, FOCUS row added);
  30-scene baseline + funnel done; limb-anchor + free-frame falsification done;
  de-smearing rules scored offline → near-null, demoted. E1 (stratified
  sampling) + E2 (robust/anisotropic chamfer) being implemented as opt-in
  knobs. Sapiens pointmap being rendered on 7 dev scenes; first result on the
  grossest scene (`ecx…0017`, torso reconstructed inside the wall) shows
  torso-anchored alignment inherits VGGT's torso error there — as expected;
  the informative test is the healthy-torso/streaked-limb scenes. Fixed a real
  loader bug on the way: the Sapiens adapter moved the fp32 5B model to GPU
  *before* casting to bf16 (~20 GB transient → OOM on half-occupied GPUs); it
  now builds+casts on CPU and moves after. External adversarial review of the
  diagnosis + plan is running.
- 2026-07-11: codex review adopted (scoring gate fixed, E1/E2 redesigned,
  guards added). Sapiens pointmap witness measured → null (flat people), both
  full-frame and crop mode. Gate=pre baseline re-derived with a wider readout
  grid. E1/E2 revised implementation landed (bit-exact neutral parity; limbs
  now hold their 200-point quota on starved scenes). Physics support-gap
  detector prototyped → 100% precise on the five floating scenes. The
  pre-registered 4-arm sweep ran: A/A exact, quota +0.009 F1 (honest, on
  target scenes, legacy suite also +0.007 — adopted), aniso −0.051 (rejected
  — depth pull is load-bearing), quota+aniso −0.057 (rejected). Sliding-feet
  cause isolated: Sapiens keypoint drift/left-right swaps (detector drag) —
  wave-3 material. Wave-2 gated de-smear iteration 1: data validated,
  consumption null; iteration 2 (stretch-level + band boost) running.
- 2026-07-11 (post-round cleanup): everything this round (and round 10)
  measured as null or left off-by-default was removed from the code, at the
  user's request to keep the repo simple. Gone: the saturating / anisotropic
  chamfer kernels (`[chamfer] kernel` + `sat_sigma_m`), round-10's
  wall-contact seating (the loss, the rigid `seat` phase, the float gate, the
  `[wall_contact]` section), two orphaned loss wrappers (`scene_attraction`,
  `clearance_repulsion` — the optimizer calls their penalty heads directly),
  the diagnose script's dead velocity-gated-contact panel (its loss was
  deleted long ago), and the `--set extreme` vertex-weight preset (measured
  no-op). What ships is unchanged: limb-quota sampling + the physics-gated
  de-smear, both default-on. Removal verified bit-exact against pre-cleanup
  runs on a de-smear-gated scene (s-ArwEzr-2M_0021) and a healthy scene
  (ecx75KwBCV8_0017), same GPU. Statements above about these knobs "staying
  in the code, off" are superseded by this entry.
