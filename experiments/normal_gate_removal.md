# Contact = near + still: removing the normal gate (2026-08-02)

Last round ended with the diagnosis; this round acted on it. The user's call:
scene points and pose are too noisy for a normal *gate* — contact should be
"near the scene AND still", on the 12 contact frames only, with the normal
deciding exactly one thing (which of the two toe frames touches). Also decided:
hands are smoothed, not optimized; and thresholds should relax at kinematics_1
instead of strict-at-fit / loose-later. Shipped as three commits (a9e3a78,
a38d729, 9f6ee3f). Evidence trees named below still exist.

## The judge: scoring against the manual annotations

`scripts/score_frame_contacts.py` (a9e3a78) folds the 12 frames onto the
annotated coarse joints — hand ← palm ∪ finger; 14-joint schema: foot ← the two
toe frames, ankle ← heel; 10-joint schema: foot ← toes ∪ heel; knee ← knee.
hip / elbow / shoulder have no frame and are excluded. All 13 `data/` scenes
have an annotation; tri-state labels, `ignored` people skipped, people matched
by `object_ids`. Every number below is this scorer.

Baseline (the pre-round `out/` artifacts): **pooled P .827 / R .188 / F1 .307**.
Precision was high; recall was the disaster — and knees had 0 true positives
over the whole corpus.

## What shipped

| change | commit | measured effect (full 13 scenes) |
|---|---|---|
| Finger smoothing at init (`finger_smooth_s = 0.2`) | a38d729 | jitter 6–20× down (3.7 → 0.35 °/frame), body reprojection ≤0.03 px change, fingertip 2D −0.06 px pooled, F1 −.004 (neutral) |
| Normal gate removed; toe-pair tiebreak; one `still_m = 0.04` | 9f6ee3f | pooled F1 .307 → **.488**, precision .827 → **.869**, 13/13 scenes improved, toe both-fire 240 → **0** |

Attribution of the +.185 (each step measured separately, trees
`out/_expG_gates/{v0_smooth,v1_nogate,s0.04}/` + the readout-only
decomposition row): smoothing −.004 · gate removal in the readout **+.107** ·
gateless kin1 labels → better bodies **+.049** · `still_m` .03 → .04 **+.029**.

The recall came from exactly where last round's diagnosis said it was lost:
fingertips at correct distance whose baked pad normal pointed away (the frozen
hand pose). Fingertip contact rate .11 → .33; hand F1 .274 → .565. The feared
failure mode — new false positives from heels/knees near walls they face away
from — did not materialize: precision went *up*.

### The toe tiebreak

`toe_under` and `toe_above` sit millimetres apart, so every distance/stillness
gate accepts them together. Rule (`suppress_toe_pair`): at most one of the pair
per step; where both are candidates, the one whose normal looks more into the
wall keeps its signal, the other is forced out before hysteresis; ties keep
`toe_under`. No knob. One composition bug was caught by measurement, not by
construction: the readout's span-merge bridges each frame independently and
resurrected a suppressed toe in 2 scenes (22 cells) — fixed by masking the
merge with the tiebreak verdict, cost 3 TP of 4558.

### The threshold relax

`[auto_contact].still_m` (0.025, the strictest gate in the chain) was deleted;
the kin1 detector now inherits `[contact_gate].still_m`, one knob for detector
and readout. 0.04 confirmed on all 13 (beats 0.03 everywhere); 0.05 scored
+.008 over 0.04 on the 6-scene subset only — inside noise, and it would call
~0.29 m/s "still" — not shipped. Net knob count: −2.

## In-loop re-detection: refuted (reverted, nothing shipped)

The tempting idea: detection is now cheap (~0.1–0.5 s), so re-detect during
refine and let attraction converge onto contacts the fit brings within reach.
Design honored the old honesty lesson — stillness stayed pinned to the
refine-start `drift_pre`; only geometry updated across re-detections.

Measured (6-scene subset, trees `out/_expH_inloop/`, LR-schedule controls run
for every phase split): re-detecting at L-BFGS start gains +.0010 (real —
the control reproduces its baseline to 1 cell in 8296 — but 5× below the ship
bar); re-detecting mid-refine **loses** ~.01 F1 with the LR confound removed.
The mechanism is not label churn poisoning precision (precision stays flat):
the extra attraction displaces the body from poses that produced more true
contacts. Labels genuinely move (+753/−342 cells; one scene acquires +286
purely additively), and acquisition saturates after a single extra chance —
but acting on those acquisitions makes the body worse.

The clarifying observation: `estimate_contacts` never reads the fit's labels —
it re-detects on the final body and inherits only `drift_pre`. So in-loop
re-detection can only help by improving the *body*, and it doesn't: frames
that arrive near the wall mid-fit are mostly not on real holds. Raising recall
further needs a different route to move the body toward the wall.

## Open problems (unchanged by this round)

1. **Knees: 0 TP at every setting**, 146→183 FP — knee contact is never
   correctly predicted on this corpus. Nothing addressed it; next candidate.
2. **The depth/placement (floater) class** — gtIGgs .015, vcRBT .090,
   9Rvr .257 (from .000): the body is 8–48 cm off along the viewing ray; no
   contact-detection change can fix placement.
3. `sport_climb` .143: annotations label long ambiguous rope-climbing spans;
   worth a look at the annotation itself before optimizing against it.
