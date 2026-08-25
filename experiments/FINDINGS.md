# Limb-length premise test — findings

> **2026-08-02 addendum.** The E1–E3 argument below was judged unsound by the
> adversarial review commissioned at the time (E1/E3 compare 2D projections as
> if they were 3D lengths; E2's slope is endogenous; the "below chance"
> arithmetic is wrong), so this document never validly closed the question.
> It has since been closed properly — same conclusion, sound route: three
> definition-free measurements show the limbs are the right length and the
> "short forearm" was an MHR-vs-SMPL-X joint-convention offset. See
> [`frames_contact_rate.md`](frames_contact_rate.md), Experiment B.

**Question.** We were deciding whether to let **limb lengths change** when we fit the
body to the depth cloud (today the body's shape is frozen). Before building that, one
question had to be answered:

> Are the body's arms and legs actually the wrong length, or does the body just *look*
> wrong because of clothing, mask bleed at the wall, and depth errors?

If the limbs are fine and the error is clothing or bleed, then making limbs adjustable
would quietly bake those artifacts into the skeleton — the fit would look *better* while
the body got *worse*.

Three diagnostics, run over ~374 scenes / ~378 people (361 corpus + 13 local). Each was
blind to the others' failure modes. The scripts printed raw numbers; the verdict below is
the reader's.

## Verdict: premise NOT supported

The residuals that motivated the limb-length idea are **not** limb-length errors, and a
person's individual limb lengths **cannot** be measured from the evidence this pipeline
has. An optimizer handed limb-length freedom would fit clothing / bleed / depth artifacts
and report a *better* fit while making the body *worse*.

This does **not** say the limbs are correct, nor that MHR's limb parameters are useless —
only that the available evidence cannot identify a per-person limb length.

## E1 — do the proportions disagree in 2D?

Compared, in pixels, the 2D keypoints SAM-3D-Body predicts for its own body against the
2D keypoints Sapiens observes. Per bone, `r = Sapiens length / SAM3D length`. Because both
live in the same image, `r` ignores depth, scale, masks, and clothing. Shoulder/hip widths
were included as controls (the most convention-sensitive measurement).

**Two different questions, two different answers.** Between the two models there is a real,
symmetric, systematic offset (thighs read ~4.4% short, forearms ~2.8% long; each bone's
left and right agree to within 0.3–1.5%). But an offset that lands the same on everybody is
a constant disagreement between two predictors — a one-time recalibration, not a per-person
limb length. Per person the signal is under the noise: real between-person spread
(`sd_true`) is 0.00–0.029, while the per-side error surviving frame averaging (`sd_noise`)
is 0.051–0.060 → signal-to-noise 0.18–0.56 (shank 0.00). The width controls (shoulder
+8.9%, hip +18%) are several times larger than any long-bone offset — the two detectors
simply place joint centres differently.

Caveat E1 cannot escape: it measures disagreement between two predictors and cannot say
which is right.

## E2 — is the 3D depth error actually a length error?

Geometry the artifacts do not obey: if a bone is short by `Δl`, the far joint sits short by
`Δl` along the bone's own direction, and the depth camera sees only the component along its
axis. So `gap ≈ Δl · u_z` — biggest when the limb points at/away from the camera, **zero**
when it lies flat across the image, and it flips sign. Clothing/bleed give a gap that is
just *there*, independent of limb direction. So per person-bone we fit `gap = a·u_z + b`:
`a` (cm) is the limb-length part, `b` the artifact offset. `a` is only identifiable when the
limb actually changes direction, so rows with `u_z` spread < 0.15 were refused.

**The slope is not a body.** The depth gap really does depend on limb direction (80.8% of
slopes beat their own permutation null; robust Theil-Sen agrees, so not outliers). But the
fitted "limb-length error" is a median **37% of the bone it claims to correct**, and for
19% of people it exceeds the whole bone (one real case: a 24.6 cm forearm with a +18.1 cm
"error"). Left and right are uncorrelated (Pearson 0.06–0.16) and disagree by 14.6–22.2 cm,
which anatomy cannot do. So `u_z` is standing in for something else that co-rotates with the
limb — foreshortening, self-occlusion, and how the depth model behaves on a limb aimed at
the camera.

Sign convention was checked before trusting any of this: the estimator recovers an injected
+3 cm error as +3.35/+2.82 cm and −3 cm as −2.69/−3.18 cm, so the E3 contradiction is real,
not a flipped sign.

## E3 — do the two roads agree?

E1 and E2 share no inputs beyond the scene list (E1 never touches depth/masks/mesh; E2 never
touches Sapiens), yet both end at the same number: how many cm is this bone off? A real error
cannot hide from one and appear in the other, nor point one way in 2D and the opposite in depth.

**They contradict.** The two roads disagree on the population median sign for **8 of 8
bones**. Per person they agree on 34–48% of cases — *below* the 50% a coin flip gives. Their
per-person corrections correlate −0.30 to +0.07. Two independent measurements that contradict
are two measurements of their own artifacts.

---

*The diagnostic scripts (`e1_proportion_2d.py`, `e2_depth_direction.py`, `e3_cross_check.py`)
and their raw JSON/markdown outputs were removed after the verdict was reached; this file is
the record. They read existing `out/` + corpus artifacts and printed numbers — nothing here
was ever imported by the pipeline or written into `out/`.*