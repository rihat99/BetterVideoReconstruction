# Contact-confidence factor audit

This is the evidence behind which signals the contact-label confidence
([`contact_confidence.md`](contact_confidence.md)) actually keeps. Every factor in
the formula was switched off one at a time, and the effect measured in two places:
the whole 361-scene corpus (6.15 M valid joint-frames, 1.44 M of them contacts) and
the four hand-annotated clips. A factor that changes *nothing* in *both* is dead
weight and can be removed without loss.

The measurement is faithful: recomputing the confidence from scratch through the
instrumented code reproduces the shipped values bit-for-bit (max deviation 0.0 on
every body), and every headline corpus number matches the round-7 figures exactly
(contacts 0.674, penetrating contacts 0.854, floating-body far 0.145, hover 0.019).

## How to read the tables

Each row is a *population* of joint-frames. `base` is the mean confidence there
today. Each remaining column is the **change** in that mean when one factor is
switched off (set to its identity — a multiplicative factor becomes 1, a witness
inside a `max(...)` is removed). A number near zero means the factor never bites in
that population.

### Corpus — mean confidence, and Δ when each factor is switched off

| population | n | base | conf_near | n_near | in_img | persist | p_close | p_pen | p_still | h_contact | h_hover | p_far | p_fast |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| contacts (all) | 1 444 340 | 0.674 | +0.000 | +0.018 | +0.002 | +0.082 | +0.055 | −0.208 | −0.062 | +0.000 | +0.000 | +0.000 | +0.000 |
| contacts, penetrating (<−1 cm) | 662 733 | 0.854 | +0.000 | +0.022 | +0.003 | +0.111 | +0.000 | −0.345 | −0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| no-contact, fast (≥0.5 m/s) | 1 058 008 | 0.973 | +0.000 | +0.000 | +0.000 | +0.027 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | −0.717 |
| no-contact, still, far ≥25 cm, healthy | 381 279 | 0.940 | +0.000 | +0.000 | +0.003 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.008 | −0.940 | +0.000 |
| no-contact, still, far ≥25 cm, floating | 197 349 | 0.145 | +0.000 | +0.000 | +0.001 | +0.000 | +0.000 | +0.000 | +0.000 | +0.012 | +0.222 | −0.145 | +0.000 |
| no-contact, still, hover 0–25 cm | 1 206 036 | 0.019 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.002 | −0.019 | +0.000 |
| no-contact, still, nothing near (NaN) | 44 280 | 0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

### Labeled clips — reliability, per factor switched off

Pooled over the four annotated clips (sport_climb excluded, as the scorer does).
`rho` is Spearman(confidence, label-is-correct) — higher means confidence tracks
correctness; `top` is accuracy in the highest-confidence tenth; `CWN` is the
confident-wrong-no-contact rate (annotated contacts the readout called no-contact
with confidence > 0.7 — lower is safer).

| switched off | rho contact | top contact | rho no-contact | top no-contact | CWN |
|---|--:|--:|--:|--:|--:|
| — (baseline) | +0.204 | 0.937 | +0.407 | 0.930 | 0.289 |
| conf_near | +0.204 | 0.937 | +0.407 | 0.930 | 0.289 |
| n_near | +0.206 | 0.937 | +0.407 | 0.930 | 0.289 |
| in_img | +0.204 | 0.937 | +0.408 | 0.930 | 0.289 |
| persist | +0.154 | 0.915 | +0.366 | 0.904 | 0.348 |
| p_close | +0.207 | 0.937 | +0.407 | 0.930 | 0.289 |
| p_pen | +0.183 | 0.918 | +0.407 | 0.930 | 0.289 |
| p_still | +0.199 | 0.937 | +0.407 | 0.930 | 0.289 |
| h_contact | +0.204 | 0.937 | +0.407 | 0.930 | 0.289 |
| h_hover | +0.204 | 0.937 | +0.407 | 0.930 | 0.289 |
| p_far | +0.204 | 0.937 | +0.247 | 0.907 | 0.287 |
| p_fast | +0.204 | 0.937 | +0.457 | 1.000 | 0.002 |

## Verdicts

**Dropped in T3 — `conf_near` (the depth-quality half of observability).** Zero
change in every corpus population, bit-identical reliability. Wherever a joint is
labelled *contact*, the depth confidence of the nearby scene points is already above
the 0.3 threshold, so this factor never lowers anything. Removed in T3 along with its
two now-dead knobs `conf_ref` and `obs_k` and its top-k neighbour compute; the
re-score confirmed the confidence was bit-identical before and after.

**Kept in T3 — `h_contact` (the "almost never touches" half of body-health).** On the
v2 formula it looked nearly redundant: switching off `h_hover` lifts the floating-body
far-confidence by +0.222, switching off `h_contact` by only +0.012, so `h_hover` alone
reproduced ~95% of body-health. But T2 gave body-health a new job — gating the
"nothing within reach" witness — and on that final formula, dropping `h_contact` lifts
the floating-body *NaN* bucket by +0.142 and un-mutes near-wall never-touch bodies. It
is no longer dead weight, so the freeze keeps both health sub-factors.

**Keep — everything else**, each with a population where it clearly bites:

- `persist` (temporal agreement) — strong in both classes; removing it worsens
  reliability *and* the safety rate.
- `p_pen` (penetration override, added in round 7) — −0.345 on penetrating
  contacts; the reason half of all contacts are no longer stuck at ~0.5.
- `p_close` and `p_still` — the near-but-not-penetrating gate and its stillness
  partner; they bite on ordinary grips (+0.055, −0.062) and leave penetrating
  contacts untouched, so they complement `p_pen` rather than duplicate it.
- `p_far` — currently the *entire* far-from-wall no-contact witness; removing it
  zeros those buckets. T2's relative-clearance is a *second* far witness beside it,
  not a replacement.
- `p_fast` — the motion witness; removing it collapses the 1.06 M fast joint-frames
  from 0.97 to near zero.
- `n_near` (occlusion guard) — weak but real (+0.018 on contacts).
- `in_img` — its average effect is small only because few joints fall off-frame;
  for those joints it is the whole reason their confidence is low, which is the
  intended behaviour.

## Two notes carried into T2

- The baseline confident-wrong-no-contact rate (0.289) is almost entirely
  `p_fast`: fast-moving joints a human annotated as contact, which the motion prior
  scores as confident no-contact. T2 only changes the *distance* witness, so this
  number is not expected to move — the T2 safety gate is well-behaved by
  construction.
- Because `p_far` is the only current far witness, T2's relative-clearance can only
  *add* confidence to the floating-body bucket (base 0.145); it cannot take
  confidence away from healthy scenes, where the closest joint is near the wall and
  the relative measure agrees with the absolute one.
