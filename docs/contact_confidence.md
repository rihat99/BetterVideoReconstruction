# How contact-label confidence is computed

Every joint of the reconstructed body, in every video frame, gets a binary
label: *in contact* with the scene, or *not in contact*. This document
explains the confidence score `label_confidence ∈ [0, 1]` attached to each of
those labels — what it means, what signals it is built from, and the exact
formulas. It is written to be read start to finish; the formulas use real
numbers rather than symbols wherever that makes them clearer.

The one-sentence version: **a label is confident when several independent
witnesses agree with it, and uncertain when the witnesses are silent or
disagree.** The witnesses are geometry (how far the joint is from the
reconstructed scene), motion (how fast the joint is moving), visibility (how
well we actually observed that part of space), and temporal consistency (do
neighbouring frames say the same thing).

Both classes are scored. A confident *contact* and a confident *no-contact*
are equally valuable to a learning pipeline — and an honest "I don't know"
(confidence near zero) is more valuable than a guess.

## The input signals

All signals come from things the pipeline has already computed: the fitted
body (52 joints per frame), the metric 3-D scene point cloud reconstructed
from the video, and the contact labels themselves.

**Distance to the scene, `d` (meters, signed).** For each joint we take the
body-surface vertices belonging to it, find their closest scene points, and
sign the distance using the local surface orientation: positive outside the
wall, negative when the body penetrates it. This distance exists in two
flavours. The *guarded* distance only trusts scene points within 10 cm that
have good depth confidence — beyond that it says "no support nearby". It is
used to decide how contact-like a joint is. The *unguarded* distance `D` is
the same measurement without the 10 cm cap (finite out to ~1.6 m) and is
used for the opposite question — how *far* from everything a joint is —
where the difference between 15 cm and 40 cm matters.

**Drift, `m` (meters).** How far the joint travelled over a 0.35-second
window. Crucially, drift is measured on the body *before* the optimizer's
contact-stabilising terms are applied, so a joint cannot look still merely
because we forced it to be still ("measure, then enforce"). Speed is
`v = m / 0.35 s`.

**Observability.** Whether we genuinely saw the space around the joint:
how many scene points were reconstructed within 15 cm of it, how confident
the depth model was about them, and whether the joint projects inside the
image frame.

**Persistence.** The fraction of frames within ±0.35 s that carry the same
label as this one. An isolated one-frame label is suspicious; a label in the
middle of a long run is trustworthy.

**Body health, `h`.** A per-person score that detects *floating bodies* —
reconstructions where depth error placed the whole climber off the wall.
Described in its own section below.

Throughout, `clamp(x)` means clipping `x` to `[0, 1]`, so every factor is a
number between 0 and 1 and the final confidence is too.

## Confidence that a contact is real

For a joint labelled *contact*:

```
conf(contact) = persistence × observability × p_contact
p_contact     = p_close × max(p_still, p_pen)
```

The three geometric/motion pieces:

```
p_close = clamp((0.06 − d) / 0.06)        1 when touching, 0.5 at 3 cm, 0 beyond 6 cm
p_still = clamp(1 − m / 0.06)             1 when still, 0.5 at 3 cm drift, 0 beyond 6 cm
p_pen   = clamp((0.03 − d) / 0.04)        0 at 3 cm outside, 1 by 1 cm inside the wall
```

Read it as a conjunction of witnesses. The joint must be *near the surface*
(`p_close`). It must be either *still* — you hold a grip by not moving — or
*visibly pressed into the wall* (`p_pen`). And we must have *seen* it
(observability) with *neighbouring frames agreeing* (persistence).

The `max(p_still, p_pen)` deserves a sentence. A climber's gripping hand
micro-adjusts by 2–3 cm within our measurement window, which puts `p_still`
around 0.5 — yet the grip is beyond doubt. Penetration settles the argument:
a joint measured *inside* the wall is touching it regardless of how much it
wiggles, so firm penetration is allowed to override motion doubt. Without
this override, half of all true contacts sat at confidence ≈ 0.5 for no good
reason. (The stillness requirement already gated the label when it was
*created*; punishing the same drift a second time in the confidence was a
design error this rule fixes.)

Observability is itself a product:

```
observability = clamp(n_15cm / 20) × in_image
```

— at least 20 reconstructed points within 15 cm of the joint, and the joint
inside the image (with a 10 px margin). A contact we could not see cannot be a
confident contact, however plausible the geometry. (A depth-confidence third
factor was dropped in round 8: it was measured inert — the reconstructed points
near a contact always carried enough depth confidence to leave it at 1.)

## Confidence that a no-contact is real

For a joint labelled *no-contact*:

```
conf(no-contact) = persistence × max(p_fast, in_image × far)
far              = max( h × max(p_far, p_far_nan),  p_far_rel )
```

There are two kinds of evidence that a joint is *not* touching — motion and
distance — and we take the stronger. Distance itself has three witnesses,
combined by `max`.

**The motion witness.** You cannot grip while flying:

```
p_fast = clamp((v − 0.15) / 0.35)         0 below 0.15 m/s, 1 above 0.5 m/s
```

A fast-moving joint is a confident no-contact *even if we never saw the
space around it* — occlusion does not matter, and (deliberately) neither
does body health: motion evidence survives a broken reconstruction.

**Distance witness 1 — absolute.** The joint is clearly far from everything:

```
p_far = clamp((D − 0.22) / 0.06)          0 at 22 cm, 0.5 at 25 cm, 1 beyond 28 cm
```

`p_far` uses the unguarded distance `D` and its threshold sits at 25 cm on
purpose — beyond the size of any climbing hold. If a hand rests on a hold
that the 3-D reconstruction failed to capture (a real and common failure,
"the carved-out hold"), the nearest *reconstructed* wall may be 10–20 cm
behind the physical hold. A 25 cm threshold means such a hand reads
*uncertain* rather than *confidently free*. This threshold was checked
against the annotation: among still, in-frame joints, the fraction actually
touching is still 29% at 12–25 cm and only 1% past 25 cm — so 25 cm is where
the ambiguity ends, and lowering it would mislabel real contacts as free. The
absolute witness requires the joint to project inside the image and is scaled
by body health `h`.

**Distance witness 2 — relative clearance.** The absolute distance is
useless on a *floating* body — a reconstruction placed 30 cm off the wall
makes every joint look far. But the joint's distance *above the body's own
closest joint* is not corrupted, because the whole-body error cancels in the
difference:

```
rel      = D − max(0, D_closest)          how far this joint sits above the nearest one
p_far_rel = clamp((rel − 0.22) / 0.06)    same 25 cm carve threshold as p_far
```

A dangling foot 60 cm above a floating body's own closest point is provably
free even when its absolute distance means nothing. The closest level is
clamped at the wall surface (0) so a firmly-pressed contact elsewhere on the
body cannot inflate the clearance. This witness carries **no** health gate —
it is immune to the whole-body error by construction — but it is immune only
to *whole-body* error, not *per-limb* error: a floater's own near-wall joints
(small `rel`) stay uncertain, which is correct.

**Distance witness 3 — nothing within reach.** If no scene point lies within
about 1.6 m of the joint at all, that is the strongest far evidence there is —
but only when the scene is real (at least 1000 reconstructed points, so an
empty cloud does not qualify) and the body is healthy. Like the absolute
witness, it is scaled by `h`, so a floater does not benefit from its own
missing geometry.

## Body health: why a floating body loses its distance witness

Some reconstructions fail globally: depth error places the entire climber
tens of centimetres off the wall, so *every* joint is "far from the scene".
In a climbing video that situation is self-contradicting — the climber
demonstrably is on the wall — so large distances there are evidence of a
broken reconstruction, not of free joints. Trusting them produced
confidently-wrong labels: before this gate, floating bodies had no-contact
confidence averaging 0.79.

Health is the product of two per-person factors, computed from the clip's
own labels and distances:

```
h = clamp(contact_rate / 0.1) × clamp(1 − hover_median / 0.15)
```

`contact_rate` is the share of frames where *any* joint touches; a body
essentially never touching (rate below 10%) is treated as floated.
`hover_median` is the median, over frames, of the *closest* joint's distance
(meters): a body whose nearest point hovers 15 cm off the wall throughout
the clip gets `h = 0`. A healthy body — touching regularly, seated on the
wall — gets `h = 1` and is completely unaffected.

`h` multiplies the *absolute* and *nothing-within-reach* distance witnesses,
never the motion witness and never the *relative*-clearance witness (relative
clearance already cancels the floating-body error, so it needs no gate — it is
what lets a floater's genuinely-free joints still score high). On the current
corpus, 310 of 363 bodies have `h ≥ 0.8`; the ~50 muted ones are floaters
found by the corpus audit.

## What stays deliberately uncertain

The design keeps confidence low where the evidence genuinely cannot decide.
A slow joint 3–25 cm from the wall (the hover band — a missed contact and a
near-miss look identical there); a slow, occluded joint on a *floating* body
or a degenerate scene, where neither the absolute distance nor the
"nothing-within-reach" test can be trusted; a joint outside the image; frames
whose contact label was bridged across a gap by the span-merge step (these
carry a `merged` flag so a consumer can treat them separately). For all of
these, low confidence is the correct answer, not a failure to be fixed — a
learner should be able to *ignore* them cheaply, which is the point of
shipping confidence at all.

## Does it behave?

Measured on the full 361-scene corpus after the latest calibration:

| population | mean confidence |
|---|---|
| contacts, all (1.44 M joint-frames) | 0.67 |
| contacts with firm penetration (46% of contacts) | 0.85 |
| no-contact, fast joints | 0.97 |
| no-contact, still + far, healthy body | 0.95 |
| no-contact, still + far, floating body | 0.48 (recovered by relative clearance) |
| no-contact, still, hover band 0–25 cm | 0.02 (honest unknown) |
| no-contact, still, nothing within reach, healthy | 0.94 |
| no-contact, still, nothing within reach, floating | 0.03 (muted, by design) |

On the four hand-annotated clips, reliability curves are monotone for both
classes — higher confidence means a higher chance the label matches the
human annotation — and the highest-confidence bins are >90% correct.

## Where it lives

Computed in one pass by `estimate_contacts` after the labels are final, in
`tools/human_optim/confidence.py` (`contact_confidence`,
`compute_body_health`); all constants sit in `CONTACT_CONFIDENCE_DEFAULTS`
and can be overridden in the `[contact_confidence]` table of
`configs/human_optimization.toml`. Saved per scene in `contacts_1.npz`:

| field | shape | meaning |
|---|---|---|
| `joint_contact` | (P, N, J) bool | the binary label |
| `label_confidence` | (P, N, J) | confidence in that label (this document) |
| `p_contact` | (P, N, J) | the geometry+motion factor alone |
| `observability` | (P, N, J) | the visibility factor alone |
| `merged` | (P, N, J) bool | label was bridged by span-merge |
| `joint_sdf_cm` | (P, N, J) | unguarded signed distance, cm |
| `body_health` | (P,) | the per-person health scalar `h` |
| `confidence_schema` | scalar int | version of this field set — **frozen at 1** |

(P = people, N = frames, J = 52 joints.)

The field set above is the frozen interface for the learning pipeline;
`confidence_schema = 1` stamps it. A future change to the fields or their
meaning increments this number.
