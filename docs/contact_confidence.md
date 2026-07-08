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
observability = clamp(n_15cm / 20) × clamp(depth_conf / 0.3) × in_image
```

— at least 20 reconstructed points within 15 cm of the joint, decent depth
confidence on the nearest of them, and the joint inside the image (with a
10 px margin). A contact we could not see cannot be a confident contact,
however plausible the geometry.

## Confidence that a no-contact is real

For a joint labelled *no-contact*:

```
conf(no-contact) = persistence × max(p_fast, in_image × h × p_far)
```

There are two independent ways to be sure a joint is *not* touching, and we
take the stronger one.

**The motion witness.** You cannot grip while flying:

```
p_fast = clamp((v − 0.15) / 0.35)         0 below 0.15 m/s, 1 above 0.5 m/s
```

A fast-moving joint is a confident no-contact *even if we never saw the
space around it* — occlusion does not matter, and (deliberately) neither
does body health: motion evidence survives a broken reconstruction.

**The distance witness.** The joint is clearly far from everything:

```
p_far = clamp((D − 0.22) / 0.06)          0 at 22 cm, 0.5 at 25 cm, 1 beyond 28 cm
```

`p_far` uses the unguarded distance `D` and its threshold sits at 25 cm on
purpose — beyond the size of any climbing hold. If a hand rests on a hold
that the 3-D reconstruction failed to capture (a real and common failure,
"the carved-out hold"), the nearest *reconstructed* wall may be 10–20 cm
behind the physical hold. A 25 cm threshold means such a hand reads
*uncertain* rather than *confidently free*. The price is honesty everywhere
else: a still joint hovering 10–20 cm off a well-reconstructed wall also
stays uncertain, because from geometry alone we cannot tell these two
situations apart. The distance witness additionally requires the joint to
project inside the image, and is scaled by body health `h`.

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

`h` multiplies only the distance witness, never the motion witness. On the
current corpus, 310 of 363 bodies have `h ≥ 0.8`; the ~50 muted ones are
floaters found by the corpus audit.

## What stays deliberately uncertain

The design keeps confidence low where the evidence genuinely cannot decide.
A slow joint 3–25 cm from the wall (the hover band — a missed contact and a
near-miss look identical there); a slow joint whose surroundings were never
reconstructed (occlusion); a joint outside the image; frames whose contact
label was bridged across a gap by the span-merge step (these carry a
`merged` flag so a consumer can treat them separately). For all of these,
low confidence is the correct answer, not a failure to be fixed — a learner
should be able to *ignore* them cheaply, which is the point of shipping
confidence at all.

## Does it behave?

Measured on the full 361-scene corpus after the latest calibration:

| population | mean confidence |
|---|---|
| contacts, all (1.42 M joint-frames) | 0.67 |
| contacts with firm penetration (46% of contacts) | 0.85 |
| no-contact, fast joints | 0.97 |
| no-contact, still + far, healthy body | 0.94 |
| no-contact, still + far, floating body | 0.15 (muted, by design) |
| no-contact, still, hover band 0–25 cm | 0.02 (honest unknown) |

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

(P = people, N = frames, J = 52 joints.)
