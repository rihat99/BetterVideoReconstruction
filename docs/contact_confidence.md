# How contact-label confidence is computed

Every joint of the reconstructed body, in every video frame, gets a binary
label: *in contact* with the scene, or *not in contact*. This document
explains the confidence score `label_confidence ∈ [0, 1]` attached to each of
those labels — what it means, what signals it is built from, and the exact
formulas. It is written to be read start to finish; the formulas use real
numbers rather than symbols wherever that makes them clearer.

The one-sentence version: **a label is confident when several independent
witnesses agree with it, and uncertain when the witnesses are silent or
disagree.** Every witness carries a one-line physical assumption. The witnesses
are geometry (how far the joint is from the reconstructed scene), motion (how
fast it is moving), visibility (how well the camera and the keypoint detector
actually pinned that joint), and temporal consistency (do neighbouring frames
say the same thing).

Both classes are scored. A confident *contact* and a confident *no-contact*
are equally valuable to a learning pipeline — and an honest "I don't know"
(confidence near zero) is more valuable than a guess.

Confidence is a **readout**: it never changes the binary labels. It is computed
once, after the labels are final.

## The input signals

All signals come from things the pipeline has already computed: the fitted
body (52 joints per frame, and its surface mesh), the metric 3-D scene point
cloud reconstructed from the video, the Sapiens 2-D keypoint scores, and the
contact labels themselves.

**Distance to the scene, `d` (meters, signed).** For each joint we take the
body-surface vertices belonging to it, find their closest scene points, and
sign the distance using the local surface orientation: positive outside the
wall, negative when the body penetrates it. Two flavours. The *guarded*
distance only trusts scene points within 10 cm — used to decide how
contact-like a joint is. The *unguarded* distance `D` is the same measurement
without the 10 cm cap (finite out to ~1.6 m) and is used for the opposite
question — how *far* from everything a joint is.

**Drift, `m` (meters).** How far the joint travelled over a 0.35-second
window, measured on the body *before* the optimizer's contact-stabilising
terms are applied ("measure, then enforce"). Speed is `v = m / 0.35 s`.

**Visibility.** Two things about whether that joint was actually observed:
- *Occlusion* — how many body-surface vertices sit between the camera and the
  joint (hiding it). A joint behind the torso from the camera's viewpoint has a
  high occlusion count; an arm reaching out into the open has zero.
- *Detector belief* — the Sapiens keypoint score for that joint. A low score
  means the 2-D detector was unsure where the joint is.
- *In frame* — whether the joint projects inside the image at all.

**Persistence.** The fraction of frames within ±0.35 s that carry the same
label as this one — an isolated one-frame label is suspicious.

**Body health, `h`.** A per-person score that detects *floating bodies* —
reconstructions where depth error placed the whole climber off the wall.
Described in its own section below.

Throughout, `clamp(x)` means clipping `x` to `[0, 1]`, so every factor is a
number between 0 and 1 and the final confidence is too.

## The visibility witness, and why it enters the two classes differently

Occlusion and low detector score both make the automatic label wrong *more
often* — but only for the **contact** class, and in a way that has a clear
direction. Measured on the four hand-annotated clips (excluding the
bad-reconstruction outlier `sport_climb`):

- An annotated **contact** that is occluded (≥5 body vertices in the way) is
  **missed** — labelled no-contact — 66% of the time, versus 25% when visible.
  Low keypoint score (< 0.5) misses 87% versus 28%.
- An annotated **no-contact** is barely affected by occlusion: its error rate
  stays around 3–7% whether visible or hidden. A joint that is clearly far or
  clearly moving is still clearly far or moving when we cannot see it.

So the visibility problem is **hidden contacts that get called no-contact**,
not free labels going wrong. That decides where the witness belongs:

- On the **no-contact side**, an occluded or low-score joint labelled *no
  contact* might really be a *hidden contact*, so its label is less certain.
  This is where the confidently-wrong misses live, so this is where visibility
  must reach.
- On the **contact side**, we penalise only the detector: a low-score joint we
  call contact is measurably less precise (false-positive rate 0.18 vs 0.10).
  We do **not** penalise occlusion here — a joint deeply hidden behind the body
  that the readout still calls contact is usually a firm grip pressed on the
  wall behind the torso (the *lowest* false-positive rate of all). "We could
  not see it" is not evidence against a contact we already have good geometric
  reason to believe.

The two factors:

```
f_occ = clamp(1 − n_occ / 8)               1 when 0 verts hide the joint, 0 by 8
f_det = clamp((score − 0.4) / 0.35)        0 at keypoint score 0.4, 1 by 0.75
                                           (= 1 for joints with no mapped keypoint)
in_img = 1 inside the image (10 px margin), else 0
```

combine into the visibility applied to each class (floored so visibility alone
never hard-zeros a label — an unseen joint is uncertain, not certainly wrong):

```
vis_contact = clamp( in_img · f_det,          floor 0.1 )     # detector only
vis_free    = clamp( in_img · f_occ · f_det,  floor 0.1 )     # occlusion + detector
```

The governing assumption, in one line: *an unseen or unpinned joint has an
unverified position; an unverified position makes a no-contact label
(which claims we would have seen a touch) uncertain — but does not, by itself,
argue against a contact we already believe on other grounds.*

## Confidence that a contact is real

For a joint labelled *contact*:

```
conf(contact) = persistence × vis_contact × p_contact
p_contact     = p_close × max(p_still, p_pen)
```

The three geometric/motion pieces:

```
p_close = clamp((0.06 − d) / 0.06)        1 when touching, 0.5 at 3 cm, 0 beyond 6 cm
p_still = clamp(1 − m / 0.06)             1 when still, 0.5 at 3 cm drift, 0 beyond 6 cm
p_pen   = clamp((0.03 − d) / 0.04)        0 at 3 cm outside, 1 by 1 cm inside the wall
```

Read it as a conjunction of witnesses: the joint must be *near the surface*
(`p_close`), and either *still* (you hold a grip by not moving) or *visibly
pressed into the wall* (`p_pen`), with the keypoint *trusted* (`vis_contact`)
and *neighbouring frames agreeing* (persistence).

The `max(p_still, p_pen)` deserves a sentence: a gripping hand micro-adjusts a
few cm within our window, putting `p_still` around 0.5 — yet the grip is beyond
doubt. Penetration settles it, so firm penetration overrides motion doubt.

## Confidence that a no-contact is real

For a joint labelled *no-contact*:

```
conf(no-contact) = persistence × max( p_fast,  vis_free × far )
far              = max( p_far_rel,  h × max(p_far, p_far_nan) )
```

There are two kinds of evidence that a joint is *not* touching — motion and
distance — and we take the stronger. Motion needs no visibility (you cannot
grip while flying whether or not we saw the joint); distance does (an occluded
"far" joint might be a hidden touch), so distance is gated by `vis_free`.

**The motion witness.**

```
p_fast = clamp((v − 0.15) / 0.35)         0 below 0.15 m/s, 1 above 0.5 m/s
```

**Distance witness 1 — relative clearance (primary).** Absolute distance is
useless on a *floating* body, so the primary witness measures clearance *above
the scene sheet the body rests on* — its own closest joint, surface-clamped to
0, so a whole-body depth error cancels in the difference:

```
anchor    = max(0, min over joints of D)   the body's closest joint, clamped at the surface
p_far_rel = clamp((D − anchor − 0.22) / 0.06)   0.5 at 25 cm above the anchor
```

The surface clamp is load-bearing: a *penetrating* closest joint (negative `D`)
would otherwise inflate every other joint's clearance and manufacture
confidently-free labels. (A raw "distance above the closest *touching* joint"
anchor was tested — it separates free from contact slightly better on seated
bodies, AUC 0.74 vs 0.62 — but it is negative 97% of the time, so its apparent
gain comes from that penetration inflation, and it does not exist at all on a
floating body, which touches nothing. The surface-clamped closest joint is the
only anchor that both avoids the inflation and survives on floaters.)

**Distance witness 2 — absolute (fallback).** For a *healthy* body in a frame
where no joint is quite touching, the absolute distance is trustworthy and can
exceed the relative clearance:

```
p_far = clamp((D − 0.22) / 0.06)          0.5 at 25 cm, health-gated by h
```

The 25 cm threshold sits beyond the size of any climbing hold on purpose: a
hand resting on a hold the 3-D reconstruction failed to capture (the
"carved-out hold") reads *uncertain* rather than *confidently free*. Checked
against the annotation: among still, in-frame joints, the fraction actually
touching is still 29% at 12–25 cm and only 1% past 25 cm.

**Distance witness 3 — nothing within reach (fallback).** If no scene point
lies within ~1.6 m of the joint at all, that is the strongest far evidence
there is — but only when the scene is real (≥ 1000 reconstructed points) and
the body is healthy:

```
p_far_nan = 1 if D is undefined on a non-degenerate scene, else 0   (health-gated by h)
```

`far` takes the relative witness first and the two absolute witnesses as the
health-gated fallback, so a floater is judged on relative clearance and a
healthy body gets whichever of the three is largest.

## Body health: why a floating body loses its absolute distance witness

Some reconstructions fail globally: depth error places the entire climber tens
of centimetres off the wall, so *every* joint is "far from the scene". In a
climbing video that is self-contradicting — the climber demonstrably is on the
wall — so those large distances are evidence of a broken reconstruction, not of
free joints.

```
h = clamp(contact_rate / 0.1) × clamp(1 − hover_median / 0.15)
```

`contact_rate` is the share of frames where *any* joint touches (below 10% ⇒
floated); `hover_median` is the median closest-joint distance (a body hovering
15 cm off the wall throughout gets `h = 0`). `h` multiplies the *absolute* and
*nothing-within-reach* witnesses only — never the motion witness and never the
*relative*-clearance witness, which already cancels the floating-body error and
is what lets a floater's genuinely-free joints still score. (Measured: a
floating body's no-contact joints recover to ≈0.47 on relative clearance while
its meaningless absolute distances are muted to 0.)

## What stays deliberately uncertain

Low confidence is the correct answer, not a failure, where the evidence cannot
decide: a slow joint 3–25 cm from the wall (the hover band — a missed contact
and a near-miss look identical there); an **occluded or low-score joint we
called no-contact** (it might be a hidden touch); a slow, occluded joint on a
*floating* body; a joint outside the image; frames whose label was bridged
across a gap by the span-merge step (flagged in `merged`). A learner should be
able to *ignore* these cheaply — which is the point of shipping confidence.

## Where it lives

Computed in one pass by `estimate_contacts` after the labels are final, in
`tools/human_optim/confidence.py` (`contact_confidence`, `compute_body_health`,
`joint_sapiens_score`; occlusion counts in `tools/human_optim/contacts.py`,
`joint_occlusion_counts`). All constants sit in `CONTACT_CONFIDENCE_DEFAULTS`
and can be overridden in the `[contact_confidence]` table of
`configs/human_optimization.toml`. Saved per scene in `contacts_1.npz`:

| field | shape | meaning |
|---|---|---|
| `joint_contact` | (P, N, J) bool | the binary label |
| `label_confidence` | (P, N, J) | confidence in that label (this document) |
| `p_contact` | (P, N, J) | the geometry+motion contact factor alone |
| `observability` | (P, N, J) | the **visibility** factor applied to this joint's label (`vis_contact` for a contact, `vis_free` for a no-contact) |
| `merged` | (P, N, J) bool | label was bridged by span-merge |
| `joint_sdf_cm` | (P, N, J) | unguarded signed distance, cm |
| `body_health` | (P,) | the per-person health scalar `h` |
| `confidence_schema` | scalar int | version of this field set — **2** (round-11) |

(P = people, N = frames, J = 52 joints.)

`confidence_schema = 2` stamps the round-11 field set: `observability` now holds
the visibility witness (occlusion × detector × in-frame) rather than the old
scene-point-count "seen" factor, which was dropped. A future change to the
fields or their meaning increments this number.
