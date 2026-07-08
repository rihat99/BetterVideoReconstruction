# UPDATE — round 8: confidence, final form

*2026-07-08, rev 10. Covers: the confidence explainer document, a
simplification pass over the confidence formula, the no-contact
recalibration (your relative-metric idea), and freezing the schema for the
learning pipeline. This is the last round that may change confidence.
Rounds 1–7 live in git history (revs 4–9).*

## Where we stand

Round 7 settled the floating-body question: two-thirds of floaters are VGGT
depth placing the whole person off the wall — not fixable by the body fit —
so they are now found (`features/audit/`), explained, and their distance
evidence is muted in the confidence. You agreed this is the right outcome
and set the focus: **make contact confidence right, once, and ship it.**

## The explainer you asked for

Written: **[`docs/contact_confidence.md`](contact_confidence.md)** — a
start-to-finish description of how the confidence is computed, in plain
text with the real numbers in the formulas: the four witnesses (geometry,
motion, visibility, persistence), the contact formula, the no-contact
formula, body health, what stays deliberately uncertain, how it behaves on
the corpus, and where every field lives. It describes the code as it is
today; if this round changes formulas, the document is updated in the same
commit — it is the reference you hand to your professor.

## What I measured before planning the changes

To ground the "simplify" and "still-far joints" asks, I decomposed every
still no-contact joint-frame on the re-scored corpus (1.88 M of them):

- **Still + far (≥25 cm) on a healthy body: mean confidence 0.94.** This
  case — your stated concern — already works after round 7. The few low
  ones are joints projecting outside the image (their persistence is 1.0,
  so the image gate is provably the cause; out-of-frame joints are
  extrapolations, and low confidence there is honest).
- **Still + far on a muted (floating) body: 197 k joint-frames at 0.15.**
  This is where your relative-metric idea belongs, and it is a real
  population worth recovering — a dangling foot 60 cm above a floating
  body's own closest point is definitely free, even though the absolute
  distances are corrupted.
- **The hover band (0–25 cm) is the true mass: 67% of all still negatives,
  at confidence 0.02.** Most of what *looks* like "far but low confidence"
  in the viewer is actually this band — inside the carve-safety threshold.
  Whether we can raise part of it is a measurable question (below), not a
  matter of taste.
- Joints with no scene within reach (distance = NaN): 2.3%, confidence 0 —
  currently treated as "can't tell", though "nothing within 1.6 m" is
  arguably the strongest far evidence there is.

And on the contact side, after round 7: overall 0.67 (was 0.47),
penetrating contacts 0.85. The contact class is in good shape; this round
is mostly about the no-contact class.

## The plan — three moves

**1. Factor-contribution audit (read-only, first).** Your "remove terms
that do not contribute": for every factor in the formula (the two scene
sub-factors of observability, the image gate, persistence, the
close/penetration pair, the two body-health sub-factors), measure how often
it actually binds — how much the corpus confidences and the labeled-clip
reliability change if it is removed. Deliverable: a keep/drop table with
evidence. My candidates going in: the depth-confidence sub-factor of
observability (observability averages 0.96 at contacts — at least one of
its three sub-factors is likely dead weight), and possibly one of the two
body-health sub-factors. Nothing is dropped in this move — it only produces
the list; the drops happen in move 3 on the final formula.

**2. No-contact recalibration.** Four changes, one per finding:

- **Relative clearance (your idea, adopted).** Per frame, each joint's
  distance *above the body's own closest joint*: `rel = D_joint −
  D_closest`. A global depth-placement error moves both terms together and
  cancels — so on a floating body, a joint with large relative clearance is
  definitely free even though its absolute distance means nothing. This
  becomes a third way to earn far-confidence, alongside the (health-gated)
  absolute distance; it needs no health gate itself. Target: the
  definitely-free slice of the muted population recovers from 0.15 to
  ≥0.6, without giving anything back on healthy scenes.
- **The hover band, decided by measurement.** The 25 cm threshold exists
  for one reason: a hand on a hold that the reconstruction failed to
  capture looks identical to a hand hovering 15 cm in open space — geometry
  alone cannot separate them. But the *rate* at which still joints at
  12–25 cm are actually touching is measurable on the four hand-annotated
  clips. If that rate is small, we lower the threshold (grid over
  {15, 20, 25} cm) and a large part of the 67% band becomes usable; if it
  is not small, we keep 25 cm and document why. The safety gate either
  way: the share of *true contacts* receiving confident no-contact must
  not increase.
- **"Nothing within reach" becomes far evidence.** Distance = NaN (no
  scene point within ~1.6 m) counts as maximally far — gated by the scene
  cloud being non-degenerate and the body healthy, so broken scenes don't
  benefit.
- **Body health stays, possibly simplified** (per move 1) — it still
  gates the absolute-distance witness. The motion witness stays untouched,
  as before.

**3. Simplify, freeze, ship.** Apply the move-1 drop list to the
post-move-2 formula; prune the corresponding knobs; update
`docs/contact_confidence.md` to match exactly; declare the npz schema
final for the learning pipeline (field list + a schema version stamp);
re-score the corpus (~16 min); sync CLAUDE.md. After this round the
confidence is an interface, not an experiment.

## How we judge it

Everything runs against the four hand-annotated clips first (reliability
curves: monotone both classes, top-decile accuracy not worse, and the
confident-wrong-negative rate — true contacts scored as confident
no-contact — must not rise), then the corpus re-score (distribution shifts
in the expected directions and nowhere else), then your visual sample.
Labels stay bit-identical throughout — everything here is readout. One
commit per move, your go-ahead each time.

## Honest limits

- The carved-hold ambiguity is physical, not a bug: if the reconstruction
  missed the hold, a touching hand and a hovering hand are geometrically
  identical. Image evidence could one day separate them; distance cannot.
  The hover-band decision is therefore a measured trade, not a fix.
- Out-of-image joints keep low confidence on purpose — their positions are
  extrapolated, and the honest label for an unobserved joint is "unknown".
- Relative clearance is immune to whole-body placement error, not to
  per-limb error: a floating body's *close* joints stay uncertain. That is
  correct — we cannot certify no-contact 5 cm from a wall we don't trust.

## Order of work

1. Factor audit (read-only) → the keep/drop table.
2. No-contact recalibration → labeled-clip gates → corpus spot-check.
3. Simplify + freeze: final formula, pruned knobs, explainer synced,
   schema stamped, corpus re-scored. Confidence is done.

## Outcome — round 8, 2026-07-08

Confidence is finished and frozen. Three commits, each with your go-ahead. The
labels never changed — everything here is the confidence score on top of them,
and we checked that the labels stayed bit-identical after every re-score.

**Which signals earn their place (the audit you asked for).** We switched off
each factor in the formula, one at a time, and measured the effect on the whole
361-scene corpus and on the four hand-annotated clips. One signal was truly
dead: the *depth-quality* half of "did we see this joint" — turning it off
changed nothing, anywhere, because the reconstruction near a real contact is
always good enough. Everything else earns its keep, with numbers. Full table:
[`docs/confidence_factor_audit.md`](confidence_factor_audit.md).

**Making the still-far joints confident (your main ask).** The old score judged
a joint's distance in absolute world units, which is meaningless on a *floating*
body — a reconstruction placed 30 cm off the wall makes every joint look far.
The fix is your idea: measure how far a joint sits *above the body's own closest
joint*. A whole-body placement error cancels in that difference, so a dangling
foot 60 cm above a floating climber's nearest point reads as confidently free
even though its absolute distance is garbage. On the corpus this lifts the
floating-body far joints from **0.15 to 0.48** on average (and the clearly-free
ones from 0.23 to 0.97), while healthy scenes barely move (+0.002) — the "clamp
the closest joint at the wall surface" detail is what stops it leaking onto good
reconstructions. We also let "no scene within 1.6 m at all" count as far, which
recovers another slice (healthy such joints go 0.00 → 0.94), gated so broken and
floating scenes don't benefit.

**The hover band — a measured trade, not a guess.** We wanted to know whether a
still joint hovering 12–25 cm from the wall could be called free. Measured on
the annotated clips: 29% of such joints are *actually touching* (a hand on a
hold the reconstruction missed looks exactly like a hand hovering). Past 25 cm
it drops to 1%. So 25 cm is where the ambiguity genuinely ends, and lowering it
would mislabel real contacts as free. We kept it, and now we can show why.

**What stayed the same, on purpose.** The confident-wrong-no-contact rate on the
annotated clips held at 0.29 (it comes from fast-moving joints, which we didn't
touch); the reliability curves stayed monotone; the contact side and the
moving-joint side are unchanged. The health gate keeps both its halves — the
"almost never touches" half looked redundant at first but turned out to matter
once distance got its new floating-body job.

**Frozen.** The saved fields carry a `confidence_schema = 1` stamp, and
[`docs/contact_confidence.md`](contact_confidence.md) describes the shipped
formula exactly — that is the document of record for the learning pipeline. The
corpus was re-scored end to end, 361/361.
