# UPDATE — round 9: taming bad VGGT depth on the human

*2026-07-08, rev 11. Covers: a tool that measures the bad-depth problem, three
cheap fixes to how the optimizer and the scale stage consume VGGT depth, and a
menu of deeper fixes for a second wave. Round 8 (contact confidence) is frozen
and untouched. Rounds 1–8 live in git history (revs 4–10).*

## The problem

Sometimes VGGT gets the human's depth wrong, and when it does, the error is
along the camera ray: at thin parts like hands and feet, the depth map smears
across the silhouette between the body and the wall behind it, so the
backprojected point cloud grows long "streaks" pointing away from the camera.
The SMPL-X body physically cannot fit a hand that is stretched over a metre of
depth, so the chamfer term (the loss that matches the body surface to those
depth points) wrenches the pose instead. This is a known, measured property of
learned depth models — errors concentrate at object boundaries — not a VGGT
quirk.

It matters in three places: the kinematic fit (chamfer), the metric scale
stage (which reads VGGT depth under the body to size the world), and fuse
(which turns the depth into the clouds everything else consumes).

## What I found in our own code before proposing anything

I audited the whole depth path and spot-checked the important claims by hand.
Three findings reframe the round:

1. **Chamfer has no outlier protection.** It is a plain nearest-neighbour
   distance — no robust kernel, no cap. Every other data term already has one
   (reprojection uses Geman-McClure, penetration uses Huber). A streak point a
   metre down the ray pulls with its full distance. Every published system
   that fits a body to noisy depth wraps this term in a saturating kernel; we
   are the outlier.
2. **Your vertex weighting is wired correctly — but it sits on the wrong side
   of the loss.** The weights only shape the *body→cloud* direction. The
   *cloud→body* direction, where every depth point (streak or not) pulls on
   the body with equal weight, has no weighting at all. That is why the
   extreme-weights sweep changed nothing: the knob isn't connected to the
   failure. Good news: this is not a bug to fix but a lever we haven't built
   yet.
3. **The scale stage leans on exactly the least reliable points.** Its
   extremity-up weighting shifts the robust median toward hand/foot vertices —
   where the streaks live. And the flying-pixel edge filter that protects the
   fused clouds is never applied inside the scale stage at all.

Two supporting facts. A 2026 study measured VGGT's depth-branch confidence as
badly overconfident exactly at geometry edges — so our confidence gating,
the current defence, is documented to be the wrong tool for streaks. And a
small doc-rot note: CLAUDE.md still mentions a "depth-push" term that no
longer exists in the code.

## What the field does

The consensus recipe for fitting bodies to noisy depth (PROX and its
descendants) is: robust kernel on both chamfer directions plus visibility —
they do no point filtering at all and let the kernel absorb the outliers.
Meanwhile, the world-frame video systems (TRAM, WHAM, SLAHMR) simply never
trust per-pixel depth on a moving person. We keep the chamfer because it is
what seats absolute depth in our pipeline — which makes robustification
mandatory rather than optional.

## The plan

**T0 — build the streak meter first.** A small diagnostic that, per frame and
per body part, measures how stretched the human cloud is along the ray
compared to the part's physical thickness, and flags "streaked" part-frames.
Run it on the five labeled clips and the seven showcase corpus scenes. Its
most important output is the go/no-go slice: on the labeled clips, are
contact mistakes concentrated in streaked frames? That tells us how much of
the precision/recall gap this whole round can possibly close.

**T1a — robust chamfer.** Wrap both chamfer directions in a Geman-McClure
kernel with a physical scale (roughly 10–20 cm): points within that range
pull normally, points far beyond it stop pulling. One subtlety: the early
"root" placement phase relies on chamfer to drag the body from a possibly
distant start, so the kernel must stay soft (or off) there and tighten only
in the later phases.

**T1b — weights on the cloud side.** Give each depth point a reliability
weight — its VGGT confidence times how far its pixel sits from the person
mask's edge (streaks are born at the mask boundary) — and use it in the
cloud→body direction that is currently uniform. This is the mirror image of
the weighting you already built, on the side that can actually mute streaks.

**T1c — fix the scale stage.** Try uniform and extremity-*down* weighting (a
ruler should be read at its most reliable marks — the torso), and apply the
same flying-pixel edge filter the fused clouds get. Scale changes flow
through fuse into everything, so each variant re-runs the downstream chain.

Each fix ships behind a knob whose default reproduces today's behaviour
exactly, gets swept on the labeled clips, and is adopted only if the numbers
clear the gates below. Winners get combined and confirmed in one final run.

## Second wave (decided after we see T1 numbers)

- **Depth corridor from the init body.** SAM-3D-Body predicts its own metric
  depth, independent of VGGT, and its reprojection is ~5 px. A hand cannot be
  a metre behind the torso — so squeeze (softly clamp) cloud points back into
  a ±30–40 cm corridor around the init body's depth instead of deleting them.
- **A per-point streak score.** Our normals code already computes the local
  shape statistics needed to detect "filament aligned with the view ray" —
  it just throws them away. Expose them and fold the score into the T1b
  weight.
- **Wall-contamination filter.** Edge pixels are mixtures of hand and wall;
  the contaminated ones sit at wall depth. Dropping human points that
  coincide with the scene cloud sounds right — but climbers' hands genuinely
  touch the wall, so this one needs a guard (only drop points far behind the
  init body) and careful contact-metric watching.
- **Along-ray-aware chamfer.** The principled version of "squeezing": measure
  the along-ray part of each residual in log-depth (where monocular error is
  actually well-behaved) and keep the sideways part metric. Nobody has
  published this for body fitting — worth trying if the simple kernel leaves
  streak damage on the table.
- **Per-part chamfer gating.** Where T0 flags a part-frame as hopelessly
  streaked, switch chamfer off for that part in that frame and let the
  init-anchor, smoothness and priors carry it.

## What we deliberately skip this round

The second-witness ideas (Sapiens pointmap/normals as a human-specialised
depth source, recovering VGGT's better-calibrated point-map branch) are the
biggest levers but need adapter/schema work and corpus re-runs. They wait
until we know how much the cheap layers recover.

## How we judge

The main yardstick is contact quality on the five labeled clips in `./data`
(boulder_1, boulder_3, competition_1, competition_2, sport_climb): precision,
recall, F1 and false-positive rate from `score_contacts.py`, against the
long-standing baseline (pooled P 0.896 / R 0.678 / F1 0.772 / FPR 0.066;
sport_climb excluded from the pool as always). Raising recall without paying
precision is the goal. As in every round: recall is the number a stronger
pull toward the wall inflates for free, so precision and FPR are the honest
signals, and the annotation files stay bit-identical.

Guards: reprojection error must not degrade, no new floaters, and the streak
slice from T0 should visibly improve — that is the mechanism check, not just
the headline number.

The seven corpus scenes you named (PfwFWSoPF_k_0026, 2nwG1Qa5k_c_0115,
9RvrpYoWDoo_0000, CRquJ8H1QLI_0001, 3DAgLpPXVCs_0132, gtIGgs_y2nY_0024,
vcRBT9Q9jXM_0029) are the qualitative showcase: no labels there, so they get
the streak meter, before/after overlays and viewer checks. Experimental runs
on them are redirected so the canonical round-8 corpus outputs stay intact.

## Honest caveats

- If T0 shows the labeled clips barely contain streaked frames, the headline
  pooled numbers won't move much even if the fix is right — then the corpus
  scenes and the streak slice carry the verdict, and you should read the
  round that way rather than as a null.
- The robust kernel trades a failure mode: it protects against streaks but
  weakens chamfer's long-range pull everywhere. The phase-aware schedule is
  there to keep the root placement safe; if depth seating degrades on healthy
  clips, the scale parameter is the dial.
- Scale-stage changes shift the whole world's size slightly on every clip,
  including healthy ones. The stability of the per-frame scale estimates
  (their spread) is the early-warning number to watch before looking at
  downstream metrics.
