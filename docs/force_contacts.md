# The force witness: finding contacts by asking what holds the body up

This page explains one idea in the contact stage. The rest of the stage —
what a contact frame is, the distance and stillness gates — is in
[`contacts.md`](contacts.md); read that first if the words *contact frame*
are new.

## The problem

The ordinary way to detect a contact is to measure a distance: if a hand is
within a few centimetres of the reconstructed wall and is holding still, call it
a touch. That works whenever the reconstruction is good.

It fails completely when the reconstruction is wrong. Monocular depth is
guesswork at the centimetre scale, and on a bad clip the whole body sits a
hand's width off the wall, or an arm comes out too short. Nothing is ever near
anything, the distance gate never fires, and every contact in the clip is lost —
even though the climber is visibly hanging off the wall. A distance gate cannot
see a contact it has mislocated.

## The idea in one paragraph

A climber on a wall is not falling. Something is holding them up. If we know the
body's motion — and we do, that is what the kinematic stage produced — then
physics tells us exactly how much force is missing, and we can ask which points
on the body could have supplied it. That question never mentions distance, so it
still has an answer when the geometry is wrong. **Contact becomes a thing we
infer from load, not from proximity.**

## Step 1: the residual, or "how much force is unexplained"

Take the finished body trajectory and freeze it. Now pretend nothing is touching
it — no wall, no holds, just this body moving through empty space under gravity.
Ask: what forces would that take?

The tool for this is **inverse dynamics** — given a motion, compute the forces
that produce it. (The implementation is a standard algorithm called RNEA; we run
BetterRobot's.) It returns one number per degree of freedom of the body. Most of
those belong to joints, and a joint can supply its own force: that is what
muscles are.

Six of the numbers are different. They belong to the **root** — the pelvis, and
through it the body's position and orientation in the world. Nothing inside the
body can push on those six. You cannot lift yourself by your own hips. So
whatever is left over there had to come in from outside, through something the
body is touching.

Those six numbers are the **residual**. With no contacts at all, the residual is
essentially the body's weight — about 600 newtons, completely unexplained. That
unexplained weight is the evidence a contact exists. The whole method is the
inverse question: *which points on the body can explain it?*

## Step 2: which points, and how a force at a point counts

We only consider the twelve contact frames — the named points on the skin
(toes, heels, knees, palms, fingertips) that the contact stage already tracks.

A force `f` applied to the world at contact frame `i` contributes `Jᵢᵀ f` to
those degree-of-freedom numbers, where `Jᵢ` is that frame's Jacobian: the
standard matrix relating how the frame moves to how the body's joints move.
Using Jacobians is a deliberate choice. It puts the contact forces in exactly
the same coordinates as the residual, so no lever arms are worked out by hand,
and two frames that share a parent joint — like the two sides of one toe — stay
completely distinct.

So the residual left after contact is

    r = (what the body needs)  −  Σᵢ Jᵢᵀ fᵢ

and the solve looks for the forces `fᵢ` that make the root part of `r` as small
as possible.

## Step 3: friction cones, which are what make the answer mean anything

Left alone, this problem is useless. With unrestricted forces, any single limb
can absorb almost the entire residual on its own — put 600 N on the left palm
and the equation balances. The solve would happily do that, and the answer would
say nothing about *which* limb touched. An earlier attempt at this failed for
exactly that reason.

The fix is to give every frame a **friction cone**. A cone is the set of pushes a
real surface can deliver: outward along the surface, plus up to 45° of sideways
friction, and never a pull — a wall can push you, it cannot grab you. The cone at
each frame points the way that patch of skin faces, negated: skin that is
touching a surface faces *into* it, so the surface pushes back the other way.

Now a frame can only explain the part of the residual it is physically oriented
to explain. A palm facing left cannot hold the body up. The answer becomes
informative because most of the possible answers have been ruled out by geometry
the body itself provides.

Two details make the cones cheap and exact. Each cone is approximated by a
four-sided pyramid inscribed in it, and the unknowns are the four nonnegative
weights on that pyramid's edges. Any nonnegative mix of edges is inside the cone,
so the constraint holds by construction — there is no penalty term and nothing to
tune. And because zero weights mean exactly zero force, "this frame is not
touching" is always reachable.

The cone direction comes from the frame's own skin normal, never from the
reconstructed scene. That is on purpose: the skin normal is available even when
the body is in completely the wrong place, which is the case this method exists
for.

## Step 4: two prices, so the load lands somewhere sensible

Cones narrow the answer but do not pin it down — several frames can usually share
the load. Two costs break the remaining ties, both physically motivated.

**The stillness price.** Force on a frame that is moving is expensive. A limb
sliding through the air is not carrying your weight. The speed used is the net
drift of that frame over a short window, and — this matters — it is the drift
measured on the body *before* any contact enforcement acted on it (`drift_pre`,
saved by the kinematic stage). Using drift measured afterwards would be circular:
the optimizer holds labelled contacts still, so it would be voting for its own
labels. A body without `drift_pre` is a hard error, never a silent
re-measurement.

**The gap price.** Force on a frame far from the wall is expensive. But this is
priced **relative to the nearest contact frame of the same step**, not as an
absolute distance — and that relative-ness is the entire floater rescue. On a
body where *every* frame is far from the wall, absolute pricing makes all force
unaffordable, the solve leaves the body's weight unexplained, and nothing crosses
the threshold — on precisely the clips this method exists to save. Relative
pricing keeps the price's real job (prefer the nearer frame) while insisting the
weight still lands somewhere.

There is also a small plain cost on force magnitude, which just says: do not
invent load you do not need.

## Step 5: no data is not the same as far away

When a step's contact frames have no scene point anywhere near them — the search
box is a metre wide, so this means the cloud genuinely has nothing there — that
is **missing evidence**, not "far from the wall". Such a frame is not allowed to
carry force at all.

This distinction is not pedantic. The nearest-neighbour search reports a huge
sentinel number when it finds nothing. Treating that number as a distance once
silently ruined an entire comparison: relative pricing subtracted sentinel minus
sentinel, got zero, and handed those steps completely free force. One policy,
applied before any pricing: no scene evidence, not a candidate. The stillness
witness gets the same treatment: a step whose pre-enforcement drift is NaN
(an untracked window, or a person detection never ran on) has no stillness
evidence and is not a candidate either. Everywhere else
in this pipeline a hole in the cloud reads as *unknown*, and unknown never
manufactures a contact.

## Step 6: solving it

Minimising the squared residual plus the prices, over nonnegative cone weights,
is a **convex** problem — one bowl, one bottom. So plain projected gradient
descent (Adam with a decaying step, clamping the weights at zero each iteration)
reaches the true optimum rather than some local dead end. The stage reports a KKT
number afterwards, which is near zero exactly when the solve really converged
rather than merely stopped.

Everything is solved in units of body weight: forces are fractions of it and the
objective is divided by its square. That is a pure rescaling and moves no
optimum, but it turns numbers that ran to 10⁵ into numbers around 1, which is the
difference between the solver converging and its line search giving up on the
first step. It did, silently, before this was done.

The whole solve takes a few seconds per scene.

## Step 7: from forces to labels

A frame is **loaded** when it carries more than `theta_bw` of body weight
(default 5%). Three rules turn that into a label.

**Hysteresis.** A frame enters contact above `theta_bw` and keeps the label until
its load drops below `theta_hold_bw` (default 3%). One threshold makes a load
hovering at the line switch on and off every other frame; a second, lower
threshold means the label only lets go on a real drop. It can only *lengthen* a
contact that already started — it never starts one.

**A contact lasts, and so does a release.** Runs shorter than `min_span_s`
(default 0.1 s, converted to frames as ceil(s·fps)) are dropped as flicker.
Gaps shorter than the same span are filled, because no real grip is released
and retaken that fast. One number, both directions.

**Union with the geometric label.** The force label is added to the distance-and-
stillness label; force only ever adds. Knee frames are excluded from the force
side — measured, their force labels are almost all wrong.

**Far labels need stillness.** The relative gap price above is what lets force
label a body that is consistently misregistered — the climbing floater rescue,
where the real labels sit a median 44 cm from the wall. The same mechanism can
fail on sports with flight phases and bad depth: whenever the kinematics carry
unexplained acceleration, the solve *must* park the residual somewhere, and
small force scraps can label contacts on limbs flying through the air. What
separates the two cases is not distance — a real beam stance whose beam cloud
sits half a metre off looks exactly as far as a fake mid-air scrap (an absolute
distance gate was tried in 2026-08 and, measured later on the annotated scenes,
killed the floater and beam-routine rescues wholesale) — it is **stillness**. A
really supported limb holds still however badly the scene around it is
registered; a flying limb does not. So the rule
(`[contact_force].far_still_x`, one value for every sport): within the
detector's own trust radius (`[contacts].max_dist_m`) a loaded frame labels no
matter how fast it moves — a sprint plant that carries weight is a contact.
Beyond it, the label also needs the frame's drift to stay under
`far_still_x × still_m`. The bar is 4× looser than the detector's own stillness
bar because a badly reconstructed body jitters even while gripping. The solve
itself is untouched — every candidate still participates, the residuals and the
toe rule still see the full allocation — only far, fast frames are refused a
label. Measured 2026-08-25: the high_jump / 1FGkg mid-air floods drop to zero,
the gtIGgs floater and both balance-beam clips keep their far labels, and
climbing F1 pays .007 against no rule at all (vs .13 for the absolute gate).

The only place the force can *remove* a geometric label is the toe pair, next.

## Step 8: the toe rule — which side did the load go through?

A foot's contact frames come in two sides facing opposite ways through the foot.
The underside (`heel` + `toe_under` in the legacy 12-frame set; `heel` +
`ball_inner` / `ball_outer` / `big_toe` in the full-body set): the foot
**stands** on a hold. `toe_above`, the top of the toe: the foot presses it
**against** a hold — a toe hook, or the whole body inverted in a bat hang. At
most one side is touching, so a label naming both is wrong — and both readouts
do produce that, because each keeps a single side but they can keep different
ones.

The decision is made **once per contact**. A contact is one unbroken run of
"this foot is touching" (any toe frame on). Across that whole run each side is
scored by how much force the solve put through it:

    standing  =  Σ |force on the underside frames|
    hooking   =  Σ |force on toe_above|

The bigger sum takes the entire run; exact ties keep the underside, since
standing is much the more common contact and the result then never depends on
evaluation order. Within the winning side each frame keeps its own label — which
underside points touch is real information (edging vs a flat foot, heel on or
off); run steps where only the losing side was labelled go to the winner's
most-loaded frame.

Why this is the right reading: every solved force lives in a cone about its own
frame's skin normal, so the solve can only put load through the side whose
orientation can explain the body's motion. "Which side carried the load" is
therefore already a normal test — using the body's normals, which are exact,
instead of the cloud's, which are not. On the eyeballed disputed contacts it
called all eight correctly, where each of the two earlier readings managed four:

* **The force's vertical sign** ("standing pushes up, hooking pulls down" — the
  first version of this rule) renames every bat hang to a stance: an inverted
  climber's hooked toe holds the body weight *up*, which the sign test declared
  impossible for a hook. boulder_3's whole inverted sequence wore `toe_under`
  because of this.
* **The cloud's normal at the nearest scene point** votes "top of toe" for
  ordinary stances on small holds: the nearest cloud points are the wall plane
  behind the hold, and a standing toe's upper surface naturally faces the wall.
  It is trustworthy only on large volumes the cloud actually resolved — exactly
  where it agrees with the load anyway.

Two properties make the rule safe. The patch cannot change while the foot stays
on the hold, by construction — the old per-step tiebreak switched patch 0.38
times per foot-contact. And the rule only ever *moves* labels between the two
sides of one foot — the union over its toe frames, which is what "this foot is
in contact" means and all that the scorer and downstream stages read, comes out
unchanged. (That invariance also means the F1 table below cannot see this rule;
its evidence is the viewer and the physics stages downstream, which inherit the
frame's normal for their friction cones.)

## What it is worth

All numbers are pooled over the 13 annotated scenes, scored per coarse joint
against the manual annotations.

| readout | precision | recall | F1 |
|---|---|---|---|
| distance + stillness gates only | .866 | .348 | **.497** |
| + the force witness | .860 | .621 | **.721** |
| + the toe rule, debounce and hysteresis | .850 | .673 | **.751** |

One number belongs next to every F1 here: labelling *every* scorable cell as
contact scores .633 on these scenes, because climbers really are in contact most
of the time. Pooled F1 therefore flatters loose detectors, and comparisons only
mean something at matched precision.

The first jump is almost entirely recall at nearly unchanged precision — the
force witness finds contacts the gates never saw, and does not invent many. The
three known floater scenes went from .02 / .09 / .26 to .81 / .79 / .80 with the
force witness, and to **.87 / .85 / .82** with this round's rules. Every one of
the 13 scenes improved at both steps.

The flicker work is measured separately, as label stability: contact starts per
contact-frame-minute of tracked video went **26.0 → 19.4** (−26%) and the average
contact got 52% longer (17.1 → 26.0 frames). Twelve of thirteen scenes improved.

### Caveats, in full

- **Held-out scenes gain much less.** On eleven never-tuned scenes from the
  ClimbingVideos database, the same frozen recipe went .789 → .837 pooled
  (.725 → .775 averaged per source video), and 2 of 5 source videos got slightly
  worse. Those scenes have good geometry, so the distance gate already worked.
  The force witness pays where the reconstruction is bad — the one held-out scene
  with poor geometry went .519 → .801.
- **On `gtIGgs` the physics is not the whole story.** A geometry-only control
  that simply thresholds the same *relative* distance recovers that floater about
  as well as the force solve does. Relative pricing is doing real work there
  independently of the dynamics, and we say so.
- **A physics-free control can beat the witness on raw in-sample F1 — and then
  falls apart.** The strongest rival we built (a bare 20 cm relative-distance
  threshold, run through the exact same candidate, span, union and toe policies)
  reaches F1 .795 on the 13 tuning scenes, above force's best .782. But that
  same frozen recipe drops *below the plain gates* on the eleven held-out scenes
  (.783 vs .796) while the force recipe holds (.837), its threshold-free ranking
  quality collapses out of sample (average precision .43 vs force's .61), and at
  the shipped precision force carries about six recall points more everywhere —
  seventeen to twenty-four more on the floaters. On bodies that no contact term
  ever touched, force is better on precision *and* recall at every operating
  point. So the honest claim is not "force gives the best F1 on the tuning set"
  — it doesn't — but "force finds contacts the distance signal cannot, at high
  precision, and generalises where the distance signal's tuning does not."
  (`scripts/diagnostics/contacts_control.py` reproduces all of this.)
- **`sport_climb` is reported separately, always.** The climber is on a rope, and
  a rope force is not in this model — only about a quarter of that scene's
  residual is explainable at all. It must never drive a conclusion.
- **Hysteresis is a modest win, not a magic one.** Against a single threshold
  retuned to the same precision (0.04 body weight), hysteresis gives slightly
  more recall (.673 vs .666) with slightly less flicker (19.4 vs 20.2 starts per
  minute). Most of the F1 movement between the last two table rows is the recall
  shift, not the hysteresis mechanism.

## The knobs

All in `[contact_force]` of `configs/human_optimization.toml`. There are four,
and only the first two are worth touching.

| knob | default | what it does |
|---|---|---|
| `theta_bw` | 0.05 | fraction of body weight at which a frame counts as loaded. **The recall knob.** |
| `theta_hold_bw` | 0.03 | the label holds until the load falls below this. Set equal to `theta_bw` for a single threshold. |
| `gap_dstar_m` | 0.05 | how quickly the gap price bites, in metres |
| `stillness_vstar_mps` | 0.2 | how quickly the stillness price bites, in m/s |

`min_span_s` is shared with the geometric readout and lives in `[contacts]`.

Everything else about the solve is fixed in code, because it was measured and it
does not want tuning: a 45° friction cone, 2000 solver iterations, a small
magnitude cost, and a 0.2 m lever arm that converts the root's torque rows into
force units so all six residual rows are one wrench in one unit.

Several knobs that used to exist are gone, having been measured neutral or worse:
a scene-derived cone direction (same result to within .002 F1, for a pile of
machinery), a weight on the non-root residual rows (neutral, shipped at 0), a
static no-motion mode, a maximum candidate distance, an absolute-gap mode,
frames exempted from having a cone at all, and force smoothing.

## Where the code is

- `tools/human_optim/forces.py` — the solve (`solve_person_forces`), the cones,
  and the label rules (`force_labels`, `resolve_toe_conflicts`, `hysteresis`,
  `min_span`, `close_gaps`).
- `scripts/stages/estimate_contacts.py` — the stage that runs it, unions the two
  readouts and writes `contacts_<tag>.npz`.
- `tests/test_forces.py` — cone feasibility, the toe rule, the span rules.
