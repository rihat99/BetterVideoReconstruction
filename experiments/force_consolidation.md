# Consolidating the force witness, and bringing the chain back (2026-08-12/13)

Round 2 on force-based contacts. The user asked for four things: prove the
forces actually help (against a rival using only relative distance and velocity),
simplify the algorithm and write a readable doc, fix the toe pair and the label
flicker, and revive the kinematics_2 → contacts_2 → kindyn_1 chain on top of the
frame-era contacts. Three parallel workstreams, one integration pass. Judge
throughout: `scripts/score_frame_contacts.py` on the 13 annotated `data/` scenes.

## 1. Does the physics actually help? Yes — but only where it matters

We built the strongest fair physics-free rival we could: the same 12 contact
frames, the same candidate policy ("no scene point within 1 m = not a
candidate"), the same min-span / union / knee / toe policies, but the label
comes from thresholding relative distance and drift-stillness alone — the very
signals the force solve prices. It got the same tuning privilege on the 13
scenes (300-recipe grid) that the force recipe had.
(`scripts/diagnostics/contacts_control.py`, tables in `scratch/round2_control/`.)

The honest scoreboard:

- **In-sample the control wins raw F1**: .795 vs force's best .782 (the winner
  degenerates to a bare 20 cm relative-gap threshold; its stillness knob never
  binds). One number deflates all in-sample F1 here: labelling *everything* as
  contact scores .633, because climbers are usually touching something.
- **Out of sample the control collapses**: the frozen in-sample winner lands
  *below plain gates* on the 11 held-out scenes (.783 vs .796) while force
  holds (.837); its ranking quality craters (AP .43 vs force's .61).
- **At the shipped precision (.86) force wins recall everywhere**: about +6
  points overall, +17 to +24 points on the three floater scenes.
- **On pure-visual bodies** (kinematics_1 run with no contact term at all — the
  one setting where circularity cannot be argued) force dominates the control
  on precision *and* recall at every operating point.

So the defensible claim, now in `docs/force_contacts.md`: force is not the best
raw-F1 detector on the tuning set — relative distance alone is — but force finds
contacts the distance signal cannot, at high precision, and generalises where
the distance signal's tuning does not.

Side findings: the geometric-gates baseline at current HEAD is .497 (the .488
quoted before predates the kin1 smoothing commit); the old held-out and
pure-visual farms had stale pre-smoothing bodies and were rebuilt at HEAD
(`out/_expL_heldout`, `out/_expL_purevis`).

## 2. Simplification, with a parity proof

`ForceKnobs` went from 15 fields to 5; every refuted or neutral knob and its
code path was deleted (scene-derived cone axes, the per-joint effort rows, the
static no-motion mode, the candidate distance cap, the absolute-gap mode, free
frames, force smoothing). The TOML `[contact_force]` section holds four knobs.
Proof it changed nothing: `frame_contact` byte-identical on all 13 scenes
before/after, judge identical to the digit (P .860 / R .621 / F1 .721) —
`scratch/round2_simplify/PARITY.md`. (`frame_force` moves by ≤ 5e-4 N, which is
GPU-atomics noise reproducible by running the same code twice.)

Three round-1 diagnostic scripts tied to the deleted knobs were removed
(`force_recovery.py`, `force_circularity.py`, `force_heldout.py` — git history
keeps them; their evidence lives in `scratch/force_recovery/` and the round-1
record). The new doc is `docs/force_contacts.md` — the whole method in plain
sentences, evidence and caveats included.

## 3. The toe pair and the flicker

The requested signal was a negative result, measured before designing: at one
frame's resolution the scene cannot adjudicate the toe pair. The wall-normal
tiebreak disagrees with the shipped force tiebreak on 48% of arbitrated steps —
a coin flip — and a position-based variant (which side of the toe the nearest
scene point is on) is 51/52%. 39% of toe_under and 42% of toe_above labels sat
on skin facing *away* from the wall.

What does carry signal is the force's vertical direction: toe_under (standing)
labels carried upward force 96.9% of the time, toe_above (hooking) only 61.4% —
so nearly half the toe_above labels were physically standing. Shipped rule: per
foot-contact span, compare total standing evidence (Σ upward force on the under
toe) against total hooking evidence (Σ downward force on the above toe); the
winner takes the whole span. Ties keep toe_under. The union the scorer and the
chain read is invariant by construction — the rule only moves labels between
the pair. After it, toe_above labels drop 879 → 131, of which 96.9% now carry
downward force, and mid-contact toe switches go 85 → 0.

Flicker: a symmetric debounce on the union (gap-closing reusing `min_frames`,
no new knob) plus force hysteresis (`theta_hold_bw = 0.03`, the one new knob:
a label enters at 5% of body weight and survives until 3%). Contact starts per
contact-frame-minute drop 26.0 → 19.4 (−26%), mean contact length 17.1 → 26.0
frames, 12/13 scenes improve (sport_climb, the rope scene, is the exception).
Pooled F1 rises .721 → **.751** (P .850 / R .673); the label method string is
now `frames-v3`. Honest notes: hysteresis beats a single retuned threshold only
modestly (recall .673 vs .666 at matched precision, with less flicker); a plain
recall re-tune (`theta_bw` 0.05 → 0.03) would score .766 on these 13 scenes but
is a same-set re-tune with 19% more flicker — flagged, not taken.

## 4. The chain is back: kinematics_2 → contacts_2 → kindyn_1

All three stages now run on frame-era artifacts, full-auto, 13/13 scenes.

- **kinematics_2** consumes `frame_contact (P,N,12)` directly: each labelled
  contact frame is its own one-point repair patch (the old 14-limb-group
  closest-vertex machinery is gone). Per-joint confidence weighting is dropped
  (no data source; refuted anyway). Seating stays off. contacts_1's `drift_pre`
  is forwarded, never re-measured — otherwise contacts_1 would confirm itself
  through contacts_2.
- **contacts_2** is just `estimate_contacts --bodies kinematic_2` — it runs the
  full frames-v3 detection, force witness included, on the repaired body.
- **kindyn_1** optimizes pose and per-contact-frame forces together. A frame's
  force enters RNEA as a wrench on its parent joint (force + lever moment,
  recomputed live from q), so a foot's two toes stay distinct. Every force is
  confined to a 45° cone about that frame's negated skin normal — the witness's
  cone, now enforced in the trajectory optimizer (`cones = true`). The old
  wreckage had a concrete cause: the physics residual was raw N² summed over
  the clip, so its pull scaled with clip length; it is now per-frame and in
  body-weight units, and the `base_wrench` ladder is recorded in the TOML.
- The old per-joint `dynamics` stage still reads `joint_contact` and stays
  frozen this era.

Measured on the round's baseline labels (pre-toe-rule): contacts_2 took pooled
F1 .721 → .747, 9 scenes up, 3 down by ≤ .014, floaters gaining most (gtIGgs
+.094). Body-joint angular jitter is flat (2.279 → 2.278 deg/frame); the cost
is reprojection on the floaters being dragged onto the wall (gtIGgs 9.5 → 26.4
px, repaid by +.094 F1; vcRBT 12.2 → 22.9 px for only +.002 — the trade to
watch). kindyn_1 cuts the root residual 57–92% (e.g. gtIGgs 1005 → 84 N RMS)
with cone violations ≤ 1e-3 N, and the refined body keeps its contacts
(re-scored .748). Integration smoke at the final state (fresh frames-v3
labels, 2 scenes): contacts_1 .859 → contacts_2 .884 at equal precision.

**Shipped result** (all 13 `out/` trees regenerated at HEAD, full chain, zero
failures): contacts_1 pooled P .850 / R .673 / **F1 .751**; contacts_2 pooled
P .838 / R .713 / **F1 .771**. The chain improves 10 of 13 scenes on top of
the frames-v3 labels — biggest 2nwG .658 → .801 and gtIGgs .870 → .894 — and
costs a little on competition_2 (.630 → .616), vcRBT (.840 → .831) and
sport_climb (.320 → .317). competition_2 is now the weakest non-rope scene in
both readouts (precision .55 — worth an eyeball).

## Evidence

- Control: `scratch/round2_control/` (frozen recipe, all tables), farms
  `out/_expL_heldout`, `out/_expL_purevis`.
- Simplify/toe/flicker: `scratch/round2_simplify/` (PARITY.md, toe_signals,
  final_compare), farm `out/_expM_final`.
- Chain: `scratch/round2_kin2/` (RESULTS.txt, ladder, logs), farm
  `out/_expD_kin2`; integration smoke `out/_int_smoke`.
- Tests: 157 passing (`tests/test_forces.py` 19, `tests/test_kin2_kindyn_frames.py` 17).

## Open problems

1. vcRBT pays 10 px of reprojection in kinematics_2 for +.002 F1 — the repair
   helps its score little; needs an eyeball.
2. The toe rule is span-rigid (a foot cannot switch toe mid-span without a gap);
   fine on this data, revisit if real toe switches show up.
3. Knees: still excluded from the force readout, still 0 true positives.
4. sport_climb still needs a rope force (or exclusion); it is the one scene
   where flicker got worse and F1 stays low (.318).
5. The `dynamics` stage (per-joint) is frozen; retire or port it.
