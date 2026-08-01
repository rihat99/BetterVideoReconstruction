# Why the frames era lost contact — and what was done about it (2026-08-01/02)

The contact-frame redesign dropped the hand contact rate from ~0.65–0.69 to
~0.10–0.15 (mean over the 13 `data/` scenes, old per-joint labels vs frame
labels). This round measured why, tested two proposed fixes, and shipped what
survived. Unlike the older records in this folder, the evidence is kept: the
experiment trees named below still exist, and every claim cites one.

## Where the misses actually come from

A gate-cascade over the stored detector signals (how many near-wall steps
survive each gate in turn) splits the drop into four separate causes:

| cause | what happens | scenes it dominates |
|---|---|---|
| normal gate on finger frames | the fingertip reaches the wall (2.5–3.5 cm) but its baked pad normal points away (`normal_dot` −0.5…−0.76), because the fingers are pinned at the converter's hand pose, which is unreliable | boulder_3, 2nwG, 9Rvr, competition_1/2, CRquJ8H1QLI |
| whole-limb depth error | the hand lands 2–6 px from its keypoint in the image but 8–48 cm short **along the viewing ray** — the old floater/placement class, not a body problem | gtIGgs, vcRBT |
| palm at grip distance | a gripping hand holds the fingertip on the hold and the palm 10–17 cm off it — real anatomy, correctly not labelled | everywhere |
| stillness gate | `drift_pre` 0.05–0.06 m vs the 0.025/0.03 thresholds blocks windows where geometry passes | boulder_3 early frames |

Key diagnostic: an error along the viewing ray is invisible in the image; a
limb that is genuinely too short is not. boulder_3's "short" left hand
reprojects 0.4–2.7 px from its Sapiens wrist keypoint — it reaches.

## Experiment A — penetration on the contact frames (shipped, `063b111`)

The scene-penetration loss moved from 1000 subsampled mesh vertices to the 12
contact frames, weight 4000 → 12000 by sweep (4 scenes × {1000, 4000, 12000}).
At 12000 the residual frame penetration reaches the term's own 1 cm dead-zone
edge (0.9–1.2 cm, was 2.3–3.5 cm) for ≤0.5 px reprojection cost, and kin1 runs
~18 % faster. Evidence: `out/_expA_framepen/{vbase,w1000,w4000,w12000,dnocham}/`.

**The hypothesis that strong penetration would stop `toe_under`+`toe_above`
firing together is refuted.** The dead zone (1 cm) sits inside the contact band
(3 cm), so penetration is exactly zero over the whole region where contact is
read: a simulated perfect push of every frame to the dead-zone edge breaks 0 of
132 both-fire steps. Stronger penetration made both-fire *worse* (133 → 197)
by lifting buried frames into the distance-gate pass band.

The real cause is geometry: the two toe frames' normals are 148.4° apart, so
their 90° acceptance windows share a 32.5° overlap in which both frames accept
the *same* surface (83/99 of competition_1's both-fire steps are same-surface
matches at 78–81°). Tightening `[contacts].normal_max_angle_deg` to ≤ 74°
(half the pair separation — a derived number, not a tuned one) makes the two
windows disjoint and zeroes both-fire on competition_1 (99→0) and boulder_3
(22→0) while keeping 8 genuine two-surface wedges on 33ODI. Cost: total
contact instances drop 12–31 % per scene, and part of that is likely true
contact against surfaces the camera never saw (a hold's occluded top face
matches only the wall behind it, at ~80°). **Not shipped** — held until the
normal-signal quality question below is settled.

A side probe: chamfer off + frame penetration runs 3.1× faster and improves
reprojection on 3/4 scenes, but the body drifts off its own depth cloud on the
hardest scene — chamfer is still the only tie to measured depth, so the mesh
cannot be fully dropped from kin1 yet.

## Experiment B — "the converter shortens limbs" (acquitted, code reverted)

The suspicion: MHR→SMPL-X conversion loses ~2 cm of forearm (MHR joints say
26.9 cm, the converted body 24.5 cm). Measured verdict: **that gap is a
joint-convention artifact, not a fitting error.** Applying SMPL-X's own joint
regressor to the *MHR mesh itself* already gives 25.3 cm — the difference
exists before any fitting. Three definition-free tests acquit the converter:
distal-surface reach (same vertex ids both meshes): all 24 limb medians within
±0.9 cm, mixed sign; the barycentric topology map: −0.01…−0.04 cm; projected
2D forearm vs Sapiens over 12 scenes: MHR joints 1.089 (long), converter 1.026
(neutral), kin1 0.968 (short).

A bone-length preservation term was implemented and swept anyway: it closed
the bone gap 90 % and made everything real worse (mesh fit 3× worse, ~1 cm of
genuine reach lost, downstream body noisier, no contact change). Reverted; the
measurement is the deliverable. Evidence: `out/_expB_bonelen/{w0,w1,recheck}/`.
No capacity wall either: 10 betas span a 21–29 cm forearm.

Consequence: the residual ~3 % 2D shortening appears **inside kin1** with bone
lengths frozen — a fixed-length limb can only project shorter by pitching
along the viewing ray. "Limbs too short" is a pose/depth symptom, not a shape
problem, and MHR bone scales are not the instrument for it.

Note on `FINDINGS.md` in this folder: its E1–E3 argument was found unsound by
its own commissioned adversarial review (category error between 2D ratios and
3D lengths; endogenous regression; wrong "below chance" arithmetic), so it
never validly closed the limb question. The question is now closed properly,
by the definition-free measurements above — same conclusion, sound route.

## Experiment C — give the hands evidence (H1 shipped `d9729f7`, H2b opt-in)

The hands had no image witness at all: `[keypoints].include_hands = false`
discarded all 21 Sapiens keypoints per hand, and the fingers were hard-frozen
at the converter pose. Three arms on 4 scenes, evidence `out/_expC_hands/`:

| arm | change | result |
|---|---|---|
| H1 | fingertip keypoints on, fingers stay frozen | fingertip 2D −23…−31 %, wrist −13 %, hand-frame contacts +4.4 %, ≤0.25 px cost elsewhere. **Shipped as default.** |
| H2 | also unfreeze the 30 finger quats | inert: the pose prior pins the fingers to the converter pose 11:1 against the fingertip observation |
| H2b | H2 + fingers dropped from the pose prior | best on every axis: contacts +7.9 % (+12.4 % on hand frames), fingertip 2D −42 % worst-case scene, stable, baseline runtime. **Opt-in** via `[stages.kinematics_1].freeze_fingers = false`. |

Why H2b is not the default yet: a fingertip *pixel* has almost no leverage on
rotation of the pad about the finger axis, and that rotation is exactly what
the normal gate needs (~130° on boulder_3). Pooled finger-frame
`normal_dot ≥ 0` moved only 0.299 → 0.338, and boulder_3's early left-hand
window stayed at zero contact in every arm. Fitting the hand needs a real hand
observation first — candidate levers: fingertip segment pairs instead of lone
tips, a hand-pose prior, or trusting SAM-3D-Body's MHR hand pose directly.

## Open decisions

1. `normal_max_angle_deg` 90° → 74°: zeroes both-fire by construction, costs
   recall partly on occluded-surface true contacts. Decide after viewing.
2. `freeze_fingers = false` (H2b) as default: better everywhere measured, but
   fits fingers on marginal evidence; safer after a real hand observation.
3. The depth/placement class (gtIGgs, vcRBT): unaddressed here; it is the old
   floater problem and needs its own round.
