# UPDATE_AGENT — round 11: confidence v3 (visibility + relative clearance, simpler)

*2026-07-09, rev 11. You are the implementing agent. Read `docs/UPDATE.md`
rev 13 first — it is the contract. Rounds 1–10: git history (round 10
shipped wall-contact seating; this round demotes it to opt-in). Trust
function/key names over line numbers; re-verify every anchor before building
on it.*

## §0 Goal, in the user's words

> "When some body part [is] occluded — not just low sapiens2 score, but
> located behind some other body part as viewed from camera — confidence
> should be decreased. But sapiens2 score is also important: if sapiens2 is
> not sure about some keypoints, we also should not be sure about its
> contact."

> "When joints are not moving but not far from wall we still lower the
> confidence, but sometimes it's not justified. We should look relative to
> other joints — how far is the closest scene point to every joint — and
> then only decide. Metric should be relative, not absolute."

> "Overall confidence should not be overcomplicated, but follow some
> concrete physical assumptions behind every term."

**The round-8 freeze is LIFTED by the user's decision.** The confidence
formula may change; `confidence_schema` bumps **1 → 2** in the same commit
that changes the formula. Everything else frozen in prior rounds stays
frozen: annotations `data/*.npz`; `joint_contact` **labels bit-identical**
(this is a pure readout round — no kinematics re-runs, no readout-knob
changes like `close_thresh_m`/`still_m`); `kinematics_2`/`dynamics`;
canonical corpus `features/` (smoke to a redirected `--out-root` first; the
one canonical re-score comes at the end, no `--skip-existing`).

Deliverable = a *simpler* confidence where every term has a one-line
physical assumption, plus the two new behaviors, validated on the labeled
clips and re-scored on the corpus. Success is measured by reliability and
the confidently-wrong gates in §5 — not by any contact P/R/F1 movement
(labels don't change, so there is none).

## §1 T1 — cleanup + wall-contact default-off (first commit)

1. `git status` first; preserve anything user-edited.
2. **Flip wall-contact seating to default OFF**: `[loss_weights.scene]
   wall_contact = [0,0,0,0]` (round 10 shipped `[0,200,0,0]`). Verify the
   weight-0 convention also skips the rigid **seat phase** (`iters.seat`,
   shipped 150) — a weight-0 run must be bit-exact with pre-round-10
   behavior on one local clip (prove it). Keep all the code + knobs
   (`[wall_contact].float_gate_m` etc.) as documented opt-in; add one line
   to CLAUDE.md saying seating exists and is off by default.
3. Delete `out/_round10/` and any redirected corpus scratch from round 10
   (list contents in the commit message). Round-10 numbers live in git
   history (this file's outcome header, commits `19954b3`, `7687a86`).
4. Leave `scripts/diagnose_depth_streaks.py` and `score_contacts.py` alone.

## §2 Verified code anchors (re-verify names before building)

- **Confidence lives in the contacts stage**: `tools/human_optim/
  confidence.py` (`contact_confidence`, `CONTACT_CONFIDENCE_DEFAULTS`,
  `compute_body_health`; round 8 worked via a factor-dump helper — find it,
  `extract_factors` or similar) called from `estimate_contacts` /
  `tools/human_optim/contacts.py`. Outputs in `contacts_<tag>.npz`:
  `label_confidence`, `p_contact`, `observability`, `body_health`,
  `merged`, `joint_sdf_cm`, `confidence_schema` (=1 today).
- **Current formula (schema 1, from the round-8 wrap — re-derive from code,
  do not trust this summary):** contact side ≈ `persist · obs · p_close ·
  max(p_still, p_pen)`; no-contact side ≈ `persist · max(p_fast,
  max(abs-far, rel-clearance, nothing-in-reach))` with the absolute + NaN
  witnesses muted by `body_health` on floated bodies, the relative one not.
  The relative witness already computes `rel_j = jsdf_j − max(0, min_j'
  jsdf_j')` — your T3b builds on exactly this quantity.
- **The explainer**: `docs/contact_confidence.md` — professor-ready
  reference; MUST be updated in the same commit as any formula change, with
  the one-line physical assumption per term (the user's governing rule).
- **Sapiens scores**: `sapiens/pose.npz` — keypoints `(P,N,308,2)` + scores
  `(P,N,308)`, goliath-308 convention. Joint→keypoint mapping:
  `tools/human_optim/keypoints.py` (body block has no wrists; wrists/hands
  live in the hand blocks). Coarse-joint map `_ANN_TO_SMPLX` in `stages.py`
  (10 groups). Score range/scale: verify empirically before designing ramps.
- **Posed geometry for the occlusion test**: `estimate_contacts` already
  poses the body per frame (vertices, joint centres, per-frame cameras from
  `geometry/transform.npz`); vertex→joint assignment by nearest joint
  centre exists in `tools/human_optim/contacts.py` (`vertex_to_joint`).
  Multi-person: P can be 2 on DB scenes — decide whether occluders include
  other persons' vertices (include if cheap; if not, note the limitation).
- **Scorer**: `scripts/score_contacts.py --reliability` (shares
  `confidence.py` — round-8 rule; keep it shared so the scorer measures the
  shipped formula). Labeled clips: the 5 locals, `out/<stem>/` +
  `data/<stem>.npz`.
- **Corpus driver**: `scripts/precompute_contacts_scenes.py` (`--scenes`,
  `--out-root`, `--devices`); full 361-scene re-score ≈ 10–16 min on 2–4
  GPUs. Canonical tree gets ONE final re-score (schema 2), smoke-redirected
  first.

## §3 T2 — probes (measure before changing; commit the numbers)

**Step 0 — factor map.** Dump the current per-joint-frame factors on the 5
labeled clips + a ~30-scene corpus sample (redirected). List every term with
its measured activation range. This is the substrate for the ablation in T3c.

**Probe A — occlusion & detector belief.** Per annotated joint-frame on the
labeled clips compute: (i) geometric self-occlusion — joint centre occluded
iff some vertex NOT assigned to its own coarse group lies near the
camera→joint ray (suggest: perpendicular distance < ~4 cm to the ray
segment) at depth < joint depth − ~5 cm; batched, resolution-independent;
validate visually by coloring occluded joints on a few overlay frames before
trusting it; (ii) the joint's Sapiens score (aggregate over its mapped
keypoints — try max and mean, pick by the probe); (iii) current confidence;
(iv) label correctness vs annotation. Then: **is the automatic label's error
rate elevated in occluded / low-score buckets?** Report the 2×N bucket table
(pooled + per clip). This both justifies the witness and calibrates its
ramp. If error rates are NOT elevated → honest null → do not ship T3a
(report and stop that thread; the user owns the verdict).

**Probe B — relative vs absolute clearance.** Population: still, in-image
joints on annotated frames with jsdf in the ambiguous band (~3–30 cm).
Compare separations of annotated contact-vs-free by: absolute `jsdf_j` vs
relative `rel_j = jsdf_j − anchor` for anchor candidates {min over joints
(surface-clamped, the round-8 quantity), p10 over joints, min over *touching*
joints only (anchor exists iff some joint < close_thresh)}. Report AUCs +
the P(contact | bucket) curves. Also measure: how often no anchor exists
(nothing touching) per clip and on the corpus sample — that population keeps
the absolute fallback. Pick the winning anchor + ramp scale from this data.

## §4 T3 — the changes (each behind the probe's evidence)

**T3a — visibility witness.** `vis_j = f_occ(occlusion margin) ·
f_det(sapiens score)`, ramps calibrated from Probe A buckets (with floors —
never hard-zero a class on visibility alone). Multiplies BOTH classes,
alongside/absorbing the existing `in_img` gate (fold `in_img` into the same
witness — out-of-image is just the extreme case of "camera can't vouch").
Physical assumption (to the explainer verbatim): *an unseen or unpinned
joint has an unverified position; an unverified position gives an
unverified contact.*

**T3b — relative-first free witness.** Rework the no-contact side: primary
= ramp of `rel_j` over the Probe-B anchor when an anchor exists; the
absolute-far and nothing-within-reach witnesses become the fallback branch
for anchor-less frames (keep the `body_health` muting logic only if the
ablation says it still earns its place once relativity is primary).
Physical assumption: *touching joints anchor the scene sheet; clearance
above that anchor is trustworthy; clearance without an anchor inherits the
body's unknown offset.*

**T3c — simplification + schema bump.** With T3a/T3b in, re-run the factor
ablation (round-8 method: dumps → offline ablation, no re-scoring loops):
drop every term that is inert or cannot state a one-line physical
assumption. Target: **term count ≤ schema 1**, ideally fewer (candidates to
scrutinize: `body_health` post-T3b, the motion ramps' overlap, `obs`
sub-factors vs the new `vis`). Then, in ONE commit: formula + `confidence_
schema = 2` + `docs/contact_confidence.md` rewritten to match EXACTLY (per
term: assumption, formula with real numbers, what fires where). Update the
`observability` field semantics in the npz if `vis` absorbs it (document in
the explainer's field table).

**T4 — validate + re-score.** Labeled clips: §5 gates. Corpus: smoke 3
scenes redirected, then the full 361-scene canonical re-score (no
`--skip-existing`); compare population distributions vs schema 1 (healthy /
floater / occluded / anchored-free populations — the round-7/8 probe style).
Spot-check confidence-shaded overlays + the viewer on a few scenes.
Prepend the outcome to this file; sync CLAUDE.md (schema 2, wall-contact
off-by-default note if not already done in T1).

## §5 Gates (all hard unless marked)

- `joint_contact` (and `merged`) bit-identical everywhere — labels are
  untouched by construction; prove it on one clip + one DB scene.
- **Confidently-wrong rate must not rise, both classes** (round-8 gate,
  extended): rate of annotated-contact joints with conf(free) > 0.7 AND rate
  of annotated-free joints with conf(contact) > 0.7, pooled + per clip.
- Reliability (Spearman of confidence vs correctness, `--reliability`)
  improves or holds pooled; no clip degrades materially.
- Target populations move as intended (numbers from the probes): occluded /
  low-score wrongly-confident joints get lower confidence; still-free
  anchored joints in the 5–30 cm band get higher confidence.
- Simplicity (the user's rule, treat as a gate): term count ≤ schema 1 and
  every term has its one-line assumption in the explainer.
- Soft: corpus distributions sane vs schema 1 (no population collapses to 0
  or saturates to 1 without a physical reason you can name).

## §6 Discipline

- Per-task commits on `main` (`Round 11 T1: …`); probe numbers in commit
  messages; raw dumps/tables accumulate in `out/_round11/`.
- Measure → change → re-measure; one formula change per commit; the
  explainer never drifts from the code.
- Skepticism: this file's formula summary is a lead, not ground truth —
  re-derive from `confidence.py` first. Report nulls plainly; the user owns
  every verdict.
- Never leave experimental output in canonical `features/`; the only
  canonical write is the final schema-2 re-score.
