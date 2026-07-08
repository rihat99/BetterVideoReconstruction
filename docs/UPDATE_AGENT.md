# UPDATE_AGENT — round 8: confidence final form (simplify, recalibrate, freeze)

*2026-07-08, rev 8. You are the implementing agent. `docs/UPDATE.md` rev 10 is
the human-facing plan — read it first. This file is the handoff: verified
state, code anchors, task specs, gates. Rounds 1–7: git history. Trust
function/key names over line numbers. This is the LAST round that may change
confidence — after it, the schema is a frozen interface for a learning
pipeline.*

## §0 State (verified 2026-07-08 by the planning session)

**Repo**: HEAD `a6c009f` on `main` (round 7 complete: `ecaeb0d` layout,
`487e567` audit, `7a6796f` autopsy, `596ab76` confidence v2, `a6c009f` wrap).
Working tree clean EXCEPT the user's uncommitted edits in
`tools/viewer/{app,loading,scene}.py` (preserve, never revert) and the new
untracked `docs/contact_confidence.md` (the explainer the planning session
wrote — commit it with your first commit, and keep it in sync with every
formula change you ship).

**Corpus** (DB root `/data3/rikhat.akizhanov/better/data/ClimbingVideos`):
361 scenes, post-T1 layout — `features/human_optim/<aa>/<bb>/<scene>/`
holds `kinematic_1.npz` + `contacts_1.npz` + `uncertainty_1.npz`. Contacts
re-scored with confidence v2 (596ab76) 361/361. Audit artifacts:
`features/audit/{bodies,autopsy,cloud_gap}.csv` + histograms. Round-7
verdicts: 80/363 bodies flagged floating; causes 65% cloud-off-wall / 22%
body-off-own-cloud / 10% broken upstream / 2% limbs; "buried" bodies were a
distance-sign artifact (seated, not behind the wall); the limb thread is
CLOSED (conversion loses ~2 cm forearm / ~5 cm thigh vs MHR but uniformly —
floaters are no shorter than healthy bodies; knee-definition mismatch, not
recoverable length).

**Labeled clips** for validation: the 4 hand-annotated tuning clips in
`out/<stem>/` + `data/<stem>.npz` (sport_climb excluded from pools by the
scorer default). Label baseline (unchanged since round 4): pooled P 0.896 /
R 0.678 / F1 0.772 / FPR 0.066. Labels are SACRED this round: bit-identical
throughout — everything is readout.

**Confidence v2 behavior — planning-session probes on the re-scored corpus
(re-derive anything you build on):**

- conf(L=1): mean **0.674** (was 0.465), >0.5: 71%; penetrating (<−1 cm,
  46% of contacts): **0.853**. The contact class is healthy — this round
  is mostly the no-contact class.
- Still no-contact joint-frames (speed < v_lo, 1.88 M total), by cause:
  - far ≥25 cm, healthy body (h ≥ 0.8): 20.0%, mean **0.94**, <0.3: 0% —
    the round-7 far branch WORKS; its rare low scores are out-of-image
    joints (verified: their persistence is 1.00, so `in_img` is the only
    possible zero).
  - far ≥25 cm, muted body (h < 0.8): 10.5% (197 k), mean **0.15** — the
    recovery target for relative clearance (T2a).
  - hover band 0–25 cm: **67.1%**, mean **0.02** — the real mass behind
    "non-contact joints get low confidence" (T2b decides how much is
    recoverable).
  - jsdf NaN (no scene within reach): 2.3%, mean 0.00 (T2c).
  - Fast (≥0.5 m/s) reference: mean 0.97.
- `body_health`: 310/363 persons ≥0.8, 17 in 0.3–0.8, 36 <0.3.

**Code map** (anchors verified today):
- `tools/human_optim/confidence.py` — `CONTACT_CONFIDENCE_DEFAULTS` :44
  (14 knobs incl. round-7 `pen_ref_m`, `health_cr`, `health_hover_cm`);
  `_sat` :62; `_scene_neighbourhood` :66 (memory-bounded chunked cdist);
  `_persistence` :90; `compute_body_health` :102 (h = h_contact · h_hover;
  contact_rate/0.1 and 1 − med_hover/15 cm); `contact_confidence` :131 —
  p_close :156, `jsdf_f` nan/±inf→0 :157, p_far :158, p_still :159,
  p_pen :163, `p_contact = p_close · max(p_still, p_pen)` :164,
  p_fast :166, obs :171 (= sat(n_near/20) · sat(conf_near/0.3) · in_img),
  persist :173, conf_pos :175, `conf_neg = persist · max(p_fast,
  in_img · h · p_far)` :179.
- `scripts/stages/estimate_contacts.py` — imports `compute_body_health` +
  `contact_confidence` :64–65; `[contact_confidence]` TOML pass-through
  :93–95; unguarded jsdf :198; drift_pre preference :210–213; gate :214;
  `merged` :221; body_health_all :178. Saved fields: joint_contact,
  joint_sdf_cm, merged, p_contact, observability, label_confidence,
  body_health, object_ids, joint_names, parents, valid_mask, fps + knob
  scalars.
- `scripts/score_contacts.py` — `--reliability` :719, `_reliability_one`
  :566. VERIFY it imports the confidence module rather than duplicating
  formulas before you change anything; if it duplicates, unify first.
- `scripts/audit_scenes.py` (round 7): per-(scene,person) metrics + flags —
  reuse its loaders/conventions for corpus sweeps.
- Guarded vs unguarded distance: `tools/human_optim/contacts.py:148`
  `joint_guarded_min_sdf` (guard `(dmin < max_dist_m=0.10) & (conf >
  thresh)`, else `torch.finfo.max` — `nan_to_num(posinf=…)` does NOT catch
  finfo.max; the `_sat` clamps are why current code is safe). Unguarded
  `lift_joint_contact` jsdf: finite to ~1.6 m, NaN = no near-surface vertex.

**The user's asks this round** (do not re-litigate): (1) the explainer doc —
written at `docs/contact_confidence.md`, keep it exact; (2) simplify —
remove factors that do not contribute; (3) still-far no-contact joints
should be confident, using a *relative* (to other joints) metric so
floating-body corruption is excluded; (4) confidence ships to a learning
pipeline after this round — LAST chance for schema changes.

## §1 Process

One task = one commit (with the explainer-doc sync folded in), user
go-ahead before each; commit directly to `main`. Labels bit-identical —
assert it in every validation. Corpus re-scores via
`precompute_contacts_scenes.py` WITHOUT `--skip-existing` (overwrite is the
point) + `precompute_uncertainty_scenes.py` if its plot consumes changed
fields; smoke to scratch (`--out-root`) first. GPUs: ask the user. Plain
language in reports. Re-derive my §0 numbers before building on them.

## §2 T1 — factor-contribution audit (read-only)

For each factor, measure whether it earns its complexity. Factors:
`obs`'s three sub-factors (n_near, conf_near, in_img) at L=1; `persist`
both classes; `p_close` vs `p_pen` overlap (is a single monotone
gsdf-function equivalent?); `h_contact` vs `h_hover` (does one alone
reproduce h?); the `v_lo/v_hi` ramp vs the `still_ref` ramp (two motion
scales — necessary?).

Method: recompute corpus confidence with each factor ablated (set to 1) —
report the conf-distribution delta per class and population (use the §0
bucket structure), plus the labeled-clip reliability delta. A factor whose
ablation moves nothing measurable in BOTH is a drop candidate. My going-in
candidates: `conf_near` (obs averages 0.96 at L=1 — at least one sub-factor
is near-inert on dense DB clouds; check the sparse LOCAL clips before
concluding), possibly one health sub-factor. Deliverable: a keep/drop table
with numbers; NO code drops yet (they land in T3 on the final formula).
Commit 1 (the report + `docs/contact_confidence.md` initial commit).

## §3 T2 — no-contact recalibration

Four changes in `confidence.py`, threaded through `estimate_contacts`:

- **(a) Relative clearance** (the user's idea). Per frame: `rel_j =
  jsdf_j − min_j' jsdf_j'` (unguarded, finite mins only). A whole-body
  placement error cancels in the difference, so this witness needs NO
  health gate. `p_far_rel = sat((rel − rel_ref + ramp)/(2·ramp))`,
  rel_ref grid start 0.25 m (same carve logic — the closest joint may
  itself be on a carved hold). Distance witness becomes
  `in_img · max(h · p_far, p_far_rel)`; motion branch untouched.
  Edge cases: all-NaN frames (rel undefined ⇒ p_far_rel 0); the closest
  joint itself (rel = 0 ⇒ no self-boost, correct); multi-person scenes
  (per person, as everything is). Target: muted-body still joints with
  rel ≥ ~0.4 m go 0.15 → ≥0.6; healthy scenes: p_far_rel ≈ p_far where
  someone touches (min ≈ 0), so distributions barely move — verify.
- **(b) Hover band, decided by measurement.** On the 4 labeled clips,
  measure P(true contact | still, jsdf ∈ [12,25] cm, in_img, healthy) —
  the actual carve/hover risk in the band. If small (report the number;
  think <~3%), grid `far_ref_m ∈ {0.15, 0.20, 0.25}` and pick by
  reliability + the safety gate below; if not small, keep 0.25 and record
  why. This is the 67%-population lever — treat it with respect.
- **(c) NaN = beyond reach.** jsdf NaN currently ⇒ p_far 0. Change to:
  NaN counts as maximally far IFF the scene cloud is non-degenerate
  (point count above a floor; reuse the audit's degenerate criterion) AND
  the body is healthy (h) — else 0 as today. 2.3% of still negatives.
- **(d) Health composition** per T1's finding (keep both sub-factors or
  simplify) — whatever T1's table says.

Safety gate (hard, both sub-tasks): on the labeled clips, the
**confident-wrong-negative rate** — true (annotated) contacts receiving
conf(L=0) > 0.7 — must NOT rise vs HEAD. Plus: reliability monotone both
classes, top-decile accuracy not worse, labels bit-identical, corpus
spot-check (5–10 scenes incl. muted + healthy + degenerate) in the expected
directions and nowhere else. Then user go-ahead → full corpus re-score.
Commit 2 (incl. explainer-doc formula sync).

## §4 T3 — simplify, freeze, ship

- Apply T1's drop list to the post-T2 formula; delete the corresponding
  knobs from `CONTACT_CONFIDENCE_DEFAULTS` + the TOML table; re-verify
  parity of what remains (dropping a factor that measured inert must not
  move reliability — re-run the labeled-clip check on the final form).
- Schema freeze: final field list in `contacts_1.npz` (add a
  `confidence_schema` version scalar, e.g. `1`); update
  `docs/contact_confidence.md` to match the shipped formula EXACTLY
  (formulas, constants, corpus table re-measured post-ship) — it is the
  document of record for the learning pipeline; sync CLAUDE.md's contacts
  row.
- Final corpus re-score (no `--skip-existing`), 361/361 verified; refresh
  `features/audit/bodies.csv` confidence columns if it carries any.
- UPDATE.md round-8 outcome section. Commit 3.

## §5 Gotchas

- The carved-hold ambiguity is PHYSICAL: a hand on an unreconstructed hold
  and a hand hovering in open space are geometrically identical. Do not
  "fix" it — measure the band risk (T2b) and trade consciously.
- Out-of-image joints stay low-confidence BY DESIGN (extrapolated
  positions). Do not add recovery paths for them.
- Relative clearance certifies FREE joints on floaters; it must never
  boost conf near the body's own closest level (rel small) — a floater's
  near-wall joints stay uncertain.
- `jsdf` NaN also occurs on degenerate scenes (`vcRBT9Q9jXM_0053` is
  all-NaN) — T2c's cloud-floor gate is what keeps those honest.
- `nan_to_num(posinf=…)` does not catch `torch.finfo.max` (the guarded
  channel's sentinel) — keep the `_sat` clamps.
- Labels +1 attraction / 0 repulsion / −1 unlabeled in the OPTIMIZER
  (`optimizer.py:334`) — irrelevant to the readout but a standing trap.
- Corpus re-score must OMIT `--skip-existing`; smoke to scratch first.
- The scorer's `--reliability` must share `confidence.py` (verify, unify
  if not) — two implementations of a frozen interface is how it rots.
- Preserve the user's uncommitted viewer edits.
