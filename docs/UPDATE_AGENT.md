# UPDATE_AGENT — round 6: betas-in-the-fit, competition_2 autopsy, calibration, cleanup, corpus

*2026-07-07, rev 6. Implementation handoff — read `docs/UPDATE.md` rev 8 first
for the rationale. Rounds 1–5: revs 2–5 of this file in git. Trust
function/key names over line numbers.*

**Scope**: `kinematics_1` betas optimization, a competition_2 investigation,
the confidence calibration follow-ups, a bounded weight ablation, the full
cleanup pass, and corpus-readiness checks.
**Out of scope**: camera poses, `estimate_scale` (reverted in round 5, stays),
dynamics, the viewer, and the actual ClimbingVideos DB re-run (Issue-22, the
user schedules it — round 6 only makes it safe).

**User decisions (do not re-litigate):**
- The final unsolved reconstruction problem is contacts missed because the
  limb is too short to reach. Scale is exhausted (rounds 4+5 measured it).
  **Betas optimization inside the fit is now IN scope** — this overrides the
  old "no betas changes" rule. Regularization = stay close to the initial
  betas (the user's explicit choice). One vector per person, never per frame.
- competition_2 is the one clip where we are extremely bad — investigate
  whether it is limb size or broken depth before expecting betas to fix it.
- This is the **final iteration**: strong emphasis on code quality — remove
  everything left from experiments, no huge scripts, elegant functions, no
  war-story comments. Performance never regresses; we must be ready to run
  the optimization over a large corpus of videos.
- A light loss-weight ablation is welcome ("slightly ablate").
- Standing: measure-then-enforce; accel/jerk/snap/radial are floors (ablate
  them UP only, never adopt a reduction); auto labels are `+1/−1` (see
  gotchas — repulsion keys on `0`); chamfer/penetration reported, not gated;
  commit directly to `main`, per task, only with the user's go-ahead.

---

## §0 State of the tree

HEAD `312fa4f`, working tree CLEAN. Round-5 commits: `14f1bac` (rounds 3+4),
`5a01dc8` (T1/T2 betas-aggregation + scale-erosion: NULL, reverted, diagnostic
kept), `86b00a1` (T3 confidence: SHIPPED), `e742d9c` (T4 kin2-on-auto:
capability, default off; + forwards `drift_pre` into `kinematic_2.npz` — a
real fix), `312fa4f` (T5 partial: doc/comment sync).

Baseline (unchanged since round 4, pooled 4 annotated clips, operating point
close 0.03 / still 0.03): **P 0.896 / R 0.678 / F1 0.772 / FPR 0.066**.
`out/`: `kinematic_1.npz` = round-4 bodies (preserved copies:
`kinematic_1_round3.npz`, `kinematic_1_round4.npz`); `contacts_1.npz`
regenerated with the T3 confidence fields (labels bit-identical);
`uncertainty_*.npz` are STALE vs the T3 schema (re-run `estimate_uncertainty`
if a task consumes them). sport_climb is P=2 with an `ignored` flag; pooled
tables use the 4 clips, sport_climb reported separately.

Code map (verified this session):

- `tools/human_optim/confidence.py` (123 lines) — `CONTACT_CONFIDENCE_DEFAULTS`
  :31, `p_far` :106, `conf_neg` :120, `_scene_neighbourhood` :50 (full-cloud
  cdist, chunk=32 — see T6 memory check), `_persistence` :71.
- `scripts/stages/estimate_contacts.py` — `jsdf` (UNGUARDED diagnostic
  distance) from `lift_joint_contact` :195, saved as `joint_sdf_cm` :224;
  guarded `gsdf` :200 (`joint_guarded_min_sdf`, finite only within
  `[contacts].max_dist_m`=0.10, else `finfo.max`); `drift_pre` gate :207–210;
  merge :216; confidence call :227–229; npz fields :239.
- `tools/human_optim/optimizer.py` — `optimize_person(...)`; label semantics
  :334 (`incontact_v = (vlabel == 1)`, `nocontact_v = (vlabel == 0)`); DOF
  freezing by grad-masking; `_detect_auto_labels` at the refine boundary.
- `tools/human_optim/stages.py` — `betas_mean = sp["betas"][p][vmask].mean(0)`
  :294; `betas_T = betas_mean[None].expand(N, -1)` :299; `optimize_person`
  call :318; `betas_out.append` :326; `_save_body` :151 (kin1 `extra=` :338).
  `_contact_labels` :372 fills **−1 = unlabeled**.
- Scorer `scripts/prototype_auto_contacts.py` (609 lines): grids, `--funnel`,
  `--body-path`, `--reliability`.
- `tools/human_optim/velocity.py` — **NOT an orphan** (used by
  `scripts/diagnose_human_optim.py`); keep.
- `scripts/diagnose_betas.py` — round-5 evidence (short-limb 1–5 mm); delete
  in T5 after T1 cites its numbers.

### The round-6 confidence recheck (already done — input to T3)

The shipped confidence is CORRECT (verified in code + numerically on out/):
monotone conf(L=0) over speed deciles 0.00→0.96; conf(L=1) falls with gsdf;
slow hover-band negatives ≈0.01 (honest unknown); fast ≈0.94–0.99; no NaN;
drift = `drift_pre`, labels post-merge. Two calibration gaps to fix in T3:

1. **The far branch is a dead sliver.** `p_far` reads the GUARDED `gsdf`,
   which is finite only ≤0.10 m — so `p_far` is effectively binary {0: seen
   within 10 cm, 1: no guarded support}, the `far_ref_m=0.10` ramp never
   engages (its "0.5 here" comment is also wrong), and the `obs` product
   (15 cm radius) strangles what remains: `obs·p_far` can only fire in the
   10–15 cm shell. Empirically conf(L=0) ≈ persistence × motion ONLY; a slow
   joint 40 cm from the wall scores ~0, indistinguishable from occluded.
2. **Merged/edge contact frames score ≈0**: inside a bridged gap gsdf may be
   up to 0.08 ⇒ `p_close`→0 ⇒ conf(L=1)≈0 on exactly the frames the merge
   deliberately bridged — 9% (boulder_1) to 29% (competition_2) of contact
   labels sit below 0.2. Consumers can't tell "bridged on strong span
   evidence" from "marginal".

## §1 Tasks and global gates

Order: **T1 autopsy → T2 betas-opt → T3 calibration (independent) → T4
ablation → T5 cleanup → T6 corpus readiness**. One commit per task, numbers in
the message.

Global gates (vs the round-4/5 baseline above): `reproj_init` vs **RAW**
targets ≤ +0.5 px; pooled FPR ≤ 0.07; P ≥ 0.88; optimizer runtime ≤ ~1.05×;
leave-one-clip-out at the end. T2 must additionally IMPROVE pooled R/F1 — a
flat result is a null ⇒ revert-and-record (round-5 style).

## §2 T1 — competition_2 autopsy (read-only, first)

Question: is competition_2 bad because of limb proportions or because its
upstream (depth/mask/keypoints) is broken? All from existing artifacts; at
most one small new plot script (delete in T5).

Compare competition_2 against the other four on:

- **Depth quality on the person**: `geometry/human.npz` per-frame confident
  point counts + mean VGGT conf (the uncertainty stage's observability inputs;
  re-run `estimate_uncertainty` if you use its npz — stale schema).
- **Scale health**: `geometry/scale.npz` diagnostics (n_used, MAD, inlier
  fraction, per-frame spread).
- **Tracking/mask**: sam3 bbox height (person size in image), valid_mask
  fraction, mask area stability.
- **Keypoints**: mean Sapiens score; converter-vs-Sapiens 2D disagreement
  (the cue axis).
- **Body-vs-cloud**: chamfer residual per frame (`diagnose_human_optim.py`),
  per-limb hover gaps (`--funnel` gap histograms), betas-implied limb lengths
  (round-5 `diagnose_betas.py` evidence — cite, don't re-derive).
- T3-adjacent datapoint (already measured this session): competition_2's
  `observability` on free joints is 0.25 vs 0.37–0.39 elsewhere, and mean
  p_contact on its predicted contacts is the lowest (0.50) — its cloud is
  sparser/less confident near the climber.

**Deliverable**: a short verdict table ("depth X / proportions Y / pose Z")
+ an explicit prediction: can T2 plausibly rescue it? If depth is the culprit,
say so — T2 is then judged on the other three clips and the funnel's
`hovering` slice, not on competition_2 recall.

## §3 T2 — betas optimization in `kinematics_1`

**Design** (user-fixed): per person ONE betas leaf `(10,)`, shared across all
frames, init `beta0 = betas_mean` (stages.py:294), optimized jointly with `q`,
leashed by an L2 prior to `beta0`. Off by default; flips on only if it wins.

Implementation:

- **Config**: `[stages.kinematics_1] optimize_betas = false` +
  `[loss_weights.kinematic] betas_prior = [0, 300, 0, 0]`. Guard at load:
  `optimize_betas` with `betas_prior[kin1] == 0` ⇒ raise (an unleashed betas
  fit is never allowed).
- **Optimizer**: make betas a leaf next to `q`; build `betas_T` from the leaf
  inside the forward (`beta[None].expand(N, -1)`) instead of the precomputed
  constant — `mid.forward` already takes betas per call, so autograd is free
  and per-iteration cost is unchanged. Phase gating like the q DOF masks:
  root NEVER optimizes betas; A/B (a) refine(+lbfgs) only vs (b) full+refine.
  Include the leaf in the L-BFGS params when active.
- **Prior**: `betas_prior · mean((beta − beta0)²)`.
- **Reports** (per person, printed + in the eval table): `‖Δβ‖₂`; implied
  limb-length change in mm (pose the REST body with `beta0` vs `beta`, chord
  shoulder→wrist and hip→ankle); implied height change in %. **Flag** height
  |Δ| > 2% — the scene's metric scale used the `beta0` body as its ruler, so a
  material size change undermines it (report, don't auto-fix).
- **Save**: `betas_out` ← the optimized betas (consumers — `estimate_contacts`,
  kinematics_2, dynamics, the viewer — all pose from the npz betas and pick it
  up automatically). Also save `betas_init` (=`beta0`) as an extra field for
  diffing.
- **Auto labels / drift_pre**: unchanged — detected at the refine boundary
  from the then-current body, pre-enforcement. No interaction to add.
- **Grid** (budgeted): prior weight {100, 300, 1000} × phase {refine-only,
  full+refine}, scouted on boulder_3 + competition_2, best cell confirmed on
  all 4. Pick the LARGEST prior weight that still closes hover (gentlest
  intervention that works).

**Why this can work where round-5 aggregation couldn't** (context for the
report): aggregation is bounded by MHR's own per-frame predictions; here the
depth cloud + wall vote. Radially (along the viewing ray) a limb-length change
barely moves the 2D projection, so the init anchor doesn't resist the exact
correction hover needs; transverse changes are resisted — that asymmetry is
the safety, not a bug.

**Accept**: pooled R/F1 UP with all global gates green; funnel `hovering`
count down; naturalness flat (grip speed, radial jitter); betas movement
reported. Null ⇒ revert code to default-off, keep the numbers in the outcome.

## §4 T3 — confidence calibration (readout-only; independent of T2)

Three changes in `tools/human_optim/confidence.py` + `estimate_contacts.py`:

1. **Re-point the far branch at the unguarded distance.** Pass `jsdf`
   (already computed :195, the `joint_sdf_cm` channel) into
   `contact_confidence` alongside the guarded `gsdf`. `p_far =
   sat((jsdf − far_ref)/(2·ramp))` with `far_ref_m = 0.25` default (grid
   {0.20, 0.25, 0.30}); non-finite `jsdf` ⇒ `p_far = 0` (unknown). Replace
   the dead `obs · p_far` term: `conf_neg = persist · max(p_fast,
   in_img · p_far)`. Carve safety is the THRESHOLD, not the obs gate:
   `far_ref` sits beyond any plausible hold protrusion, so a hand on a
   carved-out hold (nearest reconstructed wall 10–20 cm behind it) stays
   uncertain, while a hand 40 cm out in open space becomes a confident
   no-contact. Hover (≤0.10) is untouched (p_far≈0 there). Fix the
   `far_ref_m` comment while at it.
2. **Save `merged (P,N,J)` bool** in `contacts_<tag>.npz`:
   `merged = jc_post_merge & ~jc_pre_merge` (both already in hand around
   :216). Leave `label_confidence` itself honest — the flag lets a learner
   treat bridged frames on span evidence.
3. **Overlay/docstring sync** for the new field.

**Validate** (`--reliability` re-run, 4 clips): the slow-far bucket
(`jsdf > 0.3 m`, speed < 0.15, in-image) must move ~0 → ≥0.6; the hover band
(0.03–0.10, slow) stays ≤ ~0.05; per-class reliability stays monotone;
Spearman(conf, correct) not worse. Labels themselves must be bit-identical
(this task touches confidence only).

## §5 T4 — weight ablation (small, budgeted)

One-at-a-time, on the post-T2 config. Terms and directions:

- Both ways (×0.5, ×2): `reproj_init 30`, `reproj_sapiens 8`, `chamfer 10`,
  `ann_attraction 10`, `ann_velocity 500`, `pose_prior 300`,
  `penetration 4000`.
- **Up only** (×2 — the floors rule): `radial_velocity 100`, `jerk_joint 500`,
  `snap_joint 500`.

Protocol: scout every cell on boulder_3 + competition_1 (17 runs × 2 clips);
any cell with pooled-2 F1 +≥0.01 and gates green is confirmed on all 4.
Measure per cell: P/R/F1/FPR, reproj RAW, chamfer, grip speed, radial jitter.
**Adopt** only a confirmed pooled-4 F1 +≥0.01 with all gates; otherwise the
deliverable is the sensitivity table itself (a flat table = "the config is
robust", which the corpus claim needs — record it in the outcome).

## §6 T5 — the cleanup (a primary deliverable this round)

Rules: behavior-preserving; every deletion grep-justified; **parity gate** —
`estimate_contacts` + scorer on boulder_3 bit-identical labels AND confidence
(post-T3 values), one kin1 re-run float-identical, runtime ≤ 1.0×. Its own
commit(s).

- **Comment diet** (the deferred round-5 item): `configs/human_optimization.toml`
  + `losses/optimizer/stages/estimate_contacts` — every multi-line
  round-history comment becomes ≤1 contract line (units, sign, guard). KEEP
  the misconfig guards (`max_dist_m > band_m`, `prune ≥ max_dist`).
- **Scorer promotion** (deferred): `scripts/prototype_auto_contacts.py` →
  `scripts/score_contacts.py`; reusable pieces (coarse folding, rates/F1,
  funnel, reliability) → `tools/human_optim/eval.py`; CLI flags stable.
- **`estimate_uncertainty.py` shrink** (deferred): 725 → ≤ ~450 lines now the
  per-contact block is gone; kill now-unused helpers; per-contact overlay
  panel reads `contacts_*.npz` fields.
- **Experiment residue**: delete `scripts/diagnose_betas.py` (after T1 cites
  it) and any T1 autopsy scratch script; `git show 5a01dc8` and verify the
  scale/betas revert left no dead flags in `estimate_scale.py` / stages;
  dead-knob grep over every TOML key.
- **Keep**: `tools/human_optim/velocity.py` (used by diagnose_human_optim);
  the kin2-auto capability (default off — small, tested, the future
  self-training hook; flagged for user veto); `scale (P,)`≡1 npz field; the
  round-3/4 baseline copies under `out/` (data).
- **Docs sync**: CLAUDE.md stage table (`estimate_contacts` row: merge, gate
  source, `label_confidence`+`merged`; `human_optimization` row:
  `optimize_betas` if adopted); module docstrings tell the current truth;
  durable contracts land in CLAUDE.md — the UPDATE docs then close with the
  round-6 outcome and become history.

## §7 T6 — corpus readiness

- **Defaults audit**: a flagless run of every stage CLI must equal the tuned
  behavior (the TOML is the source of truth; grep CLI defaults that shadow
  TOML values and reconcile).
- **Contacts DB driver**: `precompute_scenes.py --stage contacts` (or a
  sibling driver on the symlink-farm pattern of the human_optim one), writing
  `features/contacts/<aa>/<bb>/<scene>/`. The DB feature trees are otherwise
  READ-ONLY; smoke runs must never overwrite existing feature dirs — write to
  a scratch save-dir.
- **Memory check**: `_scene_neighbourhood` (confidence.py:50) cdists
  chunk·52 joints against the FULL scene cloud, unpruned — measure peak on the
  largest DB scene; if the (chunk·J × M) matrix exceeds ~2 GB, add the same
  AABB pruning the contacts SDF uses (or shrink chunk). This is the only new
  full-cloud pass round 5 added.
- **Runtime/VRAM table** per stage on one representative clip — goes in the
  outcome (the corpus budget needs it).
- **Smoke test**: 5–10 DB scenes end-to-end (scale→fuse→kinematics_1→contacts)
  with default flags into scratch; zero failures, sane outputs, per-item
  failure tolerance intact.
- The full DB re-run stays Issue-22 (user-scheduled). Not yours.

## §8 Gotchas

- **Label semantics** (round-5 verified, optimizer.py:334): attraction keys
  on `1`, **repulsion keys on `0`**, unlabeled is **`−1`**
  (`_contact_labels` fills −1). Auto labels are `+1/−1`. Mapping auto
  no-contact to `0` unleashes the 4000-weight repulsion on every unlabeled
  joint.
- **Two distance channels**: guarded `gsdf` (contact decisions; finite ≤
  `max_dist_m`, else `torch.finfo.max` — note `nan_to_num(posinf=…)` does NOT
  catch finfo.max; the `sat()` clamps are what keep it safe) vs unguarded
  `jsdf` (diagnostics + the T3 far branch). Use the right one per purpose.
- **Betas-opt**: the init-anchor targets are FIXED 2D tracks projected once
  from the converter body — never recompute them from the re-shaped body.
  `estimate_scale` keeps its per-frame betas (not touched). Optimized betas
  flow to every consumer via the npz automatically — including dynamics
  inertia (out of scope, fine, but say so in the commit).
- **Measure-then-enforce** stands: labels + `drift_pre` are detected
  pre-enforcement and forwarded verbatim (`e742d9c` fixed kin2's forwarding).
- Ablation must not ADOPT any smoothness reduction (floors rule); ×0.5 cells
  for smoothness terms are not even run.
- T2 changes bodies ⇒ labels/drift_pre/contacts/scores all move; never mix a
  T2 body with a pre-T2 `contacts_1.npz` in one table. Copy
  `kinematic_1.npz` → `kinematic_1_round5.npz` per clip before T2 re-runs.
- `out/uncertainty_*.npz` are stale vs the T3 schema — re-run that stage
  before consuming them (T1 does).
- sport_climb: P=2, `ignored` flag, never in the pooled tune set.

## §9 Process

- One commit per task, numbers in the message; user go-ahead before the first
  commit of the round.
- Fast and elegant: T2 adds 10 parameters and one prior term — nothing else
  enters the per-iteration path all round. If an implementation starts
  wanting per-iteration recomputation of labels, stillness, confidence, or
  betas aggregation — stop and reread this plan.
- Null results are results: revert to default-off, record the numbers in the
  outcome, move on. No silent reverts, no silent keeps.
- After §7: append the round-6 outcome to `docs/UPDATE.md`, migrate durable
  contracts to CLAUDE.md, and declare the UPDATE docs historical.
