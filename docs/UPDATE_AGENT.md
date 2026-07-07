# UPDATE_AGENT — round 5: betas + scale noise, confidence rewrite, kin2-on-auto, polish

*2026-07-07, rev 5. Implementation handoff. The human-facing rationale is
`docs/UPDATE.md` rev 7 — read it first. Rounds 1–4: revs 2–4 of this file in
git. Trust function/key names over line numbers.*

**Scope**: betas aggregation, `scripts/stages/estimate_scale.py`, the contact
readout (`estimate_contacts` + `tools/human_optim/contacts.py`) and its NEW
confidence, `run_kinematics_2` label plumbing, the scorer, and a final polish
pass across the touched code.
**Out of scope**: dynamics, the viewer, camera poses, the ClimbingVideos DB
(one re-run at the very end, folded into the pending Issue-22 pass — not yours).

**User decisions (do not re-litigate):**
- The two named residual problems are **short limbs** (⇒ betas aggregation +
  scale noise, T1/T2) and **occluded contacts** (⇒ honest per-class confidence,
  T3 — occluded is labeled *uncertain*, not recovered).
- Confidence must cover **both classes** (contact AND no-contact) and use
  distance-to-wall + cartesian velocity + VGGT/Sapiens confidence. Fast-moving
  joint ⇒ no-contact confidence → 1. These labels feed future learning.
- The old per-contact confidence is considered hallucinated — **full rewrite**,
  no preservation of the old formula.
- kinematics_2 driven by our auto labels is worth one careful experiment (T4).
- **Final-iteration hygiene**: dead code removed, big scripts split, comments
  that tell war stories cut to one contract line. Code fast and elegant;
  performance never regresses.
- Standing: measure-then-enforce is the law; no per-body scale; no betas
  *optimization* (aggregation only); accel/jerk/snap/radial are floors; no
  reduction of smoothness; commit to `main` directly, only when the user says.

---

## §0 State of the tree — read before editing

HEAD is `58787d3`. **Rounds 3+4 are UNCOMMITTED**, spread over 12 files:
`configs/human_optimization.toml`, `docs/UPDATE{,_AGENT}.md`,
`scripts/prototype_auto_contacts.py`, `scripts/stages/estimate_contacts.py`,
`tools/human_optim/{__init__,config,contacts,keypoints,losses,optimizer,stages}.py`.

**Step 0 (user approves, then do): commit this tree to `main` as ONE commit**
before any round-5 edit, so every round-5 diff is separable. Suggested subject:
`Contacts in kinematics_1, rounds 3+4: frozen auto labels, radial/snap, drift_pre gate, span merge, target smoothing`.

Round-4 code map (verified in the current tree):

- `tools/human_optim/contacts.py` — `scene_sdf` :29, `joint_guarded_min_sdf`
  :148, `hysteresis_spans` :181, `joint_drift` :234, `stillness_weight` :266,
  `drift_gated_contact` :278, **`merge_contact_spans` :319** (+ `_bool_spans` :305).
- `tools/human_optim/keypoints.py` — **`smooth_tracks` :95** (SG-style order 2,
  within validity runs; applied to Sapiens + init-anchor target tracks).
- `tools/human_optim/optimizer.py` — `optimize_person(..., auto_contact=...)`;
  `_detect_auto_labels` runs once at the refine boundary and returns the
  pre-enforcement drift + labels.
- `tools/human_optim/stages.py` — kin1 saves `extra=dict(drift_pre, auto_labels)`
  :338 → `_save_body` :151. **kin2's save (:493) passes NO `extra`** — see T4.
  `betas_mean = sp["betas"][p][vmask].mean(0)` :294. `_ANN_TO_SMPLX` :358,
  `_contact_labels` :372, `resolve_annotation_path` :391, `run_kinematics_2`
  :402 (labels → `optimize_person` :486).
- `scripts/stages/estimate_contacts.py` — merge knob defaults :76–78; **honest
  gate** :185–194 (prefers `kin["drift_pre"]`, falls back to final-body drift);
  merge applied :196–199; `np.savez` :225.
- `scripts/prototype_auto_contacts.py` (609 lines) — the scorer: P/R/F1/FPR
  grids, `--funnel` miss autopsy, `--body-path` baselines.
- `scripts/stages/estimate_scale.py` — `_frame_log_scale` mask∩depth gate :108;
  per-frame betas (deliberate, so far) :198; mask read+resize :210–216; pooled
  `combine_log_scales` :230.
- `scripts/stages/estimate_uncertainty.py` (725 lines) — the OLD per-contact
  confidence: refs :112–116 (`MARGIN_REF_CM`/`SPEED_REF_MS`/`VIS_FLOOR`),
  `_joint_speed` :231, `_persistence` :246, `_score_contacts` :303 (positives
  only, NaN elsewhere). T3 deletes/replaces this.

Round-4 measured (pooled over the 4 annotated clips; sport_climb separate):

| at the round-3 readout point (close 0.02 / still 0.025) | P | R | F1 | FPR |
|---|---|---|---|---|
| round-3 winner | 0.906 | 0.565 | 0.696 | 0.049 |
| + span merge + gate=drift_pre (readout only) | 0.905 | 0.608 | 0.727 | 0.053 |
| + target smoothing 0.35 s | 0.902 | 0.622 | 0.736 | 0.056 |
| + damper `ann_velocity[kin1]=500` | 0.906 | 0.618 | 0.735 | 0.054 |

Final operating point (user's pick, already in the TOML):
`[contacts].close_thresh_m=0.03`, `[contact_gate].still_m=0.03`, detector
pinned at `[auto_contact].still_m=0.025` → **P 0.896 / R 0.678 / F1 0.772 /
FPR 0.066**; in-contact grip speed 11.5→7.3 cm/s; leave-one-out stable
(held-out F1 0.745–0.805). Funnel: `moving` ≈21% of misses (mostly genuine),
`flicker` ≈1%; still extremities sit at **−3 cm** (penetrating) while
hover-misses sit at **+4 cm** ⇒ a single global scale cannot fix hover
(bimodal ⇒ proportions ⇒ betas, hence T1). Convention probe passed
(BetterRobot lie ↔ better_human agree — no wxyz bug). Per clip,
`kinematic_1.npz` = round-4 bodies; the round-3 winner is preserved as
`kinematic_1_round3.npz`.

**Before any T1/T2 re-run: copy each clip's `kinematic_1.npz` →
`kinematic_1_round4.npz`.** The scorer's `--body-path` serves baselines.

## §1 Round-5 tasks and global gates

1. **T1 robust betas** — the proportions lever (competition_2's short limbs).
2. **T2 `estimate_scale` noise pass** — mask erosion, frame-quality weighting,
   extremity-emphasis A/B.
3. **T3 confidence rewrite** — per joint-frame label confidence for BOTH
   classes; occlusion ⇒ honest unknown.
4. **T4 kinematics_2 on auto labels** — positives-only, confidence-weighted,
   drift_pre forwarded.
5. **T5 polish** — dead code out, comment diet, scorer promoted; parity-gated.

Global gates (vs the round-4 winner at the final operating point):
`reproj_init` vs **RAW** targets ≤ +0.5 px; pooled FPR ≤ 0.07; P ≥ 0.88;
runtime: optimizer wall-clock within ~1.05× (T1/T2/T3 add nothing
per-iteration; T4 is a separate opt-in stage); funnel re-run after T1+T2;
sport_climb never in the pooled tune set. Chamfer/penetration reported, not
gated.

## §2 T1 — robust betas

**Why**: `stages.py:294` is a plain mean over valid frames. MHR betas from
small/far/truncated detections regress toward the mean shape (short limbs for
a tall climber) and dilute the good frames. The scale ruler
(`estimate_scale.py:198`) uses raw per-frame betas — the same noise enters
there.

**Diagnostic first** (`scripts/diagnose_betas.py`, small, matplotlib):
per clip/person: (a) per-dim betas spread over frames (std + MAD); (b) scatter
betas norm / leading dims vs person bbox height (`sam3/bboxes.npz`) and vs
mean Sapiens body-kp score; (c) **implied limb lengths over time** — pose the
rest body with each frame's betas (`mid.forward` at identity q), measure
shoulder→wrist and hip→ankle chord lengths, plot; (d) print weighted-median vs
mean betas delta per person, in implied-limb-length mm. If competition_2's
far/small frames dip in implied limb length, the mechanism is confirmed.

**Fix**: `robust_betas(betas (F,10), weight (F,)|None) -> (10,)` in
`tools/human_optim/body.py` — per-dim **weighted median** (reuse
`tools.geometry.human_scale.weighted_median` per column; no new math).
Weight per frame: `w = bbox_h_norm × mean_sapiens_score` (both cheap to load
where used); drop a factor if the diagnostic shows it useless. Wire:

- `stages.py:294` → `robust_betas(sp["betas"][p][vmask], w[vmask])`.
- `estimate_scale.py`: new `--betas {frame,robust}` (default `frame` = parity);
  A/B on the locals.

**Accept**: implied-limb-length spread shrinks on competition_2; after the
T1+T2 re-run, the funnel's `hovering` count drops and the −3/+4 cm imbalance
narrows; pooled label F1 not worse. **Stop-or-go**: if the diagnostic shows
betas already tight (implied-limb spread ≲2 mm), report and SKIP the wiring —
don't ship a no-op knob.

## §3 T2 — `estimate_scale` noise pass

Three independent changes, each behind a CLI flag (this stage is argparse-only;
none of this enters `config.py`), A/B'd on the 5 locals:

1. **Mask erosion** (`--mask-erode-px`, default 3; 0 = parity): `cv2.erode` on
   the VGGT-res mask after the resize (:216). Mechanism: silhouette-rim pixels
   carry background-wall depth ⇒ `d_rel` too large ⇒ log-ratio too small ⇒ `s`
   dragged DOWN ⇒ scene too close ⇒ the measured −3 cm still-extremity
   penetration. Erosion also deletes thin-limb pixels whose VGGT depth is
   smeared — same bias direction. Report the per-clip `s` shift.
2. **Frame-quality weighting** (`--frame-quality {on,off}`, default off):
   multiply each frame's pooled weight (input to `combine_log_scales` :230) by
   `q_f = normalised in-mask pixel count × median VGGT conf in mask` (both in
   hand inside the loop).
3. **Extremity-emphasis A/B** (`--emphasis {on,off}`, default on = parity):
   the ×10 hand/foot vertex weight may import exactly the worst VGGT pixels
   (thin, fast limbs) into the median. Measure only — emphasis stays default-on
   unless the A/B wins on ≥3 clips.

**Accept**: winning combo improves the funnel's hover/penetration centering
and pooled F1 ≥ round-4 final. Run T1+T2 as **one re-run batch**
(scale → fuse → kinematics_1 → contacts — both invalidate everything
downstream). Write the flags used into `geometry/scale.npz` attrs (provenance).
If any clip's `s` moves >5%, stop and inspect before re-running downstream.

## §4 T3 — contact confidence, rewritten

**Delete** the old positives-only path in `estimate_uncertainty.py`
(refs :112–116, `_score_contacts` :303, and the plot/overlay pieces that
consume per-contact conf). That stage keeps ONLY per-frame body trust; its
overlay either drops the per-contact panel or reads the new field from the
contacts npz. Do not preserve the old formula anywhere.

**Build** `tools/human_optim/confidence.py` (~120 lines), consumed by
`estimate_contacts.py` AFTER labels are final (post-merge, :199). Inputs are
already computed in the stage or cost one no-grad pass:

- `gsdf (N,J)` m — already in hand (the :198 argument). +inf ⇒ no scene support.
- `drift (N,J)` m — the SAME source as the gate (:185–194; prefer `drift_pre`).
  Speed proxy `speed = drift / window_s`.
- `n_near (N,J)`, `conf_near (N,J)` — conf-gated scene points within
  `obs_radius_m` of each joint + mean VGGT conf of the k nearest; fold into the
  existing per-frame scene pass (52 joints × pruned candidates — trivial).
- `in_img (N,J)` — project joints with `transform.npz` extrinsics/K (the
  overlay already projects); ~10 px inside-frame margin.
- `persist (N,J)` — fraction of frames within ±`window_s` sharing the final label.

Combination (`sat(x)=clip(x,0,1)`; knobs in a new `[contact_confidence]` TOML
table read by the stage's own `_load_config` + setdefaults — NOT registered in
`config.py`; the optimizer never sees readout knobs):

```
p_close   = sat((close_ref_m + ramp_m − gsdf) / (2·ramp_m))   # 0.5 at close_ref
p_still   = sat(1 − drift / (2·still_ref_m))                  # 1 at drift 0
p_contact = p_close · p_still                                 # gsdf=+inf ⇒ 0
obs       = sat(n_near / obs_min_pts) · sat(conf_near / conf_ref) · in_img
p_fast    = sat((speed − v_lo) / (v_hi − v_lo))               # motion alone
conf(L=1) = persist · obs · p_contact
conf(L=0) = persist · max(p_fast, obs · (1 − p_contact))
```

Defaults: `close_ref_m=0.03, ramp_m=0.03, still_ref_m=0.03, obs_radius_m=0.15,
obs_min_pts=20, conf_ref=0.3, v_lo=0.15, v_hi=0.5` (m/s), window shared with
`[contact_gate]`. The SEMANTICS are the contract; exact curve shapes are yours,
but every factor stays monotone and explainable in one sentence. Note the one
deliberate asymmetry: `p_fast` bypasses `obs` — a fast joint is a confident
no-contact even when occluded (you can't grip while flying); a slow occluded
joint gets conf(L=0) ≈ 0 = the honest unknown. `obs_radius_m=0.15` matches the
funnel's "invisible" guard.

**npz**: add `p_contact`, `observability`, `label_confidence` `(P,N,J)` float32
to `contacts_<tag>.npz` (:225). Overlay: contact joints stay red, free joints
green; modulate marker size or brightness by `label_confidence` — keep it
readable, no colormap zoo.

**Validate** (extend the scorer, `--reliability`): against manual annotations
on the 4 clips, per class, 10 confidence bins → accuracy per bin (pooled + per
clip) + Spearman(conf, correct). Slices straight from the user's examples:
speed deciles → conf(L=0) rises monotonically; gsdf deciles → conf(L=1) falls
with distance. Report where the known occluded misses land — the win condition
is most of them sitting in the lowest conf(L=0) bins. No hard gate (the user
reads the tables), but a non-monotone reliability curve = redesign, don't ship.

## §5 T4 — kinematics_2 on auto labels

**Plumbing** (`stages.py:run_kinematics_2` :402):

- New `--labels {manual,auto}` on `human_optimization.py`, default `manual`
  (parity). `auto`: skip `resolve_annotation_path`; load
  `human_optim/contacts_1.npz`; labels = `joint_contact` → int8 `+1`, else
  `0` — **NEVER −1**. An auto no-contact is not a negative (occlusion); with no
  −1 present, `ann_repulsion` is inert by construction. Labels are already
  per-52 — bypass `_contact_labels`/`_ANN_TO_SMPLX`, but verify joint order
  against the npz's joint names vs `mid.joint_names`; don't assume.
- Confidence weighting: thread `contact_conf (N,J)` = `label_confidence` into
  `optimize_person` next to `contact_labels`; multiply the per-joint
  contribution in `joint_velocity_loss` and the labeled joints' vertex weights
  in the attraction by their joint's conf. Default `None` ⇒ exactly current
  behavior (manual path unchanged).
- **Forward the honest gate**: kin2's save (:493) must pass
  `extra=dict(drift_pre=…, auto_labels=…)` copied person-matched from
  `kinematic_1.npz` — for BOTH label modes. Today even manual kin2 drops
  `drift_pre`, so `estimate_contacts` on a kin2 body falls back to final-body
  drift and a velocity-damped body games its own readout (the round-3 failure,
  one stage later). This is a latent gap; fix it here.
- Weights: reuse the kin2 column (attraction 20, velocity 500). If attraction
  20 visibly overpulls (reproj blowup), try 10; no wider grid.

**Eval**: `estimate_contacts --bodies kinematic_2` (confirm it picks up the
forwarded `drift_pre`) → scorer vs manual, 4 clips, against the round-4 kin1
final point; reproj RAW gate vs the kin1 body ≤ +0.5 px; naturalness numbers
(in-contact grip speed, radial jitter); the user inspects in `view_3d` (body
source `kinematics_2`). Honesty note for the report: the labels came from
kin1's readout, so kin2-auto is judged ONLY by manual annotations + naturalness
— never by agreement with its own input labels.

**v2 (only if T3's reliability shows conf(L=0) ≥ 0.9 bins are ≥95% accurate)**:
`--auto-negatives` maps those bins to −1 (enables clearance repulsion on
confident negatives). Off by default; scored separately.

## §6 T5 — polish pass (LAST, after T1–T4 settle)

Rules: behavior-preserving; every deletion justified by grep; **parity gate** —
after the pass, `estimate_contacts` + scorer on boulder_3 give bit-identical
labels, one full kin1 re-run matches the pre-polish body to float noise, and
runtime is not slower. Separate commit(s) from T1–T4.

Inventory (verify each, then act):

- `tools/human_optim/velocity.py` — orphan of the removed
  `contact_velocity_loss` (its docstring still references it). If nothing
  imports it, DELETE.
- `estimate_uncertainty.py` — after T3's deletion, restructure the remainder
  (725 → aim ≤ ~450 lines): frame-trust core + plot + overlay; kill
  now-unused helpers (`_joint_speed`, `_persistence`).
- `scripts/prototype_auto_contacts.py` → rename `scripts/score_contacts.py`
  (it is the permanent eval harness, not a prototype); move reusable pieces
  (coarse folding, rates/F1, funnel, reliability) into
  `tools/human_optim/eval.py`; keep CLI flags stable.
- **Comment diet**: the TOML and `losses/optimizer/stages/estimate_contacts`
  carry multi-line round-history comments. Compress each to ≤1 line stating
  the CONTRACT (units, sign, guard), not the history — history lives in git +
  docs. KEEP the guard comments that prevent silent misconfig
  (`max_dist_m > band_m`, `prune ≥ max_dist`).
- Dead knobs: grep every TOML key against the code; delete unread keys. Check
  `scripts/diagnose_contacts.py` still runs against current npz fields (it
  predates round 4); decide keep/fix/delete with the user's "brings nothing
  goes" rule. `scripts/diagnose_betas.py` (T1) is a candidate to delete after
  the round if it answered its question.
- Docstring/CLAUDE.md sync: `estimate_contacts` docstring gains merge +
  gate-source + confidence fields; the CLAUDE.md stage-table row mentions
  `label_confidence`. Keep both short.
- Do NOT delete: `scale (P,)`≡1 in the body npz (consumer compat), the
  round-3/4 baseline body copies under `out/` (data, not code).

## §7 End-of-round eval

1. Re-run the chain on the 5 locals with the winning T1+T2 flags; funnel
   before/after; one readout grid close×still at the end.
2. Tables: per clip + pooled(4) P/R/F1/FPR vs round-4 final; reliability
   tables (T3); kin2-auto vs kin1 (T4); per-stage runtime.
3. Leave-one-clip-out at the final operating point.
4. Report losers honestly — "emphasis stays on" or "frame-quality useless" is
   a result, not a failure.
5. Append a "Round 5 outcome" section to `docs/UPDATE.md`, same format as
   round 4's.

## §8 Gotchas

- **Auto label −1 does not exist.** Auto no-contact ⇒ `0`. Mapping it to −1
  unleashes clearance repulsion on every unlabeled joint (the round-2 failure,
  inverted). Manual kin2 keeps its tri-state untouched.
- **Never recompute `drift_pre`** from a body that had `ann_velocity` applied —
  forward it verbatim (measure-then-enforce is the law).
- Scale direction: `s` DOWN ⇒ scene smaller ⇒ walls closer ⇒ MORE still-
  extremity penetration. Erosion should push `s` slightly UP. >5% movement on
  any clip ⇒ stop and inspect.
- Erosion acts on the VGGT-res mask (post-resize, :216); 3 px there ≈ 6–10
  orig-res px. Don't apply the same count at orig res.
- T1/T2 change bodies ⇒ auto labels, drift_pre, contacts and scores ALL move.
  Never mix a T1/T2 body with a pre-T1 `contacts_1.npz` in one table.
- `[contact_confidence]` lives in the stage's own loader, NOT `config.py`
  `knob_sections` — the optimizer must not grow readout dependencies.
- Betas re-aggregation is read-time only — `sam3d/smplx_params.npz` and the DB
  stay valid; the deferred DB re-run still folds into Issue-22.
- sport_climb is P=2 with an `ignored` flag in its annotation — the scorer
  handles it; don't special-case elsewhere.

## §9 Process

- Order: **commit rounds 3+4** → T1 diagnostic (stop-or-go) → T1+T2 (one
  re-run batch) → T3 (readout-only) → T4 → T5 polish → §7 eval.
- Each task lands as its own commit with its numbers in the message.
- Fast and elegant: nothing new inside `_forward_losses` except T4's
  conf-multiplier; everything else is one-shot, trajectory-space, or
  read-time. If an implementation starts wanting per-iteration recomputation
  of labels, stillness, confidence, or betas — stop and reread this plan.
- If a task's accept criterion fails: flag stays default-off, numbers go in
  the report, move on. No silent reverts, no silent keeps.
