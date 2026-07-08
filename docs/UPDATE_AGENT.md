# UPDATE_AGENT — round 7: corpus floaters, confidence v2, limb-length lever

*2026-07-08, rev 7. You are the implementing agent. `docs/UPDATE.md` rev 9 is
the human-facing plan — read it first for intent. This file is your handoff:
verified state, code anchors, task specs, gates. Rounds 1–6: git history.
Trust function/key names over line numbers.*

## §0 State (verified 2026-07-08 by the planning session)

**Repo**: HEAD `0415bc1` on `main`. **Uncommitted edits exist** in
`scripts/view_3d.py`, `tools/viewer/{app,loading,scene}.py` (the user's own
viewer tweaks from corpus inspection). Do NOT revert or commit them blindly;
when T1 touches `view_3d.py`, edit around them and tell the user what you did.

**Corpus state** (DB root `/data3/rikhat.akizhanov/better/data/ClimbingVideos`):
361 scenes complete through `human_optim` → `contacts_1` → `uncertainty_1`
(commits `3c05a57`, `0415bc1`). Layout today (T1 changes this):
`features/human_optim/<aa>/<bb>/<scene>/kinematic_1.npz`,
`features/contacts/<aa>/<bb>/<scene>/contacts_1.npz`,
`features/uncertainty/<aa>/<bb>/<scene>/uncertainty_1.npz`.
`scenes.db` is READ-ONLY (owned by BetterParsing). Local tuning clips in
`out/<stem>/` are at round-6 state; verify freshness before comparing.

**Tuning-clip baseline** (unchanged since round 4, close 0.03 / still 0.03,
4 labeled clips): pooled P 0.896 / R 0.678 / F1 0.772 / FPR 0.066.

**The user's floating-body report**: five named scenes —
`PfwFWSoPF_k_0026`, `2nwG1Qa5k_c_0115`, `9RvrpYoWDoo_0000`,
`CRquJ8H1QLI_0001`, `3DAgLpPXVCs_0132` (**person 1**; person 0 is healthy).

**Planning-session sweep** (measured on the real corpus npz — 363
(scene,person) rows with ≥5 valid frames):

- contact_rate (share of valid frames with ≥1 of 52 joints in contact):
  percentiles [5,25,50,75,95] = [0.00, 0.54, 0.81, 0.95, 1.00].
  **21 rows at exactly 0**, 39 below 0.2, **68 below 0.4**, 80 below 0.5.
  Clearly healthy (rate ≥0.8 and med_min ∈ (−8,+3) cm): 131.
- med_min_jsdf (median over frames of min-over-joints UNGUARDED
  `joint_sdf_cm`): percentiles = [−35.4, −7.6, −3.9, −1.9, +19.2] cm. The
  median body's closest joint sits ~4 cm INSIDE the wall.
- med chamfer_cm (from uncertainty_1): [12.0, 13.9, 15.9, 18.2, 24.5].
  Floaters' chamfer is mostly unremarkable ⇒ they sit ON their own human
  cloud; the failure is elsewhere (cloud-vs-wall, limbs, upstream).
- The five named, ranked by contact_rate (of 363): `9RvrpYoWDoo_0000` #26
  (0.035, med_min +15.8 cm), `2nwG1Qa5k_c_0115` #52 (0.298, +9.1),
  `CRquJ8H1QLI_0001` #56 (0.312, +5.8), `3DAgLpPXVCs_0132` p1 #60 (0.355,
  +6.5), `PfwFWSoPF_k_0026` #75 (0.459, +2.1). Worse un-named floaters
  exist: `2nwG1Qa5k_c_0045` (0, +40 cm), `gtIGgs_y2nY_0095` (0, +65 cm),
  `UptZLFE05vc_0009` (0, med_min **−64 cm** — behind the wall),
  `vcRBT9Q9jXM_0053` (0, all-NaN jsdf — degenerate scene cloud).
- **Confidence decomposition, corpus-wide** (1,417,277 contact joint-frames):
  conf(L=1) mean **0.465** (median 0.476; 20% below 0.2). Factors at L=1:
  p_contact **0.524** (the throttle), obs 0.963 (NOT the problem on dense DB
  clouds), implied persist 0.859. By geometry: penetrating < −1 cm
  (n=652k, **46% of all contacts**) mean **0.508**; touch [−1,+3] cm (n=709k)
  0.448; gap > +3 cm (n=56k, mostly merged/edge) 0.195.
  Since p_close ≈ 1 near the wall, the throttle inside p_contact is
  **p_still**: a real gripping hand drifts 2–3 cm per 0.35 s window
  (round-4 measured ~7 cm/s grip speed), which lands at 0.4–0.5 on the ramp.
- conf(L=0): overall mean 0.538 (45% above 0.8 — fine, most joints are far).
  **On the 24 near-zero-contact scenes: mean 0.791, 74% above 0.8 —
  confidently wrong.** The user's floater-confidence complaint, quantified.

**MHR limb facts** (verified): `features/sam3d/<...>/params.npz` carries
`scale_params (P,N,28)` — per-frame skeleton scales SAM-3D predicts — plus
`pred_joint_coords (P,N,127,3)` (MHR posed joints) and `mhr_model_params
(P,N,204)`. `tools/mhr_to_smplx/converter.py:95` (`_mhr_to_camera_frame`)
poses MHR with the FULL param vector, so the conversion target already
embodies MHR's limb lengths; whatever betas cannot express is silently
dropped. ⇒ MHR-vs-SMPLX bone-length mismatch is measurable from artifacts
already on disk, per frame, corpus-wide.

**Code map** (anchors verified today; re-read before editing):
- `tools/human_optim/confidence.py` — `CONTACT_CONFIDENCE_DEFAULTS` :35;
  `_scene_neighbourhood` :54 (memory-bounded); `_persistence` :78;
  `contact_confidence` :90 — p_close :114, `jsdf_f` nan/±inf→0 :115,
  p_far :116, p_still :117, `p_contact = p_close * p_still` :118,
  p_fast :120, obs :125, `conf_pos = persist·obs·p_contact` :129,
  `conf_neg = persist·max(p_fast, in_img·p_far)` :130.
- `scripts/stages/estimate_contacts.py` — unguarded `lift_joint_contact`
  (≈:195, saved as `joint_sdf_cm`), guarded `joint_guarded_min_sdf` (≈:200),
  drift_pre preference (≈:207), `merge_contact_spans` (≈:216), confidence
  call on post-merge labels (≈:227). Fields: joint_contact, joint_sdf_cm,
  `merged`, p_contact, observability, label_confidence, valid_mask, fps.
- `tools/human_optim/contacts.py:148` `joint_guarded_min_sdf` — guard
  `(dmin < max_dist_m) & (conf > conf_thresh)`, else `torch.finfo.max`
  (NOT `inf` — `nan_to_num(posinf=…)` does not catch it; `_sat` clamps are
  why the current code is safe).
- `scripts/scene_db.py:66` `ScenePaths.feature_dir(stage, scene_id)`;
  `shard` :32 gives the `<aa>/<bb>` split.
- `scripts/precompute_contacts_scenes.py` — `_out_path` :43 →
  `feature_dir("contacts", sid)`; its docstring declares human_optim
  read-only (T1 changes that contract).
  `scripts/precompute_uncertainty_scenes.py` — `_out_path` :56, contacts
  input :72, farm symlink of contacts into the predictions dir :89–90.
- `scripts/view_3d.py:105–112` — the farm workaround merging
  `features/contacts/` into the per-scene `human_optim/` link dir (delete in
  T1; file has uncommitted user edits — be surgical).
  `tools/viewer/loading.py:95` reads `human_optim/contacts_1.npz`.
- `scripts/score_contacts.py` — `--reliability` :719, `_reliability_one`
  :566. Check whether it imports `contact_confidence` or duplicates the
  formulas; after T4 they MUST share one implementation.
- `kinematic_1.npz` carries `drift_pre (P,N,52)`, `auto_labels`, `betas
  (P,10)`, `valid_mask`, fps. `uncertainty_1.npz` carries per-frame
  `chamfer_cm`, `fit_score`, `obs_score`, `combined`, `valid`.

## §1 Process (unchanged from rounds 4–6)

One task = one commit, in order, each with a user go-ahead before committing;
commit directly to `main`. Ask the user for GPU allocation before any GPU
run. DB feature trees are written ONLY by the drivers; smoke tests write to
scratch (`--out-root`), never the real tree — the single exception is T1's
migration, which the user explicitly ordered. Labels are sacred: confidence
changes must leave `joint_contact` bit-identical. Final reports in plain
language (short sentences, terms explained — standing user feedback).
Measure-then-enforce: any health/stillness signal that feeds a loss or gate
must be computed BEFORE the term it feeds could distort it. Skepticism: the
§0 numbers are probes, not gospel — re-derive anything you build on.

## §2 T1 — layout quick fix: contacts + uncertainty into human_optim

Goal: `features/human_optim/<aa>/<bb>/<scene>/` holds `kinematic_1.npz` +
`contacts_1.npz` + `uncertainty_1.npz`, mirroring local
`out/<stem>/human_optim/`.

1. Both drivers: `_out_path` → `feature_dir("human_optim", sid)`; update the
   uncertainty driver's contacts INPUT path (:72) and drop the now-redundant
   farm symlink (:89–90 — the body dir already holds contacts after the
   move). Update both docstrings (the "human_optim is read-only" sentence).
2. Migrate: `mv` each existing `contacts_1.npz` / `uncertainty_1.npz` into
   the matching human_optim scene dir (361 + 361 files; verify counts before
   and after; then remove the emptied `features/contacts/` and
   `features/uncertainty/` trees). Show the user the plan before running.
3. Delete the `view_3d.py:105–112` merge workaround (surgical — uncommitted
   user edits live in this file).
4. Sync docs: CLAUDE.md's dataset-pipeline bullet + any row mentioning
   `features/contacts/`.

Verify: `precompute_contacts_scenes.py --dry-run --skip-existing` reports all
361 done at the NEW path; same for the uncertainty driver; `view_3d --scenes`
on one scene still shows contact colors. Commit 1.

## §3 T2 — floater detector + corpus audit

New script (suggest `scripts/audit_scenes.py`; you own naming) reading the
post-T1 tree, one row per (scene, person):

- `n_valid`, `contact_rate` (any of 52 joints), `med_min_jsdf_cm` (signed),
  `p_seated` (share of frames with min jsdf ∈ [−5,+3] cm),
  `nan_jsdf_share`, `med_chamfer_cm` / `p90_chamfer_cm` (uncertainty npz),
  `mean_obs`, scene-cloud point count, tracking `valid_share`.
- Flags: `floater` = contact_rate < 0.4 OR med_min_jsdf > +5 cm;
  `buried` = med_min_jsdf < −15 cm; `degenerate` = nan_jsdf_share > 0.5 or
  near-empty scene cloud. These seeds flag all five named (verified);
  report how many rows each flag catches (expect ~69 floaters).
- Output: a CSV (suggest `features/audit/bodies.csv`, or repo-side — ask
  the user) + a ranked console table + histogram pngs.

Validation loop with the user: they walk a sample of flagged scenes (top +
middle of the flagged band) and ~10 unflagged mid-pack scenes in
`view_3d --scenes` and confirm the flag matches their eye. A low rate is
suspicious, not damning (falls, dynos, walk-ups are legit) — the visual pass
is what buys the threshold. Deliverable: CSV, counts, agreed thresholds.
Commit 2.

## §4 T3 — autopsy: assign each flagged scene one of four causes

For every flagged (scene,person), four probes → one dominant-cause label:

- **(a) body-off-cloud** (optimizer failure): high chamfer — p90 well above
  the corpus p75 (≥~30 cm) or low fit_score. Expect RARE (see §0).
- **(b) human-cloud-off-wall** (VGGT depth layering): NEW probe — load
  `geometry/human.npz` + `geometry/scene.npz`, sample frames, nearest-
  neighbour distance of human points to the scene cloud; per-frame p10/min,
  median over frames = the cloud-to-wall gap. If that gap ≈ the body's
  hover, the reconstruction itself is detached ⇒ unfixable kinematically.
- **(c) short limbs**: from the MHR-vs-SMPLX bone table — compute it
  corpus-wide HERE (cheap; T5 reuses it): per frame, bone lengths from
  `pred_joint_coords` (127-joint MHR skeleton — joint names/parents from the
  MHR model in `third_party/` or the sam3d stage; map the four limb bones
  per side) vs converted SMPL-X rest-pose bone lengths (per-person betas →
  `tools/smplx_robot` rest joints). Robust median per scene. Deficit ≥2 cm
  on arms/legs = evidence for (c).
- **(d) broken upstream**: near-empty scene cloud, NaN jsdf, low
  valid_share, thin `geometry/scale.npz` diagnostics.

Deliverable: cause label + evidence per flagged scene (the five named
narrated explicitly), cause COUNTS, and a verdict paragraph: which diseases
dominate, what share is fixable (a: knobs, c: T5) vs flag-only (b, d). This
gates T5: if (c) is marginal, the limb experiment dies cheaply. Commit 3.

## §5 T4 — confidence v2 (readout-only; labels bit-identical)

Two calibrated changes in `tools/human_optim/confidence.py`, threaded through
`estimate_contacts`:

- **(i) Strong geometry beats motion doubt.** Today `p_contact =
  p_close · p_still` double-punishes drift: the label already passed the
  stillness gate at detection, then p_still throttles conf for the same
  drift — structurally ~0.5 on real grips, and penetration (46% of
  contacts, the strongest touch evidence) averages 0.508. Add a penetration
  term so motion cannot throttle firm contact, e.g. `p_contact = p_close ·
  max(p_still, p_pen)` with `p_pen` monotone-decreasing in gsdf, ∈[0,1],
  ≈0 for gsdf ≥ close_ref (the touch band still needs stillness), ≥0.9 by
  gsdf ≤ −1 cm. Optionally also widen the p_still ramp — still_ref is a
  detection-gate constant, not a confidence calibration. You own the exact
  form; every factor stays monotone and in [0,1].
- **(ii) Floaters lose distance-based confidence.** Compute a per-person
  body-health scalar INSIDE estimate_contacts from the final labels + jsdf
  (self-contained; same signals as T2): h = f(contact_rate, med_min_jsdf) —
  1 for healthy bodies, →0 for never-touch / all-hover bodies; smooth,
  monotone. Gate ONLY the far branch: `conf_neg = persist · max(p_fast,
  in_img · h · p_far)` — motion certainty survives a broken reconstruction,
  distances don't. Save `body_health (P,)` in the npz. (p_far is already
  safe against behind-wall negatives — a −64 cm jsdf gives p_far 0 — but h
  also mutes mixed-sign scenes; check `UptZLFE05vc_0009`.)

Validation, in order:
1. Four labeled tuning clips: `score_contacts.py --reliability` — curves
   stay monotone, top-decile accuracy not worse (both classes), Spearman
   not worse. FIRST check whether the scorer recomputes confidence with its
   own copy of the formulas; if so, refactor it to import
   `contact_confidence` before changing anything (one implementation).
2. Corpus spot re-run (5–10 scenes to scratch via the driver's
   `--out-root`): labels bit-identical; penetrating-contact mean conf
   0.508 → **≥0.75**; flagged-floater conf(L=0) mean 0.791 → **≤0.4**;
   healthy-scene conf(L=0) essentially unchanged.
3. User go-ahead → full corpus readout re-run via the drivers (contacts
   ~11 min + uncertainty ~12 min at round-6 timings; do NOT pass
   `--skip-existing` — the point is to overwrite).

New knobs live in `CONTACT_CONFIDENCE_DEFAULTS` + the `[contact_confidence]`
TOML table (readout-only — NOT `config.py`). Update the module docstring, the
CLAUDE.md contacts row, and `estimate_uncertainty`'s plot if it consumes the
fields. Commit 4.

## §6 T5 — limb length: validate the measurement, then maybe the fix

**(a) The measurement (read-only).** T3 already produced the MHR-vs-SMPLX
bone table. Add the independent third opinion on the 4 tuning clips + the 5
named scenes: lift Sapiens 2D limb endpoints (goliath-308 indices via
`tools/human_optim/keypoints.py`) with VGGT depth at those pixels
(`features/vggt/` depth exists corpus-wide; intrinsics via
`tools/geometry/intrinsics.py`) → 3D → per-frame bone lengths → robust
median + MAD per limb. Compare three sources per scene: MHR, converted
SMPL-X, VGGT-lifted. Beware: VGGT-lifting measures the FRONT SURFACE at a
2D keypoint, not the joint center — bone LENGTHS mostly cancel the thickness
offset, but occlusion and off-body pixels don't; report spread, drop
non-finite/masked-out samples.

**Go/no-go gate (record it either way):** proceed iff flagged disease-(c)
scenes show a systematic ≥2 cm limb deficit vs at least one independent
source (MHR or VGGT, ideally agreeing) AND the T3 cause counts make (c) a
material share (≳15 scenes). Otherwise: null verdict, thread closed, T5(b)
skipped — write that honestly.

**(b) The experiment (only on go).** Site depends on WHERE length is lost:
- **Conversion drops it** (MHR long, SMPL-X short): strengthen the
  converter's limb fidelity — e.g. upweight limb edges / add an explicit
  bone-length term to its existing edge loss (`converter.py:195–207`), then
  re-convert + re-run the chain on the tuning clips. Cheap, benefits every
  scene, zero optimizer risk. Re-measure post-fix — betas capacity may cap
  what conversion can keep.
- **MHR itself is short** (all sources short vs the wall evidence):
  kinematics-time extension — 4 per-person scalars (upper-arm, forearm,
  thigh, shin; LEFT/RIGHT TIED — the user's symmetry requirement), applied
  as rest-pose joint+vertex offsets along the bone axis (LBS-weighted) in
  the body forward, optimized in the refine phase only, driven by a
  **target-length loss** to the T5(a) measured lengths + an L2 leash to
  zero. NOT chamfer-driven — round 6 proved the fit does not discover
  lengthening on its own (a short limb already rests on the near face of
  its cloud). The root stage never touches them.

Gates (both variants): pooled 4-clip F1/recall UP, P ≥ 0.88, FPR ≤ 0.07,
reproj vs raw ≤ +0.5 px, runtime ≤ 1.05×, grip speed flat, contact_rate on
the five named scenes materially up, user visual pass. Null ⇒ revert and
record. Commit 5 (or the null note).

## §7 T6 — wrap

CLAUDE.md sync (new layout, `body_health` + audit fields, audit script row);
UPDATE.md round-7 outcome section; final corpus state = readouts re-run with
the shipped confidence. A corpus KINEMATICS re-run happens only if T5(b)
shipped a fit change — the user schedules it (Issue-22 pattern). Commit 6.

## §8 Gotchas

- `3DAgLpPXVCs_0132` floats for **person 1** only — everything per-person.
- Behind-wall bodies give large NEGATIVE jsdf (the nearest point's normal
  faces away). Distance-to-wall axes need sign care; "buried" is its own
  flag, not a healthy state.
- `vcRBT9Q9jXM_0053`: all-NaN jsdf (degenerate scene cloud) — every metric
  must survive NaN and empty clouds.
- `joint_guarded_min_sdf` uses `torch.finfo.max`, not inf — `nan_to_num
  (posinf=…)` will NOT catch it (historical trap; `_sat` clamps are why the
  current code is safe).
- Optimizer label convention (`optimizer.py:334`): attraction keys on
  **+1**, clearance repulsion on **0**, unlabeled is **−1**. Auto labels
  are +1/−1.
- contact_rate counts ANY of the 52 joints (finger granularity) — a
  wrist-only analysis reads lower.
- `merged (P,N,J)` marks span-merge-bridged frames (round 6); their conf is
  honestly low — consumers use the flag; do not inflate their conf.
- fps comes from the npz; scenes reach ~490 frames (VGGT cap 500).
- Corpus re-runs must OMIT `--skip-existing` when the intent is overwrite.
- The uncommitted viewer edits (§0) are the user's — preserve them.
