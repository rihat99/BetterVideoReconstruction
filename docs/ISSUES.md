# Human-Optimization Loss Review — Known Issues

**Reviewed:** 2026-07-05 (found + verified). **Fix round 1:** 2026-07-06
(uncommitted working tree). **Fix verification + this revision:** 2026-07-06.
**Scope:** the losses and their surrounding machinery in `tools/human_optim/`
(`losses.py`, `optimizer.py`, `stages.py`, `contacts.py`), plus the two other
consumers of the shared vertex-weight field (`scripts/stages/estimate_scale.py`)
and the geometry contract they read (`scripts/stages/fuse_geometry.py` outputs).

**How this list was produced:** full line review; unit probes of the loss
primitives (gradient signs, masking, edge cases); empirical probes on all five
local clips (`out/{boulder_1,boulder_3,competition_1,competition_2,sport_climb}`)
using the real posed `kinematic_1` bodies; independent verification of the
better_human `SMPLXMid` contract and the `geometry/*.npz` contract; an external
adversarial pass (codex, xhigh) which confirmed every finding and refuted none;
and a post-fix verification pass re-running the same probes against the fixed
tree + re-fused/re-optimized local artifacts. Original line numbers refer to
commit `eb50669`; "fixed-tree" lines refer to the uncommitted working tree on
top of it.

Severity: **High** = actively distorts results on the default chain or the DB;
**Medium** = wrong/misleading behavior in specific-but-real conditions, or a
stage off the default chain; **Low** = crash edge case, diagnostics, or doc rot.

---

## Status summary (2026-07-06)

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | Scene-candidate truncation at extremities | High | **Fixed** (cap 1500→6000; verified 0% truncation on locals) |
| 2 | Dead `conf_thresh` gates | High | **Fixed** (quantile gates) — see expanded section for one recommended refinement |
| 3 | `fps` missing from `transform.npz` | High | **Fixed in code + local data**; DB artifacts still stale (→ Issue 22) |
| 4 | Contacts SDF lateral asymmetry | High | **Fixed** (perpendicular SDF, matches the loss) |
| 5 | Camera-frame invalid-frame fill | High | **Partially fixed** (anchor + scene terms excluded; fill itself still camera-frame) |
| 6 | Chamfer/scale vertex-weight concentration + direction | High | **Open** (deliberately deferred — design decision pending) |
| 7 | Dilated human carve → scene holes at grips | High | **Open by decision** (dilate 2 tried, reverted to 8: mask-edge noise wins; depth-aware carve is the way to get both) |
| 8 | Chamfer crash on zero depth points | Medium | **Fixed** (empty-cloud guard) |
| 9 | One-active-vertex frames get outsized weight | Medium | **Fixed** (joint mean over active instances; pen weight retuned 2000→4000) |
| 10 | Dynamics zero torque arm | Medium | **Open** (dynamics currently deprioritized) |
| 11 | Dynamics mixed weight conventions | Medium | **Open** (dynamics currently deprioritized) |
| 12 | `joint_velocity_loss` outgoing-frame gating | Medium | **Fixed** (both-ends pair gating) |
| 13 | `depth_push` stale mechanism | Medium | **Fixed** (weight column zeroed; code retained) |
| 14 | Smoothness NaN for T<4 | Medium | **Fixed** (empty-diff → 0 guard) |
| 15 | Metrics ≠ optimized objective | Low | Open (unchanged; `_metrics` also ignores `frame_valid`) |
| 16 | Mislabeled diagnostics | Low | Open (unchanged) |
| 17 | Per-phase candidate staleness | Low | Open (less harmful now the cap doesn't bind) |
| 18 | `frame_id` / extrinsics alignment only guarded upstream | Low | Open (unchanged, latent) |
| 19 | Deleted geometry doc still referenced | Low | Open |
| 20 | **NEW:** `estimate_uncertainty` crashes on renamed config key | Medium | **Open — introduced by fix round 1** |
| 21 | **NEW:** `diagnose_human_optim` drifted from the optimizer | Low | Open |
| 22 | **NEW (operational):** all DB artifacts predate the fixes | High (for DB work) | Open |

Also changed in fix round 1, recorded as tuning (not issues): `penetration`
weight 2000→4000 (retune accompanying the Issue-9 reduction change), all four
motion weights ×2 (`accel 12/24 → 24/48`, `jerk 150/250 → 300/500`),
`depth_push` → 0 (Issue 13), and the viewer temporarily dropped its
`kinematics_2` / `dynamics` panes (`tools/viewer/loading.py` — not a loss
change; noted so nobody hunts for a loader bug).

**Remaining majors in one line:** Issue 6 (weight direction — the one open
*design* question), Issue 7 (kept at dilate 8 by decision — holes at persistent
grips remain a known limitation until a depth-aware carve exists), Issue 5
residual (world-frame fill, matters for the 9 DB gap scenes), Issue 20
(uncertainty stage hard-crashes), Issue 22 (DB re-run needed). Everything else
still open is small.

---

## What was checked and is NOT broken

Recorded so future audits don't re-hunt these:

- **Penetration / attraction gradient signs are correct** (unit-tested):
  a vertex behind the surface is pushed out along `+n`; a vertex just outside is
  pulled in along `−n`.
- **Chamfer masking is numerically sound**: empty-depth frames, no-visible-vert
  frames, and the fp32 `finfo.max`-sum overflow path all produce finite losses
  and finite, correctly-zeroed gradients (invalid frames get exactly zero grad).
- **Smoothness has no quaternion double-cover artifact**: `so3_log` flips `q`
  when `qw < 0` (`../BetterRobot/src/better_robot/lie/_torch_native_backend.py:162-183`),
  so `tangent_difference` is always shortest-arc even if consecutive frames flip
  quaternion sign. `pose_prior_loss` / `root_orient_prior_loss` use `(q·q₀)²`,
  which is sign-invariant.
- **Adam DOF freezing is correct**: gradients of frozen groups are zeroed before
  `opt.step()` and a *fresh* optimizer is built per phase, so Adam moment state
  never moves a frozen coordinate. The L-BFGS phase deliberately frees all DOFs.
- **SMPLXMid contract holds** (verified numerically against pip smplx, ≤1e-6 m):
  q layout `[transl(3), quat xyzw(4), 51×quat]`; `from_classic` folds the pelvis
  offset so `q[:7]` is the true pelvis SE3, making `fold_root_to_world`'s left
  composition a rigid transform of the whole body; `to_classic` round-trips;
  all 52 joint names match `_hand_smplx_joints` / `_ANN_TO_SMPLX` /
  `classify_joints` exactly (no silent skips). Note: `mid.forward` raises on
  non-unit quats (pypose `mat2SE3` check), so `normalize_q` before every forward
  is correctness-critical, not hygiene.
- **Metric scale is applied consistently**: human points ×s, scene points ×s
  (before voxelization), extrinsic translations ×s — one consistent metric world
  shared by `scene.npz`, `human.npz`, `transform.npz`.
- **Penetration is effective overall** (init → final, % of near-wall subsampled
  verts more than 3 cm inside the wall, pre-fix runs): boulder_1 11.6→3.8%,
  competition_1 22.5→5.3%, sport_climb 18.0→7.4%. The residual is explained by
  Issues 1, 2, 7 plus the intended margin/Huber equilibrium.
- **Fix round 1 verified sound where checked** (2026-07-06): with
  `max_candidates=6000` + the scene quantile gate, 0/26–28 sampled frames
  truncate on any local clip; the scene quantile drops exactly the configured
  10%; `frame_valid` masking, the joint-mean reductions, the chamfer empty
  guard, the smoothness guard, and the pair gating are implemented as described
  and are autograd-clean (gates/denominators built from detached quantities,
  matching the pre-fix convention).

---

## HIGH severity

### Issue 1 — Scene-candidate truncation systematically drops wall points at the extremities

- **Status (2026-07-06): FIXED** — `[penetration] max_candidates = 1500 → 6000`.
  Verified on the re-optimized locals: 0 sampled frames exceed 6000 anywhere
  (per-frame in-box candidates p50 2101–3493, max 4533). The center-topk
  *selection rule* is unchanged, which is fine while the cap never binds — but
  it will silently reintroduce the corner bias on any scene whose in-box count
  exceeds 6000 (denser voxels, larger `prune_margin_m`, big outdoor scenes).
  Re-check the "frames > cap" number when the DB is re-run (Issue 22); if it
  ever binds again, switch the top-k criterion to distance-to-nearest-*vertex*.
- **Where:** `tools/human_optim/optimizer.py` (`_padded_scene_candidates`);
  knobs `configs/human_optimization.toml` `[penetration]`.
- **Mechanism (original):** per frame, scene points inside the posed-body AABB
  + 0.30 m are candidates for penetration / attraction / annotation terms. If
  more than the cap survive, the code keeps the k **nearest to the AABB
  center**. Points near the AABB *corners* are dropped first — and an
  outstretched hand/foot reaching for a hold is exactly what defines the AABB
  corner. A dropped point can never be a nearest neighbour, so those losses
  simply did not see the wall there.
- **Evidence (pre-fix, cap 1500):** 89–100% of frames exceeded the cap
  (median 2100–3500 in-box), and 7–37% of *relevant* candidates (within
  `max_dist_m = 0.15` of the posed body) were dropped — worst frames 258–415
  points, biased to the extremities. Post-fix: 0%.
- **Note:** `estimate_contacts` was never affected (AABB prune only, no cap).

### Issue 2 — Confidence gating (was: every `conf_thresh = 0.1` gate is dead) — EXPANDED

- **Status (2026-07-06): FIXED via quantile gates** — right family of fix, one
  measured refinement recommended below (per-frame instead of per-clip for the
  human cloud). This section now documents what VGGT confidence *is*, what the
  fix implemented, its measured behavior, and the recommended follow-ups, so
  the next change here doesn't have to rediscover the groundwork.

#### 2.1 What VGGT `depth_conf` actually is

`depth_conf = 1.0 + exp(confidence_logits)` — see
`third_party/vggt-omega/vggt_omega/models/heads/dense_head.py:150`. It is the
standard aleatoric confidence used in VGGT's training loss
(`conf·L − α·log conf`): **≥ 1 by construction, unbounded above, uncalibrated**.
Consequences that any gating scheme must respect:

- An absolute threshold below 1 can *never* fire — that is exactly why the
  original `conf_thresh = 0.1` gates were dead (measured: scene conf p5 =
  5.2–12.6, human conf p5 = 2.4–5.5 across clips; 0.00% of points below 0.1).
- Only the **ranking** is meaningful. The scale drifts between clips and — as
  measured below — between frames *within* a clip (the climber's apparent size,
  distance, and lighting change), even though VGGT-Omega produces the whole
  clip in one forward pass.
- If a linear-ish quantity is ever needed (e.g. for soft weights), the natural
  transform is `log(conf − 1)` (recovers the raw logits) or a within-frame
  rank / ratio-to-median — not the raw value.

#### 2.2 What fix round 1 implemented (verified in the working tree)

- **Human cloud (chamfer input):** `_person_depth_points`
  (`tools/human_optim/stages.py:50-65`) now drops the person's lowest
  `[chamfer].conf_drop_q = 0.3` fraction of points by confidence — quantile
  computed **per person over the whole clip**, applied once at load.
- **Scene cloud (penetration / attraction / annotation terms):** `_load_scene`
  (`stages.py:112-119`) drops the lowest `[penetration].conf_drop_q = 0.1`
  fraction of scene voxels per clip, **before** PCA normal estimation (so the
  normals see the cleaned cloud — correct order). `estimate_contacts` applies
  the same gate (`scripts/stages/estimate_contacts.py:128-134`), keeping the
  loss and the contact readout consistent.
- The in-loss absolute `conf_thresh = 0.1` parameters were kept as residual,
  non-binding floors and are documented as such in the toml.
- **Architecture note (good as is):** fuse still writes the *human* cloud raw
  (un-gated) and the scene per-frame-gated (`clean_depth`, per-frame quantile
  0.3). Gating at *consume* time keeps artifacts complete and thresholds
  tunable without re-fusing — keep this property.

#### 2.3 Measured behavior of the implemented gates (2026-07-06, locals)

- Scene q=0.1 removes exactly ~10% of voxels on every clip. Of the survivors,
  only 1.1–3.6% are single-pixel (`count == 1`) voxels.
- Human per-**clip** q=0.3 is well-behaved on clips whose per-frame confidence
  is stable, and degenerates on clips where it drifts:

  | clip / person | per-frame median-conf drift (p90/p10) | frames losing >60% of their points | frames losing >90% (≈ emptied) |
  |---|---|---|---|
  | boulder_1 p0 | 1.25× | 1.6% | 0% |
  | boulder_3 p0 | 1.28× | 2.9% | 0% |
  | competition_1 p0 | 1.27× | 1.5% | 0% |
  | competition_2 p1 | 2.03× | 30.4% | 0% |
  | sport_climb p0 | **4.35×** | 39.9% | **35.4%** |
  | sport_climb p1 | 1.69× | 36.9% | 8.1% |

  On sport_climb person 0, a third of the clip's frames end up with (almost) no
  depth points: chamfer silently drops those frames (the empty-frame path is
  safe — Issue 8's guard), leaving only the reprojection terms to hold the body
  there — the body can drift off the metric cloud on exactly the frames where
  depth was weakest. That is harsher than the intent ("drop the least-reliable
  30%") and is a *distribution* artifact: a per-clip threshold concentrates the
  drop on whole low-confidence frames.

#### 2.4 Recommendation (ranked)

1. **Switch the human-cloud gate from per-clip to per-frame** (drop each
   frame's lowest 30% by confidence). One-line change inside
   `_person_depth_points` (take the quantile within each frame's slice instead
   of over the whole person). By construction every frame keeps 70% of its
   points — the emptying failure mode disappears — while within-frame ranking
   still removes the silhouette-edge bleed the gate is for. It is also
   consistent with the two existing per-frame quantile gates (`clean_depth` at
   fuse, `estimate_scale`'s `conf_drop_q`). This is the one concrete
   improvement worth making over the current implementation.
2. **Keep the scene gate per-clip at q=0.1.** The scene cloud is a single
   merged object — per-frame doesn't apply — and voxel `amax`-confidence is
   high-biased anyway, so this gate is a fringe trim, which is all it needs to
   be. Optionally also drop `count == 1` voxels (single contributing pixel):
   measured at only 2.7–4.7% of voxels (the `min_scene_voxels=25` component
   filter already killed most junk), partially overlapping the conf gate —
   a small orthogonal cleanup; take it only if flying-voxel artifacts are ever
   seen near the body, otherwise skip.
3. **If chamfer still chases bad depth after (1): prefer soft weighting over
   harder gating.** The d2v direction (each depth point pulls its nearest
   visible vertex, no robustifier) is the mechanism by which residual bad depth
   drags the body. Weight each depth point's contribution by a within-frame
   scale-free score — rank / N, or `min(conf/median(conf), c)` — alongside a
   small per-frame tail gate (q≈0.1). Soft weighting discounts flaky depth
   without sacrificing coverage; raising the hard quantile does the opposite
   trade. (The v2d direction already has the per-vertex body-part weights —
   Issue 6 — so add any conf weighting there multiplicatively, not as a
   replacement.)
4. **Not now: absolute calibration.** The human-ruler stage already computes
   per-vertex `|log d_body − log d_vggt|` against conf, so conf→metric-error
   could be calibrated per clip and thresholded at a target error in cm — the
   only *principled* absolute gate possible with an uncalibrated confidence.
   It adds cross-stage machinery for marginal gain over per-frame quantiles;
   revisit only if a clip family shows quantile gating failing.
5. **Housekeeping:** the dead `conf_thresh` floors can stay (zero cost,
   documented) or be deleted for clarity — either is fine. What is *not* fine:
   `estimate_uncertainty` still reads the removed `[chamfer].conf_thresh` key
   and now crashes (Issue 20); when fixing it, import the optimizer's actual
   loader so its "same as the optimizer" data-fit loss stays true by
   construction.

### Issue 3 — `fps` was never written to `transform.npz`; the 30.0 fallback silently propagated

- **Status (2026-07-06): FIXED in code and local data; DB stale.**
  `fuse_geometry.py` now takes fps from `resolve_source` and writes
  `fps=np.float32(fps)` into `transform.npz` (folder/image sources fall back to
  30.0, documented). Verified present in all five re-fused local clips; the
  consumers (`stages.py` `tr.get("fps", 30.0)`) now find it. **The ClimbingVideos
  DB was not re-fused** — its `transform.npz` files still lack fps and its
  `kinematic_1.npz` files still say 30 while 83% of the 361 target scenes are
  24/25/50/60 fps (scenes.db fps column) → tracked as Issue 22.
- **Impact recap (original):** wrong `fps` metadata in every DB body npz;
  dynamics `dt = 1/fps` wrong (accelerations up to 4× off at 60 fps → forces
  systematically wrong) if run on DB scenes; kinematics itself barely uses fps
  (smoothness folds dt into its weights — note this still means the effective
  smoothing *per second* varies with true clip fps across the DB).

### Issue 4 — Contacts SDF: lateral verts marginally behind the tangent plane read as deep contact

- **Status (2026-07-06): FIXED.** `scene_sdf` (`tools/human_optim/contacts.py`)
  now returns the **perpendicular** signed distance `(v−p)·n` — the same
  convention as the penetration loss — with `dmin` kept as the far-guard
  (`contact_from_sdf` unchanged: `sdf < close_thresh ∧ dmin < max_dist`).
- **Why the observed contact increase is a correction, not a regression:** the
  round-1 rerun measured +8–26% contacts, dominantly hands. With
  `close_thresh_m = 0.02` and 0.05 m scene voxels, the *old* positive-side test
  (`Euclidean < 2 cm`) was geometrically almost unsatisfiable between voxel
  samples — a vertex resting exactly ON the wall plane midway between two
  samples read as "no contact". The perpendicular test fires whenever the
  vertex is within 2 cm of the local surface plane and within 10 cm of a
  sample — the right reading for a sparsely-sampled surface. The old negative-
  side artifact (lateral verts registering as deep contact) is gone with it.
- **Residual caveat:** contact can now be attributed via a tangent plane fitted
  up to 10 cm away (`max_dist_m`), so curved/thin geometry (hold edges) is a
  bit trigger-happier; normal quality matters more. Watch the contact overlays
  on hold-heavy clips; tighten `max_dist_m` if lateral false contacts appear.

### Issue 5 — Invalid-frame filling holds the pose in the camera frame

- **Status (2026-07-06): PARTIALLY FIXED.** New `frame_valid` plumbing
  (`optimize_person(frame_valid=...)`): the **init-projection anchor** now
  averages over valid frames only (`init_reprojection_gm(frame_mask=...)`,
  masked mean with safe denominator), and the **scene terms** skip invalid
  frames (their padded candidates are masked out in `_build_scene_cand`, so
  penetration / attraction / annotation terms see zero active verts there).
  Combined with the pre-existing zeroing of Sapiens scores + depth, invalid
  frames are no longer *anchored* to the camera-following held pose.
- **What remains (why this is still open in part):** `_fill_invalid_frames`
  still fills **camera-frame** params which are then folded with each frame's
  own `world_from_cam` (`stages.py`), so the held pose itself still *moves with
  the camera* in world space during a dropout — and **smoothness and the
  pose/orient priors still act on those frames**, coupling camera motion into
  the valid neighbours (weaker than before, since nothing pins the invalid
  frames anymore, but the pull at the gap boundary is still toward a
  camera-glued trajectory). Also `_metrics` ignores `frame_valid` (diagnostics
  include held frames — cosmetic).
- **Remaining fix direction:** fill in the *world* frame (fold to world first,
  then hold the nearest valid world pose), which makes the held segment
  genuinely static; optionally down-weight smoothness across gap boundaries.
  Affects the 9 known DB gap scenes; gap-free clips are untouched either way.

### Issue 6 — Chamfer/scale vertex weights concentrate 85% of their mass on hands+feet; direction contradicts the module's own guidance for scale

- **Status (2026-07-06): OPEN — deliberately deferred (design decision).**
  Nothing changed in fix round 1: chamfer still uses the extremity-up geodesic
  field (`stages.py` → `geodesic_endeffector_weights`, log-mapped [0.1, 10]),
  and `estimate_scale` still multiplies the same field into its weighted median.
- **Measured mass distribution** (identical on the full mesh and the 1000-vert
  chamfer subsample):

  | group | % verts | % surface area | % weight mass |
  |---|---|---|---|
  | hand | 14.9 | 4.4 | **64.4** |
  | foot | 4.8 | 6.2 | **21.0** |
  | forearm+shin | 8.2 | 20.2 | 10.9 |
  | everything else | 72.1 | 69.2 | 3.7 |

  The ×10 profile compounds with tessellation density (hands: 14.9% of vertices
  on 4.4% of area ⇒ ×3.4 extra). The module's own mitigations — `normalize()`
  (mean-one) and `vertex_areas()` (density correction), plus the caveat in its
  docstring — are unused on the wired path.
- **Direction question (scale):** `vertex_weights.py` `PROFILES` itself calls
  extremity-**down** "the more defensible direction for the single global scale
  factor", and the module docstring says VGGT distorts extremity depth most —
  yet the wired field is extremity-**up**, so the global metric scale is
  effectively anchored to hand pixels. Measured sensitivity (re-running the
  scale estimator on stride-3 frames; "up" reproduces the shipped values):

  | clip | uniform | up (current) | down |
  |---|---|---|---|
  | boulder_1 | 3.198 | 3.080 | 3.312 |
  | boulder_3 | 3.826 | 3.578 | 3.956 |
  | competition_1 | 8.302 | 8.067 | 8.370 |
  | competition_2 | 7.935 | 7.760 | 7.953 |
  | sport_climb | 11.503 | 11.562 | 11.506 |

  Up-vs-down spread is ~1–10% of absolute scale; "up" sits 2–6.5% below uniform
  on 4/5 clips. Per-frame MADs are similar across directions (no internal
  winner) — adjudicating needs external ground truth or a downstream metric
  (e.g. contact recall / penetration equilibrium under each direction).
- **Decision to make:** (a) direction per consumer — extremity-up is arguable
  for chamfer (seat the parts that matter for contact), much weaker for the
  global ruler; (b) whether to apply `normalize()` + `vertex_areas` density
  correction on the wired path; (c) whether [0.1, 10] is a milder-range
  candidate. Until decided, remember both consumers share one field: changing
  it moves the metric scale AND the chamfer emphasis together.

### Issue 7 — fuse's dilated human carve leaves scene holes exactly at persistent contacts

- **Status (2026-07-06): OPEN by deliberate decision.** In fix round 1,
  `--human-dilate 8→2` was tried (it recovered gripped-hold contacts) and then
  **reverted to 8**: the 8 px carve is doing real work suppressing mask-edge
  noise (person pixels leaking into the scene cloud), and that was judged more
  important. The working tree and the 2026-07-06 re-fused locals are at
  dilate 8 (scene point counts byte-identical to the pre-fix fuses:
  12621 / 10296 / 19656 / 19349 / 185553). So the hole-at-grips limitation
  below **stands as a known, accepted tradeoff** — do not "re-fix" it by
  lowering the dilate; the way to get both properties is a smarter carve
  (below). For calibration: the dilate-2 experiment moved contacts by roughly
  +11–290% vs the old baseline (most of the headline contact increase in
  round 1 was this, not the Issue-4 SDF change), and reverting to 8 gave back
  −14–28% contacts vs the dilate-2 run.
- **Mechanism (original):** the scene carve subtracts the union person mask
  dilated 8 px at VGGT res, per frame; the scene merges across frames, so wall
  geometry within the dilated silhouette survives only if some frame saw it
  un-occluded. A hold gripped the whole clip (plus its halo) has **no scene
  points at all**. 8 px ≈ 2.5 cm (boulder_1) / 6.1 cm (competition_2) / 8.7 cm
  (sport_climb) at the person's median depth — the same order as the attraction
  band (0.10 m) and larger than the contact threshold (0.02 m).
- **Impact:** `scene_attraction` may have nothing to pull a contacting hand
  onto (nearest surviving point is lateral by up to the hole radius), and
  `estimate_contacts` under-reports exactly the load-bearing contact. Compounds
  with the (now-fixed) Issue 1 in the same region.
- **Fix direction (respecting the dilate-8 decision):** a **depth-aware carve**
  is the option that keeps both properties — carve a pixel only if its VGGT
  depth is within a few cm of the person's depth there (mask-edge noise sits AT
  person depth and still dies; the wall/hold behind a gripped hand is several
  cm deeper and survives). Alternatives: inpaint hold geometry from
  neighbouring frames before the voxel merge, or accept the limitation and let
  the annotation-guided kinematics_2 supply contact where the scene has holes.

---

## MEDIUM severity

### Issue 8 — `chamfer_loss` crashed when a person has zero depth points in every frame

- **Status (2026-07-06): FIXED.** Early `return 0` when
  `depth_padded.shape[1] == 0` (`losses.py`, mirroring the penetration guard).
  Note this guard is also what makes the Issue-2.3 frame-emptying failure mode
  *silent* — another reason to prefer the per-frame quantile (2.4.1).
- **Original trigger:** a tracked person that never yields confident human-cloud
  points; `pad_frames` → `(T, 0, 3)` → `cdist → .min(2)` raised `IndexError`
  (reproduced by unit test).

### Issue 9 — Per-frame-mean reduction gave one-active-vertex frames outsized weight

- **Status (2026-07-06): FIXED.** `scene_penetration`, `clearance_repulsion`,
  and `scene_attraction` now reduce as a **joint mean over all active
  vertex-instances** (Σ penalty ÷ total active count) instead of
  mean-per-frame-then-mean-over-frames. A frame with one surviving active
  vertex no longer carries the weight of a dense frame. The denominator is
  detached (mask built from detached quantities), matching the previous
  convention; the empty-active early-return is preserved. `penetration`'s
  weight was retuned 2000→4000 alongside (the joint mean changes the loss
  scale); the five locals were re-run and visually checked after the retune.
- **Flip side to be aware of (accepted):** gradient mass now scales with the
  active count, so a many-vert torso-against-slab frame outweighs a
  single-fingertip frame — the intended direction of the fix, but it means
  sparse-contact frames converge more slowly on the scene terms.

### Issue 10 — Dynamics applies contact forces with a zero torque arm at the representative joint

- **Status: OPEN** (unchanged; dynamics is currently deprioritized — the viewer
  temporarily dropped its dynamics pane). Original finding stands: `_build_fext`
  scatters only a linear force at the group's representative joint (wrist/foot)
  with a zero angular block, so the `r × F` moment of a fingertip contact about
  the wrist is silently dropped; wrist/forearm torques underestimated.
- **Fix direction:** apply per-member forces at the actual contacting joints,
  or add the angular part from the mean contact offset of the group's
  in-contact vertices.

### Issue 11 — Dynamics loss weights use two different conventions

- **Status: OPEN** (unchanged). `(w_base·τ)²` and `(w_mag·f)²` put the weight
  inside the square (effective w²) while `force_smooth` / `torque_smooth`
  multiply outside. Pure tuning-surprise hazard; harmless until someone
  re-tunes the dynamics column. Fix by moving all weights outside the square
  and re-folding the current effective values (`base 1.0, mag 1e-4`).

### Issue 12 — `joint_velocity_loss` gated on the outgoing frame only

- **Status (2026-07-06): FIXED.** Now gates on
  `contact[t] == 1 ∧ contact[t+1] == 1` (both ends of the step), matching the
  dynamics stage's `fpair` convention; release steps are no longer damped, and
  any step touching an invalid frame drops out for free (held frames carry
  label −1 ≠ 1). The `n_steps` diagnostic was updated to count pairs.

### Issue 13 — `depth_push_loss`'s documented mechanism was stale since scale was frozen

- **Status (2026-07-06): FIXED** by disabling: weight column zeroed
  (`depth_push = [0, 0, 0, 0]`, with an explanatory comment) — the term's
  original justification (body grows via free scale) died when scale froze, and
  chamfer owns depth. Code retained for a future re-evaluation; if it is ever
  re-enabled, rewrite the docstring to describe the actual equilibrium first.

### Issue 14 — `smoothness_loss` returned NaN for clips shorter than 4 frames

- **Status (2026-07-06): FIXED.** Each block's `.mean()` is guarded
  (`numel() > 0` else 0), so `T < 3` (accel) / `T < 4` (jerk) contribute 0
  instead of NaN.

---

## LOW severity / diagnostics / docs

### Issue 15 — Reported metrics don't measure the optimized objective

- **Status: OPEN** (unchanged, and slightly wider now): `_metrics` chamfer is
  unweighted while the optimized chamfer uses the vertex weights; the Sapiens
  metric uses a binary score threshold vs the loss's score-weighted GM; and
  `_metrics` does not apply `frame_valid`, so the before/after penetration and
  init-drift diagnostics include held invalid frames. Diagnostics only.

### Issue 16 — Two diagnostics report something other than what their names say

- **Status: HALF-GONE** (2026-07-11 cleanup). The "mean_shortfall_cm" naming
  issue lives on in `clearance_penalty` (the `clearance_repulsion` wrapper was
  removed): it averages the raw signed distance of pushed verts, not the
  shortfall. Log readability only. The velocity-gated contact diagnostic was
  removed with its long-dead loss.

### Issue 17 — Scene candidates are stale within a phase

- **Status: OPEN** (unchanged; documented tradeoff, less harmful now that the
  cap doesn't bind — a stale set is at least a *complete* stale set). Candidates
  are built once per scene phase from the phase-start pose (+0.30 m margin) and
  reused for all iterations. Rebuild every N iterations if annotation phases
  ever move labeled joints beyond the margin.

### Issue 18 — `human.npz frame_id` is the VGGT enumeration index; alignment guaranteed only indirectly

- **Status: OPEN** (unchanged, latent). Alignment with original frame indices
  holds because `estimate_body_sam3d` raises unless VGGT covered every tracked
  frame (identity mapping ≤ 500 frames); a longer clip fails upstream. If that
  guard ever changes, `run_kinematic` has no local check — add
  `assert extr.shape[0] == N` and translate via `frame_indices` to make the
  invariant local.

### Issue 19 — Deleted design doc still referenced

- **Status: OPEN.** `docs/geometry_fusion_plan.md` (deleted in `2048a56`) is
  still cited by `CLAUDE.md` and `tools/human_optim/camera.py:9`. The
  `geometry/*.npz` contract lives only in `fuse_geometry.py` — Issues 2/3/18
  are exactly the drift a written contract prevents. Restore a minimal contract
  doc (keys, shapes, frames, conf semantics incl. §2.1 above, fps) or fix the
  references.

### Issue 20 — NEW (introduced by fix round 1): `estimate_uncertainty` crashes on the renamed config key

- **Status: OPEN — hard crash.** `scripts/stages/estimate_uncertainty.py:522`
  reads `knb["chamfer"]["conf_thresh"]` by direct indexing; fix round 1 renamed
  that key to `conf_drop_q`. The defaults branch (`:526-530`) only applies when
  the config *file* is absent — with the default config present the stage
  raises `KeyError: 'conf_thresh'` at startup (reproduced via `load_config` +
  key access).
- **Also (semantics drift, pre-existing but now larger):** its private
  `_person_depth_and_conf` (`:126-141`) still gates absolutely (a no-op), so
  its "same weights + kernel as the optimizer" data-fit loss L(t) is computed
  on a differently-gated depth cloud than the optimizer now uses — the
  relative-score axis silently loses its "same objective" property.
- **Fix direction:** read `conf_drop_q` and reuse the optimizer's actual loader
  (`tools.human_optim.stages._person_depth_points`, plus its conf twin) instead
  of a private copy, so uncertainty tracks the optimizer by construction.

### Issue 21 — NEW: `diagnose_human_optim.py` drifted from the optimizer

- **Status: OPEN — silent drift, no crash.** `scripts/diagnose_human_optim.py`
  keeps a private absolute-gate copy of `_person_depth_points` (`:62-64`) and
  reads `chcfg.get("conf_thresh", 0.1)` (`:305`) — the `.get` fallback masks
  the rename, so it silently rebuilds the *un-gated* depth cloud; it applies no
  scene quantile at load; and its per-frame loss re-implementation still uses
  the old per-frame-mean reduction (`:157` area). Its per-frame *shapes* remain
  useful; its absolute values no longer reproduce the optimizer's terms.
- **Fix direction:** import the shared loaders/reductions from
  `tools/human_optim` instead of private copies (same remedy as Issue 20).

### Issue 22 — NEW (operational): all ClimbingVideos DB artifacts predate the fixes

- **Status: OPEN.** The 361-scene DB features (geometry from 2026-07-03,
  human_optim from 2026-07-04) were built entirely pre-fix: `transform.npz`
  without `fps` (30 fallback wrong for 83% of scenes), candidate truncation at
  1500 (bound on ~all frames), dead conf gates, per-frame reductions, old
  weights (`depth_push` 2, `penetration` 2000, motion ×1). The five local clips
  were re-fused + re-optimized on 2026-07-06; the DB was not.
- **Fix direction:** after the remaining decisions land (Issue 6 weight
  direction, Issue 2.4.1 per-frame quantile; Issue 7 is decided — dilate stays
  8) re-run the DB drivers once — fuse (`precompute_geometry_scenes`) then
  human_optim — rather than re-running per fix. Budget from the 07-03/07-04
  runs: ~77 min (scale+fuse) + ~3.1 h (human_optim) on 3 GPUs.

---

## Reproduction notes

- The empirical numbers came from throwaway probe scripts (session scratchpad,
  not preserved): (A) unit probes calling `chamfer_loss` / `scene_penetration`
  / `scene_attraction` on synthetic tensors with `requires_grad` to check
  signs, masking, and the empty-depth crash; (B/D/E/F) per-clip probes posing
  `kinematic_1.npz` bodies with `SMPLXMid` on CPU (stride-sampled frames),
  replaying `_padded_scene_candidates`' AABB+top-k logic against
  `geometry/scene.npz`, and reading confidence percentiles from `scene.npz` /
  `human.npz`; (G) a scale re-run importing
  `scripts.stages.estimate_scale._frame_log_scale` +
  `tools.geometry.human_scale.combine_log_scales` with uniform / up / down
  weight fields ("up" reproduces the shipped `geometry/scale.npz` values,
  validating the harness); (H) the init-vs-final SDF comparison using the same
  candidate/normals code as the losses.
- The 2026-07-06 fix-verification probe repeated (B/D) against the fixed tree:
  cap 6000 truncation counts, scene-quantile drop fractions, scene `count`
  distribution, per-frame human-conf drift + per-clip-quantile emptying rates
  (the Issue 2.3 table), artifact mtimes + scene-count byte-comparison (the
  Issue 7 non-landing evidence), and the Issue 20 KeyError repro
  (`load_config` + key access).
- DB fps distribution: `scenes/scenes.db` `fps` column over
  `scene_db.select_targets(conn, categories=(1,2), rope_supported=0)`.
- External adversarial review transcript:
  `~/.claude/handoffs/human_optim_losses.codex.out` (prompt alongside it).
