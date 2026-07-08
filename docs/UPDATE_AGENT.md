# UPDATE_AGENT — round 9: VGGT depth streaks (measure → robustify → rescale)

*2026-07-08, rev 9. You are the implementing agent. `docs/UPDATE.md` rev 11 is
the human-facing plan — read it first. This file is the handoff: verified
state, code anchors, task specs, gates. Rounds 1–8: git history. Trust
function/key names over line numbers (the planning session verified them at
HEAD `cc92cdb`, they will drift as you edit).*

## §0 State (verified 2026-07-08 by the planning session)

**Repo**: HEAD `cc92cdb` on `main`. Working tree: one modified file,
`scripts/render_vertex_weights.py` — the user's kept "extreme" montage
candidate set (viz-only, no optimization wiring). Preserve it; you may commit
it as-is in your first commit, labeled viz-only.

**Local clips**: `out/<stem>/` for the 5 labeled clips — `boulder_1`,
`boulder_3`, `competition_1`, `competition_2`, `sport_climb` — are at HEAD
state. Annotations `data/<stem>.npz` are SACRED: bit-identical all round.
Label baseline (unchanged since round 4): pooled P 0.896 / R 0.678 /
F1 0.772 / FPR 0.066 (scorer pools exclude sport_climb by default).
**Re-derive the baseline at HEAD yourself before any change** (§2).

**Corpus** (DB root `/data3/rikhat.akizhanov/better/data/ClimbingVideos`):
361 scenes, round-8 frozen state — `features/human_optim/` holds
`kinematic_1.npz` + `contacts_1.npz` (confidence `confidence_schema=1`, a
frozen interface for the learning pipeline) + `uncertainty_1.npz`. The
canonical `features/` trees are a deliverable. **Never leave experimental
outputs there** — redirect experimental runs (§2), or if a driver can't
redirect, restore the touched scenes with a baseline-config re-run before you
finish.

**Showcase scenes** (user-picked, presumed bad-depth; verify with T0):
`PfwFWSoPF_k_0026, 2nwG1Qa5k_c_0115, 9RvrpYoWDoo_0000, CRquJ8H1QLI_0001,
3DAgLpPXVCs_0132, gtIGgs_y2nY_0024, vcRBT9Q9jXM_0029`. No contact labels →
qualitative evidence only (streak meter, overlays, `view_3d --scenes`).

**Round-9 scope guards**: do NOT touch the contact-confidence formula or the
`estimate_contacts` readout knobs — only the BODY (and the geometry it is fit
to) may improve. `kinematics_2`/`dynamics` stay frozen. Commit directly to
`main` (repo rule).

## §1 Verified code anchors

- **`chamfer_loss`** `tools/human_optim/losses.py:135-186`: bidirectional,
  RAW L2 via `torch.cdist` + `min` — no robust kernel, no cap. The ONLY
  unrobustified data term (reprojection = Geman-McClure `gmof`,
  `losses.py:26-36`; penetration/attraction = Huber `_huber`).
  `vertex_weights` enters ONLY the vertex→depth mean (`losses.py:181-182`,
  normalized weighted mean — wiring is correct); the depth→vertex mean
  (`d2v`, `losses.py:177,183`) is uniform. Streaks pull mainly through `d2v`
  and by winning `v2d` nearest-neighbour races.
- **Cloud loading at optim time** `_person_depth_points`
  `tools/human_optim/stages.py:50-81`: per-frame conf quantile
  (`[chamfer].conf_drop_q=0.3`), stride cap `max_points=1500`, no distance
  cap, no body gate. Per-point conf IS stored in `geometry/human.npz`; pixel
  provenance is NOT.
- **Fuse** `scripts/stages/fuse_geometry.py:135-146`: human cloud =
  `clean_depth(depth, edge_rtol=0.04)` (NO conf arg) ∩ tight person mask;
  scene = conf-gated (`conf_drop_q=0.3`) + dilated-human-subtracted +
  voxel-merged + CC-filtered. `tools/geometry/clean.py:27-35` = the only
  depth-map filter (utils3d `depth_map_edge` flying-pixel filter); its
  `normal_map_edge` path is inert (no caller passes normals).
- **Scale stage** `scripts/stages/estimate_scale.py`: per-frame
  `weighted_median(log_ratio, w)` with `w = conf × vertex_weight`
  (`:116-117`); vertex weights hardcoded extremity-UP
  (`geodesic_endeffector_weights` defaults lo 0.1 / hi 10.0 / r 1.0,
  `:176-177`) — no config knob; NO edge filter applied to the sampled depth.
  Clip pooling `combine_log_scales` `tools/geometry/human_scale.py:44-83`
  (MAD gate 3.0 → conf-weighted mean of per-frame log-scales).
- **Init body precedes cloud load** `stages.py:299-307` (`q_world` built at
  `:299-301`, `_person_depth_points` at `:307`): a body-derived gate can be
  inserted with no reordering. SAM-3D metric depth witnesses: `pred_cam_t`
  in `sam3d/params.npz`, `transl` in `sam3d/smplx_params.npz`.
- **Normals** `tools/geometry/normals.py:43-51`: k-NN (k=16) covariance
  `eigh`, camera-oriented; eigenvalues currently discarded (T2b would return
  them).
- **Scorer** `scripts/score_contacts.py`: `--predictions <clip dirs>`,
  `--body-path` to score a variant body npz without touching `out/`,
  `--exclude-from-pool` default `sport_climb`. Reports P/R/F1 + FPR per clip,
  per coarse joint, pooled, over a close/still grid.
- **Corpus drivers** (all take `--scenes` comma-list, `--devices`):
  `precompute_geometry_scenes.py` (scale+fuse, symlink farm),
  `precompute_human_optim_scenes.py`, `precompute_contacts_scenes.py`. Only
  the contacts driver has `--out-root`. For experimental corpus runs, add a
  matching `--out-root` to the geometry + human-optim drivers (mirror the
  contacts driver's farm redirection) — or restore touched scenes afterward.
- **Doc rot** (fix in your wrap commit): CLAUDE.md still mentions a
  "depth-push" term; `grep -rn depth_push` = zero hits, the term is gone.

## §2 Eval protocol

**Baseline first.** Before any code change:
`uv run python scripts/score_contacts.py --predictions out/boulder_1 out/boulder_3 out/competition_1 out/competition_2 out/sport_climb --device cuda`
Record the full table → `out/_round9/results.md`. This is the number every
variant is judged against; do not quote the memory baseline unverified.

**Variant runs, locals.** Never clobber `out/<stem>` HEAD outputs. Per
variant, build `out/_round9/<variant>/<stem>/` as a symlink farm (pattern:
the corpus drivers): symlink the unchanged upstream stage dirs, make real
dirs only for what the variant regenerates —
- optimizer-side variants (T1a/T1b): real `human_optim/`; symlink `sam3d`,
  `sapiens`, `geometry`, `sam3`, `vggt`.
- scale-side variants (T1c): real `geometry/` + `human_optim/` (rerun
  `estimate_scale` → `fuse_geometry` → `human_optimization` → 
  `estimate_contacts`).
Then score with `--predictions out/_round9/<variant>/<stem> ...`.

**Metrics logged per variant** (raw, in `results.md`): pooled + per-clip
P/R/F1/FPR at the reference cell, the reproj/chamfer guard numbers (mean
Sapiens-reproj px and chamfer cm from `diagnose_human_optim.py` or the
optimizer's logged final losses), per-clip scale factor + per-frame scale
spread (T1c), and the T0 streak-slice metrics (below). One change at a time;
combined config only after singles are read.

**Gates** (adopt a knob only if all hold on the 4-clip pool):
- pooled F1 ≥ baseline + 0.01 (the round-6 "robust" bar) — OR the streak
  slice improves markedly while the pool is flat AND the corpus showcase
  visibly improves (say so explicitly; the user owns that call);
- pooled precision ≥ baseline − 0.005 and FPR ≤ baseline + 0.005;
- no single clip loses > 0.01 F1;
- reproj guard within noise (< +0.3 px pooled mean);
- parity: every new knob at its neutral default reproduces HEAD outputs to
  fp tolerance (assert on one clip before sweeping).

**Corpus showcase.** After a winner exists locally:
`uv run python scripts/precompute_geometry_scenes.py --scenes <7> --devices …`
(only if T1c adopted) → `precompute_human_optim_scenes.py --scenes <7>` →
`precompute_contacts_scenes.py --scenes <7>`, all REDIRECTED (§1). Compare
before/after with the T0 meter + `view_3d --scenes` + overlay mp4s; capture
screenshots/notes in `results.md`. Canonical `features/` gets the winner only
when the user approves a corpus re-run (not this round unless asked).

## §3 Tasks

### T0 — streak meter + go/no-go slice (`scripts/diagnose_depth_streaks.py`)

New diagnostic, no optimization change. Per `--predictions` clip dir, per
person, per frame: load the human cloud exactly as the optimizer does
(`_person_depth_points`, same conf gate), pose the INIT body (converter
params — independent of VGGT), assign each cloud point to a coarse part
(the 10 annotated groups, `_ANN_TO_SMPLX`) **in image space** (nearest
projected init-body joint; streaks share the part's pixel footprint, so 3-D
assignment would misfile them), then per part-frame compute the along-ray
extent = p95 − p5 of point depth along the camera ray, and the offset of the
point median from the init-body part depth. Flag "streaked" when extent >
max(3 × nominal part thickness, 0.30 m) — calibrate the threshold so healthy
clips flag ≈ few %.

Outputs: per-clip CSV (`frame, person, part, n_pts, ray_extent_m,
median_offset_m, streaked`) under `out/<stem>/geometry/diagnostics/`, a
per-clip histogram PNG, and a pooled summary table printed + saved.

**The go/no-go slice** (the round's most important number): on the labeled
clips, split annotated joint-frames into streaked vs clean by the flag and
report contact FP/FN rates in each. If contact errors do NOT concentrate in
streaked part-frames, say so loudly — T1 then can't move the pooled numbers
and the round's weight shifts to the corpus showcase + scale stability.
Run the meter on the 7 showcase scenes too (symlink farms) to confirm they
actually exhibit streaks. Commit T0 with its numbers in the message.

### T1a — robust chamfer kernel

`chamfer_loss`: apply a Geman-McClure kernel to the per-point min-distances
in BOTH directions before averaging (reuse the `gmof` pattern;
`gm(d) = ρ²·d²/(d²+ρ²)` — note current chamfer averages raw distance, not
squared; keep the kernel's scale in meters and preserve the current
magnitude at small d so the `[loss_weights]` chamfer weight keeps meaning:
e.g. `d_robust = d·ρ/(d+ρ)` (a saturating first-order kernel) is acceptable
too — pick one, justify in the commit).

Knob: `[chamfer] robust_scale_m` (0.0 = off = parity default). **Phase
hazard**: the `root` phase needs long-range chamfer pull to seat depth from a
distant init. Make the kernel phase-aware — off (or ≥ 0.5 m) during `root`,
target scale in `full`/`refine`. Implementation freedom: a second knob
(`robust_from_phase = "full"`) or a per-phase scale list; keep it simple.

Sweep ρ ∈ {0.10, 0.15, 0.25} on the locals. Log the fraction of cloud points
whose pull is > 50 % attenuated (healthy clips should sit low — if a healthy
clip attenuates half its cloud, ρ is too tight). Gate per §2.

### T1b — per-point weights on the cloud side (`d2v`)

Give each human-cloud point a reliability weight
`w = conf_norm × clip(edge_dist_px / edge_ref_px, ·, 1)`:
- `conf_norm`: per-frame normalized VGGT conf (already stored per point).
- `edge_dist_px`: distance from the point's source pixel to the person-mask
  boundary. Pixel provenance is not stored in `human.npz` — preferred:
  compute at fuse time (`cv2.distanceTransform` on the person mask) and save
  per-point `edge_dist_px` into `human.npz` (schema addition; re-run fuse on
  the 5 locals + farms — cheap). Alternative (no schema change): reproject
  points with the saved cameras at load time and sample the mask's distance
  transform; pick one and say why.

Use `w` as a weighted mean over `d2v` (mirror of the existing
`vertex_weights` weighted mean on `v2d`); optional sweep arm: also hard-drop
points with `w < w_min` from the cloud entirely (they then can't win `v2d`
races either — partially overlaps T1a; measure the marginal effect).
Knobs in `[chamfer]` with neutral defaults = parity (uniform `w≡1`).
Sweep `edge_ref_px` ∈ {4, 8} × conf on/off. Gate per §2.

### T1c — scale stage: weighting flip + edge filter

`estimate_scale.py`:
- `--vertex-weighting {extremity_up|uniform|extremity_down}` (default
  `extremity_up` = parity; `extremity_down` = swap lo/hi of the geodesic
  field — read the ruler at the torso).
- `--edge-rtol` (default 0.0 = off): apply `depth_map_edge` (as in
  `clean.py`) to invalidate flying pixels in `depth_rel` before sampling
  vertices against it (`_frame_log_scale`'s `valid`).

Report per variant, per clip: clip scale factor vs HEAD, per-frame log-scale
spread (std + MAD — the stability early-warning), n frames MAD-gated out.
Then run the downstream chain (fuse → human_optim → contacts → score) for
the best 1–2 variants only (6 variants × full chain is wasteful; pick on
scale-stability + sane factor movement first). Gate per §2 on the downstream
numbers; a scale that moves > ~5 % on a healthy clip with unchanged spread is
suspicious — flag rather than adopt.

### T1d — combine + confirm

Combine the adopted knobs, one confirmation run over all 5 locals + the 7
showcase scenes (redirected), full table into `results.md`, per-task commits
already landed; wrap commit updates CLAUDE.md (config knob docs + kill the
"depth-push" mention) and this file's §0. Set the winning values as TOML
defaults only if the user's gates passed unambiguously; otherwise leave
neutral defaults and present the table.

## §4 Second wave (run ONLY what T0/T1 evidence justifies; one at a time)

- **T2a init-body depth corridor** (preprocessing "squeeze"): in
  `_person_depth_points` (init body is in scope upstream — pass it in),
  soft-clamp each point's along-ray coordinate into
  `[init_part_depth − τ, init_part_depth + τ]`, τ ∈ {0.3, 0.4} m — clamp,
  don't delete (keeps lateral signal). Guards: skip untracked/invalid frames
  (`_fill_invalid_frames` provenance); log per-frame clamped fraction and
  alarm > 50 % (init body may be the wrong one there). Strongest candidate if
  T0 shows large `median_offset_m` (whole-part displacement), which T1a/T1b
  (outlier-shaped fixes) treat poorly.
- **T2b per-point streak score**: `estimate_point_normals` optionally
  returns eigenvalues; score = elongation (λ1 ≫ λ2) × |dominant axis · ray|;
  fold into T1b's `w`. Run if T1b helps but mask-edge distance proves a weak
  proxy.
- **T2c wall-contamination drop**: drop human points within ε of the scene
  cloud AND ≥ 20 cm behind the init-body part depth (both conditions — hands
  genuinely touch the wall in this data; naive proximity dropping will eat
  true contact evidence and inflate precision while killing recall — watch
  the contact metrics specifically).
- **T3a along-ray anisotropic chamfer**: decompose residuals into along-ray
  (compress: log-depth or ×α < 1) + lateral (keep metric) using per-point ray
  dirs (point − camera center, cameras in `transform.npz`). The principled
  version of T2a's squeeze inside the loss; try if T1a helps but costs
  healthy-clip depth seating (gate: chamfer-cm guard).
- **T3b per-part chamfer gating**: T0 flags → zero that part's vertices'
  chamfer contribution in flagged frames (vertex→part via the existing
  nearest-joint lift machinery); smoothness/init-anchor carry the part.
  Cheap and surgical; needs T0 flags to be precise (validate visually first).
- **T3c render-and-compare log-depth + free-space** — sketch only this
  round; escalate to the user before building (differentiable depth render =
  real scope).

## §5 Discipline

- Per-task commits on `main` (`Round 9 T0: …`), numbers in commit messages.
- Parity green before every sweep (neutral knob ⇒ HEAD-identical outputs on
  one clip, fp tolerance).
- Raw metrics + trajectory into `out/_round9/results.md` as you go; the user
  owns the verdict — report nulls plainly (this project ships nulls).
- Self-confirmation trap (standing): recall is inflatable by any stronger
  pull toward the wall; precision + FPR are the honest signals. Any variant
  that raises recall AND FPR together is doing the wrong thing.
- Skepticism: treat this file's claims as leads, not ground truth — re-verify
  an anchor before building on it; the planning session verified at
  `cc92cdb` only.
- Don't touch: annotations, contact-confidence/readout code and knobs,
  kinematics_2/dynamics, canonical corpus `features/` (redirect/restore),
  the user's `render_vertex_weights.py` edit (preserve/commit, never revert).
