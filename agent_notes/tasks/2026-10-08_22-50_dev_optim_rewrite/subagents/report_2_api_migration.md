# Report 2 — BetterHuman + BetterRobot API migration on dev (mars_top, 2026-10-09)

Verdict: dev now runs on BH dev + BR dev with no behaviour change. pytest on dev: 246 passed, 4 skipped, 0 failed, 0 collection errors (main: 250 pass; dev before this step: 104 passed, 1 failed, 9 files not collected). A CPU parity probe of the migrated paths (body, kindyn RNEA + Jacobians, force witness, BEDLAM readout, AMASS CoM) agrees with main to float32 rounding. All 13 stage CLIs + view_3d answer `--help`. Regression on the 4 clips: see §4.

## 1. What changed, per file (26 files, +203 / −723 lines; `git status` on dev)
- tools/body.py — BH dev API: `families.smplx(path, num_betas=10, hands=True, face=False)` → `bh.Body(spec, density=load_density_table(...), contact_frames=json)` → `.bake(betas=)` (Identity) → `.forward(q)` (HumanState). `_Posed.kinematics` (BR `KinematicState`) replaces `.data`; `body.skin_matrix` replaces the dense-LBS rebuild; landmark tables from `families.tables.SMPLX`; classic ↔ q via `smplx_from_classic` / `smplx_to_classic`. New `models_dir()` (`$BETTERHUMAN_MODELS_DIR`, else `<BH checkout>/models`) and the contact JSON resolved from the installed BH checkout — no absolute main paths. New `Body.density (52,)`.
- tools/human_optim/kindyn.py — `forward_kinematics_state` + `inverse_dynamics(model, state, v=, a=, fext=)` + `frame_jacobian(model, state, frame=)`; `forward()` runs one FK state and feeds it to both the placements and the RNEA (main ran FK twice); `.shaped().model`; mass override via `model.with_values(body_inertias=)`; finite differences imported from `.motion`.
- tools/human_optim/forces.py — same rewrite in `FrameDynamics.__init__` (one state: RNEA, 35 frame Jacobians, frame poses).
- tools/bedlam_contacts.py — `frame_jacobian(posed.model, posed.kinematics, frame=…)`; `.model`; finite-diff import.
- scripts/label_amass.py — `better_robot.dynamics.center_of_mass(model, model.fk(qn))` (the `centroidal` module is gone).
- tools/human_optim/motion.py — now hosts `velocity_acceleration_from_trajectory` (moved verbatim from smplx_robot/dynamics.py).
- tools/human_optim/vertex_weights.py — NEW (150 lines): the used part of smplx_robot/vertex_weights.py (`classify_joints`, edge graph, geodesic field, `falloff`, `scale_to_range`, `geodesic_endeffector_weights`). PROFILES, group/segmentation weights, `normalize`, `vertex_areas` went with the package (no importer).
- tools/human_optim/stages.py — `load_body(device=, frames_allow=, sole_offsets=)` replaces `load_smplx_mid`; dead `_DEFAULT_SMPLX_MODEL` + the "caller put better_human on sys.path" docstring gone.
- tools/smplx_robot/ (4 files, 431 lines) — trashed with the `trash` helper → /home/rikhat.akizhanov/better/BetterVideoReconstruction/trash/smplx_robot (the helper routes a linked worktree's trash to the main checkout; gitignored there).
- scripts/stages/estimate_scale.py — `load_body(device=device)`; sys.path hack → repo-root insert only; dead model path gone.
- scripts/stages/human_optimization.py, scripts/stages/convert_mhr_to_smplx.py, scripts/peter_gt/overlay_human.py — the dead `../better_human/src` sys.path lines removed; the repo-root insert every stage needs stays.
- scripts/peter_gt/fit_multiview.py — `identity.v_shaped`, `identity.model.values.body_inertias`.
- tools/viewer/loading.py — `_MHR_MODEL/_MHR_LOD/_MHR_CACHE`, `_load_mhr`, `_mhr_skinned` and the `mhr_1_sup` source removed; `kindyn_manual` dropped from the skinned / force / skeleton source tables (−123 lines). tools/viewer/scene.py, scripts/view_3d.py, tools/human_optim/paths.py, tools/human_optim/__init__.py — docstrings only.
- tests — test_fps_scaling (import path), test_body_density (`body.density`, `.model`), test_kin2_kindyn_frames (`.model`), test_fuse_scale (dead `import scripts.stages.human_optimization`), test_body_sole_offset (poses identity JOINT quaternions; it fed all-zero quaternions, which BR main's FK returned finite values for and BR dev's returns NaN — no pipeline path does this: every `forward` call goes through `normalize_q`, and both FKs normalise off-unit quaternions identically, measured below).

## 2. Verification that does not need the GPU
- pytest (dev venv, CPU, `BVR_DATA_ROOT=data_dev`): 246 passed, 4 skipped, 0 failed in 84 s. Every failure of report_1 (A: `update_frame_placements`, B: `compute_joint_jacobians`, and the C–G symbol breaks behind them) is gone.
- `--help`: all 12 `scripts/stages/*.py` + `scripts/view_3d.py` exit 0 (these also cover the removed sys.path hacks).
- Viewer loader: `load_clip_data(boulder_1, device="cpu")` → skinned {original, kinematics, kinematics_2, kindyn_1, prefit}, forces {kindyn_1}, skeletons {kinematics, kinematics_2, kindyn_1}.
- Parity probe (scratchpad `parity_probe.py`, same script run in main's venv and dev's venv; CPU; boulder_1 person 0, first 12 frames; `|x|max` = magnitude of the quantity):

| quantity (main vs dev) | max abs diff | note |
|---|---|---|
| q from classic (12,211), classic round trip, v_shaped, rest_joints, body_inertias (53,10), density, lbs_weights, tau_weight, total mass | 0 (lbs 1.2e-7) | identical |
| joints (12,52,3) / vertices (12,10475,3) / contact frames (12,35,3) | 3.6e-7 / 9.5e-7 / 3.6e-7 m | rounding |
| joint quats / frame quats | 4.0e-5 / 3.9e-5 deg as rotations | frame quats: 210/420 sign-flipped (same rotation) |
| ∂(Σ vertices + Σ frames)/∂q | 3.4e-3 of 1.1e4 | 3e-7 relative |
| kindyn `forward` tau (12,159), `frozen_dynamics` tau_free, jac (12,35,3,159) | 6.1e-4 / 8.5e-4 N·m of 2.8e3; jac 7.2e-7 | 3e-7 relative |
| kindyn ∂‖tau‖²/∂q, cone axes / generators, f_eff | 3.2e2 of 1.6e9; 8.6e-7; 0 | |
| force witness tau, jacobians, points, normals, body weight | 8.5e-4 N·m; 7.2e-7; 3.6e-7; 7.2e-7; 0 | |
| BEDLAM readout frames / velocities / normals / mesh_low | 4.8e-7 / 4.2e-6 / 6.9e-7 / 3.3e-7 | |
| AMASS CoM force (12,3) in BW | 4.4e-5 of 5.2 | second differences |
- Off-unit quaternions (every quaternion ×1.05) pose identically to normalised ones on both main and dev (3e-7 m): both FKs normalise internally, so stored `q` is safe.
- Raw-quaternion comparisons across calls: the only consumer of the FK quaternions (`jwq`/`fwq` → `joint_velocity_loss`, losses.py:538) uses `1 − dot²` and `|dot|`; `frame_normals`/`build_fext` go through `so3.to_matrix`; the viewer converts to rotations; no npz stores FK quaternions. Nothing to fix.

## 3. Assumptions
1. The 35-frame contact JSON stays in BetterHuman (`studio_data/frames/smplx_neutral/better_contacts.json`, identical on main and dev BH) and is found through the installed package: `Path(better_human.__file__).resolve().parents[2]` (the editable checkout). The same root is the default models dir unless `BETTERHUMAN_MODELS_DIR` is set (BH dev has no models-dir helper; its loaders read only that env var). A non-editable (wheel) install would need the env var and a copy of the JSON.
2. BR dev `Model.with_values` accepts only `joint_placements` / `body_inertias` / `frame_placements` — no `gravity` — so the gravity override keeps `dataclasses.replace(model, values=dataclasses.replace(model.values, gravity=g6))` (kindyn.py, forces.py), verified to validate and run.
3. Left for later steps (not step-2 scope): `scripts/label_amass.py:64 LOCKED_HEAD` and `scripts/import_bedlam2.py:124` absolute main-BH paths (dataset code → step 4; could use `tools.body.models_dir()`); `convert_mhr_to_smplx.py:86 _DEFAULT_SMPLX_MODELS = ../BetterHuman/models` (relative, works in both layouts); CLAUDE.md's `smplx_robot/` layout line and the `mhr_1_sup`/`kindyn_manual` viewer paragraph (docs → step 6); `tools/optim.py` untouched (step 3).

## 4. Regression on the 4 clips (GPU 2, deterministic wrapper; filled in by the main session 2026-10-09 01:05 Dubai)
Driver `artifacts/dev2/driver.sh` (pid 2179732, 00:33–00:53): chain C → `P/human_optim/`; same-input per-stage run I (fed main A's inputs) → `data_dev/iso/I/`; all 20+ stages rc=0. Dev is slower and uses more VRAM on kinematics_1 (boulder_1 86.5 s / 4874 MiB vs main 71 s / 4194 MiB; see `artifacts/dev2/timing.tsv`) — deterministic mode + BR dev FK; profiling is step 3's business.

Gate (a) iteration-0 loss lines: kinematics_1 (chain) IDENTICAL on all 4 clips. kinematics_2 (same input, run I): every loss term identical on all clips except one 3DAg `[full]` total 2.9930→2.9931; the only differing tokens are the trailing info fields (`cm_s`/`deg_s` at the 4th significant digit, e.g. deg_s 52.24→52.26). kindyn_1 warm-start "base wrench before" IDENTICAL on all 4 clips (639.7 / 644.9 / 1762.9 / 659.1 N).

Gate (b) same input per stage, dev I vs main A (`gate_table.py noise_A_vs_I.json`), worst clip vs tolerance: k2 pelvis mean 1.87/10 mm, p99 8.7/60; k2 joint 0.36°/2.2, p99 2.25/11; kindyn joint pos mean 1.13/22 mm, p99 4.05/107; force median 0.17/3.8 %BW; base-wrench Δ 0.8/16 %; contact labels exact (agreement 1.000 both tags, all clips). PASS.

Gate (c) whole chain, dev C vs main A (`noise_A_vs_C.json`), worst clip: k1 pelvis 0.43/1.7 mm (max 3.9/20), joint 0.043°/0.13 (p99 0.37/1.2); k2 pelvis 3.03/10, p99 17.1/60; k2 joint 0.65/2.2, p99 3.0/11; kindyn pos 5.1/22, p99 20.8/107; force med 1.17/3.8; bw 5.3/16; contacts agreement .993/.972 and .989/.957. PASS — all inside the main A-vs-B noise band (report_0 §4).
| clip | k1 pel mean/max mm | k1 joint mean/p99 ° | k2 pel mean/p99 mm | k2 joint mean/p99 ° | kd jpos mean/p99 mm | kd force med %BW | kd bw-after Δ% | c1 agree (F1) | c2 agree (F1) |
|---|---|---|---|---|---|---|---|---|---|
| boulder_1 | 0.23/2.4 | 0.014/0.20 | 2.62/10.5 | 0.65/2.3 | 5.10/20.8 | 1.17 | 0.7 | 0.9983 (0.996) | 0.9921 (0.985) |
| gtIGgs_y2nY_0024 | 0.34/3.5 | 0.043/0.36 | 2.29/10.4 | 0.57/2.5 | 3.14/15.5 | 0.64 | 0.5 | 0.9994 (0.999) | 0.9889 (0.982) |
| 3DAgLpPXVCs_0132 | 0.43/3.9 | 0.035/0.37 | 3.03/17.1 | 0.59/3.0 | 4.13/18.4 | 1.10 | 3.0 | 0.9991 (0.997) | 0.9953 (0.989) |
| 9RvrpYoWDoo_0001 | 0.26/1.1 | 0.030/0.28 | 2.05/13.0 | 0.64/3.0 | 3.97/13.6 | 0.88 | 5.3 | 0.9934 (0.984) | 0.9937 (0.986) |
Pooled contact F1 vs manual annotations (`score_C_contacts_{1,2}.txt`): contacts_1 0.912 (main A 0.912), contacts_2 0.931 (main A 0.932). PASS (±0.01).

Verdict: step 2 passes the report_0 gate. Full pytest on GPU 2: see the commit message / STATE.md.
