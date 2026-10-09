# REPORT — BVR dev: BetterRobot optim, residual-only losses, dataset code out of the core (2026-10-08 22:50 → 2026-10-09)

Branch `dev` of BetterVideoReconstruction (BVR), worktree `/home/rikhat.akizhanov/better/dev/BetterVideoReconstruction`, on top of main 4e2c152. Nothing merged; main untouched. Sibling dev branches: BetterRobot (BR) `/home/rikhat.akizhanov/better/dev/BetterRobot`, BetterHuman (BH) `/home/rikhat.akizhanov/better/dev/BetterHuman`.

## What was asked
Move BVR dev onto the BR dev + BH dev APIs; BVR stops owning an optimization engine and only defines problems and losses on BR's rewritten `optim`; clean out everything unnecessary; move all dataset/benchmark code out of the generic core; results must stay very close to main; rebuild the viewer on BR's viewer tools. Decisions taken at intake and during the task: `PLAN.md` (12 decisions).

## Result in one paragraph
Eight code commits on `dev` (list below). BVR no longer has an optimizer: every solve (kinematics root/kin1/kin2, kindyn with its force warm start, force witness, scale pre-fit, MHR→SMPL-X converter, multi-view fit) is a BR `Problem` + `Phase` list run by BR `TorchOptim`, with the TOML weight columns as the objective; `tools/optim.py` is gone. The body is the BH dev `Body`/`HumanState` API; kinematics and dynamics are BR dev (`forward_kinematics_state`, `inverse_dynamics`, `frame_jacobian`, `center_of_mass`). The viewer is a 1,400-line adapter on BR/BH layers (was 3,200). The code is an installable `bvr` package plus `datasets/<name>/` for every dataset, configs are deltas over a base, one runner replaces two. On the four regression clips the human chain reproduces main inside main's own run-to-run noise, and from step 3 on every stage is bit-identical to the previous step under the new `BVR_DETERMINISTIC=1` switch. pytest 229 passed, 4 skipped (main: 250; the difference is removed engine tests covered by BR, retired features and dead helpers).

## Commits on dev (oldest first)
| commit | step | content | checks |
|---|---|---|---|
| 46ebb69 | 2 | BH dev + BR dev API migration (body.py, kindyn/forces/bedlam/amass; tools/smplx_robot dissolved; MHR viewer path + kindyn_manual removed) | pytest 246; gate (report_2) |
| fffc58a | 3 | all five solves on BR optim; tools/optim.py removed; `tools/determinism.py` (`BVR_DETERMINISTIC=1`) | converter bit-identical; kin/contacts bit-identical to step 2; gate (report_3) |
| 9796bba | 5 | viewer on BR/BH layers (+ BR dev solid arrows 8828842, review fixes) | 236 tests; numeric checks vs old viewer 1e-6 m (report_5r) |
| e2e7bb7 | 4a | `tools/` → `bvr/` package; stages as `python -m bvr.stages.<name>`; retire list | 234 tests; kin_1 bit-identical |
| d282814 | 4b | `datasets/<name>/` split (examples, climbingvideos, peter, opencap, parkour, bedlam2, amass, methods) with READMEs | dry-runs identical; 48/48 --help |
| 28f023e | 4c | base+override configs (deltas 12–42 lines instead of 145–528), `bvr/runner.py` + one `scripts/pipeline.py` | merge proven identical leaf by leaf; 25 dry-run cases identical; full chain bit-identical; F1 0.912/0.931 |
| ba64306 | 6 | pyproject path deps on dev BR/BH (`uv sync` rebuilds the env), rig-neutral names, docs rewritten | 229 tests; 57/57 --help |
| (step 7) | 7 | review fixes: runner resume semantics restored for `--source`, config loader warnings, stale text, orphans | see §5 |
BR dev commits used: 72a7d02 (per-term diagnostics), 0c3d167, b6602aa, 64d3fb5, 2dd0a9d, 1f3fbe3, 4b7b1da (viewer gaps), 8828842 (solid arrows). BR suite 1674 passed.

## 1. Baseline and regression gate (subagents/report_0_reference.md)
Baseline = main 4e2c152 with main BR/BH, run twice (A, B) on the four example clips boulder_1, gtIGgs_y2nY_0024, 3DAgLpPXVCs_0132, 9RvrpYoWDoo_0001 (scratch root `/home/rikhat.akizhanov/better/data_dev/examples/out/<clip>/human_optim_{A,B}`). Main is NOT reproducible run to run (GPU scatter): A vs B differs by 0.2–0.6 mm mean pelvis after kinematics_1, 2–3.4 mm after kinematics_2, 3–7 mm joint positions after kindyn_1, forces median 0.7–1.3 %BW; contact labels agree 0.986–0.9999. Gate: (a) iteration-0 loss terms identical; (b) each stage fed main's inputs within ×3 of main's same-input noise; (c) whole chain within ×3 of A-vs-B noise and pooled contact F1 ±0.01.

## 2. Results vs main (dev step 3 run C3; report_3 §6)
Whole chain, dev vs main A, worst clip (tolerance): kin_1 pelvis 0.43 mm (1.7); kin_2 pelvis 3.0 mm mean / 17 mm p99 (10/60); kindyn joints 4.7 / 20 mm (22/107); force median 1.10 %BW (3.8); base-wrench Δ 6.9 % (16); contact agreement .993/.989 (.972/.957). Same-input per stage: contact labels exact; kin_2 pelvis ≤1.9 mm; kindyn joints ≤1.4 mm; forces median ≤0.37 %BW. Iteration-0 loss terms identical on every clip and stage. Pooled contact F1 vs manual annotations: contacts_1 0.912 (main 0.912), contacts_2 0.931 (main 0.932). Every later step (4a–6) is bit-identical to this run on every numeric key under `BVR_DETERMINISTIC=1`.
The port's own effect (step 2 engine vs BR optim, identical inputs): kinematics and contacts bit-identical; kindyn_1 0.6–1.2 mm mean joint position (strong-Wolfe L-BFGS amplifies last-bit rounding of the warm start's weight order).
Timing: kindyn_1 faster (one FK per evaluation); kinematics_1 +20–30 % on the long clips (87/70/102 s vs 71/57/77 s) and +0.2–0.7 GB VRAM — from BR dev's FK + deterministic mode, not the port; not profiled (open item).

## 3. What a user sees now
- `uv sync --managed-python` + 3 manual steps (INSTALL.md); `uv run python scripts/pipeline.py --source <video> --save-dir <out>` or `--dataset <name>`; `python -m bvr.stages.<name>` for one stage; `scripts/view_3d.py <out/clip>`.
- Layout: `bvr/` (body, config, paths, runner, determinism, geometry/, human_optim/ (problems: optimizer.py, kindyn.py, prefit.py, forces.py, losses.py, solve.py), mhr_to_smplx/, stages/, viewer/, adapters/), `datasets/<name>/{README.md, configs/, *.py, tests/}`, `scripts/{pipeline.py, view_3d.py, scene_quality.py}`, `configs/{pipeline.toml, human_optimization.toml}`.
- Configs: `base = "..."` + delta; dataset configs 12–42 lines.
- Viewer: BR Viewer + Timeline; body-source dropdown, 15 layer kinds, video pane (ImagePanel), solid force arrows as before; screenshots `vis/viewer_boulder_1.png`, `vis/viewer_boulder_1_forces.png`.
- Retired (decision 7): per-joint manual-label path, zero-weight terms, kindyn_manual, mhr_1_sup, Athletics, onboard camera, surface mesh, 3 scripts, 2 docs (→ agent_notes/docs/retired/), tools/smplx_robot, tools/optim.py. Trashed copies: `/home/rikhat.akizhanov/better/BetterVideoReconstruction/trash/` (the helper routes a worktree's trash to the main checkout).

## 4. Known differences and open items
- BEDLAM2 `label_bedlam2 --step forces` (kindyn warm start) main vs dev on CPU: max 15–21 N on forces up to 390 N, mean 0.04–0.07 N, base-wrench RMS 26.28→26.27 / 35.57→35.48 N (review_R.md). Same class as the kindyn_1 chain differences; `FORCE_METHOD` not bumped.
- `--source` runs: resume semantics (restored in step 7, see §5).
- kinematics_1 slowdown vs main (above) — profile BH dev body load + HumanState memoisation.
- L-BFGS phases log one line per phase (BR's closure is private); a BR `TorchOptim(on_closure=)` hook would restore the every-5th-closure lines.
- Not exercised numerically on dev: fit_multiview `fit_betas` path, label_amass, OpenCap floorplane/build_support_scene (review part 2 — see §5).
- Leftovers for the user: `/tmp/bvr_view_*` symlink farms from viewer checks (trash refuses /tmp); scratchpad venv `s6/venv_c` (5 GB); `data_dev` (reference runs, ~2 GB) can go when the branch is accepted.

## 5. Review and fixes
(filled in after step 7 and review part 2)

## Pointers
Task folder: `/home/rikhat.akizhanov/better/dev/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_22-50_dev_optim_rewrite/` — PLAN.md, STATE.md, subagents/{inv_A..D, report_0..7, report_G, G2, 5r, 5b, briefs}, compare_runs.py, artifacts/{ref,dev2,dev3,dev4,dev6}/ (logs, gate tables, dry-run diffs, merge proofs), artifacts/review_R.md, vis/ (viewer screenshots). Env doc: `agent_notes/docs/dev_env.md`.
