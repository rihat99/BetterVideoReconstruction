# PLAN — BVR dev: BR-dev optim, residual-only losses, dataset code out of core (2026-10-08)

Worktree: /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction (branch dev from main 4e2c152). Main checkout + main venv untouched. GPU 2 only.
Investigations: subagents/inv_A_br_optim.md (BR optim API, parity probe bit-exact), inv_B_bh_env.md (BH API delta, env), inv_C_bvr_map.md (generic vs dataset map, loss inventory, baselines), inv_D_viewer.md (BR viewer vs BVR viewer).

## Decisions (user, 2026-10-08 22:55–23:55 Dubai)
1. Dataset/benchmark code → `datasets/<name>/` inside BVR; `datasets/methods/` for cross-dataset baseline adapters.
2. "Close to main" = the example clips; regression set boulder_1, gtIGgs_y2nY_0024, 3DAgLpPXVCs_0132, 9RvrpYoWDoo_0001 (+ pytest). Per-clip tolerances from inv_C §5, calibrated on two main-vs-main runs.
3. Aggressive removal.
4. Own venv (Option A now; Option C — pyproject path deps — once the migration is stable).
5. Strict solver parity first: all five solves (kinematics+prefit, kindyn warm start, force witness, converter, multi-view fit) on BR optim via TorchOptim + scalar terms. LM/manifold upgrades only later, A/B'd.
6. Generic gaps go into BetterRobot dev (optim per-term diagnostics; viewer per-step valid mask, timeline reset on remove_group, ImagePanel, per-step sphere/arrow colours, tilted filled ground, settable fps).
7. Retire: kindyn_manual, mhr_1_sup + MHR viewer path, per-joint manual-label branch (contact_axis=joints, ann_repulsion, labels=manual), all-zero terms (velocity_*, ann_seating, ann_acceleration), [stages.kinematics_1].labels, corpus [score]; Athletics paths; score_test_contacts.py, render_sup_force_overlays.py, recompute_contact_confidence.py; docs/contact_fusion.md + contact_points.md → agent_notes; onboard camera; tools/smplx_robot (finite-diff + geodesic weights move into the human-optim package).
8. Layout: `bvr/` installable package (problems/, residuals, contacts, geometry, body, viewer, stages); pipeline.py + run_dataset.py → one clip-root runner on configs/datasets.toml; corpus.py → datasets/climbingvideos.
9. Configs: base + overrides loader; datasets keep only their delta; one loader path.
10. Viewer rebuilt on BR/BH dev layers; visual changes accepted.
11. Dev runs never write into main's trees: BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data_dev (scratch copy of the regression clips' upstream outputs).

## Steps (one commit each on dev; agents do not commit unless the brief says so)
0. Reference on MAIN (independent): scratch root data_dev with the 4 clips; run main's human chain twice (A, B) → noise floor; `compare_runs.py` tool; GPU memory measured. → verify: artifacts + noise table in subagents/report_0_reference.md.
1. Dev env (independent): Option A. → verify: imports resolve to dev/BetterRobot, dev/BetterHuman; pytest on dev lists exactly the API-break failures.
G. BR dev gaps (independent, other repo, commits on dev/BetterRobot): → verify: BR tests pass incl. new ones.
2. BH + BR API migration on dev (no behaviour change; body.py, kindyn/forces FK+RNEA+Jacobians, bedlam_contacts, label_amass, viewer MHR path removed with mhr_1_sup, smplx_robot dissolved). → verify: pytest green; regression set vs main run A within noise floor.
3. Optim port: tools/optim.py removed; bvr/problems/* define variables + cached state + scalar terms; BR TorchOptim factories; per-term logging via BR diagnostics; kindyn warm start as a second Problem; witness via bounds; converter + multi-view fit ported. → verify: regression within noise floor; per-term losses side by side; test_optim_engine removed (BR covers).
4. Structure: bvr/ package, datasets/<name>/ split, retire list, base+override configs, single runner, corpus → datasets/climbingvideos; tests and docs move with their code. → verify: pytest, --help/dry-run of every entry point, regression again, dataset dry-runs.
5. Viewer on BR/BH layers (can overlap with 3–4; touches only the viewer). → verify: opens the 4 clips headless (fake server) + one real viser session screenshot; dataset layers in datasets/.
6. Option C env, docs (CLAUDE.md/README/INSTALL for dev), adversarial review, REPORT.md.

## Assumptions (made off-hours, 2026-10-09 ~01:00 Dubai)
- Main's chain is nondeterministic (report_0 §4–5). Regression gate = report_0's recommendation: exact iteration-0 loss terms; per-stage same-input comparison within ×3 same-input noise; whole-chain pooled contact F1 ±0.01 plus ×3 A/B noise. Where ×3 noise exceeds inv_C's caps, the noise wins.
- Dev gets an opt-in deterministic switch (env var, one call per stage entry) so dev-vs-dev comparisons are exact; main is compared through the wrapper when needed.
