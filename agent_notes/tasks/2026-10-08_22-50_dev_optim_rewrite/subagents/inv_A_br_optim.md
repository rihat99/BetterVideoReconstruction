# Investigation A — BetterRobot dev optim and the API delta BVR will hit (mars_aux, read-only, 2026-10-08)

## Verdict
- BR dev `better_robot.optim` can carry every BVR fit today with a thin BVR adapter (~50 lines), no BR change required for parity.
- The port can be bit-identical: a CPU probe ran a BVR-shaped fit (masked Adam root phase + MultiStepLR, Adam, one strong-Wolfe L-BFGS call) through BVR `tools/optim.py` and through BR dev optim; final q differed by 0.0; the force witness's projected Adam matched to 0.0. Probe: scratchpad `probe_equiv.py`.
- Parity needs: B=1 one person per call; terms registered as `scalar` with weight 0.0; every phase on `TorchOptim` incl. `torch.optim.LBFGS` (BR's own `LBFGS` gave max|Δq| 7e-4); Euclidean variables, no manifolds/bounds on q.
- Missing in BR: per-term logging from the same evaluation (BVR terms return `(value, info)`; BR terms return a tensor); "column" weight semantics (BVR: unlisted term = 0; BR: registration weight, KeyError on unknown names).
- Big API break: BR dev removed the mutable `Data` API (commit 651a156). Every `forward_kinematics` / `update_frame_placements` / `compute_joint_jacobians` / `get_frame_jacobian` / `rnea` / `dynamics.centroidal` call breaks (§4).

## 1. BR dev optim public API (src/better_robot/optim/)
| Piece | Where | Notes |
|---|---|---|
| `Problem(variables, manifolds=, bounds=, data=, time=, dense=)` | problem.py:142-178 | named tensors `(B, *event)`, shared B; `time=` marks `(B,T,…)` |
| `cached(fn, name=)` | problem.py:182-191 | lazy, once per evaluation, any object |
| `residual(name, fn, weight=, kernel=, jac=, window=)` | problem.py:193-221 | loss `½·w·Σr²` (458-461) |
| `scalar(name, fn, weight=)` | problem.py:223-236 | loss `w·fn(...)`, `(B,)`; 0-dim broadcasts at B=1 |
| args resolved by name | problem.py:408-420 | `lambda st:` → cached value `st` |
| `with_(weights=, free=)` | problem.py:238-261 | unknown names KeyError; free masks bool in tangent coords; one-knot mask repeats over time only for `time=` vars |
| active iff `weights.get(name, term.weight) != 0` | problem.py:429-435 | |
| Solvers: LM 70-125, GaussNewton 128-159, `LBFGS` 170-251 (batched, Armijo), `TorchOptim` 254-321 | solvers.py | TorchOptim: new optimizer per phase init (275-282); grads masked in closure (288-295); scheduler steps per step (298-299); projection onto manifolds/bounds after each step (300-302); no done flag |
| `Phase(solver, iters, name, weights, free, on_start, on_step, stop_every=1)`, `run` | phases.py:14-34,55-104 | on_start runs even at iters=0 (78-79); each phase gets its own copy of data; stop_every=1 = host sync per step (94); run does one extra no-grad forward at the end (101-102) |
| `Result` | phases.py:37-52 | x, loss, done, history, diagnostics, state (warm start) |
| `solve_layer`, `solve_constrained` | phases.py:119-151; constrained.py:28-96 | solve_layer implicit mode only |
| Kernels (Huber/Cauchy/Tukey/GemanMcClure), manifolds | kernels.py; manifold.py:37-52 | robot terms in `better_robot.terms` |
Examples: examples/03_clip_fit_phases.py:42-67; tests/optim/test_problem.py:300-324.
Status: p1 REPORT accurate; `optim2` renamed to `optim`, old packages deleted (cb58627). Done: dense/banded/arrow normal equations, float64 linear solves, warm start, diagnostics. Not done: solve_layer unroll mode + stationarity check; the acceptance test "BVR person fit ported reproduces its losses" is open.

## 2. BVR engine today and every feature relied on
| Feature | Where |
|---|---|
| Leaves via `add_leaf`, captured by closures | optim.py:66-70; q read at optimizer.py:668,752,774,974,1023; f_free/gen_u kindyn.py:283 |
| One forward → lazy `State` (put/put_lazy) | optim.py:27-50; optimizer.py:666-721 (qn/jw/jwq/vw/fw/fwq + lazy vismask/sig/sig_f/inst_nn/d1/accel/jerk/snap); kindyn.py:277-287 |
| Terms return scalar or `(scalar, info)` | optim.py:53-55,90-91; optimizer.py:855,877,889,907,918,932,942; kindyn.py:298-306 |
| Weight column per phase, unlisted=0, error if nothing active; sum in registration order | optim.py:86-95 |
| Gradient masks float 0/1 for Adam + L-BFGS | optim.py:118-125; `_dof_mask` optimizer.py:101-120; finger mask 990; betas mask 1002-1003; prefit log_s frozen prefit.py:298-300 |
| Fresh Adam per phase; MultiStepLR at iters//2, 3·iters//4, gamma=lr_decay | optim.py:180-191; `[optimizer] lr=0.01 lr_decay=0.1`; prefit lr 0.02 no decay (estimate_scale.py:605-606) |
| L-BFGS = one step, max_iter=iters, strong-Wolfe, lr 1, history 10 | optim.py:161-177 |
| Iterations | root 150; kin1 300/200/30; kin2 100/200/250/0; kindyn L-BFGS 150; prefit 100×3; kindyn warm start 200; force witness 2000 |
| on_start even at iters=0 (closure cells: scene candidates, frozen labels, patches, betas live) | optim.py:155-160; optimizer.py:959-986 |
| Logging cadence: quarter of Adam, every 5th L-BFGS closure | optim.py:128-134,172,192 |
| Early stop: none | |
| Warm starts outside engine: kindyn forces (own strong-Wolfe L-BFGS duplicating the term math) | optimizer.py:1013-1018; kindyn.py:409-414, 422-449 vs 289-349 |
| fit_betas: leaf trained only in phases naming betas | optimizer.py:493-502,726-727,961 |
| Batching over people: none (sequential loops) | stages.py:503; kindyn.py:836; estimate_scale.py:425-438; seeded jitter stages.py:472/533 |
| Force witness: projected Adam lr 0.05, ExponentialLR to 1e-3, clamp ≥0 | forces.py:264-276 |
| Other private Adam: mhr_to_smplx converter.py:211-260 (4 stages; edge_weight switch at epoch 50; transl re-created mid-fit) | |

## 3. Mapping BVR → BR dev
| BVR feature | BR equivalent | Status | Gap owner |
|---|---|---|---|
| Leaves | `Problem({...})` `(1,…)` | exists | BVR B=1 wrapper; closures read variables by name (TorchOptim clones per phase) |
| Lazy State | `cached("st")` returning BVR State; terms `lambda st:` verbatim | exists | — |
| Scalar terms | `scalar(name, fn, weight=0.0)` | exists | — |
| `(value, info)` | none | missing | BR generic (TorchOptim keeps last closure's per-term values/info in State.extra); short term BVR cell |
| Column weights, unlisted=0 | Phase.weights | partial | BVR: register all at 0, filter column to problem.terms |
| Gradient masks | Phase.free bool | exists | `.bool()`, `time=("q",)`, masks on variable device |
| Fresh Adam + MultiStepLR | `TorchOptim(Adam, lr, scheduler=…)` | exists | 3-line factory |
| Strong-Wolfe L-BFGS one call | `Phase(TorchOptim(torch.optim.LBFGS, max_iter=n, history_size=h, line_search_fn="strong_wolfe"), iters=1)` | exists | not BR LBFGS |
| on_start + cells | Phase.on_start(view, x) | exists | data writes don't persist across phases |
| Per-term logging cadence | on_step + view.losses(x) | partial | extra forward; closure-level L-BFGS logging missing |
| Early stop | none needed | — | stop_every=0 |
| Force warm start | second Problem over forces (q frozen) reusing term fns + affine cached dyn; run(x=) | exists | removes duplicated closure |
| fit_betas | variable + per-phase free + cell toggle | exists | — |
| Batching over people | B axis | n/a | keep loop (per-person scalars; torch LBFGS couples batch) |
| Projected Adam | `bounds={"lam": (0, inf)}` + TorchOptim(Adam, ExponentialLR) | bit-identical in probe | optional later LM with bounds (changes results) |
| Prefit log_s | variable (1,1) + free False | exists | — |
| Converter mid-phase switch | none | partial | stays BVR or on_step cell |
Size: remove tools/optim.py (193) + tests/test_optim_engine.py (131; BR tests/optim has 28 tests); rewrite ~140 glue lines in optimizer.py (net −10..−30); kindyn warm start −20; forces loop −5; adapter +50. Net ≈ −250..−300 lines; ~850 lines of optimizer.py and all of losses.py (672) are problem definition and stay.

## 4. BR API delta (every better_robot use in BVR)
| BVR file:line | Symbol | BR main | BR dev | Change |
|---|---|---|---|---|
| tools/body.py:28,262-263 | `update_frame_placements(model, data)`; `data.frame_pose_world` | kinematics/forward.py:285 | removed (no Data) | `state.frame_pose[..., ids, :]` (kinematics/state.py:141); data comes from BH `shaped.fk()` → coupled to BH dev |
| tools/body.py:259 | `data.joint_pose_world` | Data field | `KinematicState.joint_pose` (state.py:134) | rename; quaternion signs may differ (converted from 3×4) |
| kindyn.py:214-218 | `forward_kinematics` + `update_frame_placements` | forward.py:167 | `forward_kinematics_state(model, q)` (state.py:152) / `model.fk(q)` | state.joint_pose / frame_pose |
| kindyn.py:232 | `so3.to_matrix(oMi[...,3:7])` | exists | unchanged | optional state.joint_rotation |
| kindyn.py:250,256 | `rnea(model, qn, v, a, fext=)` | dynamics/rnea.py:267 | `inverse_dynamics(model, state, *, v, a, fext)` (dynamics/state.py:55) | new call; fext `(…, njoints, 6)` joint frame unchanged (_world.py:226) |
| kindyn.py:368-378 | rnea, `compute_joint_jacobians`, `get_frame_jacobian` | jacobian.py:299,361 | `frame_jacobian(model, state, *, frame=fid)` (state.py:185; local_world_aligned default in both) | kinematics.jacobian no longer exports these |
| forces.py:128-162 | same | — | — | same rewrite |
| bedlam_contacts.py:57,132-133 | jacobians on `posed.data` | — | `frame_jacobian(posed.model, state, frame=…)` | `_Posed` (body.py:103,267) must carry a KinematicState |
| label_amass.py:159,166 | `dynamics.centroidal.center_of_mass(model, qn)` | centroidal.py:147 | `better_robot.dynamics.center_of_mass(model, state)` (state.py:93) | module removed |
| kindyn.py:617-628,662-675; forces.py:139-146 | `dataclasses.replace(model.values, gravity=)`, body_inertias, body_name_to_id, idx_vs, nvs, njoints | exist | exist (model.py:76-160; model_values.py:61,71) | likely none; replace vs dev validation not run |
| smplx_robot/dynamics.py:43 | `model.difference` | exists | exists | none; comment :33 cites removed better_robot.residuals |
| motion.py, camera.py, viewer/loading.py:720-750, tests/test_contact_frames.py | se3/so3 log/compose/inverse/normalize/from_matrix/to_matrix | exist | exist | none; loading.py:745-746 also uses BH native_joint_poses / inverse_bind_poses |

## 5. Numerics to match
- `TorchOptim(torch.optim.LBFGS)` not BR LBFGS (Armijo, 1/‖g‖₁ first step, no Wolfe; toy 7e-4).
- `scalar` not `residual` (½·Σ + IRLS kernels change scale).
- Register weight 0.0 and filter columns. Euclidean q, no bounds on q (TorchOptim projects after each step). B=1; variable order q, betas, f_free, gen_u (L-BFGS flattening). MultiStepLR milestones/gamma, fresh optimizer per phase, iteration counts. float32. stop_every=0 (perf).
- BR dev FK/RNEA rewritten (matrix FK lean_core, quaternions from matrices, world-frame RNEA): expect float32 rounding differences amplified over ~700 Adam iterations; quaternion signs harmless where checked (so3.to_matrix, dot² in losses.py:538). Needs per-clip A/B vs main + run-twice-on-main noise floor (GPU scatter nondeterministic).
- BH dev `terms.root_wrench` uses a different v/a stencil than BVR's `velocity_acceleration_from_trajectory` (no spacing, no dyn_row gating): do not swap without A/B. `contact_wrenches` = build_fext up to rounding.
- Force witness on LM with bounds: exact optimum vs Adam's → magnitudes shift, labels near hysteresis may flip; keep projected Adam for parity.

## Open questions
1. Per-term diagnostics into BR (generic) or a BVR cell?
2. mhr_to_smplx converter's private Adam and the kindyn warm-start L-BFGS onto BR optim too, or out of scope?
3. Strict parity first cut, or allow A/B'd upgrades (LM force witness, LM warm start)?
4. Env: dev BVR .venv empty; hard-coded `../better_human/...` (estimate_scale.py:58; stages.py:42 → non-existent dev/better_human) and absolute main-BH model path (tools/body.py:36-38).
5. Not verified: no GPU run; no real-clip A/B; `dataclasses.replace` on dev ModelValues.
