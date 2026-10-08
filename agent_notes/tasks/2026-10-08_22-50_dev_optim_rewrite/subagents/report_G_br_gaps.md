# Report G — generic gaps added to BetterRobot dev (mars_top_opus, 2026-10-09)

All 7 gaps added on /home/rikhat.akizhanov/better/dev/BetterRobot branch `dev`, one commit per item, each with tests, a docs/concepts line and a changelog line. BR suite: 1652 → 1673 passed (11 skipped, 20 deselected), 0 failures. Nothing pushed. BR task notes: dev/BetterRobot/agent_notes/tasks/2026-10-08_23-41_bvr_gaps/{PLAN.md,REPORT.md}; notes commit 5caaa8e.

## Additions
1. Optim per-term diagnostics — `72a7d02`. A term may return `(value, info: dict)` (tensors in info detached, never in the loss; plain-tensor terms unchanged). New `Problem.evaluate(x) -> (loss (B,), terms {name: (B,)}, info {name: dict})`; `loss()`/`losses()` go through it. Every solver writes `State.extra["terms"]` for the point `State.loss` describes (LM/GN/LBFGS per batch element from the accepted step); `TorchOptim` also writes `extra["info"]` from its last closure call. `Phase.on_step` reads these with no second forward (tested). Values are weighted `(B,)` device tensors (raw = terms[k]/w); a logger must convert.
```python
problem.scalar("contact", lambda st: (loss, {"n": count}), weight=0.0)
log = lambda view, s: print({k: float(v[0]) for k, v in s.extra["terms"].items()}, s.extra["info"])
run(problem, [Phase(TorchOptim(torch.optim.Adam, lr=1e-2), iters=300, weights=column, on_step=log, stop_every=0)])
```
2a. Per-sample validity — `0c3d167`. `Viewer.add(layer, ..., valid=None)`: `(T,)` bool hides any layer (incl. BH layers) on invalid samples; `(T, N)` hides single elements of SpheresLayer/ArrowsLayer. Kept by reference (edit in place + `viewer.show()`); recording camera ignores invalid samples.
2b. Timeline shrink — `b6602aa`. `remove_group` shrinks the timeline to the longest remaining layer; new `Timeline.resize(nsteps)`.
2c. ImagePanel — `64d3fb5`. `ImagePanel(name, images, *, nsteps=None, format="jpeg")`, exported from `better_robot.viewer`; images `(T,H,W,3)`, one `(H,W,3)`, or lazy `step -> image` (needs nsteps). `viewer.add(ImagePanel("video", lambda k: decode(k), nsteps=n), group="video", times=t)`.
2d. Per-step colours — `2dd0a9d`. `SpheresLayer(..., colors=(T,S,3))`, `ArrowsLayer(..., colors=(T,N,3))`.
2e. Tilted filled ground — `1f3fbe3`. `GroundLayer(..., origin=(3,), normal=(3,), fill=0.0, fill_color=RGB)`; axis-aligned default unchanged.
2f. Per-clip fps — `4b7b1da`. `times=` alone resamples onto a fixed-rate timeline (one step ≠ one frame); setting `timeline.fps` by hand broke timed layers. Added `Viewer.set_fps(fps)` (validates, pauses/resumes, re-fits timeline, re-pushes step).

## Verification
New tests in tests/optim/test_solver_state.py, tests/viewer/test_time_pointer.py, tests/viewer/test_core.py. ruff clean on optim; viewer findings unchanged (11). `examples/viewer/02_play_trajectory.py --check` runs with the fake server and a real loopback viser. viser 1.0.26 accepts every new property. Headless-Chromium `record` PNGs show tilted filled ground, per-step colours, per-element and whole-layer masks. BetterHuman dev suite against this BR tree: 317 passed + 17 FileNotFoundError without `$BETTERHUMAN_MODELS_DIR`; tests/viz with it set: 72 passed.

## Skipped / limits
Only TorchOptim writes `info` (second-order solvers drop it); `TorchOptim.init` writes no terms (appear after first step); `LBFGS.init` does one extra no-grad forward per phase; `solve_constrained` terms must return a plain tensor; PointsLayer takes only a `(T,)` mask (pad instead); PlotPanel keeps its time axis after set_fps; no `Viewer.add_image` helper; no adversarial review run.

## Open questions
1. `solve_constrained` accepting `(value, info)`? (one line; BVR does not need it)
2. OK that `remove_group` always shrinks the timeline?
