# Report 5 — BVR viewer rebuilt on BetterRobot dev + BetterHuman dev layers (mars_top_opus, 2026-10-09; saved by the main session)

**Result:** the viewer is now a thin adapter (`ClipData` → BR/BH layers) plus a small app. All 4 regression clips open on a fake server with every body source present and no exception. A real viser session on boulder_1 (in-process, same `App` code path) was screenshotted with `Viewer.record`. Viewer tests: 19 passed.

**Not done by the agent (context limit):** the `scripts/view_3d.py` process itself was never launched on a live port; the `--bedlam` loader was never run on a real BEDLAM2 scene. (Main session: see the addendum at the end.)

## Line counts (tools/viewer/*.py + scripts/view_3d.py)
| | total | breakdown |
|---|---|---|
| Before | 3,050 lines + 162 in tools/geometry/surface.py | app 551, helpers 197, `__init__` 19, loading 1,005, scene 990, view_3d 288 |
| After | 1,297 lines | app 225, layers 246, loading 479, datasets_layers 249, `__init__` 15, view_3d 83 |
Without datasets_layers the core is 1,048 lines (target ≈1,100).

## Per file
**`tools/viewer/loading.py`** (rewritten; numpy only, no viser)
- Dataclasses `BodySource` (world-frame `q (P,T,211)`, betas, valid, object_ids, contact `(P,T,F)`, forces) and `ClipData` (scene, human points, extr/fov, gravity, bodies, rig views, `videos`, sensors, plates).
- `body_source` reads a world-frame body npz. `classic_source` reads camera-frame classic SMPL-X: one `Body.from_classic` per person with a single betas vector, root composed with the camera-to-world pose; matches the stage's `fold_root_to_world` (per-frame betas) to 6e-7 m, 35× faster.
- `contact_labels` checks schema 2 and expands a frame subset by name.
- `load_forces` folds finger forces onto the wrists, orders frame forces in body-frame order, torque colour reference = 95th percentile, returns `None` for a kinematic npz (without this guard the loader crashed on `kinematic_1.npz` — found by the new test).
- `VideoFrames`: lazy cv2 decoder for `ImagePanel` (sequential reads, seeks on jumps, keeps decoded frames; seek 78 ms boulder_1 / 760 ms gtIGgs; sequential 4–13 ms/frame).
- `_hold_valid` copies a valid pose into invalid frames (all-zero quaternions give NaN in FK).
- Also: `load_gravity`, `_load_view_cameras`, `_fov_y_rad`, `camera_poses`.

**`tools/viewer/layers.py`** (new; replaces scene.py): `clip_layers(data, body, …)` → list of `Item(layer, group, kind, source, valid)`; `KINDS` = 15 checkbox kinds; FK once per source and person on CPU (~60 ms per 445 frames).

**`tools/viewer/datasets_layers.py`** (new): BEDLAM2 (`load_bedlam_data`, `_bedlam_gravity` now takes pelvis tracks, `BEDLAM_BODY`); `load_sensor_forces` + `sensor_layer` (limb→joint map; plate arrows start at the CoP); `load_plate_estimate` + `plate_layer`; scene-DB symlink farm `run_scene_db`, `_link_scene_farm` (moved from view_3d.py).

**`tools/viewer/app.py`** (rewritten): `App` uses only `server.gui` widgets except camera framing (BR gap 1). Sidebar: clip pager (dropdown, prev/random/next, "scene i/N"); BR Timeline folder; video folder (view dropdown for rigs); body-source dropdown; layer-kind checkboxes (hidden when empty); info panel. Clip switch = `remove_group` + add (timeline shrinks) + `set_fps(data.fps)`. A layer shows when its kind is ticked and its source is active or it has no source. `opening_view` + `on_client_connect` aim the camera at the person. `view_3d()` creates `Viewer(host="0.0.0.0", theme=replace(LIGHT, up=up, control_width="large"))`, body on CPU.

**`scripts/view_3d.py`**: thin CLI (target, `--scenes`, `--bedlam`, `--config`). **`tools/viewer/__init__.py`**: docstring + export.

**Trashed** (→ /home/rikhat.akizhanov/better/BetterVideoReconstruction/trash/): tools/viewer/scene.py, tools/viewer/helpers.py, tools/geometry/surface.py (viewer was its only importer), tests/test_viewer_camera_pose.py, tests/test_viewer_groups.py.

**Tests**: test_viewer_bedlam.py / test_viewer_plate_forces.py imports moved. New test_viewer_app.py (2 tests, synthetic clip, own small fake server giving skinned meshes bones): zero-quaternion invalid frame hidden + empty human-points frame hidden; subset contact colours on the right frames; source switching + force checkboxes only for kindyn; pager shrinks the timeline.

## Layer mapping actually used
| BVR thing | BR/BH layer |
|---|---|
| Scene cloud | PointsLayer |
| Scene normals | ArrowsLayer coloured by direction |
| Human depth points | PointsLayer `(T,N,3)` padded with copies of the first point; `valid (T,)` hides empty frames |
| Reconstructed camera | one moving CameraLayer (median fov) |
| Camera path | ArrowsLayer `head_size=0` as polyline |
| Rig cameras | CameraLayer T=1 per view |
| Gravity | ArrowsLayer |
| Ground | GroundLayer(origin, normal, fill=0.25) |
| Body mesh | BH `skinned_layer(identity, state)`, opacity 0.7, `valid (T,)` |
| Skeleton | SkeletonLayer per-joint radius (fingers 0.3×); kindyn joints coloured by torque grey→amber |
| Contact frames | SpheresLayer red/green per step, ArrowsLayer links, ArrowsLayer normals |
| Contact forces | ArrowsLayer per joint (default on) and per frame (default off) |
| Root wrench | one ArrowsLayer with 2 arrows (force green, torque magenta) rotated by the root quaternion |
| Sensor / plate forces | ArrowsLayer with element mask `(T,C)` |
| Video | `ImagePanel(VideoFrames, nsteps=T)` per rig view, toggled by the view dropdown |
Client skinning (top-4 weights) vs exact vertices: median 2.3 mm, max 2.7 cm — same truncation as the old viewer.

## Verification
(b) Fake server, 4 clips through `App`: every source active in turn, steps 0, T/3, T−1, 1 pushed with all kinds on; no exception. Data from a scratch farm whose `human_optim` → `human_optim_A` (the regression run was rewriting `human_optim`). Sources original/prefit/kinematics/kinematics_2/kindyn_1 on every clip; contacts on kinematics/kinematics_2/kindyn_1; forces on kindyn_1; 3DAg 2 people. 28 layers per clip (50 for 3DAg); timeline 363/104/445/471, fps 60/30/30/60.
Human-points upload (cap 6000 per climber; layer off by default): boulder_1 90 kB/step, 40.0 MB all frames (old preload 40.0 MB); 3DAg 180 kB/step, 65 MB (old 59 MB); ≈2.7 MB/s at 30 fps while on.
Load time boulder_1 CPU (load avg 88): 8–11 s total; old unchanged `_subsample_human_mask` 3.8–6 s; padding 0.24 s; layers 0.6 s.
(c) Screenshots (real viser + headless Chromium via `Viewer.record`, step 31): `/home/rikhat.akizhanov/better/dev/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_22-50_dev_optim_rewrite/vis/viewer_boulder_1.png` (overview) and `.../vis/viewer_boulder_1_forces.png` (kindyn_1 close-up: red force arrows at hands/feet/elbow/back/shoulder, 31–200 N; red/green contact spheres + links; torque-tinted joints; magenta root-torque arrow). Gravity arrow orange = boulder_1 uses `fallback_down` (ground fit rejected at 30.4°).
(d) `scripts/view_3d.py --help` exits 0; dataset modules import under pytest.

## BR gaps still hit, workaround
1. No camera framing on connect → App sets `client.camera` (only viser call outside `gui`). 2. `ImagePanel` defaults `listed=True` → App sets `listed=False`, adds it in its own folder. 3. No GUI order for BR's Timeline folder → App folders get `order=1e6+`. 4. Missing in BR: `(T,N)` mask on PointsLayer (padding), CameraLayer labels, per-bone radius on SkeletonLayer, live point-size/opacity setters (sliders dropped).

## Retired
Onboard camera; surface-mesh path (bpa/poisson + surface.py); athletics sport filter + `--scenes athletics`; point-size/camera-size/opacity sliders; torque-size + fingers toggles (torque is a colour); force-display dropdown (two checkbox kinds); legacy single-person promote shim; `runtime.device` (viewer poses on CPU); `camera_stride` (one moving frustum).

## Assumptions (reversible)
Server binds 0.0.0.0 as before (BR default loopback); every source draws a skeleton (before only kinematic sources); mesh opacity fixed 0.7; per-joint force arrows default, per-frame opt-in; video frame indices from `transform.npz` (= `vggt/camera.npz` on all 4 clips).

## Open questions / next steps
1. Run the real `scripts/view_3d.py` process on a free port and curl it. 2. Run `--bedlam` on one real scene from ../data/BEDLAM2_our. 3. Vectorise `_subsample_human_mask` (main load cost, old code)? 4. CLAUDE.md/README still describe the old viewer (step 6). 5. `[viewer]` config sections still carry `camera_stride` + a device comment — unused; clean in step 4.

Scratch: scratchpad `verify_fake.py`, `record_boulder.py`, `check_skin.py`, `check_fold.py`, `check_video.py`, `farm/`, `libasound/`; Chromium at `~/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell`.
