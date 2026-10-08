# Investigation D — BetterRobot dev viewer vs the BVR viewer (mars_aux, read-only, 2026-10-08)

Verdict: BR dev has a general viser viewer (layers holding whole `(T, ...)` clips, named groups, one timeline, `.viser` save + WebM recording); BH dev adds human layers on it. Together they cover most of what BVR draws. Gaps are app-level: clip browser, body-source dropdown, video pane, onboard camera, a few per-step styling features. Rebuilding tools/viewer on BR/BH removes ~55–65% of its 2,871 lines. BR's architecture plan already expects this port (`dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md:189-191,237-238`: "`clip_layers(ClipData)` in BVR"); never done.

Paths: BR = dev/BetterRobot/src/better_robot/viewer/, BH = dev/BetterHuman/src/better_human/viz/, BVR = dev/BetterVideoReconstruction/.

## 1. BR dev viewer (2,343 lines, viser only)
- `Viewer` (viewer.py:33): `add(layer, group, offset=, times=)` :82, `remove_group` :139, `set_visible` :149, `show(k)` :160, `save(.viser)` :171, `record(webm/gif/png)` :183, `rebuild` :231, `play` :248, `on_pointer` :275. Every visible layer gets step k in one `server.atomic()` push (:313). Sidebar "Layers" folder with one checkbox per layer (:254-266). `times=` resamples a layer onto the timeline (:299).
- Timeline (timeline.py:11): Play/Pause, Step, Speed, Loop; wall-clock playback (:96); `extend` only grows (:62), never shrinks.
- `Layer` (layer.py:11): subclass with `build`, `set_step`, optional `build_gui`, `world_points`; `listed=False` hides from the folder.
- Built-in layers: PointsLayer (fixed N, per-step colours), ArrowsLayer ((T,N,3), zero vectors hidden, static colours), FramesLayer, SkeletonLayer (per-joint radius, scalar bone radius, per-step colours), MeshLayer/RigidMeshLayer/SpheresLayer (static sphere colours), SkinnedMeshLayer (browser-side skinning, top-4 weights), CameraLayer (one moving frustum with image or lazy callable), GroundLayer (axis-aligned plane at a height, no fill), PlotPanel via `add_plot`.
- Robot adapter `robot_layers`/`RobotView` (robot.py:28,112); torque shown as colour (:145). Tools: PoseGizmo, picking. Themes LIGHT/DARK, `Theme(up="-y")`.
- By design no GUI brick: `viewer.server.gui` is the API (docs/concepts/viewer.md:196). No dataset browsing, video pane, follow-camera.
- BH dev: `mesh_layer`, `skinned_layer`, `skeleton_layer`, `frames_layer`, `forces_layer`, `human_layers`, `HumanView` from a `HumanState` (human_layers.py:36-131). `skinned_layer` refuses rigs with joint scales (MHR) → `mesh_layer`.
- Example (examples/viewer/02_play_trajectory.py:72-84): `Viewer(fps=1/DT)`, `add(GroundLayer())`, `robot_layers(model, q, skeleton=True, tau=tau)`, `add_plot`, `play()`. Headless tests use tests/viewer/fake_server.py.

## 2–3. BVR viewer features → BR/BH (BVR: view_3d.py 292 + tools/viewer 2,871 lines)
| BVR feature (≈lines) | location | BR/BH equivalent | status → gap owner |
|---|---|---|---|
| Scene cloud + confidence filter (30) | scene.py:537-553; loading.py:978-988 | PointsLayer | exists |
| Point-size slider (15) | app.py:186,269-272 | fixed at build | missing → BR live setters or drop |
| Scene normals arrows (30) | scene.py:449-467 | ArrowsLayer | exists |
| Surface mesh bpa/poisson GLB (~110 + geometry/surface.py 162) | helpers.py:132-195; loading.py:1095-1114 | — | dead (app.py:89 forces "points") → delete |
| Trajectory frustums + spline + cursor (35) | scene.py:153-173 | CameraLayer (one moving) | partial → BVR thin layer / BR LinesLayer |
| Calibrated rig cameras labelled (35) | scene.py:140-152 | CameraLayer T=1, no labels | partial → BVR |
| Gravity arrow (14) | scene.py:176-189 | ArrowsLayer | exists |
| Ground: tilted plane, fill, cells (40) | scene.py:469-497 | axis-aligned, no fill | partial → BR origin/normal/fill (~15 lines) |
| Human depth points per frame (55) | scene.py:199-213; loading.py:805-823 | PointsLayer fixed N | partial → pad; bandwidth unmeasured |
| Skinned SMPL-X per person/source (~300) | loading.py:478-675 | SkinnedMeshLayer + BH skinned_layer | exists via BH |
| `mhr_1_sup` MHR mesh (90, hard-coded main-BH path loading.py:42-43) | loading.py:678-768 | BH mesh_layer | → ClimbingVideos dataset |
| Hide person on invalid/short frames (~15) | scene.py:638,687-692,748-751 | none (layers hold last step) | missing → BR per-step `valid` mask |
| 52-joint skeleton, small fingers (100) | scene.py:233-273 | SkeletonLayer | exists (finger bones not thinner) |
| Contact frames red/green + parent link + normal (215) | scene.py:275-305,712-791; loading.py:254-361 | frames_layer axes only; static sphere/arrow colours | partial → BR per-step colours |
| Mesh translucency + opacity slider (40) | scene.py:585-593; app.py:190 | fixed | missing → drop or BR |
| Torque skeleton size/colour (90) | scene.py:307-365 | per-step colours (BR convention) | partial → colour |
| Contact-force arrows (30) | scene.py:865-891 | ArrowsLayer / BH forces_layer | exists |
| Root wrench (30) | scene.py:892-912 | ArrowsLayer | exists |
| Sensor arrows at CoP/limb (60) | scene.py:913-940; loading.py:134-163 | ArrowsLayer | exists → datasets prep |
| Plate-format estimated forces (40) | scene.py:941-956; loading.py:166-185 | ArrowsLayer | → opencap |
| Playback/fps/slider (45) | app.py:151-158,431-437 | Timeline | exists |
| Per-clip fps | app.py:453-455 | fixed at Viewer(fps=) | partial → settable fps or times= |
| Video pane synced + multi-view lag (~200) | app.py:139-150,384-396; loading.py:213-251,364-475 | none (image in frustum only) | missing → BR ImagePanel (generic) |
| Onboard camera ride (~120) | app.py:289-364; helpers.py:34-80 | none | missing → BVR or drop |
| Opening framing (40) | app.py:366-376 | fit_camera only in record (capture.py:46) | missing → BR frame on connect |
| Layer checkboxes per kind (90) | app.py:21-56,162-180 | Layers folder | partial → BVR app |
| Body-source dropdown (25) | app.py:159-161 | groups + set_visible | → BVR app ~20 lines |
| Clip pager + cache (~100) | app.py:90,118-134,204-227 | none; timeline never shrinks | → BVR app; BR: reset timeline on remove_group |
| Info panel (17) | scene.py:398-414 | server.gui | BVR |
| WebM / .viser export | — | Viewer.save/record | free |

Size after rebuild: scene.py 990 → ~250-300 (clip_layers adapter); loading.py 1,114 → ~450 generic + ~170 datasets; app.py 551 → ~200; helpers.py 197 → ~20; view_3d.py 292 → ~60. Core ≈1,000-1,100 lines (from 3,163) + ~250 in datasets. Drop onboard camera: −120; delete tools/geometry/surface.py: −162.

## 4. Dataset-specific vs generic
To datasets: ClimbingVideos/Athletics/BEDLAM DB farms (`--scenes`, `--bedlam`, scene_db; view_3d.py:147-224,232-245,273-276); athletics sport filter (app.py:122-125,209-214,498-502; data absent); BEDLAM2 loader (loading.py:838-956; app.py:507; scene.py:373,404); OpenCap plate estimate (loading.py:166-185; scene.py:941-956); sensor forces + limb→joint map (loading.py:134-163; scene.py:925-926); `mhr_1_sup` (no producer in BVR; mhr_1.npz lives in ClimbingVideos features); `kindyn_manual` (no producer; absent from examples → remove).
Suggestions: generic "measured forces (origins, vectors)" layer in core, dataset scripts write that format; if the BEDLAM2 importer wrote an `out/<stem>/`-shaped tree, the viewer's BEDLAM branch disappears.
Generic: the `out/<stem>/` tree, multi-clip pager, cameras, gravity/ground, fused cloud, human points, human_optim body sources, contacts schema 2, kindyn forces, multi-view rig (cameras.npz + views/), video pane.

## 5. Viewer tests (numpy-only, none touch viser/GUI)
test_viewer_bedlam.py (7) → bedlam2; test_viewer_camera_pose.py (4) → only if onboard camera kept; test_viewer_groups.py (3) → athletics or delete; test_viewer_multiview.py (4) generic; test_viewer_plate_forces.py (5) → opencap + runner. Moving drawing onto BR breaks no test; only import paths change.

## Other
- Dev BVR .venv is empty; main venv's editable .pth files point at main BR/BH.
- Rotation order: BR layers take xyzw; BVR stores wxyz (loading.py:566) → adapter converts, or use BH layers from HumanState.

## Open questions
1. Keep the onboard camera? 2. Keep athletics path / `kindyn_manual`? 3. Visual changes OK: torque as colour, BR arrow style, per-step human point upload? 4. Generic gaps into BR dev (per-step valid mask, timeline reset, ImagePanel, per-step colours, tilted/filled ground, live setters, framing) vs thin BVR layers — recommend BR for the first three. 5. Remote bandwidth unmeasured.
