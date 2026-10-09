# Report 5r — adversarial review of the rebuilt viewer + the two checks report_5 skipped (mars_aux, 2026-10-09; saved by the main session)

**Verdict:** no wrong convention and no data-path regression. Every numeric check against the stored artifacts and the old viewer's math agrees to ~1e-6 m. All three real `scripts/view_3d.py` processes served HTTP 200 and rendered in headless Chromium. Remaining: 3 low-severity bugs (only #1 reproduced), a few unlisted lost features (one is a lost safety check), code-quality nits, and one structural issue for step 4 (Q1: the generic core imports the dataset module). Scratch: `/tmp/claude-1003/-home-rikhat-akizhanov-better/cb45a0c1-6981-4fe2-a421-62ef38728597/scratchpad/r5/`.

## The two checks
(a) boulder_1, real process (config copy r5/pipeline_8193.toml, port 8193; log r5/live_boulder.log): viser up +1.3 s, `loading` +11.9 s (imports + `load_body` on CPU), serving +27.5 s; clip load 15.6 s. `curl /` 200 (2.89 MB); screenshot r5/live_boulder.png: camera framed on the climber, info panel filled. Only 4 sources (kindyn_1.npz absent while the regression run rewrote human_optim/). In-process breakdown (load avg ~50): `load_body` 11.8 s; human.npz read 3.5 s + `_subsample_human_mask` 8.1 s (12.07 M points); `load_clip_data` 28 s; `clip_layers` 6 s. Human points are the largest cost.
(b) `--bedlam`, real process: scene 00000000 (has bboxes, contacts, forces); no DB needed; needs `[viewer].data_root = "/home/rikhat.akizhanov/better/data/BEDLAM2_our"` (with BVR_DATA_ROOT=data_dev the default root does not exist) + `scene_ids = "00000000"` (r5/pipeline_bedlam_8194.toml). Farm +1.2 s, loading +12.5 s, scene 0.28 s; curl 200; r5/live_bedlam.png shows GT body, contact frames, forces, ground, video pane.
Extra: `--scenes` (ClimbingVideos DB) two scenes pBNShrkay8s_0002,_0003, video on: serving +27.5 s, 200, 5 sources; r5/live_scenes.png pager "scene 1/2", video matches the body. Pager on a real server (r5/check_rig_pager.py): rig → boulder_1 → back, no exception; timeline 475 → 445 → 475.
Leftovers: symlink-only farms `/tmp/bvr_view_i7lz5o_l`, `/tmp/bvr_view_upj2jo06` (atexit does not run on SIGTERM; trash refuses /tmp).

## Conventions, checked numerically (r5/check_numeric.py; farm human_optim → human_optim_A; boulder_1, gtIGgs, BEDLAM 00000000)
FK joints vs kindyn_1 `joints_world` ≤0.001 mm; root q[:3] = FK joint 0; BEDLAM joints vs forces.npz ≤0.001 mm, projected joints 100% inside bbox_full. joint_names / contact_frame_names match the body. Finger-force folding conserves the per-frame sum; finger columns 0. Root wrench `so3.act` (xyzw) vs old `_quat_mat`: 1e-8. Camera pose vs old `_camera_world_pose`: 3e-8 m, |q·q| = 1.000000 (camera-to-world, wxyz right). original/prefit root fold vs old per-joint fold: 0.001 mm. Contact frames = `Body.forward` exactly; normals outward on 100% of frames; sphere colours = kindyn_1 `frame_contact`. Human points per frame = old `fid == f`. Rig clip climb_wall_3_gt a_00_01_part6: each view's `VideoFrames` byte-identical to the old sequential decode with lag (offsets 0/2/2); sensor layer built (4 limbs). By reading: force scale 1.5e-3 m/N, torque 4.8e-3, gravity colours, ground cell formula, vertical fov, metres.

## Bugs (low severity)
1. Stale video controls after paging to a clip with no video (app.py:101-110 only updates `view_menu`/panels when the new clip has videos). Reproduced: rig → boulder_1 left `view_menu.visible=True` with the rig's options, empty "video" folder. Old viewer always reset.
2. Failed read leaves a broken viewer (by reading): app.py:92-94 removes old groups before `_read`; a read error leaves an empty viewer and `self.items` pointing at disposed layers → next `_apply` raises KeyError in `Viewer._entry`. Fix: read first. (Old viewer had the same order.)
3. Possible timeline-folder nesting: if a clip's first multi-step layer is an ImagePanel (only static 3D layers), `Viewer._add` builds "Timeline" inside `with self.video_folder` (app.py:102-106).

## Lost features not on report_5's retire list
4. Contact-file shape check gone: old loader refused a contacts npz whose (P,N,F) mismatched the body ("incompatible contact artifact", main loading.py:344-356); new `contact_labels` pads/truncates the frame axis silently (loading.py:258) → stale file draws misaligned labels without warning.
5. Info panel: camera count gone; "climbers" (ids from human points) became "people" (ids from bodies).
6. Camera drawing: per-frame fov became the median; viridis frustum trail + cursor gone (report_5 covers this only via `camera_stride`).
7. Ground `section_thickness=2.0` dropped (no BR setting); BEDLAM playback fps from camera.npz instead of the video container.

## Code quality
- Q1 generic core ↔ dataset code tangled: layers.py:31 imports datasets_layers; sensor/plate arrows drawn inside generic `_body_items` (layers.py:239-245); app.py:21 + BEDLAM dispatch and sensor/plate readers at app.py:80-85; datasets_layers.py:21-31 imports private `_fov_y_rad` from loading.py. Step 4 needs a hook (reader list / extra items on ClipData) before lifting by moving files.
- Duplicates: loading.py:160 `_fit_frames` vs datasets_layers.py:77 `_frames`; loading.py:136-141 `camera_poses` hand-rolls `se3.inverse(se3.from_matrix(extr))`.
- Dead/stale: `ClipData.path` (loading.py:52) never read; `camera_stride` still in configs/pipeline.toml:160, pipeline_peter.toml:131, pipeline_opencap.toml:132; `[runtime].device` comments ("shared by stages and the viewer") now wrong; CLAUDE.md:138-146 viewer paragraph stale (step 6).
- Calls outside `server.gui`: `client.camera` (app.py:156-157), `server.on_client_connect` (75), `get_clients` (119), `sleep_forever` (225) — none draw.
- Tests do not pin conventions: test_viewer_app.py uses identity extrinsics (:87) and identity joint quaternions (:105); a wxyz↔xyzw or c2w↔w2c swap in `camera_poses`, the root wrench or the original/prefit fold would still pass all 19 tests. The retired test_viewer_camera_pose.py was the only extrinsic-convention test. One test with a non-trivial rotation closes the gap.
- `VideoFrames` never releases its `cv2.VideoCapture` (loading.py:336-337): one open handle per cached clip.
- Opening view on orbit-camera BEDLAM scenes is top-down (same logic as the old viewer; not a regression).

Notes: tests/test_contact_confidence.py was modified at 07:50 by step 3b (imports `body_source`, `contact_labels`); its 4 viewer tests pass.
