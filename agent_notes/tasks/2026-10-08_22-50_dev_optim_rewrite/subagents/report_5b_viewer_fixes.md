# Report 5b — viewer fix pass on the review findings (mars_top_opus, 2026-10-09)

**Result:** all 7 items done; uncommitted in the dev worktree. On 6 real clips (4 regression clips, the
climb_wall_3 rig with wall sensors, an OpenCap clip with plate estimates) the new viewer builds exactly the
same layers as the pre-5b viewer: every array of every layer equal to 1e-6, same items, kinds, sources and
valid masks (scratchpad `v5b/compare_old_new.py`). 39 tests pass (viewer 21 = 19 + 2 new; contact 18 = 17 + 1).

## Per finding (what changed)
1. Stale video controls: `App.load` now resets the video folder on every clip: panels, view menu options/value,
   menu visible only with >1 view, folder hidden when the clip has no video.
2. Failed read: `load` reads first, then removes the old groups. On an exception it prints the traceback,
   keeps the clip and `self.items`, and puts the pager dropdown back (re-raises only on the very first clip).
   `load` of the clip on screen is a no-op (makes the revert free). Fake-server test for 1 + 2 added.
3. Timeline folder: video panels are `_VideoPanel(ImagePanel)` that build only their image inside the video
   folder; `Viewer.add` runs outside it, so BR's lazily built Timeline folder can never land there.
4. Contact shape check: `contact_labels` refuses `frame_contact` not shaped (people, body-file frames, file's
   frame names) with "ignoring incompatible contact artifact …"; the named-subset expansion is kept. Needs the
   body file's own frame count: new `BodySource.stored_frames`. Test added in test_contact_confidence.py
   (3 frames for a 2-frame body; 1 column for 2 names; both warned, None).
5. Info panel: "N cameras" back (rig views, else camera poses), "people" kept.
6. Duplicates/dead: `_fit_frames` + `_frames` → one public `fit_frames(arr, n, axis=1)` (zero pad; the fill
   arg was always 0/False); `camera_poses` = `se3.inverse(se3.from_matrix(extr))`: vs the hand-rolled one on
   all 4 clips + the rig: positions ≤ 2.4e-7 m, quaternions bit-equal (`v5b/check_se3.py`); `ClipData.path`
   dropped; `camera_stride` removed from configs/pipeline{,_peter,_opencap}.toml; `[runtime].device` comment
   now "compute device of the stages (the viewer poses bodies on the CPU)".
7. Convention test (test_viewer_app.py): extrinsic Rz(+90), t=(1,2,3) → frustum at (-2,1,-3), viser wxyz
   (√½,0,0,-√½); root quaternion xyzw Rz(+90), local force +x / torque +y → world +y / -x arrows. Checked that
   it fails for 3 injected swaps (w2c instead of c2w, xyzw/wxyz roll in `camera_poses`, wxyz root quaternion).
8. `VideoFrames.close()` releases the capture (decoded frames stay, reopens lazily); `App.load` calls it for
   the clip leaving the screen (after its panels are removed). Clips never leave the cache, so "on leaving the
   screen" is the only point that bounds open handles (one clip's views at a time).

## Q1 hook (no core → datasets import)
- `ClipData.extras: list[ForceArrows]`; `ForceArrows(name, kind, origins (T,C,3), vectors (T,C,3) N, color,
  valid (T,C), source)` in loading.py. `clip_layers` draws each as a solid ArrowsLayer on the solved forces'
  scale/style in the source's group, without knowing what it is.
- `App(..., read=load_clip_data)`: one callable with `load_clip_data`'s signature. `view_3d(..., markers=
  CLIP_MARKERS)`: what makes a directory a clip (core default: `geometry/scene.npz` only).
- datasets_layers: `read_clip` (BEDLAM2 scene → `load_bedlam_data`, else `load_clip_data` + `measured_forces`),
  `measured_forces` (FK of the first dynamics source's first person for the wall-sensor origins; returns
  ForceArrows), `sensor_arrows`/`plate_arrows` return arrays, `BEDLAM_MARKER`. Imports only public loading
  names (`fov_y_rad`, `fit_frames` made public).
- scripts/view_3d.py is the only importer: `read=datasets_layers.read_clip, markers=(*CLIP_MARKERS,
  BEDLAM_MARKER)`; `--bedlam` also forces `show_smplx` on (it already forced video on).
- Chosen over a list of readers: one callable is the smallest hook; step 4 can split `read_clip` per dataset.

## Verification
- pytest tests/test_viewer_*.py tests/test_contact_confidence.py: 39 passed; full suite collects 240.
- 4-clip fake-server pass (`v5b/verify_fake.py`, read=read_clip, video on): OK; 28/28/28/50 layers, timeline
  445/104/471/363, fps 30/30/60/60, contacts/forces per source as in report_5.
- Real process `scripts/view_3d.py` on port 8197 (boulder_1 + a no-video clip + an unreadable clip): curl 200;
  driven over viser's websocket (`v5b/ws_pager.py`): switch to the no-video clip → old image removed, video
  folder visible False, view menu reset to "-"; process killed. Same sequence in-process on a real viser server
  (`v5b/check_pager_real.py`): unreadable clip → traceback, clip/items/dropdown kept, all checkboxes still work;
  boulder_1 decoder released on leave, reopened on return.
- `--bedlam` real run (r5 config on port 8196, scene 00000000): curl 200; sidebar: source "ground truth",
  contact frames + force checkboxes present (no contact warning), video pane, info "443 cameras"; killed.
- Warm load of boulder_1, old vs new loader: 7.7 s vs 7.6 s.

## Line counts (tools/viewer/*.py + scripts/view_3d.py)
| | app | layers | loading | datasets_layers | __init__ | view_3d | total | core (app+layers+loading+init) |
|---|---|---|---|---|---|---|---|---|
| before 5b | 225 | 252 | 479 | 249 | 15 | 83 | 1303 | 971 |
| after 5b | 253 | 245 | 501 | 277 | 15 | 86 | 1377 | 1014 |

## Left / open
1. The core still *names* dataset things without importing them: KINDS keys/labels "sensor"/"plate", the
   gravity tag "bedlam2_gt" in the info panel, BEDLAM2 file names in docstrings.
2. Edge: a BEDLAM2 scene opened as a plain target with `[viewer].smplx = false` now fails (body None); before,
   BEDLAM ignored `smplx`. `--bedlam` is unaffected.
3. The clip cache is unbounded (ClipData + every decoded video frame per visited clip), as in the old viewer.
4. `camera_poses` via se3 can flip the quaternion sign for >90° camera rotations (Shepperd branch): same rotation.
5. 5r lost features 6–7 (per-frame fov, frustum trail, ground section thickness, BEDLAM fps source) untouched.
6. Leftover symlink-only farm /tmp/bvr_view_5_90h8hs from the --bedlam run (SIGTERM skips atexit; trash
   refuses /tmp), next to the review's two.
7. viser quirk met while driving the server: a GUI update from a client that has not yet sent its camera
   updates the server-side value but runs no callback (browsers always send the camera first).
Scratch: scratchpad `v5b/` (before/ copies, check_se3, compare_old_new, verify_fake, check_pager_real, ws_pager, live/).
