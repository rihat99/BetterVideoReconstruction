# Report G2 — solid arrows (BR dev) and the BVR force arrows on them (mars_top, 2026-10-09)

**Result:** BR dev `ArrowsLayer` draws solid arrows when given `shaft_radius=`; the BVR dev viewer uses them with the
old viewer's radii for forces, root wrench, sensors/plates, gravity and contact-frame normals. Rendered side by side
with viser's native `add_arrows` at the old radii (same session, same camera), the two look the same.

## Part A — BetterRobot dev (branch dev), commit `8828842`
API (smallest surface: two keyword args on the existing class, no `style` flag, no sibling class):
```python
ArrowsLayer(name, origins, vectors, *, scale=1.0, colors=..., head_size=0.03, line_width=3.0,
            shaft_radius: float | None = None, head_radius: float | None = None)
```
- `shaft_radius=None`: the old line style, unchanged. A float switches to solid; a `style=` flag would be redundant,
  since solid needs a radius anyway. A sibling class would repeat the colours / `valid` / world_points code.
- Solid: batched unit cylinder for shafts and batched unit cone for heads under `<root>/shafts`, `<root>/heads`.
  Per step it writes positions, wxyzs and scales only (nodes are never removed and re-added). The head has fixed
  length `head_size` and radius `head_radius` (default `head_size/2`). The shaft length is `|v| - head_size`,
  clamped at 0. A vector shorter than the head draws the head alone, its tip on the arrow tip and its base behind
  the origin, as viser does. Zero vectors and `(T,N)`-invalid vectors are hidden (scale 0). Per-step colours
  reach both shafts and heads. `scale=`, `world_points` and recording are unchanged.
- Test `test_solid_arrows_keep_a_fixed_head_and_hide_zero_and_invalid_vectors` (tests/viewer/test_core.py);
  docs/concepts/viewer.md paragraph; changelog line. ruff check + format clean on the changed files (viewer package
  still has the same 11 old findings). `mkdocs build --strict` OK.
- BR suite: **1674 passed, 11 skipped, 20 deselected** (step G: 1673 + 1 new test); viewer tests 44 passed.
- Visual: `/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-09_07-55_solid_arrows/vis/solid_vs_line.png`
  (front row line style, back row solid: long, tilted, short = head only; zero and invalid hidden). vis/ is git-ignored.
  No notes commit.

## Part B — BVR dev (uncommitted; edited only tools/viewer/layers.py + 2 lines of datasets_layers.py)
| Arrow | shaft r / head r / head length (m) | old scene.py |
|---|---|---|
| joint forces, frame forces, sensor, plate | 0.012 / 0.028 / 0.05 | same (865-912, 957-965) |
| root force + torque | 0.014 / 0.032 / 0.06 | same: old `_root_arrow` used these, not 0.012 (968-976) |
| gravity | 0.12 / 0.25 / 0.4 × camera_scale (0.1) | same (176-189) |
| contact-frame normals (6 cm) | 0.003 / 0.0072 / 0.018 | same (788-791) |
Line style kept for camera path, frame links, scene normals.
- New: joint and frame forces get a `(T,N)` mask `person valid & |f| > 1e-3 N` (`_FORCE_FLOOR_N`). This is the old
  viewer's filter. It is needed because a solid head keeps its 5 cm size: boulder_1 kindyn_1 has 953 joint-force
  entries with 0 < |f| ≤ 1e-3 N, each of which would draw a full cone. Sensors/plates were already gated.
- BVR viewer tests (test_viewer_app/bedlam/multiview/plate_forces): **19 passed**.
- Screenshots: re-recorded with scratchpad record_boulder.py, same procedure as report_5 (step 31), and overwritten.
  The old PNGs are in /home/rikhat.akizhanov/better/BetterVideoReconstruction/trash/.
  `.../dev_optim_rewrite/vis/viewer_boulder_1.png`, `.../vis/viewer_boulder_1_forces.png`.
- Comparison with the old look: `.../vis/viewer_boulder_1_forces_vs_old.png` and `..._vs_old_close.png`. Left: ours.
  Right: our layers hidden and viser `add_arrows` drawn at the old radii on the same vectors. Same browser session
  and camera. 3.3k / 10k of 1.02M pixels differ (far / close view).

## Still different from the old viewer
1. Cone 12 segments, cylinder 16 (BR unit meshes): slightly faceted next to viser's smoother cone in the close-up.
2. The old root arrows were hidden below 1e-4 m. Not ported: the shortest root arrow on boulder_1 is 7 mm.
3. The old viewer removed and re-added nodes every frame. Now the nodes persist and only their transforms are
   written. No visual difference.

## Open questions / observations
1. Viewer.record camera race: two `record` calls in one process with the BVR `App` gave a wrong camera on the second
   call once. App's `on_client_connect` framing and/or the browser's first camera report can override `set_camera`.
   The vis PNGs were checked to match the requested camera (same view as the one-session render). Worth a BR fix
   (e.g. set the camera twice or wait for the client's camera)?
2. Should `head_radius` also scale the line-style heads? Now it is solid-only, as `line_width` is line-only.
Scratch: scratchpad g2/ (compare_one_session.py, side_*.png, before_*.png), br_solid_arrows.py, br_g2_pytest.log.
