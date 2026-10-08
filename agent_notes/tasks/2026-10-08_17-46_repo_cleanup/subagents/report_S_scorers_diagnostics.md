# Report S — published-model scorers and the diagnostics helper module (mars_top_opus, 2026-10-08)

Result: two `--benchmark` scorers replace four scripts; `scripts/diagnostics/_common.py` holds every helper the live diagnostics imported from the campus_board-era scripts. Every old-vs-new output is byte-identical (JSON, PNG tables, MP4 overlays). Full suite: 250 passed. The last two campus_board files were blocked by one import in scripts/export_opencap_mot.py (outside the agent's set); applied by the main session afterwards (§5).

## 1. What replaced what
| old (HEAD 0cf1281) | new |
|---|---|
| score_baseline_contacts.py + score_gt_contacts.py | `score_published_contacts.py --benchmark {climbingvideos,climb_wall_3} [--out DIR]` |
| score_baseline_pose.py + score_gt_pose.py | `score_published_pose.py --benchmark {climbingvideos,climb_wall_3} [--out DIR]` |
| helpers in score_prediction_contacts / score_pipeline_contacts / render_prediction_overlays | `_common.py` (plain functions + constants) |

- Output locations unchanged: climbingvideos → `CLIMBING/results/baseline_{contact,pose}_*`; climb_wall_3 → `PETER/climb_wall_3_gt/others/gt_{contact,pose}_*`. `--out` is a target flag (default = that dir).
- Shared once: contacts `ours`, `six_joint`, `VertexGroups`, `vertex_rows`, `_count`, the row loop; pose `pose`, `similarity`, `camera_metrics`, `world_metrics`, `jitter`, `score_model`, `score_init`. Benchmark-specific data in one `BENCHMARKS` dict per scorer; output schemas kept exactly per benchmark.
- `_common.py` contents, all AST-identical to their HEAD origins: `_render_table`, `_draw_table`, `_table_size`, `_ROW_H`/`_MARGIN`, `_fmt`, `_rates` (score_prediction_contacts); `_limb_members`, `_limb_load`, `_FINGER` (score_pipeline_contacts); `_wired` (one copy); `_fit`, `_draw_caption`, `_draw_contact_discs`, `_force_panel`, `_wall_panel` + colour/size constants (render_prediction_overlays).

Choices: CAMPUS_ROOT/PLATE_ROOT live in render_kindyn_overlays next to the other rig roots; render_prediction_overlays' own `_draw_force_arrows` not moved (only its render() used it); `wall_clip()` dropped with its docstring paragraph; unified `_count` truncates predictions to label length and drops -1 labels on both benchmarks (CV used to raise IndexError on mismatch; no effect on current data).

## 2. Files
Created: `_common.py` (413 lines), `score_published_contacts.py` (361), `score_published_pose.py` (392).
Edited: render_kindyn_overlays.py, render_sup_force_overlays.py, render_parkour_method_overlays.py, compare_climb_wall_3_runs.py, compare_gt_contact.py, compare_opencap_forces.py, score_combined_table.py, score_test_contacts.py, score_init_world.py.
Trashed: render_prediction_overlays.py, score_baseline_contacts.py, score_gt_contacts.py, score_baseline_pose.py, score_gt_pose.py. Then (main session, §5): score_pipeline_contacts.py, score_prediction_contacts.py.
No live script runs any of the three campus_board files as a subprocess; only render_kindyn_overlays.py is run that way (run_dataset.py:208, peter_gt/run_human.py:171), CLI unchanged.

## 3. Equivalence (all byte-identical)
Setup: HEAD snapshot via `git archive HEAD scripts tools configs` under the scratchpad `S_step/head` with symlinks so old paths resolve; old `__main__` replayed with only the output dir changed; new runs via the real CLI with `--out`. Full benchmarks.

| check | result |
|---|---|
| contacts CV (108 scenes) | JSON + PNG identical |
| contacts cw3 (68 clips) | JSON + 2 PNGs identical; old JSON equals on-disk file |
| pose CV (108, CPU both) | JSON + PNG identical; 42/42 floats exact |
| pose cw3 (68, GPU 2 both) | JSON + PNG identical; 66/66 floats exact |
| score_init_world | both patched JSONs identical |
| compare_climb_wall_3_runs | JSON + PNG identical |
| score_combined_table | 2 PNGs + JSON identical |
| render_kindyn: 02_gianna_hanging (129 fr), 02_hanging (172), b_32_02_part1 cw3 (338) | mp4 bytes same, every frame np.array_equal |
| render_sup trial_01_part1 cw2 (317 fr) | same |
| render_parkour_method kv01_PKFC ours_optim (34 fr) | same |

Render harness swapped only `imageio` for a shim writing into scratch; real overlay files kept their mtimes. CV pose + init-world pairs ran on CPU after a CUDA OOM on the shared GPU 2.

## 4. Other verification
All 19 diagnostics modules import in their own process; none of the 12 live importers pulls in a retired module. `--help` exit 0 for the six argparse CLIs changed; the five non-argparse ones verified by the runs above. `_common` not shadowed in the venv. pytest: 250 passed in 40.8 s (taken while R and I were editing).

## 5. Blocked item, applied by the main session
`scripts/export_opencap_mot.py:77` imported `_limb_members` from score_pipeline_contacts (tests import export_opencap_mot). Fix: import from `_common`, then trash score_pipeline_contacts.py and score_prediction_contacts.py.

## 6. Pre-existing problems (not fixed)
- render_sup_force_overlays fails on all 24 campus_board clips (`KeyError: 'contact_probs'`, older forces_sup.npz format); only the 7 cw2 clips render.
- score_test_contacts.py fails (`KeyError: 'pred_limbs'`) on current ClimbingVideos data, old and new.
- `/home/rikhat.akizhanov/better/_talk_figs/scripts/rig_plots.py` (outside the repo) imports compare_climb_wall_2 and score_pipeline_contacts: already broken by step P.
- `CLIMBING/results/baseline_contact_metrics.json` (Sep 10) differs from a fresh HEAD run because the test split changed since; cw3 file matches.

## Open questions
1. score_test_contacts.py and the campus branch of render_sup_force_overlays no longer run on current data: keep, fix, or retire?
2. render_kindyn_overlays' docstring still says "Four rigs" and describes the first wall's corner-cell reading; trim in step D.
