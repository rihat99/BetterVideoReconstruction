# Report D — docs in line with the cleaned repo (2026-10-08, HEAD b0b3dce)

Result: 8 files edited in the repo + 2 in sibling repos, all uncommitted. Brief grep leaves 2 hits, both correct (external runners). Full suite 250 passed.

## Files edited
| file | change |
|---|---|
| CLAUDE.md | branch section → "Working principles" on `main` (4 rules kept); Git workflow → commit to `main`; new "Data layout" paragraph; config list = the 9 TOMLs (corpus `[export]` → `[split]`, which is what exists); "Three pipelines" + run_dataset paragraph; stage table + `measure_contacts_boards.py`, `build_support_scene.py` rows; athletics clause "(not present on this machine)"; Run block → `uv run --no-sync` + `../data/examples/...` paths + one run_dataset line; layout tree regenerated (run_dataset, import_*/label_*/sensor_forces/opencap_*, diagnostics/, methods/, peter_gt/, tools/paths, body, npz_io, bedlam_forces; third_party: 5 submodules + vendored vggt-omega) |
| README.md | `--stages dynamics` row → `--stages kindyn_1` row (the dynamics stage is no longer dispatched, scripts/stages/human_optimization.py:49-81); export_contact_dataset sentence + "the export's test labels" clause removed; section "Dataset pipelines" + one run_dataset paragraph; both commands `uv run --no-sync` |
| INSTALL.md | :23-25 five submodules + vendored vggt-omega (.gitmodules has 5; `git ls-files third_party/vggt-omega` shows plain files; pyproject.toml:100-102); :52-54 "vendored", not "editable submodule" |
| docs/board_contacts.md | :109-114, :156 run_gt_contact → `run_dataset.py --dataset climb_wall_3\|parkour --variant gt_contact` (same --shard/--clips/--steps) |
| configs/pipeline_opencap.toml:1 | names `run_dataset.py --dataset opencap` |
| configs/pipeline_peter.toml:1 | names run_dataset.py / datasets.toml (it also serves parkour) |
| scripts/methods/opencap_physpt_export.py:263 | help → "where PhysPT's own run_opencap.py wrote" |
| scripts/diagnostics/render_kindyn_overlays.py | module docstring: "Four rigs" → "Five output trees"; disc paragraph (dropped the dead white-rimmed model-only disc: every TREES clip now has truth); wall panel paragraph (dropped "limb panels shaded, wall never measured a limb": all walls set limbs_measured=True); usage `trial_10` (first-wall clip, gone) → `trial_01_part1`; wall2_clip docstring "Unlike the first wall," dropped. AST ok |
| contact_anything/CLAUDE.md:258-259 | `../BetterVideoReconstruction/peter/...` → `../data/peter/...` |
| BetterContactAnnotator/README.md:36-37 | default → `../data/examples/out` |

## Deviation from the brief
- scripts/methods/opencap_estmf_export.py:4 NOT changed and :263 not pointed at run_dataset: both name the EXTERNAL runners, `data/Estimating-3D-Motion-Forces/scripts/run_opencap.py` (writes `<clip>_pred.npz`) and `other/PhysPT/run_opencap.py` (writes `<clip>_output.npz`), both present. run_dataset produces neither file. :263 only got "PhysPT's own" to disambiguate.
- The corner-cell rig paragraph was already removed by step S; what remained were the "Four rigs" count and two paragraphs describing that rig's display.
- CLAUDE.md Run block was not "already fixed": it said `--save-dir out/` (would now create an untracked out/ in the repo, since .gitignore no longer lists it). Rewritten to `../data/examples/...`.

## Verification
- Brief grep leftovers: `scripts/methods/opencap_estmf_export.py:4` and `opencap_physpt_export.py:263` (external runners, above). Nothing else.
- Every written path exists (`ls -d` on all data dirs, configs, scripts, tools, third_party; sibling paths from each sibling root).
- Doc commands `--dry-run`: climb_wall_3 gt_contact `--shard 1/4` (17 clips), `--clips a_00_01_part1 --steps contacts_1`, parkour gt_contact (28): all rc 0; no new file in the out trees (`find -newer`).
- `opencap_physpt_export.py --help` exit 0. pytest (CUDA_VISIBLE_DEVICES=2): **250 passed in 38.96s**.

## Left as is (outside this brief; for the caller)
- scripts/stages/human_optimization.py:3-11,94 docstring/help still list a `dynamics` stage that is no longer dispatched (frozen core).
- scripts/diagnostics/_common.py:336-340 `_wall_panel` docstring still describes "the first wall"; render_kindyn_overlays.py:155-158, :403 code comments say "corner cell".
- README "Documentation" list lacks force_contacts / board_contacts / quality docs (CLAUDE.md has them).
- view_3d.py `--scenes athletics` code path kept; ../data/AthleticsVideos absent.

## Open questions
1. Fix the frozen-core `dynamics` mentions in human_optimization.py, or leave?
2. Is "Five output trees" fine, or should the gt tree count as part of climb_wall_3 ("four rigs")?
