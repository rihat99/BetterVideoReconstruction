# Report F — review fixes (mars_top_opus, 2026-10-08)

All 7 items done; full suite 250 passed.

## 1. Batch manifests (data only)
12 `_batch_manifest.jsonl` found under DATA/peter and DATA/examples; originals copied to `<repo>/trash/manifests/<path>` (cmp-identical). 7 rewritten (Edit replace_all per prefix; every line still parses as JSON):

| manifest (under DATA/) | resolved before → after |
|---|---|
| peter/out_climb_wall_2_single | 0/8 → 8/8 (was `-dev/peter/`) |
| peter/out_campus_board_multi | 0/10 → 10/10 |
| peter/out_campus_board_single | 0/2 → 2/2 |
| peter/out_climb_wall_3_single | 0/34 → 34/34 |
| examples/out_2 | 0/10 → 10/10 (was `-dev/data_2/`, `-dev/out_2/`) |
| peter/climb_wall_3_gt/scene | 0/10 → 10/10 |
| peter/climb_wall_3_gt/human/_manifests | 0/18 → 18/18 |
| peter/out_campus_board_multi/_cam_right | empty |
| peter/_provenance/cw2_gravity, cw3_gravity/{00,16,32}/out | 0/2 each, unchanged (point at old scratch/; provenance only) |

`_manifest_video` now resolves `out_climb_wall_3_single/a_00_01_part4` → `peter/climb_wall_3/a_00_01_part4/cam_left.mp4` and `examples/out_2/-ZQhn42Myp8_0031` → `examples/data_2/-ZQhn42Myp8_0031.mp4`.

## 2. scripts/run_dataset.py
- New `_farm_link` (:136): same-target link kept; any other link or a real file raises SystemExit "`<link>` already points to X, not Y; move it aside to rebuild this farm". Used by build_farm (:153) for farm/inherit/geometry links and by build_view_farm (:189). A synthetic-scene farm whose geometry is a symlink now raises (:179) instead of being unlinked.
- Config check (:468-470): `[run_dataset] config not found: <path>` for human and pipeline config, after the step guards (keeps the existing tests' expectations).
- `build_farm(..., dry_run=True)` under --dry-run checks links and creates nothing (:522). `--view-farm --dry-run` prints `would link ...` (:412, :204).
Verification: test_support_scene + test_viewer_plate_forces 34 passed. opencap `--variant floorplane_edgev --clips subject10_DJ1_cam1 --dry-run` → exit 1 with the conflict message (link unchanged). climb_wall_3 `--variant gt_contact --dry-run` → exit 0, farm listing hash identical. `--variant nosuchvariant` → exit 1 config not found. `--view-farm --dry-run` creates nothing.

## 3. baseline_farm.py:145-150
`results/test_scenes.txt` written only without `--clips`; with `--clips` prints "left untouched". --help updated. Only --help run.

## 4. .gitignore:228-234
Anchored `/data/ /data_2/ /out/ /out_2/ /peter/ /scratch/` with a comment. `git check-ignore -v out/x` → `.gitignore:231`. `scripts/out/x` not ignored (anchoring works). 0 tracked files under those dirs.

## 5. CLAUDE.md:53-56
Sentence on corpus.toml `data_root` being repo-relative; claim checked against corpus.py:216,242, assign_dataset_split.py:47, calibrate_contact_confidence.py:408-410, score_test_contacts.py:67.

## 6. DATA/Estimating-3D-Motion-Forces/scripts/run_climb_wall_3.py:40
SOURCE → DATA/peter/climb_wall_3; original copied to `<repo>/trash/external/run_climb_wall_3.py` (file untracked there).

## 7. Final
250 passed in 37.8 s. Diff this step: run_dataset.py +51/−22, baseline_farm.py +8/−5, .gitignore +7, CLAUDE.md +3.

## Open
- The 4 `_provenance/*` manifests still point at old scratch paths (provenance only).
- run_climb_wall_3.py:18 docstring names the retired cw3_estmf_inputs.py (successor: scripts/methods/estmf_inputs.py --dataset climb_wall_3).
- gt_contact farm links for a_00_01_part1 dated today 18:59: step R's real --skip-existing run rebuilt them with identical targets (harmless).
