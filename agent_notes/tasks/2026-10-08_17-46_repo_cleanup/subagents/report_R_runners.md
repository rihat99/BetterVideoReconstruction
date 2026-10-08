# Report R — one dataset runner (mars_top, 2026-10-08)

Result: `scripts/run_parkour.py`, `run_opencap.py`, `run_peter.py`, `run_gt_contact.py` (1,088 lines) are replaced by `scripts/run_dataset.py` (531 lines) + `configs/datasets.toml` (97 lines). Commands identical to the HEAD runners in 21 synthetic cases; real `--skip-existing` runs on one processed clip per dataset launch nothing; full suite 250 passed.

## 1. What replaced what
| old | new |
|---|---|
| `run_parkour.py [--clips] [--steps]` | `run_dataset.py --dataset parkour [--clips] [--steps] [--shard]` |
| `run_opencap.py [--shard] [--variant v] [--view-farm]` | `run_dataset.py --dataset opencap [--shard] [--variant v] [--view-farm]` |
| `run_peter.py --dataset campus_board\|climb_wall_2\|climb_wall_3 [--views a,b] [--shard]` | `run_dataset.py --dataset <same> [--views a,b] [--shard]` |
| `run_gt_contact.py [--dataset climb_wall_3\|parkour]` | `run_dataset.py --dataset climb_wall_3\|parkour --variant gt_contact` |

New flags: `--skip-existing` (also leaves out a model sweep every selected clip has finished, incl. the extra-view sam3 sweeps; without it the sweeps launch exactly as before and the stage TOML's `skip_existing` does the skipping) and `--dry-run` (prints every `=== command ===`, touches nothing: no manifest, no farm, no mkdir). Dropped: the `--config` / `--human-config` CLI overrides (now `pipeline_config` / `human_config` per dataset and `config` per variant in datasets.toml, as the brief's CLI prescribes; no script or doc passed them — grep).

`configs/datasets.toml`: `[datasets.<name>]` with `root`, `clips` glob (one-video datasets) or `views` (rigs; first view = the single-view tree, any other `--views` selection → `out_multi` with `multi_steps`), `out`, `out_multi`, `pipeline_config`, `human_config`, `steps` (the chain; the rigs have no gravity step, as in run_peter). `[variants.<name>]` with `steps` (default: plain chain), `config` (default: `human_optimization_<name>.toml`, run_opencap's rule, so unlisted names still work), `inherit`, `contacts = "boards"`, `requires`. Tables: parkour, opencap, campus_board, climb_wall_2, climb_wall_3; floorplane, floorplane_edgev, gt_contact. `owns_scene` is derived from the chain (a support_scene step), as the test demands, not a key.

Public names kept in `scripts/run_dataset.py`: `MANIFEST_STEPS`, `CLIP_OUTPUT`, `SUPPORT_STEPS`, `VARIANT_STEPS`, `VARIANT_CHAINS` (loaded from the TOML at import), `DATASETS`, `VARIANTS`, `owns_scene`, `build_farm(clip_dir, variant, *, synthetic_scene=False, inherit=())` (run_opencap's + run_gt_contact's INHERITED links), `build_view_farm`. One `_clip_command` (fuse joins all views; `boards=True` → `measure_contacts_boards.py`; `overlay` → `render_kindyn_overlays.py <clip>/human_optim_<variant>`), one `_manifest_name`, one `_parse_shard`, one per-clip loop.

## 2. Files
- Created: `configs/datasets.toml`, `scripts/run_dataset.py`.
- Edited: `tests/test_support_scene.py` (imports → `scripts.run_dataset`; `_point_opencap_at()` helper monkeypatches `DATASETS["opencap"]["root"/"out"]` instead of `run_opencap._OPENCAP`; `main(argv)` → `main(["--dataset", "opencap", *argv])`; assertions untouched), `tests/test_viewer_plate_forces.py:10` (import).
- Trashed (`trash.py` → `trash/`): `scripts/run_parkour.py`, `run_opencap.py`, `run_peter.py`, `run_gt_contact.py`.
- Not touched (checked): `scripts/peter_gt/run_all_clips.sh` and `run_human.py` call no runner; `configs/pipeline_peter.toml` / `pipeline_opencap.toml` needed no key move.

## 3. Equivalence (HEAD copies vs new)
Harness in `/tmp/claude-1003/-home-rikhat-akizhanov-better/cb45a0c1-6981-4fe2-a421-62ef38728597/scratchpad/eq/`: `make_tree.py` builds a fake `BVR_DATA_ROOT` (2 clips per dataset, one OpenCap clip pre-fused, one campus_board clip with single-view masks to reuse, `sensor_forces.npz` for the gt_contact selection, 1 of 2 climb_wall_2 clips); `run_fake.py` runs a runner's `main()` with `subprocess.run` stubbed to print nothing extra, create the step's output file and return 0; `equiv.py` runs, per case, old runner (`PYTHONPATH=repo`, from `scratchpad/old/`), new runner through the same stub, and new `--dry-run`, each on a fresh tree, then compares the `=== ... ===` lines (paths normalised) and a full snapshot of both trees (entries, symlink targets, json contents).

`uv run --no-sync python scratchpad/eq/equiv.py` — 21 cases: parkour (all; `--clips kv02 --steps fuse,scale,kinematics_1`), opencap (all; `--shard 1/2`; `--shard 2/2 --steps sam3,fuse,gravity`; `--variant floorplane`; floorplane `--steps support_scene_floor,kinematics_1 --clips ...`; `--variant floorplane_edgev`; `--variant somevariant --steps kinematics_1,kindyn_1`), campus_board (all; `--views cam_left,cam_right`), climb_wall_2 (all), climb_wall_3 (all; `--shard 2/2 --steps sapiens,scale,kindyn_1`; `--views cam_left,cam_right,cam_3`; `--views cam_right`), gt_contact on climb_wall_3 (all; `--clips b --steps contacts_1,kinematics_2`; `--shard 2/2`), climb_wall_2, parkour.
- Commands: identical in all 21 cases (old = new = new `--dry-run`; 2–21 commands per case, 200 in total), incl. manifests named by the same hash, the multi-view `sources` manifest, extra-view sam3 sweep into `_cam_right/`, fuse with both videos, farm paths, board contacts and the overlay.
- Trees: identical in 16/21. The other 5 (parkour_sub, opencap_floorplane, opencap_fp_sub, opencap_edgev, opencap_plain) differ by exactly one entry: the old runner wrote an unused `_clips_<hash>.json` although no sweep ran; the new runner writes the manifest only when a sweep launches (run_gt_contact never wrote one; nothing else reads them — grep `_clips_`).

## 4. Real data (CUDA_VISIBLE_DEVICES=2, nothing launched)
- `--dry-run` discovery: parkour 28, opencap 161, campus_board 24, climb_wall_2 7, climb_wall_3 68 clips; `--variant gt_contact`: climb_wall_3 68, parkour 28 (= clips with `human_optim/sensor_forces.npz`).
- `--skip-existing` (real run, after a `--dry-run` of the same selection showed 0 commands): parkour kv01_PKFC, opencap subject2_DJ1_cam1, campus_board 01_hanging, climb_wall_2 trial_01_part1, climb_wall_3 a_00_01_part1, climb_wall_3 `--variant gt_contact` a_00_01_part1, campus_board `--views cam_left,cam_right` 01_hanging: all rc=0, 0 commands launched, every step "skipping", 0.1 s each; the people report prints `01_hanging 1/1`. Side effects: none — no new `_clips_*.json` under any out tree, and the `_gt_contact/` + `human_optim_gt_contact/` symlinks of a_00_01_part1 have the same targets before and after (md5 of `find -type l -printf '%p -> %l'` equal).
- `--help`: exit 0.

## 5. Tests
- `tests/test_support_scene.py` + `tests/test_viewer_plate_forces.py`: 34 passed.
- Full suite `CUDA_VISIBLE_DEVICES=2 uv run --no-sync pytest tests/ -q -p no:cacheprovider`: 250 passed in 38.6 s (same count as report M).

## 6. Behaviour unified across the four runners (deliberate, each was one runner's rule)
1. `--steps` must be a subset of this run's chain in chain order, for every dataset (run_opencap's guard; run_parkour / run_peter / run_gt_contact ran steps in the order given, unchecked).
2. A per-clip step counts as failed when rc != 0 OR its output is missing (run_peter / run_gt_contact); exit code 1 when any clip failed (run_parkour / run_opencap; run_peter / run_gt_contact always returned 0).
3. Manifest written only when a sweep launches (§3).
4. gt_contact clips are the source clips that have `requires` in their out tree (run_gt_contact listed out dirs with sensor_forces.npz; same set when every out dir has a source clip — 68/68 and 28/28 here).
5. Log prefix `[run_dataset]`; per-clip line `clip/step: ok|FAILED (rc=..) in N min`.
6. Now expressible, not before: `--shard` on parkour; `--variant` on any dataset (a `gt_contact` run on opencap would select the 12 clips with sensor_forces.npz; floorplane on a rig). Not exercised.

## 7. Known gap (unchanged, audit A §1c)
`--variant floorplane_edgev` runs the plain chain on the clip's real fused cloud with `human_optimization_floorplane_edgev.toml`, exactly as `run_opencap.py` did. The edgev / massedgev runs on disk were kindyn-only reruns inside the floorplane farm (geometry and kinematics/contacts linked from `human_optim_floorplane`); no code builds that layout, and none was added. Stated in the `[variants.floorplane_edgev]` comment.

## 8. Not verified
- No real stage was launched: a run that actually executes a sweep or a per-clip stage through the new runner was not done (every real run skipped everything, as intended for ~minutes checks). The command strings are proven identical to the old runners' (§3).
- `build_view_farm` on real data (covered by `test_view_farm_aliases_each_variant_farm_by_clip_name` only).
- 531 lines vs the brief's ~450 target: the four runners' behaviours (multi-view, variants, boards, dry-run, skip-existing) plus a 45-line docstring; no logic was cut to fit.

## 9. References to the trashed scripts outside my file set (for steps D / I / S)
`configs/pipeline_peter.toml:1`, `configs/pipeline_opencap.toml:1` (header comments), `docs/board_contacts.md:109,112,154` (`run_gt_contact.py` usage), `scripts/methods/opencap_physpt_export.py:263` and `scripts/methods/opencap_estmf_export.py:4` (docstrings naming run_opencap.py).

## Open questions
1. Should `floorplane_edgev` stay a listed variant at all, given it cannot reproduce the on-disk run?
2. `--variant` + multi-view `--views` is accepted and fails only through the step check ("must be a subset of this run's chain"); refuse it explicitly?
3. Manifest-only-when-a-sweep-launches (§3): keep, or restore "always write" for byte-identical trees?
