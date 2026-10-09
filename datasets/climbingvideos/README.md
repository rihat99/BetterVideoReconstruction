# climbingvideos — the ClimbingVideos scene corpus

The large scene database of curated climbing clips (indoor boulder + rope) the pipeline fans out over, stage by stage,
into a stage-first feature tree; the contact-label confidence and the train/test split are calibrated and assigned here.

- **Where:** `<data root>/ClimbingVideos/` (`paths.py: CLIMBING`): `scenes/scenes.db` (owned by the upstream `../BetterParsing`
  pipeline, read-only except the explicit split / discard writes), `frames/`, `features/<stage>/<aa>/<bb>/<scene>/`
  (`<aa>/<bb> = scene_id[:2]/scene_id[2:4]`). `configs/corpus.toml` `data_root` is resolved against the repo root, not `$BVR_DATA_ROOT`.
- **Run:** `uv run --no-sync python datasets/climbingvideos/corpus.py --stage <sam3|vggt|sapiens|sam3d|smplx|geometry|gravity|human_optim|kinematics_2|kindyn|contacts> [--scenes ID … | --sample N | --limit N] [--dry-run]`
  — any stage over the DB, stride-sharded across `[corpus].devices`; model stages sweep with one load per GPU, farm stages build a
  local `out/<stem>/` symlink farm and call `python -m bvr.stages.<name>`. Knobs: `configs/corpus.toml` (`[corpus]`, `[overlay]`,
  per-stage overrides, `[split]`) on top of `configs/pipeline.toml` + `configs/human_optimization.toml`.
- **Curate:** `assign_dataset_split.py [--dry-run]` assigns `dataset_split` per source video (`[split]`);
  `calibrate_contact_confidence.py --config <toml …> --contacts 1|2 [--write]` fits `[contact_confidence]` on the annotated scenes.
- **View / files:** `scripts/view_3d.py --scenes` pages the DB (`viewer.py`; selection from `[viewer]` in `configs/corpus.toml`);
  `scene_db.py` (DB access, feature layout, row selection), `tests/test_corpus.py`.
