# climbingvideos — the ClimbingVideos scene corpus

The large scene database of curated climbing clips (indoor boulder + rope) that PACT trains on: the pipeline's outputs per
scene, kept stage first; the contact-label confidence and the train/test split are calibrated and assigned here.

- **Where:** `<data root>/ClimbingVideos/` (`paths.py: CLIMBING`, `$BVR_DATA_ROOT`): `scenes/scenes.db` (owned by the upstream
  `../BetterParsing` pipeline, read-only except the explicit split write), `scenes/<aa>/<bb>/<scene>.mp4`,
  `features/<stage>/<aa>/<bb>/<scene>/` (`<aa>/<bb> = scene_id[:2]/scene_id[2:4]`). Which scenes: `[corpus]` of `configs/corpus.toml`.
- **Run:** `uv run python bvr_datasets/climbingvideos/corpus.py --steps <steps> [--save-dir <root>] [--scenes=<ids> | --sample N | --limit N] [--config <delta>] [--skip-existing] [--shard i/n] [--dry-run]`.
  A view links each scene's video and stage folders into a clip-first run root, so `bvr.run.chain` runs the steps as on any clip.
  With `--save-dir` the outputs go to a root of their own (the corpus is the upstream, untouched); without it they go through the
  view `<corpus>/bvr` into the feature tree, and what they replace to `<corpus>/bvr/_trash`. The model steps that write a whole
  stage folder (sam3, vggt, sapiens, gravity) need `--save-dir`. One process per GPU: `CUDA_VISIBLE_DEVICES=<g> ... --shard i/n`.
- **View:** `uv run python scripts/view_3d.py --source <root>/_corpus --save-dir <root> --upstream <root>/_corpus` (a run root), or
  `--source <corpus>/bvr --save-dir <corpus>/bvr` (the corpus itself; `corpus.py` without `--steps` builds that view).
- **Curate:** `assign_dataset_split.py [--dry-run]` assigns `dataset_split` per source video (`[split]`);
  `calibrate_contact_confidence.py [--contacts 1|2] [--write]` fits `[contact_confidence]` of `configs/bvr.toml` on the annotated scenes.
- **Files:** `scene_db.py` (DB access, feature layout, row selection), `tests/test_corpus.py`.
