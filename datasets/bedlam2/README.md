# bedlam2 — BEDLAM2 pose ground truth

BEDLAM2's synthetic renders with exact SMPL-X pose ground truth (the bodies are AMASS motions). The dataset ships no per-frame
label file — pose, camera and placement are composed from `be_seq.csv`, the render database and the motion npz — and its floors
are flat planes, so person masks, ground contact and frozen-pose contact forces can all be measured from the pose alone.

- **Where:** the download `<data root>/BEDLAM2/` (`b2_imagedata_download/`, `bedlam2_render_database-*/bedlam2.sqlite`, `b2_motions_npz_training/`);
  our copy `<data root>/BEDLAM2_our/` in the ClimbingVideos layout (`scenes/`, `frames/`, one feature stage `features/gt/<aa>/<bb>/<scene>/`
  with `smplx.npz`, camera and gravity, plus the labels below). `paths.py: BEDLAM2, BEDLAM2_OUR`.
- **Import:** `uv run --no-sync python datasets/bedlam2/import_bedlam2.py --step plan|scenes|features|frames|db|verify|all [--sample 12 | --scenes … | --limit N] [--workers N --skip-existing]`.
- **Label:** `uv run --no-sync python datasets/bedlam2/label_bedlam2.py --step masks|contacts|forces|select|all [--scenes … | --sample N | --limit N] [--apply]` —
  `masks`: z-buffered instance maps + `bboxes.npz` with the per-frame `valid` flag (`masks.py`, `rasterize.py`); `contacts`: the 35 contact frames
  against the floor from height, stillness and skin facing (`contacts.py`) → `contacts.npz`; `forces`: `kindyn_1`'s force solve at the frozen
  pose (`forces.py`) → `forces.npz`; `select`: scenes with no valid person, `--apply` clears `human_selected` (rollback JSON kept).
  Knobs: `configs/bedlam.toml` (`[bedlam]`, `[bedlam_forces]`). Nothing is deleted; `smplx.npz` is never rewritten.
- **View / tests:** `scripts/view_3d.py --bedlam` (`viewer.py`: one `ground truth` body source, cameras, floor, gravity, contacts, forces);
  `tests/test_bedlam_labels.py`, `test_bedlam_forces.py`, `test_viewer_bedlam.py`.
