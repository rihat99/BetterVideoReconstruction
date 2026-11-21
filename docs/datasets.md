# Datasets

`bvr_datasets/` is the code around the measured and labelled datasets. It builds on `bvr` and runs the same chain;
nothing in `bvr` imports it. Each dataset is a folder with a `README.md` that says what the data is, where it lives
under `$BVR_DATA_ROOT`, and the commands. Run every command from the repository root.

| folder | what it is | what the code does |
|---|---|---|
| `examples/` | the short annotated climbing clips, `examples/data/<clip>.mp4` with a manual contact annotation `<clip>.npz` | scores a run's contact labels against the annotations |
| `bedlam2/` | BEDLAM2 synthetic renders with exact SMPL-X ground truth | imports the scenes; labels masks, floor contacts and frozen-pose forces from the ground truth (PACT's BEDLAM2 labels) |
| `climbingvideos/` | the ClimbingVideos scene corpus PACT trains on | runs any steps of the chain over the corpus; fits `[contact_confidence]`; assigns the train and test split |
| `peter/` | the climbing force rigs, a force board under each limb's hold | the recipe of the static-camera rigs; reads measured contacts from the boards; scores estimated against measured |
| `parkour/` | LAAS Parkour: floor plates and an instrumented bar | imports the clips and forces; scores pose (Procrustes MPJPE) and forces, through the peter recipe |
| `opencap/` | OpenCap LabValidation: two floor force plates, motion capture | imports and synchronises the trials; scores ground reaction forces against the plates, through the peter recipe |

## Scoring contacts on the example clips

```bash
uv run python bvr_datasets/examples/score_frame_contacts.py --predictions <run root> --contacts 1
```

It folds `<clip>/human_optim/contacts_<n>.npz` onto the annotated joints (hand from the palm and finger frames, foot
from the toe and ball frames, ankle from the heel, knee from the knee) and prints precision, recall and F1 per joint
and pooled, in under a second for the 13 clips. This is the contact number of the acceptance checks.

## Comparing two runs

```bash
uv run python scripts/compare_runs.py <root a> <root b> [--clips a,b] [--annotations $BVR_DATA_ROOT/examples/data]
```

It reads `kinematic_1`, `contacts_1`, `kinematic_2`, `contacts_2` and `kindyn_1` of every clip present in both roots
and prints, per clip and pooled: body-joint distance in mm, root orientation, each side's root acceleration, label
agreement and F1 between the two, each side's F1 against the annotations, and for `kindyn_1` the force difference
in body weights, support, root residual and force confidence. It uses numpy and the SMPL-X model file only, so it
compares runs of any two versions of the code; contact frames are matched by name (`head` read as `head_top`). One
clip takes about 8 s on the CPU.

## The force rigs

The peter rigs, Parkour and OpenCap share one recipe: a static camera, one person, a body whose contact frames are
on the hands and feet only. `scripts/pipeline.py` runs the model steps with `bvr_datasets/peter/configs/peter.toml`
(OpenCap: `bvr_datasets/opencap/configs/opencap.toml`), and `python -m bvr_datasets.peter.chain` runs the human steps
on that body. With `--measured` the chain reads the contact labels from the force sensors instead and writes them to a
second run root beside the first (`bvr_datasets/peter/board_contacts.md`). These datasets keep one folder per clip,
and a run names a clip by its video, so a run reads a folder of links `<clip>.mp4`; each README has the one-line loop
that makes it.

OpenCap's stored scores used a synthetic floor and a 3 cm shoe heel on the contact frames. The rebuilt core has no
contact frame offset along the normal, so that recipe does not run on it; OpenCap runs the plain rig recipe.

## Labelling for PACT

`bedlam2/label_bedlam2.py --step masks|contacts|forces|select` writes person masks, floor contacts and the forces of
the frozen ground-truth pose beside BEDLAM2's ground truth; `climbingvideos/corpus.py --steps <steps>` runs the chain
over the corpus scenes, either into a run root of its own (`--save-dir`, the corpus is then the upstream and stays
untouched) or into the corpus's feature tree. Both write the files PACT's loaders read, with the contact frame
`head_top`.

## Tests

```bash
uv run pytest bvr_datasets -q
```

36 tests on the CPU in about 10 s, with `BETTERHUMAN_MODELS_DIR` set. Four OpenCap plate tests read raw trials from
the data root and skip when it does not hold them.
