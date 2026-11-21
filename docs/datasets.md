# Datasets

`bvr_datasets/` is the code around the measured and labelled datasets. It builds on `bvr` and runs the same chain;
nothing in `bvr` imports it. Each dataset is a folder with a `README.md` that says what the data is, where it lives
under `$BVR_DATA_ROOT`, and the commands. Run every command from the repository root.

| folder | what it is | what the code does |
|---|---|---|
| `examples/` | the short annotated climbing clips, `examples/data/<clip>.mp4` with a manual contact annotation `<clip>.npz` | scores a run's contact labels against the annotations |
| `bedlam2/` | BEDLAM2 synthetic renders with exact SMPL-X ground truth | imports the scenes; labels masks, floor contacts and frozen-pose forces from the ground truth (PACT's BEDLAM2 labels) |
| `climbingvideos/` | the ClimbingVideos scene corpus PACT trains on | runs any steps of the chain over the corpus; fits `[contact_confidence]`; assigns the train and test split |
| `peter/` | the climbing force rigs, a force board under each limb's hold | the recipe of the static-camera rigs; reads measured contacts from the boards; scores estimated against measured; the three-camera ground truth of climb_wall_3 |
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
in body weights, support, root residual and force confidence. It reads the files with numpy and the poses with the
SMPL-X model file, so it compares runs of any two versions of the code; contact frames are matched by name (`head`
read as `head_top`). Support and root residual are each side's own, over the frames where its inverse dynamics count
by today's rule (stencil tracked, person labelled): a person without contact labels has none. One clip takes about
8 s on the CPU.

## The force rigs

The peter rigs, Parkour and OpenCap share one recipe: a static camera, one person, a body whose contact frames are
on the hands and feet only. `scripts/pipeline.py` runs the model steps with `bvr_datasets/peter/configs/peter.toml`
(OpenCap: `bvr_datasets/opencap/configs/opencap.toml`), and `python -m bvr_datasets.peter.chain` runs the human steps
on that body. With `--measured` the chain reads the contact labels from the force sensors instead and writes them to a
second run root beside the first (`bvr_datasets/peter/board_contacts.md`). These datasets keep one folder per clip,
and a run names a clip by its video, so a run reads a folder of links `<clip>.mp4`; each README has the one-line loop
that makes it.

## The ground truth of climb_wall_3

Three calibrated cameras filmed climb_wall_3, so its trials have a body that does not depend on one camera's depth.
`python -m bvr_datasets.peter.gt.fit_multiview` runs the human steps on a trial with the same problem, terms and
solvers as the pipeline and these differences: the keypoints and the SAM 3D Body estimate of every camera stand where
one camera's stood, each camera weighing by how well its keypoints agree; the contact labels are the boards';
nothing of the body may be behind the wall's plane; and in `kindyn_1` each limb's summed force is pulled to what its
board measured. The climber's shape is frozen, one per climber: `python -m bvr_datasets.peter.gt.shape` fits it to
their measured height and mass and the three cameras. The weights are the delta `bvr_datasets/peter/gt/gt.toml`.

The run reads prepared trees (`peter/climb_wall_3_gt/human/<trial>/`: the calibrated cameras, each camera's model
outputs, the wall's cloud and plane, gravity, the boards' forces) as its upstream and writes
`<run>/<trial>/human_optim/`; `bvr_datasets/peter/README.md` has the commands. PACT's ForceWall benchmark reads
`kindyn_1.npz` (`q`, `betas`, `valid_mask`) and `contacts_1.npz` (`limb_contact`) of these trials.

The wall and the trees come from the scene scripts of `bvr_datasets/peter/gt/`, which write under a root of their own
(`<root>/scene/<set>/`, `<root>/human/<trial>/`): a people-free frame per camera, the chessboard calibration of the
three cameras, VGGT on the three stills (through the `vggt` stage's `model`), and the wall's cloud as VGGT's depth
placed by the calibrated cameras, one scale per camera measured against them (`scene_calib.npz`). `prepare_human.py`
then writes a trial's tree from its videos, the calibration (with each camera's frame lag behind `cam_left`), that
cloud and its plane, gravity and the boards. The pipeline's model steps run on each camera of the trial, and
`prepare_human.py --cam3` completes the cameras for the fit: `cam_3`, bumped between recordings, gets its pose
re-solved from the climber's keypoints triangulated by the front pair, and each camera the spread of its keypoints,
which weighs it in the fit.

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

36 tests on the CPU in about 10 s, with `BETTERHUMAN_MODELS_DIR` and the data root set. Four OpenCap plate tests read
raw trials of LabValidation and fail where the data root does not hold them, as a test of a body fails without the
body models: nothing skips.
