# BetterVideoReconstruction

Reconstruct a **per-person SMPL-X body trajectory** from one monocular video,
and the **contact forces** acting on it.

Given a clip, the chain tracks everyone in it (SAM 3), recovers the camera and
depth (VGGT-Omega), fits a 3D body to every tracked person in every frame
(SAM-3D-Body), converts it to SMPL-X, fuses the depth into a scene point cloud,
anchors the scene's metric scale to the human body, and refines the bodies
against the video (**visually-guided kinematic refinement**: 2D keypoints, depth
chamfer, pose prior, smoothness). Standalone stages then detect scene contact on
the refined bodies, estimate gravity, and solve pose and per-contact forces
together by inverse dynamics (`kindyn_1`).

```
video ─▶ sam3 ─▶ vggt ─▶ sapiens ─▶ sam3d ─▶ smplx ─▶ fuse ─▶ scale ─▶ kinematics_1
                                   ─▶ contacts_1 ─▶ kinematics_2 ─▶ contacts_2 ─▶ kindyn_1
```

| Stage | What it does | Output (under `out/<stem>/`) |
|-------|--------------|------------------------------|
| sam3 | SAM 3 video tracker | per-person masks + bbox tracks (`sam3/`) |
| vggt | VGGT-Omega | multi-view cameras + depth (`vggt/camera.npz`, `vggt/depth/`) |
| sapiens | Sapiens2-5B | Goliath-308 keypoints (`sapiens/pose.npz`) |
| sam3d | SAM-3D-Body | per-frame MHR body for every person (`sam3d/params.npz`) |
| smplx | MHR → SMPL-X fit | per-frame SMPL-X parameters (`sam3d/smplx_params.npz`) |
| fuse | point-cloud fuse | up-to-scale scene + per-frame human points (`geometry/`) |
| scale | human-body ruler + pre-fit | metric scale, rescales `geometry/` in place (`geometry/scale.npz`, `prefit.npz`) |
| kinematics_1 | Adam + L-BFGS | the refined world-frame bodies (`human_optim/kinematic_1.npz`) |
| contacts_1/2 | contact readout | per-contact-frame labels + confidence (`human_optim/contacts_<tag>.npz`) |
| kinematics_2 | contact-patch repair | each labelled contact pulled onto the surface it touches (`human_optim/kinematic_2.npz`) |
| kindyn_1 | kinematics + dynamics | pose and per-contact-frame forces solved together (`human_optim/kindyn_1.npz`) |

The optimization runs on [`BetterRobot`](../BetterRobot) (kinematics, inverse
dynamics, the optimizer) with [`BetterHuman`](../BetterHuman) bodies; both are
installed editable from the checkouts next to this repo.

## Install

See [`INSTALL.md`](INSTALL.md): `uv sync` builds the environment from the lock,
then three manual steps (detectron2, the `sam-3d-body` path, the MHR assets) and
the model checkpoints. Data lives outside the repo in `../data` (`BVR_DATA_ROOT`
moves it).

## Run one clip

```bash
uv run python scripts/pipeline.py --source ../data/examples/data/boulder_1.mp4 --save-dir ../data/examples/out/
# then the contact / force stages on it
uv run python scripts/pipeline.py --source ../data/examples/data/boulder_1.mp4 --save-dir ../data/examples/out/ --steps contacts_1,kinematics_2,contacts_2,kindyn_1
```

Every behaviour knob lives in a TOML (`configs/pipeline.toml`,
`configs/human_optimization.toml`); the CLI takes only targets. A finished step
is skipped on a re-run; `--dry-run` prints the commands without running them.
Each stage is also runnable on its own as `python -m bvr.stages.<name>`.

## View it

```bash
uv run python scripts/view_3d.py ../data/examples/out/boulder_1/
```

A viser web viewer: the fused scene, the cameras, every body source
(`original` / `prefit` / `kinematics` / `kinematics_2` / `kindyn_1`) as a skinned
mesh with its skeleton, the contact frames (red touching, green free), the
solved forces, and the video in sync. Point it at `../data/examples/out/` to page
through every clip.

## Run a dataset

```bash
uv run python scripts/pipeline.py --dataset parkour --dry-run
uv run python scripts/pipeline.py --dataset climb_wall_3 --variant gt_contact
uv run python datasets/climbingvideos/corpus.py --stage geometry --limit 10 --dry-run
```

The measured datasets (LAAS Parkour, OpenCap LabValidation, the peter climbing
rigs) are tables in `datasets/<name>/configs/dataset.toml`: the same chain over
every clip, plus variant chains (`gt_contact`: contacts measured by the rig's
force sensors; `floorplane`: a synthetic support scene). The ClimbingVideos
scene DB runs any stage through `datasets/climbingvideos/corpus.py`. Each
`datasets/<name>/README.md` says what the data is, where it lives, and how to
import, run and score it.

## Documentation

- [`INSTALL.md`](INSTALL.md) — environment, submodules, model assets.
- [`CLAUDE.md`](CLAUDE.md) — architecture, the `out/<stem>/` contract, configs, layout, conventions.
- [`docs/scale.md`](docs/scale.md) — the metric scale + in-place metricization.
- [`docs/fusion.md`](docs/fusion.md) — the point clouds + the relative two-pass voxel merge.
- [`docs/human_optimization.md`](docs/human_optimization.md) — the four-stage refinement, every loss term.
- [`docs/contacts.md`](docs/contacts.md) — per-contact-frame contact detection.
- [`docs/contact_confidence.md`](docs/contact_confidence.md) — how much a contact label can be trusted, and how that number is fitted.
- [`docs/force_contacts.md`](docs/force_contacts.md) — contact detection by inverse dynamics, end to end.
- [`docs/quality.md`](docs/quality.md) — depth-smear downweighting, the scene-quality table, force confidence.
