# BetterVideoReconstruction

Reconstruct a **per-person SMPL-X body trajectory** from a monocular video.
Given a clip, the pipeline tracks everyone in it, recovers the camera, fits a 3D
body every frame for each tracked person, converts each to SMPL-X, fuses the
up-to-scale point clouds, anchors the scene's metric scale to the human body
(rescaling the fused geometry in place), and runs a
**visually-guided kinematic refinement** (2D reprojection + depth chamfer + pose
prior + acceleration/jerk smoothness). Standalone SMPL-X stages then add scene
contact, world-frame gravity, and per-contact-joint **forces** via inverse dynamics
(RNEA).

It is the video-focused sibling of
[`BetterImageReconstruction`](../BetterImageReconstruction) (image/scene) and
[`BetterHumanForce`](../BetterHumanForce) (the original research repo this was
split from). All three sit next to the [`BetterRobot`](../BetterRobot)
PyTorch-native robotics library, which provides the kinematics, RNEA, and
optimization residuals used here.

## Pipeline

The default chain (`scripts/pipeline.py`) runs these stages in order:

```
video ─▶ sam3 ─▶ vggt ─▶ sapiens ─▶ sam3d ─▶ smplx ─▶ fuse ─▶ scale ─▶ human_optimization
```

| Stage | Model | Output (under `out/<stem>/`) |
|-------|-------|--------|
| sam3 | SAM 3 (video) | per-person masks + bbox tracks |
| vggt | VGGT-Omega | multi-view cameras + depth (≤500 frames; records the VGGT inference resolution) |
| sapiens | Sapiens2-5B | Goliath-308 keypoint pose |
| sam3d | SAM-3D-Body | per-frame MHR body params (all tracked people) |
| smplx | MHR→SMPL-X fit | per-frame SMPL-X params |
| fuse | point-cloud fuse | up-to-scale scene + per-frame human point clouds (`geometry/`) |
| scale | human-body ruler + pre-fit | metric scale anchored to the body, refined by a dual scale+body alignment; rescales the fused geometry in place (`geometry/scale.npz`, `geometry/prefit.npz`) |
| human_optimization | Adam + L-BFGS | visually-guided kinematic refinement (`human_optim/kinematic_1.npz`) |

**VGGT-Omega** runs right after the tracker as the multi-view camera/geometry
stage; it writes `vggt/camera.npz` (cross-frame extrinsics + depth, plus the
inference resolution) and the body stage reads that camera directly. Its
access-gated checkpoints are not bundled (see [`INSTALL.md`](INSTALL.md)).

**Metric scale** is recovered by `scale` (`estimate_scale`), which uses the
SAM-3D-Body human as a ruler against the fused human points, so it runs after
`fuse` and rescales the saved geometry in place.

### Standalone SMPL-X stages

Off the default chain, each runs on a single clip's `out/<stem>/` dir:

| Stage | Output |
|-------|--------|
| `human_optimization --stages kinematics_2` | optional annotation- or auto-contact-guided refinement (`human_optim/kinematic_2.npz`) |
| `estimate_contacts` | per-joint scene contact and confidence (`human_optim/contacts_1.npz` + overlay) |
| `estimate_gravity_geocalib` | world-frame gravity (`geocalib/gravity.npz`) |
| `human_optimization --stages dynamics` | force-only RNEA: per-contact-joint forces + torques (`human_optim/dynamics.npz`) |

### Dataset pipeline (ClimbingVideos DB)

For the large scene database, `scripts/corpus.py` seeds SAM 3 from the per-scene
YOLO track and sweeps every corpus stage over the batch (shared helpers in
`scripts/scene_db.py`).
`scripts/export_contact_dataset.py` packages the resulting frames, masks,
intrinsics, and 22-joint contact labels into a self-contained train/test dataset.
Manual annotations are not used by `kinematics_1` or by confidence generation.
They are used only by the optional manual `kinematics_2` mode, confidence auditing,
and the export's test labels.

## Quick start

```bash
# See INSTALL.md for env + submodule + checkpoint setup first.
uv run python scripts/pipeline.py --source data/boulder_1.mp4 --save-dir out/

# Then view the fused scene + refined SMPL-X bodies in a viser web viewer:
uv run python scripts/view_3d.py out/boulder_1/
```

Outputs land under `out/<stem>/` (per clip). Each stage is also runnable on its
own; behavior knobs live in `configs/*.toml` (CLIs only take targets). See
[`CLAUDE.md`](CLAUDE.md) for the full stage table, the `out/<stem>/` contract,
and conventions.

## Documentation

- [`INSTALL.md`](INSTALL.md) — environment, submodules, model assets.
- [`CLAUDE.md`](CLAUDE.md) — architecture, pipeline contract, conventions.
- [`docs/scale.md`](docs/scale.md) — the metric scale + in-place metricization.
- [`docs/fusion.md`](docs/fusion.md) — the point clouds + the relative two-pass voxel merge.
- [`docs/human_optimization.md`](docs/human_optimization.md) — the four-stage refinement, every loss term.
- [`docs/contacts.md`](docs/contacts.md) — per-joint contact detection.
- [`docs/contact_confidence.md`](docs/contact_confidence.md) — per-joint contact-confidence formula and artifact schema.
