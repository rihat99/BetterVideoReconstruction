# BetterVideoReconstruction

Reconstruct a **per-person SMPL-X body trajectory** from a monocular video.
Given a clip, the pipeline tracks everyone in it, recovers the camera, fits a 3D
body every frame for each tracked person, converts each to SMPL-X, anchors the
scene's metric scale to the human body, fuses the metric geometry, and runs a
**visually-guided kinematic refinement** (2D reprojection + depth chamfer + pose
prior + acceleration/jerk smoothness). Standalone SMPL-X stages then add scene
contact, world-frame gravity, per-contact-joint **forces** via inverse dynamics
(RNEA), and reconstruction-trust scoring.

It is the video-focused sibling of
[`BetterImageReconstruction`](../BetterImageReconstruction) (image/scene) and
[`BetterHumanForce`](../BetterHumanForce) (the original research repo this was
split from). All three sit next to the [`BetterRobot`](../BetterRobot)
PyTorch-native robotics library, which provides the kinematics, RNEA, and
optimization residuals used here.

## Pipeline

The default chain (`scripts/pipeline.py`) runs these stages in order:

```
video ─▶ sam3 ─▶ vggt ─▶ sapiens ─▶ sam3d ─▶ smplx ─▶ scale ─▶ fuse ─▶ optimize_smplx
```

| Stage | Model | Output (under `out/<stem>/`) |
|-------|-------|--------|
| sam3 | SAM 3 (video) | per-person masks + bbox tracks |
| vggt | VGGT-Omega | multi-view cameras + depth (≤500 frames; records the VGGT inference resolution) |
| sapiens | Sapiens2-5B | Goliath-308 keypoint pose |
| sam3d | SAM-3D-Body | per-frame MHR body params (all tracked people) |
| smplx | MHR→SMPL-X fit | per-frame SMPL-X params |
| scale | human-body ruler | metric scale anchored to the body (`geometry/scale.npz`) |
| fuse | metric fuse | metric scene + per-frame human point clouds (`geometry/`) |
| optimize_smplx | Adam + L-BFGS | visually-guided kinematic refinement (`sam3d/smplx_kinematic.npz`) |

**VGGT-Omega** runs right after the tracker as the multi-view camera/geometry
stage; it writes `vggt/camera.npz` (cross-frame extrinsics + depth, plus the
inference resolution) and the body stage reads that camera directly. Its
access-gated checkpoints are not bundled (see [`INSTALL.md`](INSTALL.md)).

**Metric scale** is recovered by `scale` (`estimate_scale`), which uses the
SAM-3D-Body human as a ruler, so it runs after the body stages and before `fuse`
(which applies it).

### Standalone SMPL-X stages

Off the default chain, each runs on a single clip's `out/<stem>/` dir:

| Stage | Output |
|-------|--------|
| `optimize_smplx_2` | annotation-guided refinement (`sam3d/smplx_kinematic_2.npz`) |
| `estimate_contacts` | per-joint scene contact (`sam3d/cnt_kin.npz` + overlay) |
| `estimate_gravity_geocalib` | world-frame gravity (`geocalib/gravity.npz`) |
| `optimize_dynamics_smplx` | force-only RNEA: per-contact-joint forces + torques (`sam3d/smplx_dynamics.npz`) |
| `estimate_uncertainty` | per-frame + per-contact reconstruction-trust scoring |

### Dataset pipeline (ClimbingVideos DB)

For the large scene database, `scripts/track_sam3_scenes.py` seeds SAM 3 from the
per-scene YOLO track, then
`scripts/precompute_scenes.py --stage {vggt,sapiens,sam3d,smplx}` sweeps each model
over the batch (shared helpers in `scripts/scene_db.py`).

## Quick start

```bash
# See INSTALL.md for env + submodule + checkpoint setup first.
uv run python scripts/pipeline.py --source data/boulder_1.mp4 --save-dir out/

# Then view the fused scene + refined SMPL-X bodies in a viser web viewer:
uv run python scripts/view_3d.py out/boulder_1/
```

Outputs land under `out/<stem>/` (per clip). The current build is file-based
and small-batch — a DB-backed dataset is planned later. Each stage is also
runnable on its own; see [`CLAUDE.md`](CLAUDE.md) for the full stage table, the
`out/<stem>/` contract, and conventions.

## Documentation

- [`INSTALL.md`](INSTALL.md) — environment, submodules, model assets.
- [`CLAUDE.md`](CLAUDE.md) — architecture, pipeline contract, conventions.
- [`docs/geometry_fusion_plan.md`](docs/geometry_fusion_plan.md) — `fuse_geometry` design + `geometry/*.npz` contract.
