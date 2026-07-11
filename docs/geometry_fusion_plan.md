# `fuse_geometry` — design + the `geometry/*.npz` contract

`fuse_geometry` turns VGGT's up-to-scale world into the one metric world every
downstream stage shares. It reads the raw VGGT depth + cameras (`vggt/`), the
human-ruler clip scale (`geometry/scale.npz`, from `estimate_scale`), and the SAM3
masks (`sam3/`), and writes three files under `geometry/`. Everything after fuse —
the kinematic optimizer, the contact readout, the trust scoring, the viewer —
consumes these files and nothing upstream of them, so fuse is the single owner of
"what the world looks like and how big it is".

## The build

Per frame: clean the VGGT depth (edge-aware, `--edge-rtol`), backproject through the
VGGT intrinsics, split points into **scene** (outside every tracked person's dilated
mask) and **human** (inside a reconstructed climber's mask; tracked non-climbers are
excluded from both clouds). Scene points are voxel-merged across the clip
(`--voxel-scene`, default 5 cm) and small disconnected clusters are dropped
(`--min-scene-voxels`); human points stay per-frame (keyed by `frame_id`/`obj_id`).
All positions are multiplied by the clip scale, and the camera translations with
them, so `transform.npz`'s extrinsics and both clouds live in one metric frame.

## The body-anchored rescale (round 13)

The human-ruler scale compares the body to *its own* VGGT depth, so it is
structurally blind to a human-vs-scene bias: on some clips VGGT places the whole
person 0.2–1.0 m off the wall they are climbing, and every downstream stage then
sees a body that floats in front of (or is buried inside) the scene. Fuse therefore
gates every clip on an independent witness — the SAM-3D init body, whose camera-frame
depth is metric from apparent size and does not scale with the clip factor:

- **Gate** (`tools/human_optim/depth_anchor.py`): per person, the median over valid
  frames of the unsigned distance from the four end-effector joint centres to the
  fused scene fires at > `--float-gate-m` (default 0.28 m — calibrated on the hard
  test set, where healthy rows top out at 0.225 and floaters start at 0.346), and a
  signed variant fires at burial < −0.10 m. Support guards (≥20 valid frames, ≥60%
  finite) block low-evidence clips.
- **Rescale**: from the exact per-frame correspondences, choose the scale ratio `r`
  closest to 1 such that a quarter of frames touch the wall (≤ 2 cm) without pushing
  the worst 5% deeper than 8 cm into it (the penetration loss stops looking past
  10 cm, so the optimizer must stay inside that authority). The chosen `r` is then
  verified against ground-truth distances in the candidate world (nearest-neighbour
  correspondences switch under rescale), backed off toward 1 if the burial budget
  breaks, and finally re-verified against the actually re-voxelized candidate scene —
  any failure vetoes the rescale and the baseline build ships instead. Multi-person
  clips rescale only when every person agrees (one world per clip).
- **The fired person's human cloud stays at the body-consistent scale** (it is
  per-frame consistent with the body to ±5% even on floaters), so the chamfer
  evidence, the init body and the wall all land in agreement; the init body keeps its
  camera-frame position under the rescale, so reprojection is strained by exactly
  zero.

`--body-anchor off` reproduces the pre-round-13 build byte-for-byte (modulo a known
≤1e-5 m CUDA voxel-centroid nondeterminism); on but not fired changes nothing except
adding the diagnostic keys.

## `geometry/transform.npz`

| key | shape | meaning |
|---|---|---|
| `scale` | `()` | the **effective** metric scale applied to everything (= `scale_base · rescale_r`) |
| `extrinsics` | `(T,4,4)` | world→cam OpenCV, translations at `scale` |
| `intrinsics_px_orig` / `intrinsics` | `(T,3,3)` | original-resolution / normalized VGGT intrinsics |
| `fov_x_deg`, `fov_y_deg`, `frame_indices`, `image_*`, `vggt_*` | | VGGT camera bookkeeping |
| `fps` | `()` | the source clip's real fps (consumers fall back to 30 when absent — pre-round-13 files) |
| `scale_method` | `()` | provenance string |
| `scale_base` | `()` | the human-ruler scale before the body anchor (absent = anchor never ran) |
| `rescale_r` | `()` | the applied body-anchor ratio (1.0 = not applied) |
| `depth_refuted` | `(P,)` | per climber: this person's VGGT human depth was refuted and the rescale applied |
| `anchor_gap_m` / `anchor_burial_m` / `anchor_support` | `(P,)` | gate diagnostics (median EE gap, signed burial, finite-frame fraction) |
| `anchor_pred_gap_q` | `(P,3)` | predicted post-rescale gap p05/p25/p50 |
| `anchor_true_touch` / `anchor_true_burial` | `()` | ground-truth touch-q25 / burial-q05 verified on the built world |

## `geometry/scene.npz`

`points (M,3)` metric world, voxel-merged static scene (dilated humans excluded);
`colors (M,3)`, `confidence (M,)` (per-voxel max VGGT depth-conf), `count (M,)`
(points per voxel), `voxel_size ()`.

## `geometry/human.npz`

Per-frame, unmerged climber points: `points (K,3)`, `colors`, `confidence`,
`frame_id (K,)`, `obj_id (K,)`, `frame_indices`. For a `depth_refuted` person these
points are backprojected at `scale_base` (body-consistent) and mapped through the
`scale`-consistent cameras — deliberately NOT at the scene's scale.

## Staleness

A fired rescale changes the clip's world wholesale: `kinematic_1/2`, `contacts_*`,
`uncertainty_*` and `dynamics` computed against the old geometry are stale and must
be re-run (fuse prints a warning listing them). Files without the new keys behave
exactly as before everywhere.
