# Fusion: the point clouds

The step `fuse` (`bvr/stages/fuse_geometry.py`) turns VGGT's depth maps into one static scene cloud and per-frame
human points. VGGT gives every frame a camera and a depth map that agree across the clip up to one unknown factor;
fuse builds the clouds in that unscaled world, and `scale` measures the factor next ([`scale.md`](scale.md)).

## Per frame

1. The depth loses its flying pixels at depth edges (`edge_rtol`).
2. The SAM 3 masks split it. The human points are the pixels of every person SAM 3D Body fitted, tagged with the
   person (`obj_id`) and the frame (`frame_id`). The scene is the rest, less a band of `human_dilate` pixels around
   every tracked person and less its least confident `conf_drop_q`.
3. Both are back-projected with the frame's camera.

## Once per clip, the scene

1. Voxel merge. The voxel is `voxel_px` pixel footprints wide at the median scene depth, so its size holds in a world
   of unknown scale. The merge runs twice, the second time on the grid shifted by half a voxel: a surface that lies
   on a cell face leaves two copies after one pass.
2. Hidden points go. VGGT's depth of one wall differs by a few centimetres between frames, which leaves a slab whose
   back no camera sees; a point stays only if it is in front in some frame.
3. Connected clusters of fewer than `min_scene_voxels` voxels go: floating fragments.
4. Every point gets a normal, from its `normal_k` nearest neighbours, turned towards the nearest camera. The contact steps and the scene terms use these.

A shot counts as locked off when the scene slides less than `static_motion_max` of the image width across the frames
(`bvr.geometry.camera_motion`); `transform.npz` stores `static_camera` and `camera_motion`.

## Output

`geometry/fused/`, written as one folder:

| file | holds |
|---|---|
| `transform.npz` | `extrinsics (T, 4, 4)` camera from world, `intrinsics_px_orig (T, 3, 3)` in pixels of the video, normalised `intrinsics`, `fov_x_deg`, `fov_y_deg`, `frame_indices (T,)`, image and VGGT sizes, `fps`, `static_camera`, `camera_motion`, `scale = 1`, `metric = false` |
| `scene.npz` | `points (M, 3)`, `colors (M, 3)` uint8, `confidence (M,)` (the largest of its voxel), `normals (M, 3)`, `count (M,)` depth pixels merged, `voxel_size` |
| `human.npz` | `points (n, 3)`, `colors`, `confidence`, `frame_id (n,)` (a row of the transform), `obj_id (n,)`, `frame_indices (T,)` |

`T` is the number of frames of the clip: VGGT sees every frame, and the `vggt` step stops on a clip of more than
`[vggt].max_frames`. `frame_indices` is `0..T-1`, kept for the readers of older trees. The step holds the per-frame clouds on the GPU until the merge (5 to 13 GB on the example
clips).
