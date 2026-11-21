# Scale: making the world metric

VGGT reconstructs the cameras and the depth right up to one unknown factor `s` per clip: nothing in the video says
whether the wall is three metres away or thirty. The SMPL-X bodies of the converter have metric size and sit in the
same cameras, so over a person's pixels the body's depth and the cloud's depth differ by `s`. The step `scale`
(`bvr/stages/estimate_scale.py`) reads `s` there, seats the bodies on their depth, and writes the metric world.

## The ruler

Per person and tracked frame, the converter's body is posed with that frame's own shape. Each vertex that faces the
camera looks up the fused human point at its pixel; the least confident `conf_drop_q` drop, and the
confidence-weighted median of `log z_body - log z_cloud` is the frame's reading. A frame with fewer than
`min_inlier_verts` such vertices has none. The readings of all people pool into one `s`: frames beyond `mad_k`
median absolute deviations drop, and the rest are averaged by weight.

- Log ratios, so that a factor too large and one too small weigh alike.
- Medians inside a frame, because depth has gross outliers at mask edges and on smeared limbs.
- One scale per clip: VGGT has one by construction, and the per-frame readings are measurements to pool.

## The pre-fit

The converter's bodies sit a little off their depth, and that offset is in the ruler's `s`. With `[scale].prefit`
(the default) the bodies of the people with keypoints are fitted to their depth points and keypoints, and `s` is read
again from the fit, for up to `rounds` rounds of `iters` Adam steps. The rounds stop once `s` moves less than 0.1 %.

- In the first round each person's `log s` is a variable of the fit: depth points cannot tell a nearer body from a
  smaller world, so the keypoints set the depth and the scale follows. The later rounds hold every person at the
  pooled `s`.
- After a round, each depth point's nearest visible vertex votes its log ratio, weighted by the robust kernel's own
  weight for that match, and the votes pool as the ruler's do. Reading through the fit's own matches gives the
  rounds a fixed point.
- A person's shape is the median of their `betas` and stays fixed: the body's size is the ruler. The fingers stay
  too.

The objective, one loss per person, weighted by `[scale.weights]`:

| term | what it asks |
|---|---|
| `chamfer` | every depth point near its nearest visible vertex, Geman-McClure at `gm_chamfer_m`. One-sided: the cloud sees only the front, and pulling hidden vertices onto it folds limbs. It is measured in the cloud's units (body and kernel divided by `s`); in metres the cloud's noise grows with `s`, and the term would shrink the world whatever the evidence |
| `reprojection` | the Sapiens body and face keypoints, Geman-McClure at `gm_reproj_px` |
| `pose`, `orientation` | the joints and the root near the converter's rotations |
| `translation` | the root near the converter's position (m^2): holds the depth where the image says little |
| `start_velocity` | the joints move as the converter's do, per reference step ([`frame_rate.md`](frame_rate.md)) |

## Depth smear

When a person moves fast, VGGT can streak their depth along the view ray into a streak metres long. The smear of a
frame (`bvr.geometry.depth_smear`) is the extent of the person's points along the ray over their largest extent
across it: well below 1 for a real body, several for a streak. A frame above 2 weighs `(2 / smear)^p` (about 0.4 at
2.5, 0.2 at 3) in every reading of `s` and in the pre-fit; a frame at or below 2 keeps weight 1. The median smear
over people and frames is stored as `smear_median` for the quality flag of the contact steps
([`contacts.md`](contacts.md)).

## Output

Points, camera translations and the voxel size of `geometry/fused/` times `s`, in metres:

| file | holds |
|---|---|
| `geometry/transform.npz` | the fused transform with metric extrinsics, `scale = s`, `metric = true` |
| `geometry/scene.npz`, `geometry/human.npz` | the clouds in metres |
| `geometry/scale.npz` | `scale`, `scale_ruler` (the ruler's reading), `log_scale`, `scale_mad`, `n_frames_used`, `n_people`, `method` (`prefit-dual` or `human-log-ratio`), `per_frame_scale` and `per_frame_weight (P, T)` of the last reading, `smear_median` |
| `geometry/prefit.npz` | the classic SMPL-X fields of `sam3d/smplx_params.npz`, the fitted people's tracked frames replaced by the fit; `valid_mask`, `object_ids`, `scale` |

`kinematics_1` starts from `prefit.npz` when it exists, from the converter otherwise. A large gap between `scale`
and `scale_ruler` means the pre-fit moved the world far from the converter's opinion.

## Limits

- The ruler sizes the person, not the wall. If VGGT puts the person's depth at the wrong distance from the wall,
  `s` is still clean but the body, seated on its own depth, hovers off the wall.
- Where the keypoints are weak (blur, occlusion), the pull towards the converter's position is what remains, so a
  depth bias of the converter can leak into `s`.
- A shape bias of the converter is invisible: if it makes the person 5 % too tall, the world is 5 % too large.
- A static camera has no parallax, so its human depth can sit at the wrong depth; the force-rig recipe turns the
  pre-fit and the kinematic chamfer off (`bvr_datasets/peter/configs/peter.toml`).
