# Per-joint depth fit (the "spaghettization" detector)

VGGT depth sometimes stretches a person along the camera rays. A hand or a leg
then reads much longer or farther than a real limb can be. The fitted SMPL-X
body has fixed human proportions, so it cannot follow. That limb ends up away
from the wall it really touches. We call this failure spaghettization.

This metric measures the problem, per joint and per frame. It does not fix
anything. It answers one question: how well does the fitted body match the
human depth cloud around each joint?

## How it works

Computed inside `estimate_contacts` (`tools/human_optim/depth_fit.py`), per
person, per frame:

1. Take the person's human cloud points for the frame (`geometry/human.npz`).
   Each point came from one depth pixel, so projecting it with that frame's
   camera recovers its pixel.
2. Project the posed body vertices with the same camera. Split the image into
   small cells (`cell_px`, default 8 px). In each cell keep the vertex closest
   to the camera. That vertex is the body's front surface there, and it belongs
   to a joint (nearest-joint assignment, the same map the contact labels use).
3. Match every cloud point to the nearest front vertex within `cell_px` pixels.
   The pixel cap matters: without it a thin limb could claim pixels far outside
   itself and read background depth as its own error (we measured metre-scale
   false spikes on clean clips before the cap).
4. For every matched point the residual is the camera-depth difference
   `z_cloud − z_mesh`. Positive means the cloud lies beyond the body — the
   stretch direction.
5. Per joint, take the median residual over its matched points. Fewer than
   `min_points` matches (default 5) gives NaN — no data, not zero error.

Matching happens in 2-D on purpose. The body is well aligned in the image even
when the depth is wrong, so pixels are the one reliable correspondence. It also
separates body parts that are close in 3-D, like a foot next to a hand.

## The saved fields

Written to `human_optim/contacts_<tag>.npz`, all `(P, N, J)` float32:

| Field | Meaning |
|---|---|
| `joint_depth_gap_cm` | signed median depth gap in cm; NaN = too few points |
| `joint_depth_points` | matched cloud point count |
| `joint_depth_trust` | `exp(−(median log(z_cloud/z_mesh) / 0.05)²)` in [0, 1] |

Trust uses the log depth ratio, not centimetres. A 10 cm error at 10 m is
nothing; at 2 m it is serious. The log ratio scales that away. Roughly: a 5%
depth error gives trust 0.37, a 10% error gives 0.02. No data gives trust 1.0 —
the absence of a witness leaves other evidence (motion) unchanged instead of
erasing it.

## What it catches, and what it cannot

- It catches limb-level stretch: the cloud runs away along a ray, the body
  cannot follow, and the gap shows exactly which joint and when.
- It does NOT catch a whole-body floater. There the whole cloud is displaced
  and the body aligns to its own cloud, so the local fit looks fine. That
  problem needs its own mechanism.
- Thin parts (fingers) cover few pixels, so their medians are the noisiest.
  Read finger spikes together with `joint_depth_points`.

## Where it is used

**Currently unwired (contact-frames era).** The per-joint contact stage that
consumed this metric was replaced by the contact-frame detector (see
[`contacts.md`](contacts.md)), and the viewer's "depth trust" layer was retired
with it. The module (`tools/human_optim/depth_fit.py`) remains for the frozen
per-joint stages. Historical consumers:

- **Viewer** (retired): the "depth trust (spaghetti)" layer colored each joint
  sphere from calm green (trusted) to bright magenta (spaghettized).
- **Confidence** (frozen): `joint_depth_trust` multiplied the no-contact motion
  evidence (see [`contact_confidence.md`](contact_confidence.md)).

Knobs live in `[depth_fit]` in `configs/human_optimization.toml`: `cell_px`,
`min_points`, `trust_scale_log`.
