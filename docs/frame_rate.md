# Frame rate: one rule for every time derivative

Clips come at 24 to 120 frames per second. Every time derivative of the pipeline follows one rule, so that a weight
means the same at every rate. The rule lives in `bvr.terms.motion.Time`.

## The rule

- The stride is `h = max(1, round(fps / 30))` frames: 1 at 24 to 44 fps, 2 at 60, 4 at 120. A derivative steps
  over `h` frames, so it always spans about one thirtieth of a second.
- A change over one stride is reported per reference step `tau0 = 1/30 s`: it is multiplied by `tau0 / tau`, where
  `tau = h / fps` is the stride's time. At 30 fps that factor is 1 and a difference is the plain difference of
  neighbouring frames.
- Order k reads frames `t, t + h, ..., t + k h`: velocity (k = 1), acceleration, jerk, snap (k = 4). The velocity of
  the pose is `kinematics.difference(pose[t], pose[t + h])`, the root's twist and three rotation coordinates per
  joint; points, forces and torques use the plain difference.
- Inverse dynamics at frame t takes central differences over `t - h`, `t`, `t + h`. The first and last `h` frames
  of a clip have no such stencil and count with the body at rest.
- A row counts only if every frame from its first to its last is tracked, the frames between the strides included;
  a contact term's row counts only where the contact is labelled on all of them. Rows that do not count carry no
  loss and are not in the mean.

The weights in `configs/bvr.toml` are therefore per change over 1/30 s, a fixed physical unit, and a 30 fps clip
gets the plain frame differences it was tuned on.

## Windows are seconds, counts are counts

Every window over time is given in seconds and converted where it is used: the keypoint and start-pose smoothing
(`smooth_window_s = 0.35`), the finger smoothing (`finger_smooth_s = 0.2`), the stillness window
(`[contact_gate].window_s = 0.35`), the shortest contact (`[contacts].min_span_s = 0.1`, `ceil(seconds * fps)`
frames), the merge gap (`merge_max_gap_s = 2.0`), the confidence's distance to a label change
(`transition_cap_s = 1.0`).

Two kinds of number stay frame counts. The raw detections of both contact witnesses must last
`[contact_gate].detect_min_frames = 3` frames at every rate: a contact at 60 fps first shows as a flicker of 3 to 5
frames that the readout welds afterwards, and a bar of 6 frames cut recall. And counts that are not time:
`[sam3].part_length`, `[vggt].max_frames`, `[gravity].num_frames`.

## Neighbouring frames at a stride above one

At a stride of `h > 1` no strided derivative ties frame t to frame t + 1: a 60 fps clip is two 30 fps trajectories
that share only their evidence, and they drift apart. The root then zig-zags by millimetres between neighbours
(1.3 to 3.8 mm on average at 60 fps, and 11 to 21 mm at 120 fps, in `kinematics_1` without a remedy).

The terms `neighbour_root` and `neighbour_joint` (`bvr.terms.motion.neighbour`) price the second difference of the
pose over neighbouring frames, in the rule's unit (times `(fps / 30)^2`). A real acceleration reads there as in the
strided term, so a weight well below that term's leaves the motion alone, while a zig-zag of `z` reads as
`4 z (fps / 30)^2` and is damped. At a stride of one frame the term is zero, so clips up to 44 fps are untouched.
The weights are 24 (root) and 48 (joints) in the root phase, `kinematics_1` and `kinematics_2`, and ten times that
in `kindyn_1`, whose root residual pulls on the pose. With them the root of the 60 fps example clips zig-zags
0.3 to 0.7 mm on average, about as little as before the rule. A delta that sets the eight weights to 0 turns the
terms off.

## What the rule changes, by rate

| clip rate | h | against the plain frame differences the weights were tuned on |
|---|---|---|
| 30 fps (29.97 included) | 1 | nothing (29.97 differs by under 3 %) |
| 24 and 25 fps | 1 | a difference term of order k weighs `(fps / 30)^(2k)`: at 25 fps 0.69 for velocity, 0.48 for acceleration, 0.23 for snap. Slower clips are smoothed less than with 30 fps weights on their frame differences |
| 60 fps | 2 | the same physical restraint as at 30 fps; the band above 30 Hz is left to the neighbour terms. The contact witness reads a 2-frame stencil |
| 100 to 120 fps | 3 or 4 | as at 60 fps |

Measured on the 13 annotated example clips: at 24 and 25 fps the rule costs 0.025 of force confidence and gains
0.006 to 0.013 of annotation F1. One 60 fps clip, `2nwG1Qa5k_c_0115`, loses 0.04 to 0.05 of annotation F1 on one
hand, through the witness's stride; the pooled F1 holds.
