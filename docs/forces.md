# Forces: pose and contact forces solved together

The step `kindyn_1` (`bvr/stages/solve_dynamics.py`) fits every person's trajectory and the forces at the labelled
contact frames as one problem, so the physics can reshape the motion. It reads `kinematic_2.npz`, the labels of
`contacts_2.npz` and gravity.

## The physics

Inverse dynamics (BetterRobot, on BetterHuman's body) gives the generalised force a body needs to move the way it
does:

    torques = M(q) a + b(q, v) + g(q) - sum over contact frames of J^T f

The first six rows belong to the root, which has no motor: what is left there once the contact forces `f` are counted
is the root residual, zero for a motion its contacts explain. The other rows are the joint torques. Velocity and
acceleration are central differences over the stride ([`frame_rate.md`](frame_rate.md)); a frame beside a tracking
gap does not count, and the first and last frames count with the body at rest.

A contact frame carries force only where `contacts_2` labels it, and only the way its skin faces: the force is a
non-negative combination of the four edges of a friction pyramid about the negated skin normal
(`[friction].half_angle_deg` = 45), so skin pushes and does not pull. The pyramids stand where the starting pose puts
the skin and stay fixed during the solve: a pyramid that turned with the pose would be a target the pose could rotate
to make force cheap. The variable is an activation per edge in body weights,
`force = sum over edges of softplus(weight * activation) * edge` newtons. With the variable in newtons BetterRobot's
L-BFGS does not get going: its first step moves a force by about 1e-4 N, and the change of the gradient is lost in
float32 rounding.

Mass comes from BetterHuman's default densities: the neutral SMPL-X body at mean shape weighs 77.4 kg. Compare forces
across bodies in body weights (`total_mass * 9.81`).

## Two solves

Both are BetterRobot's L-BFGS, people as the batch:

| solve | steps | what moves | weights |
|---|---|---|---|
| `forces` | up to 200, history 20, ends once no gradient entry is above `gradient_tolerance` | the forces alone, at the held pose | the 12 dynamics terms of `[stages.kindyn_1.weights]` |
| `joint` | 125 | pose and forces under one line search, with the scene | the whole table |

The first is there because forces start at zero, where the root residual says the body is in free fall and its
gradient on the pose means nothing. At a held pose inverse dynamics runs once and the torques are affine in the
forces, so that solve is a problem of its own with the activations as its one variable. The joint solve is stopped,
not converged: left to run on it keeps trading image evidence for less joint torque, and the body slides off its
depth. The step count is therefore a setting with an effect: 100 steps leave a 24 fps example clip's body 37 mm from
the result before the rebuild instead of 50 mm, and cost the wild clips 0.014 of force confidence.

A person without contact labels has no dynamics term and is fitted kinematically, with no forces.

## The terms

The kinematic terms of [`fitting.md`](fitting.md) but `chamfer` and `contact_acquisition`, and twelve more. `R` is the
number of frames whose inverse dynamics count and `W` the body weight in newtons; every dynamics term is divided by
`2 R W^2`, so a weight means the same for every body and clip length.

| term | sum over counted frames of |
|---|---|
| `root_force`, `root_torque` | the squared root residual, linear rows and angular rows |
| `force_magnitude` | `|f|^2` |
| `force_rate_*`, `force_acceleration_*`, `force_jerk_*` | the squared first, second, third strided difference of each held contact's force, split `_magnitude` along and `_direction` across the window's mean force direction |
| `torque_magnitude`, `torque_rate`, `torque_acceleration` | the squared joint torques and their first and second differences, each joint group weighted by `[torque_multipliers]` (200 N m over the group's maximum voluntary torque) |

## Force confidence

How far to trust a body's forces on a frame (`bvr.contacts.confidence.force_confidence`), in `[0, 1]`. Contact forces
are solved for the whole body at once, so trust is a property of the body in a frame, not of one force. Three things
each cost confidence along `1 / (1 + (x / x*)^2)`, and the result is their product:

- the residual force after the solve, `x*` = 0.5 body weights: where the solve could not balance the motion, what it
  did assign is an allocation of error;
- the residual torque, `x*` = 1.5 body weights times 0.2 m;
- total force beyond 3 body weights, `x*` = 1 body weight: that explains reconstruction error, not motion.

It is 0 on untracked frames. `force_confidence_clip` is its mean over each person's tracked frames. Read forces
through it: on clips filmed at 100 fps and more, force spikes remain on people the camera hardly sees, and their
confidence is low there.

## Output

`human_optim/kindyn_1.npz`, format `bvr-kindyn`: a body file ([README](../README.md)) with `drift_pre` and
`auto_labels` passed on, and

| key | holds |
|---|---|
| `frame_forces (P, N, 35, 3)` | the force at each contact frame, world, newtons |
| `frame_contact (P, N, 35)` | where a force was allowed: the labels of `contacts_2` on tracked frames |
| `contact_forces_world (P, N, 52, 3)` | the forces summed onto each frame's parent joint; `contact_frame_parents (35,)` |
| `joint_contact (P, N, 52)` | the labels folded onto joints |
| `joint_torques (P, N, 159)` | inverse dynamics with the forces, the six root rows first; zero where the stencil reaches an untracked frame |
| `base_wrench (P, N, 6)` | the root residual in the root's frame, zero where the frame does not count |
| `torque_mag (P, N, 52)` | one size per joint: the root's six rows, then three rows per joint |
| `joints_world (P, N, 52, 3)` | the joint positions |
| `total_mass (P,)`, `gravity_world (3,)` | the bodies' mass in kg and the clip's gravity |
| `force_confidence (P, N)`, `force_confidence_clip (P,)` | as above |

## Measured

On the 13 annotated example clips, from the same upstream files as the code before the rebuild: support (total force)
0.89 body weights on average, root residual 0.15 body weights at the median, force confidence 0.84. The step took
93 s for the 13 clips in one process on a shared GPU, at most 450 MB of GPU memory.
