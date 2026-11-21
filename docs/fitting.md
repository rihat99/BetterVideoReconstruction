# Fitting the bodies: kinematics_1 and kinematics_2

The steps `kinematics_1` and `kinematics_2` (`bvr/stages/refine_kinematics.py`) fit every person's body trajectory
to the video. `kindyn_1` adds forces to the same problem ([`forces.md`](forces.md)).

## One problem per clip

A fit is a BetterRobot `optim.Problem` (`bvr.terms.objective.kinematics_problem`). Its batch is the people of the
clip and its variable is the trajectory `q (P, T, 211)`: per frame the root pose in the world and 51 joint
quaternions. Every term returns one loss per person, `(P,)`, a mean over that person's tracked frames, so the people
of a clip are solved together but independently. Cached values are computed at most once per evaluation: `pose`
(`q` with unit quaternions), `body` (BetterHuman's `HumanState`: joints, vertices, contact frames), `frames` (the 35
contact-frame poses), the strided derivatives `velocity` to `snap` ([`frame_rate.md`](frame_rate.md)), and
`distance` (each contact frame against the scene).

Every term is registered with weight 0. A phase passes its stage's weight table of `configs/bvr.toml`
(`[stages.<name>.weights]`), so the table is the objective: a term the table does not name is off, and a name that is
no term raises. What a phase holds fixed is data, rebuilt from the pose when a scene phase starts: the scene points
near each body (`scene`), the contact labels (`labels`) and, in `kinematics_2`, the wall patches (`patches`).

The solvers are Adam (learning rate `[optimizer].learning_rate`, cut by `learning_rate_decay` at one half and three
quarters of a phase) and BetterRobot's L-BFGS, which keeps one memory and one line search per person (`bvr/fit.py`).
Each phase logs every person's loss and terms.

## kinematics_1

It starts from the scale step's pre-fit bodies, or the converter's where there is no pre-fit. One shape per person
(the mean of their `betas` over tracked frames); an untracked frame holds the nearest tracked pose. The fingers keep
the converter's hand pose, smoothed over `finger_smooth_s`, and are frozen in every phase (`freeze_fingers`).

| phase | solver, steps | what moves | weights |
|---|---|---|---|
| `root` | Adam, `[stages.root].iterations` = 150 | the root pose alone | `[stages.root.weights]` |
| `full` | Adam, 300 | all joints | `[stages.kinematics_1.weights]` without the contact terms |
| `refine` | Adam, 200 | all joints, with the scene | `[stages.kinematics_1.weights]` |
| `polish` | L-BFGS, 30 | the same | the same |

Between `full` and `refine` contacts are detected once on the placed body, before any contact term has acted, and
frozen: a contact frame touches where it is near the scene and still (`[auto_contact]`, `[contact_gate]`;
[`contacts.md`](contacts.md)). The file keeps what was measured then: `drift_pre (P, N, 35)`, how far each contact
frame moved across the stillness window, and `auto_labels` (1 for contact, -1 for none). The contact steps read
stillness from `drift_pre` and never measure it again: a drift measured after contact terms held a frame still would
confirm the labels that held it.

## kinematics_2

It starts from `kinematic_1.npz` and the labels of `contacts_1.npz`, and runs three Adam phases, `root` (100),
`full` (200) and `refine` (250), with the scene on throughout. In each phase every labelled contact frame is pulled
to the wall the camera sees at that frame's own pixel (`contact_acquisition`), which seats a body that hovered in
front of its wall. `drift_pre` and `auto_labels` pass through unchanged.

## The terms

`g(r2, c) = r2 / (c^2 + r2)` is the robust cost of a squared pixel error: a square near 0 that levels off at 1, so a
wrong detection cannot drag the pose. A difference is per reference step of 1/30 s.

| term | per person | unit |
|---|---|---|
| `reprojection_keypoints` | Sapiens keypoints against the joints and landmarks they name, `sum s g(r2, 40 px) / sum s`, `s` the score (0 below `score_thresh`); the tracks smoothed over `smooth_window_s` first | none |
| `reprojection_start` | the 22 body joints against where the start pose projects them, `g(r2, 30 px)`: holds the body on the image while the depth terms move it along the rays | none |
| `chamfer` | the visible body surface against the person's depth points, both ways; a distance counts as `0.25 tanh(d / 0.25)`; hands and feet weigh more (`emphasis_*`), smeared frames less | m |
| `pose_prior`, `orientation_prior` | the mean of `1 - <q, q_start>^2` over the 51 joints, or over the root | none |
| `acceleration_*`, `jerk_*`, `snap_*` | the mean square of the strided second, third and fourth difference of the pose; `_root` is the root's 6 coordinates, `_joint` the other 153 | rad^2 or m^2 |
| `neighbour_root`, `neighbour_joint` | the acceleration between neighbouring frames, zero at a stride of 1 ([`frame_rate.md`](frame_rate.md)) | rad^2 or m^2 |
| `radial_velocity` | the squared speed of the joints along their viewing rays, where one camera sees worst | m^2 |
| `contact_velocity` | a labelled contact frame stays put: its squared step plus `angular_weight` times its turn | m^2 |
| `penetration` | contact frames stay out of the scene: `Huber(relu(-s - margin_m), huber_m)` of the signed distance `s` | m^2 |
| `contact_attraction` | a labelled contact frame hovering within `band_m` comes onto the scene: the mean of `s` | m |
| `contact_acquisition` | `kinematics_2`: each labelled frame comes to its own wall patch, a pseudo-Huber with knee `delta_m` | m^2 |

The weights per stage are the tables of `configs/bvr.toml`. `kinematics_1` uses `[auto_contact]` for its attraction
band, `kinematics_2` and `kindyn_1` use `[contact_attraction]`. Nearest neighbours and visibility carry no gradient:
a term moves the body, never its match.

## What it costs

The 13 annotated example clips, from the same inputs, one process per step on a shared GPU: `kinematics_1` 494 s
in all (22 to 55 s a clip) and `kinematics_2` 206 s (13 to 22 s a clip), at most 1.8 GB of GPU memory. Half of an evaluation is the
chamfer's two nearest-neighbour searches and its visibility pass; the body forward is about 9 ms.
