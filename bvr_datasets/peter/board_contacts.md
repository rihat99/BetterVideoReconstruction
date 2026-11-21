# Contacts measured by force boards, instead of estimated from video

Everywhere else in this pipeline, contact is a guess. The stage looks at a body
part, asks whether it is near the scene and holding still, asks whether the
physics needs it to be carrying load, and decides. It is a good guess, but it is
a guess, and a wrong one quietly poisons everything downstream: a limb the solve
thinks is on the wall gets given force it never had.

On the peter climbing rigs we do not have to guess. Each of the four limbs
climbs on its own hold, and each hold sits on its own three-axis force board. The
board reads the force that limb is pushing with, about a hundred times a second.
So for those clips contact is **measured**, and we can run the rest of the chain
on the truth and see what the guessing was costing us.

That is what this stage is for.

## What the boards know, and what they don't

A board knows **which limb** and **when**. It does not know **which part** of the
hand or foot is touching — it is one sensor under the whole hold.

That matters, because the optimization does not put contact on limbs. It puts
contact on **contact frames**: named points on the skin, each facing a particular
way (see [`docs/contacts.md`](../../docs/contacts.md)). The peter rigs use eighteen of them, nine
per side — palm, palm base, fingertip, thumb for a hand; heel, both balls of the
foot, big toe, top of the toe for a foot.

So there is a gap to close: the board says *left foot is loaded*, and something
has to decide *which of that foot's five points is the one on the hold*. Closing
that gap is the whole job of this stage, and it takes two steps.

## Step 1 — when is a limb loaded

Take the measured force, take its size in newtons, and call the limb in contact
while that size is large enough.

Using a single threshold would make a limb flicker on and off whenever the force
wobbled around it, so there are two: contact **starts** at 50 N and only **ends**
below 30 N. A grip that eases off to 35 N stays a grip. This is called
hysteresis, and it is the same trick the estimated-contact stage uses on
distance.

A board sample that is missing or corrupt reads as zero force, which means no
contact — the honest reading when the sensor did not speak.

## Step 2 — which of the limb's points is on the hold

Now we know the left foot is loaded from frame 100 to frame 260. Which of its
five contact frames carries the label?

Pose the body, and for each of the foot's five points measure how far it is from
the fused scene cloud — the reconstructed wall. A point that sits within 5 cm of
the wall is a candidate.

The tempting thing is to decide this per video frame: label whichever points are
near the wall right now. That turns out to be a bad idea. The chosen set jitters
from frame to frame, and the downstream terms that pull a contact onto the wall
and hold it still end up chasing a moving target.

So the decision is made **once per loaded span**, and it holds for the whole
span: a point gets the label if it was within 5 cm of the wall for at least a
third of the time the limb was loaded. Steady points win; a point that brushed
past for two frames does not.

If no point of the limb qualifies — the whole foot was reconstructed well off the
wall — the label falls back to whichever point came closest, and that span is
flagged `rescued` in the output. Those flags are worth reading: they are exactly
the places where the reconstruction disagrees with a sensor that cannot be wrong.

## The chain

`kinematic_1` is shared with the normal run and never recomputed — it is the
visual fit, and it never sees a contact label, so there is nothing for the boards
to improve. From there:

| step | what it does |
|---|---|
| `contacts_1` | the boards, read on the `kinematic_1` pose |
| `kinematics_2` | the usual contact-patch repair, now on measured labels |
| `contacts_2` | the boards again, re-read on the repaired pose |
| `kindyn_1` | pose and per-contact forces solved together, as usual |

The boards are measured twice on purpose. Which limb is loaded never changes — the
sensor said so. But *which point* of that limb is on the hold is a question about
the body, and after `kinematics_2` has repaired the body, the answer can be
better. This mirrors what the normal pipeline does with its own two contact
readouts.

**Forces are not supervised.** The boards decide contact and nothing else. The
solved forces stay an independent prediction, which is the only way force error
measured against those same boards means anything.

## Where it lands

The measured run is a second run root beside the normal one. It reads what it does
not compute (the videos' model outputs, the geometry, `kinematic_1.npz`, the boards'
`sensor_forces.npz`) from the normal run through `--upstream`, and writes its own
`<clip>/human_optim/`, so the two runs can be compared clip by clip.

## Running it

```bash
C=bvr_datasets/peter/configs/peter.toml
uv run python bvr_datasets/peter/sensor_forces.py --rig climb_wall_3 --save-dir <run>
uv run python -m bvr_datasets.peter.chain --measured --source <videos> --save-dir <run 2> --upstream <run> --config $C
# one clip, one step
uv run python -m bvr_datasets.peter.chain --measured --source <videos> --save-dir <run 2> --upstream <run> --config $C \
  --clips a_00_01_part1 --steps contacts_1
```

`--skip-existing` resumes: a step whose manifest still vouches for it is skipped. A
clip that fails is recorded and the batch moves on.

The thresholds (50 N in, 30 N out, 5 cm) are constants of
[`bvr_datasets/_rigs.py`](../_rigs.py); every loss weight is shared with the normal
run, so the two runs differ by their contact labels and nothing else.

## The artifact

`contacts_<1|2>.npz` is the same schema-2 file (format `bvr-contacts`) the pipeline's
own contact step writes, so `kinematics_2` and `kindyn_1` read it unchanged. Beyond the labels it
carries what the boards said, which is what makes it auditable:

| field | what it is |
|---|---|
| `frame_contact (1, N, F)` | the labels the downstream stages use |
| `frame_rescued (1, N, F)` | spans labelled although nothing was near the wall |
| `frame_scene_dist_m (1, N, F)` | how far each point was from the wall |
| `limb_contact (1, N, 4)` | the boards' own per-limb on/off |
| `limb_force_world (1, N, 4, 3)` | the measured force vectors |
| `contact_label_method` | `boards-v3: …`, the rule above in words |

## The same stage on the Parkour clips

The peter rigs are not the only instrumented data we have. The LAAS Parkour
dataset films twenty-eight jumps and vaults with the feet landing on floor force
plates and the hands on a sensed bar, so each of the same four limbs has its own
measured 3D force, one row per video frame. That is the same measurement the
peter boards make, so the same stage runs on it.

Only the file shape differed: the import stores a 6D wrench plus a mask saying
which samples the rig actually caught.
[`bvr_datasets/parkour/sensor_forces.py`](../parkour/sensor_forces.py) writes
the force part of it as the `sensor_forces.npz` the stage reads, and then the
chain is one command:

```bash
uv run python bvr_datasets/parkour/sensor_forces.py --save-dir <run>
uv run python -m bvr_datasets.peter.chain --measured --source <videos> --save-dir <run 2> --upstream <run> \
  --config bvr_datasets/peter/configs/peter.toml
```

Two things about this rig are unlike the climbing boards. Both are written into
the file, so no reader has to know which dataset it is looking at:

- **A limb can be unmeasured.** A foot that lands beside the plate is not a
  released foot — it is a foot nobody watched. Those samples are stored as `NaN`
  (3.6% of limb-frames here, nearly all of them the right foot) and drop out of
  every number, rather than counting as "not touching".
- **The vectors live in the motion-capture world, not ours.** The two worlds do
  not share a compass direction, so force *sizes* compare and force *directions*
  do not. The file says `world_aligned = False`, and the comparison prints a
  blank angle column instead of a meaningless number.

Neither touches the labels: whether a force is big enough to count as contact
does not depend on which way the world is facing.
