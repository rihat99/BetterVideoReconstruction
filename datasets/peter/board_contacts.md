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
way (see [`contacts.md`](contacts.md)). The peter rigs use eighteen of them, nine
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

Everything goes to `<clip>/human_optim_gt_contact/`, beside the normal
`<clip>/human_optim/`, so the two runs can be compared clip by clip.

The stages themselves are untouched. Each clip gets a small directory of symlinks
at `<clip>/_gt_contact/` whose `human_optim` points at the variant directory and
whose inputs point at the real ones, so the stock stage CLIs write into the
variant tree without knowing it exists. It is the same trick the corpus driver
uses to fan a stage over the scene database.

## Running it

```bash
# the whole tree, four GPUs
for i in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$((i + 1)) uv run --no-sync python scripts/run_dataset.py \
    --dataset climb_wall_3 --variant gt_contact --shard $i/4 &
done
# one clip, one step
uv run --no-sync python scripts/run_dataset.py --dataset climb_wall_3 --variant gt_contact \
  --clips a_00_01_part1 --steps contacts_1
```

Every step is resumable: a step whose output already exists is skipped. A clip
that fails is reported and the batch moves on.

The thresholds live in `[boards]` in
[`datasets/peter/configs/human_optimization_peter.toml`](configs/human_optimization_peter.toml);
every loss weight is shared with the normal run, so the two trees differ by their
contact labels and nothing else.

## The artifact

`contacts_<1|2>.npz` is the same schema-2 file the pipeline's own contact stage
writes, so `kinematics_2` and `kindyn_1` read it unchanged. Beyond the labels it
carries what the boards said, which is what makes it auditable:

| field | what it is |
|---|---|
| `frame_contact (1, N, F)` | the labels the downstream stages use |
| `frame_rescued (1, N, F)` | spans labelled although nothing was near the wall |
| `frame_scene_dist_m (1, N, F)` | how far each point was from the wall |
| `limb_contact (1, N, 4)` | the boards' own per-limb on/off |
| `limb_force_world (1, N, 4, 3)` | the measured force vectors |
| `contact_label_method` | `boards-v3`, the same string the three-camera GT chain stamps |

## The same stage on the Parkour clips

The peter rigs are not the only instrumented data we have. The LAAS Parkour
dataset films twenty-eight jumps and vaults with the feet landing on floor force
plates and the hands on a sensed bar, so each of the same four limbs has its own
measured 3D force, one row per video frame. That is the same measurement the
peter boards make, so the same stage runs on it.

Only the file shape differed: the import stores a 6D wrench plus a mask saying
which samples the rig actually caught.
[`datasets/parkour/parkour_sensor_forces.py`](../parkour/parkour_sensor_forces.py) writes
the force part of it as the `sensor_forces.npz` the stage reads, and then the
chain is one command:

```bash
uv run --no-sync python datasets/parkour/parkour_sensor_forces.py
CUDA_VISIBLE_DEVICES=6 uv run --no-sync python scripts/run_dataset.py --dataset parkour --variant gt_contact
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

## What it showed on climb_wall_3

All 68 single-view clips, both runs, compared against the same boards
(`datasets/peter/compare_gt_contact.py`).

The estimated labels turn out to be decent: pooled against the boards they score
precision 0.89, recall 0.93, F1 0.91 — they over-claim contact slightly more
often than they miss it. So there was never a huge amount of label error to
remove, and pooled over everything the measured labels buy a modest amount:
limb force error 80.1 → 72.0 N, correlation 0.65 → 0.71, load-share error 10.8 →
10.0 pp.

The pooled number hides the actual result, though, and the `rescued` flag is what
splits it. Across the tree **15.6 % of measured contact labels are rescued** —
the board says the limb was loaded, and no point of that limb was within 5 cm of
the reconstructed wall. Nine clips have none at all; thirteen are more than half
rescued, with median gaps of 10–29 cm (mostly the 32° overhang clips).

Split on that, the two halves say opposite things:

| | limb MAE [N] | corr | share err [pp] | root resid [BW] |
|---|---|---|---|---|
| **22 clips, body on the wall** — estimated | 67.0 | 0.761 | 9.27 | 0.058 |
| **22 clips, body on the wall** — measured | **53.4** | **0.867** | **7.19** | 0.076 |
| **13 clips, body off the wall** — estimated | 85.9 | 0.588 | 12.38 | 0.093 |
| **13 clips, body off the wall** — measured | 87.8 | 0.489 | 13.09 | 0.171 |

Where the reconstruction has the body in roughly the right place, perfect
contacts are worth about 20 % of the limb force error and lift the correlation
from 0.76 to 0.87. Where it does not, they make things worse: `kinematics_2` pulls
the limb onto a wall patch that is not the one it was really on, and `kindyn_1`
then solves forces on a pose that has been bent to fit a true label.

Two things follow. First, on these clips contact detection is not the bottleneck —
the single-view reconstruction geometry is, and no amount of label quality fixes
a hand that is 29 cm from where it should be. Second, the root residual gets
*worse* with true labels (0.068 → 0.095 BW pooled). That is not a regression: our
estimated labels over-claim contact, and every spurious contact is one more handle
the solve can use to absorb residual it cannot otherwise explain. A lower residual
bought that way is flattery, not physics.

## What it showed on Parkour

All 28 clips, both runs, against the plates and the bar.

Here the estimated labels are *not* decent, and the failure is one-sided. Pooled
they score precision 0.69 and recall 0.97 — the pipeline almost never misses a
real contact, and constantly invents ones that were not there. Split by limb, it
is the feet:

| limb | measured frames | claimed | precision | recall |
|---|---|---|---|---|
| left hand | 1333 | 1600 | 0.83 | 0.99 |
| right hand | 1145 | 1404 | 0.80 | 0.98 |
| left foot | 855 | 1357 | 0.62 | 0.98 |
| right foot | 547 | 1067 | **0.46** | 0.89 |

A parkour foot is near the ground and moving slowly a lot of the time without
carrying any weight, which is exactly the situation the near-and-still test
cannot tell from a real landing. The right foot is claimed in contact roughly
twice as often as the plates say.

That has a clean price, and removing it a clean payoff:

| | limb MAE [N] | corr | share err [pp] | total MAE [N] | root resid [BW] |
|---|---|---|---|---|---|
| Estimated contacts (video) | 92.3 | 0.629 | 16.23 | 110.6 | 0.086 |
| Measured contacts (boards) | **47.5** | **0.851** | **5.48** | 110.6 | 0.130 |

The per-limb force error halves and the load-share error drops by two thirds,
while the **total** force is identical to within 0.1 N. That is the signature of
a distribution error, not a magnitude error: the solve always knew how hard the
body was pushing, and was putting a large part of it through the wrong limb.

Unlike climb_wall_3, the split by `rescued` does not reverse the result. No
Parkour clip is more than 25 % rescued (climb_wall_3 had thirteen above 50 %),
so every clip here is a clip whose body is roughly in the right place, and all of
them gain — the five worst-placed go 82.5 → 46.9 N, the fifteen best 104.5 →
48.1 N. The single-view geometry on these clips is good enough that contact
detection really is the bottleneck, which is the opposite of what the climbing
wall said.

The root residual rises with true labels here too (0.086 → 0.130 BW), for the
same reason as on the wall: spurious contacts are extra handles for absorbing
residual, and a low residual bought that way is flattery, not physics.

## Shared with the ground-truth chain

The reading itself lives in
[`bvr/human_optim/boards.py`](../../bvr/human_optim/boards.py) and is used by
two callers: this single-view stage, and the three-camera ground-truth chain in
`datasets/peter/gt/fit_multiview.py`, which has always labelled from the boards.
One implementation, so the two cannot drift into disagreeing about what the
sensors said.
