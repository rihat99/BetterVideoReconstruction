# Contact points on the mesh — investigation

**Status: investigated, not implemented** (2026-07-23). This doc records what we
found when we asked: can contact move from *joints* to a small set of named
points on the body *surface*?

## The idea

Today every contact — the detected labels, the kinematic contact losses, and the
dynamics force solve — attaches to a skeleton joint. But real contact happens on
the skin: the heel and toes of a foot, the palm and fingertips of a hand. The
proposal is to pick roughly twelve mesh vertices as named contact points — three
per foot (heel and two toe-side points), one per knee, one per elbow, two or
three per palm — and hang the whole contact machinery off them instead of the
joints.

What that buys:

- **Real lever arms.** A force applied at a joint *center* produces no torque at
  that joint. A force at the heel, 10 cm from the ankle center, loads the ankle
  the way a real foothold does.
- **A grasp wrench for hands.** A hand gripping a hold applies both a force and
  a twist (a *wrench*). One force at the wrist can never produce that twist; two
  or three forces at different palm points can, so the pull-dominated torque
  residual we see on climbing scenes finally has a way to close.
- **Cones at the true surface.** The friction cone (see
  [human_optimization.md](human_optimization.md)) needs the surface normal *at
  the contact*. A surface point queries the wall where the contact actually is,
  not at a joint center buried inside the leg.

## Use mesh vertices, not robot frames

The natural-sounding vehicle is a *frame* — a named point rigidly attached to a
body, the way robotics libraries model tool tips. BetterRobot has full support
for these (frame kinematics and Jacobians). We should **not** use them here, for
one decisive reason: frames carry a *fixed* local offset, and the body's shape
parameters (betas) move the skin relative to the joints. BetterHuman re-bakes
joint placements and inertias when the shape changes, but a frame's offset would
go stale — a "heel" frame calibrated on one body sits inside or outside the foot
on another.

A mesh vertex has no such problem: its position already tracks shape and pose
through the model's skinning, for free. And BetterHuman already ships the right
table — its landmark list contains **big toe, small toe and heel for each foot**
plus five fingertips per hand, tracked exactly this way. Feet come for free;
knee, elbow and palm points would be hand-picked once into a small
mirror-symmetric table of vertex ids.

## What each part of the pipeline needs

- **Dynamics (small).** A force `f` at a surface point is the same as the force
  plus a lever torque `r × f` at the joint it belongs to (`r` = offset from the
  joint center to the point). BetterRobot's RNEA already consumes that torque
  half of an external wrench correctly, so the solve changes by a few dozen
  lines: compute each contact point's posed position, add `r × f`, and let
  several points on one body sum. No library work.
- **Kinematic losses (small–medium).** The attraction and repulsion terms
  already act on *vertices* — labels are expanded to a vertex mask today. A
  contact-point version swaps that expansion for the chosen vertex ids: mostly a
  relabeling. The velocity damper works on joints and needs a small variant that
  damps the posed contact vertex instead (its angular part can reuse the parent
  joint — a rigidly attached point turns with its body).
- **Detection (medium).** The per-joint readout in [contacts.md](contacts.md)
  becomes per-point: each point gets its own signed distance and drift. All the
  primitives (guarded SDF, stillness gate, hysteresis, span merge) transfer.
  One point is noisier than a joint's whole vertex set — use a small vertex
  neighborhood around each point, not the bare vertex.
- **Contracts (mechanical but wide).** Manual annotations, the 22-joint dataset
  export, the scorer, corpus provenance and the viewer all speak per-joint
  labels. Point labels can coexist by *folding up*: a joint is in contact when
  any of its points is. Save `point_contact` next to the folded `joint_contact`
  and bump the contact schema version. The manual annotation vocabulary (ten
  coarse body parts) actually maps more faithfully to ~12 surface points than to
  52 joints.

## The path, in phases

1. **Dynamics only.** Keep detection exactly as it is; fold the existing joint
   labels down onto the points; apply forces there with the lever torque.
   Nothing upstream changes, and the two biggest wins (ankle torque, grasp
   wrench) land immediately.
2. **Point-native detection.** Per-point SDF + drift readout, `point_contact`
   output, schema bump.
3. **Kinematic losses on points.**

## The main risk

Not the math — the mesh. Every benefit above is a better *moment balance*, and
that rests on the heel/palm vertices sitting where the real contact is. If the
fitted body is a few centimetres off (the error the whole pipeline fights), the
lever arms are wrong and the solved torques become confidently wrong instead of
absent. Before trusting phase-1 numbers, perturb the contact points by a few
centimetres and check how much the solved forces move.
