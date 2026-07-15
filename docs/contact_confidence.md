# Per-joint contact-label confidence (schema 5)

`label_confidence` is a continuous weight in `[0, 1]`. It says how much a
learner should trust one final contact or no-contact label. It is not a
calibrated probability. Zero means “ignore this label.” Confidence never
changes the binary label.

Schema 5 uses four terms:

- `joint_visibility` says whether the joint can be seen in the image and in 3-D.
- `surface_evidence` supports a contact label near a plausible scene surface.
- `motion_evidence` supports a no-contact label when motion is clear and the
  local human depth fit is trustworthy.
- `joint_depth_trust` measures agreement between the posed body and the human
  depth cloud along the same camera rays.

There is no wall-clearance evidence term. Wall distance only changes the speed
scale inside `motion_evidence`. Distance alone can never raise confidence. This
matters for a floating reconstruction. Its local depth fit can look good while
its apparent distance from the wall is wrong.

Let

```text
smoothstep(x) = u²(3 - 2u), where u = clamp(x, 0, 1)
sigmoid(x)    = 1 / (1 + exp(-x))
```

The formulas below use defaults from `configs/human_optimization.toml`.

## Visibility

```text
edge  = min(x, y, W - 1 - x, H - 1 - y)
image = smoothstep(edge / 10 px) × [camera_depth > 0]

body_visibility = smoothstep(1 - n_occluding_vertices / 5)
joint_visibility = image × body_visibility
```

Occlusion uses a deterministic sample of 2,500 vertices per tracked body. A
candidate must be within 2 cm of the camera-to-joint ray. It must also be at
least 5 mm in front of that joint's local visible skin. Complete hands and
ankle/foot pairs are local groups. Their own members do not occlude one another.
Other tracked people use separate samples and can occlude the target person.

Five or more occluding vertices give zero confidence for both label classes.
Counts from one to four reduce visibility smoothly. An out-of-frame,
behind-camera, or non-finite projection also gives zero visibility.

## Contact evidence

Let `d` be the guarded signed body-surface distance in meters. Positive is
outside the scene. Negative is inside it.

```text
near                    = sigmoid((0.05 - d) / 0.01)
not_deep                = sigmoid((d - (-0.02)) / 0.01)
penetration_reliability = 0.40 + 0.60 × not_deep

surface_evidence = near × penetration_reliability
```

Evidence fades beyond 5 cm outside the wall. Deep penetration is suspicious
geometry, not proof of contact. A missing guarded distance gives zero contact
evidence.

## No-contact evidence

Let `m` be net joint displacement over the centered contact window. Let `T` be
the window duration, normally `0.35 s`. Speed is `v = max(m, 0) / T`.

Let `q` be scene proximity. It changes the speed scale smoothly. It is zero
when guarded scene distance is missing.

```text
q = sigmoid((0.05 m - d) / 0.01 m)
s = 0.10 m/s + q × (0.15 m/s - 0.10 m/s)
p = 1 + q × (3 - 1)

raw_motion      = 1 - exp(-(v / s)^p)
motion_evidence = joint_depth_trust × raw_motion
```

The response is the original exponential far from the surface. It becomes
cubic near the surface. The larger near-wall scale also asks for more motion
near a possible contact. Together these changes keep slow near-wall drift close
to zero while a clear pass still rises quickly. At zero speed the result is
zero for every distance.

Depth trust is
`exp(-(median(log(z_cloud / z_mesh)) / 0.05)²)`. A value of one means the local
human depth agrees with the body. A value near zero means the local body fit is
stretched in depth. When fewer than five cloud points match a joint, depth
trust is one. Missing data leaves existing motion evidence unchanged.

The motion source is the persisted pre-enforcement drift when available. Final
body drift is the per-element fallback. This stops attraction or damping from
manufacturing apparent stillness after the measurement.

The first table holds depth trust at one. “Near” uses `d = 0.02 m`. “Far” uses
the limiting `q = 0` scale.

| speed | raw motion, near | raw motion, far |
|---:|---:|---:|
| 0.000 m/s | 0.000 | 0.000 |
| 0.005 m/s | 0.000 | 0.049 |
| 0.010 m/s | 0.000 | 0.095 |
| 0.020 m/s | 0.003 | 0.181 |
| 0.040 m/s | 0.022 | 0.330 |
| 0.050 m/s | 0.042 | 0.393 |
| 0.100 m/s | 0.276 | 0.632 |
| 0.200 m/s | 0.911 | 0.865 |

Missing or non-finite drift gives zero motion evidence. Motion is also missing
when the current frame or a centered-window endpoint is an invalid tracking
frame.

## Final weight

```text
label_evidence = surface_evidence  if joint_contact is true
                 motion_evidence   otherwise

label_confidence = joint_visibility × label_evidence
```

Depth trust affects only no-contact labels. Contact labels keep their existing
surface evidence.

These worked examples use full visibility unless stated otherwise.

| label and inputs | label confidence |
|---|---:|
| no-contact, `d=2 cm`, `v=0.04 m/s`, depth trust `1` | 0.022 |
| no-contact, `d=2 cm`, `v=0.20 m/s`, depth trust `1` | 0.911 |
| no-contact, `d=2 cm`, `v=0.20 m/s`, depth trust `0.25` | 0.228 |
| no-contact, any finite speed, depth trust `0` | 0.000 |
| contact, `d=2 cm`, depth trust `0` or `1` | 0.942 |
| contact, `d=2 cm`, visibility `0.5` | 0.471 |

## Saved fields

`estimate_contacts` writes these fields to `human_optim/contacts_1.npz`. It uses
`contacts_2.npz` for `kinematic_2`.

| Field | Shape | Meaning |
|---|---|---|
| `joint_contact` | `(P, N, J)` bool | Final binary contact label |
| `label_confidence` | `(P, N, J)` float | Final per-joint training weight |
| `joint_visibility` | `(P, N, J)` float | Image × 3-D body visibility |
| `joint_image_visibility` | `(P, N, J)` float | Image-border and camera-depth term |
| `joint_occlusion_visibility` | `(P, N, J)` float | Smooth body-occlusion term |
| `joint_occluding_vertices` | `(P, N, J)` float | Sampled-body occluder count |
| `surface_evidence` | `(P, N, J)` float | Contact-label evidence |
| `motion_evidence` | `(P, N, J)` float | Proximity-adaptive, depth-trusted no-contact evidence actually used |
| `joint_depth_gap_cm` | `(P, N, J)` float | Signed median camera-ray depth gap; NaN with too few matches |
| `joint_depth_points` | `(P, N, J)` float | Matched human-cloud point count |
| `joint_depth_trust` | `(P, N, J)` float | Smooth local human-depth fit trust |
| `joint_sdf_cm` | `(P, N, J)` float | Unguarded signed-distance diagnostic |
| `joint_distance_cm` | `(P, N, J)` float | Euclidean scene-distance diagnostic; not a confidence input |
| `merged` | `(P, N, J)` bool | Label was filled by span merging |
| `confidence_schema` | scalar | `5` |
| `confidence_method` | scalar string | `depth-trust-motion-v1` |
| `confidence_motion_source` | scalar string | `drift_pre_with_final_fallback` |

The artifact also stores source hashes and every effective confidence knob.
Invalid tracked frames have every confidence term set to zero. Manual
annotations are used only for evaluation.

## Limitations

- The weight is not learned or calibrated.
- Net displacement can miss out-and-back motion inside the centered window.
- Motion magnitude does not distinguish approaching, leaving, or sliding.
- Body occlusion is a sampled-vertex proxy. Thin occluders can be missed.
- Wall and hold occlusion are not modeled.
- The depth-fit metric does not detect a whole-body floater.
- Confidence does not repair an incorrect binary contact label.
