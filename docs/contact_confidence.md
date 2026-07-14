# Per-joint contact-label confidence (schema 4)

`label_confidence` is a continuous weight in `[0, 1]` saying how much a learner
should trust one final per-joint contact or no-contact label. It is a heuristic
training weight, not a calibrated probability. Zero means “ignore this label,”
not “the opposite label is true.” Confidence never changes the binary label.

Schema 4 deliberately has only three terms:

- `joint_visibility`: image visibility multiplied by 3-D body visibility;
- `surface_evidence`: support for a contact label;
- `motion_evidence`: support for a no-contact label.

There is no wall-distance, relative-clearance, persistence, detector-score, or
whole-body/floating-person term.

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
candidate must lie within 2 cm of the camera-to-joint ray and at least 5 mm in
front of that joint's local visible skin. Complete hands and ankle/foot pairs
are local groups, so their own members do not occlude one another. Other tracked
people use independent samples and can occlude the target person.

`n_occluding_vertices >= 5` gives exactly zero confidence for both label
classes. Counts one through four reduce visibility continuously. Out-of-frame,
behind-camera, and non-finite projections also give zero visibility.

## Contact evidence

Let `d` be the guarded signed body-surface distance in meters: positive outside
the scene and negative inside it.

```text
near                    = sigmoid((0.05 - d) / 0.01)
not_deep                = sigmoid((d - (-0.02)) / 0.01)
penetration_reliability = 0.40 + 0.60 × not_deep

surface_evidence = near × penetration_reliability
```

Evidence fades beyond 5 cm outside the wall. Implausibly deep penetration is
treated as suspicious geometry rather than proof of contact. Missing guarded
surface distance gives zero contact evidence.

## No-contact evidence

Let `m` be net joint displacement over the centered contact window and `T` the
window duration, normally `0.35 s`. Speed is `v = max(m, 0) / T`.

```text
motion_evidence = 1 - exp(-v / 0.10 m/s)
```

The motion source is the persisted pre-enforcement drift when available, with
final-body drift as a per-element fallback. This prevents contact attraction or
damping from manufacturing apparent stillness after the measurement.

The curve has no hard dead zone, but stationary and very slow joints stay low:

| speed | motion evidence |
|---:|---:|
| 0.000 m/s | 0.000 |
| 0.005 m/s | 0.049 |
| 0.010 m/s | 0.095 |
| 0.020 m/s | 0.181 |
| 0.050 m/s | 0.393 |
| 0.100 m/s | 0.632 |
| 0.200 m/s | 0.865 |

Missing or non-finite drift gives zero motion evidence. Motion is also missing
when the current frame or either centered-window endpoint is an invalid tracking
frame.

## Final weight

```text
label_evidence = surface_evidence  if joint_contact is true
                 motion_evidence   otherwise

label_confidence = joint_visibility × label_evidence
```

Therefore a stationary no-contact joint has zero confidence even if bad depth
places it far from the wall. A slowly moving no-contact joint has proportionally
low confidence. A genuinely approaching limb can remain strongly trusted right
up to the contact transition. Stationary contact labels are not penalized by
motion; they use current surface evidence instead.

## Saved fields

`estimate_contacts` writes these fields to `human_optim/contacts_1.npz` (or
`contacts_2.npz` for `kinematic_2`):

| Field | Shape | Meaning |
|---|---|---|
| `joint_contact` | `(P, N, J)` bool | Final binary contact label |
| `label_confidence` | `(P, N, J)` float | Final per-joint training weight |
| `joint_visibility` | `(P, N, J)` float | Image × 3-D body visibility |
| `joint_image_visibility` | `(P, N, J)` float | Image-border and camera-depth term |
| `joint_occlusion_visibility` | `(P, N, J)` float | Smooth body-occlusion term |
| `joint_occluding_vertices` | `(P, N, J)` float | Sampled-body occluder count |
| `surface_evidence` | `(P, N, J)` float | Contact-label evidence |
| `motion_evidence` | `(P, N, J)` float | No-contact-label evidence |
| `joint_sdf_cm` | `(P, N, J)` float | Unguarded signed-distance diagnostic |
| `joint_distance_cm` | `(P, N, J)` float | Euclidean scene-distance diagnostic; not a confidence input |
| `merged` | `(P, N, J)` bool | Label was filled by span merging |
| `confidence_schema` | scalar | `4` |
| `confidence_method` | scalar string | `motion-only-joint-label-v1` |
| `confidence_motion_source` | scalar string | `drift_pre_with_final_fallback` |

The artifact also stores hashes of its source body, scene, and camera transform,
plus every effective confidence configuration value. Invalid tracked frames are
written with every confidence term equal to zero. Manual annotations are not an
input to confidence generation; they are used only for evaluation.

## Limitations

- The weight is not learned or calibrated.
- Net displacement can miss out-and-back motion within the centered window.
- Motion magnitude does not distinguish approaching, leaving, or sliding.
- Body occlusion is a sampled-vertex proxy, not exact triangle rasterization.
  Thin occluders can be missed, and wall/hold occlusion is not modeled.
- Confidence does not repair an incorrect binary contact label.
