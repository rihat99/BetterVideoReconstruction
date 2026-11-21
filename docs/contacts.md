# Contacts: which points of the body touch the scene, and when

The steps `contacts_1` and `contacts_2` (`bvr/stages/estimate_contacts.py`) read contact on the bodies of
`kinematics_1` and `kinematics_2`. Contact lives on the body's 35 contact frames (`bvr.body`): named points of the
skin, each with its outward normal, from the soles and toes, the hands and the knees to the elbows, the back and
`head_top`. For every person, frame and contact frame two witnesses answer, and the readout unions them
(`bvr/contacts/`).

## The geometric witness: near and still

`bvr.contacts.detector`. Per frame and contact frame it finds the nearest scene point: the signed distance along that
point's normal, the distance, and the point's confidence. A distance is trusted only next to a confident point
(`max_dist_m`, `conf_thresh`): a hole in the cloud is unknown, not "no wall".

Stillness is the drift, the net distance a contact frame covers across `[contact_gate].window_s` (0.35 s). Speed at
one instant cannot tell a gripping hand from a slow pass along the wall: the fit's jitter of a few millimetres per
frame is a speed of 0.1 to 0.3 m/s. Over a third of a second the jitter cancels, a grip nets under 2 cm and a slow
pass 3 to 7 cm. The drift is `drift_pre` of the body file, measured by `kinematics_1` before any contact term acted;
the contact steps never measure it again ([`fitting.md`](fitting.md)).

A label is a span with hysteresis: it starts below `[contacts].close_thresh_m` (3 cm), lasts while the frame stays
below `exit_thresh_m`, and is dropped if shorter than `detect_min_frames` (3 frames at every rate). The detection of
`kinematics_1` is the same rule with the looser `[auto_contact]` distances (8 and 12 cm). A frame that is not
still (drift above twice `still_m`) cannot touch, unless it presses into the surface (closer than `still_override_m`)
and the solved load confirms the press (`confirm_bw`): a planted hand presses as close as one sweeping along the wall.
A short gap between two spans is filled where the frame never left (`merge_*`). The skin's own normal decides nothing
except the side of the toes: scene normals come from a reconstructed cloud and skin normals from a fitted pose, and a
gate on "does the skin face the wall" threw away real contacts.

## The force witness: what holds the body up

`bvr.contacts.witness`. Inverse dynamics of the fixed trajectory with nothing touching it leaves, on the six rows of
the root, the wrench contact must have supplied (`bvr.terms.dynamics.balance`). The witness asks which forces at the
contact frames explain it, for every person and frame at once:

    per frame:  0.5 |S (tau_free - J^T f)_root|^2 + 0.5 (m^2 + (v / v*)^2 + (gap / d*)^2) |f|^2

- Forces are in body weights, and each is a non-negative combination of the four edges of a friction pyramid about
  the negated skin normal (`[friction].half_angle_deg` = 45). A free force at one limb could absorb any residual;
  in a cone a frame explains only what its skin is turned to explain, and the two sides of a foot's toes face
  opposite ways. `S` divides the torque rows by a 0.2 m lever.
- Force on a moving frame is expensive (`v` its drift speed, `v*` = `stillness_vstar_mps`), and so is force far
  from the scene (`gap` its distance beyond the nearest candidate frame of the same instant, `d*` = `gap_dstar_m`).
  Pricing the distance relative to the nearest frame lets a body that is misplaced as a whole, every frame far from
  the wall, still be given force: the case geometry cannot see.
- `m` = `magnitude_weight` is the plain price of force. A frame with no measured distance or drift is no candidate.
- The problem is convex, solved by `iterations` steps of Adam with the bound enforced after each, the learning rate
  decaying from `learning_rate` to `learning_rate_final`.

## The readout

`bvr.contacts.labels.read`, per person, in numpy along time:

1. The solved load labels a frame where it exceeds `theta_bw` (0.02 body weights), with hysteresis down to
   `theta_hold_bw`. Beyond the detector's trust radius, load counts only on a frame that is still at a looser bar
   (`far_still_x` times `still_m`): real far support holds still, the scraps of residual a bad fit parks on a limb
   do not. Heels take no label from load: the split of load inside a foot is not determined, and a flat foot can
   stand on its ball alone.
2. Knees take no label from either witness: a knee rides near the wall, still, for most of a climb without touching.
3. Where the solve explained nothing it is mute: on a frame whose inverse dynamics do not count, or whose root
   residual after the solve stays above `mute_res_bw` body weights (a rope holds the climber, the trajectory is
   broken). Its silence there is not evidence, and a press counts without its confirmation.
4. The labels are the union of both witnesses. A contact lasts at least `min_span_s` (0.1 s), and so does a
   release; a span that presses is kept however short.
5. Each foot gets one side of its toes per contact, the side whose frames carried more solved load.

## How far to trust a label

`bvr.contacts.confidence`. The labels are folded onto report joints, a contact frame's parent joint (every frame of
a hand reports at its wrist), because that is what people annotate. Each folded label gets the probability that a
human annotator gives the same label, from one logistic head per label value (contact or free):

    logit = intercept + group + log_drift * log(drift + drift_floor_m)
                      + log_transition * log(1 + 30 * min(seconds to the nearest change, transition_cap_s))

`group` is the joint's group (hand, foot, ankle, knee), `drift` the joint's stillest frame. The coefficients are
`[contact_confidence]`, fitted on the annotated ClimbingVideos scenes by
`bvr_datasets/climbingvideos/calibrate_contact_confidence.py --write`; do not edit them by hand. Signed distance,
solved load, keypoint scores, reprojection residuals, visibility and the clip's quality each added at most 0.01 AUROC
to these three inputs. The label itself is never changed; joints outside the four groups get NaN.

## The quality flag

The contact steps are the first where the clip's three quality signals exist, so they write a flag:

- `camera`: the camera travels more than `cam_depth_max` of the median scene depth and the root residual before any
  force is above `res_before_bw_max` body weights. VGGT turned a pan or a zoom into travel.
- `depth`: the median depth smear of the human points is above `smear_med_max` ([`scale.md`](scale.md)).

An empty flag passes; a missing signal never raises one.

## Output

`human_optim/contacts_<n>.npz`, format `bvr-contacts`, all `(P, N, 35)` unless said:

| key | holds |
|---|---|
| `frame_contact` | the labels the later steps read |
| `frame_contact_geom`, `frame_force_label` | each witness's own labels |
| `frame_force` `(P, N, 35, 3)` | the witness's forces (N); `body_weight_n (P,)` |
| `residual_force_before_n`, `residual_force_after_n` `(P, N)` | the root residual force without and with the forces, NaN where the witness heard nothing |
| `frame_sdf_cm`, `frame_distance_cm`, `frame_drift_m` | the measured signals, NaN where nothing was measured |
| `joint_contact`, `joint_label_confidence` `(P, N, 52)` | the labels folded onto joints and their confidence |
| `frame_report_joint (35,)`, `frame_label_confidence` | each contact frame's report joint, and the confidence there |
| `quality_flag`, `quality_cam_ratio`, `quality_smear_med`, `quality_res_before_bw` | the flag and its signals |
| `contact_frame_names`, `frame_parent_joint`, `frame_vertex`, `joint_names`, `parents`, `object_ids`, `valid_mask`, `fps` | what the arrays index |
| `contact_label_schema` (2), `contact_label_method`, `confidence_method` | the version of the rules; readers refuse another schema |

## Measured

On the 13 annotated example clips (`bvr_datasets/examples/score_frame_contacts.py`, from the same upstream files as
the code before the rebuild) the readout scores pooled F1 0.842 (`contacts_1`: precision 0.881, recall 0.807) and
0.856 (`contacts_2`) against the manual annotations. Precision above recall: it misses more contacts than it invents,
and the 782 annotated knee frames are all missed, as rule 2 says. A step takes 3 to 4 s on a 104-frame clip.
