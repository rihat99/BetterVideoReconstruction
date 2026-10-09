# Report 5c — old viewer sidebar and skeleton look restored (mars_top_opus, 2026-10-09; saved by the main session; BR dev 26dfb8d, BVR dev "Dev 9/N")

Result: the "layers" folder and the collapsed "dynamics" folder are back; skeleton at the old sizes with thin finger bones; the torque toggle sizes the spheres (×(1+min(τ/ref,1))) and tints them yellow, never colour alone; contact-frame links are solid 3.6 mm cylinders.

## BR dev 26dfb8d
`SkeletonLayer(..., radius: float | (J,) | (T, J), bone_radius: float | (J,) | None)`: per-step joint radius (marker scales written per step only then), per-bone radius indexed like `parents` (entry j = bone parents[j]→j, deviates from the brief's (B,)), radius 0 hides. Solid frame links needed no change (`ArrowsLayer(head_size=0, shaft_radius=r)`). New test `test_skeleton_takes_per_bone_radii_and_per_step_joint_radii`; docs/concepts/viewer.md + changelog line; ruff/mkdocs clean; suite 1673 passed, 13 skipped (CPU-only run).

## Sidebar (folder → control → default)
- layers (expanded): SMPL-X body on, skeleton on, contact frames on, frame normal arrows off, human points off, scene points on, scene normals off, cameras on, gravity on, ground plane on.
- dynamics (collapsed; only while a source with forces is active): force display per-joint/per-frame (per-joint), joint torques (sphere size) off, fingers/palm off, contact forces on, root force on, root torque on; sensor forces (measured) off and estimated forces (plate) off when the clip has them.
- Mechanism: each layer item has a `when` map control→required value; a layer shows when its checkbox is on, every `when` matches and its body source is active; controls with nothing to toggle are hidden. A forces source has four SkeletonLayers per person (body/fingers × plain/torque-sized); root wrench split into root-force and root-torque layers (arrows < 1e-4 m not drawn).

## Sizes (old scene.py constants)
Kinematic joints/bones 0.028/0.012 m, fingers 0.0084/0.0036; dynamics joints/bones 0.022/0.012; torque on: size ×(1+min(τ/ref,1)), yellow (245,200,40), root grey; dynamics fingers hidden unless toggled, then 0.0066/0.0036 (old drew them full size); contact frames sphere 0.012, link r 0.0036 colour (150,190,160), normal 6 cm; force/root arrows unchanged from G2.

## Verification
BVR viewer tests 22 passed (new `test_skeleton_sizes_and_dynamics_toggles_match_the_old_viewer`). Fake-server pass (scratchpad v5c/verify_fake.py) over the 4 regression clips + climb_wall_3 a_00_01_part6 (sensor arrows) + OpenCap subject8_DJ1_cam1 (plate arrows), every source × torque × fingers × force display at 4 steps: no exception, radii finite; layers per clip 32 (58 two-person). Screenshots vis/viewer_boulder_1{,_forces,_torque}.png (previous two trashed with suffix .20261009-131535).

## Still different from the old viewer
1. dynamics fingers at 0.3× when shown (user: "too big"). 2. force-display dropdown hidden for a source without per-frame forces. 3. Not restored (outside brief): size sliders, playback folder + onboard camera, mesh translucency only under an overlay (opacity fixed 0.7). 4. Any checkbox with nothing to toggle hides (step 5 behaviour). 5. BR tessellation differs from the old icosphere/10-section cylinder.

## Side effects
`uv run ruff` re-synced the venv and reinstalled decord 0.6.0 from the lockfile (main session checked: detectron2 0.6, decord, sam_3d_body still import). Stale text in CLAUDE.md:150-153 and scripts/view_3d.py:10 fixed by the main session in the same commit.
