# The viewer

`scripts/view_3d.py` shows the clips of a run in a browser, on BetterRobot's viewer (viser), with BetterHuman's
posed meshes. It takes the pipeline's targets:

```bash
uv run python scripts/view_3d.py --source <video or folder> --save-dir <run root> [--upstream <root>] [--config <delta>]
uv run python scripts/view_3d.py --source <video or folder> --save-dir <run root> --record <file.mp4|.webm|.png>
```

It opens every clip of `--source` whose tree has `geometry/transform.npz` (the scale step), at
`http://localhost:<port>` (`[viewer].port`, 8082). Each file is read from the clip's own tree, else from
`--upstream`'s, so a run that redid a few steps shows on top of the run it continued. The video comes from
`--source`. Display settings are `[viewer]`: point sizes, camera size, the up axis, how many human points to draw
per frame and person, the radius of the skeleton's joint spheres (`joint_radius`, `finger_joint_radius`), whether
torque sizes them at the start (`torque_size`) and how far it grows them (`torque_growth`). A scene point's radius is
`point_size_spacings` times the cloud's own sample spacing (the voxel the fuse step merged it on, `voxel_size` of
`geometry/scene.npz`, 1.8 to 7 cm on the example clips), so a wall reads as a surface at any spacing; a tree that does
not record its spacing gets `point_size`.

## What it shows

The sidebar holds, in the order of the viewer before the rebuild:

- Clip: a pager over the clips, when there are several. A switch takes the old clip off the screen at once, then
  reads and draws the new one (a few seconds for the bodies); a clip that fails to read leaves the screen empty.
- Timeline: a Play/Pause button, step, speed. Images: the video, in step with the timeline.
- Body: the source of the bodies, among those the clip holds: `converter` (`sam3d/smplx_params.npz`), `prefit`,
  `kinematics_1`, `kinematics_2`, `kindyn_1`. Contact labels come from the contacts file beside each body file
  (`contacts_1` for `kinematics_1`, `contacts_2` for `kinematics_2`, the labels of `kindyn_1` for itself). A switch
  keeps the frame. The default is the newest source the run's own tree wrote. "Torque as size": a `kindyn_1` body's
  joint torque sizes the joint spheres, drawn orange, instead of colouring them: a sphere grows to `torque_growth`
  times its size (1.4) at the clip's 95th percentile of torque.
- Layers, one checkbox per tag: Scene points, Human points (off at the start, read from `geometry/human.npz` the
  first time they are switched on), Cameras (the frustum, a wireframe, and the camera's path), Gravity, Ground plane
  (the floor the gravity step fitted and accepted, `ground_accepted` of `geocalib/gravity.npz`: ten times its extent,
  half-metre cells, a see-through grey fill; none where the gravity step found no floor), Body (the exact posed mesh,
  see-through, one colour per person: navy, then salmon, green, lilac, gold), Skeleton (the joint
  tree, each joint a sphere of one size, smaller on the fingers, coloured by joint torque on a `kindyn_1` body),
  Contact frames (red in contact, green free, with their normals and a thin link to the joint each hangs on; grey
  without labels), Forces (the contact forces: `kindyn_1`'s solved ones, and for `kinematics_1` and `kinematics_2` the
  witness's of their contacts file, `frame_force`), Root wrench (the root's six rows of `joint_torques`, the raw inverse
  dynamics: the residual force and torque the contact forces leave; for a person without contact labels, whom no
  force holds, about the body weight on every frame; nothing where the stencil reaches an untracked frame). A layer
  switched off stays off across switches.
- Display: the bodies' opacity (kept across switches) and the scene's point size (each clip starts at its own).
- Robot: BetterRobot's skeleton style and joint frames.
- Info: the number of people, the source of gravity, and the sources read from `--upstream`.

The opening view holds the people over the whole clip, so a person who travels stays in it, seen from 25 degrees
above the filming camera's side (BetterRobot's `Viewer.aim`: the body layers of 32 frames spread over the clip fit in
80 % of a 16:9 view). The scene, the cameras and the floor do not widen it; a clip without people is framed whole.

## Recording

`--record` writes the first clip with the opening view to a video or a still through a headless Chromium
(`BETTER_ROBOT_CHROME`, or `chromium`, `google-chrome` or `chrome-headless-shell` on `PATH`) and, for `.mp4`, ffmpeg
(`BETTER_ROBOT_FFMPEG` or on `PATH`; the binary of the `imageio-ffmpeg` package works). It recorded a two-person clip
in 67 s on the CPU.

## Notes

- Bodies are BetterHuman's exact meshes, sent each frame, not skinned in the browser: viser 1.1.1 turns the page white
  when a skinned mesh is removed with a browser open, and the app removes the bodies on every switch.
- Bodies are posed once per clip and source, on the CPU, when the source is chosen.
- The floor's fill is see-through, yet viser lets it hide what lies beyond it: a body that the fitted plane cuts, or
  seen from under the floor, loses that part of its mesh while the skeleton stays whole.
- A switch of clip reads only what it draws: the human depth points (310 MB on boulder_1) wait until they are shown.
- A tree written before the rebuild opens too: `bvr.files.load_npz` reads it and maps its contact frame `head` to
  `head_top`.
