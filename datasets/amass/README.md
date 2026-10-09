# amass — contacts and forces for raw AMASS motions

Raw AMASS SMPL-X motions (`*_stageii.npz`, 120 fps mocap) labelled with ground contact and contact forces by the BEDLAM2 readout:
BEDLAM2's bodies are AMASS motions, so `datasets.bedlam2.contacts` (35 contact frames vs a floor plane: height, stillness, skin facing)
and `datasets.bedlam2.forces` (the `kindyn_1` force solve at the frozen pose) apply as they are. The floor, which AMASS does not
give, is measured as the median height of each frame's lowest still contact frame.

- **Where:** AMASS lives at `<data root>/AMASS/`; the script takes any motion file or directory and writes to `--save-dir`.
- **Run:** `CUDA_VISIBLE_DEVICES=N EGL_DEVICE_ID=N uv run --no-sync python datasets/amass/label_amass.py --motions <stageii.npz | dir> --save-dir <dir> [--step label|render|all]`.
- **Output:** `<stem>_contacts.npz` in the AMASS world at 30 fps (every 4th frame; `source_frame` maps back): `frame_contact (N, 35)`,
  `frame_force (N, 35, 3)` newtons, `frame_position`, the same per SMPL-X joint (`joint_contact (N, 52)`, `joint_force`, `joint_position`),
  names, the frame→vertex table, the floor, `total_mass_kg`, `q`, the three measurements behind every label; plus one `contact_frame_vertices.npz`.
  `--step render` draws `<stem>_contacts.mp4` (two mesh views with red/green frames and force arrows, a force-vs-`m(a_com − g)` strip, a contact timeline).
- **Knobs:** `[bedlam]` and `[bedlam_forces]` of `datasets/bedlam2/configs/bedlam.toml` (`--config`). The pose goes in the way
  `datasets/bedlam2/import_bedlam2.py` brings it in: 16 locked-head betas truncated to 10, the hand mean subtracted.
