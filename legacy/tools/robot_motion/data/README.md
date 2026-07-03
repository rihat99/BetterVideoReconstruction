# Builder data files

Runtime data shipped with BetterRobot builders. Access via
`importlib.resources.files("better_robot.io.builders.data")`.

## `smpl_body_densities.json`

Per-body densities (kg/m³) for the 24 SMPL joints under five systems:

- **Naive** — uniform 1010 kg/m³.
- **Dempster**, **Chandler**, **Clauser** — biomechanics tables.
- **Custom** — an SMPL-tuned hybrid (default).

Provenance: copied verbatim from
`better_human/src/better_human/smpl/config/body_densities.json`.

## `smpl_3d_segmentation.npy`

Pickled dict `{"body_faces": list[np.ndarray]}` — a length-24 list of face-index
arrays partitioning the 13776-face / 6890-vertex SMPL mesh into per-joint
submeshes. Load with `np.load(path, allow_pickle=True).item()`.

Provenance: copied verbatim from
`better_human/src/better_human/smpl/config/smpl_3d_segmentation.npy`.
