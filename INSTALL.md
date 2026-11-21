# Installation

## What you need

- [uv](https://docs.astral.sh/uv/) and git.
- A CUDA GPU for the model steps. The lock pins torch 2.13 with the CUDA 12.9 wheels (`pytorch-cu129` index), the
  torch BetterRobot and BetterHuman are developed on.
- [BetterRobot](https://github.com/rihat99/BetterRobot) and [BetterHuman](https://github.com/rihat99/better_human)
  checked out beside this repository, as `../BetterRobot` and `../BetterHuman`: `[tool.uv.sources]` of
  `pyproject.toml` installs both from there, editable.

## 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/rihat99/BetterVideoReconstruction.git
cd BetterVideoReconstruction
```

In a clone without them: `git submodule update --init --recursive`. Under `third_party/`:

| folder | what it is |
|---|---|
| `sam3`, `sam-3d-body`, `sapiens2`, `geocalib` | the models of the steps `sam3`, `sam3d`, `sapiens`, `gravity` (submodules) |
| `MHR` | one file of it: the map from MHR's mesh to SMPL-X's that the `smplx` step reads (`tools/mhr_smpl_conversion/assets/mhr2smplx_mapping.npz`; submodule) |
| `vggt-omega` | the model of the `vggt` step, a copy patched for a chunked low-memory forward |

## 2. Build the environment

```bash
uv sync --managed-python
```

This makes `.venv` with a uv-managed CPython 3.12 (`xtcocotools` builds a C extension, which fails on a system
Python without headers) and installs the lock: torch, BetterRobot and BetterHuman, the `third_party/` packages
editable (`sam3`, `sapiens`, `vggt-omega`, `geocalib`), and the dependencies of SAM 3D Body, listed in
`pyproject.toml` because that submodule has no `pyproject.toml` of its own. For the same reason SAM 3D Body goes on
the path by hand, once:

```bash
echo "$PWD/third_party/sam-3d-body" > "$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/sam-3d-body.pth"
```

A later `uv sync` keeps that file.

## 3. Body models

BetterHuman reads the licensed body models from `$BETTERHUMAN_MODELS_DIR`; set it (BetterHuman's README gives the
layout). BVR needs:

- `smplx/SMPLX_NEUTRAL.npz`, and `SMPLX_MALE.npz` or `SMPLX_FEMALE.npz` if `[smplx].gender` asks for one;
- `MHR/converted/mhr_lod1.npz`, the MHR body the `smplx` step converts from. BetterHuman's converter
  (`tools/convert/mhr/` in BetterHuman) makes it once, in that tool's own environment with pymomentum; BVR's
  environment has no pymomentum.

## 4. Model weights

Each model step downloads its weights on first use into the Hugging Face cache (GeoCalib into torch's hub cache). The
repositories marked gated need access: accept the licence on the model's page, then log in once with
`uv run hf auth login`. A config key points a step at a local copy instead.

| step | weights | local copy |
|---|---|---|
| `sam3` | `facebook/sam3` (`sam3.pt`), gated | `[sam3].checkpoint` |
| `vggt` | `facebook/VGGT-Omega` (`vggt_omega_1b_512.pt`), gated | `[vggt].checkpoint`, a `.pt` file |
| `sapiens` | `facebook/sapiens2-pose-5b` (`sapiens2_5b_pose.safetensors`, 19 GB) | `[sapiens].checkpoint_root`, holding `pose/sapiens2_5b_pose.safetensors` |
| `sam3d` | `facebook/sam-3d-body-dinov3`, gated; its snapshot carries the TorchScript MHR body the step runs | `[sam3d].hf_repo` |
| `gravity` | GeoCalib's `pinhole` or `distorted` weights from its GitHub release | `[gravity].weights` picks one |

## 5. Environment variables

| variable | meaning | default |
|---|---|---|
| `BVR_DATA_ROOT` | the data root: `examples/` and the datasets | the folder `data` beside the repository; an error if that does not exist |
| `BETTERHUMAN_MODELS_DIR` | the body models of section 3 | none |
| `CUDA_VISIBLE_DEVICES` | the GPU a step runs on (`[run].device = "cuda"`); empty for the CPU tests | every GPU |
| `BETTER_ROBOT_CHROME`, `BETTER_ROBOT_FFMPEG` | a headless Chromium and an ffmpeg for the viewer's `--record` (ffmpeg only for `.mp4`) | found on `PATH` |

## 6. Check

```bash
uv run python -c "import better_robot, better_human, bvr, bvr_datasets, sam3, sam_3d_body, sapiens, vggt_omega, geocalib"
CUDA_VISIBLE_DEVICES= uv run pytest tests -q
```

The import prints `Momentum is not enabled`: SAM 3D Body's warning that it runs its TorchScript MHR body, which is
what BVR wants.

## Notes

- `sam3` and `vggt-omega` pin `numpy<2`, BetterRobot needs `numpy>=2.0`; `[tool.uv] override-dependencies` forces
  `numpy>=2.0`, which all three run with.
- `xtcocotools` needs numpy and cython at build time: `[tool.uv.extra-build-dependencies]`.
- To follow new BetterRobot or BetterHuman commits, pull their checkouts and run `uv sync --managed-python`; it
  re-locks if their dependencies changed.
