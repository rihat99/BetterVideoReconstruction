"""Write our solved foot forces in the OpenCap dataset's own OpenSim ``.mot`` layout.

The dataset states a plate measurement as a *triple* — a force ``F``, a centre of
pressure ``p`` on the floor, and a free moment ``T`` about it — while our solve states a
set of point forces ``f_i`` at the contact frames ``r_i`` of a foot. The two are the same
wrench written two ways, and this script is the conversion, so our estimate can be read
by anything that reads the dataset (OpenSim's ``ExternalLoads``, the paper's own
scripts) and plotted against it column for column.

**The math.** Our side gives the wrench directly: ``F = Σ f_i`` and ``M_O = Σ r_i × f_i``
about the ground origin. :mod:`datasets.opencap.opencap_plates` inverts that to the triple; its
module docstring carries the derivation, sign by sign, and the tests round-trip the
dataset's own force files through it.

Frames: our forces live in our world (VGGT's first camera), the file wants the OpenSim
ground frame, so every point and vector is taken back through the importer's calibration
before the triple is formed — ``f' = Rᵀ f``, ``r' = Rᵀ (r − t)`` with ``R, t`` the
``R_cam_from_mocap`` of ``meta.json``. Contact-frame positions ``r_i`` are the TRUE ones,
re-posed from the saved ``q`` + ``betas`` (:func:`frame_positions`), not the parent joint.

``y0`` is the floor plane the dataset itself reports (the median ``ground_force_py`` over
the clip's loaded samples, ≈0.006 m). Each row is written one of three ways, and the
distinction matters because our solve is not a plate and can produce a force a plate never
could:

* ``‖F‖ < LOADED_N`` — nothing is being carried. The WHOLE row is zeroed, which is how the
  dataset writes an unloaded plate.
* ``F_y > LOADED_N`` — the ordinary case, the inversion above.
* ``‖F‖ ≥ LOADED_N`` but ``F_y ≤ LOADED_N`` — a real force with no vertical component to
  divide by, so the line of action never meets the floor and the COP does not exist. The
  force is kept (deleting hundreds of solved newtons would make the file disagree with the
  comparison), and the point is put at the force-weighted mean of that foot's own loaded
  contact frames, projected onto the floor plane; ``T_y`` is then the vertical moment left
  over about that point. **The horizontal moment residual is dropped** — a plate triple has
  nowhere to put it — so such a row preserves the force and the vertical moment but not the
  full wrench. They are counted in ``cop_undefined_rows`` and printed.

Only the feet are exported — the layout has no column for a hand, and these trials have no
hand contact.

Which contact frames belong to a foot is read off their PARENT JOINT rather than their
name, exactly as the comparison's ``_limb_members`` does, so the two cannot disagree and
a body carrying non-limb frames (``sacrum``, ``chest``, ``head``) does not trip a name
rule that was written for the hands-and-feet rigs.

Output ``processed/out/<clip>/<run>/estimated_forces.mot`` (19 columns at 60 Hz, the
dataset's header style) and ``estimated_forces.npz`` beside it. ``--run`` names the run
directory, so a variant chain (``human_optim_floorplane``) exports its own file.

Usage::

    uv run --no-sync python datasets/opencap/export_opencap_mot.py
    uv run --no-sync python datasets/opencap/export_opencap_mot.py --run human_optim_floorplane
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


from datasets.opencap.opencap_plates import LOADED_N, cop_free_torque, vertical_free_torque
from datasets.opencap.opencap_sensor_forces import PROCESSED, world_from_mocap
from datasets._common import _limb_members
from bvr.body import load_body
from bvr.human_optim.boards import LIMBS
from bvr.human_optim.qmath import normalize_q
from bvr.npz_io import atomic_savez

#: The 19 columns, in the dataset's order. ``R`` is the right foot, ``L`` the left.
COLUMNS = ["time"] + [f"{side}_ground_{field}" for side in ("R", "L")
                      for field in [f"force_v{a}" for a in "xyz"]
                      + [f"force_p{a}" for a in "xyz"] + [f"torque_{a}" for a in "xyz"]]

#: ``.mot`` side prefix -> the limb name the force files use.
SIDES = {"R": "right_foot", "L": "left_foot"}


def plate_triple(force: np.ndarray, moment: np.ndarray, fallback: np.ndarray,
                 y0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Our foot wrench → the dataset's ``(force, cop, free_torque)`` plus the undefined count.

    :param force: ``(N, 3)`` foot force in the OpenSim ground frame, newtons.
    :param moment: ``(N, 3)`` its moment about the ground origin, N·m.
    :param fallback: ``(N, 3)`` a point on the floor to use where no COP exists — the
        force-weighted mean of that foot's loaded contact frames (:func:`foot_wrenches`).
    :param y0: the floor plane, metres.
    :returns: the three ``(N, 3)`` blocks and how many rows took the fallback point. See
        the module docstring for the three cases.
    """
    cop, torque, defined = cop_free_torque(force, moment, y0)
    carrying = np.linalg.norm(force, axis=-1) >= LOADED_N
    undefined = carrying & ~defined
    if undefined.any():
        cop[undefined] = fallback[undefined]
        torque[undefined] = vertical_free_torque(
            fallback[undefined], force[undefined], moment[undefined])
    gate = (defined | undefined)[:, None]
    return force * gate, cop * gate, torque * gate, int(undefined.sum())


@torch.no_grad()
def frame_positions(kindyn: dict, chunk: int = 64) -> np.ndarray:
    """Re-pose the saved ``q`` → contact-frame world positions ``(N, F, 3)``.

    The stored global ``scale`` is re-applied about the pelvis, the same operation
    ``estimate_contacts`` and ``measure_contacts_boards`` do, so the frames sit at the
    body's metric size.
    """
    names = [str(x) for x in kindyn["contact_frame_names"]]
    body = load_body(device="cpu", frames_allow=names)
    if list(body.contact_frame_names) != names:
        raise SystemExit("the body's contact frames do not match the run's")
    betas = torch.as_tensor(kindyn["betas"][0], dtype=torch.float32)
    q = torch.as_tensor(kindyn["q"][0], dtype=torch.float32)
    scale = float(np.asarray(kindyn["scale"], np.float32)[0])
    posed = []
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        out = body.forward(betas[None].expand(end - start, -1),
                           normalize_q(q[start:end], body.num_joints))
        root = out.joints[:, :1, :]
        posed.append(root + scale * (out.frames_world[..., :3] - root))
    return torch.cat(posed, 0).numpy().astype(np.float64)


def floor_y(clip: Path) -> float:
    """The dataset's own floor plane for this clip: median COP height over loaded samples."""
    gt = np.load(clip / "forces.npz", allow_pickle=True)
    cop = np.asarray(gt["cop"], np.float64)[:, 2:]                 # the two feet
    loaded = np.asarray(gt["force"], np.float64)[:, 2:, 1] > LOADED_N
    height = cop[..., 1][loaded & np.isfinite(cop[..., 1])]
    return float(np.median(height)) if height.size else 0.0


def foot_wrenches(kindyn: dict, rotation: np.ndarray, translation: np.ndarray, y0: float
                  ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per foot, ``(force, moment-about-the-ground-origin, fallback point)`` in mocap coordinates.

    The fallback point is where that foot is pressing — the ‖f‖-weighted mean of its
    contact frames, flattened onto the floor plane. It is only used on the rare row whose
    COP does not exist (see :func:`plate_triple`), never where the inversion applies.
    """
    members = _limb_members([str(x) for x in kindyn["joint_names"]])       # LIMBS order
    parents = [int(p) for p in kindyn["contact_frame_parents"]]
    points = frame_positions(kindyn)                                # (N, F, 3) our world
    forces = np.asarray(kindyn["frame_forces"][0], np.float64)      # (N, F, 3) our world
    points_mocap = (points - translation) @ rotation                # Rᵀ (r − t)
    forces_mocap = forces @ rotation                                # Rᵀ f
    out = {}
    for side, limb in SIDES.items():
        joints = set(members[LIMBS.index(limb)])
        cols = [c for c, parent in enumerate(parents) if parent in joints]
        if not cols:
            raise SystemExit(f"the run has no contact frame on the {limb}")
        weight = np.linalg.norm(forces_mocap[:, cols], axis=-1)[..., None]     # (N, C, 1)
        centre = ((points_mocap[:, cols] * weight).sum(1)
                  / np.maximum(weight.sum(1), 1e-9))
        centre[:, 1] = y0
        out[side] = (forces_mocap[:, cols].sum(1),
                     np.cross(points_mocap[:, cols], forces_mocap[:, cols]).sum(1), centre)
    return out


def write_mot(path: Path, table: np.ndarray) -> None:
    """Write ``(N, 19)`` as a ``.mot`` in the dataset's header and column style."""
    header = [f"name {path.stem}", f"datacolumns {table.shape[1]}",
              f"datarows {table.shape[0]}",
              f"range {table[0, 0]:.6f} {table[-1, 0]:.6f}", "endheader "]
    rows = ["\t".join(f"{v:20.8f}" for v in row) + "\t" for row in table]
    path.write_text("\n".join(header + ["\t".join(COLUMNS) + "\t"] + rows) + "\n")


def export(clip: Path, run_dir: Path) -> int:
    """Convert one clip's solved forces into the dataset's force-file form."""
    kindyn = dict(np.load(run_dir / "kindyn_1.npz", allow_pickle=True))
    rotation, translation = world_from_mocap(clip)
    y0 = floor_y(clip)
    wrenches = foot_wrenches(kindyn, rotation, translation, y0)
    fps = float(np.asarray(kindyn["fps"], np.float32))
    n_frames = len(next(iter(wrenches.values()))[0])

    columns, saved, undefined = [np.arange(n_frames) / fps], {}, {}
    for side in SIDES:
        force, moment, fallback = wrenches[side]
        *triple, undefined[SIDES[side]] = plate_triple(force, moment, fallback, y0)
        saved[SIDES[side]] = triple
        columns += [component for block in triple for component in block.T]
    table = np.stack(columns, 1)

    write_mot(run_dir / "estimated_forces.mot", table)
    atomic_savez(
        run_dir / "estimated_forces.npz",
        columns=np.array(COLUMNS), table=table.astype(np.float32),
        limbs=np.array(list(saved)),
        force=np.stack([saved[k][0] for k in saved], 1).astype(np.float32),
        cop=np.stack([saved[k][1] for k in saved], 1).astype(np.float32),
        free_torque=np.stack([saved[k][2] for k in saved], 1).astype(np.float32),
        cop_undefined_rows=np.array([undefined[k] for k in saved], np.int32),
        floor_y_m=np.float32(y0), fps=np.float32(fps),
        frame=np.str_("opensim_ground_y_up"),
        R_world_from_mocap=rotation.astype(np.float32),
        t_world_from_mocap=translation.astype(np.float32))
    peak = {side: float(np.abs(w[0][:, 1]).max()) for side, w in wrenches.items()}
    print(f"{clip.name}: wrote estimated_forces.mot  ({n_frames} rows, floor y {y0 * 1000:.1f} mm, "
          f"peak Fy R {peak['R']:.0f} N  L {peak['L']:.0f} N, COP undefined on "
          f"{undefined['right_foot']} R / {undefined['left_foot']} L rows)")
    return n_frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=PROCESSED, help="the processed clip root")
    ap.add_argument("--run", default="human_optim", help="run directory under processed/out/<clip>/")
    args = ap.parse_args(argv)

    written = 0
    for clip in sorted(p for p in args.source.iterdir()
                       if (p / "forces.npz").exists() and (p / "meta.json").exists()):
        run_dir = args.source / "out" / clip.name / args.run
        if not (run_dir / "kindyn_1.npz").exists():
            print(f"{clip.name}: no kindyn_1.npz, skipped")
            continue
        export(clip, run_dir)
        written += 1
    print(f"\n[export_opencap_mot] {written} clip(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
