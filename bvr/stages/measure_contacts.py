"""Contact labels MEASURED by a force rig (``sensor_forces.npz``) instead of estimated.

The drop-in replacement for ``estimate_contacts.py`` on a clip with a force sensor under
each limb (the peter boards, the Parkour and OpenCap plates): instead of inferring contact
from how near and how still a body part is, it reads the contact state off the rig. The
dataset's own script writes ``human_optim/sensor_forces.npz`` in :mod:`bvr.human_optim.measured_contacts`
schema, so the sensors say exactly which limb is loaded and when — ground truth for
everything downstream that consumes contact labels.

What the boards cannot say is *which part* of a hand or foot is touching, and the
optimization's contact points are the finer contact **frames** (palm, thumb, heel,
big toe, ...). :mod:`bvr.human_optim.measured_contacts` closes that gap: measured force with
hysteresis gives the per-limb on/off, then each loaded span is put on the frames of
that limb the body holds near the wall (a span with no such frame falls back to its
nearest one, flagged ``frame_rescued`` — which is how a limb the reconstruction put
well off the wall shows up).

The artifact is the same schema-2 contacts file the pipeline's ``kinematics_2`` and
``kindyn_1`` already read, so those stages run unchanged on measured labels.

Run from CLI::

    python -m bvr.stages.measure_contacts \\
        --predictions out/clip/ --bodies kinematic_1 \\
        --config datasets/peter/configs/human_optimization_peter.toml

Reads:  human_optim/<bodies>.npz, human_optim/sensor_forces.npz,
        geometry/scene.npz, geometry/transform.npz
Writes: human_optim/contacts_<1|2>.npz
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import argparse

import numpy as np
import torch
from scipy.spatial import cKDTree

from bvr import determinism
from bvr.body import load_body
from bvr.config import load_config
from bvr.human_optim import paths
from bvr.human_optim.measured_contacts import LIMBS, frame_contact, limb_contact, limb_of_frame
from bvr.human_optim.qmath import normalize_q
from bvr.npz_io import atomic_savez

#: What the labels mean, stamped into the artifact. Shared verbatim with the
#: three-camera ground-truth chain so the two trees are directly comparable.
CONTACT_LABEL_METHOD = (
    "boards-v3: measured |F| with hysteresis; per loaded span, on the limb's "
    "frames within near_scene_m of the wall for >= 1/3 of the span, else its nearest")


def _sha256(path: Path) -> str:
    with path.open("rb") as source_file:
        return hashlib.file_digest(source_file, "sha256").hexdigest()


@torch.no_grad()
def _frame_positions(body, betas: torch.Tensor, q: torch.Tensor, scale: float,
                     *, chunk: int = 64) -> np.ndarray:
    """Pose all frames → contact-frame world positions ``(N, F, 3)``.

    Re-applies the stored global ``scale`` about the pelvis, the same operation
    ``estimate_contacts`` does, so the frames sit at the body's metric size.
    """
    posed = []
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        out = body.forward(betas[None].expand(end - start, -1),
                           normalize_q(q[start:end], body.num_joints))
        root = out.joints[:, :1, :]
        posed.append(root + scale * (out.frames_world[..., :3] - root))
    return torch.cat(posed, 0).cpu().numpy()


def run(predictions_dir: str | Path, config_path: str | Path, *,
        bodies: str = "kinematic_1") -> Path:
    """Write ``contacts_<tag>.npz`` from the boards for one clip."""
    pred = Path(predictions_dir)
    cfg = load_config(config_path)
    measured_cfg = cfg["measured_contacts"]
    enter_n, exit_n = float(measured_cfg["force_enter_n"]), float(measured_cfg["force_exit_n"])
    near_m = float(measured_cfg["near_scene_m"])
    device = cfg.get("runtime", {}).get("device", "cuda")

    tag = bodies.removeprefix("kinematic_")
    body_path = paths.body_path(pred, bodies)
    sensor_path = paths.human_optim_dir(pred) / "sensor_forces.npz"
    scene_path, transform_path = pred / "geometry" / "scene.npz", pred / "geometry" / "transform.npz"
    for needed in (body_path, sensor_path, scene_path):
        if not needed.exists():
            raise SystemExit(f"[measure_contacts] missing {needed}")

    kin = dict(np.load(body_path, allow_pickle=True))
    num_people, num_frames = int(kin["num_people"]), int(kin["num_frames"])
    if num_people != 1:
        raise SystemExit(f"[measure_contacts] {pred.name} tracks {num_people} people; "
                         "the boards measure one climber and cannot say which")
    valid = np.asarray(kin["valid_mask"][0], bool)

    body = load_body(device=device, frames_allow=cfg["contacts"].get("frames") or None,
                     sole_offsets=cfg["contacts"].get("sole_offset_m") or None)
    frame_names = list(body.contact_frame_names)
    if frame_names != [str(x) for x in kin["contact_frame_names"]]:
        raise SystemExit("[measure_contacts] the config's contact frames do not match the body's")
    unowned = [n for n in frame_names if limb_of_frame(n) is None]
    if unowned:
        raise SystemExit(f"[measure_contacts] no board covers {unowned}; restrict "
                         "[contacts].frames to the hands and feet")

    positions = _frame_positions(
        body,
        torch.as_tensor(kin["betas"][0], dtype=torch.float32, device=device),
        torch.as_tensor(kin["q"][0], dtype=torch.float32, device=device),
        float(kin["scale"][0]))
    scene_points = np.asarray(np.load(scene_path)["points"], np.float64)
    distance = cKDTree(scene_points).query(positions.reshape(-1, 3), k=1, workers=-1)[0]
    distance = distance.reshape(num_frames, len(frame_names)).astype(np.float32)

    limb_on, force_world = limb_contact(sensor_path, num_frames, enter_n, exit_n)
    contact, rescued = frame_contact(limb_on, valid, distance, frame_names, near_m)

    frame_limb = [limb_of_frame(name) for name in frame_names]
    for i, limb in enumerate(LIMBS):
        cols = [c for c, owner in enumerate(frame_limb) if owner == limb]
        gaps = distance[:, cols][rescued[:, cols]]
        far = f", far {int(rescued[:, cols].any(1).sum())}f med {np.median(gaps) * 100:.0f} cm" if gaps.size else ""
        print(f"[measure_contacts] {limb}: {int(limb_on[:, i].sum())} loaded frames{far}")
    labelled = distance[contact] * 100
    print(f"[measure_contacts] labelled frames' distance to the scene: "
          + (f"med {np.median(labelled):.1f} p90 {np.percentile(labelled, 90):.1f} cm"
             if labelled.size else "no contact at all"))

    out_path = paths.contacts_path(pred, tag)
    atomic_savez(
        out_path,
        contact_label_schema=np.int32(2),
        contact_label_method=np.str_(CONTACT_LABEL_METHOD),
        frame_contact=contact[None], contact_frame_names=np.array(frame_names),
        frame_parent_joint=np.array([int(x) for x in body.contact_frame_parents], np.int64),
        frame_vertex=np.array([int(x) for x in body.contact_frame_vertices], np.int64),
        frame_limb=np.array([owner or "" for owner in frame_limb]),
        frame_scene_dist_m=distance[None], frame_rescued=rescued[None],
        limb_names=np.array(LIMBS), limb_contact=limb_on[None],
        limb_force_world=force_world[None],
        limb_force_n=np.linalg.norm(force_world, axis=-1)[None],
        force_enter_n=np.float32(enter_n), force_exit_n=np.float32(exit_n),
        near_scene_m=np.float32(near_m),
        object_ids=np.asarray(kin["object_ids"], np.int32), valid_mask=valid[None],
        num_people=np.int32(num_people), num_frames=np.int32(num_frames),
        fps=np.float32(kin["fps"]),
        source_body=np.str_(body_path.name), source_body_sha256=np.str_(_sha256(body_path)),
        source_scene_sha256=np.str_(_sha256(scene_path)),
        source_transform_sha256=np.str_(_sha256(transform_path)),
        source_sensor_sha256=np.str_(_sha256(sensor_path)))
    print(f"[measure_contacts] wrote {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    determinism.apply()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--bodies", default="kinematic_1", choices=["kinematic_1", "kinematic_2"],
                    help="which refined body the frame choice is read on (its contacts tag "
                         "follows: kinematic_1 -> contacts_1, kinematic_2 -> contacts_2)")
    ap.add_argument("--config", type=Path, required=True,
                    help="the dataset's human-optimization TOML ([measured_contacts] thresholds + [contacts] frames)")
    args = ap.parse_args(argv)
    run(args.predictions, args.config, bodies=args.bodies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
