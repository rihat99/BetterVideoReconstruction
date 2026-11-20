"""Convert every SAM 3D Body MHR body to SMPL-X by fitting SMPL-X to it, frame by frame.

The MHR body of each fitted frame is posed with BetterHuman's MHR family from the parameters
``sam3d/params.npz`` stores, moved into the camera frame, and carried onto SMPL-X's mesh with MHR's
barycentric map (``bvr.paths.mhr_to_smplx_mapping``). SMPL-X (BetterHuman's family, 10 betas, full
axis-angle hands with their mean pose) is then fitted to that target with Adam, following MHR's
reference converter: the target is centred, then four phases free more and more of the body:

1. 40 steps, global orientation, against the mesh edges;
2. 80 steps, plus body pose and betas;
3. 40 steps, plus the hand poses;
4. ``[smplx].iterations`` steps, everything, against the vertices; the edges stay for the first 50.

Jaw, eyes and expression stay zero: MHR does not predict them. Body models come from
BetterHuman's models folder (``$BETTERHUMAN_MODELS_DIR``).

Writes, under the clip's directory::

    sam3d/smplx_params.npz   bvr-smplx, people stacked as (P, N, ...), NaN where a person was not fitted:
                             betas (10), global_orient (3), body_pose (63), left_hand_pose (45),
                             right_hand_pose (45), transl (3) in the camera frame, jaw_pose, leye_pose,
                             reye_pose (3, zero), expression (10, zero), fit_error (mean vertex distance, m);
                             cam_intrinsics, valid_mask (P, N), object_ids (P,), num_people, num_frames,
                             video_height, video_width, gender, model_type, num_betas, num_expression_coeffs
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import better_human as bh
import numpy as np
import torch
from better_human.families import MHRClassic, SMPLXClassic
from better_robot import optim
from torch.optim.lr_scheduler import MultiStepLR

from bvr import files, paths, run

log = logging.getLogger(__name__)

_POSES = {"global_orient": 3, "body_pose": 63, "left_hand_pose": 45, "right_hand_pose": 45}
_ZERO = {"jaw_pose": 3, "leye_pose": 3, "reye_pose": 3, "expression": 10}


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    """Load both body models and the map between their meshes once; return the per-clip conversion."""
    settings = stage.config["smplx"]
    device = stage.device
    mhr = bh.load("mhr", device=device, density=None)
    smplx = bh.HumanModel.from_spec(bh.families.smplx(gender=settings["gender"], num_betas=10, num_expression=10,
                                                      flat_hand_mean=False, device=device), density=None)
    mapping = np.load(paths.mhr_to_smplx_mapping())
    # Each SMPL-X vertex is a barycentric point of one MHR triangle.
    corners = mhr.spec.faces[torch.as_tensor(mapping["triangle_ids"], device=device)]  # (V, 3)
    weights = torch.as_tensor(mapping["baryc_coords"], dtype=torch.float32, device=device)  # (V, 3)
    edges = _edges(smplx.spec.faces)

    def target(shape: torch.Tensor, parameters: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """SMPL-X-mesh points ``(B, V, 3)`` of MHR bodies, in the camera frame (metres)."""
        with torch.no_grad():
            identity, q, streams = bh.from_classic(mhr, MHRClassic(identity_coeffs=shape, model_parameters=parameters))
            vertices = identity.forward(q, streams).vertices
            # SAM 3D Body's camera frame is MHR's with y and z turned around.
            vertices = vertices * vertices.new_tensor([1.0, -1.0, -1.0]) + translation[:, None]
            return (vertices[:, corners] * weights[..., None]).sum(dim=2)

    def process(clip: paths.Clip) -> None:
        bodies = files.load_npz(clip.find(paths.SAM3D_PARAMS), "bvr-mhr")
        valid = bodies["valid_mask"].astype(bool)
        people, count = valid.shape
        fitted = {name: np.full((people, count, size), np.nan, np.float32) for name, size in
                  {"betas": 10, **_POSES, "transl": 3}.items()}
        fit_error = np.full((people, count), np.nan, np.float32)
        for person in range(people):
            frames = np.flatnonzero(valid[person])
            for start in range(0, len(frames), settings["batch_size"]):
                batch = frames[start:start + settings["batch_size"]]
                goal = target(*(torch.as_tensor(bodies[key][person, batch], device=device)
                                for key in ("shape_params", "mhr_model_params", "pred_cam_t")))
                result, fit_error[person, batch] = _fit(smplx, goal, edges, settings["iterations"])
                for name, value in result.items():
                    fitted[name][person, batch] = value
            log.info("person %d: %d/%d frames, median fit error %.1f mm", int(bodies["object_ids"][person]),
                     len(frames), count, 1000 * float(np.nanmedian(fit_error[person])) if len(frames) else float("nan"))
        files.save_npz(
            clip.path(paths.SMPLX_PARAMS), "bvr-smplx", **fitted,
            **{name: np.zeros((people, count, size), np.float32) for name, size in _ZERO.items()},
            fit_error=fit_error, cam_intrinsics=bodies["cam_intrinsics"].astype(np.float32), valid_mask=valid,
            object_ids=bodies["object_ids"].astype(np.int32), num_people=np.int32(people), num_frames=np.int32(count),
            video_height=np.int32(bodies["video_height"]), video_width=np.int32(bodies["video_width"]),
            gender=np.str_(settings["gender"]), model_type=np.str_("smplx"), num_betas=np.int32(10),
            num_expression_coeffs=np.int32(10))

    return process


def _fit(smplx: bh.HumanModel, goal: torch.Tensor, edges: torch.Tensor,
         iterations: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """SMPL-X fields ``{name: (B, k)}`` fitted to ``goal (B, V, 3)``, and each frame's mean vertex distance ``(B,)``.

    Every frame is its own problem. The goal is centred first, so the template at the origin
    overlaps it before any pose moves; the translation is found in the centred frame and the centre
    added back at the end.
    """
    centre = (goal.amin(dim=1, keepdim=True) + goal.amax(dim=1, keepdim=True)) / 2
    centred = goal - centre
    goal_edges = centred[:, edges[:, 1]] - centred[:, edges[:, 0]]
    problem = optim.Problem({name: goal.new_zeros(len(goal), size) for name, size in {"betas": 10, **_POSES, "transl": 3}.items()})

    def vertices(betas, global_orient, body_pose, left_hand_pose, right_hand_pose, transl):
        identity, q, streams = bh.from_classic(smplx, SMPLXClassic(
            betas=betas, global_orient=global_orient, body_pose=body_pose, left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose, transl=transl))
        return identity.forward(q, streams).vertices

    with_edges = [True]  # the last phase drops the edge term after its 50th step

    def edge(vertices):
        if not with_edges[0]:
            return vertices.new_zeros(len(vertices))
        return ((vertices[:, edges[:, 1]] - vertices[:, edges[:, 0]]) - goal_edges).abs().mean(dim=(1, 2))

    problem.cached(vertices, name="vertices")
    problem.scalar("edge", edge)
    problem.scalar("vertex", lambda vertices: (vertices - centred).square().mean(dim=(1, 2)), weight=0.0)

    def frozen(*names: str) -> dict[str, torch.Tensor]:
        return {name: torch.zeros(problem.variables[name].shape[-1], dtype=torch.bool, device=goal.device)
                for name in names}

    def centre_translation(view: optim.Problem, x: dict) -> tuple[dict, dict]:
        with torch.no_grad():
            return {**x, "transl": x["transl"] + (centred - vertices(**x)).mean(dim=1)}, {}

    coarse = optim.TorchOptim(lambda parameters: torch.optim.Adam(parameters, lr=0.1))
    fine = optim.TorchOptim(lambda parameters: torch.optim.Adam(parameters, lr=0.01),
                      scheduler=lambda optimizer: MultiStepLR(optimizer, [iterations // 3, 2 * iterations // 3], 0.1))
    x = optim.run(problem, [
        optim.Phase(coarse, 40, free=frozen("betas", "body_pose", "transl", "left_hand_pose", "right_hand_pose"), stop_every=0),
        optim.Phase(coarse, 80, free=frozen("transl", "left_hand_pose", "right_hand_pose"), stop_every=0),
        optim.Phase(coarse, 40, free=frozen("transl"), stop_every=0),
        optim.Phase(fine, iterations, weights={"vertex": 1.0}, on_start=centre_translation, stop_every=0,
              on_step=lambda view, state: with_edges.__setitem__(0, state.iteration < 50)),
    ]).x
    x["transl"] = x["transl"] + centre[:, 0]
    with torch.no_grad():
        error = (vertices(**x) - goal).norm(dim=-1).mean(dim=1)
    return {name: value.cpu().numpy() for name, value in x.items()}, error.cpu().numpy()


def _edges(faces: torch.Tensor) -> torch.Tensor:
    """The unique undirected edges ``(E, 2)`` of a triangle mesh, each as sorted vertex ids."""
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).long().sort(dim=1).values
    return torch.unique(edges, dim=0)


if __name__ == "__main__":
    run.main(load)
