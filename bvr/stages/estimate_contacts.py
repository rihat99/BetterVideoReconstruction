"""Steps ``contacts_1`` and ``contacts_2``: which contact frames of the fitted bodies touch the scene.

``contacts_1`` reads the bodies of ``kinematics_1`` and ``contacts_2`` those of ``kinematics_2``.
For every person, frame of the clip and contact frame the step asks two witnesses and unions
their answers (:mod:`bvr.contacts`):

    geometry  the contact frame is near the scene and still
    force     at this pose the frame has to carry load for the motion to be possible

Stillness is read from ``drift_pre`` of the body file, the motion ``kinematics_1`` measured before
any contact term acted. The step never measures it again on the body it is given.

The labels are folded onto report joints (a contact frame's parent joint, the wrist for every
frame of a hand), and each folded label gets the fitted probability that an annotator agrees with
it. The file also carries the clip's quality flag, because this is the first step where its three
signals exist: camera travel, depth smear, and the root residual before any force.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import better_human as bh
import numpy as np
import torch

from bvr import body, camera, files, paths, run
from bvr.contacts import confidence, detector, labels, witness
from bvr.geometry import quality
from bvr.geometry.normals import load_scene_cloud
from bvr.terms import dynamics, motion, objective, scene

log = logging.getLogger(__name__)

#: The force witness looks for the scene this far around the contact frames (m). The detector's margin answers
#: "is it touching"; here an empty box has to mean that the body is far from everything that was reconstructed.
SEARCH_MARGIN = 1.0

#: Version of the label arrays, and of the rules that make them; readers refuse another schema.
LABEL_SCHEMA, LABEL_METHOD, CONFIDENCE_METHOD = 2, "frames-v4", "calibrated-v2"


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    human = body.load(stage.device)
    bodies, written = {"contacts_1": (paths.KINEMATIC_1, paths.CONTACTS_1),
                       "contacts_2": (paths.KINEMATIC_2, paths.CONTACTS_2)}[stage.step]
    return lambda clip: contacts(clip, human, stage.config, bodies, written)


def contacts(clip: paths.Clip, human: bh.HumanModel, config: dict, bodies: str, written: str) -> None:
    """Read the contacts of the bodies in the clip's file ``bodies`` and write them to ``written``."""
    options, gate, force = config["contacts"], config["contact_gate"], config["contact_force"]
    spec, names, device = human.spec, human.contact_frames.names, human.spec.device

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, device=device)

    fitted = files.load_npz(clip.find(bodies), "bvr-kinematics")
    transform = files.load_npz(clip.find(paths.TRANSFORM), "bvr-transform")
    static = files.load_npz(clip.find(paths.SCENE), "bvr-scene")
    down = files.load_npz(clip.find(paths.GRAVITY), "bvr-gravity")["gravity_world"]
    time = motion.Time(float(transform["fps"]))
    cloud = scene.Scene(*load_scene_cloud(
        static, camera.centers(tensor(transform["extrinsics"])), conf_drop_q=options["conf_drop_q"],
        k=options["normal_k"], device=device))

    tracked, drift = fitted["valid_mask"].astype(bool), tensor(fitted["drift_pre"])
    identity = dynamics.under_gravity(body.bake(human, tensor(fitted["betas"])), tensor(down))
    state = objective.posed(identity, tensor(fitted["q"]))
    frames = body.contact_frames(state)
    normals, weight = body.frame_normals(frames), dynamics.weight(identity)

    far = torch.stack([detector.scene_signals(*person, cloud, prune_margin_m=SEARCH_MARGIN).nearest
                       for person in zip(frames[..., :3], normals)])
    solved = witness.solve(
        state, tensor(tracked), drift, far, time, window_s=gate["window_s"],
        half_angle_deg=config["friction"]["half_angle_deg"], **{key: force[key] for key in (
            "gap_dstar_m", "stillness_vstar_mps", "magnitude_weight", "iterations", "learning_rate",
            "learning_rate_final")})
    carried = (solved.forces.norm(dim=-1) / weight[:, None, None]).cpu().numpy()
    mute = (~solved.heard | (solved.after / weight[:, None] > force["mute_res_bw"])).cpu().numpy()

    near = [detector.scene_signals(*person, cloud, prune_margin_m=options["prune_margin_m"])
            for person in zip(frames[..., :3], normals)]
    readouts = [
        labels.read(signals, drift[person], frames[person, :, :, :3], carried[person], far[person].cpu().numpy(),
                    mute[person], tracked[person], names=names, fps=time.fps, contacts=options, gate=gate, force=force)
        for person, signals in enumerate(near)]
    for object_id, readout, seen in zip(fitted["object_ids"], readouts, tracked):
        counts = ", ".join(f"{name} {count}" for name, count in zip(names, readout.contact.sum(axis=0)) if count)
        log.info("person %d: contact on %d of %d tracked frames; %s", object_id, readout.contact.any(axis=1).sum(),
                 seen.sum(), counts)
    contact, geometric, loaded = (np.stack(part) for part in zip(*readouts))

    # A drift is evidence only on a tracked frame; the confidence reads the stillest frame of each joint.
    measured = np.where(tracked[..., None], drift.cpu().numpy(), np.nan)
    report = confidence.report_joints(human.contact_frames.parent_joints, spec.joint_names, spec.parents)
    joint_contact = confidence.fold_contacts(contact, report, len(spec.joint_names))
    joint_drift = confidence.fold_drift(measured, report, len(spec.joint_names))
    joint_confidence = np.stack([
        confidence.label_confidence(*person, time.fps, spec.joint_names, config["contact_confidence"])
        for person in zip(joint_contact, joint_drift, tracked)])

    heard = solved.heard.cpu().numpy()
    before = np.where(heard, (solved.before / weight[:, None]).cpu().numpy(), np.nan)
    unexplained = float(np.nanmedian(before)) if heard.any() else float("nan")
    travel = quality.cam_motion_ratio(transform["extrinsics"], static["points"])
    human_points = files.load_npz(clip.find(paths.HUMAN), "bvr-human-points")
    smear, _ = quality.smear_stats(
        human_points["points"], human_points["frame_id"], human_points["obj_id"], transform["extrinsics"])
    flag = quality.quality_flag(travel, unexplained, smear, **config["quality"])
    log.info("quality: camera travel %.2f, smear %.2f, residual before forces %.2f body weights: %s",
             travel, smear, unexplained, flag or "pass")

    def centimetres(metres: torch.Tensor) -> np.ndarray:
        """Centimetres, NaN where nothing was measured: on an untracked frame, or with no scene point near."""
        values = (100.0 * metres).cpu().numpy()
        return np.where(np.isfinite(values) & tracked[..., None], values, np.nan)

    files.save_npz(
        clip.path(written), "bvr-contacts",
        contact_label_schema=np.int32(LABEL_SCHEMA), contact_label_method=np.array(LABEL_METHOD),
        confidence_method=np.array(CONFIDENCE_METHOD),
        frame_contact=contact, frame_contact_geom=geometric, frame_force_label=loaded,
        frame_label_confidence=joint_confidence[..., report], frame_report_joint=report,
        joint_contact=joint_contact, joint_label_confidence=joint_confidence,
        frame_force=solved.forces.cpu().numpy(), body_weight_n=weight.cpu().numpy(),
        residual_force_before_n=np.where(heard, solved.before.cpu().numpy(), np.nan),
        residual_force_after_n=np.where(heard, solved.after.cpu().numpy(), np.nan),
        frame_sdf_cm=centimetres(torch.stack([signals.signed for signals in near])),
        frame_distance_cm=centimetres(torch.stack([signals.nearest for signals in near])),
        frame_drift_m=measured,
        quality_flag=np.array(flag), quality_cam_ratio=np.float32(travel), quality_smear_med=np.float32(smear),
        quality_res_before_bw=np.float32(unexplained),
        contact_frame_names=np.array(names), frame_parent_joint=np.array(human.contact_frames.parent_joints),
        frame_vertex=np.array(human.contact_frames.vertex_ids), joint_names=np.array(spec.joint_names),
        parents=np.array(spec.parents, np.int32), object_ids=fitted["object_ids"], valid_mask=fitted["valid_mask"],
        num_people=np.int32(len(tracked)), num_frames=np.int32(tracked.shape[1]), fps=np.float32(time.fps))


if __name__ == "__main__":
    run.main(load)
