"""The kinematic objective: one problem that holds every value and term of a kinematic fit.

A fit is a BetterRobot ``optim.Problem`` whose batch is the people of a clip. Its one variable is
the trajectory ``q (P, T, 211)``; a term is a function of cached values that returns one loss per
person. A term is off until a phase's weight table names it, so the tables of ``configs/bvr.toml``
are the objective of each stage. What a phase holds fixed is data: the scene ``Candidates`` near
the body (``scene``), the contact ``labels`` and, for ``kinematics_2``, the wall ``patches``.
"""
from __future__ import annotations

from functools import partial
from itertools import pairwise

import better_human as bh
import torch
from better_robot import optim, terms

from bvr import body, camera
from bvr.terms import motion, scene, visual

#: The terms that read the scene or the labels; a phase that runs before either exists leaves them off.
CONTACT_TERMS = ("penetration", "contact_attraction", "contact_acquisition", "contact_velocity")


def kinematics_problem(
    identity: bh.Identity, start: torch.Tensor, *, keypoints: torch.Tensor, scores: torch.Tensor,
    tracked: torch.Tensor, cameras: camera.Cameras, time: motion.Time, attraction: dict, config: dict,
    cloud: tuple[torch.Tensor, torch.Tensor] | None = None, variables: dict[str, torch.Tensor] | None = None,
) -> optim.Problem:
    """The problem of fitting ``identity``'s trajectory to a clip, from ``start (P, T, 211)``, before it is solved.

    :param keypoints: Sapiens keypoints ``(P, T, 308, 2)`` in pixels, with ``scores (P, T, 308)``.
    :param tracked: ``(P, T)`` bool, the frames each person is tracked on.
    :param cameras: ``(extrinsics (T, 4, 4), intrinsics (T, 3, 3))``.
    :param attraction: the band of the attraction term, ``[auto_contact]`` or ``[contact_attraction]``.
    :param config: the config tables of the terms.
    :param cloud: each person's depth points ``(P, T, M, 3)`` and the mask ``(P, T, M)`` of the real ones
        (``visual.human_points``); without it the problem has no ``chamfer`` term.
    :param variables: more variables ``(P, ...)`` beside ``q``, for the terms a caller registers (contact forces).

    Cached values: ``pose`` (``q`` with unit quaternions), ``body`` (its ``HumanState``), ``frames`` (the contact
    frames), ``velocity`` to ``snap`` (the strided derivatives of ``pose``) and ``distance`` (the frames against
    the scene). A caller adds a term by registering it, and sets ``scene``, ``labels`` and ``patches`` with
    ``Problem.with_(data=)`` or from a phase's ``on_start``.
    """
    human, penetration = identity.human, config["penetration"]
    with torch.no_grad():
        start_body = identity.forward(start)
    problem = optim.Problem({"q": start, **(variables or {})}, data={"scene": None, "labels": None, "patches": None}, time=("q",))
    problem.cached(terms.RobotManifold(human.model).project, name="pose", reads=("q",))
    problem.cached(bh.terms.human_state(identity, q="pose"), name="body")
    problem.cached(body.contact_frames, name="frames", reads=("body",))
    problem.cached(motion.velocity(human.model, time), name="velocity")
    for lower, higher in pairwise(("velocity", "acceleration", "jerk", "snap")):
        problem.cached(time.difference, name=higher, reads=(lower,))
    problem.cached(scene.signed_distance(config["chamfer"]["chunk"]), name="distance")

    term = partial(problem.scalar, weight=0.0)
    term("reprojection_keypoints", visual.keypoint_reprojection(
        human, keypoints, scores, tracked, cameras, time.fps, **config["keypoints"]))
    term("reprojection_start", visual.start_reprojection(
        start_body, tracked, cameras, time.fps, **config["reprojection_start"]))
    if cloud is not None:
        term("chamfer", visual.chamfer(human, *cloud, cameras, **{
            key: config["chamfer"][key]
            for key in ("sat_m", "vertex_subsample", "emphasis_lo", "emphasis_hi", "emphasis_radius", "chunk")}))
    term("pose_prior", visual.pose_prior(start))
    term("orientation_prior", visual.orientation_prior(start))
    for order, name in enumerate(("acceleration", "jerk", "snap"), start=2):
        rows = time.rows(tracked, order)
        term(f"{name}_root", motion.mean_square(rows, slice(0, 6)), reads=(name,))
        term(f"{name}_joint", motion.mean_square(rows, slice(6, None)), reads=(name,))
    for part, columns in (("root", slice(0, 6)), ("joint", slice(6, None))):
        term(f"interleave_{part}", motion.interleave(human.model, time, tracked, columns), reads=("pose",))
    term("radial_velocity", motion.radial_velocity(time, camera.centers(cameras[0]), tracked))
    term("contact_velocity", motion.contact_velocity(time, **config["contact_velocity"]))
    term("penetration", scene.penetration(**{
        key: penetration[key] for key in ("conf_thresh", "max_dist_m", "max_pen_m", "margin_m", "huber_m")}))
    term("contact_attraction", scene.attraction(**{
        key: attraction[key] for key in ("conf_thresh", "max_dist_m", "band_m")}))
    term("contact_acquisition", scene.acquisition(delta_m=config["contact_acquisition"]["delta_m"]))
    return problem


@torch.no_grad()
def posed(identity: bh.Identity, q: torch.Tensor) -> bh.HumanState:
    """The body at ``q (P, T, 211)`` as the fit sees it: unit quaternions, no gradient."""
    return identity.forward(terms.RobotManifold(identity.human.model).project(q))


def reach(state: bh.HumanState) -> torch.Tensor:
    """The points ``(P, T, K, 3)`` a body's scene candidates are drawn around: its joints and contact frames."""
    return torch.cat((state.joint_translations, body.contact_frames(state)[..., :3]), dim=-2)
