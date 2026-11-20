"""Ground contact for BEDLAM2 ground truth: which skin points are on the floor.

BEDLAM2 is the easy case for contact detection and it is worth saying why. The
motion is mocap, so it does not drift or float the way an estimated body does; the
floor is a plane the renderer itself stood everyone on, so it is known exactly
rather than fitted to a point cloud; and the scenes are flat ground, so there is no
geometry to reason about. Three measurements are enough:

* **height** — how far a contact frame sits above the stored floor plane. The band
  is wide (:data:`ENTER_M`), and the reason is the shoes: BEDLAM2 lifts the *whole
  body* by the sole thickness and poses the naked mesh inside a flat shoe, so with the
  heel on the ground the toes ride 5–7 cm up, and a kneeling body's knees and palms
  sit 4–7 cm up, all visibly on the floor in the render.
* **stillness** — the contact frame's speed. It is BetterRobot's: the frame's
  Jacobian times the generalised velocity of the trajectory
  (:func:`generalised_velocity`), averaged over a tenth of a second on whichever side
  of now is calmer (:func:`calm_speed`), so a plant is still right up to its last frame
  and mocap jitter does not flicker a label.
* **facing** — whether the skin at that point looks at the floor. Each contact frame
  carries its own outward surface normal. This is what keeps the wide height band
  honest: the ball of a vertical foot (toes tucked under, kneeling) sits 7–10 cm up
  and perfectly still, and only its normal — 60–75° from straight down — says it is
  not resting on anything. :data:`MAX_TILT_DEG` is set just above the kneeling
  patella (45°) and the sacrum (46°), which are real contacts.

Height alone would label a foot swinging low through mid-stride; stillness alone
would label a hand held motionless in the air; facing alone would label the back of
every hand that passes near the ground. The readout is the contact detector's own
rules (:mod:`bvr.contacts.detector`) wired to a plane instead of a scene cloud: gate
the height by the speed and the facing, then run hysteresis with a minimum duration
so a label is an episode rather than a flicker.

All 35 contact frames are eligible. BEDLAM2's motion library is full of floor work —
yoga, sitting, lying — where the sacrum, back, head and sit bones are genuinely the
contact, so restricting the readout to feet and hands would be wrong here.

Two limits worth knowing before trusting a label:

* **The facing gate is on the authored normal, not the posed skin.** The frames were
  authored on the rest mesh, so a rounded part touches with whatever bit of it faces
  down: with the hand flat the thumb pad faces inward (61–69°), so the thumb is
  nearly never labelled on a flat hand, and toes tucked under lose their pad (55–68°)
  — the toe tip is the contact there, and no frame sits on it. Measured on the 200
  viewer scenes against the posed mesh vertex under each frame (touching when within
  a centimetre of the plane), the gate costs 4.6 points of recall, almost all of it
  the thumb.
* **The floor is one plane per sequence.** A person on an unmodelled step is blind to
  the readout; ``scene_contacts`` reports the share of frames where the whole body is
  between the enter height and :data:`SUSPECT_M` as ``plane_suspect``.
"""
from __future__ import annotations

import math
from pathlib import Path

import better_human as bh
import numpy as np
import torch
from better_robot import kinematics

from bvr import body
from bvr.contacts import detector

#: Window (seconds) the speed is averaged across — a tenth of a second either side of
#: now. Long enough to reject a foot swinging through at walking speed, short enough
#: that a running stance (~0.25 s) fits inside a half-window.
WINDOW_S = 0.20

#: Speed (m/s) at which a point is half in contact; a label dies at twice this. A point provably on the floor reads 0.03 m/s at the median and
#: 0.12 m/s at p90 from mocap jitter and foot roll, so 0.2 m/s is where contact ends.
STILL_MPS = 0.10

#: Height (metres) above the floor to start a contact, and the looser height that
#: holds one. 8 cm is what the data needs: with the heel down and the shoe flat on the
#: ground, the naked big toe sits 5.4–7.5 cm up (scene ``1a000787``), and a body
#: kneeling on the floor has its knees at 4–7 cm and its palms at 5–7 cm
#: (``0j000077``). Between 3.5 and 8 cm the label rate on the 200 viewer scenes goes
#: from 17.8 % to 20.2 % and the toes of that flat shoe from 0 % to 75 % labelled.
ENTER_M = 0.08
EXIT_M = 0.10

#: How far a contact frame's outward normal may lean from straight down. 55° sits just
#: above the kneeling patella (45°) and the sacrum (46°), which the wide height band
#: must keep, and below the ball of a vertical foot (60–75°), which it must not label
#: — at 90° that ball is labelled two thirds of the time while kneeling on tucked toes.
#: The cost is the thumb on a flat hand (61–69°); see the module docstring.
MAX_TILT_DEG = 55.0

#: Shortest contact that is not flicker, in seconds.
MIN_SPAN_S = 0.10

#: How close the body's lowest *mesh vertex* must be to the floor for the body to
#: count as touching it. Only used to report the support check, never to label — and
#: measured on the mesh rather than on the contact frames on purpose, so "was this
#: person on the floor?" and "did the detector find a contact?" stay independent
#: questions. Every touching frame should carry at least one label.
TOUCH_M = 0.01

#: Top of the band where the body is too high to label and too low to be airborne.
#: The support check cannot see into it — it uses the same plane the labels do, so a
#: person standing on an unmodelled step scores as correctly unlabelled. Counting
#: the band is the one number that says "this scene's floor may be wrong".
SUSPECT_M = 0.15

#: The label schema of ``contacts_<n>.npz``, which this file shares.
CONTACT_LABEL_SCHEMA = 2
CONTACT_LABEL_METHOD = "bedlam2-gt-floor-speed-v3"

POSE_CHUNK = 256


def generalised_velocity(model, q: torch.Tensor, *, fps: float) -> torch.Tensor:
    """Tangent-space velocity ``(N, nv)`` of a configuration trajectory ``(N, nq)``, per second.

    The central difference of neighbouring frames on the BetterRobot manifold. At the first
    and last frame it is the neighbour's: a contact readout must not see the clip's edges as
    still.
    """
    step = kinematics.difference(model, q[:-1], q[1:]) * fps         # (N - 1, nv)
    velocity = 0.5 * (step[:-1] + step[1:])                           # (N - 2, nv)
    return torch.cat((velocity[:1], velocity, velocity[-1:]))


def calm_speed(speed: torch.Tensor, *, fps: float, window_s: float) -> torch.Tensor:
    """Speed averaged over the calmer half-window either side of now ``(N, F)``.

    A window centred on ``t`` needs the whole window inside a contact and so erodes
    a foot plant by half a window at each end (a 0.4 s stance at 2 m/s got zero
    frames). Averaging forwards and backwards separately and taking the smaller means
    one side lies inside the contact right up to its last frame. Where the clip has
    no room for a half-window the other side decides alone.
    """
    reach = max(1, int(round(float(window_s) * float(fps) / 2.0)))
    n, device = speed.shape[0], speed.device
    here = torch.arange(n, device=device)
    step = torch.arange(1, reach + 1, device=device)
    infinity = torch.full((), float("inf"), device=device, dtype=speed.dtype)

    def average(index: torch.Tensor, usable: torch.Tensor) -> torch.Tensor:
        return torch.where(usable[:, None], speed[index.clamp(0, n - 1)].mean(dim=1), infinity)

    ahead = average(here[:, None] + step, here + reach <= n - 1)
    behind = average(here[:, None] - step, here - reach >= 0)
    calm = torch.minimum(ahead, behind)
    return torch.where(torch.isfinite(calm), calm, torch.zeros_like(calm))


def frame_positions(identity: bh.Identity, q: torch.Tensor, *, up: torch.Tensor, centroid: torch.Tensor,
                    fps: float, chunk: int = POSE_CHUNK
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pose one person's whole clip and measure everything the readout needs.

    :param identity: one person's body (:func:`bvr.body.bake` of one shape).
    :param q: ``(N, 211)`` world-frame trajectory.
    :returns: ``(frames (N, F, 3), velocities (N, F, 3), normals (N, F, 3),
        mesh_low (N,))`` — the contact frames' world positions and velocities, the
        outward direction the skin faces at each of them, and how far the body's
        **lowest vertex** sits above the floor. The last comes free from the same
        forward pass and is what makes the support check independent of the labels:
        it says whether the body was touching at all, without consulting the 35 points
        the detector looks at.
    """
    names = identity.human.contact_frames.names
    qdot = generalised_velocity(identity.model, q, fps=fps)
    origins = q.new_zeros(len(names), 3)
    parts = []
    for start in range(0, len(q), chunk):
        with torch.no_grad():
            state = identity.forward(q[None, start:start + chunk])
            frames = body.contact_frames(state)[0]
            jacobians = kinematics.point_jacobian(state.model, state.kinematics, frames=names, points=origins)[0]
            parts.append((frames[..., :3], (jacobians @ qdot[start:start + chunk, None, :, None])[..., 0],
                          body.frame_normals(frames), ((state.vertices[0] - centroid) @ up).amin(-1)))
    return tuple(torch.cat(part) for part in zip(*parts))


def contact_from_floor(positions: torch.Tensor, velocities: torch.Tensor,
                       normals: torch.Tensor, *, up: torch.Tensor, centroid: torch.Tensor,
                       fps: float, window_s: float = WINDOW_S, still_mps: float = STILL_MPS,
                       enter_m: float = ENTER_M, exit_m: float = EXIT_M,
                       min_span_s: float = MIN_SPAN_S,
                       max_tilt_deg: float = MAX_TILT_DEG) -> dict[str, torch.Tensor]:
    """Label one person's contact frames against a known floor plane.

    :param positions: contact-frame world positions ``(N, F, 3)``.
    :param velocities: their world linear velocities ``(N, F, 3)``, m/s.
    :param normals: the outward unit normal at each of them ``(N, F, 3)``.
    :param up: unit world up ``(3,)`` — BEDLAM2's is ``-y``.
    :param centroid: any point on the floor plane ``(3,)``.
    :param fps: frames per second, for the speed window and the minimum span.
    :returns: ``contact (N, F)`` bool plus the three measurements behind it —
        ``height (N, F)`` in metres, ``speed (N, F)`` in m/s as the gate read it
        (:func:`calm_speed`), and ``tilt (N, F)`` in degrees from straight down.
    """
    height = (positions - centroid) @ up                         # (N, F) above the floor
    tilt = torch.rad2deg(torch.arccos((normals @ -up).clamp(-1.0, 1.0)))
    speed = calm_speed(velocities.norm(dim=-1), fps=fps, window_s=window_s)
    # A point that moves, or whose skin faces away from the floor, is not resting on it
    # however low it is: it looks infinitely far from the ground, so it can neither start
    # nor hold a span.
    resting = ((speed < 2.0 * still_mps) & (tilt <= max_tilt_deg)).cpu().numpy()
    gated = np.where(resting, height.cpu().numpy(), np.inf)
    contact = detector.min_span(detector.hysteresis(gated < enter_m, gated < exit_m),
                                max(1, math.ceil(min_span_s * fps)))
    return {"contact": torch.as_tensor(contact), "height": height, "speed": speed, "tilt": tilt}


def scene_contacts(gt_dir: Path, *, human: bh.HumanModel, knobs: dict | None = None) -> dict:
    """Label every person in one BEDLAM2 scene.

    :param gt_dir: the scene's ``features/gt/<aa>/<bb>/<scene_id>`` directory.
    :param human: the pipeline's body (:func:`bvr.body.load`), on the device the scene is posed on.
    :param knobs: overrides for the module defaults, by keyword name.
    :returns: the arrays :func:`save_contacts` stores, keyed by npz field name.
    """
    device = human.spec.device
    knobs = dict(knobs or {})
    touch_m = float(knobs.pop("touch_m", TOUCH_M))
    suspect_m = float(knobs.pop("suspect_m", SUSPECT_M))
    meta = np.load(gt_dir / "smplx.npz")
    grav = np.load(gt_dir / "gravity.npz")
    up = torch.as_tensor(np.asarray(grav["up_world"], np.float32), device=device)
    centroid = torch.as_tensor(np.asarray(grav["ground_centroid"], np.float32), device=device)
    fps = float(meta["fps"])
    q, betas = meta["q"], meta["betas"]
    people, frames = q.shape[:2]
    names = list(human.contact_frames.names)

    contact = np.zeros((people, frames, len(names)), bool)
    height = np.zeros((people, frames, len(names)), np.float32)
    speed = np.zeros((people, frames, len(names)), np.float32)
    tilt = np.zeros((people, frames, len(names)), np.float32)
    lowest = np.zeros((people, frames), np.float32)
    for person in range(people):
        identity = body.bake(human, torch.as_tensor(betas[person:person + 1], device=device))
        positions, velocities, normals, mesh_low = frame_positions(
            identity, torch.as_tensor(q[person], device=device), up=up, centroid=centroid, fps=fps)
        out = contact_from_floor(positions, velocities, normals, up=up, centroid=centroid,
                                 fps=fps, **knobs)
        contact[person] = out["contact"].cpu().numpy()
        height[person] = out["height"].cpu().numpy()
        speed[person] = out["speed"].cpu().numpy()
        tilt[person] = out["tilt"].cpu().numpy()
        lowest[person] = mesh_low.cpu().numpy()

    return {
        "contact_label_schema": np.int32(CONTACT_LABEL_SCHEMA),
        "contact_label_method": np.str_(CONTACT_LABEL_METHOD),
        "frame_contact": contact,
        "frame_height_m": height,
        "frame_speed_mps": speed,
        "frame_tilt_deg": tilt,
        "mesh_low_m": lowest,
        "touching": lowest <= touch_m,
        # Frames the readout is blind to: above the enter threshold, below airborne.
        "plane_suspect": float(np.mean((lowest > knobs.get("enter_m", ENTER_M))
                                       & (lowest <= suspect_m))),
        "contact_frame_names": np.array(names),
        "frame_parent_joint": np.array(human.contact_frames.parent_joints, dtype=np.int64),
        "frame_vertex": np.array(human.contact_frames.vertex_ids, dtype=np.int64),
        "object_ids": np.asarray(meta["object_ids"], np.int32),
        "num_people": np.int32(people), "num_frames": np.int32(frames),
        "fps": np.float32(fps),
        "window_s": np.float32(knobs.get("window_s", WINDOW_S)),
        "still_mps": np.float32(knobs.get("still_mps", STILL_MPS)),
        "enter_m": np.float32(knobs.get("enter_m", ENTER_M)),
        "exit_m": np.float32(knobs.get("exit_m", EXIT_M)),
        "max_tilt_deg": np.float32(knobs.get("max_tilt_deg", MAX_TILT_DEG)),
        "min_span_s": np.float32(knobs.get("min_span_s", MIN_SPAN_S)),
        "touch_m": np.float32(touch_m),
        "suspect_m": np.float32(suspect_m),
    }


def save_contacts(path: Path, fields: dict) -> None:
    """Write ``gt/contacts.npz`` — the label schema the viewer already reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **fields)


__all__ = ["calm_speed", "contact_from_floor", "frame_positions", "generalised_velocity", "save_contacts",
           "scene_contacts"]
