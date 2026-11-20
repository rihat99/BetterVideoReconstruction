"""Time: the one rule for frame rate, the differences it gives, and the terms that hold motion smooth.

Every time derivative of the pipeline steps over a stride of ``h = round(fps / 30)`` frames, so a
derivative always spans about one thirtieth of a second, whatever the clip's rate. A difference is
reported per reference step ``tau0 = 1/30 s``: the change over one stride times ``tau0 / tau``,
where ``tau = h / fps`` is the stride's time. A weight therefore prices "change per thirtieth of a
second", a fixed physical unit, and at 30 fps a difference is the plain difference of neighbouring
frames. A row of a derivative counts only if every frame from its first to its last is tracked.

Tensors carry people first and time second, ``(P, T, ...)``. Every term returns one loss per
person, ``(P,)``: a mean over that person's counted rows.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import better_human as bh
import torch
from better_robot import Model, kinematics

from bvr.terms import Term

#: The frame rate the weights are tuned at (frames per second); ``1 / REFERENCE_RATE`` is ``tau0``.
REFERENCE_RATE = 30.0


@dataclass(frozen=True)
class Time:
    """The time axis of a clip recorded at ``fps`` frames per second."""

    fps: float

    @property
    def stride(self) -> int:
        """``h``: the frames a derivative steps over, the whole number nearest to ``tau0``."""
        return max(1, round(self.fps / REFERENCE_RATE))

    @property
    def step(self) -> float:
        """``tau``: the seconds of one stride."""
        return self.stride / self.fps

    @property
    def rate(self) -> float:
        """``tau0 / tau``: what a change over one stride is multiplied by to read per reference step."""
        return self.fps / (REFERENCE_RATE * self.stride)

    def difference(self, values: torch.Tensor) -> torch.Tensor:
        """The change of ``values (P, T, ...)`` per reference step, ``(P, T - h, ...)``: frame ``t + h`` less ``t``."""
        return (values[:, self.stride:] - values[:, :-self.stride]) * self.rate

    def rows(self, tracked: torch.Tensor, order: int) -> torch.Tensor:
        """The rows of a derivative of ``order`` that count, ``(P, T - order h, ...)`` bool.

        Row ``t`` reads frames ``t`` to ``t + order h`` and counts if ``tracked (P, T, ...)`` holds on every one of
        them, the frames between the strides included.
        """
        return tracked.unfold(1, order * self.stride + 1, 1).all(dim=-1)


def velocity(model: Model, time: Time) -> Callable[[torch.Tensor], torch.Tensor]:
    """Cached value: the tangent step of ``pose (P, T, nq)`` per reference step, ``(P, T - h, nv)``.

    Row ``t`` is ``difference(pose[t], pose[t + h])``: the root's twist in its own frame (linear, angular), then
    three rotation coordinates per joint. ``Time.difference`` of it is the acceleration, and so on.
    """
    def value(pose: torch.Tensor) -> torch.Tensor:
        return kinematics.difference(model, pose[:, :-time.stride], pose[:, time.stride:]) * time.rate
    return value


def mean_square(rows: torch.Tensor, columns: slice) -> Term:
    """Term: the mean square of ``columns`` of a derivative ``(P, R, C)`` over the rows that count, ``rows (P, R)``."""
    def loss(derivative: torch.Tensor) -> torch.Tensor:
        return _mean(derivative[..., columns].square(), rows)
    return loss


def interleave(model: Model, time: Time, tracked: torch.Tensor, columns: slice) -> Term:
    """Term: neighbouring frames lie on one trajectory. Zero at a stride of one frame.

    The strided derivatives tie frame ``t`` to ``t + h`` and never to ``t + 1``, so a clip filmed at 60 fps or more
    is ``h`` trajectories that share only their evidence, and they drift apart: the root zig-zags by millimetres
    between neighbours. The cost is the mean square of ``columns`` of the second difference of ``pose`` over
    neighbouring frames, in m^2 and rad^2: how far a frame sits off the midpoint of its two neighbours, in plain
    frame units. That damps the zig-zag and hardly prices a real acceleration, which is ``h^2`` times smaller
    between neighbours than over a stride. At a stride of one frame the acceleration terms are this term already.
    """
    if time.stride == 1:
        return lambda pose: pose.new_zeros(len(pose))
    rows = tracked.unfold(1, 3, 1).all(dim=-1)

    def loss(pose: torch.Tensor) -> torch.Tensor:
        step = kinematics.difference(model, pose[:, :-1], pose[:, 1:])[..., columns]
        return _mean((step[:, 1:] - step[:, :-1]).square(), rows)
    return loss


def radial_velocity(time: Time, centers: torch.Tensor, tracked: torch.Tensor) -> Term:
    """Term: the mean squared speed of the joints along their viewing rays, in m per reference step, squared.

    Depth along the ray is what one camera sees worst, so that is where a trajectory jitters. The ray, from the
    camera centre ``centers (T, 3)`` to the joint, is held constant: the term slows motion along it and never turns it.
    """
    rows = time.rows(tracked, 1)

    def loss(body: bh.HumanState) -> torch.Tensor:
        joints = body.joint_translations
        ray = torch.nn.functional.normalize(joints[:, :-time.stride] - centers[:-time.stride, None], dim=-1).detach()
        return _mean(((time.difference(joints) * ray).sum(dim=-1)).square(), rows)
    return loss


def contact_velocity(time: Time, *, angular_weight: float) -> Term:
    """Term: a contact frame that stays labelled stays put, in m^2 per reference step squared.

    The cost of a row is ``|step|^2 + angular_weight (1 - <r, r'>^2) (tau0 / tau)^2``. ``frames (P, T, F, 7)`` are
    the contact-frame poses and ``labels (P, T, F)`` say where a frame touches; a row counts where the label holds
    on every frame it reads. ``r`` and ``r'`` are the frame's orientation at the two ends of a stride, so the
    second part is the squared sine of half the turn.
    """
    stride = time.stride

    def loss(frames: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        held = time.rows(labels, 1)
        moved = time.difference(frames[..., :3]).square().sum(dim=-1)
        turned = 1.0 - (frames[:, stride:, :, 3:] * frames[:, :-stride, :, 3:]).sum(dim=-1).square()
        cost = moved + angular_weight * time.rate**2 * turned
        return (cost * held).sum(dim=(1, 2)) / held.sum(dim=(1, 2)).clamp_min(1)
    return loss


def _mean(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Per-person mean ``(P,)`` of ``values (P, R, C)`` over the rows ``(P, R)`` that count; 0 where none does."""
    return (values * rows[..., None]).sum(dim=(1, 2)) / (rows.sum(dim=1) * values.shape[-1]).clamp_min(1)
