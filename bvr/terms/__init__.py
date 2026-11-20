"""What an objective is made of.

A fit is a BetterRobot ``optim.Problem`` whose batch is the people of a clip. A term is a function
of values the problem caches (the trajectory, the posed body, its contact frames) that returns one
loss per person, ``(P,)``: a mean over that person's own tracked frames. ``visual`` holds what the
video says, ``motion`` the frame-rate rule and smoothness, ``scene`` contact with the static scene,
and ``objective`` puts them on one problem.
"""
from collections.abc import Callable

import torch

#: A term: called with the cached values and data its parameters name, it returns the loss ``(P,)``.
Term = Callable[..., torch.Tensor]
