"""Shared optimization engine for the body-fitting stages.

A stage describes its fit as a :class:`Problem`: named leaf tensors, one
``forward`` that poses the body once per iteration and returns a :class:`State`
of shared products, and named loss **terms** that read that state. A list of
:class:`Phase` rows then drives the fit — each phase brings its own weight
column, iteration budget, optimizer (Adam or L-BFGS) and trainable-DOF masks.
:func:`run_phases` owns the loop: weighted sum, backward, gradient masking,
stepping, per-term logging. Loss math stays in the stage modules; nothing here
knows about bodies, cameras or scenes.

A term is active in a phase iff its weight there is non-zero, so a config's
per-stage weight column *is* the objective definition. Terms that share an
expensive intermediate (the scene nearest-neighbour pass, the tangent-difference
chain) read it from a lazy :class:`State` entry, computed at most once per
iteration and only when some active term asks for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


class State:
    """Per-iteration products shared by the loss terms.

    ``put`` stores an eager value; ``put_lazy`` a maker called at most once, the
    first time a term reads that name (so an intermediate no active term needs is
    never computed). Read with ``state["name"]``.
    """

    __slots__ = ("_values", "_makers")

    def __init__(self):
        self._values: dict[str, object] = {}
        self._makers: dict[str, Callable[[], object]] = {}

    def put(self, name: str, value) -> None:
        self._values[name] = value

    def put_lazy(self, name: str, make: Callable[[], object]) -> None:
        self._makers[name] = make

    def __getitem__(self, name: str):
        if name not in self._values:
            self._values[name] = self._makers.pop(name)()
        return self._values[name]


#: A loss term: reads the shared :class:`State`, returns the raw scalar loss —
#: optionally with a dict of detached diagnostic scalars for logging.
TermFn = Callable[[State], "torch.Tensor | tuple[torch.Tensor, dict]"]


class Problem:
    """Named leaf variables + a shared forward + named loss terms."""

    def __init__(self, forward: Callable[[], State]):
        self.leaves: dict[str, torch.Tensor] = {}
        self.terms: dict[str, TermFn] = {}
        self._forward = forward

    def add_leaf(self, name: str, init: torch.Tensor) -> torch.Tensor:
        """Register a trainable leaf starting at ``init``; returns the leaf tensor."""
        leaf = init.detach().clone().requires_grad_(True)
        self.leaves[name] = leaf
        return leaf

    def add_term(self, name: str, fn: TermFn) -> None:
        """Register a loss term. Registration order fixes the summation order."""
        self.terms[name] = fn

    def evaluate(self, weights: dict[str, float]) -> tuple[torch.Tensor, dict]:
        """One forward + the weighted sum of the active (non-zero-weight) terms.

        Returns ``(loss, report)`` — ``report`` maps each active term to
        ``(raw_value_tensor, weight, diagnostics_dict)``; tensors are converted to
        floats only at log time, so no per-iteration GPU sync sneaks in.
        """
        state = self._forward()
        total = None
        report: dict[str, tuple[torch.Tensor, float, dict]] = {}
        for name, fn in self.terms.items():
            w = float(weights.get(name, 0.0))
            if w == 0.0:
                continue
            out = fn(state)
            value, info = out if isinstance(out, tuple) else (out, {})
            total = w * value if total is None else total + w * value
            report[name] = (value, w, info)
        if total is None:
            raise ValueError("no active loss term (every weight is zero)")
        return total, report


@dataclass
class Phase:
    """One optimization phase over a :class:`Problem`.

    ``weights`` is this phase's ``{term: weight}`` column (0 ⇒ the term is off
    here). ``grad_mask`` multiplies each named leaf's gradient after backward —
    zeros freeze DOFs without touching the optimizer, under Adam **and** L-BFGS.
    ``on_start`` runs once before the phase's first iteration (build scene
    candidates, detect labels…) even when ``iters`` is 0.
    """

    name: str
    iters: int
    weights: dict[str, float]
    optimizer: str = "adam"          # "adam" | "lbfgs"
    grad_mask: dict[str, torch.Tensor] | None = None
    on_start: Callable[[], None] | None = None


def _apply_grad_mask(problem: Problem, grad_mask: dict[str, torch.Tensor] | None) -> None:
    """Zero the frozen coordinates of each masked leaf's gradient, in place."""
    if not grad_mask:
        return
    for name, mask in grad_mask.items():
        grad = problem.leaves[name].grad
        if grad is not None:
            grad.mul_(mask)


def _log(prefix: str, phase: str, it: int, loss: torch.Tensor, report: dict) -> None:
    parts = []
    for name, (value, w, info) in report.items():
        v = float(value.detach())
        extras = "".join(f" {k}={float(x):.4g}" for k, x in info.items())
        parts.append(f"{name}={v:.4f}·{w:g}={w * v:.3f}{extras}")
    print(f"{prefix}[{phase:9s}] it {it:4d}  loss={float(loss.detach()):.4f}  ({'  '.join(parts)})")


def run_phases(
    problem: Problem,
    phases: list[Phase],
    *,
    lr: float,
    lr_decay: float | None = None,
    lr_lbfgs: float = 1.0,
    lbfgs_history: int = 10,
) -> None:
    """Run ``phases`` in order, mutating the problem's leaves in place.

    Adam phases get a fresh optimizer per phase (no stale momentum leaks across a
    DOF unfreeze) with a MultiStep decay at 1/2 and 3/4 of the budget when
    ``lr_decay`` is given. L-BFGS phases run one strong-Wolfe ``step`` whose
    ``max_iter`` is the phase budget. Logging: every quarter of an Adam phase,
    every 5th L-BFGS closure call.
    """
    params = list(problem.leaves.values())
    for ph in phases:
        if ph.on_start is not None:
            ph.on_start()
        iters = int(ph.iters)
        if iters <= 0:
            continue
        if ph.optimizer == "lbfgs":
            opt = torch.optim.LBFGS(params, lr=lr_lbfgs, max_iter=iters,
                                    history_size=lbfgs_history, line_search_fn="strong_wolfe")
            step = [0]

            def closure(ph=ph, opt=opt, step=step):
                opt.zero_grad()
                loss, report = problem.evaluate(ph.weights)
                loss.backward()
                _apply_grad_mask(problem, ph.grad_mask)
                step[0] += 1
                if step[0] == 1 or step[0] % 5 == 0:
                    _log("    ", ph.name, step[0], loss, report)
                return loss

            opt.step(closure)
            continue
        if ph.optimizer != "adam":
            raise ValueError(f"unknown optimizer {ph.optimizer!r} (adam | lbfgs)")
        opt = torch.optim.Adam(params, lr=lr)
        sched = (torch.optim.lr_scheduler.MultiStepLR(
                    opt, milestones=[iters // 2, 3 * iters // 4], gamma=lr_decay)
                 if lr_decay is not None else None)
        for it in range(iters):
            opt.zero_grad()
            loss, report = problem.evaluate(ph.weights)
            loss.backward()
            _apply_grad_mask(problem, ph.grad_mask)
            opt.step()
            if sched is not None:
                sched.step()
            if it % max(1, iters // 4) == 0 or it == iters - 1:
                _log("    ", ph.name, it, loss, report)
