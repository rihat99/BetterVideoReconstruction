"""What the stage problems share on top of BetterRobot's optimizer.

A stage describes its fit as a BetterRobot ``Problem`` — named variables with a
batch axis of one, one ``cached("st", forward)`` that returns a lazy :class:`State`
of shared products, named scalar terms reading it — and a list of ``Phase`` rows;
``better_robot.optim.run`` owns the loop (optimizers, gradient masks, per-term
diagnostics). This module holds the four pieces every such fit needs: the lazy
:class:`State`, the term registration (``weight=0`` — a phase's weight column
switches terms on), the ``TorchOptim`` factories (Adam with the half / three-quarter
decay, strong-Wolfe L-BFGS as one step) and the per-term logger.
"""

from __future__ import annotations

from typing import Callable

import torch
from better_robot.optim import Phase, Problem, TorchOptim


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


def add_term(problem: Problem, name: str, fn: TermFn) -> None:
    """Register ``fn(st)`` as a scalar term, off until a phase's weight column names it."""
    problem.scalar(name, fn, weight=0.0, reads=("st",))


def phase_weights(problem: Problem, column: dict[str, float]) -> dict[str, float]:
    """``column`` over the registered terms (unlisted = 0); a phase must have an active term."""
    weights = {name: float(column.get(name, 0.0)) for name in problem.terms}
    if not any(weights.values()):
        raise ValueError("no active loss term (every weight is zero)")
    return weights


def adam(lr: float, iters: int, lr_decay: float | None = None) -> TorchOptim:
    """Adam; with ``lr_decay`` a MultiStep decay at 1/2 and 3/4 of the ``iters`` budget."""
    scheduler = None if lr_decay is None else (lambda opt: torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[iters // 2, 3 * iters // 4], gamma=lr_decay))
    return TorchOptim(torch.optim.Adam, lr=lr, scheduler=scheduler)


def lbfgs(iters: int, lr: float = 1.0, history: int = 10, **kwargs) -> TorchOptim:
    """One strong-Wolfe ``torch.optim.LBFGS`` call of ``iters`` iterations: run it as a 1-step phase."""
    return TorchOptim(torch.optim.LBFGS, lr=lr, max_iter=iters, history_size=history,
                      line_search_fn="strong_wolfe", **kwargs)


def term_logger(name: str, iters: int) -> Callable:
    """``Phase.on_step`` printing the loss and every active term each quarter of the phase.

    Values come from the step's own evaluation (``State.extra["terms"]`` / ``["info"]``),
    weighted; the raw value is recovered by dividing the weight back out.
    """
    def on_step(view: Problem, state) -> None:
        it = state.it - 1
        if it % max(1, iters // 4) != 0 and it != iters - 1:
            return
        parts = []
        for term, weighted in state.extra["terms"].items():
            w = float(view.weights.get(term, view.terms[term].weight))
            wv = float(weighted)
            info = state.extra.get("info", {}).get(term, {})
            extras = "".join(f" {k}={float(x):.4g}" for k, x in info.items())
            parts.append(f"{term}={wv / w:.4f}·{w:g}={wv:.3f}{extras}")
        print(f"    [{name:9s}] it {it:4d}  loss={float(state.loss):.4f}  ({'  '.join(parts)})")
    return on_step


__all__ = ["Phase", "Problem", "State", "TermFn", "adam", "add_term", "lbfgs", "phase_weights", "term_logger"]
