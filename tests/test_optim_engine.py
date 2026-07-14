"""Unit tests for the shared optimization engine (tools/optim.py)."""
from __future__ import annotations

import pytest
import torch

from tools.optim import Phase, Problem, State, run_phases


def _quadratic_problem(target: torch.Tensor):
    """A problem whose single leaf should move to ``target``."""

    def forward() -> State:
        st = State()
        st.put("x", problem.leaves["x"])
        return st

    problem = Problem(forward)
    problem.add_leaf("x", torch.zeros_like(target))
    problem.add_term("fit", lambda st: (st["x"] - target).square().mean())
    return problem


def test_adam_phase_converges_to_target():
    target = torch.tensor([1.0, -2.0, 0.5])
    problem = _quadratic_problem(target)
    run_phases(problem, [Phase("fit", 400, {"fit": 1.0})], lr=0.05)
    assert torch.allclose(problem.leaves["x"], target, atol=1e-2)


def test_lbfgs_phase_converges_to_target():
    target = torch.tensor([0.3, 0.7])
    problem = _quadratic_problem(target)
    run_phases(problem, [Phase("fit", 50, {"fit": 1.0}, optimizer="lbfgs")], lr=0.05)
    assert torch.allclose(problem.leaves["x"], target, atol=1e-4)


def test_zero_weight_term_is_never_evaluated():
    calls = {"n": 0}

    def forward() -> State:
        return State()

    problem = Problem(forward)
    x = problem.add_leaf("x", torch.zeros(2))

    def poisoned(st):
        calls["n"] += 1
        return x.sum()

    problem.add_term("live", lambda st: x.square().sum())
    problem.add_term("dead", poisoned)
    run_phases(problem, [Phase("p", 3, {"live": 1.0, "dead": 0.0})], lr=0.1)
    assert calls["n"] == 0


def test_lazy_state_entry_computes_once_and_only_on_demand():
    made = {"n": 0}

    def forward() -> State:
        st = State()

        def make():
            made["n"] += 1
            return torch.ones(3)

        st.put_lazy("shared", make)
        return st

    problem = Problem(forward)
    x = problem.add_leaf("x", torch.zeros(3))
    problem.add_term("a", lambda st: (x - st["shared"]).square().sum())
    problem.add_term("b", lambda st: (x * st["shared"]).sum())
    problem.add_term("c", lambda st: x.sum())

    # both a and b read "shared": one make() per evaluate, not two
    loss, report = problem.evaluate({"a": 1.0, "b": 1.0, "c": 1.0})
    assert made["n"] == 1
    assert set(report) == {"a", "b", "c"}

    # nothing reads it: never made
    made["n"] = 0
    problem.evaluate({"c": 1.0})
    assert made["n"] == 0


def test_grad_mask_freezes_dofs():
    target = torch.tensor([5.0, 5.0])
    problem = _quadratic_problem(target)
    mask = torch.tensor([1.0, 0.0])   # second DOF frozen
    run_phases(problem, [Phase("fit", 200, {"fit": 1.0}, grad_mask={"x": mask})], lr=0.1)
    x = problem.leaves["x"].detach()
    assert torch.allclose(x[0], target[0], atol=1e-2)
    assert float(x[1]) == 0.0


def test_on_start_runs_even_for_zero_iteration_phase():
    ran = []
    problem = _quadratic_problem(torch.zeros(1))
    run_phases(problem, [Phase("noop", 0, {"fit": 1.0}, on_start=lambda: ran.append(True))], lr=0.1)
    assert ran == [True]


def test_all_zero_weights_raise():
    problem = _quadratic_problem(torch.zeros(1))
    with pytest.raises(ValueError):
        problem.evaluate({"fit": 0.0})
