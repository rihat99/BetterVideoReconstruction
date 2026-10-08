"""The body's per-part densities — the table every kindyn inertia is baked with."""

from __future__ import annotations

import pytest
import torch

from tools.body import load_body

# Dempster's segment densities (kg/m³) as shipped in BetterHuman's body_densities.json.
_DEMPSTER = {"pelvis": 1010.0, "left_hip": 1050.0, "spine2": 920.0, "head": 1110.0,
             "right_elbow": 1130.0, "left_wrist": 1160.0}
_FINGER_KEYWORDS = ("index", "middle", "ring", "pinky", "thumb")
# One real clip's fitted shape (../data/examples/out_2/0Zgts-jmjR4_0006), so the check is off-template.
_BETAS = torch.tensor([[-0.73, 0.03, -0.83, -0.06, 0.27, -0.78, -0.60, 0.31, -0.35, -1.40]])


@pytest.fixture(scope="module")
def bodies():
    """The pipeline body (Dempster by default) and the same body at a uniform density."""
    return load_body(), load_body(density=1000.0)


def test_the_default_body_carries_dempster_densities(bodies) -> None:
    body, _ = bodies
    density = body.density
    names = list(body.joint_names)
    assert density.shape == (52,)
    for name, value in _DEMPSTER.items():
        assert float(density[names.index(name)]) == value
    fingers = [i for i, name in enumerate(names) if any(k in name for k in _FINGER_KEYWORDS)]
    assert len(fingers) == 30
    assert all(float(density[i]) == _DEMPSTER["left_wrist"] for i in fingers)   # a finger is hand


def test_kindyn_inertia_is_baked_with_those_densities(bodies) -> None:
    body, uniform = bodies
    dempster = body.shaped(_BETAS).model.values.body_inertias[0]          # (53, 10), row 0 = universe
    plain = uniform.shaped(_BETAS).model.values.body_inertias[0]
    density = body.density
    # Same part volumes, so every packed row (mass, first and second moments) scales
    # with that part's own density — the moment of inertia follows the table too.
    torch.testing.assert_close(dempster[1:] / density[:, None], plain[1:] / 1000.0,
                               rtol=2e-5, atol=2e-5)
    assert float(dempster[1:, 0].sum()) != pytest.approx(float(plain[1:, 0].sum()), rel=1e-3)
