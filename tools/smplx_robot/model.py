"""Load the SMPL-X-mid body used across the pipeline.

The body comes from the rewritten BetterHuman (a ``use_face=False``
:class:`better_human.SMPLX`, 52 joints / ``q (211)``), which is itself built on a
BetterRobot ``Model`` — so the kinematic FK, the Lie ``difference`` and the baked
per-part inertia the dynamics stage needs all come straight off the body
(``mid.shaped(betas).robot``). No separate robot-model build or ``q`` remap is
required: the model uses native ``better_human`` joint order.
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_smplx_mid(model_path: str | Path | None = None, device: str | torch.device = "cpu"):
    """Load the 52-joint SMPL-X body used across the pipeline.

    Redirects to :func:`tools.body.load_body`. ``model_path`` is kept for call-site
    compatibility but ignored — BetterHuman bundles its own validated
    ``SMPLX_NEUTRAL.npz``.
    """
    from tools.body import load_body

    return load_body(device=device)
