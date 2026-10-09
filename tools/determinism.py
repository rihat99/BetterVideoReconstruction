"""Opt-in bit-reproducible GPU runs: ``BVR_DETERMINISTIC=1``.

The stages are not reproducible run to run by default (scatter / index kernels
pick a nondeterministic reduction order; the chain amplifies ~0.5 mm of noise in
``kinematics_1`` to centimetres of joint position by ``kindyn_1``). With the
variable set, :func:`apply` turns on torch's deterministic algorithms (``warn_only``:
an op with no deterministic version runs anyway and warns), disables cuDNN
autotuning, and sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` unless the caller already
chose one — cuBLAS reads that variable when it creates its handle, so :func:`apply`
must run before the first CUDA call: every stage calls it at the top of ``main``.
Measured on the regression clips: two runs are then bit-identical, at no cost in
wall time. Default off.
"""

from __future__ import annotations

import os

import torch


def apply() -> None:
    """Switch deterministic mode on when ``BVR_DETERMINISTIC=1``; a no-op otherwise."""
    if os.environ.get("BVR_DETERMINISTIC") != "1":
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
