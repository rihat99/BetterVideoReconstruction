"""Gemma 4 VLM triage / contact-graph stage — PLACEHOLDER.

Intended: run a Gemma-4 vision-language model over each clip (frames + the
SAM 3 ``sam3/bboxes.npz`` / ``sam3/``) to produce a triage verdict and/or a
human–object contact graph. The contact graph is the natural upstream for
``build_robot_motion.py``'s ``joint_contact`` input (which today must be
hand-supplied), closing the loop to the dynamics stage.

**Not yet implemented** — running this stage raises ``NotImplementedError``.
The ``GemmaVLM`` adapter (``tools/adapters/gemma_vlm.py``) is in place with the
full ``__init__ / generate(text, images, *, system, enable_thinking, ...) ->
(thinking, answer) / release`` interface; the prompt/parse layer and the output
schema remain to be written.

Per-item I/O (planned)::

    reads:  the source video / folder of frames, sam3/bboxes.npz, sam3/
    writes: gemma/result.json   (schema TBD — triage + contact edges)

Standalone CLI (once implemented)::

    uv run python scripts/stages/classify_gemma.py \\
        --source video.mp4 --save-dir out/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.precompute_runner import InputItem, add_standard_args, run_batch

_NOT_IMPLEMENTED = (
    "classify_gemma is a placeholder. The GemmaVLM adapter "
    "(tools/adapters/gemma_vlm.py) is ready, but the prompt/parse layer and "
    "output schema are not implemented yet. See the module docstring."
)


class GemmaClassifyStep:
    """PLACEHOLDER Gemma-4 triage / contact-graph stage (see module docstring)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def load(self) -> None:
        # Fail fast — do NOT download ~30 GB of Gemma weights for a stub.
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def process(self, item: InputItem) -> None:  # pragma: no cover - unreachable
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def release(self) -> None:
        pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(p)
    p.add_argument("--model-id", default="google/gemma-4-31B-it",
                   help="Gemma 4 VLM model id (default: %(default)s)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_batch(GemmaClassifyStep(args), args)


if __name__ == "__main__":
    raise SystemExit(main())
