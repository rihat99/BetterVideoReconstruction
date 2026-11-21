"""Run the reconstruction chain over a video, or over every video of a folder.

    uv run python scripts/pipeline.py --source <video or folder> --save-dir <root>
        [--config <delta.toml>] [--steps a,b] [--upstream <root>] [--skip-existing]
        [--shard i/n] [--dry-run]

Each clip gets its own tree, ``<save-dir>/<stem>/``. The options name what to process; every knob
is a key of ``configs/bvr.toml``, and ``--config`` names a delta over it. The steps, their order
and their files are the table ``bvr.run.STEPS``; without ``--steps`` the chain is ``[run].steps``.

``--upstream`` names another run root: a stage reads from it what this run does not have, and
writes only under ``--save-dir``. ``--skip-existing`` skips a step for a clip while its manifest
says complete and its config, code, libraries and inputs are unchanged. ``--shard i/n`` takes every n-th
clip, so n runs on n GPUs fill one root. The exit code is 0 when every step of every clip is
complete.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bvr import run
from bvr.config import ConfigError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="a video, or a folder of videos (each a clip)")
    parser.add_argument("--save-dir", type=Path, required=True, help="the run root: <save-dir>/<stem>/ per clip")
    parser.add_argument("--config", type=Path, help="a delta over configs/bvr.toml")
    parser.add_argument("--steps", help=f"comma-separated steps, in chain order: {','.join(s.name for s in run.STEPS)}")
    parser.add_argument("--upstream", type=Path, help="a run root to read missing inputs from")
    parser.add_argument("--skip-existing", action="store_true", help="skip steps whose manifest still holds")
    parser.add_argument("--shard", default="1/1", help="'i/n': every n-th clip, starting at the i-th")
    parser.add_argument("--dry-run", action="store_true", help="log the commands and touch nothing")
    options = parser.parse_args()
    index, _, count = options.shard.partition("/")
    if not (index.isdigit() and count.isdigit() and 1 <= int(index) <= int(count)):
        parser.error(f"--shard {options.shard}: expected i/n with 1 <= i <= n")
    try:
        return run.chain(options.source, options.save_dir, config_path=options.config,
                         steps=options.steps.split(",") if options.steps else None, upstream=options.upstream,
                         skip_existing=options.skip_existing, shard=(int(index), int(count)),
                         dry_run=options.dry_run)
    except (ConfigError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
