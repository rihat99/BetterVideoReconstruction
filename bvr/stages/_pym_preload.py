"""LD_PRELOAD shim for pymomentum PyPI wheels.

PyPI `pymomentum-gpu` / `pymomentum-cpu` wheels are built against GCC 14 and
need `GLIBCXX_3.4.33`+ at runtime. The wheel ships the required libstdc++ as
`libstdc++-pymomentum.so.6`, but the system's older libstdc++ (e.g. GCC 12 on
Ubuntu 22.04 only exports up to `GLIBCXX_3.4.30`) still gets loaded alongside
it via torch/libc, and mixing the two causes a segfault deep inside
`momentum::addBlendShapeParameters` when `Character.with_blend_shape` is
called (see facebookresearch/momentum#939).

Fix: preload the wheel's bundled libstdc++ as `libstdc++.so.6` via
`LD_PRELOAD` so every subsequent symbol lookup resolves against the newer
ABI. Import this module before any `torch` / `pymomentum` / `mhr` import.

If `LD_PRELOAD` is already set (or the wheel is not installed), this is a
no-op. On first run it re-execs the current Python process with the preload
appended, so callers see a single deterministic interpreter.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

_MARKER_ENV = "_PYM_PRELOAD_DONE"


def _find_bundled_libstdcxx() -> Path | None:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    for libs_dir in ("pymomentum_gpu.libs", "pymomentum_cpu.libs"):
        cand = site_packages / libs_dir / "libstdc++-pymomentum.so.6"
        if cand.exists():
            return cand
    return None


def ensure_preloaded() -> None:
    if os.environ.get(_MARKER_ENV) == "1":
        return
    lib = _find_bundled_libstdcxx()
    if lib is None:
        os.environ[_MARKER_ENV] = "1"
        return

    existing = os.environ.get("LD_PRELOAD", "")
    lib_str = str(lib)
    if lib_str in existing.split(":"):
        os.environ[_MARKER_ENV] = "1"
        return

    env = os.environ.copy()
    env["LD_PRELOAD"] = f"{lib_str}:{existing}" if existing else lib_str
    env[_MARKER_ENV] = "1"
    # A stage run as ``python -m bvr.stages.<name>`` must come back the same way.
    spec = getattr(sys.modules["__main__"], "__spec__", None)
    argv = [sys.executable, "-m", spec.name, *sys.argv[1:]] if spec else [sys.executable, *sys.argv]
    os.execvpe(sys.executable, argv, env)


ensure_preloaded()
