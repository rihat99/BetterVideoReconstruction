"""No tracked file names a path on one machine: every path derives from a root of ``bvr.paths``."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: A path that starts at a home, a mount or a data disk; a path relative to some root is fine.
ABSOLUTE = re.compile(r"(?:^|[^\w.~/-])/(?:home|Users|mnt|data\d*)/")


def test_no_tracked_file_holds_an_absolute_path() -> None:
    tracked = subprocess.run(["git", "ls-files", "--", "*.py", "*.toml", "*.md", "*.sh", ":!third_party"],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.split()
    found = [f"{name}:{number}: {line.strip()}"
             for name in tracked if (REPO_ROOT / name).is_file()
             for number, line in enumerate((REPO_ROOT / name).read_text().splitlines(), 1) if ABSOLUTE.search(line)]
    assert not found, "\n".join(found)
