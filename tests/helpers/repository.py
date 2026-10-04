"""Paths in the Git index, or an exported artifact when .git is absent.

Repository tests inspect working contents for these paths. Untracked local work
is not a shipped file and must not be deleted to make package/nav checks pass.
"""

import subprocess
from pathlib import Path


def maintained_paths(root):
    root = Path(root).resolve()
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        top = None
    if top and Path(top).resolve() == root:
        output = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], text=True)
        return {name for name in output.split("\0") if name}
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
