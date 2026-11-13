import subprocess
from pathlib import Path


def maintained_paths(root):
    """List indexed paths in a checkout, or all files in an exported tree without its own Git root."""
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
