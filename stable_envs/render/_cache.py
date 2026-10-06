import os
from pathlib import Path


def cache_path(*parts: str) -> Path:
    """Resolve a cache location without creating directories."""
    root = os.environ.get("STABLE_ENVS_CACHE")
    if root:
        return Path(root).expanduser().joinpath(*parts)
    base = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    return base.joinpath("stable-envs", *parts)
