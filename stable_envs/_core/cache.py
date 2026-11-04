"""The stable-envs cache directory, shared by the tuning cache and model downloads."""

from __future__ import annotations

import os
from pathlib import Path


def cache_dir() -> Path:
    """Return the ``stable-envs`` cache root: ``$XDG_CACHE_HOME/stable-envs``, or ``~/.cache/stable-envs`` when unset."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "stable-envs"
