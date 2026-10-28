"""Resolve and compile the MS-Human-700 MuJoCo model (LNSGroup/MS-Human-700, Apache-2.0).

The model is not on PyPI, so first use clones a pinned commit into the cache; set
``STABLE_ENVS_MSHUMAN_HOME`` to an existing checkout to skip the download.
"""

from __future__ import annotations

import os
import subprocess
from functools import cache
from pathlib import Path

import mujoco

from ..._core import environ

_HOME_VAR = "STABLE_ENVS_MSHUMAN_HOME"
_REPO = "https://github.com/LNSGroup/MS-Human-700.git"
_COMMIT = "2d686957aefd5739cf4d2859a6acfd94c8f84400"
_XML = "MS-Human-700-Locomotion.xml"


def home() -> Path:
    override = environ.read(_HOME_VAR)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "stable-envs" / "ms-human-700"


def _ensure(root: Path) -> Path:
    if not (root / _XML).exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", _REPO, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "fetch", "--depth", "1", "origin", _COMMIT], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", _COMMIT], check=True)
    return root


@cache
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(_ensure(home()) / _XML))
