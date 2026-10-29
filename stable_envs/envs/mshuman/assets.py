"""Resolve and compile the MS-Human-700 MuJoCo model (LNSGroup/MS-Human-700, Apache-2.0).

The model is not on PyPI, so first use downloads the pinned commit into the cache; set
``STABLE_ENVS_MSHUMAN_HOME`` to an existing checkout to skip the download.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import tarfile
import tempfile
import urllib.request
from functools import cache
from pathlib import Path

import mujoco
from filelock import FileLock

from ..._core import environ
from ..._core.cache import cache_dir

_HOME_VAR = "STABLE_ENVS_MSHUMAN_HOME"
_COMMIT = "2d686957aefd5739cf4d2859a6acfd94c8f84400"
_ARCHIVE = f"https://github.com/LNSGroup/MS-Human-700/archive/{_COMMIT}.tar.gz"
_SHA256 = "3fefdd72d7b24e9af7d65124bae9707089675e5e2bf2d0b1905d62f65e6df3b3"
_XML = "MS-Human-700-Locomotion.xml"


def home() -> Path:
    override = environ.read(_HOME_VAR)
    if override:
        return Path(override).expanduser()
    return cache_dir() / "ms-human-700"


def _ensure(root: Path) -> Path:
    if (root / _XML).exists():
        return root
    if environ.read(_HOME_VAR):
        raise FileNotFoundError(f"{_HOME_VAR}={root} has no {_XML}; point it at an MS-Human-700 checkout or unset it")
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(root.with_name(f"{root.name}.lock")):
        if not (root / _XML).exists():
            with urllib.request.urlopen(_ARCHIVE, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != _SHA256:
                raise RuntimeError(f"{_ARCHIVE} does not match its pinned sha256; set {_HOME_VAR} to a checkout")
            with tempfile.TemporaryDirectory(dir=root.parent, prefix=f".{root.name}-") as tmp:
                tarfile.open(fileobj=io.BytesIO(data)).extractall(tmp, filter="data")
                shutil.rmtree(root, ignore_errors=True)
                Path(tmp, f"MS-Human-700-{_COMMIT}").replace(root)
    return root


@cache
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(_ensure(home()) / _XML))
