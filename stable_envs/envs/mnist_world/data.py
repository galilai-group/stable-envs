"""The MNIST training digits MNIST World draws, downloaded once and cached.

The files are the four-byte-header IDX archives that torchvision downloads, from the same mirror and
checked against the same MD5 sums. Only the training split is used: every digit of the published
MNIST World episodes, validation included, is an unmodified MNIST training image.
"""

from __future__ import annotations

import contextlib
import functools
import gzip
import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
}
NUM_IMAGES = 60000
DIGIT = 28


def default_dir() -> Path:
    """``$XDG_CACHE_HOME/stable-envs/mnist``, with ``~/.cache`` when XDG_CACHE_HOME is unset."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "stable-envs" / "mnist"


@contextlib.contextmanager
def _lock(directory: Path):
    """Hold an exclusive lock in ``directory`` so concurrent processes download a file once."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(directory / ".lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _fetch(directory: Path, name: str) -> bytes:
    path = directory / name
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        with _lock(directory):
            if not path.exists():
                try:
                    with urllib.request.urlopen(URL + name, timeout=60) as response:
                        data = response.read()
                except OSError as e:
                    raise RuntimeError(
                        f"MNIST World could not download {URL + name} ({e}). Put {name} into {directory}, "
                        "or pass mnist_dir= naming a directory that holds it."
                    ) from e
                partial = path.with_suffix(".part")
                partial.write_bytes(data)
                os.replace(partial, path)
    data = path.read_bytes()
    if hashlib.md5(data).hexdigest() != FILES[name]:
        raise RuntimeError(f"{path} does not match the MD5 of the MNIST release; delete it to download it again")
    return data


def _idx(data: bytes) -> np.ndarray:
    raw = gzip.decompress(data)
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(raw[3])]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * raw[3]).reshape(dims)


@functools.cache
def mnist_train(directory: str) -> tuple[np.ndarray, np.ndarray]:
    """The ``(60000, 28, 28)`` uint8 training images and their ``(60000,)`` labels."""
    root = Path(directory)
    images = _idx(_fetch(root, "train-images-idx3-ubyte.gz"))
    labels = _idx(_fetch(root, "train-labels-idx1-ubyte.gz"))
    assert images.shape == (NUM_IMAGES, DIGIT, DIGIT) and labels.shape == (NUM_IMAGES,)
    return images, labels


@functools.cache
def device_table(directory: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The training images and labels as tensors on ``device``, shared by every env there."""
    images, labels = mnist_train(directory)
    return torch.from_numpy(images.copy()).to(device), torch.from_numpy(labels.astype(np.int64)).to(device)
