"""Golden frames of our own renderers: fixed render rows per class and an exact comparison."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).parents[1] / "fixtures" / "golden"
DEVICES = ("cuda", "cpu")
SEED = 20261021
N = 8

#: Packages whose version can change our frames. A package that is not installed records None.
PACKAGES = (
    "torch",
    "numpy",
    "triton",
    "mujoco",
    "mujoco-warp",
    "warp-lang",
    "nvdiffrast",
    "pygame",
    "pymunk",
    "jax",
)

CAPTURE = "scripts/testing/capture_golden.py"


def directory(device: str) -> Path:
    return FIXTURES / device


def profile(device: str) -> dict:
    """The versions, platform and device that frames of ``device`` depend on."""
    import torch

    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    out = {"device": device, "versions": versions, "machine": platform.machine()}
    out["gpu"] = torch.cuda.get_device_name() if device == "cuda" else None
    return out


def manifest(device: str) -> dict | None:
    path = directory(device) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def profile_mismatch(device: str, recorded: dict) -> list[str]:
    """Each way this interpreter differs from the recorded profile; empty when it matches."""
    here = profile(device)
    out = [
        f"{k}: recorded {recorded.get(k)!r}, here {here[k]!r}"
        for k in ("machine", "gpu")
        if recorded.get(k) != here[k]
    ]
    for name, want in sorted(recorded.get("versions", {}).items()):
        got = here["versions"].get(name)
        if got != want:
            out.append(f"{name}: recorded {want!r}, here {got!r}")
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, states: np.ndarray, frames: np.ndarray, size: int) -> None:
    np.savez_compressed(path, states=states, frames=frames, size=np.int64(size))


def load(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as data:
        return data["states"], data["frames"], int(data["size"])


def assert_frames(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    assert actual.dtype == expected.dtype == np.uint8, f"{label}: dtype {actual.dtype} vs {expected.dtype}"
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape} vs {expected.shape}"
    if np.array_equal(actual, expected):
        return
    d = np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max(-1)
    rows = sorted({int(i) for i in np.nonzero(d.reshape(len(d), -1).max(1))[0]})
    raise AssertionError(
        f"{label}: {int((d > 0).sum())} pixels differ (max {int(d.max())}/255) in frames {rows}. "
        f"If the render change is intended, capture new frames with {CAPTURE} and review the diff."
    )


def check_class(name: str, device: str) -> None:
    """Render the recorded rows of ``name`` on ``device`` and require the recorded frames exactly."""
    import pytest

    import stable_envs as se

    recorded = manifest(device)
    if recorded is None:
        pytest.skip(f"no golden fixture; run {CAPTURE} --device {device}")
    entry = recorded["files"].get(name)
    if entry is None:
        pytest.skip(f"no golden fixture for {name} on {device}; run {CAPTURE} --device {device}")
    bad = profile_mismatch(device, entry["profile"])
    if bad:
        pytest.skip(f"{name} {device} golden frames were captured on another profile: " + "; ".join(bad))
    path = directory(device) / entry["file"]
    assert sha256(path) == entry["sha256"], f"{path} does not match the sha256 in its manifest"
    states, expected, size = load(path)
    cls = se.registry.REGISTRY[name]
    kw = {} if size == cls.default_size else {"size": size}
    try:
        env = se.make(name, num_envs=len(states), device=device, **kw)
    except ModuleNotFoundError as e:
        pytest.skip(f"{name} needs {e.name}, which this interpreter does not have")
    try:
        actual = env._render(states).cpu().numpy()
    finally:
        env.close()
    assert_frames(actual, expected, f"{name} on {device}")
