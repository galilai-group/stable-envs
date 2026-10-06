"""Sealed numeric trajectories for one registration.

State arrays have shape [T+1, B, ...]; actions and task outputs use [T, B, ...].
The manifest records source and producer identities. Loading verifies hashes
and schema; registration acceptance requires separate comparisons and gates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from .evidence import EvidenceRun, verify_seal

FORMAT = "stable-envs.state-task-trajectory.v1"
TASKS = {"reward", "terminated", "truncated", "observation", "info_state"}


def _validate(metadata, arrays):
    if metadata.get("format") != FORMAT:
        raise ValueError("Unknown trajectory format")
    if not isinstance(metadata.get("registration"), str) or not metadata["registration"]:
        raise ValueError("Missing registration")
    for key in ("batch", "steps"):
        if type(metadata.get(key)) is not int or metadata[key] < 1:
            raise ValueError(f"Invalid {key}")
    if type(metadata.get("seed")) is not int:
        raise ValueError("Missing seed")
    for identity in ("source", "producer"):
        value = metadata.get(identity, {})
        if not all(value.get(key) for key in ("commit", "files")):
            raise ValueError(f"Missing {identity} identity")
    if not metadata["source"].get("versions"):
        raise ValueError("Missing source versions")
    batch, steps = metadata["batch"], metadata["steps"]
    if not {"actions", "reward", "terminated", "truncated"} <= arrays.keys():
        raise ValueError("Missing action/task arrays")
    actions = arrays["actions"]
    if actions.ndim != 3 or actions.shape[:2] != (steps, batch) or actions.shape[2] < 1:
        raise ValueError("Actions must have shape [T,B,A]")
    leaves = set()
    for key, value in arrays.items():
        if value.dtype.kind not in "buif" or (value.dtype.kind == "f" and not np.isfinite(value).all()):
            raise ValueError(f"Non-numeric or non-finite array: {key}")
        if key.startswith("state/"):
            path = tuple(key.split("/")[1:])
            if any(not p or p in {".", ".."} for p in path):
                raise ValueError(f"Invalid State path: {key}")
            leaves.add(path)
            if value.ndim < 2 or value.shape[:2] != (steps + 1, batch):
                raise ValueError(f"State must have [T+1,B,...] shape: {key}")
        elif key in TASKS:
            if value.shape[:2] != (steps, batch):
                raise ValueError(f"Task must have [T,B,...] shape: {key}")
            if key in {"reward", "terminated", "truncated"} and value.shape != (steps, batch):
                raise ValueError(f"Scalar task column required: {key}")
            if key in {"terminated", "truncated"} and value.dtype != np.bool_:
                raise ValueError(f"Boolean task column required: {key}")
        elif key != "actions":
            raise ValueError(f"Unknown array: {key}")
    if not leaves or any(path[:i] in leaves for path in leaves for i in range(1, len(path))):
        raise ValueError("Empty or overlapping State paths")
    expected = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()}
    if metadata.get("arrays") != expected:
        raise ValueError("Trajectory schema mismatch")


@dataclass(frozen=True)
class Trajectory:
    metadata: dict
    arrays: dict[str, np.ndarray]

    def state(self, index, *, device="cpu"):
        """Copy one time index into an owned batched State on the requested device."""
        if type(index) is not int or not 0 <= index <= self.metadata["steps"]:
            raise IndexError(index)
        result = TensorDict({}, batch_size=[self.metadata["batch"]], device=device)
        for key, value in self.arrays.items():
            if key.startswith("state/"):
                result[tuple(key.split("/")[1:])] = torch.tensor(value[index], device=device)
        return result


def write_trajectory(directory, *, registration, seed, source, producer, arrays):
    """Validate arrays and write a sealed corpus in a new directory; never replace evidence."""
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    actions = arrays.get("actions")
    if actions is None or actions.ndim != 3:
        raise ValueError("Actions must have shape [T,B,A]")
    metadata = dict(
        format=FORMAT,
        registration=registration,
        seed=seed,
        batch=actions.shape[1],
        steps=actions.shape[0],
        source=source,
        producer=producer,
        arrays={key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
    )
    _validate(metadata, arrays)
    run = EvidenceRun(directory)
    run.json("manifest.json", metadata)
    run.arrays("trajectory.npz", **arrays)
    run.seal()
    return load_trajectory(directory, registration=registration)


def load_trajectory(directory, *, registration):
    """Verify hashes, registration, and schema; return read-only arrays without pickle."""
    root = Path(directory)
    entries = verify_seal(root)
    if set(entries) != {"manifest.json", "trajectory.npz"}:
        raise ValueError("Unexpected corpus files")
    metadata = json.loads((root / "manifest.json").read_text())
    if metadata.get("registration") != registration:
        raise ValueError("Corpus registration mismatch")
    with np.load(root / "trajectory.npz", allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Duplicate archive fields")
        arrays = {key: archive[key] for key in archive.files}
    _validate(metadata, arrays)
    for value in arrays.values():
        value.setflags(write=False)
    return Trajectory(metadata, arrays)
