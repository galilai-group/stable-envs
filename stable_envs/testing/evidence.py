from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(root, name):
    path = Path(name)
    if (
        not name
        or any(c in name for c in "\n\r\0")
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != name
    ):
        raise ValueError(f"Invalid artifact path: {name!r}")
    target = Path(root) / path
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError(f"Symlink is not an evidence artifact: {name}")
    return target


class EvidenceRun:
    """Write artifacts once into a new directory, then seal their hashes.

    A crash can leave an incomplete directory. Integrity checking requires the
    seal itself to be trusted; it does not authenticate source or prove acceptance.
    """

    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self._sealed = False

    def _open(self, name, mode):
        if self._sealed or name == "SHA256SUMS":
            raise ValueError("Cannot write a sealed run or the reserved seal name")
        path = _artifact(self.root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open(mode)

    def json(self, name, value):
        # Serialize before opening: invalid JSON leaves no misleading empty file.
        text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with self._open(name, "x") as stream:
            stream.write(text)

    def arrays(self, name, **arrays):
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        if not arrays or any(value.dtype.kind not in "buif" for value in arrays.values()):
            raise TypeError("Evidence arrays must be numeric; pickle/object arrays are forbidden")
        with self._open(name, "xb") as stream:
            np.savez_compressed(stream, **arrays)

    def frame_diagnostic(self, prefix, ours, source, *, metadata):
        """Save the worst raw RGB pair, signed difference, batch index, and error summary."""
        from .metrics import normalized_mse, summarize_errors

        a, b = np.asarray(ours), np.asarray(source)
        errors = normalized_mse(a, b)
        summary = summarize_errors(errors)
        index = np.unravel_index(summary["worst_index"], errors.shape)
        self.arrays(
            prefix + ".npz",
            ours=a[index],
            source=b[index],
            difference=a[index].astype(np.int16) - b[index].astype(np.int16),
        )
        self.json(prefix + ".json", dict(metadata=metadata, frame_index=[int(i) for i in index], metrics=summary))

    def seal(self):
        if self._sealed:
            raise ValueError("Run is already sealed")
        paths = sorted(p for p in self.root.rglob("*") if p.is_file())
        if not paths:
            raise ValueError("Cannot seal an empty run")
        lines = []
        for path in paths:
            name = path.relative_to(self.root).as_posix()
            lines.append(f"{sha256(_artifact(self.root, name))}  {name}\n")
        with (self.root / "SHA256SUMS").open("x") as stream:
            stream.writelines(lines)
        self._sealed = True
        return verify_seal(self.root)


def verify_seal(directory) -> dict[str, str]:
    """Verify complete file coverage and hashes, rejecting duplicates and unsafe paths."""
    root = Path(directory).resolve()
    entries = {}
    for line in _artifact(root, "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in entries or name == "SHA256SUMS" or len(digest) != 64:
            raise ValueError(f"Invalid or duplicate seal entry: {name}")
        if sha256(_artifact(root, name)) != digest:
            raise ValueError(f"Artifact hash mismatch: {name}")
        entries[name] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if not entries or actual != set(entries) | {"SHA256SUMS"}:
        raise ValueError("Seal does not cover the complete nonempty run")
    return entries
