"""Pin checks, rollouts and manifest entries shared by the ``freeze_*_oracle.py`` generators."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np

PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"


def refuse_overwrite(paths, manifest, what="frozen semantic oracle"):
    """Raise FileExistsError when any output in ``paths`` or ``manifest`` already exists."""
    existing = [str(path) for path in paths if path.exists()]
    if existing or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {what}: {existing or [manifest]}")


def pinned_commit(clean=True):
    """Return the installed stable-worldmodel checkout's commit, asserting it is ``PIN`` and, with ``clean``, unmodified."""
    import stable_worldmodel

    source = Path(stable_worldmodel.__file__).resolve().parents[1]
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert pin == PIN
    if clean:
        assert not subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"], text=True)
    return pin


def versions(names):
    """Return the installed version of each distribution in ``names``."""
    return {name: importlib.metadata.version(name) for name in names}


def _name(key):
    return key if isinstance(key, str) else "/".join(key)


def rollout(ref, actions):
    """Reset ``ref`` with seed 17, step it through ``actions(ref)`` and return the outcomes and stacked State.

    The returned values hold ``actions`` and each State leaf as ``state/<path>``, initial State first.
    """
    initial = ref.reset(seed=17)
    actions = actions(ref)
    outcomes = [ref.step(action) for action in actions]
    values = {"actions": actions}
    for key in initial.keys(True, True):
        values[f"state/{_name(key)}"] = np.stack(
            [initial[key].numpy(), *[outcome["state"][key].numpy() for outcome in outcomes]]
        )
    return initial, outcomes, values


def add_outputs(values, outcomes, keys=None, sort=False):
    """Stack task outputs into ``values``.

    The named ``keys`` are stored as given; without them every non-State leaf is
    stored as ``output/<path>``, in leaf order or, with ``sort``, ordered by ``str(key)``.
    """
    if keys is not None:
        for key in keys:
            values[key] = np.stack([outcome[key].numpy() for outcome in outcomes])
        return
    leaves = outcomes[0].keys(True, True)
    for key in sorted(leaves, key=str) if sort else leaves:
        if key == "state" or (isinstance(key, tuple) and key[0] == "state"):
            continue
        values[f"output/{_name(key)}"] = np.stack([outcome[key].numpy() for outcome in outcomes])


def arrays(values):
    """Return the shape and dtype of each saved array."""
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in values.items()}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, values):
    """Write ``values`` to ``path`` and return its manifest entry."""
    np.savez_compressed(path, **values)
    return {"file": path.name, "sha256": sha256(path), "arrays": arrays(values)}


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)
