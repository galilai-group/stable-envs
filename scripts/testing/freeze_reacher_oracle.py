"""Freeze a full-State Reacher oracle from the pinned stable-worldmodel wrapper only."""

import importlib.metadata
import json
from pathlib import Path

import numpy as np

from scripts.testing._oracle import add_outputs, arrays, pinned_commit, rollout, sha256, versions
from tests.helpers.reference.dmc_semantics import ReacherReference

root = Path("tests/fixtures/semantics/reacher")
root.mkdir(parents=True, exist_ok=True)
path = root / "swm_176d545_mujoco317.npz"
if path.exists():
    raise FileExistsError(f"Refusing to overwrite frozen semantic oracle: {path}")
pin = pinned_commit(clean=False)
assert importlib.metadata.version("mujoco") == "3.17.0"
ref = ReacherReference(3)
try:
    _, outcomes, values = rollout(ref, lambda ref: np.random.default_rng(217).uniform(-1, 1, (32, 3, 2)))
    add_outputs(values, outcomes, ("reward", "terminated", "truncated", "observation", "score"))
    np.savez_compressed(path, **values)
    manifest = {
        "reference_commit": pin,
        "file": path.name,
        "sha256": sha256(path),
        "seed": 17,
        "batch": 3,
        "steps": 32,
        "generator": "scripts/testing/freeze_reacher_oracle.py",
        "versions": versions(("numpy", "mujoco", "dm-control", "stable-worldmodel")),
        "source": "Real SWM wrapper only; independent of candidate adapter",
        "arrays": arrays(values),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(path, manifest["sha256"])
finally:
    ref.close()
