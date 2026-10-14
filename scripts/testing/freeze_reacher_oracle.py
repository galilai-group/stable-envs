"""Freeze a full-State Reacher oracle from the pinned SWM wrapper only."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import stable_worldmodel

from tests.helpers.reference.dmc_semantics import ReacherReference

root = Path("tests/fixtures/semantics/reacher")
root.mkdir(parents=True, exist_ok=True)
path = root / "swm_176d545_mujoco312.npz"
if path.exists():
    raise FileExistsError(f"Refusing to overwrite frozen semantic oracle: {path}")
source = Path(stable_worldmodel.__file__).resolve().parents[1]
pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
assert pin == "176d54532b3c0a251a4f509b74f2fdacc143a63d"
assert importlib.metadata.version("mujoco") == "3.12.0"
ref = ReacherReference(3)
try:
    initial = ref.reset(seed=17)
    actions = np.random.default_rng(217).uniform(-1, 1, (32, 3, 2))
    outcomes = [ref.step(a) for a in actions]
    values = {"actions": actions}
    for key in initial.keys(True, True):
        name = key if isinstance(key, str) else "/".join(key)
        values[f"state/{name}"] = np.stack([initial[key].numpy(), *[x["state"][key].numpy() for x in outcomes]])
    for key in ("reward", "terminated", "truncated", "observation", "score"):
        values[key] = np.stack([x[key].numpy() for x in outcomes])
    np.savez_compressed(path, **values)
    manifest = {
        "reference_commit": pin,
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": 17,
        "batch": 3,
        "steps": 32,
        "generator": "scripts/testing/freeze_reacher_oracle.py",
        "versions": {k: importlib.metadata.version(k) for k in ("numpy", "mujoco", "dm-control", "stable-worldmodel")},
        "source": "Real SWM wrapper only; independent of candidate adapter",
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in values.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(path, manifest["sha256"])
finally:
    ref.close()
