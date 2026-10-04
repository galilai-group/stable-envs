"""Create the immutable reference-only Humanoid Walk oracle."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import stable_worldmodel

from tests.helpers.reference.dmc_semantics import DMCHumanoidReference

PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
ROOT = Path("tests/fixtures/semantics/dmc_humanoid")
ROOT.mkdir(parents=True, exist_ok=True)
PATH = ROOT / "humanoid_walk_swm_176d545_mujoco312.npz"
MANIFEST = ROOT / "manifest.json"
if PATH.exists() or MANIFEST.exists():
    raise FileExistsError(f"Refusing to overwrite frozen semantic oracle under {ROOT}")

source = Path(stable_worldmodel.__file__).resolve().parents[1]
pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
assert pin == PIN
assert not subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"], text=True)
assert importlib.metadata.version("mujoco") == "3.12.0"

reference = DMCHumanoidReference(3)
try:
    initial = reference.reset(seed=17)
    actions = np.random.default_rng(217).uniform(-1, 1, (32, 3, reference.action_dim))
    outcomes = [reference.step(action) for action in actions]
    values = {"actions": actions}
    for key in initial.keys(True, True):
        name = key if isinstance(key, str) else "/".join(key)
        values[f"state/{name}"] = np.stack(
            [initial[key].numpy(), *[outcome["state"][key].numpy() for outcome in outcomes]]
        )
    for key in outcomes[0].keys(True, True):
        if key == "state" or (isinstance(key, tuple) and key[0] == "state"):
            continue
        name = key if isinstance(key, str) else "/".join(key)
        values[f"output/{name}"] = np.stack([outcome[key].numpy() for outcome in outcomes])
    np.savez_compressed(PATH, **values)
finally:
    reference.close()

manifest = {
    "reference_commit": pin,
    "seed": 17,
    "batch": 3,
    "steps": 32,
    "generator": "scripts/recovery/freeze_dmc_humanoid_oracle.py",
    "versions": {
        name: importlib.metadata.version(name)
        for name in ("numpy", "mujoco", "dm-control", "stable-worldmodel")
    },
    "source": "Real pinned SWM HumanoidDMControlWrapper default Walk only; independent of candidate adapter",
    "domain": {
        "file": PATH.name,
        "sha256": hashlib.sha256(PATH.read_bytes()).hexdigest(),
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in values.items()
        },
    },
}
MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
print(MANIFEST)
