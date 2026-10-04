"""Create additive full-State oracles from only the three pinned SWM wrappers."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import stable_worldmodel

from tests.helpers.reference.dmc_semantics import DMCReference as DMCFirstFamilyReference

PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
DOMAINS = ("acrobot", "pendulum", "cartpole")
root = Path("tests/fixtures/semantics/dmc_first_family")
root.mkdir(parents=True, exist_ok=True)
paths = {domain: root / f"{domain}_swm_176d545_mujoco312.npz" for domain in DOMAINS}
existing = [str(path) for path in paths.values() if path.exists()]
if existing or (root / "manifest.json").exists():
    raise FileExistsError(f"Refusing to overwrite frozen semantic oracle: {existing or [root / 'manifest.json']}")
source = Path(stable_worldmodel.__file__).resolve().parents[1]
pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
assert pin == PIN
assert not subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"], text=True)
assert importlib.metadata.version("mujoco") == "3.12.0"

manifest = {
    "reference_commit": pin,
    "seed": 17,
    "batch": 3,
    "steps": 32,
    "generator": "scripts/recovery/freeze_dmc_first_family_oracle.py",
    "versions": {
        name: importlib.metadata.version(name) for name in ("numpy", "mujoco", "dm-control", "stable-worldmodel")
    },
    "source": "Real SWM wrappers only; independent of candidate adapter",
    "domains": {},
}
for domain in DOMAINS:
    ref = DMCFirstFamilyReference(domain, 3)
    try:
        initial = ref.reset(seed=17)
        actions = np.random.default_rng(217).uniform(-1, 1, (32, 3, 1))
        outcomes = [ref.step(action) for action in actions]
        values = {"actions": actions}
        for key in initial.keys(True, True):
            name = key if isinstance(key, str) else "/".join(key)
            values[f"state/{name}"] = np.stack(
                [initial[key].numpy(), *[outcome["state"][key].numpy() for outcome in outcomes]]
            )
        for key in ("reward", "terminated", "truncated", "observation", "score"):
            values[key] = np.stack([outcome[key].numpy() for outcome in outcomes])
        path = paths[domain]
        np.savez_compressed(path, **values)
        manifest["domains"][domain] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in values.items()},
        }
    finally:
        ref.close()
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(root / "manifest.json")
