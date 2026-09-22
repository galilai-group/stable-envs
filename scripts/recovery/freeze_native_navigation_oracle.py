"""Freeze additive real-reference-only oracles for three native navigation defaults."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import stable_worldmodel

from tests.helpers.reference.native_navigation_semantics import NativeNavigationReference

PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
DOMAINS = ("two_room", "piecewise", "simple_point_maze")
HISTORICAL_IDS = {
    "two_room": "swm/TwoRoom-v1",
    "piecewise": "swm/Piecewise-v0",
    "simple_point_maze": "swm/SimplePointMaze-v0",
}
root = Path("tests/fixtures/semantics/native_navigation")
root.mkdir(parents=True, exist_ok=True)
version = importlib.metadata.version("gymnasium").replace(".", "")
paths = {domain: root / f"{domain}_swm_176d545_gymnasium{version}.npz" for domain in DOMAINS}
existing = [str(path) for path in paths.values() if path.exists()]
if existing or (root / "manifest.json").exists():
    raise FileExistsError(f"Refusing to overwrite frozen semantic oracle: {existing or [root / 'manifest.json']}")
source = Path(stable_worldmodel.__file__).resolve().parents[1]
pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
assert pin == PIN
assert not subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"], text=True)

manifest = {
    "reference_commit": pin,
    "seed": 17,
    "batch": 3,
    "steps": 32,
    "generator": "scripts/recovery/freeze_native_navigation_oracle.py",
    "versions": {
        name: importlib.metadata.version(name)
        for name in ("numpy", "gymnasium", "torch", "tensordict", "stable-worldmodel", "matplotlib")
    },
    "source": "Real pinned SWM environments only; independent of stable-envs production equations",
    "scope": "Empty-kwargs TwoRoom-v1, Piecewise-v0 and SimplePointMaze-v0 defaults only",
    "domains": {},
}
for domain in DOMAINS:
    ref = NativeNavigationReference(domain, 3)
    try:
        initial = ref.reset(seed=17)
        bound = 0.4 if domain == "simple_point_maze" else 1.5
        actions = np.random.default_rng(217).uniform(-bound, bound, size=(32, 3, 2)).astype(np.float32)
        outcomes = [ref.step(action) for action in actions]
        values = {"actions": actions}
        for key in initial.keys(True, True):
            name = key if isinstance(key, str) else "/".join(key)
            values[f"state/{name}"] = np.stack(
                [initial[key].numpy(), *[outcome["state"][key].numpy() for outcome in outcomes]]
            )
        output_keys = set(outcomes[0].keys(True, True)) - {
            ("state", *key) if isinstance(key, tuple) else ("state", key)
            for key in initial.keys(True, True)
        }
        for key in sorted(output_keys, key=str):
            if key == "state" or (isinstance(key, tuple) and key[0] == "state"):
                continue
            name = key if isinstance(key, str) else "/".join(key)
            values[f"output/{name}"] = np.stack([outcome[key].numpy() for outcome in outcomes])
        path = paths[domain]
        np.savez_compressed(path, **values)
        manifest["domains"][domain] = {
            "historical_id": HISTORICAL_IDS[domain],
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in values.items()},
        }
    finally:
        ref.close()
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(root / "manifest.json")
