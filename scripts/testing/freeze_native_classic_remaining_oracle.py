"""Freeze additive real-reference-only oracles for the remaining classic defaults."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import stable_worldmodel

from tests.helpers.reference.native_classic_semantics import NativeClassicReference

PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
DOMAINS = ("acrobot", "mountain_car", "mountain_car_continuous")
HISTORICAL_IDS = {
    "acrobot": "swm/AcrobotControl-v1",
    "mountain_car": "swm/MountainCarControl-v0",
    "mountain_car_continuous": "swm/MountainCarContinuousControl-v0",
}
root = Path("tests/fixtures/semantics/native_classic_remaining")
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
    "generator": "scripts/testing/freeze_native_classic_remaining_oracle.py",
    "versions": {
        name: importlib.metadata.version(name)
        for name in ("numpy", "gymnasium", "torch", "tensordict", "stable-worldmodel")
    },
    "source": "Real pinned SWM wrappers only; independent of stable-envs production adapter",
    "domains": {},
}
for domain in DOMAINS:
    ref = NativeClassicReference(domain, 3)
    try:
        initial = ref.reset(seed=17)
        rng = np.random.default_rng(217)
        actions = (
            rng.integers(0, 3, size=(32, 3, 1), dtype=np.int64)
            if domain in {"acrobot", "mountain_car"}
            else rng.uniform(-1, 1, size=(32, 3, 1)).astype(np.float32)
        )
        outcomes = [ref.step(action) for action in actions]
        values = {"actions": actions}
        for key in initial.keys(True, True):
            name = key if isinstance(key, str) else "/".join(key)
            values[f"state/{name}"] = np.stack(
                [initial[key].numpy(), *[outcome["state"][key].numpy() for outcome in outcomes]]
            )
        for key in ("reward", "terminated", "truncated", "observation", "info_state"):
            values[key] = np.stack([outcome[key].numpy() for outcome in outcomes])
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
