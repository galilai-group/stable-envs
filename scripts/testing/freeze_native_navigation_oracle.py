"""Freeze reference-only oracles for three native navigation defaults."""

import importlib.metadata
from pathlib import Path

import numpy as np

from scripts.testing._oracle import (
    add_outputs,
    pinned_commit,
    refuse_overwrite,
    rollout,
    save,
    versions,
    write_manifest,
)
from tests.helpers.reference.native_navigation_semantics import NativeNavigationReference

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
refuse_overwrite(paths.values(), root / "manifest.json")
pin = pinned_commit()

manifest = {
    "reference_commit": pin,
    "seed": 17,
    "batch": 3,
    "steps": 32,
    "generator": "scripts/testing/freeze_native_navigation_oracle.py",
    "versions": versions(("numpy", "gymnasium", "torch", "tensordict", "stable-worldmodel", "matplotlib")),
    "source": "Real pinned SWM environments only; independent of stable-envs production equations",
    "scope": "Empty-kwargs TwoRoom-v1, Piecewise-v0 and SimplePointMaze-v0 defaults only",
    "domains": {},
}
for domain in DOMAINS:
    ref = NativeNavigationReference(domain, 3)
    try:
        bound = 0.4 if domain == "simple_point_maze" else 1.5
        _, outcomes, values = rollout(
            ref,
            lambda ref: np.random.default_rng(217).uniform(-bound, bound, size=(32, 3, 2)).astype(np.float32),
        )
        add_outputs(values, outcomes, sort=True)
        manifest["domains"][domain] = {"historical_id": HISTORICAL_IDS[domain], **save(paths[domain], values)}
    finally:
        ref.close()
write_manifest(root / "manifest.json", manifest)
