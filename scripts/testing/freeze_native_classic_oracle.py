"""Freeze reference-only oracles for the CartPole and Pendulum classic-control defaults."""

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
from tests.helpers.reference.native_classic_semantics import NativeClassicReference

DOMAINS = ("cartpole", "pendulum")
root = Path("tests/fixtures/semantics/native_classic")
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
    "generator": "scripts/testing/freeze_native_classic_oracle.py",
    "versions": versions(("numpy", "gymnasium", "torch", "tensordict", "stable-worldmodel")),
    "source": "Real pinned SWM wrappers only; independent of stable-envs production adapter",
    "domains": {},
}


def actions(domain):
    rng = np.random.default_rng(217)
    return (
        rng.integers(0, 2, size=(32, 3, 1), dtype=np.int64)
        if domain == "cartpole"
        else rng.uniform(-2, 2, size=(32, 3, 1)).astype(np.float32)
    )


for domain in DOMAINS:
    ref = NativeClassicReference(domain, 3)
    try:
        _, outcomes, values = rollout(ref, lambda ref: actions(domain))
        add_outputs(values, outcomes, ("reward", "terminated", "truncated", "observation", "info_state"))
        manifest["domains"][domain] = {
            "historical_id": f"swm/{'CartPole' if domain == 'cartpole' else 'Pendulum'}Control-v1",
            **save(paths[domain], values),
        }
    finally:
        ref.close()
write_manifest(root / "manifest.json", manifest)
