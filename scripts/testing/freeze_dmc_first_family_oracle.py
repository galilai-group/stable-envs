"""Freeze full-State oracles for the first DMC family from the pinned stable-worldmodel wrappers only."""

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
from tests.helpers.reference.dmc_semantics import DMCReference as DMCFirstFamilyReference

DOMAINS = ("acrobot", "pendulum", "cartpole")
root = Path("tests/fixtures/semantics/dmc_first_family")
root.mkdir(parents=True, exist_ok=True)
paths = {domain: root / f"{domain}_swm_176d545_mujoco317.npz" for domain in DOMAINS}
refuse_overwrite(paths.values(), root / "manifest.json")
pin = pinned_commit()
assert importlib.metadata.version("mujoco") == "3.17.0"

manifest = {
    "reference_commit": pin,
    "seed": 17,
    "batch": 3,
    "steps": 32,
    "generator": "scripts/testing/freeze_dmc_first_family_oracle.py",
    "versions": versions(("numpy", "mujoco", "dm-control", "stable-worldmodel")),
    "source": "Real SWM wrappers only; independent of candidate adapter",
    "domains": {},
}
for domain in DOMAINS:
    ref = DMCFirstFamilyReference(domain, 3)
    try:
        _, outcomes, values = rollout(ref, lambda ref: np.random.default_rng(217).uniform(-1, 1, (32, 3, 1)))
        add_outputs(values, outcomes, ("reward", "terminated", "truncated", "observation", "score"))
        manifest["domains"][domain] = save(paths[domain], values)
    finally:
        ref.close()
write_manifest(root / "manifest.json", manifest)
