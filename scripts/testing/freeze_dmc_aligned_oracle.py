"""Create immutable reference-only oracles for DMC Ball in Cup, Finger, Manipulator and Quadruped."""

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
from tests.helpers.reference.dmc_semantics import DMCReference as DMCAlignedReference

DOMAINS = ("ball_in_cup", "finger", "manipulator", "quadruped")
ROOT = Path("tests/fixtures/semantics/dmc_aligned")
ROOT.mkdir(parents=True, exist_ok=True)
PATHS = {domain: ROOT / f"{domain}_swm_176d545_mujoco317.npz" for domain in DOMAINS}
refuse_overwrite(PATHS.values(), ROOT / "manifest.json", "frozen oracle")
pin = pinned_commit()
assert importlib.metadata.version("mujoco") == "3.17.0"
manifest = {
    "reference_commit": pin,
    "seed": 17,
    "batch": 3,
    "steps": 32,
    "generator": "scripts/testing/freeze_dmc_aligned_oracle.py",
    "versions": versions(("numpy", "mujoco", "dm-control", "stable-worldmodel")),
    "source": "Real pinned SWM wrappers only; independent of candidate adapter",
    "domains": {},
}
for domain in DOMAINS:
    ref = DMCAlignedReference(domain, 3)
    try:
        _, outcomes, values = rollout(
            ref, lambda ref: np.random.default_rng(217).uniform(-1, 1, (32, 3, ref.action_dim))
        )
        add_outputs(values, outcomes)
        manifest["domains"][domain] = save(PATHS[domain], values)
    finally:
        ref.close()
write_manifest(ROOT / "manifest.json", manifest)
