"""Create the immutable reference-only Humanoid Walk oracle."""

import importlib.metadata
from pathlib import Path

import numpy as np

from scripts.testing._oracle import add_outputs, pinned_commit, rollout, save, versions, write_manifest
from tests.helpers.reference.dmc_semantics import DMCHumanoidReference

ROOT = Path("tests/fixtures/semantics/dmc_humanoid")
ROOT.mkdir(parents=True, exist_ok=True)
PATH = ROOT / "humanoid_walk_swm_176d545_mujoco317.npz"
MANIFEST = ROOT / "manifest.json"
if PATH.exists() or MANIFEST.exists():
    raise FileExistsError(f"Refusing to overwrite frozen semantic oracle under {ROOT}")

pin = pinned_commit()
assert importlib.metadata.version("mujoco") == "3.17.0"

reference = DMCHumanoidReference(3)
try:
    _, outcomes, values = rollout(
        reference, lambda ref: np.random.default_rng(217).uniform(-1, 1, (32, 3, ref.action_dim))
    )
    add_outputs(values, outcomes)
    entry = save(PATH, values)
finally:
    reference.close()

manifest = {
    "reference_commit": pin,
    "seed": 17,
    "batch": 3,
    "steps": 32,
    "generator": "scripts/testing/freeze_dmc_humanoid_oracle.py",
    "versions": versions(("numpy", "mujoco", "dm-control", "stable-worldmodel")),
    "source": "Real pinned SWM HumanoidDMControlWrapper default Walk only; independent of candidate adapter",
    "domain": entry,
}
write_manifest(MANIFEST, manifest)
