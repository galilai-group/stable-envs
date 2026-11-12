"""Freeze a full-State OGBMaze oracle from the pinned stable-worldmodel wrapper only."""

import importlib.metadata
import json
from pathlib import Path

import numpy as np

from stable_envs._core.mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

from scripts.testing._oracle import add_outputs, arrays, pinned_commit, rollout, sha256, versions
from tests.helpers.reference.ogbench_maze_semantics import OGBMazeReference

root = Path("tests/fixtures/semantics/ogbench_maze")
root.mkdir(parents=True, exist_ok=True)
path = root / "swm_176d545_ogbench121_mujoco317.npz"
if path.exists():
    raise FileExistsError(f"Refusing to overwrite frozen semantic oracle: {path}")
pin = pinned_commit(clean=False)
assert importlib.metadata.version("ogbench") == "1.2.1"
assert importlib.metadata.version("mujoco") == "3.17.0"
ref = OGBMazeReference(3)
try:
    _, outcomes, values = rollout(
        ref, lambda ref: np.random.default_rng(217).uniform(-1, 1, (32, 3, 2)).astype(np.float32)
    )
    add_outputs(values, outcomes)
    np.savez_compressed(path, **values)
    manifest = {
        "reference_commit": pin,
        "historical_id": "swm/OGBMaze-v0",
        "historical_defaults": {
            "loco_env_type": "point",
            "maze_env_type": "maze",
            "maze_type": "large",
            "ob_type": "states",
            "width": 224,
            "height": 224,
        },
        "file": path.name,
        "sha256": sha256(path),
        "seed": 17,
        "batch": 3,
        "steps": 32,
        "generator": "scripts/testing/freeze_ogbench_maze_oracle.py",
        "versions": versions(("numpy", "gymnasium", "ogbench", "mujoco", "stable-worldmodel")),
        "source": "Real pinned SWM wrapper only; independent of candidate adapter",
        "arrays": arrays(values),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(path, manifest["sha256"])
finally:
    ref.close()
