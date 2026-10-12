import hashlib
import json
from pathlib import Path

import numpy as np

import stable_envs as se

ROOT = Path("tests/fixtures/semantics/native_navigation")
NAMES = {
    "two_room": "TwoRoom",
    "piecewise": "Piecewise",
    "simple_point_maze": "SimplePointMaze",
}


def _array(value):
    return value.detach().cpu().numpy()


def test_frozen_native_navigation_oracles_replay_exactly():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for domain, metadata in manifest["domains"].items():
        path = ROOT / metadata["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
        fixture = np.load(path)
        with se.make(NAMES[domain], num_envs=3, device="cpu") as env:
            state = env.reset(seed=manifest["seed"])
            for key in state.keys(True, True):
                name = key if isinstance(key, str) else "/".join(key)
                assert np.array_equal(_array(state[key]), fixture[f"state/{name}"][0])
            for step, action in enumerate(fixture["actions"]):
                output = env.step(action)
                for key in output["state"].keys(True, True):
                    name = key if isinstance(key, str) else "/".join(key)
                    assert np.array_equal(_array(output["state"][key]), fixture[f"state/{name}"][step + 1])
                assert _array(output["valid"]).all()
                for name in (k.removeprefix("output/") for k in fixture.files if k.startswith("output/")):
                    assert np.array_equal(_array(output[tuple(name.split("/"))]), fixture[f"output/{name}"][step])
