"""Source parity for every persistent Rocket leaf, including scalar precision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/recovery"))
from rocket_feasibility import RocketLandingEnv as Oracle
from rocket_feasibility import capture, differences

import stable_envs
from stable_envs._adapters.rocket_spaces import nodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    records = []
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.glob("stable_envs/_adapters/rocket*.py"))
        + [ROOT / "stable_envs/envs/rocket_semantics.py"]
    }
    config = dict(render_mode="rgb_array", render_resolution=(64, 64))
    for seed in [0, 17, 42]:
        template = Oracle(**config)
        try:
            cases = []
            for path, node in nodes(template.variation_space):
                if hasattr(node, "spaces"):
                    continue
                name = path[1:].replace("/", ".")
                values = [node.low.copy(), node.high.copy(), copy.deepcopy(node.init_value)]
                if node.shape == ():
                    values += [
                        0.8123456789012345,
                        np.float32(0.8123456789012345),
                        np.float64(0.8123456789012345),
                    ]
                for value in values:
                    cases.append((name, value))
        finally:
            template.close()
        for name, value in cases:
            oracle = Oracle(**config)
            e = stable_envs.make("PyFlytRocketLanding", simulator_config=config, device="cpu")
            try:
                node = oracle.variation_space
                for part in name.split("."):
                    node = node[part]
                node.set_init_value(copy.deepcopy(value))
                e.set_variation({name: copy.deepcopy(value)})
                # Source colors apply only if a rocket/pad selection triggers its
                # visual path. Select a different leaf to retain this override.
                selection = (
                    ["rocket.fin_color" if name == "rocket.body_color" else "rocket.body_color"]
                    if name.startswith(("rocket.", "pad."))
                    else []
                )
                oracle.reset(seed=seed, options={"variation": selection})
                e.reset(seed=seed, options={"variation": selection})
                for step in range(4):
                    if step:
                        action = np.array([0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1])
                        oracle.step(action.copy())
                        e.step(action[None], states=None, frames=None)
                    assert not differences(capture(oracle), capture(e._simulator.envs[0].env))
                    assert np.array_equal(oracle.render(), e._simulator.envs[0].env.render())
                snapshot = e.get_state()
                e.reset(seed=42)
                e.set_state(snapshot)
                assert all(torch.equal(snapshot[k], e.get_state()[k]) for k in snapshot.keys(True, True))
                oracle.step(np.zeros(7))
                e.step(np.zeros((1, 7)), states=None, frames=None)
                assert not differences(capture(oracle), capture(e._simulator.envs[0].env))
                records.append(
                    dict(
                        seed=seed,
                        leaf=name,
                        value=np.asarray(value).tolist(),
                        representation=type(value).__name__,
                        dtype=str(np.asarray(value).dtype),
                        selection=selection,
                        source_reset_steps_pixels_restore_suffix_exact=True,
                    )
                )
            finally:
                e.close()
                oracle.close()
                if oracle.modified_pad_urdf:
                    Path(oracle.modified_pad_urdf).unlink(missing_ok=True)
        print("PERSISTENT_VARIATION", seed, "PASS", flush=True)
    (args.output / "result.json").write_text(
        json.dumps(dict(status="PERSISTENT_VARIATION_PASS", records=records, production_hashes=hashes), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
