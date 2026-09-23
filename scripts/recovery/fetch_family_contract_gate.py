"""Fetch public readiness and persistent/reset visual Variation checks."""

import json
from pathlib import Path

import numpy as np
import torch

import stable_envs.envs.fetch as fetch
from scripts.recovery.fetch_public_gate import equal_state


def main():
    results = {}
    for task in ("PickAndPlace", "Reach", "Push", "Slide"):
        for suffix in ("", "Dense", "Dict"):
            name = f"Fetch{task}{suffix}"
            with getattr(fetch, name)(3, seed=17) as env:
                state = env.get_state()
                assert env.has_dynamics and env.execution_mode == "DELEGATED_CPU"
                assert env.action_spec.shape == (4,)
                for key in state.keys(True, True):
                    assert env.state_spec[key].dtype == state[key].dtype
                key = "table.color" if task == "Reach" else "object.color"
                off = next(o for k, o, _ in env.variation_space.flat_layout() if k == key)
                persistent = [0.2, 0.8, 0.3]
                override = [0.8, 0.3, 0.2]
                env.set_variation({key: persistent})
                equal_state(env.get_state(), state)
                for seed in (17, 42):
                    env.reset(seed=seed)
                    assert torch.equal(env.get_style()[:, off : off + 3], torch.tensor([persistent] * 3))
                env.reset(seed=17, options={"variation_values": {key: np.array(override)}})
                env.reset(seed=42)
                assert torch.equal(env.get_style()[:, off : off + 3], torch.tensor([override] * 3))
                env.set_variation({key: persistent})
                env.reset(seed=42)
                assert torch.equal(env.get_style()[:, off : off + 3], torch.tensor([persistent] * 3))
                results[name] = dict(
                    public_readiness=True,
                    typed_state=True,
                    persistent_visual_across_resets=True,
                    reset_visual_remains_until_replaced=True,
                    persistent_visual_can_be_restored=True,
                    visual_set_nonmutation=True,
                    observation_integration="DEFERRED",
                )
    Path("docs/research/semantic_recovery_evidence/fetch_family/public-contract.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print("FETCH_FAMILY_PUBLIC_CONTRACT_PASS")


if __name__ == "__main__":
    main()
