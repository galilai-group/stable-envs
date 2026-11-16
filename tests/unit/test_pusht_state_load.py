"""A folded State saved with torch.save resolves its schema in a new process: it renders and restores there."""

import subprocess
import sys
import textwrap

import pytest
import torch

import stable_envs as se

CHILD = textwrap.dedent(
    """
    import sys
    import torch
    import stable_envs as se

    name, path = sys.argv[1], sys.argv[2]
    saved = torch.load(path, weights_only=False)
    state = saved["state"]
    with se.make(name, device="cpu", **({"size": 32} if name.startswith("PushT") else {})) as env:
        assert se.schema_of(state) is not None, "the loaded State resolves no schema"
        assert torch.equal(env.render(state), saved["frame"])
        env.set_state(state)
        assert torch.equal(env.render(), saved["frame"])
    """
)


@pytest.mark.parametrize(
    "name,variation",
    [("PushT", None), ("PushT", {"agent.shape": 3, "block.shape": 7}), ("CartPole", None)],
)
def test_a_saved_state_renders_and_restores_in_a_new_process(name, variation, tmp_path):
    if name not in se.list_envs():
        pytest.skip(f"{name} is not installed")
    with se.make(name, device="cpu", **({"size": 32} if name.startswith("PushT") else {})) as env:
        if variation is not None:
            env.set_variation(variation)
        env.reset(seed=3)
        env.step(env.action_spec.sample(1).to(env.device), frames=None)
        state = env.get_state()
        if se.schema_of(state) is None:
            pytest.skip(f"{name} has no folded State")
        torch.save({"state": state, "frame": env.render(state)}, tmp_path / "state.pt")
    child = subprocess.run(
        [sys.executable, "-c", CHILD, name, str(tmp_path / "state.pt")], capture_output=True, text=True, timeout=300
    )
    assert child.returncode == 0, child.stderr[-2000:]
