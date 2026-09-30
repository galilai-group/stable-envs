"""Supplied-state, batch, nonmutation and validation checks for explicit GL paths."""

import os

import pytest
import torch

from scripts.benchmarks.renderer_table.common import public_render, state_hash


@pytest.mark.parametrize("name", ["OGBCube", "OGBScene", "DMCWalkerWalk", "FetchPush", "FetchReach"])
def test_opengl_supplied_state_and_out(name):
    import stable_envs as se

    selection = os.environ.get("SE_FIDELITY_TEST_FAMILY", "")
    if selection and not name.startswith(selection):
        pytest.skip("Different pinned family profile")
    device = os.environ.get("SE_FIDELITY_TEST_DEVICE", "cpu")
    env = se.make(name, num_envs=1, device=device, seed=17, backend="opengl")
    try:
        state = env.get_state()
        before = state_hash(state)
        image = public_render(env, state, backend="opengl")
        assert image.dtype == torch.uint8 and image.device.type == device and image.shape[0] == 1
        out = torch.empty_like(image)
        supplied = env.to_render_state(state) if hasattr(env, "to_render_state") else state
        other = env.render(supplied, backend="opengl", out=out)
        assert other.data_ptr() == out.data_ptr()
        # A one-byte floating rasterization difference can occur in source GL.
        assert (image.to(torch.int16) - other.to(torch.int16)).abs().max() <= 1
        assert state_hash(state) == before and state_hash(env.get_state()) == before
        if name.startswith("OGB"):
            rs = env.to_render_state(state).expand(3).clone()
            rs["appearance", "floor.color"][1] = 0.3
            rs["appearance", "cube.color"][2] = torch.tensor([[0.1, 0.7, 0.3]], device=device)
            digest = state_hash(rs)
            batch = env.render(rs, backend="opengl", chunk_size=2)
            for i in range(3):
                single = env.render(rs[i : i + 1], backend="opengl")
                assert (batch[i].to(torch.int16) - single[0].to(torch.int16)).abs().max() <= 1
            assert state_hash(rs) == digest
            for key, value in [(("renderer", "signature"), 0), (("pose", "qpos"), float("nan"))]:
                bad = rs.clone()
                bad[key][-1] = value
                with pytest.raises(ValueError):
                    env.render(bad, backend="opengl")
            assert state_hash(env.get_state()) == before
    finally:
        env.close()
    with pytest.raises(RuntimeError):
        env.render(state, backend="opengl")


def test_cube_fidelity_batch_and_nonmutation():
    if os.environ.get("SE_FIDELITY_TEST_DEVICE") != "cuda" or os.environ.get("SE_FIDELITY_TEST_FAMILY") != "OGB":
        pytest.skip("Requires Cube CUDA profile")
    import stable_envs as se

    env = se.make("OGBCube", device="cuda", seed=17)
    try:
        state = env.get_state()
        before = state_hash(state)
        rows = env.to_render_state(state).expand(3).clone()
        rows["appearance", "floor.color"][1] = 0.3
        rows["appearance", "cube.color"][2] = torch.tensor([[0.1, 0.7, 0.3]], device="cuda")
        saved = state_hash(rows)
        images = env.render(rows, backend="mjwarp_fidelity", chunk_size=2)
        for i in range(3):
            assert torch.equal(images[i], env.render(rows[i : i + 1], backend="mjwarp_fidelity")[0])
        expanded = torch.cat([rows[i % 3 : i % 3 + 1] for i in range(257)])
        batch = env.render(expanded, backend="mjwarp_fidelity")
        assert torch.equal(batch, images[torch.arange(257, device="cuda") % 3])
        assert state_hash(rows) == saved and state_hash(env.get_state()) == before
    finally:
        env.close()
