"""Fetch render(state) checks the drawn qpos and goal; set_state checks the full State; backend names are checked."""

import pytest
import torch

import stable_envs as se
from tests.helpers.render_checks import assert_render_checks_drawn_fields_only


@pytest.mark.parametrize("name", ["FetchReach", "FetchPush"])
def test_render_checks_drawn_fields_only(name):
    with se.make(name, num_envs=2, device="cpu") as env:
        assert {key for key, _ in env._drawn_fields()} == {("qpos",), ("goal",)}
        assert_render_checks_drawn_fields_only(env, env.reset(seed=5))
        assert torch.is_tensor(env.render())


@pytest.mark.parametrize("backend", ["raster", "mjwarp", "nvdiff", "Numba"])
def test_an_unknown_render_backend_is_refused(backend):
    with pytest.raises(ValueError, match="Fetch render backend"):
        se.make("FetchReach", num_envs=1, device="cpu", backend=backend)
    with se.make("FetchReach", num_envs=1, device="cpu") as env:
        state = env.reset(seed=0)
        with pytest.raises(ValueError, match="Fetch render backend"):
            env.render(state, backend=backend)
