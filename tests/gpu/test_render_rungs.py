"""CUDA render rungs: a small batch draws with the CPU renderer of its class and moves the frames in one copy.

The CPU and GPU renderers of each class here draw the same frames, to the bound in ``CASES``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs import registry, tuning

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

#: ``{class: (family, largest per-frame MSE between the CPU and the GPU renderer)}``.
CASES = {"TwoRoom": ("navigation", 0.01), "PushT": ("pusht", 0.0), "DMCReacher": ("dmc", 0.5)}
LANES = 4


def _actions(env):
    return torch.as_tensor(env.action_spec.sample(env.num_envs, rng=np.random.default_rng(0))).to("cuda")


def _mse(a, b):
    a, b = a.cpu().numpy().astype(np.float64), b.cpu().numpy().astype(np.float64)
    return ((a - b) ** 2).reshape(len(a), -1).mean(1)


@pytest.mark.parametrize("name", CASES)
def test_a_small_cuda_batch_draws_on_the_cpu_renderer(name):
    family, bound = CASES[name]
    missing = registry.missing_extra(name)
    if missing is not None:
        pytest.skip(f"{name} needs the {missing} extra")
    with se.make(name, num_envs=LANES, device="cuda") as env:
        state = env.reset(seed=0)
        with tuning.override(large_renderer_min_rows=1):
            gpu = env.render(state)
        with tuning.override(large_renderer_min_rows=LANES + 1):
            cpu = env.render(state)
            out = torch.zeros_like(gpu)
            env.render(state, out=out)
            stepped = env.step(_actions(env))["frames"]
        assert cpu.device == gpu.device and cpu.shape == gpu.shape and cpu.dtype == gpu.dtype
        assert _mse(cpu, gpu).max() <= bound
        assert torch.equal(out, cpu)
        assert stepped.device.type == "cuda" and stepped.shape == (LANES, 1, *gpu.shape[1:])
        with tuning.override(large_renderer_min_rows=1):
            assert _mse(stepped[:, 0], env.render(env.get_state())).max() <= bound


@pytest.mark.parametrize("name", CASES)
def test_tune_times_both_renderers(name, tmp_path, monkeypatch):
    if registry.missing_extra(name) is not None:
        pytest.skip(f"{name} needs its extra")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("STABLE_ENVS_LARGE_RENDERER_MIN_ROWS")
    tuning._clear_caches()
    from stable_envs._tune import tune

    try:
        with se.make(name, num_envs=2, device="cuda") as env:
            tune(env, knobs=["large_renderer_min_rows"], echo=None)
        data, current = tuning.read_cache("cuda")
        assert current
        keys = [k for k, v in data["entries"].items() if "large_renderer_min_rows" in v]
        assert keys and all(k.endswith(f"/{name}|0|2-3") for k in keys)
    finally:
        tuning._clear_caches()
