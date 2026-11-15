"""CUDA render rungs: a small batch draws with the CPU renderer of its class and moves the frames in one copy.

The CPU and CUDA renderers of every switched class draw the same bytes, so the switch point never changes a frame.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs import registry, tuning

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

#: Every class that ships a CUDA render switch point.
NAMES = ("PushT", "PushTDiscrete", "TwoRoom", "TwoRoomTarget")
LANES = 16


def _require(name):
    try:
        missing = registry.missing_extra(name)
    except KeyError:
        missing = name
    if missing is not None:
        pytest.skip(f"{name} needs the {missing} extra")


def _actions(env, rng):
    return torch.as_tensor(env.action_spec.sample(env.num_envs, rng=rng)).to("cuda")


def _varied(env, rng):
    """Install a sampled appearance and physics variation on every lane, then reset with it."""
    env.set_variation(env.sample_variation(env.num_envs, rng=rng))
    return env.reset(seed=1)


@pytest.mark.parametrize("name", NAMES)
def test_switched_classes_declare_cuda_rungs(name):
    _require(name)
    with se.make(name, num_envs=1, device="cuda") as env:
        assert env.render_rungs is not None and env.render_rungs["cuda"].min_envs is not None


@pytest.mark.parametrize("varied", [False, True], ids=["default", "varied"])
@pytest.mark.parametrize("name", NAMES)
def test_a_small_cuda_batch_draws_the_same_bytes_on_the_cpu_renderer(name, varied):
    _require(name)
    rng = np.random.default_rng(0)
    with se.make(name, num_envs=LANES, device="cuda") as env:
        state = _varied(env, rng) if varied else env.reset(seed=0)
        for _ in range(3):
            state = env.step(_actions(env, rng), frames=None)["state"]
        with tuning.override(large_renderer_min_rows=1):
            gpu = env.render(state)
            gpu_part = env.render(state[:5])
        with tuning.override(large_renderer_min_rows=LANES + 1):
            cpu = env.render(state)
            cpu_part = env.render(state[:5])
            out = torch.zeros_like(gpu)
            env.render(state, out=out)
            stepped = env.step(_actions(env, rng))["frames"]
        # TwoRoom draws in a CPU copy of the env, which receives the installed variation with the rows.
        assert ("_host_twin" in env.__dict__) == name.startswith("TwoRoom")
        assert cpu.device == gpu.device and cpu.shape == gpu.shape and cpu.dtype == gpu.dtype
        assert torch.equal(cpu, gpu)
        assert torch.equal(cpu_part, gpu_part) and torch.equal(cpu_part, gpu[:5])
        assert torch.equal(out, cpu)
        assert stepped.device.type == "cuda" and stepped.shape == (LANES, 1, *gpu.shape[1:])
        with tuning.override(large_renderer_min_rows=1):
            assert torch.equal(stepped[:, 0], env.render(env.get_state()))


@pytest.mark.parametrize("name", NAMES)
def test_tune_times_both_renderers(name, tmp_path, monkeypatch):
    _require(name)
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
