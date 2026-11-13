"""MiniWorldMovingBlockWorld on CUDA: the Triton step and the Torch reset give the Numba kernel's episodes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stable_envs.envs.miniworld.simulation.adapter import configure_pyglet

configure_pyglet()  # before miniworld imports pyglet.gl
pytest.importorskip("miniworld.envs.moving_blockworld")
pytest.importorskip("numba")

import stable_envs as snv  # noqa: E402

pytestmark = [
    pytest.mark.family("miniworld"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

#: Configurations that reach every branch: grid and continuous motion, block-block and agent-block collisions,
#: torus wrapping, balls and random textures.
CASES = {
    "flowm": ("flowm", {}),
    "textured": ("flowm-textured", {}),
    "upstream": ("upstream", {}),
    "colliding": ("flowm", {"box_allow_overlap": False, "agent_box_allow_overlap": False, "room_size": 10}),
    "torus": ("upstream", {"block_torus_wrap": True, "box_speed_scale": 3.0}),
    "wrapped": ("flowm", {"wrap_heading": True, "max_episode_steps": 30}),
}


def _close(got, want, what):
    for key in want.keys(True, True):
        atol = 1e-9 if want[key].is_floating_point() else 0
        torch.testing.assert_close(
            got[key].cpu(), want[key], rtol=0, atol=atol, msg=lambda m, k=key: f"{what} {k}: {m}"
        )


@pytest.mark.parametrize("case", sorted(CASES))
def test_triton_steps_and_torch_resets_give_the_numba_episodes(case):
    """Resets, and each step from the same State, agree with Numba: grid motion exactly, continuous to 1e-9.

    Each step starts from the Numba State: CUDA's cosine and sine round differently from libm's, and a
    last-bit difference can flip a later collision test.
    """
    preset, options = CASES[case]
    batch = 37  # not a multiple of the kernel block
    rng = np.random.default_rng(3)
    # Mostly walk, turn and pick up, so that blocks are carried.
    actions = rng.choice([0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8], size=(80, batch))
    with (
        snv.MiniWorldMovingBlockWorld(num_envs=batch, size=32, device="cpu", preset=preset, **options) as cpu,
        snv.MiniWorldMovingBlockWorld(num_envs=batch, size=32, device="cuda", preset=preset, **options) as cuda,
    ):
        assert (cuda.simulation_backend, cuda.execution_mode) == ("triton", "NATIVE_TRITON")
        _close(cuda.reset(seed=7), cpu.reset(seed=7), "reset")
        _close(cuda.reset(), cpu.reset(), "next reset")
        carried = 0
        for t, a in enumerate(actions):
            cuda.set_state(cpu.get_state().to("cuda"))
            if t >= 70:  # sequences: environments that finish step on under an active mask
                a = np.stack([a, a], 1)[:, :, None]
            else:
                a = a[:, None]
            want = cpu.step(a, frames=None)
            got = cuda.step(torch.as_tensor(a, device="cuda"), frames=None)
            assert got["state"]["agent", "pos"].device.type == "cuda"
            _close(got["state"], want["state"], f"step {t}")
            assert torch.equal(got["truncated"].cpu(), want["truncated"]) and torch.equal(
                got["valid"].cpu(), want["valid"]
            )
            carried += int((want["state"]["agent", "carrying"] >= 0).sum())
        if case in ("upstream", "colliding"):
            assert carried > 0


@pytest.mark.parametrize("batch,expected", [(8, "numba"), (64, "triton")])
def test_auto_routes_small_cuda_batches_to_numba(batch, expected):
    with snv.MiniWorldMovingBlockWorld(
        num_envs=batch, size=32, device="cuda", simulation_backend="auto", triton_min_envs=64
    ) as env:
        assert env.simulation_backend == expected
        out = env.step(torch.full((batch, 1), 2, device="cuda"), frames=None)
        assert out["state"]["agent", "pos"].device.type == "cuda" and out["truncated"].device.type == "cuda"
