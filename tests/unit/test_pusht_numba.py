"""PushT Numba and Triton physics: backend routing, agreement with the JAX solver, and Pymunk tracking."""

from contextlib import closing

import numpy as np
import pytest
import torch

pytest.importorskip("numba")

from stable_envs.envs.pusht.semantics import PushT  # noqa: E402
from stable_envs.envs.pusht.simulation.execution import TRITON_MIN_ENVS, pusht_backend  # noqa: E402
from stable_envs.envs.pusht.simulation.numba_batch import PushTNumbaAdapter  # noqa: E402
from stable_envs.envs.pusht.simulation.numba_physics import advance_batch  # noqa: E402
from stable_envs.envs.pusht.simulation.replay_batch import PushTReplayAdapter  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402


def test_backend_routing(monkeypatch):
    from stable_envs.envs.pusht import raster

    monkeypatch.setattr(raster, "HAVE_TRITON", True)
    for count in (1, 7, 8, 4096):
        assert pusht_backend(count, "cpu") == "numba"
        assert pusht_backend(count, "cuda") == "triton"
        assert pusht_backend(count, "cpu", "auto") == "numba"
        assert pusht_backend(count, "cuda", "auto") == ("triton" if count >= TRITON_MIN_ENVS else "numba")
        assert pusht_backend(count, "cuda", "auto", triton_min_envs=8) == ("triton" if count >= 8 else "numba")
        assert pusht_backend(count, "cuda", "triton") == "triton"
        for device in ("cpu", "cuda"):
            assert pusht_backend(count, device, "jax") == "jax"
            assert pusht_backend(count, device, "numba") == "numba"
    with pytest.raises(ValueError, match="requires device='cuda'"):
        pusht_backend(1, "cpu", "triton")
    with pytest.raises(ValueError, match="simulation_backend"):
        pusht_backend(1, "cpu", "pymunk")
    for bad in (0, True, 2.0):
        with pytest.raises(ValueError, match="triton_min_envs"):
            pusht_backend(1, "cuda", "auto", triton_min_envs=bad)
    # Without Triton, CUDA physics runs in JAX.
    monkeypatch.setattr(raster, "HAVE_TRITON", False)
    assert pusht_backend(8, "cuda") == "jax"
    assert pusht_backend(8, "cuda", "auto", triton_min_envs=4) == "jax"


def test_public_env_records_its_backend():
    with PushT(num_envs=2, device="cpu", size=32) as env:
        assert env.simulation_backend == "numba" and env.execution_mode == "NATIVE_NUMBA"
        assert isinstance(env._simulator, PushTNumbaAdapter)
    if torch.cuda.is_available():
        with PushT(num_envs=2, device="cuda", size=32) as env:
            assert env.simulation_backend == "triton" and env.execution_mode == "NATIVE_TRITON"
        with PushT(num_envs=2, device="cuda", size=32, simulation_backend="auto", triton_min_envs=4) as env:
            assert env.simulation_backend == "numba" and env.execution_mode == "NATIVE_NUMBA"
            out = env.step(torch.full((2, 2), 0.3, device="cuda"), frames=None)
            assert out["reward"].is_cuda and out["state"]["current", "observation"].is_cuda


@pytest.mark.parametrize("variety", [False, True])
def test_one_step_matches_the_jax_solver(variety):
    """From the same solver state, a Numba step equals a JAX CPU step to rounding, contacts and all."""
    import jax

    from stable_envs.envs.pusht.simulation.jax_batch import enable_x64
    from stable_envs.envs.pusht.simulation.jax_physics import advance

    count = 16
    options = None
    if variety:
        options = [
            dict(variation=[], variation_values={"agent.shape": i % 8, "block.shape": i % 7 + 1}) for i in range(count)
        ]
    with closing(PushTNumbaAdapter(count, "cpu", 3, config=dict(resolution=32))) as env:
        env.reset(5, options)
        rng = np.random.default_rng(0)
        geometry = tuple(value.numpy() for value in env._geometry)
        order = env._tree_order.numpy()
        names = ("body", "bias", "impulse", "identifiers", "tree", "tree_bounds")
        contacts = 0
        for _ in range(60):
            body = env._body.numpy()
            toward = body[:, 1, :2] - body[:, 0, :2]
            toward /= np.linalg.norm(toward, axis=1, keepdims=True)
            target = body[:, 0, :2] + 60 * toward + rng.normal(0, 20, (count, 2))
            before = {name: getattr(env, "_" + name).numpy().copy() for name in names}
            with enable_x64(), jax.default_device(jax.devices("cpu")[0]):
                expected = [
                    np.asarray(v) for v in advance(*before.values(), order, geometry, target, env._circle, False)
                ]
            counts = np.zeros((count, 2), np.int64)
            lines = np.zeros((count, 90, 8))
            advance_batch(
                *(getattr(env, "_" + name).numpy() for name in names),
                order,
                *geometry,
                target,
                False,
                env._circle,
                np.ones(count, np.bool_),
                counts,
                lines,
            )
            for name, want in zip(names[:3], expected, strict=False):
                np.testing.assert_allclose(getattr(env, "_" + name).numpy(), want, rtol=0, atol=1e-6, err_msg=name)
            # An identifier matters only where it keeps an impulse; elsewhere ties between equal faces may differ.
            held = expected[2] > 0
            np.testing.assert_array_equal(env._identifiers.numpy()[held], expected[3][held])
            np.testing.assert_array_equal(env._tree.numpy(), expected[4])
            np.testing.assert_allclose(env._tree_bounds.numpy(), expected[5], rtol=0, atol=1e-6)
            np.testing.assert_array_equal(counts[:, 0], expected[6])
            np.testing.assert_array_equal(counts[:, 1], expected[8])
            np.testing.assert_allclose(lines, expected[7], rtol=0, atol=1e-6)
            contacts += int(counts[:, 0].sum())
            # Continue from the JAX state so rounding does not compound.
            for name, want in zip(names, expected, strict=False):
                getattr(env, "_" + name).numpy()[:] = want
        assert contacts > 0


@pytest.mark.parametrize(
    "initial,action,steps,atol",
    [
        ([480, 100, 200, 250, 0, 0, 0], [1.0, 0.0], 200, 5e-4),
        ([100, 215, 200, 200, 0, 0, 0], [0.5, 0.0], 200, 5e-4),
        ([300, 300, 200, 250, 0, 0, 0], [1.6, -1.3], 200, 2e-2),
    ],
)
def test_relative_controls_and_contact_track_pymunk(initial, action, steps, atol):
    """The cases and bounds of the JAX solver's Pymunk tracking test."""
    options = dict(
        variation=[], state=np.array(initial, np.float64), goal_state=np.array([100, 100, 300, 250, np.pi / 4, 0, 0])
    )
    with (
        closing(PushTNumbaAdapter(1, "cpu", 0, config=dict(resolution=32))) as env,
        closing(PushTReplayAdapter(1, "cpu", 0, config=dict(resolution=32))) as ref,
    ):
        env.reset(0, options)
        ref.reset(0, options)
        for _ in range(steps):
            got = env.step(torch.tensor([action], dtype=torch.float32))
            want = ref.step(np.array([action], np.float32))
            np.testing.assert_allclose(
                got["state", "current", "observation"], want["state", "current", "observation"], atol=atol, rtol=0
            )
            assert got["reward"].item() == pytest.approx(want["reward"].item(), abs=atol)
            assert got["success"].item() == want["success"].item()


@pytest.mark.parametrize("discrete", [False, True])
def test_masked_steps_checkpoints_and_cuda_tensors(discrete):
    """A masked step leaves the inactive lanes alone, and a checkpoint restores without physics."""
    cases = [(PushTNumbaAdapter, "cpu")]
    if torch.cuda.is_available():
        from stable_envs.envs.pusht.simulation.triton_batch import PushTTritonAdapter

        cases += [(PushTNumbaAdapter, "cuda"), (PushTTritonAdapter, "cuda")]
    results = []
    for cls, device in cases:
        with closing(cls(3, device, 4, discrete=discrete, config=dict(resolution=32))) as env:
            env.reset(9)
            a = torch.tensor([[3, 4], [8, 9], [15, 0]] if discrete else [[0.2, -0.3], [-0.4, 0.5], [1.0, 1.0]])
            a = a.to(device)
            for _ in range(5):
                env.step(a)
            before = env.get_state()
            mask = torch.tensor([True, False, True], device=device)
            out = env.step(a, active=mask)
            for key, value in before["current"].items(True, True):
                assert torch.equal(value[1], out["state", "current"][key][1]), key
            checkpoint = out["state"]
            expected = env.step(a)
            env.set_state(checkpoint)
            assert_same_state(checkpoint, env.get_state())
            assert_same_state(expected, env.step(a))
            results.append(expected)
    for result in results[1:]:
        np.testing.assert_allclose(
            result["state", "current", "observation"].cpu(), results[0]["state", "current", "observation"], atol=0.05
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("variety", [False, True])
def test_triton_steps_like_numba(variety):
    """The float32 Triton kernel and the float64 Numba kernel take the same steps from the same State."""
    from stable_envs.envs.pusht.simulation.triton_batch import PushTTritonAdapter

    count = 8
    options = None
    if variety:
        options = [
            dict(variation=[], variation_values={"agent.shape": i % 8, "block.shape": i % 7 + 1}) for i in range(count)
        ]
    config = dict(resolution=32)
    with (
        closing(PushTTritonAdapter(count, "cuda", 3, config=config)) as gpu,
        closing(PushTNumbaAdapter(count, "cpu", 3, config=config)) as cpu,
    ):
        gpu.reset(5, options)
        cpu.reset(5, options)
        rng = np.random.default_rng(1)
        for _ in range(20):
            # Each step starts both kernels from the Numba State, so float32 rounding does not compound.
            gpu.set_state(cpu.get_state().to("cuda"))
            action = torch.as_tensor(rng.uniform(-1, 1, (count, 2)), dtype=torch.float32)
            got, want = gpu.step(action.cuda()), cpu.step(action)
            observation = ("current", "observation")
            torch.testing.assert_close(got["state"][observation].cpu(), want["state"][observation], rtol=0, atol=0.05)
            torch.testing.assert_close(got["reward"].cpu(), want["reward"], rtol=0, atol=0.05)


def test_cpu_frames_and_states_do_not_depend_on_the_step_thread_counts():
    def rollout(native_threads):
        before = torch.get_num_threads()
        torch.set_num_threads(4)
        try:
            with closing(PushT(num_envs=6, size=96, device="cpu")) as env:
                assert env.native_step_threads
                env.native_step_threads = native_threads
                env.reset(seed=3)
                actions = torch.linspace(-1, 1, 6 * 5 * 2).reshape(6, 5, 2)
                out = env.step(actions, frames=slice(None))
                assert torch.get_num_threads() == 4
                return out["frames"], out["states"]
        finally:
            torch.set_num_threads(before)

    frames, states = rollout(True)
    other_frames, other_states = rollout(False)
    assert torch.equal(frames, other_frames)
    assert (states == other_states).all()
