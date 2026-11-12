from contextlib import closing

import numpy as np
import pytest
import torch

from stable_envs.envs.pusht.semantics import PushT
from stable_envs.envs.pusht.simulation.jax_batch import PushTJaxAdapter
from stable_envs.envs.pusht.simulation.replay_batch import PushTReplayAdapter, resolve_variation
from tests.helpers.reference.semantics import assert_same_state


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return request.param


@pytest.mark.parametrize("discrete", [False, True])
def test_mixed_action_dtypes_and_reset_restore(device, discrete):
    with closing(PushTJaxAdapter(2, device, 8, discrete=discrete, config=dict(resolution=32))) as env:
        a = torch.tensor([[3, 4], [8, 9]] if discrete else [[0.2, -0.3], [-0.4, 0.5]], device=device)
        env.step(a)
        env.step(a.to(torch.int32 if discrete else torch.float64))
        checkpoint = env.get_state()
        expected = env.step(a)
        env.set_state(checkpoint)
        assert_same_state(expected, env.step(a))
        before = env.get_state()
        expected_reset = env.reset(seed=17)
        env.set_state(before)
        assert_same_state(expected_reset, env.reset(seed=17))


@pytest.mark.parametrize(
    "initial,action,steps,atol",
    [
        ([480, 100, 200, 250, 0, 0, 0], [1.0, 0.0], 200, 5e-4),
        ([100, 215, 200, 200, 0, 0, 0], [0.5, 0.0], 200, 5e-4),
        # Outside the action space, as SWM accepts. The faster agent lets float32 rounding grow: over 200 steps
        # the largest gap measured was 0.0078 px of observation and 0.0048 of reward.
        ([300, 300, 200, 250, 0, 0, 0], [1.6, -1.3], 200, 2e-2),
    ],
)
def test_relative_controls_and_contact_track_main(device, initial, action, steps, atol):
    options = dict(
        variation=[], state=np.array(initial, np.float64), goal_state=np.array([100, 100, 300, 250, np.pi / 4, 0, 0])
    )
    with (
        closing(PushTJaxAdapter(1, device, 0, config=dict(resolution=32))) as env,
        closing(PushTReplayAdapter(1, "cpu", 0, config=dict(resolution=32))) as ref,
    ):
        env.reset(0, options)
        ref.reset(0, options)
        for _ in range(steps):
            got = env.step(torch.tensor([action], device=device))
            want = ref.step(np.array([action], np.float32))
            np.testing.assert_allclose(
                got["state", "current", "observation"].cpu(),
                want["state", "current", "observation"],
                atol=atol,
                rtol=0,
            )
            assert got["reward"].item() == pytest.approx(want["reward"].item(), abs=atol)
            assert got["success"].item() == want["success"].item()
            assert not got["truncated"].any()


def test_shapes_variations_reward_mask_and_source_checkpoint(device, monkeypatch):
    options = [
        dict(
            variation=[],
            variation_values={"agent.shape": i, "block.shape": i % 7 + 1},
            state=np.array([100, 100, 350, 300, 0.2, 0, 0], np.float64),
            goal_state=np.array([100, 100, 300, 250, 0.7, 0, 0]),
        )
        for i in range(8)
    ]
    with (
        closing(PushTJaxAdapter(8, device, 0, config=dict(resolution=32))) as env,
        closing(PushTReplayAdapter(8, "cpu", 0, config=dict(resolution=32))) as ref,
    ):
        source = ref.reset(0, options)
        env.set_state(source.to(device))
        np.testing.assert_array_equal(
            env.get_state()["current", "observation"].cpu(), source["current", "observation"]
        )
        before = env.get_state()

        def forbidden(*args, **kwargs):
            raise AssertionError("CPU physics during a JAX transition")

        for owner in env.envs:
            monkeypatch.setattr(owner.env, "advance", forbidden)
            monkeypatch.setattr(owner.env.space, "step", forbidden)
        result = env.step(torch.full((8, 2), 0.2, device=device), active=torch.arange(8, device=device) != 0)
        state = result["state"]
        assert torch.isfinite(state["current", "observation"]).all()
        for key, value in before["current"].items(True, True):
            assert torch.equal(value[0], state["current"][key][0])
        expected = -(state["current", "observation"] - state["current", "goal", "value"]).norm(dim=1)
        torch.testing.assert_close(result["reward"], expected, rtol=0, atol=0)
        assert not result["truncated"].any()
        env.set_state(state)
        assert_same_state(state, env.get_state())


@pytest.mark.parametrize("failure", [RuntimeError("Unknown backend cuda"), IndexError()])
def test_missing_jax_cuda_fails_before_constructing_owners(monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise failure

    def forbidden(*args, **kwargs):
        raise AssertionError("Constructed CPU owners before checking JAX CUDA support")

    monkeypatch.setattr("jax.devices", unavailable)
    monkeypatch.setattr(PushTReplayAdapter, "__init__", forbidden)
    with pytest.raises(RuntimeError, match=r"pip install 'stable-envs\[pusht-gpu\]'"):
        PushTJaxAdapter(1, "cuda:0")


@pytest.mark.parametrize("discrete", [False, True])
def test_episode_checkpoint_restores_without_physics_replay(device, discrete, monkeypatch):
    with closing(PushTJaxAdapter(2, device, 0, discrete=discrete, config=dict(resolution=32))) as env:
        env.reset(0, dict(variation=[], state=np.array([100, 215, 200, 200, 0, 0, 0], np.float64)))
        for _ in range(200):
            relative = (env._body[:, 1, :2] - env._body[:, 0, :2]) / 100
            if discrete:
                angle = torch.atan2(relative[:, 1], relative[:, 0]).remainder(2 * np.pi)
                action = torch.stack((relative.norm(dim=1) * 16, angle * (16 / (2 * np.pi))), 1)
                action = action.floor().clamp(0, 15).to(torch.int64)
            else:
                action = relative.clamp(-1, 1).to(torch.float32)
            env.step(action, capture_state=False)
        assert env._impulse.count_nonzero() > 0
        checkpoint = env.get_state()
        expected = env.step(action)

        def forbidden(*args, **kwargs):
            raise AssertionError("Checkpoint restore replayed physics")

        with monkeypatch.context() as patch:
            patch.setattr("stable_envs.envs.pusht.simulation.jax_batch.advance", forbidden)
            env.set_state(checkpoint)
        assert_same_state(checkpoint, env.get_state())
        assert_same_state(expected, env.step(action))
        bad = checkpoint.clone()
        bad["current", "jax", "bias"][1, 0, 0] = float("nan")
        before = env.get_state()
        with pytest.raises(ValueError, match="solver cache"):
            env.set_state(bad)
        assert_same_state(before, env.get_state())


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("processes", [0, 1])
def test_cpu_restores_and_renders_jax_checkpoints(device, discrete, processes, monkeypatch):
    import stable_envs as se

    monkeypatch.setattr("stable_envs.envs.pusht.simulation.execution.process_count", lambda *a, **kw: processes)

    with closing(PushTJaxAdapter(1, device, 0, discrete=discrete, config=dict(resolution=32))) as env:
        action = torch.tensor([[3, 4]] if discrete else [[0.5, 0.0]], device=device)
        state = env.step(action)["state"].cpu()
        cls = se.PushTDiscrete if discrete else se.PushT
        with cls(device="cpu", size=32) as cpu:
            frame = cpu.render(state)
            assert frame.shape == (1, 32, 32, 3)
            assert torch.equal(frame, cpu.render(cpu.to_render_state(state)))
            cpu.set_state(state)
            assert_same_state(state, cpu.get_state())
            assert torch.equal(frame, cpu.render())


@pytest.mark.parametrize("discrete", [False, True])
def test_reset_workers_match_in_process_resets(device, discrete):
    count, config = 5, dict(resolution=32)
    with (
        closing(PushTJaxAdapter(count, device, 3, discrete=discrete, config=config)) as ref,
        closing(PushTJaxAdapter(count, device, 3, discrete=discrete, config=config, processes=2)) as env,
    ):
        adapters = (ref, env)

        def same(call):
            expected, actual = (call(adapter) for adapter in adapters)
            assert_same_state(actual, expected)
            for key, value in ref.render_fields().items():
                assert torch.equal(env.render_fields()[key], value), key
            return actual

        def play(steps):
            for step in range(steps):
                action = torch.tensor(
                    [[3 + step % 5, 4]] * count if discrete else [[0.3, -0.2 + 0.1 * step]] * count, device=device
                )
                active = torch.arange(count, device=device) != step % count
                same(lambda adapter, a=action, m=active: adapter.step(a, active=m))

        same(lambda adapter: adapter.reset(seed=17))
        play(3)
        shapes = [
            dict(variation=[], variation_values={"agent.shape": lane % 8, "block.shape": (lane + 2) % 7 + 1})
            for lane in range(count)
        ]
        same(lambda adapter: adapter.reset(seed=18, options=shapes))
        play(4)
        values = {"block.start_position": np.linspace([120.0, 150.0], [380.0, 300.0], count)}
        same(lambda adapter: adapter.reset(seed=19, options=dict(variation_values=values)))
        play(2)

        # Variation changes while the workers own the lanes.
        space = ref.template.env.variation_space
        lanes = resolve_variation(
            space,
            {"agent.color": np.arange(3 * count, dtype=np.uint8).reshape(count, 3), "block.scale": 45.0},
            count,
        )
        for adapter in adapters:
            adapter.install_variation(lanes, PushT.immediate_variation_leaves)
        same(lambda adapter: adapter.get_state())
        keys = ["agent.color", "agent.start_position", "block.scale"]
        for attribute in ("value", "init_value"):
            np.testing.assert_array_equal(env.variation_rows(attribute, keys), ref.variation_rows(attribute, keys))
        same(lambda adapter: adapter.reset(seed=20))
        play(2)

        # After a restore the parent owns the lanes; the next reset gives the workers their metadata.
        checkpoint = ref.get_state()
        same(lambda adapter: adapter.reset(seed=21))
        same(lambda adapter: (adapter.set_state(checkpoint), adapter.get_state())[1])
        play(1)
        same(lambda adapter: adapter.reset(seed=22))

        # A reset that fails in one lane changes no lane.
        before = env.get_state()
        bad = [dict(variation=[])] * (count - 1) + [dict(variation=[], variation_values={"agent.shape": 99})]
        for adapter in adapters:
            with pytest.raises(ValueError):
                adapter.reset(seed=23, options=bad)
        assert_same_state(env.get_state(), before)
        same(lambda adapter: adapter.reset(seed=24))
        assert env.reset(seed=None)["current", "observation"].isfinite().all()

        # The parent owners rebuilt from the batch columns equal the in-process owners.
        same(lambda adapter: adapter.reset(seed=25))
        assert env._stale
        for expected, actual in zip(ref.envs, env.envs, strict=True):
            fields = expected.static_fields()
            assert actual.static_fields().keys() == fields.keys()
            for key, value in fields.items():
                np.testing.assert_array_equal(actual.static_fields()[key], value, err_msg=key)
        assert not env._stale
