"""Fetch physics backends: routing, the CPU batch bit for bit against gymnasium-robotics, MuJoCo Warp rollouts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium_robotics")

import stable_envs as snv  # noqa: E402
from stable_envs.envs.fetch.simulation.adapter import FetchAdapter  # noqa: E402
from stable_envs.envs.fetch.simulation.batch import FetchBatchAdapter  # noqa: E402
from stable_envs.envs.fetch.simulation.execution import MJWARP_MIN_ENVS, fetch_backend  # noqa: E402
from tests.helpers.reference.fetch_policy import scripted_actions, source_rollouts  # noqa: E402

TASKS = ("Reach", "Push", "Slide", "PickAndPlace")


def _cuda_problem() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required"
    from stable_envs._core.warp_cuda import warp_cuda_problem

    return warp_cuda_problem()


def cuda(fn):
    problem = _cuda_problem()
    return pytest.mark.gpu(pytest.mark.skipif(problem is not None, reason=str(problem))(fn))


def assert_identical(expected, actual, what):
    for key in expected.keys(True, True):
        a, b = expected[key], actual[key]
        assert a.dtype == b.dtype and torch.equal(a, b), (what, key)


def test_routing():
    assert fetch_backend(1, "cpu") == "native"
    assert fetch_backend(1, "cuda") == "mjwarp"
    assert fetch_backend(4096, "cpu", "auto") == "native"
    assert fetch_backend(MJWARP_MIN_ENVS - 1, "cuda", "auto") == "native"
    assert fetch_backend(MJWARP_MIN_ENVS, "cuda", "auto") == "mjwarp"
    assert fetch_backend(8, "cuda", "auto", 8) == "mjwarp"
    assert fetch_backend(8, "cuda", "native") == "native"
    assert fetch_backend(8, "cuda", "source") == "source"
    with pytest.raises(ValueError, match="requires device='cuda'"):
        fetch_backend(8, "cpu", "mjwarp")
    with pytest.raises(ValueError, match="simulation_backend"):
        fetch_backend(8, "cpu", "mjx")
    with pytest.raises(ValueError, match="mjwarp_min_envs"):
        fetch_backend(8, "cuda", "auto", 0)


def test_public_env_steps_on_the_cpu_batch():
    with snv.make("FetchPush", num_envs=3, device="cpu") as env:
        assert (env.simulation_backend, env.execution_mode) == ("native", "DELEGATED_CPU")
        assert isinstance(env._simulator, FetchBatchAdapter)
    with snv.make("FetchPush", num_envs=3, device="cpu", simulation_backend="source") as env:
        assert env.simulation_backend == "source" and not isinstance(env._simulator, FetchBatchAdapter)


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("task", TASKS)
def test_cpu_batch_matches_the_source_loop_bit_for_bit(task, threads):
    """Every State field and output, through masks, restores, seeded and unseeded resets and variations."""
    lanes = 5
    for reward_type in ("sparse", "dense"):
        reference = FetchAdapter(lanes, seed=7, task=task, reward_type=reward_type)
        candidate = FetchBatchAdapter(lanes, seed=7, task=task, reward_type=reward_type, threads=threads)
        try:
            assert_identical(reference.get_state(), candidate.get_state(), "construction")
            rng = np.random.default_rng(1)
            for episode in range(2):
                variations = None
                if episode:
                    variations = [{} for _ in range(lanes)]
                    variations[1] = {"agent.start_position": np.array([1.3, 0.7])}
                    variations[3] = {"goal.start_position": np.array([1.2, 0.7, 0.4247])}
                    if task != "Reach":
                        variations[2] = {
                            "block.start_position": np.array([1.3, 0.8]),
                            "block.angle": np.array([0.4]),
                            "block.mass": np.array([3.0]),
                        }
                seed = None if episode else 11
                assert_identical(
                    reference.reset(seed, variations), candidate.reset(seed, variations), f"reset {episode}"
                )
                for step in range(55):
                    # float64 and float32 actions round differently in the source's action scaling.
                    dtype = np.float32 if step % 2 else np.float64
                    actions = rng.uniform(-1.3, 1.3, (lanes, 4)).astype(dtype)
                    active = None if step % 3 else rng.random(lanes) < 0.7
                    expected = reference.step(actions, active=active)
                    assert_identical(expected, candidate.step(actions, active=active), f"step {step}")
                    if step == 20:
                        state = reference.get_state()
                        reference.set_state(state)
                        candidate.set_state(state)
        finally:
            reference.close()
            candidate.close()


def _replay(adapter, actions):
    qpos, success = [], None
    for step in range(actions.shape[1]):
        out = adapter.step(actions[:, step])
        qpos.append(out["state"]["qpos"].cpu().numpy())
        success = out["success"].cpu().numpy()
    return np.stack(qpos, 1), success


def _closed_loop(adapter, task, steps):
    """The controller closed loop on ``adapter``, from its sites; returns the final success per environment."""
    t = adapter.task
    adapter.reset(seed=0)
    goal = adapter.get_state()["goal"].cpu().numpy()
    phase, count = np.zeros(adapter.num_envs, int), np.zeros(adapter.num_envs, int)
    for _ in range(steps):
        sites = adapter._site_xpos
        sites = (sites.cpu().numpy() if torch.is_tensor(sites) else sites).astype(np.float64)
        grip, obj = sites[:, t.grip_site], sites[:, t.goal_site]
        out = adapter.step(scripted_actions(task, grip, obj, goal, phase, count))
    return out["success"].cpu().numpy()


@pytest.mark.parametrize("task", TASKS)
def test_cpu_batch_replays_scripted_rollouts_exactly(task):
    episodes, steps = 4, 120
    actions, qpos, success = source_rollouts(task, range(episodes), steps)
    adapter = FetchBatchAdapter(episodes, seed=0, task=task)
    try:
        adapter._set_episode_limits(np.full(episodes, steps))
        adapter.reset(seed=0)
        replayed, replay_success = _replay(adapter, actions)
        assert np.array_equal(replayed, qpos)
        assert np.array_equal(replay_success, success)
    finally:
        adapter.close()


#: Per task: most episodes (of 16) whose replayed outcome may differ, largest robot joint deviation over a
#: 200-step rollout, and largest mean object position deviation. Float32 contacts make pushes and slides
#: diverge more than reaches and grasps.
WARP_BOUNDS = {
    "Reach": (0, 0.005, 0.0),
    "Push": (3, 0.05, 0.03),
    "Slide": (3, 1.0, 0.2),
    "PickAndPlace": (1, 0.02, 0.01),
}


@cuda
@pytest.mark.parametrize("task", TASKS)
def test_mjwarp_rollouts_reach_the_source_outcomes(task):
    from stable_envs.envs.fetch.simulation.mjwarp import FetchWarpAdapter

    episodes, steps = 16, 200
    actions, qpos, success = source_rollouts(task, range(episodes), steps)
    adapter = FetchWarpAdapter(episodes, "cuda", seed=0, task=task)
    try:
        adapter._set_episode_limits(np.full(episodes, steps))
        adapter.reset(seed=0)
        replayed, replay_success = _replay(adapter, torch.as_tensor(actions, device="cuda"))
        closed = _closed_loop(adapter, task, steps)
    finally:
        adapter.close()
    differ, robot_bound, object_bound = WARP_BOUNDS[task]
    assert (replay_success != success).sum() <= differ
    assert abs(int(closed.sum()) - int(success.sum())) <= max(differ, 1)
    assert np.abs(replayed[:, :, :15] - qpos[:, :, :15]).max() <= robot_bound
    if task != "Reach":
        assert np.linalg.norm(replayed[:, :, 15:18] - qpos[:, :, 15:18], axis=-1).mean() <= object_bound


@cuda
def test_mjwarp_resets_masks_and_routing():
    from stable_envs.envs.fetch.simulation.mjwarp import FetchWarpAdapter

    adapter = FetchWarpAdapter(6, "cuda", seed=3, task="PickAndPlace")
    try:
        assert adapter.execution_mode == "NATIVE_WARP"
        start = adapter.get_state()
        reference = FetchBatchAdapter(6, seed=3, task="PickAndPlace")
        # Resets draw the same starts and goals as the CPU batch, up to float32 storage.
        expected = reference.get_state()
        for key in ("goal", ("task", "initial_gripper_xpos"), ("rng", "state_words")):
            assert torch.equal(start[key].cpu(), expected[key])
        assert torch.allclose(start["qpos"].cpu(), expected["qpos"], atol=1e-6)
        reference.close()
        actions = torch.rand(6, 4, device="cuda") * 2 - 1
        active = torch.tensor([True, False, True, False, True, True], device="cuda")
        out = adapter.step(actions, active=active)
        held = ~active.cpu()
        for key in ("qpos", "qvel", ("controller", "position"), ("lifecycle", "elapsed_steps")):
            assert torch.equal(out["state"][key].cpu()[held], start[key].cpu()[held]), key
        assert out["state"]["lifecycle", "elapsed_steps"].cpu()[~held].eq(1).all()
    finally:
        adapter.close()
    with snv.make("FetchReach", num_envs=2, device="cuda") as env:
        assert env.simulation_backend == "mjwarp"
    with snv.make("FetchReach", num_envs=2, device="cuda", simulation_backend="auto") as env:
        assert env.simulation_backend == "native"
        out = env.step(torch.zeros(2, 4, device="cuda"), frames=None)
        assert out["state"]["qpos"].device.type == "cuda" and out["reward"].device.type == "cuda"
