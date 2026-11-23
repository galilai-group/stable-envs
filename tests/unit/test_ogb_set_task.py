"""``OGBCube.set_task`` and ``OGBScene.set_task`` against OGBench's data-collection mode."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

import stable_envs as snv
from stable_envs.envs.ogbench.simulation.compatibility import TARGET_TASKS
from stable_envs.envs.ogbench.simulation.manipulation import FLAGS, PCG_KEYS, OGBManipulationAdapter, site_ids
from stable_envs.envs.ogbench.simulation.pcg import pcg_decode
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.unit

FAMILIES = ("Cube", "Scene")


def _upstream(family):
    """OGBench's own environment, built in data-collection mode."""
    if family == "Cube":
        from ogbench.manipspace.envs.cube_env import CubeEnv

        env = CubeEnv(env_type="single", mode="data_collection", ob_type="states", visualize_info=False)
    else:
        from ogbench.manipspace.envs.scene_env import SceneEnv

        env = SceneEnv(env_type="scene", mode="data_collection", ob_type="states", visualize_info=False)
    env.reset(seed=0)
    env.family = family
    return env


def _load(env, state, i):
    """Put row ``i`` of a State into the OGBench environment ``env``, as the State keeps it."""
    get = lambda name: snv.get_field(state, name)[i].numpy()  # noqa: E731
    for _ in range(2):
        for name, flag in FLAGS.items():
            if mujoco.mj_stateSize(env.model, flag):
                mujoco.mj_setState(env.model, env.data, get(f"physics/{name}").astype(np.float64), flag)
        mujoco.mj_forward(env.model, env.data)
    for name, flag in FLAGS.items():
        if mujoco.mj_stateSize(env.model, flag):
            mujoco.mj_setState(env.model, env.data, get(f"physics/{name}").astype(np.float64), flag)
    pos, rot = site_ids(env)
    env.data.site_xpos[pos] = get("controller/site_xpos").reshape(-1, 3)
    env.data.site_xmat[rot] = get("controller/site_xmat").reshape(-1, 9)
    env.data.body("ur5e/robotiq/right_pad").cfrc_ext[:] = get("output/gripper_force")
    env._prev_qpos, env._prev_qvel = get("output/prev_qpos").copy(), get("output/prev_qvel").copy()
    env._success = bool(get("task/success")[0])
    if env.family == "Scene":
        env._cur_button_states = get("scene/cur_button_states").copy()
        env._prev_button_states = get("scene/prev_button_states").copy()
        env._target_button_states = get("scene/target_button_states").copy()
        env._target_drawer_pos = float(get("scene/target_drawer_pos")[0])
        env._target_window_pos = float(get("scene/target_window_pos")[0])
        env.model.site("drawer_handle_center_target").pos[1] = env._target_drawer_pos
        env.model.site("window_handle_center_target").pos[0] = env._target_window_pos
        env._apply_button_states()
    env.np_random.bit_generator.state = pcg_decode({key: get(f"rng/environment/{key}") for key in PCG_KEYS})


def _draw(env, seed):
    """OGBench's target draw on ``env``, from a generator seeded as Gymnasium seeds ``np_random``."""
    from gymnasium.utils import seeding

    kept = env._np_random, env._np_random_seed
    env._np_random, env._np_random_seed = seeding.np_random(seed)
    try:
        env.set_new_target(return_info=False)
    finally:
        env._np_random, env._np_random_seed = kept


def _oracle(env):
    from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
    from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
    from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle

    kind = {"cube": CubeMarkovOracle, "button": ButtonMarkovOracle}
    kind.update(drawer=DrawerMarkovOracle, window=WindowMarkovOracle)
    oracle = kind[str(env._target_task)](env=env)
    oracle.reset(None, env.compute_ob_info())
    return oracle


def _seeds_by_task(family, state):
    """The first seed that draws each target task for row 0 of ``state``."""
    env, found = _upstream(family), {}
    for seed in range(400):
        _load(env, state, 0)
        _draw(env, seed)
        found.setdefault(str(env._target_task), seed)
        if len(found) == (1 if family == "Cube" else 4):
            break
    env.close()
    return found


CASES = [("Cube", "cube")] + [("Scene", task) for task in TARGET_TASKS]


@pytest.mark.parametrize(("family", "task"), CASES)
def test_target_and_rollout_equal_ogbench_data_collection_mode(family, task):
    """OGBench built with mode="data_collection" draws the same target, and its oracle reaches it in both."""
    np.random.seed(0)
    env = snv.make(f"OGB{family}", num_envs=2, device="cpu", seed=5)
    upstream = _upstream(family)
    try:
        state = env.reset(seed=8)
        for _ in range(3):
            state = env.step(torch.full((2, 5), 0.2))["state"]
        seed = _seeds_by_task(family, state)[task]
        _load(upstream, state, 0)
        _draw(upstream, seed)
        state = env.set_task([0], seed=[seed])
        assert snv.get_field(state, "task/target/task")[0, 0] == TARGET_TASKS.index(task)
        want = upstream.compute_ob_info()
        info = env.source_info(state)
        for key, value in want.items():
            if key.startswith("privileged/target") and key != "privileged/target_task":
                assert np.array_equal(info[key][0].numpy(), np.asarray(value)), key
        assert np.array_equal(snv.get_field(state, "physics/mocap_pos")[0].numpy(), upstream.data.mocap_pos.ravel())
        oracle, reached = _oracle(upstream), False
        for _ in range(300):
            action = np.asarray(oracle.select_action(None, upstream.compute_ob_info()), np.float32)
            _, reward, terminated, _, out = upstream.step(action)
            got = env.step(torch.from_numpy(np.stack([action, np.zeros(5, np.float32)])))
            assert bool(got["success"][0]) == bool(out["success"]), "success"
            assert float(got["reward"][0]) == reward and bool(got["terminated"][0]) == terminated
            assert np.array_equal(snv.get_field(got["state"], "physics/qpos")[0].numpy(), upstream.data.qpos)
            if out["success"]:
                reached = True
                break
        assert reached, f"the OGBench oracle did not reach the {task} target"
    finally:
        upstream.close()
        env.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_state_and_outputs_equal_the_ogbench_objects(family):
    """Every row and output after ``set_task``, against the OGBench objects of the per-environment adapter."""
    rng = np.random.default_rng(1)
    ref = OGBManipulationAdapter(family, 3, seed=5)
    env = snv.make(f"OGB{family}", num_envs=3, device="cpu", seed=5)
    try:
        env.reset(seed=8)
        ref.reset(seed=8)
        for _ in range(4):
            action = rng.uniform(-1, 1, (3, 5)).astype(np.float32)
            ref.step(action)
            env.step(torch.from_numpy(action))
        state = env.set_task([2, 0], seed=[21, None])
        for i, seed in ((2, 21), (0, None)):
            ref.envs[i]._mode = "data_collection"
            if seed is None:
                ref.envs[i].set_new_target(return_info=False)
            else:
                _draw(ref.envs[i], seed)
        ref._forget_static(ref.envs)
        assert_same_state(state, ref.get_state())
        goal = env.source_info(state)["goal"]
        assert goal[[0, 2]].isnan().all() and not goal[1].isnan().any()
        for _ in range(30):
            action = rng.uniform(-1, 1, (3, 5)).astype(np.float32)
            want, got = ref.step(action), env.step(torch.from_numpy(action))
            assert_same_state(got["state"], want["state"])
            for key in ("reward", "success", "terminated", "truncated"):
                assert torch.equal(got[key], want[key]), key
        assert not snv.get_field(env.reset(seed=3), "task/mode").any()
    finally:
        ref.close()
        env.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_a_seeded_draw_leaves_the_row_generator_as_it_was(family):
    """The unseeded reset after a seeded ``set_task`` equals that of a twin row that got none."""
    a = snv.make(f"OGB{family}", num_envs=2, device="cpu", seed=4)
    b = snv.make(f"OGB{family}", num_envs=2, device="cpu", seed=4)
    try:
        for env in (a, b):
            env.reset(seed=1)
            env.step(torch.full((2, 5), 0.1))
        a.set_task([0], seed=[9])
        assert_same_state(a.reset(), b.reset())
    finally:
        a.close()
        b.close()


def test_scene_repairs_a_lost_cube_in_task_mode_only():
    ref = OGBManipulationAdapter("Scene", 2, seed=0)
    env = snv.make("OGBScene", num_envs=2, device="cpu", seed=0)
    try:
        env.reset(seed=2)
        state = env.set_task([0], seed=[5]).clone()
        cube = int(env._simulator._cube)
        snv.get_field(state, "physics/qpos")[:, cube + 2] = -1.0
        env.set_state(state)
        ref.set_state(state)
        action = np.zeros((2, 5), np.float32)
        got, want = env.step(torch.from_numpy(action)), ref.step(action)
        assert_same_state(got["state"], want["state"])
        z = snv.get_field(got["state"], "physics/qpos")[:, cube + 2]
        assert z[0] < -0.5 and z[1] == 0.02
    finally:
        ref.close()
        env.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_rows_after_steps_equal_the_same_rows_of_a_full_set_task(family):
    a = snv.make(f"OGB{family}", num_envs=3, device="cpu")
    b = snv.make(f"OGB{family}", num_envs=3, device="cpu")
    try:
        for env in (a, b):
            env.reset(seed=2)
            env.step(torch.full((3, 5, 5), 0.3))
        start = a.get_state()
        part = a.set_task([1], seed=[40])
        full = b.set_task(torch.tensor([True, True, True]), seed=[30, 40, 50])
        for key, value in part.items(True, True):
            assert torch.equal(value[1], full[key][1]), key
            assert torch.equal(value[[0, 2]], start[key][[0, 2]]), key
        action = torch.full((3, 5), -0.2)
        assert torch.equal(
            a.step(action)["state"]["physics", "float64"][1], b.step(action)["state"]["physics", "float64"][1]
        )
        assert a.set_task([], seed=[]) is not None
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_set_task_with_worker_processes_equals_in_process(family):
    from stable_envs.envs.ogbench.simulation.manipulation_batch import OGBManipulationBatchAdapter
    from stable_envs.envs.ogbench.simulation.manipulation_process import OGBManipulationProcessAdapter

    local = OGBManipulationBatchAdapter(family, 3, seed=17)
    remote = OGBManipulationBatchAdapter(
        family, 3, seed=17, instances=OGBManipulationProcessAdapter(family, 3, seed=17, processes=2)
    )
    try:
        local.reset(seed=5)
        remote.reset(seed=5)
        action = np.full((3, 5), 0.3, np.float32)
        local.step(action)
        remote.step(action)
        tasks = [(11,), None, (None,)]
        local.set_task(tasks)
        remote.set_task(tasks)
        assert_same_state(remote.get_state(), local.get_state())
        for _ in range(5):
            assert_same_state(remote.step(action)["state"], local.step(action)["state"])
        assert_same_state(remote.reset(seed=None), local.reset(seed=None))
    finally:
        local.close()
        remote.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_set_task_state_restores_and_has_no_goal_frame(family):
    env = snv.make(f"OGB{family}", num_envs=2, device="cpu")
    other = snv.make(f"OGB{family}", num_envs=2, device="cpu")
    try:
        env.reset(seed=6)
        state = env.set_task([1], seed=[3])
        other.reset(seed=1)
        other.set_state(state)
        assert_same_state(other.get_state(), state)
        action = torch.full((2, 5), 0.2)
        assert_same_state(other.step(action)["state"], env.step(action)["state"])
        with pytest.raises(ValueError, match=r"rows \[1\]"):
            env.goal_render_state(state)
        env.goal_render_state(state[[0]])
    finally:
        env.close()
        other.close()


def test_set_task_refuses_bad_arguments():
    env = snv.make("OGBCube", num_envs=2, device="cpu")
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError, match="seed"):
            env.set_task([0, 1], seed=[1])
        with pytest.raises(ValueError, match="takes no options"):
            env.set_task([0], seed=[1], options={"p_stack": 0.1})
        with pytest.raises(ValueError, match="distinct"):
            env.set_task([0, 0], seed=[1, 2])
        with pytest.raises(ValueError, match="mask"):
            env.set_task(torch.tensor([True]), seed=[1])
        with pytest.raises(ValueError, match="CPU tensor"):
            env.set_task([0], seed=torch.tensor([1], device="meta"))
        env.set_task([1], seed=torch.tensor([4]))
    finally:
        env.close()
