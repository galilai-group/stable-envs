"""Roadster on CUDA: routing, the native batch on a CUDA env, and MuJoCo Warp against the source's expert.

MuJoCo Warp steps in float32, so a rollout follows the source approximately. The check replays the actions of
the source's scripted expert, run closed loop on the source, for 300 steps under wind and rain: each
environment must keep its task outcome (success or failure and when) and stay close to the source's path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.gpu]

pytest.importorskip("mujoco_warp")

import stable_envs as snv  # noqa: E402
from stable_envs.envs.roadster.simulation.execution import MJWARP_MIN_ENVS  # noqa: E402
from stable_envs.envs.roadster.source import GoalExpert, RoadsterPlanEnv  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402

LANES, STEPS, SEED = 8, 300, 100
WIND, RAIN = 5.0, 0.3
#: Largest distance in metres between the source's and MuJoCo Warp's car, up to the end of each episode. On an
#: RTX 4090 the gaps over 8 environments were 0.006 m (traffic), 0.011 m (carhop), 0.27-0.33 m (rooftop) and
#: 2.7-3.4 m (loop, at 20 m/s around the half-pipe).
PATH_BOUNDS = {"traffic": 0.1, "carhop": 0.1, "rooftop": 1.5, "loop": 8.0}
#: The mission's car crosses the loop at 20 m/s about 75 steps in. The source amplifies any difference there:
#: a 1e-9 m/s change of the start velocity moves the source's own car 13 m by step 300. The path bound covers the
#: steps before the loop; the outcome check covers the whole rollout.
MISSION_STEPS_BEFORE_LOOP, MISSION_BOUND = 60, 0.5
#: Tasks that cross the half-pipe loop at 20 m/s, where the source amplifies any difference (in Loop a 1e-9 m/s
#: change of the start velocity moves the source's car 5-95 m by step 300): one environment of the 8 may change
#: outcome, which MuJoCo Warp's run-to-run differences cause about once in ten runs.
CHAOTIC = ("loop", "mission")
#: Steps by which the first success or failure may come earlier or later than in the source.
OUTCOME_SLACK = 3


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("Roadster on MuJoCo Warp requires CUDA")


def _first_end(terminated: np.ndarray, success: np.ndarray) -> tuple[int, bool] | None:
    hits = np.flatnonzero(terminated)
    return None if not hits.size else (int(hits[0]), bool(success[hits[0]]))


def test_cuda_batches_route_to_mujoco_warp_and_auto_keeps_small_batches_native():
    _require_cuda()
    with snv.RoadsterTrafficJam(num_envs=2, device="cuda") as env:
        assert env.simulation_backend == "mjwarp" and env.execution_mode == "MUJOCO_WARP"
        out = env.step(torch.zeros(2, 4, device="cuda"), frames=None)
        assert out["state"]["sim", "qpos"].device.type == "cuda" and out["reward"].device.type == "cuda"
    with snv.RoadsterTrafficJam(num_envs=MJWARP_MIN_ENVS - 1, device="cuda", simulation_backend="auto") as env:
        assert env.simulation_backend == "native" and env.execution_mode == "DELEGATED_CPU"


def test_the_native_batch_on_a_cuda_env_steps_as_on_cpu():
    _require_cuda()
    actions = np.random.default_rng(0).uniform(-1, 1, (3, 6, 4)).astype(np.float32)
    options = {"variation_values": {"weather.wind": [6.0], "weather.rain": [0.4]}}
    with (
        snv.RoadsterCarHop(num_envs=3, device="cuda", simulation_backend="native") as gpu,
        snv.RoadsterCarHop(num_envs=3, device="cpu") as cpu,
    ):
        gpu.reset(seed=SEED, options=options)
        cpu.reset(seed=SEED, options=options)
        a, b = gpu.step(torch.as_tensor(actions, device="cuda"), frames=None), cpu.step(actions, frames=None)
        assert_same_state(a.cpu(), b)


def test_an_inactive_environment_holds_its_state_on_mujoco_warp():
    _require_cuda()
    with snv.RoadsterCarHop(num_envs=3, device="cuda") as env:
        env.reset(seed=SEED, options={"variation_values": {"weather.wind": [6.0]}})
        for _ in range(3):
            env.step(torch.full((3, 4), 0.3, device="cuda"), frames=None)
        before = env.get_state()
        out = env._step(
            torch.full((3, 4), 0.5, device="cuda"), active=torch.tensor([True, False, True], device="cuda")
        )
        held = out["state"][1]
        for key in before.keys(True, True):
            assert torch.equal(held[key], before[1][key]), key
        assert not torch.equal(out["state"]["sim", "qpos"][0], before["sim", "qpos"][0])
        for key in out.exclude("state").keys(True, True):
            assert not out[key][1].any(), key


@pytest.mark.parametrize("task", ["loop", "mission", "traffic", "rooftop", "carhop"])
def test_mujoco_warp_replays_the_source_expert_with_its_outcome(task):
    _require_cuda()
    from stable_envs.envs.roadster.simulation.warp import RoadsterWarpAdapter

    sources = [RoadsterPlanEnv(task=task) for _ in range(LANES)]
    for i, src in enumerate(sources):
        if task == "loop":
            src._loop_goal_state()  # the batch takes the reference run once, before any episode
        src.reset(seed=SEED + i, wind=WIND, rain=RAIN)
    expert = GoalExpert()
    actions = np.zeros((STEPS, LANES, 4), np.float32)
    src_pos, src_term, src_success = (
        np.zeros((STEPS, LANES, 3)),
        np.zeros((STEPS, LANES), bool),
        np.zeros((STEPS, LANES), bool),
    )
    for t in range(STEPS):
        for i, src in enumerate(sources):
            actions[t, i] = expert(src)
            _, terminated, _, info = src.step(actions[t, i])
            src_pos[t, i], src_term[t, i], src_success[t, i] = src.data.qpos[:3], terminated, info["success"]
    gpu = RoadsterWarpAdapter(task, LANES, "cuda")
    try:
        gpu.configure_physics(np.tile([WIND, RAIN], (LANES, 1)))
        gpu.reset(SEED)
        pos, term, success = np.zeros_like(src_pos), np.zeros_like(src_term), np.zeros_like(src_success)
        for t in range(STEPS):
            out = gpu.step(torch.as_tensor(actions[t], device="cuda"))
            pos[t] = out["state"]["sim", "qpos"][:, :3].cpu().numpy()
            term[t], success[t] = out["terminated"].cpu().numpy(), out["success"].cpu().numpy()
    finally:
        gpu.close()
    flips = 0
    for i in range(LANES):
        want, got = _first_end(src_term[:, i], src_success[:, i]), _first_end(term[:, i], success[:, i])
        same = (want is None and got is None) or (
            want is not None and got is not None and want[1] == got[1] and abs(want[0] - got[0]) <= OUTCOME_SLACK
        )
        flips += not same
        if task not in CHAOTIC:
            assert same, f"{task} environment {i}: source ends {want}, MuJoCo Warp ends {got}"
        if task != "mission" and same:
            end = STEPS if want is None else want[0] + 1
            gap = float(np.abs(pos[:end, i] - src_pos[:end, i]).max())
            assert gap <= PATH_BOUNDS[task], f"{task} environment {i}: {gap:.3f} m from the source's path"
    if task in CHAOTIC:
        assert flips <= 1, f"{task}: {flips} of {LANES} environments changed outcome"
    if task == "mission":
        gap = float(np.abs(pos[:MISSION_STEPS_BEFORE_LOOP] - src_pos[:MISSION_STEPS_BEFORE_LOOP]).max())
        assert gap <= MISSION_BOUND, f"mission: {gap:.3f} m from the source's path before the loop"
