"""The Numba CPU step and the path-seeded reset give the bytes of the Torch step and the full-tree reset."""

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs._core.reset_tree import PathSeeder
from stable_envs._core.reset_tree import sample as sample_space

pytestmark = pytest.mark.family("navigation")

NAMES = ["TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones"]

VARIATION = {
    "TwoRoom": lambda b: {
        "wall.axis": np.arange(b).reshape(-1, 1) % 2,
        "wall.thickness": np.linspace(7.5, 40.25, b).reshape(-1, 1),
        "agent.speed": np.linspace(1.25, 17.75, b).reshape(-1, 1),
        "agent.radius": np.linspace(7, 13.75, b).reshape(-1, 1),
        "door.number": np.linspace(1.0, 3.0, b).reshape(-1, 1),
        "door.size": np.tile([[3.5, 9.25, 20.0]], (b, 1)),
        "door.position": np.tile([[30.0, 112.5, 190.25]], (b, 1)),
    },
    "Piecewise": lambda b: {
        "agent.speed": np.linspace(0.5, 21.5, b).reshape(-1, 1),
        "agent.radius": np.linspace(7, 13.75, b).reshape(-1, 1),
        **{f"zones.bias_{i}": np.linspace(-9.75, 9.5 - i, 2 * b).reshape(b, 2) for i in range(4)},
    },
}
VARIATION["TwoRoomTarget"] = VARIATION["TwoRoom"]
VARIATION["PiecewiseZones"] = VARIATION["Piecewise"]


def _leaves(result):
    return [(key, value.dtype, tuple(value.shape), value.numpy().tobytes()) for key, value in result.items(True, True)]


def _rollout(name, backend, batch, seed, actions, variation=True):
    with se.make(name, num_envs=batch, device="cpu", simulation_backend=backend) as env:
        assert env.simulation_backend == backend
        if variation:
            env.set_variation(VARIATION[name](batch))
        state = env.reset(seed=seed)
        out = [_leaves(state)]
        for t0 in range(0, actions.shape[1], 8):
            out.append(_leaves(env.step(actions[:, t0 : t0 + 8], states=slice(None), frames=None)))
        for action in actions[:, :4].unbind(1):
            out.append(_leaves(env.step(action, frames=None)))
        return out


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("batch", [1, 5, 64])
def test_numba_steps_match_torch_bytes(name, batch):
    rng = np.random.default_rng(batch)
    actions = torch.from_numpy(rng.uniform(-3, 3, (batch, 48, 2)).astype(np.float32))
    for variation in (False, True):
        assert _rollout(name, "numba", batch, 11, actions, variation) == _rollout(
            name, "torch", batch, 11, actions, variation
        )


@pytest.mark.parametrize("name", ["TwoRoom", "Piecewise"])
def test_numba_masks_finished_lanes_like_torch(name):
    # The first lanes start on their targets and finish at once; the sequence holds them.
    agent = np.array([[60, 112], [60, 112], [100, 60], [150, 150]], np.float32)
    target = np.array([[60, 112], [70, 112], [180, 180], [30, 30]], np.float32)
    actions = torch.tensor([[[1.0, 0.0]] * 12] * 4)
    results = []
    for backend in ("numba", "torch"):
        with se.make(name, num_envs=4, device="cpu", simulation_backend=backend) as env:
            env.reset(seed=0, options={"state": agent, "target_state": target})
            results.append(_leaves(env.step(actions, states=slice(None), frames=None)))
    assert results[0] == results[1]


@pytest.mark.parametrize("name", NAMES)
def test_numba_float64_and_strided_actions_match_torch(name):
    actions = torch.from_numpy(np.random.default_rng(3).uniform(-2, 2, (6, 9, 4)))[..., ::2]
    assert actions.dtype == torch.float64 and not actions.is_contiguous()
    results = []
    for backend in ("numba", "torch"):
        with se.make(name, num_envs=6, device="cpu", simulation_backend=backend) as env:
            env.reset(seed=4)
            results.append([_leaves(env.step(actions[:, t], frames=None)) for t in range(9)])
            rows = env._render_state.clone()
            step = torch.full((6, 2), 0.75)
            results.append(env._advance_rows(rows, step, env._style).numpy().tobytes())
            results.append(env._advance_rows(rows, step.double(), env._style).numpy().tobytes())
    assert results[:3] == results[3:]


@pytest.mark.parametrize("name", ["TwoRoomTarget", "PiecewiseZones"])
def test_numba_non_finite_state_raises_like_torch(name):
    for backend in ("numba", "torch"):
        with se.make(name, num_envs=2, device="cpu", simulation_backend=backend) as env:
            env.reset(seed=0)
            with pytest.raises(ValueError, match="non-finite|nonfinite|NaN"):
                env.step(torch.tensor([[0.0, 0.0], [float("nan"), 0.0]]), frames=None)


def test_numba_selection():
    with se.Piecewise(num_envs=2, device="cpu") as env:
        assert (env.simulation_backend, env.execution_mode) == ("numba", "NATIVE_NUMBA")
    with se.PiecewiseZones(num_envs=2, device="cpu", simulation_backend="torch") as env:
        assert (env.simulation_backend, env.execution_mode) == ("torch", "NATIVE_TORCH")
    with pytest.raises(ValueError, match="simulation_backend"):
        se.TwoRoomTarget(simulation_backend="jax")


def _full_tree_values(env, seed):
    """Reset values drawn the slow way: seed the whole tree for each lane and call ``Box.sample``."""
    tree = env.reset_tree_factory()
    out = {leaf.path: [] for leaf in env.reset_leaves}
    for lane in range(env.B):
        tree.seed(seed + lane)
        for leaf in env.reset_leaves:
            node = tree
            for part in leaf.path.split("."):
                node = node[part]
            while True:
                value = node.sample()
                if env._reset_value_ok(leaf.path, value, lane):
                    break
            out[leaf.path].append(value)
    return {path: np.stack(rows) for path, rows in out.items()}


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seed", [0, 7, 2**31 - 3])
def test_path_seeded_reset_matches_full_tree_seeding(name, seed):
    with se.make(name, num_envs=37, device="cpu") as env:
        env.set_variation(VARIATION[name](37))
        values = env.draw_reset_values(seed)
        expected = _full_tree_values(env, seed)
        for path, rows in expected.items():
            assert values[path].dtype == rows.dtype
            assert values[path].tobytes() == rows.tobytes(), path


def test_path_seeder_matches_tree_seed_and_sample():
    from stable_envs.envs.navigation.envs import _piecewise_reset_tree, _two_room_reset_tree

    for factory, extra in ((_two_room_reset_tree, "door.number"), (_piecewise_reset_tree, "zones.bias_2")):
        fast, full = factory(), factory()
        seeder = PathSeeder(fast, ["agent.position", "target.position", extra])
        assert seeder.exact
        for seed in (0, 1, 123456):
            seeder.seed(seed)
            full.seed(seed)
            for path in ("agent.position", "target.position", extra):
                a, b = fast, full
                for part in path.split("."):
                    a, b = a[part], b[part]
                for _ in range(3):
                    x, y = sample_space(a), b.sample()
                    assert x.dtype == y.dtype and x.tobytes() == y.tobytes()
