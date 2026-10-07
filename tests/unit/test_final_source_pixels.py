from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from tests.helpers.reference import fetch_pixels
from tests.helpers.reference.profiles import PROFILES
from tests.helpers.reference.pusht_pixels import compare_padded_states


@pytest.mark.parametrize("attribute", ["_renderer", "custom_renderer"])
def test_source_trace_closes_owned_renderer_once(attribute):
    from tests.helpers.reference.ogbench_pixels import _close_renderer

    closed = []
    owner = SimpleNamespace(**{attribute: SimpleNamespace(close=lambda: closed.append(True))})
    _close_renderer(owner, attribute)
    _close_renderer(owner, attribute)
    assert closed == [True] and getattr(owner, attribute) is None


def history_state(width):
    actions = torch.zeros(2, width, 2)
    actions[:, 0] = 0.25
    dtypes = torch.zeros(2, width, dtype=torch.int64)
    dtypes[:, 0] = 1
    return TensorDict(
        {
            "history": TensorDict(
                dict(
                    actions=actions,
                    dtypes=dtypes,
                    valid=torch.arange(width)[None].expand(2, -1) < 1,
                    lengths=torch.ones(2, 1, dtype=torch.int64),
                ),
                [2],
            ),
            "position": torch.zeros(2, 2),
        },
        [2],
    )


def test_push_history_allows_only_zero_padding():
    compare_padded_states(history_state(2), history_state(5))


@pytest.mark.parametrize("corruption", ["tail", "length", "prefix", "content", "dtype", "position"])
def test_push_history_rejects_corruption(corruption):
    expected, actual = history_state(2), history_state(5)
    if corruption == "tail":
        actual["history", "actions"][0, 4] = 1
    elif corruption == "length":
        actual["history", "lengths"][0] = 2
        actual["history", "valid"][0, 1] = True
    elif corruption == "prefix":
        actual["history", "valid"][0, :2] = torch.tensor([False, True])
    elif corruption == "content":
        actual["history", "actions"][0, 0] = 2
    elif corruption == "dtype":
        actual["history", "dtypes"][0, 0] = 2
    else:
        actual["position"][0] = 1
    with pytest.raises(AssertionError):
        compare_padded_states(actual, expected)


@pytest.fixture
def fake_fetch(monkeypatch):
    """A renderer with a visible controller mutation and delayed physics effects."""
    sources = []

    class Source:
        def __init__(self, **kwargs):
            self.spec = SimpleNamespace(max_episode_steps=50)
            self.unwrapped = self
            self.mujoco_renderer = SimpleNamespace(_viewers={})
            self.renders = 0
            self.closed = False
            sources.append(self)

        def reset(self, seed, options):
            self.position = float(seed)
            self.controller = 0.0
            self.elapsed = 0

        def step(self, action):
            self.position += float(action.sum()) + self.controller
            self.elapsed += 1
            return np.asarray([self.position]), self.position, False, self.elapsed >= 50, {"is_success": np.float64(0)}

        def close(self):
            self.closed = True

    def snapshot(source):
        return {key: np.asarray(getattr(source, key)) for key in ("position", "controller", "elapsed")}

    def image(source, device):
        source.renders += 1
        source.controller = 3.0
        source.mujoco_renderer._viewers["rgb_array"] = True
        return np.zeros((64, 64, 3), np.uint8), {"vendor": "fake"}

    monkeypatch.setattr(fetch_pixels, "load_source", lambda *args: None)
    original_import = fetch_pixels.importlib.import_module
    monkeypatch.setattr(
        fetch_pixels.importlib,
        "import_module",
        lambda name, *args: (
            None if name == "stable_worldmodel.envs.gymnasium_robotics" else original_import(name, *args)
        ),
    )
    monkeypatch.setattr(fetch_pixels.gym, "make", lambda *args, **kwargs: Source(**kwargs))
    monkeypatch.setattr(fetch_pixels, "snapshot", snapshot)
    monkeypatch.setattr(fetch_pixels, "_image", image)
    monkeypatch.setattr(
        fetch_pixels,
        "source_state",
        lambda rows: TensorDict(
            {key: torch.from_numpy(np.stack([row[key] for row in rows])) for key in rows[0]}, [len(rows)]
        ),
    )
    return sources


@pytest.mark.parametrize("scenario", ["reached", "horizon"])
def test_disposable_fetch_branches_keep_physics_independent(fake_fetch, scenario):
    case = next(case for case in PROFILES["fetch"] if case.domain == "PickAndPlace")
    trace, identities = fetch_pixels._pickandplace_trace(case, 17, "float64", 8, scenario, "cpu", "unused")
    physics, branches = fake_fetch[:2], fake_fetch[2:]
    assert all(source.renders == 0 and source.controller == 0 for source in physics)
    assert len(branches) == 16 and all(source.renders == 1 and source.closed for source in branches)
    assert all(
        row["differences"] == {"/controller": {"max_abs": 3.0, "shape": [], "dtype": "float64"}}
        for lane in identities
        for row in lane["mutations"]
    )
    assert all(bool((state["controller"] == 0).all()) for state in trace.states)
    if scenario == "horizon":
        torch.testing.assert_close(trace.valid, torch.arange(8)[None] < torch.tensor([[1], [3]]))


def test_continuous_fetch_render_remains_rejected(fake_fetch):
    case = next(case for case in PROFILES["fetch"] if case.domain == "PickAndPlace")
    witness = fetch_pixels._pickandplace_mutation(case, 17, "float64", "cpu", "unused")
    assert witness["mutation"] and witness["continuation"]
    plain = fake_fetch[0]
    with pytest.raises(AssertionError, match="changed continuation"):
        fetch_pixels._render(plain, "cpu")


def test_fetch_mutation_witness_requires_an_observed_mutation(fake_fetch, monkeypatch):
    case = next(case for case in PROFILES["fetch"] if case.domain == "PickAndPlace")
    monkeypatch.setattr(fetch_pixels, "_image", lambda source, device: (np.zeros((64, 64, 3), np.uint8), {}))
    with pytest.raises(AssertionError, match="was not witnessed"):
        fetch_pixels._pickandplace_mutation(case, 17, "float64", "cpu", "unused")
