"""The Numba classic-control renderer draws the Torch port's and the Gymnasium source's frames byte for byte."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numba")

import stable_envs as se

pytestmark = pytest.mark.family_classic

CLASSIC = ("Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum")
PI = np.pi


def _edge(name):
    if name == "CartPole":
        x = [0.0, 2.4, -2.4, 3.5, -5.0, 9.0, -9.0]
        t = [0.0, 0.2095, -0.2095, PI / 2, -PI / 2, PI, 7 * PI, -24 * PI]
        return np.array([(a, b) for a in x for b in t], np.float32)
    if name == "Pendulum":
        th = [0.0, PI, -PI, PI / 2, 24 * PI, -24 * PI]
        u = [-2.0, 2.0, 0.0, 1e-3, -1e-3, 1.99, -1.99, 0.5]
        return np.array([(a, b) for a in th for b in u], np.float32)
    if name == "Acrobot":
        a = [0.0, PI / 2, PI, -PI, 24 * PI, -PI / 2]
        return np.array([(p, q) for p in a for q in a], np.float32)
    return np.array([[v] for v in (-1.2, 0.6, 0.45, 0.5, 0.0, -6.0, 5.0, -0.5, 0.3, 1e-4)], np.float32)


def _extreme(name, n, rng):
    if name == "CartPole":
        return np.stack([rng.uniform(-9.0, 9.0, n), rng.uniform(-24 * PI, 24 * PI, n)], 1).astype(np.float32)
    if name == "Pendulum":
        return np.stack([rng.uniform(-24 * PI, 24 * PI, n), rng.uniform(-2.0, 2.0, n)], 1).astype(np.float32)
    if name == "Acrobot":
        return rng.uniform(-24 * PI, 24 * PI, (n, 2)).astype(np.float32)
    return rng.uniform(-6.0, 5.0, (n, 1)).astype(np.float32)


def _rows(env, name, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([env.sample_states(64, rng), _edge(name), _extreme(name, 16, rng)])


GEOMETRY = {
    "CartPole": {"physics.length": (0.25, 0.75)},
    "Acrobot": {"physics.link_length_1": (0.5, 1.45), "physics.link_length_2": (0.5, 1.45)},
}


def _variation(env, n, seed):
    """Per-lane colours from a small palette, random anti-aliasing and varied geometry where the class has any."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, (12, 3))
    out = {}
    for path in env.variation_space.names():
        if path.endswith("color"):
            out[path] = torch.as_tensor(palette[rng.integers(0, 12, n)], dtype=torch.float32)
        elif path == "rendering.antialias":
            out[path] = torch.as_tensor(rng.integers(0, 2, (n, 1)), dtype=torch.float32)
    for path, (lo, hi) in GEOMETRY.get(type(env).__name__, {}).items():
        out[path] = torch.as_tensor(rng.uniform(lo, hi, (n, 1)), dtype=torch.float32)
    return out


def _make(name, n):
    if name not in se.registry.REGISTRY:
        pytest.skip(f"{name} is not registered in this interpreter")
    return se.make(name, num_envs=n, device="cpu")


def _torch_port(name, states, var):
    """Draw with the per-primitive Torch port, in chunks under the render-worker threshold (in this process)."""
    env = _make(name, 1)
    env.use_fused_render = False
    parts = []
    for i in range(0, len(states), 8):
        part = None if var is None else {k: v[i : i + 8] for k, v in var.items()}
        parts.append(env._render(states[i : i + 8], variation=part, backend="port").clone())
    return torch.cat(parts)


def _differs(a, b):
    """Return a (B,) bool tensor: whether each frame differs."""
    return (a.to(torch.int16) - b.to(torch.int16)).abs().flatten(1).amax(1).gt(0)


@pytest.mark.parametrize("varied", (False, True), ids=("default", "varied"))
@pytest.mark.parametrize("name", CLASSIC)
def test_the_numba_frames_equal_the_torch_port(name, varied):
    env = _make(name, 1)
    states = torch.as_tensor(_rows(env, name, 7))
    n = len(states)
    var = _variation(env, n, 11) if varied else None
    fast = env._render(states, variation=var, backend="port").clone()
    slow = _torch_port(name, states, var)
    assert fast.shape == (n, *env.native, 3)
    bad = int(_differs(fast, slow).sum())
    assert bad == 0, f"{name} varied={varied}: {bad} of {n} frames differ from the Torch port"
    assert len({f.numpy().tobytes() for f in fast}) > 8


@pytest.mark.parametrize("name", CLASSIC)
def test_the_numba_frames_equal_the_source_in_the_default_appearance(name):
    """Wherever the Torch port draws the source's frame, so does the Numba port.

    Float32 trigonometry at exact angle multiples moves a few port frames by a pixel; those rows stay in the
    Torch-port comparison above and drop out here. Every sampled row must remain.
    """
    pytest.importorskip("pygame")
    pytest.importorskip("gymnasium")
    env = _make(name, 1)
    rows = _rows(env, name, 13)
    states = torch.as_tensor(rows)
    fast = env._render(states, backend="port").clone()
    # Chunks under the render-worker threshold draw in this process.
    source = torch.cat([env._render(states[i : i + 8], backend="source").clone() for i in range(0, len(states), 8)])
    exact = ~_differs(_torch_port(name, states, None), source)
    assert int(exact[:64].sum()) == 64, f"{name}: the Torch port misses the source on a sampled row"
    bad = int(_differs(fast, source)[exact].sum())
    assert bad == 0, f"{name}: {bad} of {int(exact.sum())} frames differ from the source"


def test_the_output_buffer_is_filled_and_returned():
    env = _make("CartPole", 1)
    states = torch.as_tensor(env.sample_states(6, np.random.default_rng(3)))
    want = env._render(states, backend="port").clone()
    buf = torch.zeros(6, 400, 600, 3, dtype=torch.uint8)
    assert env._render(states, backend="port", out=buf).data_ptr() == buf.data_ptr()
    assert torch.equal(buf, want)


@pytest.mark.parametrize("name", CLASSIC)
def test_auto_draws_the_same_frames_for_small_and_large_batches(name):
    env = _make(name, 1)
    states = torch.as_tensor(env.sample_states(16, np.random.default_rng(5)))
    auto = torch.cat([env._render(states[i : i + 2]).clone() for i in range(0, 16, 2)])
    assert torch.equal(auto, env._render(states, backend="port"))
