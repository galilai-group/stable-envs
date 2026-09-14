"""The CPU backend for the MJWarp-only DM Control cells: honest device, invariances, and MJWarp's picture.

CPU only, no GL for rendering (the trackcom cells never touch ``mujoco.Renderer`` on the CPU path;
``_bundle`` still constructs one, so ``MUJOCO_GL`` must name a backend the host has -- the conftest
does not set it, so this module picks ``glfw``/``egl`` per platform before ``mujoco`` is imported).
The oracle for the picture is a committed slice of OUR MJWarp frames at the first four frozen Step-4
states of each cell (``examples/results/step6/cpu-3d/replay/mjwarp_frames_first4.npz``, from the
texture-fidelity workstream's stacks rendered in ``se-step5`` on an RTX 3090); the claim tested is
the one the design makes -- the same picture to within a level -- not bit-identity, which a different
renderer cannot promise and MJWarp does not keep against itself.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "examples/results/step6/cpu-3d/replay/mjwarp_frames_first4.npz"
CELLS = ("DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk", "DMCHumanoidStand")

pytest.importorskip("dm_control")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def se():
    import stable_envs as se
    if not all(c in se.REGISTRY for c in CELLS):
        pytest.skip("DM Control trackcom cells are not registered in this interpreter")
    return se


def _states(cell: str, n: int) -> np.ndarray:
    return np.load(ROOT / f"examples/results/step4/states/{cell}.npz")["states"][:n]


def test_cpu3d_imports_no_gpu_runtime():
    """The CPU backend has no GPU runtime behind it: no warp, no mujoco_warp, no torch.cuda call."""
    import re
    src = (ROOT / "stable_envs/render/cpu3d.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"^\s*(import|from)\s+(warp|mujoco_warp)\b", code, re.M)
    assert "torch.cuda" not in code


@pytest.mark.parametrize("cell", CELLS)
def test_cpu_device_builds_the_cpu_backend_and_no_mjwarp_context(se, cell):
    env = se.make(cell, batch=2, device="cpu")
    assert env.device.type == "cpu"
    assert int(env.size) in env._cpu_scenes                    # the CPU scene was built at init
    from stable_envs.envs import dmcontrol as D
    assert not any(k[0] == env.DOMAIN for k in D._TC_CACHE)     # and no MJWarp context was
    f = env.render(_states(cell, 2))
    assert f.shape == (2, env.size, env.size, 3) and f.dtype == torch.uint8 and f.device.type == "cpu"


@pytest.mark.parametrize("device", ["mps", "meta"])
def test_unknown_device_raises_rather_than_falling_back(se, device):
    """No silent fallback: a device with no 3-D backend is refused by name, whether torch can host
    tensors on it (mps on this Mac, meta anywhere) or not."""
    cls = se.REGISTRY["DMCHopperHop"]
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device on this host")
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)) as e:
        cls(batch=1, device=device)
    msg = str(e.value)
    assert "no 3-D backend" in msg or device in msg


def test_swimmer_is_refused_on_cpu_by_name_and_geom_type(se):
    """DMCSwimmerSwimmer6's bodies are mjGEOM_ELLIPSOID, which no CPU intersector covers. It used to
    render on the GPU behind a device='cpu' request; now it raises, naming the env and the geom."""
    if "DMCSwimmerSwimmer6" not in se.REGISTRY:
        pytest.skip("swimmer not registered")
    with pytest.raises(ValueError) as e:
        se.REGISTRY["DMCSwimmerSwimmer6"](batch=1, device="cpu")
    msg = str(e.value)
    assert "DMCSwimmerSwimmer6" in msg and "ELLIPSOID" in msg and "no CPU 3-D backend" in msg


def test_no_gpu_runtime_is_loaded_by_a_cpu_render(se):
    env = se.make("DMCCheetahRun", batch=2, device="cpu")
    env.render(_states("DMCCheetahRun", 2))
    assert "warp" not in sys.modules and "mujoco_warp" not in sys.modules


@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("size", [42, 84, 168])
def test_window_and_footprint_bounds_are_exact(se, cell, size):
    """The mover window and the shadow footprint are conservative BOUNDS: casting every ray and
    shadow-testing every lit hit must give the identical frame, byte for byte, at every size."""
    from stable_envs.render import cpu3d
    st = _states(cell, 6)
    env = se.make(cell, batch=6, device="cpu")
    ref = env.render(st, size=size).numpy()
    try:
        cpu3d.FORCE_FULL_FRAME = True
        full = env.render(st, size=size).numpy()
        cpu3d.FORCE_FULL_FRAME = False
        cpu3d.SHADOW_TEST_ALL_HITS = True
        allhits = env.render(st, size=size).numpy()
    finally:
        cpu3d.FORCE_FULL_FRAME = False
        cpu3d.SHADOW_TEST_ALL_HITS = False
    assert np.array_equal(full, ref), "window cast differs from the full-frame cast"
    assert np.array_equal(allhits, ref), "footprint-bounded shadows differ from testing every hit"


@pytest.mark.parametrize("cell", CELLS)
def test_batch_chunk_and_order_invariance(se, cell):
    from stable_envs.render import cpu3d
    n = cpu3d.CPU_CHUNK + 3                                     # straddles a chunk boundary
    st = _states(cell, n)
    env = se.make(cell, batch=n, device="cpu")
    a = env.render(st).numpy()
    b = env.render(st).numpy()
    assert np.array_equal(a, b), "not deterministic"
    for i in (0, cpu3d.CPU_CHUNK - 1, cpu3d.CPU_CHUNK, n - 1):
        one = env.render(st[i:i + 1]).numpy()[0]
        assert np.array_equal(one, a[i]), f"frame {i} depends on its batch"
    perm = np.random.default_rng(0).permutation(n)
    c = env.render(st[perm]).numpy()
    assert np.array_equal(c, a[perm]), "frame depends on batch order"
    # the same three properties with a DIFFERENT appearance per row (materials and light per env)
    sty = env.sample_style(n, rng=np.random.default_rng(1))
    d1 = env.render(st, style=sty).numpy()
    assert np.array_equal(d1, env.render(st, style=sty).numpy())
    for i in (0, cpu3d.CPU_CHUNK, n - 1):
        assert np.array_equal(env.render(st[i:i + 1], style=sty[i:i + 1]).numpy()[0], d1[i]), \
            f"frame {i} with a per-row style depends on its batch"
    assert np.array_equal(env.render(st[perm], style=sty[perm]).numpy(), d1[perm])
    assert not np.array_equal(d1, a), "the per-row styles should have changed the picture"


@pytest.mark.parametrize("cell", CELLS)
def test_matches_mjwarp_to_within_a_level(se, cell):
    z = np.load(FIXTURE)
    ref = z[f"{cell}_mjwarp"]
    env = se.make(cell, batch=len(ref), device="cpu")
    ours = env.render(_states(cell, len(ref))).numpy()
    d = np.abs(ours.astype(np.int16) - ref.astype(np.int16)).max(-1)
    within1 = float((d <= 1).mean())
    mean = float(d.mean())
    # the design's claim: the same picture to within a level; measured on the full 64-state corpus at
    # exact 0.982-0.997, within-1 >= 0.9994, mean 0.004-0.018 (examples/results/step6/cpu-3d/m1_local)
    assert within1 >= 0.998, (cell, within1, mean)
    assert mean <= 0.06, (cell, within1, mean)
    assert float((d == 0).mean()) >= 0.95, (cell, float((d == 0).mean()))


def test_render_at_another_resolution_is_a_pure_function_of_size(se):
    env = se.make("DMCCheetahRun", batch=2, device="cpu")
    st = _states("DMCCheetahRun", 2)
    big = env.render(st, size=168)
    assert big.shape == (2, 168, 168, 3)
    assert np.array_equal(big.numpy(), env.render(st, size=168).numpy())
