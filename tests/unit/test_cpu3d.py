"""CPU device honesty, batching and spatial-bound invariants for trackcom rendering.

Source-image acceptance lives in contracts/test_dmc_source_pixels.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from stable_envs._core.mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

ROOT = Path(__file__).resolve().parents[2]
CELLS = ("DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk", "DMCHumanoidWalk")

pytest.importorskip("dm_control")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def se():
    import stable_envs as se

    assert all(c in se.registry.REGISTRY for c in CELLS), "Missing DM Control trackcom registration"
    return se


def _states(cell: str, n: int) -> np.ndarray:
    if cell == "DMCHumanoidWalk":
        path = ROOT / "tests/fixtures/semantics/dmc_humanoid/humanoid_walk_swm_176d545_mujoco312.npz"
        return np.load(path)["state/qpos"].reshape(-1, 28)[:n]
    return np.load(ROOT / f"tests/fixtures/states/fidelity/{cell}.npz")["states"][:n]


@pytest.mark.gl
@pytest.mark.parametrize("cell", CELLS)
def test_cpu_device_builds_the_cpu_backend_and_no_mjwarp_context(se, cell):
    env = se.make(cell, num_envs=2, device="cpu")
    assert env.device.type == "cpu"
    assert int(env.size) in env._cpu_scenes  # the CPU scene was built at init
    from stable_envs.envs import dmcontrol as D

    assert not any(k[0] == env.DOMAIN for k in D._TC_CACHE)  # and no MJWarp context was
    f = env._render(_states(cell, 2))
    assert f.shape == (2, env.size, env.size, 3) and f.dtype == torch.uint8 and f.device.type == "cpu"


@pytest.mark.parametrize("device", ["mps", "meta"])
def test_unknown_device_raises_rather_than_falling_back(se, device):
    """A device with no 3-D backend is refused by name instead of falling back."""
    cls = se.registry.REGISTRY["DMCHopperHop"]
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device on this host")
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)) as e:
        cls(num_envs=1, device=device)
    msg = str(e.value)
    assert "no 3-D backend" in msg or device in msg


def test_swimmer_builds_the_cpu_backend(se):
    """Swimmer renders on CPU: its frames meet the source-pixel bar against dm_control."""
    try:
        env = se.make("DMCSwimmerSwimmer6", num_envs=2, device="cpu")
    except RuntimeError as e:
        if "OpenGL" not in str(e):
            raise
        pytest.skip(f"the CPU scene bakes its sky with dm_control's renderer: {e}")
    assert int(env.size) in env._cpu_scenes
    f = env._render(env.sample_states(2, np.random.default_rng(0)))
    assert f.shape == (2, env.size, env.size, 3) and f.dtype == torch.uint8 and f.device.type == "cpu"


def test_trackcom_material_knobs_are_geom_wearers_only(se):
    """Neither trackcom backend draws a site, so a material worn only by sites is not a knob.

    On the certified cells (sites hidden) the list equals the base class's; on quadruped it drops
    ``eye``, worn by the two pupil sites alone."""
    from stable_envs.envs import dmcontrol as D

    for name in CELLS:
        cls = se.registry.REGISTRY[name]
        assert cls._used_materials() == D._DMCModel._used_materials.__func__(cls), name
    if "DMCQuadrupedWalk" in se.registry.REGISTRY:
        cls = se.registry.REGISTRY["DMCQuadrupedWalk"]
        base = D._DMCModel._used_materials.__func__(cls)
        ours = cls._used_materials()
        assert "eye" in base and "eye" not in ours and ours == [m for m in base if m != "eye"]
        assert cls.CAMERA == 2 and cls.default_size == 224  # the registered default


def test_the_transcribed_quadruped_mixin_overrides_only_the_loader_hook_and_the_size(se):
    """``_TranscribedQuadrupedModel`` swaps the model source (``_load_env``, ``default_size=224``) and nothing else."""
    from stable_envs.envs import dmcontrol as D

    own = {k for k in vars(D._TranscribedQuadrupedModel) if not k.startswith("__")}
    assert own == {"default_size", "_load_env"}, own
    cls = se.registry.REGISTRY["DMCQuadrupedWalk"]
    assert cls._bundle.__func__ is D._DMCModel._bundle.__func__
    assert cls.reference_env.__func__ is D._DMCModel.reference_env.__func__
    assert cls.sample_states is D._DMCModel.sample_states
    assert cls._load_env.__func__ is not D._DMCModel._load_env.__func__
    assert cls.default_size == 224 and D._DMCModel.default_size == 84


@pytest.mark.gl
def test_no_gpu_runtime_is_loaded_by_a_cpu_render(se):
    env = se.make("DMCCheetahRun", num_envs=2, device="cpu")
    env._render(_states("DMCCheetahRun", 2))
    assert "warp" not in sys.modules and "mujoco_warp" not in sys.modules


@pytest.mark.gl
@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("size", [42, 84, pytest.param(168, marks=pytest.mark.nightly)])
def test_window_and_footprint_bounds_are_exact(se, cell, size):
    """The mover window and the shadow footprint are conservative BOUNDS: casting every ray and
    shadow-testing every lit hit must give the identical frame, byte for byte, at every size."""
    from stable_envs.render import cpu3d

    st = _states(cell, 6)
    env = se.make(cell, num_envs=6, device="cpu")
    ref = env._render(st, size=size).numpy()
    try:
        cpu3d.FORCE_FULL_FRAME = True
        full = env._render(st, size=size).numpy()
        cpu3d.FORCE_FULL_FRAME = False
        cpu3d.SHADOW_TEST_ALL_HITS = True
        allhits = env._render(st, size=size).numpy()
    finally:
        cpu3d.FORCE_FULL_FRAME = False
        cpu3d.SHADOW_TEST_ALL_HITS = False
    assert np.array_equal(full, ref), "window cast differs from the full-frame cast"
    assert np.array_equal(allhits, ref), "footprint-bounded shadows differ from testing every hit"


@pytest.mark.nightly
@pytest.mark.gl
@pytest.mark.parametrize("cell", CELLS)
def test_batch_chunk_and_order_invariance(se, cell):
    from stable_envs.render import cpu3d

    n = cpu3d.CPU_CHUNK + 3  # straddles a chunk boundary
    st = _states(cell, n)
    env = se.make(cell, num_envs=n, device="cpu")
    a = env._render(st).numpy()
    b = env._render(st).numpy()
    assert np.array_equal(a, b), "not deterministic"
    for i in (0, cpu3d.CPU_CHUNK - 1, cpu3d.CPU_CHUNK, n - 1):
        one = env._render(st[i : i + 1]).numpy()[0]
        assert np.array_equal(one, a[i]), f"frame {i} depends on its batch"
    perm = np.random.default_rng(0).permutation(n)
    c = env._render(st[perm]).numpy()
    assert np.array_equal(c, a[perm]), "frame depends on batch order"
    # the same three properties with a DIFFERENT appearance per row (materials and light per env)
    sty = env.sample_style(n, rng=np.random.default_rng(1))
    d1 = env._render(st, style=sty).numpy()
    assert np.array_equal(d1, env._render(st, style=sty).numpy())
    for i in (0, cpu3d.CPU_CHUNK, n - 1):
        assert np.array_equal(env._render(st[i : i + 1], style=sty[i : i + 1]).numpy()[0], d1[i]), (
            f"frame {i} with a per-row style depends on its batch"
        )
    assert np.array_equal(env._render(st[perm], style=sty[perm]).numpy(), d1[perm])
    assert not np.array_equal(d1, a), "the per-row styles should have changed the picture"


@pytest.mark.gl
def test_render_at_another_resolution_is_a_pure_function_of_size(se):
    env = se.make("DMCCheetahRun", num_envs=2, device="cpu")
    st = _states("DMCCheetahRun", 2)
    big = env._render(st, size=168)
    assert big.shape == (2, 168, 168, 3)
    assert np.array_equal(big.numpy(), env._render(st, size=168).numpy())
