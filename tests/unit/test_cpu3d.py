"""The CPU backend for the MJWarp-only DM Control cells: honest device, invariances, and MJWarp's picture.

CPU only, no GL for rendering (the trackcom cells never touch ``mujoco.Renderer`` on the CPU path;
``_bundle`` still constructs one, so ``MUJOCO_GL`` must name a backend the host has --
``configure_mujoco_gl`` picks that before ``mujoco`` or ``dm_control`` is imported. Do not default
to GLFW on macOS: ``from dm_control import mujoco`` hangs inside ``dm_control._render``).
The oracle for the picture is a committed slice of OUR MJWarp frames at the first four frozen Step-4
states of each cell (``tests/fixtures/regression/join/step6/cpu-3d/replay/mjwarp_frames_first4.npz``, from the
texture-fidelity workstream's stacks rendered in ``se-step5`` on an RTX 3090); the claim tested is
the one the design makes -- the same picture to within a level -- not bit-identity, which a different
renderer cannot promise and MJWarp does not keep against itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from stable_envs._mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/regression/join/step6/cpu-3d/replay/mjwarp_frames_first4.npz"
CELLS = ("DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk", "DMCHumanoidStand")

pytest.importorskip("dm_control")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def se():
    import stable_envs as se

    if not all(c in se.registry.REGISTRY for c in CELLS):
        pytest.skip("DM Control trackcom cells are not registered in this interpreter")
    return se


def _states(cell: str, n: int) -> np.ndarray:
    return np.load(ROOT / f"tests/fixtures/regression/join/step4/states/{cell}.npz")["states"][:n]


def test_cpu3d_imports_no_gpu_runtime():
    """The CPU backend has no GPU runtime behind it: no warp, no mujoco_warp, no torch.cuda call."""
    import re

    src = (ROOT / "stable_envs/render/cpu3d.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"^\s*(import|from)\s+(warp|mujoco_warp)\b", code, re.M)
    assert "torch.cuda" not in code


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
    """No silent fallback: a device with no 3-D backend is refused by name, whether torch can host
    tensors on it (mps on this Mac, meta anywhere) or not."""
    cls = se.registry.REGISTRY["DMCHopperHop"]
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device on this host")
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)) as e:
        cls(num_envs=1, device=device)
    msg = str(e.value)
    assert "no 3-D backend" in msg or device in msg


def test_swimmer_is_refused_on_cpu_by_the_allow_list(se):
    """DMCSwimmerSwimmer6 used to be refused on device='cpu' because its bodies are mjGEOM_ELLIPSOID
    and no CPU intersector covered them. Step 7 added the intersector (for the quadruped torso), so
    the geom-type refusal is gone -- and swimmer, which is not an SWM id and has no CPU certification,
    is now refused by an explicit allow-list (``CPU_UNCERTIFIED``) naming the env and the reason,
    rather than drawn uncertified. Certification decides whether it is listed and the entry removed."""
    if "DMCSwimmerSwimmer6" not in se.registry.REGISTRY:
        pytest.skip("swimmer not registered")
    with pytest.raises(ValueError) as e:
        se.registry.REGISTRY["DMCSwimmerSwimmer6"](num_envs=1, device="cpu")
    msg = str(e.value)
    assert "DMCSwimmerSwimmer6" in msg and "not certified" in msg and "CPU" in msg


def test_trackcom_material_knobs_are_geom_wearers_only(se):
    """Neither trackcom backend draws a site, so a material worn only by sites is not a knob here.
    ``DMControlTrackCom._used_materials`` restricts the base census to geoms: on the four certified
    cells (sites hidden) the list is IDENTICAL to the base class's -- their declared variation space
    does not move -- and on quadruped it drops ``eye``, worn by the two pupil sites alone."""
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
    """``_TranscribedQuadrupedModel`` is the non-suite.load quadruped MJCF (``default_size=224``).

    ``_DMCModel._load_env`` is the one place a cell's model comes from, and this mixin overrides it
    and ``default_size`` and NOTHING else -- so ``_bundle``/``sample_states``/``reference_env`` are
    the base class's and cannot drift from it (INTEG-1: a copied ``_bundle`` drifted the moment W3
    changed the base one). Every loader in the base goes through the hook."""
    import inspect

    from stable_envs.envs import dmcontrol as D

    own = {k for k in vars(D._TranscribedQuadrupedModel) if not k.startswith("__")}
    assert own == {"default_size", "_load_env"}, own
    cls = se.registry.REGISTRY["DMCQuadrupedWalk"]
    assert cls._bundle.__func__ is D._DMCModel._bundle.__func__
    assert cls.reference_env.__func__ is D._DMCModel.reference_env.__func__
    assert cls.sample_states is D._DMCModel.sample_states
    for name in ("_bundle", "sample_states", "reference_env"):
        src = inspect.getsource(getattr(D._DMCModel, name))
        assert "_load_env(" in src and "_suite().load(" not in src, name
    assert cls._load_env.__func__ is not D._DMCModel._load_env.__func__
    assert cls.default_size == 224 and D._DMCModel.default_size == 84


def test_no_gpu_runtime_is_loaded_by_a_cpu_render(se):
    env = se.make("DMCCheetahRun", num_envs=2, device="cpu")
    env._render(_states("DMCCheetahRun", 2))
    assert "warp" not in sys.modules and "mujoco_warp" not in sys.modules


@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("size", [42, 84, 168])
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


@pytest.mark.parametrize("cell", CELLS)
def test_matches_mjwarp_to_within_a_level(se, cell):
    z = np.load(FIXTURE)
    ref = z[f"{cell}_mjwarp"]
    env = se.make(cell, num_envs=len(ref), device="cpu")
    ours = env._render(_states(cell, len(ref))).numpy()
    d = np.abs(ours.astype(np.int16) - ref.astype(np.int16)).max(-1)
    within1 = float((d <= 1).mean())
    mean = float(d.mean())
    # the design's claim: the same picture to within a level; measured on the full 64-state corpus at
    # exact 0.982-0.997, within-1 >= 0.9994, mean 0.004-0.018 (tests/fixtures/regression/join/step6/cpu-3d/m1_local)
    assert within1 >= 0.998, (cell, within1, mean)
    assert mean <= 0.06, (cell, within1, mean)
    assert float((d == 0).mean()) >= 0.95, (cell, float((d == 0).mean()))


def test_render_at_another_resolution_is_a_pure_function_of_size(se):
    env = se.make("DMCCheetahRun", num_envs=2, device="cpu")
    st = _states("DMCCheetahRun", 2)
    big = env._render(st, size=168)
    assert big.shape == (2, 168, 168, 3)
    assert np.array_equal(big.numpy(), env._render(st, size=168).numpy())
