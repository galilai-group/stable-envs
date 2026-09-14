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


def test_unknown_device_raises_rather_than_falling_back(se):
    cls = se.REGISTRY["DMCHopperHop"]
    with pytest.raises((ValueError, RuntimeError)) as e:
        cls(batch=1, device="meta")
    # either our own message or torch refusing the device before we get there -- never a CUDA render
    assert "no 3-D backend" in str(e.value) or "meta" in str(e.value)


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
