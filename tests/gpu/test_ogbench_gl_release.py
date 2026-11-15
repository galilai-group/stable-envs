"""Every OpenGL context an OGBench env creates is released by ``env.close()``, and none is left at exit."""

import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.regression]

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

SCRIPT = textwrap.dedent(
    """
    import gc

    import numpy as np

    import stable_envs as se
    from stable_envs.envs.ogbench import OGBCube, OGBScene
    from tests._runner import resource_guard as guard
    from tests.helpers import ogbench_golden as golden

    guard.install()
    before = guard.take()
    for family, backend in CASES:
        if family == "maze":
            env = se.make(backend, num_envs=2, device="cuda")
            env._render(env.sample_states(2, rng=np.random.default_rng(0)))
        else:
            state, _ = golden.load(golden.FIXTURES / f"{family}-{backend}.npz")
            env = (OGBCube if family == "Cube" else OGBScene)(device="cuda", size=64, seed=17)
            env.render(state[:2], backend=backend)
        env.close()
    gc.collect()
    d = guard.delta(before)
    print("COUNTS", CASES, d["context_created"], d["context_freed"], d["context_live"])
    assert d["context_created"] > 0, "no context was created, so nothing was checked"
    assert d["context_live"] == 0, f"{d['context_live']} contexts live after close"
    """
)

CASES = {
    "maze": (("maze", "PointMazeMediumNavigate"),),
    "cube": (("Cube", "mjwarp"), ("Cube", "mjwarp_fidelity")),
    "scene": (("Scene", "mjwarp"),),
    "mixed": (("Cube", "mjwarp"), ("maze", "PointMazeMediumNavigate"), ("Scene", "mjwarp")),
}


@pytest.mark.parametrize("cases", CASES.values(), ids=CASES.keys())
def test_closing_the_env_releases_its_contexts_and_exit_is_clean(cases):
    code = f"CASES = {cases!r}\n" + SCRIPT
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out
    assert "EGLError" not in out, out
    assert "Exception ignored" not in out, out


POOL_SCRIPT = textwrap.dedent(
    """
    import multiprocessing as mp

    import stable_envs as se
    from stable_envs.envs.ogbench import manipulation_opengl_pool as pool

    env = se.make("OGBCube", num_envs=16, device="cuda", seed=17, backend="opengl")
    env.reset(seed=3)
    with se.tuning.override(ogb_gl_workers=2):
        if pool.workers(16, "cuda") == 0:
            print("SKIP fewer than 2 CPUs")
            raise SystemExit(0)
        env.render(env.to_render_state(), backend="opengl")
    started = [p for p in mp.active_children() if p.name == "ogb-opengl"]
    assert len(started) == 2, started
    env.close()
    assert not pool._POOLS, pool._POOLS
    assert not [p for p in started if p.is_alive()], "OpenGL workers outlive the last env"
    """
)


def test_closing_the_last_env_stops_the_opengl_workers():
    p = subprocess.run([sys.executable, "-c", POOL_SCRIPT], capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out
    if "SKIP" in p.stdout:
        pytest.skip(p.stdout.strip())
    assert "EGLError" not in out, out
    assert "Exception ignored" not in out, out
