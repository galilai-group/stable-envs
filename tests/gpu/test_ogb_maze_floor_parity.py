"""Both pixel-scene floor encodings of the OGBench mazes agree with MuJoCo's own frames.

``reference_env`` shows the XML checker, which is the default ``floor.pixel_encoding=0``. The
position-encoding gradient (encoding 1) is compared here against the unchanged ogbench source.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.regression]

if not torch.cuda.is_available():
    pytest.skip("OGBench maze parity requires a CUDA device", allow_module_level=True)

pytest.importorskip("ogbench")

import stable_envs as se
from stable_envs.envs.ogbench import maze as M
from stable_envs.testing import conformance as C
from tests.helpers import source_mse as SM

GRADIENT_MSE = SM.PATH.parent / "ogb_maze_gradient_mse.json"

CELLS = ("PointMazeMediumNavigate", "AntMazeMediumNavigate", "HumanoidMazeMediumNavigate")
SEED = 20260811

#: Gradient-floor source envs live for the process, as conformance.reference_env does for its own.
_GRADIENT_REFS: dict = {}


def _gradient_reference(cls, size):
    key = (cls.LOCO, cls.MAZE, int(size))
    if key not in _GRADIENT_REFS:
        _GRADIENT_REFS[key] = M._build_reference(cls.LOCO, cls.MAZE, int(size), "pixels")
    return _GRADIENT_REFS[key]


def _mse(ours, official) -> dict:
    return C.mse_summary(C.frame_mse(ours, official))


@pytest.mark.parametrize("name", CELLS)
def test_reference_floor_is_the_xml_checker(name):
    import importlib

    import mujoco

    cls = se.registry.REGISTRY[name]
    env = C.reference_env(cls, cls.default_size)
    m = env.unwrapped.model
    module = importlib.import_module(f"ogbench.locomaze.{cls.LOCO}")
    xml = mujoco.MjModel.from_xml_path(getattr(module, f"{cls.LOCO.capitalize()}Env").xml_file)
    attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
    src, dst = xml.tex("grid"), m.tex("grid")
    n = 3 * int(dst.height[0]) * int(dst.width[0])
    got = np.asarray(getattr(m, attr)[int(dst.adr[0]) : int(dst.adr[0]) + n])
    want = np.asarray(getattr(xml, attr)[int(src.adr[0]) : int(src.adr[0]) + n])
    assert np.array_equal(got, want)


@pytest.mark.parametrize("name", CELLS)
def test_both_floor_encodings_match_the_source(name):
    """The checker floor stays under the class's recorded worst-frame MSE, and the gradient floor stays in
    the band of its own record in ``ogb_maze_gradient_mse.json``."""
    rec = SM.record(name)
    if rec is None:
        pytest.skip(f"no MSE record for {name}; run {SM.RECORD}")
    env = se.make(name, num_envs=16, device="cuda")
    cls = type(env)
    states = C.sample_render_rows(env, 32, rng=np.random.default_rng(SEED))

    checker = cls.reference_render(C.reference_env(cls, env.size), states)
    on_checker = _mse(env._render(states).cpu().numpy(), checker)
    assert on_checker["mse_max"] <= SM.ceiling(rec["mse_max"]), f"{name} checker floor: {on_checker}, record {rec}"

    gradient = cls.reference_render(_gradient_reference(cls, env.size), states)
    ours = env._render(states, variation={"floor.pixel_encoding": [1]}).cpu().numpy()
    gradient_rec = json.loads(GRADIENT_MSE.read_text())
    if gradient_rec["gpu"] != torch.cuda.get_device_name():
        pytest.skip(f"the gradient-floor record was measured on {gradient_rec['gpu']}")
    bad = SM.problems(
        f"{name} gradient floor",
        _mse(ours, gradient),
        gradient_rec["classes"][name],
        rerecord=f"write the new values to {GRADIENT_MSE.name}",
    )
    assert not bad, "\n".join(bad)
    assert not np.array_equal(checker, gradient)
