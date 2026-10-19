"""Both pixel-scene floor encodings of the OGBench mazes match MuJoCo's own frames.

``reference_env`` shows the XML checker, which is the default ``floor.pixel_encoding=0``. The
position-encoding gradient (encoding 1) is compared here against the unchanged ogbench source.
"""

from __future__ import annotations

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
from tests.helpers import pixel_tolerances as PT

CELLS = ("PointMazeMediumNavigate", "AntMazeMediumNavigate", "HumanoidMazeMediumNavigate")
SEED = 20260811

#: Gradient-floor source envs live for the process, as conformance.reference_env does for its own.
_GRADIENT_REFS: dict = {}


def _gradient_reference(cls, size):
    key = (cls.LOCO, cls.MAZE, int(size))
    if key not in _GRADIENT_REFS:
        _GRADIENT_REFS[key] = M._build_reference(cls.LOCO, cls.MAZE, int(size), "pixels")
    return _GRADIENT_REFS[key]


def _assert_exact(cls, ours, official, label):
    r = C.pixel_agreement(ours, official)
    floors = PT.exact_floors(cls.__name__)
    if r["interior_exact_frac"] is not None:
        assert r["interior_exact_frac"] >= floors["interior_exact_min"], f"{cls.__name__} {label}: {r}"
    assert r["exact_frac"] >= floors["all_exact_min"], f"{cls.__name__} {label}: {r}"
    assert r["mean_abs"] <= floors["mean_abs_max"], f"{cls.__name__} {label}: {r}"


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
    env = se.make(name, num_envs=16, device="cuda")
    cls = type(env)
    states = C.sample_render_rows(env, 32, rng=np.random.default_rng(SEED))

    checker = cls.reference_render(C.reference_env(cls, env.size), states)
    _assert_exact(cls, env._render(states).cpu().numpy(), checker, "checker floor")

    gradient = cls.reference_render(_gradient_reference(cls, env.size), states)
    ours = env._render(states, variation={"floor.pixel_encoding": [1]}).cpu().numpy()
    _assert_exact(cls, ours, gradient, "gradient floor")
    assert not np.array_equal(checker, gradient)
