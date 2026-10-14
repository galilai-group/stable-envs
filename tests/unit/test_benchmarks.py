"""Guards in ``examples/bench_gpu_vs_cpu.py``: supersample resolution and the optional draw-alone split.

The script is imported and its decision functions are driven with fakes, so this runs on CPU without
any reference simulator.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load(name: str):
    """Import a script from ``examples/`` by path, with ``examples/`` on the path.

    The scripts are meant to be run as ``python examples/<script>.py``, not imported as packages.
    """
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _load("bench_gpu_vs_cpu")


# --------------------------------------------------------------------- supersample


def test_native_supersample_is_not_passed_as_one(bench, monkeypatch):
    """``ss=None`` leaves the env to its own ``default_supersample``.

    Checked as an absent keyword: an explicit ``supersample=None`` would fall through to the
    ``supersample: int = 1`` default of ``StableEnv.__init__``.
    """
    seen = {}

    def fake_make(name, **kw):
        seen.update(kw)
        return object()

    monkeypatch.setattr(bench.se, "make", fake_make)
    bench.make_env("Whatever", 8, 84, None)
    assert "supersample" not in seen, f"ss=None must not pass supersample to se.make, got {seen}"
    assert seen["num_envs"] == 8 and seen["size"] == 84 and seen["device"] == "cuda"

    seen.clear()
    bench.make_env("Whatever", 8, 84, 2)
    assert seen["supersample"] == 2, "an explicitly requested supersample must still be honoured"


@pytest.mark.parametrize(
    "tok,want",
    [("native", None), ("default", None), ("env", None), ("none", None), ("NATIVE", None), ("1", 1), ("2", 2)],
)
def test_parse_ss(bench, tok, want):
    assert bench.parse_ss(tok) == want


def test_ss_flag_defaults_to_native(bench):
    """``--ss`` defaults to the env's own supersample."""
    src = (EXAMPLES / "bench_gpu_vs_cpu.py").read_text()
    assert '"--ss"' in src and 'default="native"' in src, "bench_gpu_vs_cpu.py must default --ss to native"


# --------------------------------------------------------------------- cpu_throughput


class _Ref:
    """A reference object that is NOT a gymnasium env: no ``unwrapped``, like dm_control's."""

    closed = False

    def close(self):
        self.closed = True


class _GymLikeRef(_Ref):
    class _U:
        def _render(self, visualize=False):
            return None

    unwrapped = _U()


def _cls_for(ref_obj, calls):
    class _Cls:
        reference_id = "fake/Env-v0"

        @classmethod
        def reference_env(cls, size):
            return ref_obj

        @classmethod
        def reference_render(cls, env, states):
            calls.append(len(states))
            return np.zeros((len(states), 8, 8, 3), np.uint8)

        @classmethod
        def reference_set_state(cls, u, st):
            return None

    return _Cls


@pytest.mark.parametrize("ref,expect_draw", [(_Ref(), False), (_GymLikeRef(), True)])
def test_cpu_throughput_keeps_the_timing_it_already_took(bench, ref, expect_draw):
    """A reference with no ``unwrapped`` still reports ``set_state_render``.

    The draw-alone split is optional; a non-gymnasium reference (dm_control, Jumanji, PGX) must not
    discard the timing already taken.
    """
    calls: list[int] = []
    pool = np.zeros((32, 3), np.float32)
    res = bench.cpu_throughput(_cls_for(ref, calls), "X", 8, 8, pool)
    assert "error" not in res, res
    assert res["set_state_render"]["fps"] > 0
    assert ("render_only" in res) is expect_draw
    assert calls, "set_state_render was not measured"


def test_cpu_throughput_returns_none_without_a_reference(bench):
    class _NoRef:
        reference_id = None

    assert bench.cpu_throughput(_NoRef, "X", 8, 8, np.zeros((4, 2), np.float32)) is None
