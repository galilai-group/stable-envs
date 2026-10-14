"""The benchmark scripts' guards, tested. Every one of these is a number that got published wrong.

``examples/`` is not library code, which is exactly why it needs this file: a mistake in a benchmark
does not raise, it produces a plausible number that goes into a table and is read as a measurement.
The two guards asserted here each replaced such a number.

  **supersample resolution** — ``--ss`` defaulted to 1 and was passed to every ``se.make``. Forty-four
  registered classes declare ``default_supersample = 2``, so the sweep measured a picture their
  defaults do not produce and did it up to four times cheaper: measured in one process on an idle
  node, ``supersample=1`` against the class's own value is 1.16x on DMCBallInCupCatch, 1.26x on
  AntmazeLargeNavigate and 3.3x on MujocoPusher. The guard is that ``ss=None`` must reach
  ``se.make`` as *no keyword at all*.

  **``ref.unwrapped`` outside its guard** — the draw-alone split is an optional extra on top of a
  timing that has already been taken. Reaching for ``unwrapped`` on a reference that is not a
  gymnasium env (dm_control hands back an ``Environment``, Jumanji a tuple, PGX a dict) threw out of
  the enclosing ``try`` and discarded the measurement, reporting the whole row as an error.

CPU-only and reference-free: the script is imported and its decision functions are driven with
fakes, so this costs a second and runs anywhere rather than being a thing to remember on a GPU node.
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
    """``ss=None`` must leave the env to its own ``default_supersample``.

    Asserted as "the keyword is absent", not as "the keyword is None", because ``se.make(...,
    supersample=None)`` reaches ``StableEnv.__init__``'s ``supersample: int = 1`` default for every
    class that does not override ``__init__`` — which is the same substitution under another name.
    """
    seen = {}

    def fake_make(name, **kw):
        seen.update(kw)
        return object()

    monkeypatch.setattr(bench.se, "make", fake_make)
    bench.make_env("Whatever", 8, 84, None)
    assert "supersample" not in seen, (
        f"ss=None must not reach se.make at all, got {seen} — passing supersample=1 measures a "
        f"different picture than se.make(name) produces for the 44 classes declaring 2"
    )
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
    """The DEFAULT is the thing that got published, so pin it rather than the parser."""
    src = (EXAMPLES / "bench_gpu_vs_cpu.py").read_text()
    assert '"--ss"' in src and 'default="native"' in src, (
        "bench_gpu_vs_cpu.py must default --ss to the env's own supersample; a numeric default silently "
        "overrides default_supersample for 44 classes"
    )


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
    """A reference with no ``unwrapped`` must still report ``set_state_render``.

    This is the regression: the optional draw-alone split reached ``ref.unwrapped`` outside its own
    guard, so any non-gymnasium reference raised past the measurement that had already completed and
    the row came back as an error. dm_control, Jumanji and PGX are all in that position, which is 30+
    envs whose official-renderer column silently went blank.
    """
    calls: list[int] = []
    pool = np.zeros((32, 3), np.float32)
    res = bench.cpu_throughput(_cls_for(ref, calls), "X", 8, 8, pool)
    assert "error" not in res, res
    assert res["set_state_render"]["fps"] > 0
    assert ("render_only" in res) is expect_draw
    assert calls, "the combined number must actually have been measured"


def test_cpu_throughput_returns_none_without_a_reference(bench):
    class _NoRef:
        reference_id = None

    assert bench.cpu_throughput(_NoRef, "X", 8, 8, np.zeros((4, 2), np.float32)) is None
