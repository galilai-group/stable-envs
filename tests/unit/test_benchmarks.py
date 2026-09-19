"""The benchmark scripts' guards, tested. Every one of these is a number that got published wrong.

``examples/`` is not library code, which is exactly why it needs this file: a mistake in a benchmark
does not raise, it produces a plausible number that goes into a table and is read as a measurement.
The four guards asserted here each replaced such a number.

  **supersample resolution** — ``--ss`` defaulted to 1 and was passed to every ``se.make``. Forty-four
  registered classes declare ``default_supersample = 2``, so the sweep measured a picture their
  defaults do not produce and did it up to four times cheaper: measured in one process on an idle
  node, ``supersample=1`` against the class's own value is 1.16x on DMCBallInCupCatch, 1.26x on
  AntmazeLargeNavigate and 3.3x on MujocoPusher. ``bench_storage.py`` never passed a supersample and
  so measured the other picture, and the two published tables disagreed for that reason on the same
  44 envs. The guard is that ``ss=None`` must reach ``se.make`` as *no keyword at all*.

  **a requested size an env refuses** — several envs are pinned to their reference's output size on
  purpose. Recording that as an error drops them from the table and reads as "does not work" rather
  than "fixed size".

  **``size`` that means something else** — Craftax reads it as a PER-TILE edge, so ``size=256`` is a
  1792x2304 frame and batch 1024 asks the allocator for 47 GiB. Unguarded that reads as "this env
  OOMs". The aspect-preserving envs (highway-env draws 150x600) must NOT trip the same guard.

  **``ref.unwrapped`` outside its guard** — the draw-alone split is an optional extra on top of a
  timing that has already been taken. Reaching for ``unwrapped`` on a reference that is not a
  gymnasium env (dm_control hands back an ``Environment``, Jumanji a tuple, PGX a dict) threw out of
  the enclosing ``try`` and discarded the measurement, reporting the whole row as an error.

CPU-only and reference-free: the scripts are imported and their decision functions are driven with
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
    """Import a script from ``examples/`` by path.

    They import each other by bare module name (``from bench_gpu_vs_cpu import ...``), which only
    works with ``examples/`` on the path, so put it there rather than rewriting the scripts to be
    packages: they are meant to be run as ``python examples/bench_vs_decode.py``.
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


@pytest.fixture(scope="module")
def storage():
    _load("bench_gpu_vs_cpu")
    return _load("bench_storage")


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
    assert seen["batch"] == 8 and seen["size"] == 84 and seen["device"] == "cuda"

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
    for script in ("bench_gpu_vs_cpu", "bench_vs_decode"):
        src = (EXAMPLES / f"{script}.py").read_text()
        assert '"--ss", default="native"' in src, (
            f"{script}.py must default --ss to the env's own supersample; a numeric default silently "
            f"overrides default_supersample for 44 classes"
        )


# --------------------------------------------------------------------- size guards


@pytest.mark.parametrize(
    "hw,requested,tripped",
    [
        # Every shape here is one this sweep has actually produced; see examples/results/storage/*.json.
        ((1792, 2304), 256, True),  # CraftaxClassic: size is a PER-TILE edge
        ((256, 1024), 256, False),  # Highway, Merge: 1:4 and the requested 256 IS the short edge
        ((256, 512), 256, False),  # Parking
        ((256, 256), 256, False),
        ((512, 512), 256, False),  # exactly 2x is NOT over the bar
        ((513, 513), 256, True),
        ((224, 352), 224, False),  # OvercookedAsymmAdvantages at its own tile-aligned size
    ],
)
def test_per_tile_note(storage, hw, requested, tripped):
    note = storage.per_tile_note(hw, requested)
    assert (note is not None) == tripped, f"{hw} at size={requested}: got {note!r}"
    if tripped:
        assert "per tile" in note and f"{hw[0]}x{hw[1]}" in note


def test_per_tile_note_does_not_fire_on_the_long_edge(storage):
    """The regression the ``min``/``max`` choice is: a 1:4 env honours the request and must be kept.

    Highway, Merge and Parking render 256x1024, 256x1024 and 256x512 at ``size=256``. A guard written
    on ``max(hw)`` re-measures all three at native and rewrites three published rows for envs that did
    exactly what they were asked.
    """
    for hw in [(256, 1024), (256, 512), (150, 600)]:
        assert storage.per_tile_note(hw, min(hw)) is None, hw


class _FakeEnv:
    """Minimal stand-in for a StableEnv: knows its size and produces frames of a chosen shape."""

    def __init__(self, size, hw, state_dim=3, n_pool=64):
        self.size = size
        self._hw = hw
        self._sd = state_dim
        self._n = n_pool

    def sample_states(self, n, rng=None):
        return np.zeros((min(n, self._n), self._sd), np.float32)

    def render(self, states, **kw):
        import torch

        return torch.zeros((len(states), *self._hw, 3), dtype=torch.uint8)


def _fake_registry(monkeypatch, storage, table):
    """``table`` maps the requested size (or None for native) to an env or an exception."""

    def fake_make(name, num_envs=1, device="cuda", size=None, **kw):
        got = table[size]
        if isinstance(got, Exception):
            raise got
        return got

    monkeypatch.setattr(storage.se, "make", fake_make)


def test_probe_frames_happy_path(storage, monkeypatch):
    _fake_registry(monkeypatch, storage, {256: _FakeEnv(256, (256, 256))})
    pool, frames, hw, size, note = storage.probe_frames("X", 256, 8, 64)
    assert (hw, size, note) == ((256, 256), 256, None)
    assert len(frames) == 8


def test_probe_frames_env_refuses_the_size_at_construction(storage, monkeypatch):
    _fake_registry(
        monkeypatch,
        storage,
        {
            256: ValueError("renders at 224px only"),
            None: _FakeEnv(224, (224, 224)),
        },
    )
    pool, frames, hw, size, note = storage.probe_frames("X", 256, 8, 64)
    assert size == 224 and hw == (224, 224)
    assert note and "renders at 224px only" in note, (
        "an env pinned to its reference's output size must be measured at that size WITH the reason "
        "recorded, not dropped from the table as an error"
    )


def test_probe_frames_env_refuses_the_size_at_render(storage, monkeypatch):
    class _RefusesAtRender(_FakeEnv):
        def render(self, states, **kw):
            raise RuntimeError("needs a whole number of tiles")

    _fake_registry(
        monkeypatch,
        storage,
        {
            256: _RefusesAtRender(256, (256, 256)),
            None: _FakeEnv(64, (64, 64)),
        },
    )
    pool, frames, hw, size, note = storage.probe_frames("X", 256, 8, 64)
    assert size == 64 and hw == (64, 64)
    assert note and "whole number of tiles" in note


def test_probe_frames_size_means_per_tile(storage, monkeypatch):
    """Craftax: the size was accepted and meant something else. Fall back and say which."""
    _fake_registry(
        monkeypatch,
        storage,
        {
            256: _FakeEnv(256, (1792, 2304)),
            None: _FakeEnv(16, (112, 144)),
        },
    )
    pool, frames, hw, size, note = storage.probe_frames("X", 256, 4, 64)
    assert hw == (112, 144) and size == 16
    assert note and "per tile" in note


def test_probe_frames_reraises_when_native_also_fails(storage, monkeypatch):
    """A failure at the env's OWN size is a real error and must not be swallowed into a note."""
    _fake_registry(
        monkeypatch,
        storage,
        {
            256: ValueError("no"),
            None: RuntimeError("cannot instantiate at all"),
        },
    )
    with pytest.raises(RuntimeError, match="cannot instantiate at all"):
        storage.probe_frames("X", 256, 4, 64)


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
