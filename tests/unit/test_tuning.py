"""Tuning knobs: precedence, validation, environment variables, the measured cache, ``tune``, hint and autotune."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import stable_envs as snv
from stable_envs import _tune, cli, doctor, registry, tuning
from stable_envs._execution import policy
from stable_envs.env import Env
from stable_envs.spaces import Dict

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("loguru_to_caplog")]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in ("STABLE_ENVS_AUTOTUNE", "STABLE_ENVS_TUNING_HINT", *(k.env_var for k in tuning.KNOBS.values())):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    tuning.reset()
    tuning._clear_caches()
    yield
    tuning.reset()
    tuning._clear_caches()


class Probe(Env):
    """A render-only env on the navigation path whose render time depends on its chunk."""

    family = "navigation"
    default_size = 8
    state_dim = 1
    state_names = ("x",)

    def make_variation_space(self):
        return Dict({})

    def sample_states(self, batch, rng=None):
        return np.zeros((batch, 1), np.float32)

    def chunk(self):
        return tuning.get("nav_render_chunk", path="navigation", size=self.size, batch=self.B)

    def _render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None):
        tuning.on_render("navigation", self.size, self.B, self)
        time.sleep(0.002 * (1 + abs(self.chunk() - 64) / 64))
        return torch.zeros(self.B, self.size, self.size, 3, dtype=torch.uint8)


def _draw(env):
    return env._render(torch.zeros(env.B, 1))


def _cache(value=16, batch=4, size=8):
    return tuning.write_cache({tuning.entry_key("navigation", size, batch): {"nav_render_chunk": value}}, "cpu")


# --------------------------------------------------------------------------- registry, validation, precedence


def test_every_knob_documents_itself_and_has_a_valid_fallback():
    paths = {p for ps in tuning.RENDER_PATHS.values() for p in ps}
    for name, knob in tuning.KNOBS.items():
        assert knob.name == name and knob.doc and set(knob.paths) <= paths
        assert knob.device in ("cpu", "cuda", "any") and knob.check(knob.fallback) == knob.fallback


@pytest.mark.parametrize(
    "knobs",
    [{"no_such_knob": 1}, {"cpu3d_chunk": 0}, {"cpu3d_chunk": 2.0}, {"cpu3d_chunk": True}, {"cpu3d_chunk": "8"}]
    + [{"raster2d_block": 96}, {"navdot_warps": 3}],
)
def test_invalid_names_and_values_raise_and_change_nothing(knobs):
    with pytest.raises(ValueError):
        tuning.set(**knobs)
    with pytest.raises(ValueError), tuning.override(**knobs):
        pass
    assert tuning.set() == {}
    with pytest.raises(ValueError, match="unknown tuning knob"):
        tuning.get("chunk")


def test_layers_take_precedence_in_order(monkeypatch):
    where = dict(path="navigation", size=8, batch=4)
    assert tuning.get("nav_render_chunk", **where) == 256
    assert tuning.get("nav_render_chunk", default=32, **where) == 32
    _cache(16)
    assert tuning.get("nav_render_chunk", default=32, **where) == 16
    assert tuning.get("nav_render_chunk") == 256
    monkeypatch.setenv("STABLE_ENVS_NAV_RENDER_CHUNK", "4")
    assert tuning.get("nav_render_chunk", **where) == 4
    tuning.set(nav_render_chunk=8)
    assert tuning.get("nav_render_chunk", **where) == 8
    with tuning.override(nav_render_chunk=2):
        with tuning.override(nav_render_chunk=1):
            assert tuning.get("nav_render_chunk") == 1
        assert tuning.get("nav_render_chunk") == 2
    tuning.set(nav_render_chunk=None)
    assert tuning.get("nav_render_chunk", **where) == 4
    assert tuning.set(workers=0) == {"workers": 0} and tuning.get("workers") == 0
    assert tuning.describe()["workers"] == (0, "set")


# --------------------------------------------------------------------------- environment variables


@pytest.mark.parametrize(("text", "value"), [("8", 8), (" 16 ", 16), ("1_024", 1024), ("2K", 2048), ("6GiB", 6 << 30)])
def test_environment_values_parse(monkeypatch, text, value):
    monkeypatch.setenv("STABLE_ENVS_NVD_BUDGET_BYTES", text)
    assert tuning.get("nvd_budget_bytes") == value


def test_auto_defers_to_the_lower_layers(monkeypatch):
    monkeypatch.setenv("STABLE_ENVS_CPU3D_CHUNK", "auto")
    assert tuning.get("cpu3d_chunk") == 16


@pytest.mark.parametrize("text", ["eight", "8.5", "-4", "0", "1e3"])
def test_bad_environment_values_name_the_variable(monkeypatch, text):
    monkeypatch.setenv("STABLE_ENVS_CPU3D_CHUNK", text)
    with pytest.raises(ValueError, match="STABLE_ENVS_CPU3D_CHUNK"):
        tuning.get("cpu3d_chunk")


# --------------------------------------------------------------------------- the measured cache


@pytest.mark.parametrize(("batch", "lo", "hi"), [(1, 1, 1), (2, 2, 3), (64, 64, 127), (100, 64, 127)])
def test_batch_ranges_are_power_of_two_buckets(batch, lo, hi):
    assert tuning.batch_range(batch) == (lo, hi)


def test_cache_round_trip_applies_to_its_path_size_and_batch_range_only(tmp_path):
    path = _cache(16)
    assert path.parent == tmp_path / "stable-envs" / "tuning" and all(c.isalnum() or c in "._-" for c in path.stem)
    data = json.loads(path.read_text())
    assert data["entries"] == {"navigation|8|4-7": {"nav_render_chunk": 16}}
    assert data["stamp"] == tuning.stamp("cpu")
    assert [Probe(num_envs=n).chunk() for n in (4, 7, 8)] == [16, 16, 256]
    assert Probe(num_envs=4, size=16).chunk() == 256


def test_a_stale_or_corrupt_cache_is_ignored():
    path = _cache(16)
    data = json.loads(path.read_text())
    data["stamp"]["stable_envs"] = "0.0.0"
    path.write_text(json.dumps(data))
    tuning._clear_caches()
    assert Probe().chunk() == 256 and tuning.read_cache("cpu")[1] is False
    path.write_text("{not json")
    tuning._clear_caches()
    with pytest.warns(RuntimeWarning, match="ignoring tuning cache"):
        assert Probe().chunk() == 256
    with pytest.raises(ValueError):
        _cache(0)


# --------------------------------------------------------------------------- tune


def test_tune_sweeps_from_the_default_and_later_envs_on_the_path_use_it(capsys):
    best = snv.tune(Probe(num_envs=512), repeats=2)
    out = capsys.readouterr().out
    assert best == {"nav_render_chunk": 64}
    assert "saved for: navigation, 8 px, batch 512-1023" in out and "speedup" in out
    assert Probe(num_envs=600).chunk() == 64 and Probe(num_envs=4).chunk() == 256


def test_tune_starts_from_the_env_call_site_default(capsys):
    class Own(Probe):
        def _tuning_default(self, name, size):
            return 32 if name == "nav_render_chunk" else None

    snv.tune(Own(num_envs=512), repeats=1, write=False)
    row = next(line.split() for line in capsys.readouterr().out.splitlines() if "nav_render_chunk" in line)
    assert row[1] == "32"


def test_tune_without_write_leaves_no_cache():
    snv.tune(Probe(num_envs=512), repeats=1, write=False, verbose=False)
    assert not tuning.cache_path("cpu").exists()


def test_candidates_start_from_the_default_and_prune_equal_splits():
    env = Probe(num_envs=100)
    assert _tune.candidates(tuning.KNOBS["nav_render_chunk"], 256, env) == [64, 256]
    assert _tune.candidates(tuning.KNOBS["navdot_warps"], 4, env) == [1, 2, 4, 8, 16]


def test_cli_show_clear_and_all(monkeypatch, capsys):
    assert cli.main(["tune", "--show"]) == 0 and "no tuning cached" in capsys.readouterr().out
    _cache(16)
    assert cli.main(["tune", "--show"]) == 0 and "nav_render_chunk=16" in capsys.readouterr().out
    assert _tune.status("cpu").split(" of ")[0] in ("0", "1")
    assert cli.main(["tune", "--clear"]) == 0 and not tuning.cache_path("cpu").exists()
    monkeypatch.setattr(_tune.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    assert cli.main(["tune", "--all", "--device", "cpu"]) == 1 and "--yes" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("knobs", "batch", "expected"),
    [({}, 64, 8), ({"workers": 0}, 64, 0), ({"workers": 3}, 64, 3), ({}, 32, 0), ({"worker_min_lanes": 8}, 8, 8)],
)
def test_worker_knobs_drive_process_count(monkeypatch, knobs, batch, expected):
    monkeypatch.setattr(policy.sys, "platform", "linux")
    monkeypatch.setattr(policy.mp, "current_process", lambda: SimpleNamespace(daemon=False))
    with tuning.override(**knobs):
        assert policy.process_count(batch, list(range(16)), path="classic-sim") == expected


@pytest.mark.parametrize(("knobs", "expected"), [({}, 120), ({"worker_timeout": 7}, 7)])
def test_worker_timeout_sets_how_long_a_reply_can_take(monkeypatch, knobs, expected):
    from stable_envs._execution import requests

    clock = iter(range(10_000))
    monkeypatch.setattr(requests.time, "monotonic", lambda: next(clock))
    owner = requests.ProcessRequests()
    owner._connections = [SimpleNamespace(poll=lambda _: False)]
    owner._processes = [SimpleNamespace(is_alive=lambda: True, pid=1)]
    with tuning.override(**knobs), pytest.raises(RuntimeError, match="timed out"):
        owner._responses()
    # The deadline is read once; each poll reads the clock once more.
    assert next(clock) == expected + 1


@pytest.mark.parametrize(
    ("capability", "arches", "ok"),
    [
        ((8, 6), ["sm_80"], True),
        ((8, 0), ["sm_86"], False),
        ((9, 0), ["sm_80", "compute_80"], True),
        ((12, 0), ["sm_80", "sm_90", "sm_100"], False),
        ((12, 0), ["sm_90", "sm_120"], True),
        ((10, 0), ["sm_90a", "sm_100"], True),
    ],
)
def test_torch_arch_compatibility(capability, arches, ok):
    assert doctor.runs_on(capability, arches) is ok


# --------------------------------------------------------------------------- hint and autotune


def _logged(caplog, text):
    return [r for r in caplog.records if r.name == "stable_envs.tuning" and text in r.getMessage()]


def test_a_large_untuned_render_logs_one_hint_per_path(caplog):
    caplog.set_level("INFO", logger="stable_envs.tuning")
    _draw(Probe(num_envs=tuning.HINT_LANES - 1))
    assert _logged(caplog, "stable-envs tune") == []
    env = Probe(num_envs=tuning.HINT_LANES)
    _draw(env)
    _draw(env)
    _draw(Probe(num_envs=tuning.HINT_LANES))
    (hint,) = _logged(caplog, "stable-envs tune Probe")
    assert hint.levelname == "INFO"


def test_the_hint_is_hidden_by_its_variable_and_by_a_cached_entry(caplog, monkeypatch):
    caplog.set_level("INFO", logger="stable_envs.tuning")
    monkeypatch.setenv("STABLE_ENVS_TUNING_HINT", "0")
    _draw(Probe(num_envs=tuning.HINT_LANES))
    monkeypatch.delenv("STABLE_ENVS_TUNING_HINT")
    tuning._clear_caches()
    _cache(64, batch=tuning.HINT_LANES)
    _draw(Probe(num_envs=tuning.HINT_LANES))
    assert _logged(caplog, "stable-envs tune") == []


@pytest.fixture
def autotuning(monkeypatch, caplog):
    monkeypatch.setenv("STABLE_ENVS_AUTOTUNE", "1")
    monkeypatch.setattr(registry, "make", lambda name, **kw: Probe(**kw))
    caplog.set_level("INFO", logger="stable_envs.tuning")


def test_autotune_caches_the_path_on_first_render(autotuning):
    _draw(Probe(num_envs=tuning.HINT_LANES))
    assert Probe(num_envs=tuning.HINT_LANES).chunk() == 64


def test_autotune_falls_back_without_waiting_when_another_process_holds_the_lock(autotuning, caplog):
    fcntl = pytest.importorskip("fcntl")
    lock = tuning.cache_path("cpu").with_suffix(".lock")
    lock.parent.mkdir(parents=True)
    with open(lock, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        start = time.perf_counter()
        assert _draw(Probe(num_envs=tuning.HINT_LANES)).shape[0] == tuning.HINT_LANES
        assert time.perf_counter() - start < 1.0
    assert not tuning.cache_path("cpu").exists() and _logged(caplog, "another process is tuning")


def test_a_failing_autotune_never_reaches_the_render(autotuning, caplog, monkeypatch):
    monkeypatch.setattr(_tune, "tune", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _draw(Probe(num_envs=tuning.HINT_LANES)).shape[0] == tuning.HINT_LANES
    assert _logged(caplog, "autotune of Probe failed")
