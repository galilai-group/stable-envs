"""Tuning knobs: precedence, validation, environment variables, the measured cache and ``tune``."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import stable_envs as snv
from stable_envs import _tune, cli, tuning
from stable_envs._execution import policy
from stable_envs.env import Env
from stable_envs.spaces import Dict

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for knob in tuning.KNOBS.values():
        monkeypatch.delenv(knob.env_var, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    tuning.reset()
    tuning._clear_caches()
    yield
    tuning.reset()
    tuning._clear_caches()


class Probe(Env):
    """A render-only env on the navigation path whose frames record the chunk it saw."""

    family = "navigation"
    default_size = 8
    seen: list = []

    def __init__(self, num_envs=4, size=None, device="cpu", seed=0):
        super().__init__(num_envs=num_envs, size=size, device=device, seed=seed)
        self.built_with = tuning.get("nav_render_chunk")

    def make_variation_space(self):
        return Dict({})

    def reset(self, seed=None):
        return torch.arange(self.B, dtype=torch.float32)

    def render(self, state=None, **kwargs):
        chunk = tuning.get("nav_render_chunk")
        Probe.seen.append(chunk)
        time.sleep(0.002 * (1 + abs(chunk - 64) / 64))
        return torch.zeros(self.B, self.size, self.size, 3, dtype=torch.uint8)


# --------------------------------------------------------------------------- registry and validation


def test_every_knob_documents_itself_and_has_a_valid_fallback():
    for name, knob in tuning.KNOBS.items():
        assert knob.name == name and knob.doc and knob.paths
        assert knob.device in ("cpu", "cuda", "any")
        assert knob.check(knob.fallback) == knob.fallback
        assert set(knob.paths) <= {p for paths in tuning.RENDER_PATHS.values() for p in paths}
        assert knob.env_var == "STABLE_ENVS_" + name.upper()


@pytest.mark.parametrize(
    "knobs",
    [
        {"no_such_knob": 1},
        {"cpu3d_chunk": 0},
        {"cpu3d_chunk": -3},
        {"cpu3d_chunk": 2.0},
        {"cpu3d_chunk": True},
        {"cpu3d_chunk": "8"},
        {"raster2d_block": 96},
        {"navdot_warps": 3},
    ],
)
def test_invalid_names_and_values_raise_and_change_nothing(knobs):
    with pytest.raises(ValueError):
        tuning.set(**knobs)
    with pytest.raises(ValueError):
        with tuning.override(**knobs):
            pass
    assert tuning.set() == {}


def test_unknown_names_raise_on_get():
    with pytest.raises(ValueError, match="unknown tuning knob"):
        tuning.get("chunk")


def test_workers_accepts_zero():
    tuning.set(workers=0)
    assert tuning.get("workers") == 0


# --------------------------------------------------------------------------- precedence


def test_layers_take_precedence_in_order(monkeypatch):
    assert tuning.get("cpu3d_chunk") == 16
    assert tuning.get("cpu3d_chunk", default=32) == 32
    monkeypatch.setenv("STABLE_ENVS_CPU3D_CHUNK", "4")
    assert tuning.get("cpu3d_chunk", default=32) == 4
    tuning.set(cpu3d_chunk=8)
    assert tuning.get("cpu3d_chunk") == 8
    with tuning.override(cpu3d_chunk=2):
        assert tuning.get("cpu3d_chunk") == 2
        with tuning.override(cpu3d_chunk=1):
            assert tuning.get("cpu3d_chunk") == 1
        assert tuning.get("cpu3d_chunk") == 2
    assert tuning.get("cpu3d_chunk") == 8
    tuning.set(cpu3d_chunk=None)
    assert tuning.get("cpu3d_chunk") == 4


def test_instance_tuning_outranks_every_other_layer_only_while_that_env_runs():
    with tuning.override(nav_render_chunk=32):
        env = Probe(tuning={"nav_render_chunk": 8})
        other = Probe()
        assert env.built_with == 8 and other.built_with == 32
        Probe.seen.clear()
        env.render()
        other.render()
        assert Probe.seen == [8, 32]
    assert tuning.get("nav_render_chunk") == 256


def test_instance_tuning_is_validated():
    with pytest.raises(ValueError, match="unknown tuning knob"):
        Probe(tuning={"nav_chunk": 8})
    with pytest.raises(ValueError):
        Probe(tuning={"nav_render_chunk": 0})


def test_describe_reports_value_and_source():
    tuning.set(cpu3d_chunk=8)
    info = tuning.describe()
    assert info["cpu3d_chunk"]["value"] == 8 and info["cpu3d_chunk"]["source"] == "set"
    assert info["workers"]["source"] in ("fallback", "derived")
    assert set(info) == set(tuning.KNOBS)


# --------------------------------------------------------------------------- environment variables


@pytest.mark.parametrize(
    ("text", "value"),
    [("8", 8), (" 16 ", 16), ("1_024", 1024), ("2K", 2048), ("6G", 6 << 30), ("6GiB", 6 << 30), ("3m", 3 << 20)],
)
def test_environment_values_parse(monkeypatch, text, value):
    monkeypatch.setenv("STABLE_ENVS_NVD_BUDGET_BYTES", text)
    assert tuning.get("nvd_budget_bytes") == value


@pytest.mark.parametrize("text", ["auto", "", "AUTO"])
def test_auto_defers_to_the_lower_layers(monkeypatch, text):
    monkeypatch.setenv("STABLE_ENVS_CPU3D_CHUNK", text)
    assert tuning.get("cpu3d_chunk") == 16


@pytest.mark.parametrize("text", ["eight", "8.5", "-4", "0", "1e3"])
def test_bad_environment_values_name_the_variable(monkeypatch, text):
    monkeypatch.setenv("STABLE_ENVS_CPU3D_CHUNK", text)
    with pytest.raises(ValueError, match="cpu3d_chunk|STABLE_ENVS_CPU3D_CHUNK"):
        tuning.get("cpu3d_chunk")


# --------------------------------------------------------------------------- the measured cache


def test_cache_path_follows_xdg_and_sanitizes_the_device_name(tmp_path):
    path = tuning.cache_path("cpu")
    assert path.parent == tmp_path / "stable-envs" / "tuning"
    assert path.suffix == ".json" and all(c.isalnum() or c in "._-" for c in path.stem)


@pytest.mark.parametrize(("batch", "lo", "hi"), [(1, 1, 1), (2, 2, 3), (64, 64, 127), (100, 64, 127), (128, 128, 255)])
def test_batch_ranges_are_power_of_two_buckets(batch, lo, hi):
    assert tuning.batch_range(batch) == (lo, hi)


def test_cache_round_trip_applies_to_its_path_size_and_batch_range_only():
    key = tuning.entry_key("navigation", 8, 4)
    path = tuning.write_cache({key: {"knobs": {"nav_render_chunk": 16}}}, "cpu")
    data = json.loads(path.read_text())
    assert data["entries"][key]["knobs"] == {"nav_render_chunk": 16}
    assert data["stamp"] == tuning.stamp("cpu")
    assert Probe(num_envs=4).built_with == 16
    assert Probe(num_envs=7).built_with == 16
    assert Probe(num_envs=8).built_with == 256
    assert Probe(num_envs=4, size=16).built_with == 256
    assert tuning.get("nav_render_chunk") == 256  # no environment is running
    tuning.set(nav_render_chunk=32)
    assert Probe(num_envs=4).built_with == 32


def test_a_stale_cache_is_ignored(monkeypatch):
    key = tuning.entry_key("navigation", 8, 4)
    path = tuning.write_cache({key: {"knobs": {"nav_render_chunk": 16}}}, "cpu")
    data = json.loads(path.read_text())
    data["stamp"]["stable_envs"] = "0.0.0"
    path.write_text(json.dumps(data))
    tuning._clear_caches()
    assert Probe(num_envs=4).built_with == 256
    assert tuning.read_cache("cpu")[1] is False


def test_a_corrupt_cache_warns_and_is_ignored():
    path = tuning.cache_path("cpu")
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    with pytest.warns(RuntimeWarning, match="ignoring tuning cache"):
        assert Probe(num_envs=4).built_with == 256


def test_write_cache_validates_values():
    with pytest.raises(ValueError):
        tuning.write_cache({tuning.entry_key("navigation", 8, 4): {"knobs": {"nav_render_chunk": 0}}}, "cpu")


# --------------------------------------------------------------------------- tune


def test_tune_sweeps_from_the_default_saves_for_the_path_and_later_envs_use_it(capsys):
    env = Probe(num_envs=512)
    best = snv.tune(env, repeats=2)
    assert set(best) == {"nav_render_chunk"}
    out = capsys.readouterr().out
    assert "saved for: navigation, 8 px, batch 512-1023" in out and "speedup" in out
    assert best["nav_render_chunk"] == 64
    assert Probe(num_envs=600).built_with == 64
    assert Probe(num_envs=4).built_with == 256


def test_tune_without_write_leaves_no_cache():
    snv.tune(Probe(num_envs=512), repeats=1, write=False, verbose=False)
    assert not tuning.cache_path("cpu").exists()


def test_candidates_start_from_the_default_and_prune_equal_splits():
    env = Probe(num_envs=100)
    knob = tuning.KNOBS["nav_render_chunk"]
    values = _tune.candidates(knob, 256, env)
    assert 256 in values and values == sorted(values)
    assert [v for v in values if v >= 100] == [256]
    assert _tune.candidates(tuning.KNOBS["navdot_warps"], 4, env) == [1, 2, 4, 8, 16]


def test_cli_show_and_clear(capsys):
    assert cli.main(["tune", "--show"]) == 0
    assert "no tuning cached" in capsys.readouterr().out
    tuning.write_cache({tuning.entry_key("navigation", 8, 4): {"knobs": {"nav_render_chunk": 16}}}, "cpu")
    assert cli.main(["tune", "--show"]) == 0
    assert "nav_render_chunk=16" in capsys.readouterr().out
    assert cli.main(["tune", "--clear"]) == 0
    assert not tuning.cache_path("cpu").exists()


def test_cli_all_asks_before_starting(monkeypatch, capsys):
    monkeypatch.setattr(_tune.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    assert cli.main(["tune", "--all", "--device", "cpu"]) == 1
    assert "pass --yes" in capsys.readouterr().out


def test_status_counts_tuned_paths():
    before = _tune.status("cpu")
    assert before.startswith("0 of ")
    tuning.write_cache({tuning.entry_key("navigation", 8, 4): {"knobs": {"nav_render_chunk": 16}}}, "cpu")
    assert _tune.status("cpu").startswith(("1 of ", "0 of "))


# --------------------------------------------------------------------------- worker policy


@pytest.mark.parametrize(
    ("knobs", "batch", "expected"),
    [({}, 64, 8), ({"workers": 0}, 64, 0), ({"workers": 3}, 64, 3), ({}, 32, 0), ({"worker_min_lanes": 8}, 8, 8)],
)
def test_worker_knobs_drive_process_count(monkeypatch, knobs, batch, expected):
    monkeypatch.setattr(policy.sys, "platform", "linux")
    monkeypatch.setattr(policy.mp, "current_process", lambda: SimpleNamespace(daemon=False))
    with tuning.override(**knobs):
        assert policy.process_count(batch, list(range(16))) == expected


# --------------------------------------------------------------------------- doctor


@pytest.mark.parametrize(
    ("capability", "arches", "ok"),
    [
        ((8, 6), ["sm_80"], True),
        ((8, 0), ["sm_86"], False),
        ((9, 0), ["sm_80", "sm_86"], False),
        ((9, 0), ["sm_80", "compute_80"], True),
        ((12, 0), ["sm_80", "sm_90", "sm_100"], False),
        ((12, 0), ["sm_90", "sm_120"], True),
        ((10, 0), ["sm_90a", "sm_100"], True),
    ],
)
def test_torch_arch_compatibility(capability, arches, ok):
    from stable_envs import doctor

    assert doctor.runs_on(capability, arches) is ok


# --------------------------------------------------------------------------- hint and autotune


def _hints(caplog):
    return [r for r in caplog.records if r.name == "stable_envs.tuning" and "stable-envs tune" in r.getMessage()]


def test_a_large_untuned_render_logs_one_hint_per_path(caplog, monkeypatch):
    monkeypatch.delenv("STABLE_ENVS_AUTOTUNE", raising=False)
    caplog.set_level("INFO", logger="stable_envs.tuning")
    Probe(num_envs=tuning.HINT_LANES - 1).render()
    assert _hints(caplog) == []
    env = Probe(num_envs=tuning.HINT_LANES)
    env.render()
    env.render()
    Probe(num_envs=tuning.HINT_LANES).render()
    (hint,) = _hints(caplog)
    assert hint.levelname == "INFO" and "stable-envs tune Probe" in hint.getMessage()


def test_the_hint_is_silenced_by_its_variable_and_by_a_cached_entry(caplog, monkeypatch):
    monkeypatch.delenv("STABLE_ENVS_AUTOTUNE", raising=False)
    caplog.set_level("INFO", logger="stable_envs.tuning")
    monkeypatch.setenv("STABLE_ENVS_TUNING_HINT", "0")
    Probe(num_envs=tuning.HINT_LANES).render()
    assert _hints(caplog) == []
    monkeypatch.delenv("STABLE_ENVS_TUNING_HINT")
    tuning._clear_caches()
    key = tuning.entry_key("navigation", 8, tuning.HINT_LANES)
    tuning.write_cache({key: {"knobs": {"nav_render_chunk": 64}}}, "cpu")
    Probe(num_envs=tuning.HINT_LANES).render()
    assert _hints(caplog) == []


def test_autotune_caches_the_path_on_first_render(monkeypatch):
    monkeypatch.setenv("STABLE_ENVS_AUTOTUNE", "1")
    Probe(num_envs=tuning.HINT_LANES).render()
    data, current = tuning.read_cache("cpu")
    assert current and tuning.entry_key("navigation", 8, tuning.HINT_LANES) in data["entries"]
    assert Probe(num_envs=tuning.HINT_LANES).built_with == 64


def test_autotune_falls_back_without_waiting_when_another_process_holds_the_lock(monkeypatch, caplog):
    pytest.importorskip("fcntl")
    monkeypatch.setenv("STABLE_ENVS_AUTOTUNE", "1")
    caplog.set_level("INFO", logger="stable_envs.tuning")
    lock = tuning.cache_path("cpu").with_suffix(".lock")
    with tuning._try_lock(lock) as held:
        assert held
        start = time.perf_counter()
        frames = Probe(num_envs=tuning.HINT_LANES).render()
        assert time.perf_counter() - start < 1.0
    assert frames.shape[0] == tuning.HINT_LANES
    assert not tuning.cache_path("cpu").exists()
    assert any("another process is tuning" in r.getMessage() for r in caplog.records)


def test_a_failing_autotune_never_reaches_the_render(monkeypatch, caplog):
    monkeypatch.setenv("STABLE_ENVS_AUTOTUNE", "1")
    caplog.set_level("INFO", logger="stable_envs.tuning")

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(_tune, "tune", broken)
    assert Probe(num_envs=tuning.HINT_LANES).render().shape[0] == tuning.HINT_LANES
    assert any("autotune of navigation failed" in r.getMessage() for r in caplog.records)
