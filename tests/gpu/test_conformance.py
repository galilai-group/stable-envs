"""GPU conformance suite, parameterized over the env registry.

Requires a CUDA device; the module skips without one.

Run:  pytest tests/gpu -v          (on a host with a CUDA device)
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

import _replay_state as RS

import stable_envs as se
from stable_envs.testing import conformance as C
from stable_envs.testing._sampling import has_render_rows
from stable_envs.testing.provenance import reference_version_mismatch
from tests.helpers import source_mse as SM
from tests.helpers.reference import dmc_source_render
from tests.helpers.reference.certification import has_reference_renderer

ENV_NAMES = sorted(se.registry.REGISTRY)

#: Minimum ``1 - mean|d|(declared) / mean|d|(flipped)`` inside a task object's footprint, pooled over
#: the state stack: the declared visibility must be at least twice as close to the official frame as
#: the flipped one. Measured margins over five draws per cell at 84 px:
#:
#:     cell                margin (CPU)      margin (RTX 3090)   declared mean|d|   flipped mean|d|
#:     DMCReacher      0.8730 - 0.8806   0.8513 - 0.8604     17.0 - 20.3        132 - 144
#:     DMCFingerTurnEasy   0.8263 - 0.8506   0.9445 - 0.9503      5.1 - 16.9         97 - 104
#:     DMCFingerTurnHard   0.8199 - 0.8713   0.9090 - 0.9147      9.0 - 18.4         99 - 108
#:
#: Drawing an object the reference does not draw gives a margin around -8. A within-8/255 fraction is
#: not used because on footprints that are nearly all silhouette it differs between CPU and GPU.
TASK_OBJECT_PRESENCE_MARGIN = 0.50

#: Seed for the states the reference-based tests draw, so a failure is reproducible regardless of test
#: order. It is not tuned to clear any floor, and is separate from ``render_worker.STATE_SEED``.
PARITY_STATE_SEED = 20260811


def _installed(packages) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in packages:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


@pytest.fixture(scope="module", params=ENV_NAMES)
def env(request):
    env = se.make(request.param, num_envs=16, device="cuda")
    if not has_render_rows(env):
        env.reset(seed=0)  # a replay State such as Craftax's exists only after the first reset
    return env


def test_registry_is_not_empty():
    errs = se.import_errors()
    assert ENV_NAMES, f"no envs registered; env-module import errors: {errs}"
    if errs:
        # A missing optional dependency removes its envs; printed for diagnosis.
        print(f"\nenv modules that did not import: {errs}")


def test_required_env_modules_actually_imported():
    """Every env module named in ``SE_REQUIRE_ENV_MODULES`` imported and registered envs.

    Without it a missing module only removes its tests. Skips when the variable is unset.
    """
    want = [m.strip() for m in os.environ.get("SE_REQUIRE_ENV_MODULES", "").split(",") if m.strip()]
    if not want:
        pytest.skip("SE_REQUIRE_ENV_MODULES is not set; import failures are reported, not asserted")
    errs = se.import_errors()
    missing = {
        m: errs.get(m, "imported but registered no envs")
        for m in want
        if m in errs or not any(cls.__module__.endswith(f".{m}") for cls in se.registry.REGISTRY.values())
    }
    assert not missing, (
        "this run was launched to cover env modules it does not have, so every test for them "
        "collected zero cases:\n  " + "\n  ".join(f"{k}: {v}" for k, v in missing.items())
    )


def test_required_envs_are_all_reachable():
    """Every env named in ``SE_REQUIRE_ENVS`` is registered in this interpreter.

    No single interpreter holds every family's dependencies, so each job in the test matrix covers the
    envs assigned to it in ``tests/fixtures/regression/census/assignment.json``. Checked at registry
    level without building envs. Skips when the variable is unset.
    """
    want = [n.strip() for n in os.environ.get("SE_REQUIRE_ENVS", "").split(",") if n.strip()]
    if not want:
        pytest.skip("SE_REQUIRE_ENVS is not set, so there is no env assignment to check")
    errs = se.import_errors()
    missing = sorted(n for n in want if n not in se.registry.REGISTRY)
    assert not missing, (
        f"this run was launched to cover {len(want)} env(s) and {len(missing)} of them are not "
        f"registered in this interpreter, so every test parameterised over them collected zero "
        f"cases:\n  "
        + "\n  ".join(missing[:20])
        + (f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else "")
        + f"\n\nenv-module import errors here: {errs or 'none'}\n"
        + "If the assignment changed, edit tests/fixtures/regression/census/assignment.json (shared with "
        "the throughput census)."
    )


def test_describe_is_complete(env):
    d = env.describe()
    label = d.get("class_name") or d.get("id")
    if not has_render_rows(env):
        # A replay State has no flat render row, so describe() reports its State fields instead.
        assert d["render_state_names"] == [], f"{label}: render_state_names without a render row"
        live = {"/".join(k): v for k, v in env.get_state().items(True, True)}
        assert set(d["state_fields"]) == set(live), (
            f"{label}: describe() state_fields {sorted(set(d['state_fields']) ^ set(live))} disagree with get_state()"
        )
        for key, v in live.items():
            want = {"shape": list(v.shape[1:]), "dtype": str(v.dtype).removeprefix("torch.")}
            assert d["state_fields"][key] == want, f"{label}: {key} is {want} in get_state(), {d['state_fields'][key]}"
        return
    assert d["render_state_dim"] > 0, f"{label}: render_state_dim must be positive"
    assert len(d["render_state_names"]) == d["render_state_dim"], (
        f"{label}: {len(d['render_state_names'])} render_state_names for {d['render_state_dim']} dims"
    )


def test_render_state_roundtrip(env):
    if not has_render_rows(env):
        r = RS.check_state_roundtrip(env)
        assert not r["mismatched_leaves"], f"set_state(get_state()) is not the identity on {r['mismatched_leaves']}"
        return
    r = C.check_render_state_roundtrip(env)
    assert r["max_abs_err"] == 0.0, f"set_state(get_state()) is not the identity: {r}"


def test_every_state_dim_reaches_the_render(env):
    if not has_render_rows(env):
        r = RS.check_state_reaches_the_render(env)
        assert r["duplicate_frames"] == 0 and r["steps_change_the_frame"], (
            f"the State does not reach the frame: lanes reset from different seeds give "
            f"{r['duplicate_frames']} duplicate frame(s), and steps change the frame: {r['steps_change_the_frame']}"
        )
        return
    r = C.check_state_dims_matter(env)
    assert not r["dead_dims"], (
        f"state dims {r['dead_dims']} change nothing in the frame: either the state is not minimal "
        f"or the renderer ignores them"
    )


def test_batch_invariance(env):
    r = (C if has_render_rows(env) else RS).check_batch_invariance(env)
    assert r["max_abs_diff"] == 0, f"frame differs when rendered in a batch vs alone (max {r['max_abs_diff']}/255)"


def test_per_env_variation_is_per_env(env):
    r = (C if has_render_rows(env) else RS).check_per_env_variation(env)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert not r["collapsed_variations"], (
        f"variations {r['collapsed_variations']} collapse to one value across the batch; one value "
        f"per environment must give one appearance per environment"
    )


def test_style_is_state(env):
    r = (C if has_render_rows(env) else RS).check_style_is_state(env)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["max_abs_diff"] == 0, (
        f"a fresh env given the stored (state, style) renders differently (max {r['max_abs_diff']}/255); "
        f"the frame depends on something not in (state, style)"
    )


def test_default_style_is_the_official_appearance(env):
    r = (C if has_render_rows(env) else RS).check_default_style_is_official(env)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["max_abs_diff"] == 0, f"render(style=default_style()) differs from render() by {r['max_abs_diff']}/255"


def test_color_knobs_apply_the_requested_color(env):
    """Every RGB knob puts its color in the frame, except leaves the class declares not applicable.

    A declared leaf's lighting cannot reach unit gain, in the official renderer too (see
    ``conformance.color_exactness_applicability``). Its measured distance must still be recorded.
    """
    r = (C if has_render_rows(env) else RS).check_colors_apply_exactly(env)
    part = C.partition_colour_exactness(r)
    assert not part["asserted"], (
        f"colour knobs that did not put the requested colour in the frame: {part['asserted']}"
        + (
            f" (these are not covered by the class's colour-exactness declaration, which covers "
            f"{[d['probe'] for d in part['not_applicable']]})"
            if part["honoured"]
            else ""
        )
    )
    for declared in part["not_applicable"]:
        assert declared["best_abs_diff"] is not None, (
            f"{declared['probe']} is declared not_applicable but the probe recorded no "
            f"best_abs_diff; the declaration skips the assertion, not the measurement"
        )
    if part["not_applicable"]:
        print(f"declared not_applicable, measured and reported: {part['not_applicable']} ({part['why']})")


def test_determinism(env):
    assert (C if has_render_rows(env) else RS).check_determinism(env)["identical"], (
        "two renders of the same states differ"
    )


def test_backend_agreement(env):
    if not has_render_rows(env):
        pytest.skip(f"{type(env).__name__} renders a full State without a Scene2D, so there is one backend")
    r = C.check_backend_agreement(env)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["exact_frac"] >= 0.99, (
        f"the fused Triton kernel and the torch reference disagree on {100 * (1 - r['exact_frac']):.2f}% "
        f"of pixels (mean {r['mean_abs']:.3f}): {r}"
    )


def test_variations_all_do_something(env):
    r = (C if has_render_rows(env) else RS).check_variations_do_something(env)
    assert not r["inert_variations"], f"declared variations that change nothing: {r['inert_variations']}"


@pytest.mark.reference
def test_a_declared_task_object_is_drawn_iff_the_reference_draws_it(env):
    """A task object declared by a ``visibility.*`` leaf is drawn exactly when the reference draws it.

    ``interior_mask`` keeps only pixels with a flat 3x3 neighborhood, so an object on a textured floor
    has no interior pixels and the parity test cannot see it appear or disappear. This checks the
    object's own footprint instead.
    """
    # Default visibility is a property of the stable-worldmodel wrappers the DMC classes follow.
    reference = dmc_source_render.appearance_reference(type(env), env.size)
    r = C.check_task_object_visibility(env, n=16, seed=PARITY_STATE_SEED % 1000, reference=reference)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    print(f"\n{type(env).__name__} task objects: {r['objects']}")
    for path, o in r["objects"].items():
        assert o["footprint_px"] > 0, (
            f"{path} changes no pixel on any of {r['n_states']} states, so the env declares a "
            f"task-relevant object it cannot actually draw or hide"
        )
        assert o["per_row_max_abs_diff"] == 0, (
            f"{path} does not survive a per-row style: a batch with the knob alternating down the "
            f"rows differs from the two uniform renders by {o['per_row_max_abs_diff']}/255"
        )
        assert "why_no_margin" not in o, f"{path}: {o['why_no_margin']} — {o}"
        if o["presence_margin"] is None:
            continue  # no official renderer to agree with
        assert o["presence_margin"] >= TASK_OBJECT_PRESENCE_MARGIN, (
            f"inside {path}'s own {o['footprint_px']} pixels the declared appearance sits "
            f"{o['presence_mean_abs_declared']:.2f}/255 from the official frame and the flipped one "
            f"sits {o['presence_mean_abs_flipped']:.2f}/255, margin {o['presence_margin']:.4f} "
            f"against a floor of {TASK_OBJECT_PRESENCE_MARGIN}. A margin at or below zero means an "
            f"object is drawn that the reference does not draw, or the reverse: {o}"
        )


@pytest.mark.reference
def test_source_mse_matches_the_record(env):
    """Per-frame MSE against the source stays in the recorded band of ``tests/fixtures/source_mse.json``.

    The mean, the 95th percentile and the worst frame are measured on the recorded seeded render rows.
    A value above the band is a regression; a value far below it means the record is out of date.
    """
    cls = type(env)
    if not has_reference_renderer(cls):
        pytest.skip(f"{cls.__name__}: no official reference renderer to compare against")
    records = SM.load()
    if not records:
        pytest.skip(f"no MSE record; run {SM.RECORD}")
    rec = records.get(cls.__name__)
    assert rec is not None, (
        f"{cls.__name__} has a reference renderer but no MSE record; run {SM.RECORD} --update {cls.__name__}"
    )
    assert rec["size"] == env.size, f"{cls.__name__}: the record is at {rec['size']} px, the env renders {env.size} px"
    bad = reference_version_mismatch(rec.get("reference_versions"), _installed(rec.get("reference_versions") or {}))
    if bad:
        pytest.skip(f"{cls.__name__}: the record was measured against another reference: {'; '.join(bad)}")
    gpu = torch.cuda.get_device_name()
    if rec.get("gpu") != gpu:
        pytest.skip(f"{cls.__name__}: the record was measured on {rec.get('gpu')}, this host has {gpu}")
    try:
        got = C.check_source_mse(env, n=rec["n"], seed=rec["seed"])
    except ImportError as e:
        pytest.skip(f"{cls.__name__}: reference env not installed ({e})")
    print(f"\n{cls.__name__} vs {env.reference_id}: " + ", ".join(f"{k} {got[k]:.4f}" for k in SM.KEYS))
    problems = SM.problems(cls.__name__, got, rec)
    assert not problems, "\n".join(problems) + f"\nrecorded on {rec.get('gpu')}; per-frame MSE {got['per_frame']}"


@pytest.mark.reference
def test_gpu_step_tracks_the_reference_simulator(env):
    """A batched GPU simulator tracks its reference simulator within stated bounds.

    Contact solvers diverge chaotically, so the block is held to a median error (PushT: 0.68 px median
    after 100 steps, mean 20.7). The contact-free kinematic agent must agree to float precision.
    """
    # Require the class itself to define _step and opt in: pointing the gym_pusht-shaped reference at an
    # unrelated env can abort the interpreter.
    cls = type(env)
    if "_step" not in cls.__dict__ or not getattr(cls, "step_reference_checkable", False):
        pytest.skip(
            f"{cls.__name__}: no batched GPU simulator declared as checkable against a "
            f"reference (set step_reference_checkable = True on the class to opt in)"
        )
    try:
        r = C.check_step_matches_reference(env)
    except (ImportError, NotImplementedError) as e:
        pytest.skip(f"{type(env).__name__}: {e}")
    print(f"\n{type(env).__name__} step vs reference: {r}")
    assert r["agent_max_err"] < 1e-3, (
        f"the kinematic agent drifts {r['agent_max_err']:.3e} px from the reference integrator; with no "
        f"contacts this is an integration bug: {r}"
    )
    assert r["block_median_err"] < 2.0, (
        f"block position tracks the reference to a median {r['block_median_err']:.2f} px after "
        f"{r.get('steps', '?')} steps, above the 2.0 px bound: {r}"
    )


#: Largest history dependence accepted from an official MuJoCo renderer: one level on two pixels.
REFERENCE_DRAW_NOISE = {"max_delta": 1, "pixels": 2}


@pytest.mark.reference
def test_state_is_sufficient(env):
    """The official renderer gives the same frame for a state reached through two different histories.

    A property of the reference alone: the compact state determines the frame.
    """
    cls = type(env)
    if not has_reference_renderer(cls):
        pytest.skip(f"{cls.__name__}: no official reference renderer")
    try:
        # Use the process-wide cache: dropping a MuJoCo reference env releases an EGL context while
        # others are still live, which conformance.py forbids.
        ref = C.reference_env(cls, env.size)
    except ImportError as e:
        pytest.skip(f"{cls.__name__}: reference env not installed ({e})")
    # One reference env, two histories: the forward pass reaches state i from i-1, the reverse pass from
    # i+1. A second env would hold another unreleasable EGL context per cell.
    states = C.sample_render_rows(env, 16, rng=np.random.default_rng(PARITY_STATE_SEED))
    fa = cls.reference_render(ref, states)
    fb = cls.reference_render(ref, states[::-1])[::-1]
    diff = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(-1)
    d = int(diff.max())
    worst = int((diff > 0).sum(axis=(1, 2)).max())
    # MuJoCo's OpenGL output can move one pixel by one level between draws of an identical scene
    # (mjData and mjvScene bit-identical); REFERENCE_DRAW_NOISE tolerates exactly that.
    assert d <= REFERENCE_DRAW_NOISE["max_delta"] and worst <= REFERENCE_DRAW_NOISE["pixels"], (
        f"the official renderer gives different pixels for the same state depending on history "
        f"(max {d}/255 on up to {worst} pixels per frame); the compact state does not determine the "
        f"frame for {cls.__name__}"
    )
