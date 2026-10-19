"""GPU conformance suite, parameterised over the env registry.

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

import stable_envs as se
from stable_envs.testing import conformance as C
from tests.helpers import pixel_tolerances as PT
from tests.helpers.reference import dmc_source_render
from tests.helpers.reference.certification import has_reference_renderer

ENV_NAMES = sorted(se.registry.REGISTRY)

#: Minimum ``1 - mean|d|(declared) / mean|d|(flipped)`` inside a task object's footprint, pooled over
#: the state stack: the declared visibility must be at least twice as close to the official frame as
#: the flipped one. Measured margins over five draws per cell at 84 px:
#:
#:     cell                margin (CPU)      margin (RTX 3090)   declared mean|d|   flipped mean|d|
#:     DMCReacherEasy      0.9646 - 0.9676   0.9548 - 0.9575      5.2 -  7.2        156 - 161
#:     DMCReacherHard      0.8730 - 0.8806   0.8513 - 0.8604     17.0 - 20.3        132 - 144
#:     DMCFingerTurnEasy   0.8263 - 0.8506   0.9445 - 0.9503      5.1 - 16.9         97 - 104
#:     DMCFingerTurnHard   0.8199 - 0.8713   0.9090 - 0.9147      9.0 - 18.4         99 - 108
#:
#: Drawing an object the reference does not draw gives a margin around -8. A within-8/255 fraction is
#: not used because on footprints that are nearly all silhouette it differs between CPU and GPU.
TASK_OBJECT_PRESENCE_MARGIN = 0.50

#: Seed for the states the reference-based tests draw, so a failure is reproducible regardless of test
#: order. It is not tuned to clear any floor, and is separate from ``render_worker.STATE_SEED``.
PARITY_STATE_SEED = 20260811


@pytest.fixture(scope="module", params=ENV_NAMES)
def env(request):
    return se.make(request.param, num_envs=16, device="cuda")


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


def test_relaxed_parity_floors_are_all_explained():
    """Every registered class with a lowered exact-pixel floor gives a ``parity_note`` naming the mechanism.

    Checked on the class, so it applies even where the reference package is not installed.
    """
    bad = []
    for name, cls in sorted(se.registry.REGISTRY.items()):
        lowered = {k: v for k, v in PT.exact_floors(name).items() if v < PT.EXACT[k]}
        if lowered and len(str(getattr(cls, "parity_note", "") or "").strip()) < 40:
            bad.append(f"{name} lowers {lowered} with no usable parity_note")
    assert not bad, "parity floors lowered without a parity_note:\n  " + "\n  ".join(bad)


def test_describe_is_complete(env):
    d = env.describe()
    label = d.get("class_name") or d.get("id")
    assert d["render_state_dim"] > 0, f"{label}: render_state_dim must be positive"
    assert len(d["render_state_names"]) == d["render_state_dim"], (
        f"{label}: {len(d['render_state_names'])} render_state_names for {d['render_state_dim']} dims"
    )


def test_render_state_roundtrip(env):
    r = C.check_render_state_roundtrip(env)
    assert r["max_abs_err"] == 0.0, f"set_state(get_state()) is not the identity: {r}"


def test_every_state_dim_reaches_the_render(env):
    r = C.check_state_dims_matter(env)
    assert not r["dead_dims"], (
        f"state dims {r['dead_dims']} change nothing in the frame: either the state is not minimal "
        f"or the renderer ignores them"
    )


def test_batch_invariance(env):
    r = C.check_batch_invariance(env)
    assert r["max_abs_diff"] == 0, f"frame differs when rendered in a batch vs alone (max {r['max_abs_diff']}/255)"


def test_per_env_variation_is_per_env(env):
    r = C.check_per_env_variation(env)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert not r["collapsed_variations"], (
        f"variations {r['collapsed_variations']} collapse to one value across the batch; one value "
        f"per environment must give one appearance per environment"
    )


def test_style_is_state(env):
    r = C.check_style_is_state(env)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["max_abs_diff"] == 0, (
        f"a fresh env given the stored (state, style) renders differently (max {r['max_abs_diff']}/255); "
        f"the frame depends on something not in (state, style)"
    )


def test_default_style_is_the_official_appearance(env):
    r = C.check_default_style_is_official(env)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["max_abs_diff"] == 0, f"render(style=default_style()) differs from render() by {r['max_abs_diff']}/255"


def test_color_knobs_apply_the_requested_color(env):
    """Every RGB knob puts its colour in the frame, except leaves the class declares not applicable.

    A declared leaf's lighting cannot reach unit gain, in the official renderer too (see
    ``conformance.color_exactness_applicability``). Its measured distance must still be recorded.
    """
    r = C.check_colors_apply_exactly(env)
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
    assert C.check_determinism(env)["identical"], "two renders of the same states differ"


def test_backend_agreement(env):
    r = C.check_backend_agreement(env)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["exact_frac"] >= 0.99, (
        f"the fused Triton kernel and the torch reference disagree on {100 * (1 - r['exact_frac']):.2f}% "
        f"of pixels (mean {r['mean_abs']:.3f}): {r}"
    )


def test_variations_all_do_something(env):
    r = C.check_variations_do_something(env)
    assert not r["inert_variations"], f"declared variations that change nothing: {r['inert_variations']}"


@pytest.mark.reference
def test_a_declared_task_object_is_drawn_iff_the_reference_draws_it(env):
    """A task object declared by a ``visibility.*`` leaf is drawn exactly when the reference draws it.

    ``interior_mask`` keeps only pixels with a flat 3x3 neighbourhood, so an object on a textured floor
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


def test_fidelity_declaration_matches_the_tolerance_table(env):
    """A class declares ``visual_fidelity`` if and only if the tolerance table compares it by agreement."""
    from stable_envs.registry import check_fidelity_declaration

    cls = type(env)
    check_fidelity_declaration(cls)
    assert bool(env.visual_fidelity) == (PT.agreement_limits(cls.__name__) is not None), (
        f"{cls.__name__}: visual_fidelity and tests/helpers/pixel_tolerances.AGREEMENT disagree"
    )


@pytest.mark.reference
def test_render_parity_with_official_renderer(env):
    """Frames match the official renderer on sampled reachable states, to the exact-pixel floors.

    Classes in ``pixel_tolerances.AGREEMENT`` are skipped here and checked by
    ``test_render_agreement_with_official_renderer``.
    """
    cls = type(env)
    if PT.agreement_limits(cls.__name__) is not None:
        pytest.skip(f"{cls.__name__} is compared by agreement in test_render_agreement_with_official_renderer")
    if not has_reference_renderer(cls):
        pytest.skip(f"{cls.__name__}: no official reference renderer to compare against")
    try:
        # Use the process-wide cache: dropping a MuJoCo reference env releases an EGL context while
        # others are still live, which conformance.py forbids.
        ref = C.reference_env(cls, env.size)
    except ImportError as e:
        pytest.skip(f"{cls.__name__}: reference env not installed ({e})")

    states = C.sample_render_rows(env, 64, rng=np.random.default_rng(PARITY_STATE_SEED))
    official = cls.reference_render(ref, states)
    ours = env._render(states).cpu().numpy()
    assert ours.shape == official.shape, f"shape {ours.shape} vs official {official.shape}"

    r = C.pixel_agreement(ours, official)
    print(f"\n{cls.__name__} vs {env.reference_id}: {r}")
    floors = PT.exact_floors(cls.__name__)
    floor = floors["interior_exact_min"]
    if floor < 1.0:
        assert getattr(env, "parity_note", ""), (
            f"{cls.__name__} relaxes interior_exact_min to {floor} but gives no parity_note"
        )
        print(f"  relaxed to {floor}: {env.parity_note}")
    if r["interior_exact_frac"] is not None:
        assert r["interior_exact_frac"] >= floor, (
            f"{r.get('interior_bad_px')} of {r.get('interior_px')} interior pixels disagree "
            f"(max |delta| {r['max_abs']}/255), i.e. {100 * r['interior_exact_frac']:.5f}% exact against "
            f"a floor of {100 * floor:.5f}%, on states from default_rng({PARITY_STATE_SEED}). Interior "
            f"disagreement is normally a geometry or colour error. A few 1/255 pixels may be a "
            f"quantisation boundary in a smoothly shaded region (see conformance.interior_mask), which "
            f"is declared in tests/helpers/pixel_tolerances.py with a parity_note: {r}"
        )
    # All-pixel exactness also depends on how much of the frame moves, so a class can lower it.
    all_floor = floors["all_exact_min"]
    if all_floor < PT.EXACT["all_exact_min"]:
        assert getattr(env, "parity_note", ""), (
            f"{cls.__name__} relaxes all_exact_min to {all_floor} but gives no parity_note"
        )
        print(f"  all-pixel floor relaxed to {all_floor}")
    assert r["exact_frac"] >= all_floor, (
        f"only {100 * r['exact_frac']:.1f}% of pixels exact (floor {100 * all_floor:.1f}%): {r}"
    )
    assert r["mean_abs"] <= floors["mean_abs_max"], f"mean |diff| {r['mean_abs']:.3f}/255 too high: {r}"


@pytest.mark.reference
def test_render_agreement_with_official_renderer(env):
    """A class in ``pixel_tolerances.AGREEMENT`` meets its ``mean_abs``, ``within8_frac`` and ``ssim`` limits.

    ``mean_abs`` catches a global shift, ``within8_frac`` a wrong region, ``ssim`` lost structure.
    Interior-pixel exactness is not asserted for these classes.
    """
    cls = type(env)
    vf = PT.agreement_limits(cls.__name__)
    if vf is None:
        pytest.skip(f"{cls.__name__} is compared exactly in test_render_parity_with_official_renderer")
    try:
        r = C.check_visual_fidelity(env)
    except ImportError as e:
        pytest.skip(f"{cls.__name__}: reference env not installed ({e})")
    print(f"\n{cls.__name__} vs {env.reference_id}: {r}")
    print(f"  limits: {vf}; declared measurement: {env.visual_fidelity}")
    assert r["mean_abs"] <= vf["mean_abs_max"], (
        f"mean |diff| {r['mean_abs']:.3f}/255 exceeds the declared ceiling {vf['mean_abs_max']}: {r}"
    )
    assert r["within8_frac"] >= vf["within8_min"], (
        f"only {100 * r['within8_frac']:.1f}% of pixels are within 8/255 (declared floor "
        f"{100 * vf['within8_min']:.1f}%): {r}"
    )
    assert r["ssim"] >= vf["ssim_min"], f"SSIM {r['ssim']:.4f} below the declared floor {vf['ssim_min']}: {r}"


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
