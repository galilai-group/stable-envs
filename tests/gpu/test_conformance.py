"""GPU conformance suite, parameterised over the env registry.

Requires a CUDA device by design. The repo's product is a GPU renderer, so a test suite that passes on
CPU-only CI proves nothing about it; these are marked ``gpu`` and skip loudly rather than silently
weakening into something that can pass without the hardware.

Run:  pytest tests/gpu -v          (on a GPU node; see scripts/run_gpu_tests.sbatch)
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

ENV_NAMES = sorted(se.REGISTRY)

#: Parity thresholds. Interior pixels must be EXACT — that is the strong claim and it is achievable,
#: since interior pixels have no anti-aliasing ambiguity. All-pixel agreement is held to a much looser
#: bar on purpose: the reference rasterizers are aliased and disagree with any other rasterizer at
#: edges, so demanding exactness there would force us to reimplement pygame's coverage rules rather
#: than to render the scene correctly.
INTERIOR_EXACT_MIN = 1.0
ALL_EXACT_MIN = 0.90
MEAN_ABS_MAX = 4.0

#: How much CLOSER to the official frame the env's DECLARED visibility must be than the flipped one,
#: inside a task object's own footprint: ``1 - mean|d|(declared) / mean|d|(flipped)``. A NEW bound on
#: a NEW quantity -- it neither redefines nor relaxes anything above; see
#: ``test_a_declared_task_object_is_drawn_iff_the_reference_draws_it`` and
#: ``conformance.check_task_object_visibility`` for what the quantity is and why the interior bar
#: cannot answer it. 0.50 reads as "at least twice as close", which is the sentence a reader can check.
#: The ratio is POOLED over the whole state stack -- one number per object, not one per frame; the
#: only per-row quantity in that check is ``per_row_max_abs_diff``.
#:
#: RE-DERIVED AFTER A FAILURE, and recorded as such rather than presented as a first measurement. The
#: first version of this bound compared two WITHIN-8 AGREEMENT FRACTIONS and was floored at 0.25 from
#: CPU draws (worst 0.3966 on DMCReacherHard); the frozen 64-state corpus then measured 0.2422 on the
#: 3090 and the branch failed its own new bound on the certification card by 0.008. The renderer had
#: not changed -- the statistic was machine-dependent, because a task object's footprint is nearly all
#: silhouette (DMCReacherHard's target is ~15 px a frame) and the fraction of it landing within 8/255
#: is a statement about two rasterizers' edge rules: 0.60 on CPU, 0.26 on the GPU, identical code. The
#: quantity was replaced with the error ratio, which does not move that way, and the floor re-measured
#: on BOTH machines before being set.
#:
#: Set from the SPREAD over independent draws, clear of the worst, like every other floor here. Five
#: draws (four of 16 sampled states plus the frozen 64-state Step-4 corpus) on each of the four cells
#: that declare a ``visibility.*`` leaf, at 84 px, on BOTH the machine this is authored on and the
#: machine it is enforced on -- because the first draft of this bound was set from CPU numbers alone
#: and the GPU then measured under it:
#:
#:     cell                margin (CPU)      margin (RTX 3090)   declared mean|d|   flipped mean|d|
#:     DMCReacherEasy      0.9646 - 0.9676   0.9548 - 0.9575      5.2 -  7.2        156 - 161
#:     DMCReacherHard      0.8730 - 0.8806   0.8513 - 0.8604     17.0 - 20.3        132 - 144
#:     DMCFingerTurnEasy   0.8263 - 0.8506   0.9445 - 0.9503      5.1 - 16.9         97 - 104
#:     DMCFingerTurnHard   0.8199 - 0.8713   0.9090 - 0.9147      9.0 - 18.4         99 - 108
#:
#: The failure this exists to catch inverts the ratio: a renderer drawing an object the reference does
#: not draw puts ~140/255 in the numerator and ~15 in the denominator, i.e. a margin around -8. So the
#: pass and fail populations are not close and any bound between them separates them; 0.50 is chosen
#: to leave room for the CPU/GPU difference in edge pixels rather than to sit just under the worst.
#:
#: WHY NOT THE WITHIN-8 FRACTION, which this bound used in its first draft: on ``DMCReacherHard``'s
#: ~15 footprint pixels per frame that fraction is 0.60 on CPU and 0.26 on a 3090 for one unchanged
#: renderer, because the footprint is nearly all silhouette. It would have put the floor within 0.008
#: of the frozen corpus on the certification hardware. The mean error does not move that way.
TASK_OBJECT_PRESENCE_MARGIN = 0.50

#: The states the two reference-based tests draw. A CONSTANT, because both of them used to call
#: ``sample_states(n)`` with no generator, which takes the env's own ``rng`` — and the ``env`` fixture
#: is module-scoped, so the draw depended on how many tests had already consumed from that generator,
#: i.e. on test ORDER. Under ``pytest-randomly`` the order changes per run, so the same commit
#: produced a different comparison every time: a failure could not be reproduced and a pass was not
#: evidence. Measured on the swm nav family, one draw in eight disagrees on one or two interior pixels
#: (see ``swm_nav``'s module docstring), so roughly a fifth of full runs went red on a different env
#: each time with nothing to look at afterwards.
#:
#: Seeding is NOT the fix for that disagreement and must never be used as one — the floor an env is
#: held to is declared on the env, with a ``parity_note`` naming the mechanism, and this seed is not
#: chosen to clear anybody's floor. It is here so that a failure is a fact somebody can reproduce.
#: Deliberately not ``render_worker.STATE_SEED``: that one is a cross-process protocol constant, and
#: coupling the parity draw to it would make a change to either silently move the other.
PARITY_STATE_SEED = 20260811


@pytest.fixture(scope="module", params=ENV_NAMES)
def env(request):
    return se.make(request.param, batch=16, device="cuda")


def test_registry_is_not_empty():
    errs = se.envs.import_errors()
    assert ENV_NAMES, f"no envs registered; env-module import errors: {errs}"
    if errs:
        # Not a failure: a missing optional dependency legitimately removes one env. Printed so a
        # silently absent env is diagnosable instead of looking like a registration bug.
        print(f"\nenv modules that did not import: {errs}")


def test_required_env_modules_actually_imported():
    """``SE_REQUIRE_ENV_MODULES=dmcontrol,ogbench`` makes a missing family a FAILURE, not a print.

    An env module that does not import is normally a fact about the interpreter, not a defect, and
    ``test_registry_is_not_empty`` prints it. The cost of that being only a print was measured: the
    DM Control family's 16 envs were absent from the suite interpreter for a version skew
    (``mujoco 3.9.0`` against a ``dm_control`` that needs ``flex_bandwidth``), so every DM Control
    test — including the reference-based parity test, which is the repo's whole warrant — silently
    COLLECTED ZERO CASES. The suite was green and the family had no coverage at all, and nothing in
    the run said so louder than one line among a dozen other import notes.

    A run cannot know by itself which families it was meant to cover, so the launcher says: the
    DM Control sbatch exports the variable, and the run fails if the family it exists to test is not
    there. Unset, this is a no-op and the print remains the right level for everything else.
    """
    want = [m.strip() for m in os.environ.get("SE_REQUIRE_ENV_MODULES", "").split(",") if m.strip()]
    if not want:
        pytest.skip("SE_REQUIRE_ENV_MODULES is not set; import failures are reported, not asserted")
    errs = se.envs.import_errors()
    missing = {m: errs.get(m, "imported but registered no envs") for m in want
               if m in errs or not any(cls.__module__.endswith(f".{m}")
                                       for cls in se.REGISTRY.values())}
    assert not missing, (
        "this run was launched to cover env modules it does not have, so every test for them "
        "collected zero cases and the run would have been green anyway:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in missing.items()))


def test_required_envs_are_all_reachable():
    """``SE_REQUIRE_ENVS=PushT,DMCWalkerWalk`` fails the run if a named env is not reachable HERE.

    The env-name counterpart to the module check above, and the guard the test MATRIX rests on. No
    single interpreter can hold the whole library — robocasa pins ``mujoco==3.3.1``, ``mujoco_warp``
    needs ``>=3.11``, dm_control needs a build in between — so the release gate is a fan-out over the
    set cover in ``scratch/census/assignment.json``, and each job is told which envs it OWNS.

    Without this, the fan-out decays into the thing it replaces. An interpreter that quietly lost a
    dependency would skip its entire assignment with a per-env reason that is individually honest,
    and report green: which is precisely how a single-interpreter run showed 2580 passed / 648
    skipped while two thirds of the registry had never been through the gate at all.

    Checked at REGISTRY level rather than by constructing each env: construction needs a GPU and an
    EGL context and takes minutes across 147 envs, and a name absent from the registry is already
    conclusive — the tests parameterised over it cannot have run.
    """
    want = [n.strip() for n in os.environ.get("SE_REQUIRE_ENVS", "").split(",") if n.strip()]
    if not want:
        pytest.skip("SE_REQUIRE_ENVS is not set; this interpreter was not launched to cover a "
                    "named set of envs, so there is no assignment to hold it to")
    errs = se.envs.import_errors()
    missing = sorted(n for n in want if n not in se.REGISTRY)
    assert not missing, (
        f"this run was launched to cover {len(want)} env(s) and {len(missing)} of them are not "
        f"registered in this interpreter, so every test parameterised over them collected ZERO "
        f"cases and the run would have reported green:\n  "
        + "\n  ".join(missing[:20])
        + (f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else "")
        + f"\n\nenv-module import errors here: {errs or 'none'}\n"
        + "If the assignment is what changed, edit scratch/census/assignment.json — it is the one "
          "definition of the cover, shared with the throughput census, so that the envs that are "
          "MEASURED and the envs that are TESTED cannot drift apart.")


def test_relaxed_parity_floors_are_all_explained():
    """Every class that lowers a parity floor names the mechanism, whether or not it can be measured.

    ``test_render_parity_with_official_renderer`` already refuses a relaxed floor without a
    ``parity_note`` — and skips entirely when the reference package is not installed in this
    interpreter, which for this repo is the normal case for most families. So the rule that a
    tolerance must come with a stated mechanism was enforced exactly where it was least needed. This
    checks it on the class, over the whole registry, with no reference env and no GPU work: a floor
    lowered in a family whose reference is not importable here is caught in the interpreter it was
    lowered in.
    """
    bad = []
    for name, cls in sorted(se.REGISTRY.items()):
        floors = {k: getattr(cls, k, None) for k in ("parity_interior_min", "parity_all_exact_min")}
        lowered = {k: v for k, v in floors.items()
                   if v is not None and float(v) < (INTERIOR_EXACT_MIN if k == "parity_interior_min"
                                                    else ALL_EXACT_MIN)}
        if lowered and len(str(getattr(cls, "parity_note", "") or "").strip()) < 40:
            bad.append(f"{name} lowers {lowered} with no usable parity_note")
    assert not bad, (
        "a tolerance without a stated mechanism is indistinguishable from a bug being tolerated:\n  "
        + "\n  ".join(bad))


def test_describe_is_complete(env):
    d = env.describe()
    assert d["state_dim"] > 0, f"{d['name']}: state_dim must be positive"
    assert len(d["state_names"]) == d["state_dim"], (
        f"{d['name']}: {len(d['state_names'])} state_names for {d['state_dim']} dims — a state vector "
        f"whose fields are unnamed cannot be checked or reused by anyone")


def test_state_roundtrip(env):
    r = C.check_state_roundtrip(env)
    assert r["max_abs_err"] == 0.0, f"set_state(get_state()) is not the identity: {r}"


def test_every_state_dim_reaches_the_render(env):
    r = C.check_state_dims_matter(env)
    assert not r["dead_dims"], (
        f"state dims {r['dead_dims']} change nothing in the frame — either the state is not minimal "
        f"or the renderer ignores them; both make the state->frame claim false")


def test_batch_invariance(env):
    r = C.check_batch_invariance(env)
    assert r["max_abs_diff"] == 0, (
        f"frame differs when rendered in a batch vs alone (max {r['max_abs_diff']}/255) — every "
        f"throughput number in this repo assumes batching is semantically free")


def test_per_env_variation_is_per_env(env):
    r = C.check_per_env_variation(env)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert not r["collapsed_variations"], (
        f"variations {r['collapsed_variations']} collapse to one value across the batch — passing one "
        f"value per environment must give one appearance per environment, which is the whole point of "
        f"batched domain randomisation")


def test_style_is_state(env):
    r = C.check_style_is_state(env)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["max_abs_diff"] == 0, (
        f"a fresh env given the stored (state, style) renders differently (max {r['max_abs_diff']}/255) "
        f"— the frame depends on something a dataset would not have recorded, so 'store states, "
        f"regenerate live' does not hold for this env")


def test_default_style_is_the_official_appearance(env):
    r = C.check_default_style_is_official(env)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["max_abs_diff"] == 0, (
        f"render(style=default_style()) differs from render() by {r['max_abs_diff']}/255 — parity is "
        f"measured under the default, so these two paths disagreeing makes every parity number describe "
        f"an appearance nobody gets")


def test_color_knobs_apply_the_requested_color(env):
    """Every RGB knob puts its colour in the frame — except where the CLASS declares it cannot.

    The check is unchanged and still probes every leaf. What changed in Step 8 is what this assertion
    is entitled to conclude from the result: a leaf whose renderer applies a documented lighting law
    that cannot reach unit gain can never contain the literal triple, and the OFFICIAL renderer
    cannot either, so asserting it here measures the reference's shading model. Sami's ruling of
    2026-09-16; the declaration and its five required pieces of evidence live on the class and are
    read by ``conformance.color_exactness_applicability``.

    A declared leaf is NOT silently dropped: its measured distance is asserted to still exist, so a
    knob that comes off its primitive still fails something rather than disappearing into the
    declaration. Every leaf nobody declared is asserted exactly as before.
    """
    r = C.check_colors_apply_exactly(env)
    part = C.partition_colour_exactness(r)
    assert not part["asserted"], (
        f"colour knobs that did not put the requested colour in the frame: {part['asserted']} — "
        f"a knob wired to the wrong primitive, scaled, or channel-swapped still passes the weaker "
        f"'does it change anything' check"
        + (f" (these are NOT covered by the class's colour-exactness declaration, which covers "
           f"{[d['probe'] for d in part['not_applicable']]})" if part["honoured"] else ""))
    for declared in part["not_applicable"]:
        assert declared["best_abs_diff"] is not None, (
            f"{declared['probe']} is declared not_applicable but the probe recorded no "
            f"best_abs_diff — the declaration is only allowed to stop the ASSERTION, never the "
            f"measurement, or a wiring regression on this leaf would move no number at all")
    if part["not_applicable"]:
        print(f"declared not_applicable, measured and reported: {part['not_applicable']} "
              f"({part['why']})")


def test_determinism(env):
    assert C.check_determinism(env)["identical"], "two renders of the same states differ"


def test_backend_agreement(env):
    r = C.check_backend_agreement(env)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["exact_frac"] >= 0.99, (
        f"the fused Triton kernel and the torch reference disagree on {100*(1-r['exact_frac']):.2f}% "
        f"of pixels (mean {r['mean_abs']:.3f}) — they must draw the same picture: {r}")


def test_variations_all_do_something(env):
    r = C.check_variations_do_something(env)
    assert not r["inert_variations"], (
        f"declared variations that change nothing: {r['inert_variations']} — a knob in the space but "
        f"not in scene() advertises support that does not exist")


@pytest.mark.reference
def test_a_declared_task_object_is_drawn_iff_the_reference_draws_it(env):
    """The complement to the interior bar, for the one thing the interior bar cannot see.

    ``test_render_parity_with_official_renderer`` above is the strong claim, and on a scene whose
    floor is textured it is silent about whole OBJECTS: ``interior_mask`` keeps only pixels with a
    flat 3x3 neighbourhood in the reference, so an object that lands on a checker floor contributes
    no interior pixels and can appear or disappear without moving ``interior_exact_frac``. Measured,
    not supposed — two DM Control reacher renders that differ by an entire red disc report interior
    exactness identical to sixteen digits and the same ``interior_px``
    (``examples/results/step5/reacher/parity_probe.json``), and the frame's ``interior_frac`` at 84 px
    is 0.000000, so the assertion there does not even apply.

    Nothing about the interior metric is changed to fix that. This asserts a SEPARATE bound on the
    population the interior mask throws away — the object's own pixels — and only for envs that
    declare a ``visibility.*`` leaf, i.e. that have named a task-relevant object at all.
    """
    r = C.check_task_object_visibility(env, n=16, seed=PARITY_STATE_SEED % 1000)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    print(f"\n{type(env).__name__} task objects: {r['objects']}")
    for path, o in r["objects"].items():
        assert o["footprint_px"] > 0, (
            f"{path} changes no pixel on any of {r['n_states']} states, so the env declares a "
            f"task-relevant object it cannot actually draw or hide")
        assert o["per_row_max_abs_diff"] == 0, (
            f"{path} does not survive a PER-ROW style: a batch with the knob alternating down the "
            f"rows differs from the two uniform renders by {o['per_row_max_abs_diff']}/255. "
            f"check_per_env_variation cannot ask this of a Discrete leaf, which is why it is here")
        assert "why_no_margin" not in o, (
            f"{path}: {o['why_no_margin']} — {o}")
        if o["presence_margin"] is None:
            continue                                   # no official renderer to agree with
        assert o["presence_margin"] >= TASK_OBJECT_PRESENCE_MARGIN, (
            f"inside {path}'s own {o['footprint_px']} pixels our DECLARED appearance sits "
            f"{o['presence_mean_abs_declared']:.2f}/255 from the official frame and the FLIPPED one "
            f"sits {o['presence_mean_abs_flipped']:.2f}/255 — margin {o['presence_margin']:.4f} "
            f"against a floor of {TASK_OBJECT_PRESENCE_MARGIN}. A margin at or below zero means we "
            f"draw an object the reference does not draw (or the reverse), which is exactly what the "
            f"interior-pixel metric is blind to on a textured surface: {o}")


def test_fidelity_tier_is_fully_declared(env):
    """An env is parity or visual, and either way it has said everything that tier requires.

    ``register`` already refuses a half-declared tier, so this cannot fail for a registered env — which
    is the point of asserting it here anyway: it is the line that fails, with the reason, if that
    enforcement is ever weakened or bypassed. The tier decides which of the two parity tests below
    applies, so an env with an under-specified tier would silently be checked by neither.
    """
    from stable_envs.registry import check_fidelity_declaration
    check_fidelity_declaration(type(env))
    assert env.describe()["fidelity_tier"] == env.fidelity_tier, (
        "describe() must report the tier — it is how a consumer finds out that these frames are not "
        "the official pixels without reading the source")


@pytest.mark.reference
def test_render_parity_with_official_renderer(env):
    """The claim, measured: our frame vs the OFFICIAL renderer on states the env actually reaches.

    THIS TEST IS NOT RELAXED FOR THE VISUAL TIER. A visual-tier env is skipped here and asserted by
    ``test_visual_tier_matches_its_declared_fidelity`` instead, against its own declared numbers. The
    alternative — one test with a per-env threshold spanning "every interior pixel is bit-identical"
    and "the picture looks right" — is how a strong claim becomes a weak one: the floor would be read
    as the contract, and the next env that could not reach 1.0 would move it again.
    """
    cls = type(env)
    if env.fidelity_tier == "visual":
        pytest.skip(f"{cls.__name__} declares the VISUAL tier: it is a different render of the same "
                    f"scene and is asserted by test_visual_tier_matches_its_declared_fidelity")
    if not hasattr(cls, "reference_env") or env.reference_id is None:
        pytest.skip(f"{cls.__name__}: no official reference renderer to compare against")
    try:
        # Through the process-wide cache, NOT cls.reference_env directly. A reference env bound
        # to a local here is dropped the moment the test returns, and for the MuJoCo families
        # that RELEASES an EGL context while every other env still holds live ones — the one
        # thing conformance.py's rule forbids. Instrumented on the full 44-env run: 83 contexts
        # created and 50 released mid-run, all of them here and in test_state_is_sufficient.
        ref = C.reference_env(cls, env.size)
    except ImportError as e:
        pytest.skip(f"{cls.__name__}: reference env not installed ({e})")

    states = env.sample_states(64, rng=np.random.default_rng(PARITY_STATE_SEED))
    official = cls.reference_render(ref, states)
    ours = env.render(states).cpu().numpy()
    assert ours.shape == official.shape, f"shape {ours.shape} vs official {official.shape}"

    r = C.pixel_agreement(ours, official)
    print(f"\n{cls.__name__} vs {env.reference_id}: {r}")
    floor = float(getattr(env, "parity_interior_min", INTERIOR_EXACT_MIN))
    if floor < 1.0:
        assert getattr(env, "parity_note", ""), (
            f"{cls.__name__} relaxes parity_interior_min to {floor} but gives no parity_note — a "
            f"tolerance without a stated mechanism is indistinguishable from a bug being tolerated")
        print(f"  relaxed to {floor}: {env.parity_note}")
    if r["interior_exact_frac"] is not None:
        assert r["interior_exact_frac"] >= floor, (
            f"{r.get('interior_bad_px')} of {r.get('interior_px')} interior pixels disagree "
            f"(max |delta| {r['max_abs']}/255), i.e. {100*r['interior_exact_frac']:.5f}% exact against "
            f"a floor of {100*floor:.5f}% — an interior disagreement is normally a real "
            f"geometry/colour error rather than anti-aliasing. The states are drawn from "
            f"default_rng({PARITY_STATE_SEED}), so this is reproducible; if a HANDFUL of pixels "
            f"disagree by 1/255 the next question is whether they sit on a quantisation boundary "
            f"inside a smoothly shaded region, which the 3x3 interior mask cannot see (see "
            f"conformance.interior_mask). That case is declared with parity_interior_min AND a "
            f"parity_note naming the mechanism, measured over many draws — never by moving the "
            f"seed: {r}")
    # The all-pixel bar is shared unless an env DECLARES a different one, with a note, the same way the
    # interior bar works. The distinction the two make is the point: interior exactness describes the
    # renderer, all-pixel exactness describes the renderer weighted by how much of the frame moves — a
    # scene whose robot covers 13% of the frame cannot reach the number a scene whose robot covers 9%
    # reaches, on the identical backend with the identical per-pixel residual. Nothing here loosens the
    # shared bar for anyone: the default is still ALL_EXACT_MIN.
    all_floor = getattr(env, "parity_all_exact_min", None)
    all_floor = ALL_EXACT_MIN if all_floor is None else float(all_floor)
    if all_floor < ALL_EXACT_MIN:
        assert getattr(env, "parity_note", ""), (
            f"{cls.__name__} relaxes parity_all_exact_min to {all_floor} but gives no parity_note")
        print(f"  all-pixel floor relaxed to {all_floor}")
    assert r["exact_frac"] >= all_floor, (
        f"only {100*r['exact_frac']:.1f}% of pixels exact (floor {100*all_floor:.1f}%): {r}")
    assert r["mean_abs"] <= MEAN_ABS_MAX, f"mean |diff| {r['mean_abs']:.3f}/255 too high: {r}"


@pytest.mark.reference
def test_visual_tier_matches_its_declared_fidelity(env):
    """A VISUAL-tier env must hit the numbers it declares. Its own thresholds, separately checkable.

    Three assertions, on three quantities that fail for different reasons, because one aggregate would
    let a real regression hide: ``mean_abs`` catches a global shift (a light, a material, a colour
    resolution rule), ``within8_frac`` catches a REGION going wrong while the mean stays small, and
    ``ssim`` catches structure — a surface losing its texture moves SSIM and can leave both of the
    others where they were.

    The thresholds live on the env, not here, and they are measured before they are declared; the env
    also carries the measurements themselves so the margin between "what it does" and "what it must
    do" is visible rather than implied. What this test deliberately does NOT do is assert anything
    about interior-pixel exactness: the visual tier does not claim it, and asserting a low floor on it
    would turn a number that is reported for honesty into a target.
    """
    cls = type(env)
    if env.fidelity_tier != "visual":
        pytest.skip(f"{cls.__name__} is a parity env; test_render_parity_with_official_renderer "
                    f"applies to it")
    try:
        r = C.check_visual_fidelity(env)
    except ImportError as e:
        pytest.skip(f"{cls.__name__}: reference env not installed ({e})")
    vf = env.visual_fidelity
    print(f"\n{cls.__name__} vs {env.reference_id} [VISUAL TIER]: {r}")
    print(f"  declared: {vf}")
    assert r["mean_abs"] <= vf["mean_abs_max"], (
        f"mean |diff| {r['mean_abs']:.3f}/255 exceeds the declared ceiling {vf['mean_abs_max']} — the "
        f"visual tier is a measured claim, and this env is no longer meeting it: {r}")
    assert r["within8_frac"] >= vf["within8_min"], (
        f"only {100*r['within8_frac']:.1f}% of pixels are within 8/255 (declared floor "
        f"{100*vf['within8_min']:.1f}%) — a region of the frame is wrong even though the mean may not "
        f"show it: {r}")
    assert r["ssim"] >= vf["ssim_min"], (
        f"SSIM {r['ssim']:.4f} below the declared floor {vf['ssim_min']} — the frame has lost "
        f"structure the official one has, which is the failure a pixel mean is worst at seeing: {r}")


@pytest.mark.reference
def test_a_learned_decoder_beats_the_render_it_replaces(env):
    """A network in the render path must be better than not having one. On the same states, live.

    The measured reason this is a gate and not a preference: a corrector applied where it does not
    belong is WORSE than a plain resize — 5.94/255 on a camera 40 cm from the one it was fitted on
    against 6.14 for doing nothing and 4.75 for the render being replaced, and 11.35 against 4.39 on a
    wrist camera. It fails by painting plausible texture that the official frame does not contain, so
    the three comparisons are made separately: **the mean does not catch it and SSIM and within-8
    do**. On a different kitchen the mean TIES (33.19 against 32.85) while SSIM falls from 0.792 to
    0.763, and that is the whole failure.

    ``examples/rendernet/train_corrector.py`` runs the same three comparisons before it will write a
    checkpoint, and ``tests/gpu/test_rendernet.py`` asserts them on the declaration. This one asserts
    them on frames, in this interpreter, against this machine's official renderer — which is the only
    one of the three that can notice a render path that has drifted since the weights were sealed.
    """
    cls = type(env)
    if getattr(env, "render_decoder", "renderer") != "learned":
        pytest.skip(f"{cls.__name__} draws its frames with a renderer; there is nothing to compare "
                    f"a learned decoder against")
    base = cls.__name__.removesuffix("Corrected")
    assert base in se.REGISTRY, (
        f"{cls.__name__} has no plain sibling {base}: a learned decoder is a second way to draw a "
        f"cell that already exists, and the claim it makes is relative to the first one")
    try:
        ref = C.reference_env(cls, env.size)
    except ImportError as e:
        pytest.skip(f"{cls.__name__}: reference env not installed ({e})")

    plain = se.make(base, batch=env.B, size=env.size, device="cuda")
    states = env.sample_states(64, rng=np.random.default_rng(PARITY_STATE_SEED))
    official = cls.reference_render(ref, states)
    got, was = env.render(states).cpu().numpy(), plain.render(states).cpu().numpy()
    a, b = C.pixel_agreement(got, official), C.pixel_agreement(was, official)
    a["ssim"], b["ssim"] = C.ssim(got, official), C.ssim(was, official)
    print(f"\n{cls.__name__} [LEARNED] {a}\n  {base} [renderer] {b}")
    worse = []
    for k, higher_is_better in (("mean_abs", False), ("within8_frac", True), ("ssim", True)):
        ok = a[k] > b[k] if higher_is_better else a[k] < b[k]
        if not ok:
            worse.append(f"{k}: {a[k]:.4f} vs the plain render's {b[k]:.4f}")
    assert not worse, (
        f"{cls.__name__} runs a network over {base}'s frames and comes out no better on: "
        + "; ".join(worse) + ". A learned decoder that does not beat the renderer it replaces is a "
        "hallucination budget spent for nothing, and the way it fails is invisible in the picture.")


@pytest.mark.reference
def test_a_learned_decoder_holds_up_under_its_declared_variation(env):
    """The measurement that a default-appearance declaration cannot make, asserted separately.

    A rasterizer that agrees with the official one at one camera agrees at another — same arithmetic,
    pointed somewhere else — so for a renderer the default-appearance number describes the whole
    variation space. A learned decoder does not generalise that way: fitted at one camera pose it
    scores 3.25/255 there and 9.06 under the env's own declared ±8° of pan, which is worse than not
    correcting at all (8.91) AND worse than the render it replaces (7.93). Every one of those three
    numbers is measured. The class therefore declares ``variation_fidelity`` — ``register`` refuses a
    learned decoder without it — and this is where it is checked.
    """
    if getattr(env, "render_decoder", "renderer") != "learned":
        pytest.skip("a renderer's agreement does not depend on where the camera is pointed; this "
                    "measurement exists for decoders that were FITTED at one appearance")
    vf = env.variation_fidelity
    try:
        r = C.check_visual_fidelity_under_variation(env, n=64, seed=0)
    except ImportError as e:
        pytest.skip(f"{type(env).__name__}: reference env not installed ({e})")
    assert "skipped" not in r, (
        f"{type(env).__name__} carries a variation_fidelity declaration that cannot be re-measured "
        f"here ({r['skipped']}). A declaration nothing can check is the shape of the failure this "
        f"whole path is guarding against.")
    print(f"\n{type(env).__name__} under its declared variation: {r}\n  declared: {vf}")
    assert r["mean_abs"] <= vf["mean_abs_max"], (
        f"mean |diff| {r['mean_abs']:.3f}/255 under the declared variation exceeds the declared "
        f"ceiling {vf['mean_abs_max']} — the network is being shown appearances it does not cover: {r}")
    assert r["within8_frac"] >= vf["within8_min"], (
        f"only {100*r['within8_frac']:.1f}% of pixels within 8/255 under the declared variation "
        f"(floor {100*vf['within8_min']:.1f}%): {r}")
    assert r["ssim"] >= vf["ssim_min"], (
        f"SSIM {r['ssim']:.4f} under the declared variation is below the declared floor "
        f"{vf['ssim_min']} — structure is the first thing a decoder loses off its training "
        f"distribution, and it is the thing the pixel mean does not report: {r}")


@pytest.mark.reference
def test_gpu_step_tracks_the_reference_simulator(env):
    """Where an env ships a batched GPU simulator, it must track the reference within a stated bound.

    Deliberately a DIFFERENT and weaker claim than render parity, and the two must never be conflated.
    A renderer can be bit-exact against the official one; a reimplemented contact solver cannot, because
    contact is chaotic and any two solvers separate exponentially once they disagree by an ulp. So this
    asserts a median, not a maximum: PushT tracks pymunk to a median 0.68 px after 100 steps while the
    mean is 20.7, because a few episodes diverge outright. Asserting the mean would either fail on
    correct physics or force a tolerance loose enough to hide a real error.

    The kinematic bound is the honest control: the agent is driven by an integrator with no contacts, so
    it must agree to float precision (measured 9.5e-05 px). If that ever regresses, the discrepancy is
    an integration bug, not solver divergence.
    """
    # ``hasattr`` is too loose a guard. ``check_step_matches_reference`` builds a gym_pusht-shaped CPU
    # reference, and pointing it at an env with an unrelated ``step`` does not raise — it can abort the
    # interpreter, which no try/except can catch and which takes the whole suite down with exit 134
    # rather than one failure. So require the class to DECLARE step itself and to be checkable.
    cls = type(env)
    if "step" not in cls.__dict__ or not getattr(cls, "step_reference_checkable", False):
        pytest.skip(f"{cls.__name__}: no batched GPU simulator declared as checkable against a "
                    f"reference (set step_reference_checkable = True on the class to opt in)")
    try:
        r = C.check_step_matches_reference(env)
    except (ImportError, NotImplementedError) as e:
        pytest.skip(f"{type(env).__name__}: {e}")
    print(f"\n{type(env).__name__} step vs reference: {r}")
    assert r["agent_max_err"] < 1e-3, (
        f"the kinematic agent drifts {r['agent_max_err']:.3e} px from the reference integrator — with no "
        f"contacts acting on it this is an integration bug, not solver divergence: {r}")
    assert r["block_median_err"] < 2.0, (
        f"block position tracks the reference to a median {r['block_median_err']:.2f} px after "
        f"{r.get('steps', '?')} steps, above the 2.0 px bound: {r}")


@pytest.mark.reference
def test_state_is_sufficient(env):
    """Two reference envs driven to the same state must render the same frame.

    This is the thesis stated as a property of the REFERENCE, independent of our renderer: if the
    official env can reach the same pixels from the same compact state via two different histories,
    then the state really does determine the frame and nothing hidden is carried along.
    """
    cls = type(env)
    if not hasattr(cls, "reference_env") or env.reference_id is None:
        pytest.skip(f"{cls.__name__}: no official reference renderer")
    try:
        # Through the process-wide cache, NOT cls.reference_env directly. A reference env bound
        # to a local here is dropped the moment the test returns, and for the MuJoCo families
        # that RELEASES an EGL context while every other env still holds live ones — the one
        # thing conformance.py's rule forbids. Instrumented on the full 44-env run: 83 contexts
        # created and 50 released mid-run, all of them here and in test_state_is_sufficient.
        ref = C.reference_env(cls, env.size)
    except ImportError as e:
        pytest.skip(f"{cls.__name__}: reference env not installed ({e})")
    # ONE reference env, driven through two different histories — not two envs. The difference that
    # matters to this test is the history preceding each state, and a single env gives that: the
    # forward pass reaches state i from state i-1, the reverse pass reaches it from state i+1.
    #
    # Two envs were also a resource bug. MuJoCo reference envs hold EGL contexts that cannot be closed
    # (releasing one corrupts rendering in the others), so with 25 OGBench cells this test alone kept
    # 50 live contexts and every maze env failed with 69-90/255 differences that looked exactly like a
    # state-sufficiency violation. The renderer was fine; the test was exhausting the driver.
    states = env.sample_states(16, rng=np.random.default_rng(PARITY_STATE_SEED))
    fa = cls.reference_render(ref, states)
    fb = cls.reference_render(ref, states[::-1])[::-1]
    d = int(np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max())
    assert d == 0, (
        f"the official renderer gives different pixels for the same state depending on history "
        f"(max {d}/255) — the compact state does NOT determine the frame for {cls.__name__}, and no "
        f"GPU renderer can fix that")
