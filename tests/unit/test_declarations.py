"""What an env is allowed to CLAIM: the fidelity tier, the render decoder, and what registration refuses.

Two declarations decide how the rest of the suite judges an env, and both of them are ways to make a
strong assertion stop applying. ``fidelity_tier = "visual"`` removes the parity gate;
``render_decoder = "learned"`` says a neural checkpoint drew part of the frame. ``core/env.py`` pays for
each at the moment it is adopted — measured numbers, thresholds, a stated mechanism, and words in the
class docstring that a reader will actually see — and ``register`` enforces that.

WHY THIS FILE EXISTS. Until now the whole ``render_decoder`` / ``variation_fidelity`` contract was
exercised in exactly one place: ``tests/gpu/test_rendernet.py``, parameterised over the classes with a
sealed corrector on the machine. That means it needed a GPU, a RoboCasa install and a trained
checkpoint — and on every interpreter without all three it collected zero cases and asserted nothing,
which is precisely the coverage hole this suite has twice been caught in. The enforcement itself is
twelve lines of string and dict checking and needs none of that.

So the refusals are asserted here against SYNTHETIC classes, one per rule, on CPU. The synthetic class
is the honest instrument: a rule is only proved by a class that breaks it, and no real env in this
repository breaks any of them (that is what ``register`` is for). The registry-wide half of the file
then asserts that the real classes still satisfy the rules — the tripwire for the enforcement being
bypassed or weakened, on every env rather than on the ones that happen to have weights here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import stable_envs as se
from stable_envs import spaces
from stable_envs.core.env import (REQUIRED_VISUAL_FIDELITY, StableEnv, check_decoder_declaration,
                                  check_fidelity_declaration, format_registry, make, register,
                                  registry_table)

#: A complete, self-consistent measurement. Individual tests below remove exactly one thing from it,
#: which is what makes each of them a test of one rule rather than of the whole block.
VF = {"mean_abs": 2.45, "within8_frac": 0.935, "ssim": 0.97,
      "mean_abs_max": 3.2, "within8_min": 0.88, "ssim_min": 0.92,
      "measured_at": "a test", "decoder": "espcn_112t128_w48_d4_u2"}


@pytest.fixture(autouse=True)
def _registry_is_restored():
    """Every synthetic class below calls ``register``, which writes into the process-wide REGISTRY.

    Leaking one would parameterise every registry-swept test in the GPU suite over a class that cannot
    render, and the failure would surface in a completely unrelated file.
    """
    before = dict(se.REGISTRY)
    yield
    se.REGISTRY.clear()
    se.REGISTRY.update(before)


def _cls(name: str, doc: str, **attrs):
    """A minimal registerable env class. Nothing here renders; the declarations are the subject."""
    body = {"__doc__": doc, "state_dim": 2, "state_names": ("a", "b"), "default_size": 8,
            "reference_id": "fake/Env-v0",
            "make_variation_space": lambda self: spaces.Dict({}),
            "sample_states": lambda self, batch, rng=None: np.zeros((batch, 2), np.float32)}
    body.update(attrs)
    return type(name, (StableEnv,), body)


def _visual(name="Vis", **over):
    """A well-formed VISUAL-tier class, before whatever the caller breaks."""
    attrs = {"fidelity_tier": "visual", "visual_fidelity": dict(VF),
             "visual_note": "a different rasterizer draws the same scene", "parity_interior_min": None}
    attrs.update(over)
    return _cls(name, "A test env. VISUAL TIER: the frames are a different render.", **attrs)


def _learned(name="Lrn", **over):
    """A LEARNED-decoder class, complete except for whatever the caller breaks.

    Built on ``_visual`` because a learned decoder is only admissible inside the visual tier — the one
    test that needs it outside builds its class directly, so that the illegal combination cannot be
    produced here by accident.
    """
    attrs = {"render_decoder": "learned", "variation_fidelity": dict(VF)}
    attrs.update(over)
    return _visual(name, **attrs)


# ------------------------------------------------------------------ the tier


def test_a_well_formed_declaration_of_each_kind_registers():
    """The control. Without it every refusal below could be passing for an unrelated reason."""
    register(_cls("Plain", "An ordinary parity env."))
    register(_visual("GoodVisual"))
    register(_learned("GoodLearnedCorrected",
                      __doc__="A test env. VISUAL TIER, LEARNED decoder: a network drew the detail."))
    assert {"Plain", "GoodVisual", "GoodLearnedCorrected"} <= set(se.REGISTRY)


def test_an_unknown_tier_is_refused_by_name():
    with pytest.raises(ValueError, match="fidelity_tier must be one of"):
        register(_cls("Bad", "doc", fidelity_tier="approximate"))


def test_a_parity_env_carrying_visual_fidelity_is_refused():
    """An env makes exactly ONE of the two claims; carrying both blurs which one is being asserted.

    It is also the shape a half-finished tier change leaves behind — the numbers added, the tier not
    yet moved — and the suite would then hold it to the parity gate while a reader saw a visual
    declaration.
    """
    with pytest.raises(ValueError, match="declares fidelity_tier='parity' and also carries"):
        register(_cls("Bad", "doc", visual_fidelity=dict(VF)))


@pytest.mark.parametrize("missing", REQUIRED_VISUAL_FIDELITY)
def test_a_visual_env_missing_any_required_measurement_is_refused(missing):
    """Each of the seven separately, because a check written as "is the dict non-empty" passes on six.

    Three MEASURED numbers and three THRESHOLDS, kept apart on purpose: a threshold equal to the
    measurement flakes on the next state draw, and a measurement inferred from a threshold is not a
    measurement. ``measured_at`` is what makes either reproducible.
    """
    vf = {k: v for k, v in VF.items() if k != missing}
    with pytest.raises(ValueError, match="declares the VISUAL tier without"):
        register(_visual("Bad", visual_fidelity=vf))


def test_a_visual_env_with_no_visual_note_is_refused():
    """The note is what tells a reader which residual they are accepting, and what it disqualifies."""
    with pytest.raises(ValueError, match="VISUAL tier with no visual_note"):
        register(_visual("Bad", visual_note=""))


def test_a_visual_env_that_leaves_parity_interior_min_set_is_refused():
    """Leaving a number there is how the strong claim ends up loosened to accommodate a weak one.

    The two tiers are kept apart MECHANICALLY rather than by convention: with
    ``parity_interior_min = None`` the parity assertion cannot be applied to this env, cannot be
    silently relaxed for it, and cannot be passed by it.
    """
    with pytest.raises(ValueError, match="leaves parity_interior_min at"):
        register(_visual("Bad", parity_interior_min=1.0))


def test_a_visual_env_that_does_not_say_so_in_its_docstring_is_refused():
    """``help(cls)`` is where a user looks first and it must not read like a parity env.

    Mechanical on purpose: ``describe()`` and the registry listing both mark the tier, but a marker
    that depends on remembering to write it is the kind of rule this repo has learned to make
    impossible instead.
    """
    with pytest.raises(ValueError, match="does not say so in its class docstring"):
        register(_visual("Bad", __doc__="A fast env with no warning in it."))


# ------------------------------------------------------------------ the decoder


def test_an_unknown_decoder_is_refused_by_name():
    with pytest.raises(ValueError, match="render_decoder must be one of"):
        register(_cls("Bad", "doc", render_decoder="diffusion"))


def test_a_learned_decoder_outside_the_visual_tier_is_refused():
    """A frame a network drew is not the official frame and cannot make the parity claim.

    Whatever it measures — and a good corrector measures BETTER than the renderer it replaces on every
    declared number, which is exactly why this has to be blocked structurally rather than by judgement.
    """
    with pytest.raises(ValueError, match="declares render_decoder='learned' outside the VISUAL tier"):
        register(_cls("Bad", "doc. LEARNED.", render_decoder="learned",
                      variation_fidelity=dict(VF)))


def test_a_learned_decoder_whose_numbers_do_not_name_the_checkpoint_is_refused():
    """``visual_fidelity`` for a learned decoder describes a (renderer, checkpoint) PAIR.

    Without the second half the numbers read as a claim about the renderer, which is the one misreading
    this field exists to prevent — and a checkpoint replaced without its declaration would otherwise
    keep being described by the previous one's numbers.
    """
    vf = {k: v for k, v in VF.items() if k != "decoder"}
    with pytest.raises(ValueError, match="no 'decoder' entry naming the checkpoint"):
        register(_learned("BadCorrected", visual_fidelity=vf))


@pytest.mark.parametrize("missing", ["mean_abs", "within8_frac", "ssim", "mean_abs_max",
                                     "within8_min", "ssim_min", "measured_at"])
def test_a_learned_decoder_without_a_complete_variation_measurement_is_refused(missing):
    """THE asymmetry, and it is measured rather than stylistic.

    A rasterizer that agrees with the official renderer at one camera agrees at another — same
    arithmetic pointed somewhere else — so one declaration at the default appearance describes the
    whole variation space. A learned decoder does not generalise that way: fitted at one camera pose it
    scores 3.25/255 there and 9.06 under the env's own declared ±8° of pan, which is worse than not
    correcting at all (8.91) AND worse than the render it replaces (7.93). So for a decoder the
    default-appearance declaration is not the claim, it is the BEST CASE, and the variation regime is
    measured or the class does not register.
    """
    vf = {k: v for k, v in VF.items() if k != missing}
    with pytest.raises(ValueError, match="variation_fidelity is missing"):
        register(_learned("BadCorrected", variation_fidelity=vf))


def test_a_learned_decoder_with_no_variation_fidelity_at_all_is_refused():
    with pytest.raises(ValueError, match="variation_fidelity is missing"):
        register(_learned("BadCorrected", variation_fidelity={}))


def test_a_learned_decoder_that_does_not_say_so_in_its_docstring_is_refused():
    """The difference between "MuJoCo's renderer approximated" and "a learned approximation of MuJoCo's
    renderer" is INVISIBLE IN THE PICTURE, and ``help(cls)`` is where someone finds it."""
    with pytest.raises(ValueError, match="does not say so in its class docstring"):
        register(_learned("BadCorrected",
                          __doc__="A test env. VISUAL TIER: a fast re-render of the same scene."))


# ------------------------------------------------------------------ registration itself


def test_two_classes_cannot_share_a_lookup_key():
    """A silent overwrite would mean ``make("X")`` returning whichever module imported last.

    Auto-discovery makes that reachable without anyone editing a shared file, so it is an error rather
    than a warning, and the message names the module that got there first.
    """
    a = register(_cls("Twin", "doc"))
    b = _cls("Twin", "a different class with the same name")
    with pytest.raises(ValueError, match="env name collision"):
        register(b)
    assert se.REGISTRY["Twin"] is a


def test_registering_the_same_class_twice_is_not_a_collision():
    """Re-importing a module must be idempotent, or a reload turns into a spurious failure."""
    a = _cls("Same", "doc")
    register(a)
    register(a)
    assert se.REGISTRY["Same"] is a


def test_an_unknown_env_name_lists_the_ones_that_exist():
    with pytest.raises(KeyError) as e:
        make("Pusht")                                   # the plausible casing mistake
    assert "unknown env" in str(e.value)
    assert len(str(e.value)) > 100, "the error must list the known names, not merely reject one"


# ------------------------------------------------------------------ what the listing shows


def test_the_listing_marks_a_visual_env_and_a_learned_one_differently():
    """``sorted(REGISTRY)`` says nothing about what the frames are worth, which is why this exists.

    Anything that shows a user the registry shows this instead, so the envs whose frames are not the
    official pixels cannot be picked by name alone — and the two marks are different because the two
    facts are different: a visual env's residual is a rasterizer disagreeing, a learned decoder's is
    partly SYNTHESISED.
    """
    register(_cls("PlainOne", "An ordinary parity env."))
    register(_visual("VisualOne"))
    register(_learned("LearnedOneCorrected",
                      __doc__="A test env. VISUAL TIER, LEARNED decoder: a network drew the detail."))
    rows = {r["name"]: r for r in registry_table()}
    assert rows["PlainOne"]["tier"] == "parity" and rows["PlainOne"]["decoder"] == "renderer"
    assert rows["VisualOne"]["tier"] == "visual" and rows["VisualOne"]["decoder"] == "renderer"
    assert rows["LearnedOneCorrected"]["decoder"] == "learned"

    txt = format_registry()
    assert "VISUAL TIER — LEARNED decoder" in txt
    assert "not the official pixels" in txt
    for line in txt.splitlines():
        if line.strip().startswith("PlainOne"):
            assert "VISUAL" not in line, "a parity env must not be marked"


def test_registry_table_reports_no_state_dim_rather_than_a_wrong_one():
    """Envs that compute ``state_dim`` at construction (Fetch from its MJCF, RoboCasa from a measured
    live-qpos set) carry the class default of 0, and reporting 0 is a number that is WRONG rather than
    absent."""
    register(_cls("Deferred", "doc", state_dim=0))
    row = {r["name"]: r for r in registry_table()}["Deferred"]
    assert row["state_dim"] is None


def test_describe_tells_a_consumer_what_the_frames_are_not_for():
    """The answer to "can I train on these and evaluate on the official ones" is not in the other fields.

    A caller who prints ``describe()`` and reads nothing else still must not mistake a visual env for a
    parity one, and must not mistake a learned decoder for a rasterizer — so the learned case adds its
    own sentence in the caller's own words rather than leaving it to ``visual_note``.
    """
    plain = _cls("P2", "doc")(batch=1, device="cpu").describe()
    assert plain["fidelity_tier"] == "parity" and plain["render_decoder"] == "renderer"
    assert "not_for" not in plain and plain["parity_interior_min"] == 1.0

    vis = _visual("V2")(batch=1, device="cpu").describe()
    assert "different render of the same scene" in vis["not_for"]
    assert vis["visual_fidelity"]["ssim_min"] == 0.92

    lrn = _learned("L2Corrected",
                   __doc__="A test env. VISUAL TIER, LEARNED decoder.")(batch=1,
                                                                        device="cpu").describe()
    assert "LEARNED approximation" in lrn["not_for"]
    assert "synthesised rather than measured" in lrn["not_for"]
    assert lrn["variation_fidelity"]["ssim_min"] == 0.92


# ------------------------------------------------------------------ the real registry


def test_every_registered_class_still_satisfies_both_declarations():
    """The tripwire. ``register`` already enforces this, which is exactly why it is asserted again.

    If the enforcement is ever weakened, bypassed (a class put into ``REGISTRY`` directly), or reached
    through a path that does not call ``register``, this is the line that fails and says which class
    and which rule. It runs over the WHOLE registry on CPU with no env built, so it covers the families
    whose reference package is not installed in this interpreter — which is where a bad declaration
    would otherwise sit unnoticed, since every test that could see it needs the package.
    """
    bad = []
    for name, cls in sorted(se.REGISTRY.items()):
        for check in (check_fidelity_declaration, check_decoder_declaration):
            try:
                check(cls)
            except ValueError as e:
                bad.append(f"{name}: {e}")
    assert not bad, "registered classes with an inadmissible declaration:\n  " + "\n  ".join(bad)


def test_a_variation_fidelity_declared_by_anyone_is_complete():
    """Optional for a renderer, required for a decoder — but never PARTIAL, whoever declares it.

    ``register`` only checks completeness for the learned case, so a renderer that declared three of the
    seven fields would carry a dict that ``describe()`` publishes and no test reads. Half a measurement
    published as a measurement is the failure this closes.
    """
    bad = []
    for name, cls in sorted(se.REGISTRY.items()):
        vf = dict(getattr(cls, "variation_fidelity", {}) or {})
        if not vf:
            continue
        missing = [k for k in REQUIRED_VISUAL_FIDELITY if k not in vf]
        if missing:
            bad.append(f"{name} declares variation_fidelity without {missing}")
    assert not bad, "\n  ".join(bad)


def test_a_learned_class_has_the_plain_sibling_its_claim_is_relative_to():
    """A learned decoder is a SECOND way to draw a cell that already exists.

    Everything it claims — better mean, better within-8, better SSIM — is relative to the renderer it
    replaces, so without that sibling registered there is nothing to compare it against and the three
    numbers in its declaration describe nothing. Asserted on the class rather than on frames (which
    ``tests/gpu/test_conformance.py`` does) so it holds in interpreters with no checkpoint at all.
    """
    for name, cls in sorted(se.REGISTRY.items()):
        if getattr(cls, "render_decoder", "renderer") != "learned":
            continue
        base = name.removesuffix("Corrected")
        assert base != name, (
            f"{name} draws through a learned decoder but is not named <Plain>Corrected, so the sibling "
            f"it is a correction OF cannot be found by name")
        assert base in se.REGISTRY, (
            f"{name} has no plain sibling {base} in the registry")
        assert se.REGISTRY[base].render_decoder == "renderer", (
            f"{base} is supposed to be the plain way to draw this cell and it is not")


def test_the_declared_numbers_beat_the_render_they_replace_wherever_a_baseline_is_declared():
    """A learned decoder that is not better than the renderer it replaces must not exist.

    Asserted on the DECLARATION, which is free and runs without weights, in addition to the live-frame
    version in ``tests/gpu/test_conformance.py``. The measured reason it is a gate and not a
    preference: applied where it does not belong a corrector is WORSE than a plain resize — 5.94/255 on
    a camera 40 cm from the one it was fitted on against 6.14 for doing nothing — and it fails by
    painting plausible texture the official frame does not contain, so the mean does not catch it and
    SSIM and within-8 do.
    """
    checked = 0
    for name, cls in sorted(se.REGISTRY.items()):
        if getattr(cls, "render_decoder", "renderer") != "learned":
            continue
        for label, decl in (("visual_fidelity", dict(getattr(cls, "visual_fidelity", {}) or {})),
                            ("variation_fidelity", dict(getattr(cls, "variation_fidelity", {}) or {}))):
            if "plain_mean_abs" not in decl:
                continue                      # sealed before the baseline was recorded alongside it
            checked += 1
            assert decl["mean_abs"] < decl["plain_mean_abs"], f"{name} {label}: mean is not better"
            assert decl["within8_frac"] > decl["plain_within8_frac"], f"{name} {label}: within-8"
            assert decl["ssim"] > decl["plain_ssim"], f"{name} {label}: SSIM is not better"
    if not checked:
        pytest.skip("no registered class declares a learned decoder with a recorded plain baseline; "
                    "the live-frame version of this check is in tests/gpu/test_conformance.py")


def test_the_import_error_report_is_a_readable_mapping_and_costs_nothing():
    """``import_errors()`` is what makes "my env vanished" answerable in one line.

    A machine without ``gym_pusht`` should lose exactly PushT, not the package — so the report has to
    be a plain ``{module: reason}`` of strings, and it must not be able to fail while being read (it is
    called from a test that runs before anything else, and from the suite's terminal summary).
    """
    errs = se.envs.import_errors()
    assert isinstance(errs, dict)
    for k, v in errs.items():
        assert isinstance(k, str) and isinstance(v, str) and v, f"{k!r} -> {v!r}"
        assert ":" in v, f"the reason for {k} does not name an exception type: {v!r}"
    errs["scribbled"] = "x"
    assert "scribbled" not in se.envs.import_errors(), (
        "import_errors() hands out its own dict, so a caller can corrupt the record it reads")


def test_a_family_that_failed_to_import_contributed_no_envs_and_took_none_away():
    """One module's failure removes its own envs and nothing else. The point of auto-discovery.

    Asserted as the conjunction rather than by counting: for every module that did NOT import, no
    registered class claims to come from it, and the registry is still non-empty.
    """
    errs = se.envs.import_errors()
    assert se.REGISTRY, f"no envs registered at all; import errors: {errs}"
    for mod in errs:
        offenders = [n for n, c in se.REGISTRY.items() if c.__module__.endswith(f".{mod}")]
        assert not offenders, (
            f"{mod} is recorded as having failed to import, yet {offenders} claim to come from it")


# ------------------------------------------------------------- base/subclass attribute collisions

def test_no_env_shadows_an_attribute_the_base_class_owns():
    """An env module must not assign ``self.<name>`` for a ``<name>`` ``StableEnv`` assigns itself.

    THE BUG THIS EXISTS FOR, in full, because it is the most expensive shape of defect this repository
    has and it is invisible by construction. ``StableEnv`` grew a ``self._live`` to remember how many
    rows of the buffer a short ``set_state`` filled. ``robocasa.py`` had used ``self._live`` since it
    was written, for something entirely different — the indices of the qpos dimensions the camera can
    actually see. Nothing raised. Nothing warned. The base class simply won, and RoboCasa's
    ``sample_states`` started returning ``(4,)`` where it had returned ``(4, 40)``.

    It was caught only because the release gate was fanned out over the interpreters that can actually
    build a kitchen, so ``tests/gpu/test_rendernet.py`` ran for the first time instead of skipping.
    Five tests failed at once; before the fan-out, all five had been a green skip.

    Checked STATICALLY, from the source text, on purpose: constructing an env needs a GPU, an EGL
    context and — for RoboCasa — a compiled kitchen, so a test that instantiated things would itself
    skip on most machines, which is the failure mode that hid the bug in the first place.
    """
    import ast

    root = Path(__file__).resolve().parents[2]

    def self_attrs(node: ast.AST) -> set[str]:
        out: set[str] = set()
        for n in ast.walk(node):
            targets = ([n.target] if isinstance(n, ast.AnnAssign) else
                       list(n.targets) if isinstance(n, ast.Assign) else [])
            for t in targets:
                if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id == "self"):
                    out.add(t.attr)
        return out

    owned = self_attrs(ast.parse((root / "stable_envs" / "core" / "env.py").read_text()))
    assert {"_state", "device", "_live_rows"} <= owned, (
        f"read the base class wrong — it must assign self._state, self.device and self._live_rows, "
        f"and it appears to assign {sorted(owned)}. The collision check below is worthless if this "
        f"set is empty or partial, so it is asserted rather than trusted.")

    def env_classes(tree: ast.Module) -> list[ast.ClassDef]:
        """Only classes that actually INHERIT the base, transitively within the file.

        Scanning whole modules instead flagged ``GriddlyDecoder``, ``OvercookedAtlas`` and sokoban's
        ``_Atlas`` — plain helper classes that live beside an env and legitimately own a ``device``.
        A helper sharing a name with a base-class attribute is not a collision, because nothing ever
        assigns both to the same object.
        """
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        derived, changed = {"StableEnv"}, True
        while changed:
            changed = False
            for c in classes:
                names = {b.id for b in c.bases if isinstance(b, ast.Name)}
                names |= {b.attr for b in c.bases if isinstance(b, ast.Attribute)}
                if c.name not in derived and names & derived:
                    derived.add(c.name)
                    changed = True
        return [c for c in classes if c.name in derived]

    offenders: dict[str, set[str]] = {}
    for path in sorted((root / "stable_envs" / "envs").glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        clash = set().union(*(self_attrs(c) for c in env_classes(tree)), set()) & owned
        # ``device``/``B``/``size`` are re-derived by some envs from their own arguments before
        # delegating; the collision that matters is one where the two meanings DIFFER, and the guard
        # cannot see that. It is left strict on purpose: an env that genuinely needs to restate a base
        # attribute should say so here rather than have the check quietly widened.
        if clash:
            offenders[path.name] = clash

    assert not offenders, (
        "these env modules assign attributes that stable_envs/core/env.py also assigns, so whichever "
        "runs last silently wins and the other meaning is destroyed with no error:\n  "
        + "\n  ".join(f"{f}: {sorted(c)}" for f, c in sorted(offenders.items()))
        + "\n\nRename the env's attribute, or the base class's if the base class is the newcomer — "
          "that is what was done for _live (base -> _live_rows, robocasa -> _live_idx).")


# -------------------------------------------------------------------- the release gate's exclusions

def test_every_gate_exclusion_states_its_mechanism():
    """An env left out of the release gate must say WHY, and the why must name a mechanism.

    The gate's whole value is that a missing env fails the run instead of skipping it. That only holds
    if the exceptions are declared: "it did not import here" is a missing package and belongs in
    ``import_errors()``, whereas "it cannot be installed alongside the others" is a fact about the
    dependency graph and belongs in ``gate.EXCLUDED`` with the conflict written down. Without this
    test the list is a place to quietly move anything inconvenient, which is the same silent-skip
    failure one level up.
    """
    from stable_envs.testing import gate

    for name, why in gate.EXCLUDED.items():
        assert why and len(why) > 120, (
            f"gate.EXCLUDED[{name!r}] must explain the CONFLICT, not merely assert one. An exclusion "
            f"is the one way an env leaves the gate without failing it, so it carries the burden of "
            f"proof: name the packages and the versions that cannot be satisfied together.")
        assert any(tok in why for tok in ("==", "<=", ">=", "<", ">")), (
            f"gate.EXCLUDED[{name!r}] does not name a version constraint. Every real exclusion so far "
            f"has been two packages whose pins cannot both hold; if this one is not, it is probably a "
            f"missing install rather than an exclusion.")


def test_gate_envs_removes_exactly_the_excluded_and_nothing_else():
    """``gate_envs`` is subtraction, not filtering-by-availability."""
    from stable_envs.testing import gate

    names = ["PushT", "OvercookedCrampedRoom", "DMCWalkerWalk", "OvercookedCoordRing"]
    assert gate.gate_envs(names) == ["DMCWalkerWalk", "PushT"]
    assert gate.is_excluded("PushT") is None
    assert gate.is_excluded("OvercookedCrampedRoom")
    # An excluded env is still a first-class env: the carve-out is about which INTERPRETER runs the
    # gate, never about removing the family. If this ever fails, someone deleted an env believing the
    # exclusion licensed it.
    assert (Path(__file__).resolve().parents[2] / "stable_envs" / "envs" / "overcooked.py").exists()


def test_the_excluded_family_documents_itself_where_a_user_would_look():
    """The exclusion must be in the module's own docstring, not only in the gate's table.

    Someone hitting this reads the env module or its ImportError; they do not read the test harness.
    """
    src = (Path(__file__).resolve().parents[2] / "stable_envs" / "envs" / "overcooked.py").read_text()
    # Case-insensitive: this is prose, and the first version of this test asserted "own interpreter"
    # against a docstring that shouts "OWN INTERPRETER". A guard that fails on capitalisation trains
    # people to edit the guard, which is the opposite of what it is for.
    head = src[:src.index('"""', 3) + 3].lower()
    for token in ("scipy", "own interpreter", "gate.py"):
        assert token in head, (
            f"stable_envs/envs/overcooked.py's module docstring does not mention {token!r}. The "
            f"family is excluded from the release gate, and a user who hits that must be able to "
            f"find out why from the module itself.")


def test_the_live_context_cap_scales_with_the_registry():
    """The EGL tripwire must not fire on an interpreter that simply holds more envs.

    A context is created once per env that owns one and then deliberately NEVER released, so the
    legitimate live count grows with the registry. As a fixed 192 this was comfortable at 64 envs and
    became a false positive the moment the library moved into one interpreter: a 191-env run reached
    193 live contexts doing exactly what the design requires, and every test after that failed
    ``EGL_CONTEXTS_RUNAWAY`` — 1,776 of them having created no context at all.

    The rate-based rules are the ones that catch real runaway creation, and they are per-test, so
    scaling this one costs nothing they were covering.
    """
    from stable_envs.testing import resource_guard as RG

    lim = RG.limits()
    assert lim["session_live"] >= 192, "the floor must never make an existing run's bound tighter"
    assert lim["session_live"] >= 2 * len(se.REGISTRY), (
        f"session_live is {lim['session_live']} for a registry of {len(se.REGISTRY)}; one context per "
        f"env is legitimate and never released, so a cap below that fails healthy runs")
    assert lim["per_test"] == 8, ("per_test is the RATE bound and must stay fixed — it is what still "
                                  "catches a context created per state, per style value or per call")


def test_a_test_that_creates_no_context_is_never_a_runaway():
    """The exact shape of the false positive: zero created, zero freed, many live."""
    from stable_envs.testing import resource_guard as RG

    d = {"context_created": 0, "context_freed": 0, "renderer_created": 0, "renderer_closed": 0,
         "context_live": 2 * len(se.REGISTRY), "sites": []}
    names = [n for n, _ in RG.violations(d)]
    assert names == [], (
        f"a test that created and released nothing was reported as {names}. Holding many live "
        f"contexts is the DESIGN — they are cached for the life of the process — so it cannot by "
        f"itself be a violation, or the guard fails every test that runs after the registry is warm.")


# ------------------------------------------------------------------ reference versions

from stable_envs.core.env import MEASURED_WITH_KEY, REFERENCE_VERSIONS_KEY, reference_versions_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

RV = {"mujoco": "3.11.0", "mujoco_warp": "3.11.0"}


def test_a_visual_env_may_declare_the_reference_versions_it_was_measured_against():
    """The field the certifier compares with the measuring interpreter's ``reference_packages``.

    It lives next to ``measured_at`` because that is where the same fact was already written as prose;
    ``describe()`` publishes it so a consumer can see which reference the numbers describe.
    """
    cls = _visual("Versioned", visual_fidelity=dict(VF, **{REFERENCE_VERSIONS_KEY: dict(RV)}))
    register(cls)
    assert reference_versions_of(cls) == RV
    assert cls(batch=1, device="cpu").describe()[REFERENCE_VERSIONS_KEY] == RV


def test_a_parity_env_carries_reference_versions_as_a_class_attribute():
    cls = _cls("VersionedParity", "An ordinary parity env.", reference_versions=dict(RV))
    register(cls)
    assert reference_versions_of(cls) == RV
    assert cls(batch=1, device="cpu").describe()[REFERENCE_VERSIONS_KEY] == RV


def test_an_undeclared_reference_version_is_absent_not_empty():
    """Absent means "no claim about the reference version", which the certifier must not confuse with
    "measured against nothing"; ``describe()`` leaves the key out rather than publishing ``{}``."""
    cls = _cls("Plain", "An ordinary parity env.")
    register(cls)
    assert reference_versions_of(cls) == {}
    assert REFERENCE_VERSIONS_KEY not in cls(batch=1, device="cpu").describe()


@pytest.mark.parametrize("bad", [[], "3.11.0", {}, {"mujoco": 3.11}, {"mujoco": ""}, {"": "3.11.0"},
                                 {"mujoco": None}])
def test_a_malformed_reference_version_field_is_refused(bad):
    """A version the certifier cannot compare is worse than none, because it looks like provenance."""
    with pytest.raises(ValueError, match="reference_versions"):
        register(_visual("Bad", visual_fidelity=dict(VF, **{REFERENCE_VERSIONS_KEY: bad})))
    with pytest.raises(ValueError, match="reference_versions"):
        register(_cls("BadParity", "doc", reference_versions=bad))


def _dmc_declaration_table() -> dict:
    """``stable_envs.envs.dmcontrol.VISUAL_FIDELITY``, read off the SOURCE rather than the module.

    The module imports mujoco with ``MUJOCO_GL=egl`` at import time, so on a host without EGL (this
    laptop) the family does not register and the module cannot be imported -- but the declaration is
    a table of literals and names, and the claim under test is about the table. Only the module-level
    assignments the table is built from are executed, in an empty namespace.
    """
    import ast
    src = (ROOT / "stable_envs" / "envs" / "dmcontrol.py").read_text()
    want = {"_AT", "_SWIMMER_AT", "_AT_VERSIONS", "_SWIMMER_AT_VERSIONS", "_AT_MEASURED_WITH",
            "_SWIMMER_AT_MEASURED_WITH", "_QUADRUPED_AT", "_QUADRUPED_AT_MEASURED_WITH",
            "VISUAL_FIDELITY"}
    keep = [n for n in ast.parse(src).body
            if isinstance(n, (ast.Assign, ast.AnnAssign))
            and any(isinstance(t, ast.Name) and t.id in want
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "dmcontrol.py<declarations>", "exec"), ns)
    return ns["VISUAL_FIDELITY"]


def _declared_reference_versions() -> list[tuple[str, dict, str]]:
    """``(name, versions, measured_at)`` for every declaration in the tree that names its reference
    versions: the registered classes, plus the DM Control table read off the source so the check runs
    where that family cannot register."""
    rows = []
    for name, cls in sorted(se.REGISTRY.items()):
        rv = reference_versions_of(cls)
        if rv:
            # A visual env corroborates its declared versions with ``visual_fidelity["measured_at"]``.
            # A PARITY env cannot: ``check_fidelity_declaration`` forbids it from carrying
            # ``visual_fidelity`` at all, so its provenance text is ``parity_note`` -- the field that
            # already exists to say what the declared bound is and is not. Either way the rule is the
            # same and is the point of this check: a version may not be declared without prose in the
            # class that says the same thing, so a bump to one without the other is caught.
            at = str((getattr(cls, "visual_fidelity", {}) or {}).get("measured_at", ""))
            rows.append((name, rv, at or str(getattr(cls, "parity_note", "") or "")))
    for name, vf in sorted(_dmc_declaration_table().items()):
        for key in (REFERENCE_VERSIONS_KEY, MEASURED_WITH_KEY):
            if key in vf:
                rows.append((f"dmcontrol.VISUAL_FIDELITY[{name}].{key}", vf[key],
                             str(vf.get("measured_at", ""))))
    return rows


def test_every_declared_reference_version_is_the_one_measured_at_names():
    """Nothing invented: a declared version must appear, package and version, in ``measured_at``.

    The field was populated FROM that text (the DM Control ``_AT`` string names mujoco 3.11.0 and
    mujoco-warp 3.11.0 and no dm_control version, so no dm_control version is declared), and this
    keeps the two from drifting apart: a bump to one without the other is a declaration that
    disagrees with itself.
    """
    rows = _declared_reference_versions()
    assert rows, "no declaration in the tree names its reference versions; the DM Control table should"
    bad = []
    for name, rv, at in rows:
        text = at.lower().replace("-", "_")
        for pkg, ver in rv.items():
            if ver not in at or pkg.lower().replace("-", "_") not in text:
                bad.append(f"{name}: declares {pkg}={ver}, measured_at says {at!r}")
    assert not bad, "\n  ".join(bad)


def test_the_dm_control_trackcom_declarations_name_the_reference_they_were_measured_against():
    """The measured case this field exists for: declared under mujoco 3.11.0, judged under 3.12.0 in
    Step 4, and 3.12.0 re-anchors finite-plane textures (changelog item 15, commit cc7fb98c)."""
    table = _dmc_declaration_table()
    for name in ("DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk", "DMCHumanoidStand"):
        assert table[name][REFERENCE_VERSIONS_KEY] == {"mujoco": "3.11.0"}
        assert table[name][MEASURED_WITH_KEY] == {"mujoco_warp": "3.11.0"}
        assert table[name]["measured_at"].startswith("84 px")          # the text is kept as it was
    assert table["DMCSwimmerSwimmer6"][REFERENCE_VERSIONS_KEY] == {"mujoco": "3.11.0"}
    assert table["DMCSwimmerSwimmer6"][MEASURED_WITH_KEY] == {"warp": "1.16.0"}


OUR_RENDERER_PACKAGES = ("mujoco_warp", "mujoco-warp", "warp", "warp-lang", "nvdiffrast", "torch", "triton")


def test_reference_versions_never_name_our_own_renderer():
    """The field gates: with mujoco_warp in it, bumping OUR renderer would turn a real regression of
    ours into blocked_environment. Our side goes in measured_with, which is recorded and never compared."""
    bad = []
    for name, rv, _ in _declared_reference_versions():
        if name.endswith(f".{MEASURED_WITH_KEY}"):
            continue
        bad += [f"{name}: {pkg}" for pkg in rv if pkg in OUR_RENDERER_PACKAGES]
    assert not bad, "\n  ".join(bad)


def test_measured_with_is_validated_like_reference_versions_but_is_not_a_reference_version():
    cls = _visual("Measured", visual_fidelity=dict(VF, **{REFERENCE_VERSIONS_KEY: {"mujoco": "3.11.0"},
                                                          MEASURED_WITH_KEY: {"mujoco_warp": "3.11.0"}}))
    register(cls)
    assert reference_versions_of(cls) == {"mujoco": "3.11.0"}
    with pytest.raises(ValueError, match="measured_with"):
        register(_visual("Bad", visual_fidelity=dict(VF, **{MEASURED_WITH_KEY: {"mujoco_warp": 3}})))


def test_visual_fidelity_reference_versions_take_precedence_over_the_class_attribute():
    """``reference_versions_of`` is the one reading; ``describe()`` and the certifier both use it."""
    cls = _visual("Both", reference_versions={"mujoco": "3.10.0", "gymnasium": "1.3.0"},
                  visual_fidelity=dict(VF, **{REFERENCE_VERSIONS_KEY: {"mujoco": "3.11.0"}}))
    register(cls)
    assert reference_versions_of(cls) == {"mujoco": "3.11.0", "gymnasium": "1.3.0"}
