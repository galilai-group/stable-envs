"""What an env is allowed to CLAIM: the fidelity tier, and what registration refuses.

``fidelity_tier = "visual"`` removes the parity gate. ``core/env.py`` asks for something in return:
measured numbers, thresholds, a stated mechanism, and words in the class docstring that a reader will
see. ``register`` enforces that.

The refusals are asserted here against SYNTHETIC classes, one per rule, on CPU. A rule is only proved
by a class that breaks it, and no real env in this repository breaks any of them. The registry-wide
half of the file then asserts that the real classes still satisfy the rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import stable_envs as se
from stable_envs import spaces
from stable_envs.env import REQUIRED_VISUAL_FIDELITY
from stable_envs.env import Env as StableEnv
from stable_envs.registry import (
    check_fidelity_declaration,
    format_registry,
    make,
    register,
    registry_table,
)

#: A complete, self-consistent measurement. Individual tests below remove exactly one thing from it,
#: which is what makes each of them a test of one rule rather than of the whole block.
VF = {
    "mean_abs": 2.45,
    "within8_frac": 0.935,
    "ssim": 0.97,
    "mean_abs_max": 3.2,
    "within8_min": 0.88,
    "ssim_min": 0.92,
    "measured_at": "a test",
}


@pytest.fixture(autouse=True)
def _registry_is_restored():
    """Every synthetic class below calls ``register``, which writes into the process-wide REGISTRY.

    Leaking one would parameterise every registry-swept test in the GPU suite over a class that cannot
    render, and the failure would surface in a completely unrelated file.
    """
    before = dict(se.registry.REGISTRY)
    yield
    se.registry.REGISTRY.clear()
    se.registry.REGISTRY.update(before)


def _cls(name: str, doc: str, **attrs):
    """A minimal registerable env class. Nothing here renders; the declarations are the subject."""
    body = {
        "__doc__": doc,
        "state_dim": 2,
        "state_names": ("a", "b"),
        "default_size": 8,
        "reference_id": "fake/Env-v0",
        "family": "test",
        "make_variation_space": lambda self: spaces.Dict({}),
        "sample_states": lambda self, batch, rng=None: np.zeros((batch, 2), np.float32),
    }
    body.update(attrs)
    return type(name, (StableEnv,), body)


def _visual(name="Vis", **over):
    """A well-formed VISUAL-tier class, before whatever the caller breaks."""
    attrs = {
        "fidelity_tier": "visual",
        "visual_fidelity": dict(VF),
        "visual_note": "a different rasterizer draws the same scene",
        "parity_interior_min": None,
    }
    attrs.update(over)
    return _cls(name, "A test env. VISUAL TIER: the frames are a different render.", **attrs)


# ------------------------------------------------------------------ the tier


def test_a_well_formed_declaration_of_each_kind_registers():
    """The control. Without it every refusal below could be passing for an unrelated reason."""
    register(_cls("Plain", "An ordinary parity env."))
    register(_visual("GoodVisual"))
    assert {"Plain", "GoodVisual"} <= set(se.registry.REGISTRY)


def test_an_unknown_tier_is_refused_by_name():
    with pytest.raises(ValueError, match="fidelity_tier must be one of"):
        register(_cls("Bad", "doc", fidelity_tier="approximate"))


def test_a_parity_env_carrying_visual_fidelity_is_refused():
    """An env makes exactly one of the two claims; carrying both blurs which one is asserted."""
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
    assert se.registry.REGISTRY["Twin"] is a


def test_registering_the_same_class_twice_is_not_a_collision():
    """Re-importing a module must be idempotent, or a reload turns into a spurious failure."""
    a = _cls("Same", "doc")
    register(a)
    register(a)
    assert se.registry.REGISTRY["Same"] is a


def test_an_unknown_env_name_lists_the_ones_that_exist():
    with pytest.raises(KeyError) as e:
        make("Pusht")  # the plausible casing mistake
    assert "unknown env" in str(e.value)
    assert len(str(e.value)) > 100, "the error must list the known names, not merely reject one"


# ------------------------------------------------------------------ what the listing shows


def test_the_listing_marks_a_visual_env():
    """``sorted(REGISTRY)`` says nothing about what the frames are worth, which is why this exists.

    Anything that shows a user the registry shows this instead, so the envs whose frames are not the
    official pixels cannot be picked by name alone.
    """
    register(_cls("PlainOne", "An ordinary parity env."))
    register(_visual("VisualOne"))
    rows = {r["name"]: r for r in registry_table()}
    assert rows["PlainOne"]["tier"] == "parity"
    assert rows["VisualOne"]["tier"] == "visual"

    txt = format_registry()
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
    assert row["render_state_dim"] is None


def test_describe_tells_a_consumer_what_the_frames_are_not_for():
    """The answer to "can I train on these and evaluate on the official ones" is not in the other fields.

    A caller who prints ``describe()`` and reads nothing else still must not mistake a visual env for a
    parity one.
    """
    plain = _cls("P2", "doc")(num_envs=1, device="cpu").describe()
    assert plain["fidelity_tier"] == "parity"
    assert "not_for" not in plain and plain["parity_interior_min"] == 1.0

    vis = _visual("V2")(num_envs=1, device="cpu").describe()
    assert "different render of the same scene" in vis["not_for"]
    assert vis["visual_fidelity"]["ssim_min"] == 0.92


# ------------------------------------------------------------------ the real registry


def test_every_registered_class_still_satisfies_its_fidelity_declaration():
    """The tripwire. ``register`` already enforces this, which is exactly why it is asserted again.

    If the enforcement is ever weakened, bypassed (a class put into ``REGISTRY`` directly), or reached
    through a path that does not call ``register``, this is the line that fails and says which class
    and which rule. It runs over the WHOLE registry on CPU with no env built, so it covers the families
    whose reference package is not installed in this interpreter — which is where a bad declaration
    would otherwise sit unnoticed, since every test that could see it needs the package.
    """
    bad = []
    for name, cls in sorted(se.registry.REGISTRY.items()):
        try:
            check_fidelity_declaration(cls)
        except ValueError as e:
            bad.append(f"{name}: {e}")
    assert not bad, "registered classes with an inadmissible declaration:\n  " + "\n  ".join(bad)


def test_the_import_error_report_is_a_readable_mapping_and_costs_nothing():
    """``import_errors()`` is what makes "my env vanished" answerable in one line.

    A machine without ``gym_pusht`` should lose exactly PushT, not the package — so the report has to
    be a plain ``{module: reason}`` of strings, and it must not be able to fail while being read (it is
    called from a test that runs before anything else, and from the suite's terminal summary).
    """
    errs = se.import_errors()
    assert isinstance(errs, dict)
    for k, v in errs.items():
        assert isinstance(k, str) and isinstance(v, str) and v, f"{k!r} -> {v!r}"
        assert ":" in v, f"the reason for {k} does not name an exception type: {v!r}"
    errs["scribbled"] = "x"
    assert "scribbled" not in se.import_errors(), (
        "import_errors() hands out its own dict, so a caller can corrupt the record it reads"
    )


def test_a_family_that_failed_to_import_contributed_no_envs_and_took_none_away():
    """One module's failure removes its own envs and nothing else. The point of auto-discovery.

    Asserted as the conjunction rather than by counting: for every module that did NOT import, no
    registered class claims to come from it, and the registry is still non-empty.
    """
    errs = se.import_errors()
    assert se.registry.REGISTRY, f"no envs registered at all; import errors: {errs}"
    for mod in errs:
        offenders = [n for n, c in se.registry.REGISTRY.items() if c.__module__.endswith(f".{mod}")]
        assert not offenders, f"{mod} is recorded as having failed to import, yet {offenders} claim to come from it"


# ------------------------------------------------------------- base/subclass attribute collisions


def test_no_env_shadows_an_attribute_the_base_class_owns():
    """An env module must not assign ``self.<name>`` for a ``<name>`` ``StableEnv`` assigns itself.

    THE BUG THIS EXISTS FOR, in full, because it is the most expensive shape of defect this repository
    has and it is invisible by construction. ``StableEnv`` grew a ``self._live`` to remember how many
    rows of the buffer a short ``set_state`` filled. ``robocasa.py`` had used ``self._live`` since it
    was written, for something entirely different — the indices of the qpos dimensions the camera can
    actually see. Nothing raised. Nothing warned. The base class simply won, and RoboCasa's
    ``sample_states`` started returning ``(4,)`` where it had returned ``(4, 40)``.

    Checked STATICALLY, from the source text, on purpose: constructing an env needs a GPU, an EGL
    context and — for RoboCasa — a compiled kitchen, so a test that instantiated things would itself
    skip on most machines, which is the failure mode that hid the bug in the first place.
    """
    import ast

    root = Path(__file__).resolve().parents[2]

    def self_attrs(node: ast.AST) -> set[str]:
        out: set[str] = set()
        for n in ast.walk(node):
            targets = (
                [n.target] if isinstance(n, ast.AnnAssign) else list(n.targets) if isinstance(n, ast.Assign) else []
            )
            for t in targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    out.add(t.attr)
        return out

    owned = self_attrs(ast.parse((root / "stable_envs" / "env.py").read_text()))
    assert {"_render_state", "device", "_live_rows"} <= owned, (
        f"read the base class wrong — it must assign self._render_state, self.device and self._live_rows, "
        f"and it appears to assign {sorted(owned)}. The collision check below is worthless if this "
        f"set is empty or partial, so it is asserted rather than trusted."
    )

    def env_classes(tree: ast.Module) -> list[ast.ClassDef]:
        """Only classes that actually INHERIT the base, transitively within the file.

        Scanning whole modules instead flagged ``GriddlyDecoder``, ``OvercookedAtlas`` and sokoban's
        ``_Atlas`` — plain helper classes that live beside an env and legitimately own a ``device``.
        A helper sharing a name with a base-class attribute is not a collision, because nothing ever
        assigns both to the same object.
        """
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        derived, changed = {"Env", "StableEnv"}, True
        while changed:
            changed = False
            for c in classes:
                names = {b.id for b in c.bases if isinstance(b, ast.Name)}
                # ``gym.Env``/``gymnasium.Env`` are upstream simulators (e.g. PushT's source task under
                # envs/pusht/simulation/), not stable_envs.Env; their attributes never meet ours.
                names |= {
                    b.attr
                    for b in c.bases
                    if isinstance(b, ast.Attribute)
                    and not (isinstance(b.value, ast.Name) and b.value.id in {"gym", "gymnasium"})
                }
                if c.name not in derived and names & derived:
                    derived.add(c.name)
                    changed = True
        return [c for c in classes if c.name in derived]

    # Base attributes an env deliberately restates WITH THE BASE MEANING. Navigation's reset()
    # sets ``_live_rows = self.B`` after installing a full batch, exactly as ``Env.set_state`` does.
    restated = {"envs/navigation/envs.py": {"_live_rows"}}
    offenders: dict[str, set[str]] = {}
    for path in sorted((root / "stable_envs" / "envs").rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        clash = set().union(*(self_attrs(c) for c in env_classes(tree)), set()) & owned
        clash -= restated.get(path.relative_to(root / "stable_envs").as_posix(), set())
        # ``device``/``B``/``size`` are re-derived by some envs from their own arguments before
        # delegating; the collision that matters is one where the two meanings DIFFER, and the guard
        # cannot see that. It is left strict on purpose: an env that genuinely needs to restate a base
        # attribute should say so here rather than have the check quietly widened.
        if clash:
            offenders[path.relative_to(root / "stable_envs").as_posix()] = clash

    assert not offenders, (
        "these env modules assign attributes that stable_envs/core/env.py also assigns, so whichever "
        "runs last silently wins and the other meaning is destroyed with no error:\n  "
        + "\n  ".join(f"{f}: {sorted(c)}" for f, c in sorted(offenders.items()))
        + "\n\nRename the env's attribute, or the base class's if the base class is the newcomer — "
        "that is what was done for _live (base -> _live_rows, robocasa -> _live_idx)."
    )


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
    from tests._runner import gate

    for name, why in gate.EXCLUDED.items():
        assert why and len(why) > 120, (
            f"gate.EXCLUDED[{name!r}] must explain the CONFLICT, not merely assert one. An exclusion "
            f"is the one way an env leaves the gate without failing it, so it carries the burden of "
            f"proof: name the packages and the versions that cannot be satisfied together."
        )
        assert any(tok in why for tok in ("==", "<=", ">=", "<", ">")), (
            f"gate.EXCLUDED[{name!r}] does not name a version constraint. Every real exclusion so far "
            f"has been two packages whose pins cannot both hold; if this one is not, it is probably a "
            f"missing install rather than an exclusion."
        )


def test_the_live_context_cap_scales_with_the_registry():
    """The EGL tripwire must not fire on an interpreter that simply holds more envs.

    A context is created once per env that owns one and never released, so the legitimate live count
    grows with the registry; the per-test rate rules are the ones that catch runaway creation.
    """
    from tests._runner import resource_guard as RG

    lim = RG.limits()
    assert lim["session_live"] >= 192, "the floor must never make an existing run's bound tighter"
    assert lim["session_live"] >= 2 * len(se.registry.REGISTRY), (
        f"session_live is {lim['session_live']} for a registry of {len(se.registry.REGISTRY)}; one context per "
        f"env is legitimate and never released, so a cap below that fails healthy runs"
    )
    assert lim["per_test"] == 8, (
        "per_test is the RATE bound and must stay fixed — it is what still "
        "catches a context created per state, per style value or per call"
    )


def test_a_test_that_creates_no_context_is_never_a_runaway():
    """The exact shape of the false positive: zero created, zero freed, many live."""
    from tests._runner import resource_guard as RG

    d = {
        "context_created": 0,
        "context_freed": 0,
        "renderer_created": 0,
        "renderer_closed": 0,
        "context_live": 2 * len(se.registry.REGISTRY),
        "sites": [],
    }
    names = [n for n, _ in RG.violations(d)]
    assert names == [], (
        f"a test that created and released nothing was reported as {names}. Holding many live "
        f"contexts is the DESIGN — they are cached for the life of the process — so it cannot by "
        f"itself be a violation, or the guard fails every test that runs after the registry is warm."
    )


# ------------------------------------------------------------------ reference versions

from stable_envs.env import MEASURED_WITH_KEY, REFERENCE_VERSIONS_KEY, reference_versions_of  # noqa: E402

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
    assert cls(num_envs=1, device="cpu").describe()[REFERENCE_VERSIONS_KEY] == RV


def test_a_parity_env_carries_reference_versions_as_a_class_attribute():
    cls = _cls("VersionedParity", "An ordinary parity env.", reference_versions=dict(RV))
    register(cls)
    assert reference_versions_of(cls) == RV
    assert cls(num_envs=1, device="cpu").describe()[REFERENCE_VERSIONS_KEY] == RV


def test_an_undeclared_reference_version_is_absent_not_empty():
    """Absent means "no claim about the reference version", which the certifier must not confuse with
    "measured against nothing"; ``describe()`` leaves the key out rather than publishing ``{}``."""
    cls = _cls("Plain", "An ordinary parity env.")
    register(cls)
    assert reference_versions_of(cls) == {}
    assert REFERENCE_VERSIONS_KEY not in cls(num_envs=1, device="cpu").describe()


@pytest.mark.parametrize("bad", [[], "3.11.0", {}, {"mujoco": 3.11}, {"mujoco": ""}, {"": "3.11.0"}, {"mujoco": None}])
def test_a_malformed_reference_version_field_is_refused(bad):
    """A version the certifier cannot compare is worse than none, because it looks like provenance."""
    with pytest.raises(ValueError, match="reference_versions"):
        register(_visual("Bad", visual_fidelity=dict(VF, **{REFERENCE_VERSIONS_KEY: bad})))
    with pytest.raises(ValueError, match="reference_versions"):
        register(_cls("BadParity", "doc", reference_versions=bad))


def _dmc_declaration_table() -> dict:
    """``stable_envs.envs.dmcontrol.VISUAL_FIDELITY``, read off the source rather than the module.

    Reading the source keeps the claim about the declaration table valid on hosts whose GL backend cannot
    register the cells. Only the module-level assignments the table is built from are executed, in one
    shared namespace: every module of the package is walked, ``_cells`` first because per-domain entries
    name ``_AT``/``_AT_VERSIONS``/``_AT_MEASURED_WITH``, and the tables are merged.
    """
    import ast

    pkg = ROOT / "stable_envs" / "envs" / "dmcontrol"
    want = {
        "_AT",
        "_SWIMMER_AT",
        "_AT_VERSIONS",
        "_SWIMMER_AT_VERSIONS",
        "_AT_MEASURED_WITH",
        "_SWIMMER_AT_MEASURED_WITH",
        "_QUADRUPED_AT",
        "_QUADRUPED_AT_MEASURED_WITH",
        "_FROZEN_STAND_VISUAL_FIDELITY",
        "VISUAL_FIDELITY",
    }
    # ``__init__.py`` and ``_family.py`` are skipped: their ``VISUAL_FIDELITY`` is a VIEW of the
    # per-domain tables (``{n: globals()[n].visual_fidelity ...}``), not a declaration, and reading
    # it would either need the classes or hide the literals this check exists to read.
    skip = {"_cells.py", "__init__.py", "_family.py"}
    paths = [pkg / "_cells.py"] + sorted(p for p in pkg.glob("*.py") if p.name not in skip)
    ns: dict = {}
    table: dict = {}
    for path in paths:
        keep = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(t, ast.Name) and t.id in want
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
            )
        ]
        if not keep:
            continue
        ns.pop("VISUAL_FIDELITY", None)
        exec(compile(ast.Module(body=keep, type_ignores=[]), f"{path.name}<declarations>", "exec"), ns)
        table.update(ns.get("VISUAL_FIDELITY", {}))
    assert table, f"no VISUAL_FIDELITY entries found under {pkg}; the walk has drifted"
    return table


def _declared_reference_versions() -> list[tuple[str, dict, str]]:
    """``(name, versions, measured_at)`` for every declaration in the tree that names its reference
    versions: the registered classes, plus the DM Control table read off the source so the check runs
    where that family cannot register."""
    rows = []
    for name, cls in sorted(se.registry.REGISTRY.items()):
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
                rows.append((f"dmcontrol.VISUAL_FIDELITY[{name}].{key}", vf[key], str(vf.get("measured_at", ""))))
    return rows


@pytest.mark.family("dmc")
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


@pytest.mark.family("dmc")
def test_the_dm_control_trackcom_declarations_name_the_reference_they_were_measured_against():
    """Declared under mujoco 3.11.0; 3.12.0 re-anchors finite-plane textures (changelog item 15, commit cc7fb98c)."""
    table = _dmc_declaration_table()
    for name in (
        "DMCCheetahRun",
        "DMCHopperHop",
        "DMCWalkerWalk",
        "DMCHumanoidWalk",
        "DMCSwimmerSwimmer6",
        "DMCQuadrupedWalk",
    ):
        assert table[name][REFERENCE_VERSIONS_KEY] == {"mujoco": "3.12.0"}
        assert table[name][MEASURED_WITH_KEY] == {"mujoco_warp": "3.12.0"}
        assert "mujoco 3.12.0" in table[name]["measured_at"]


OUR_RENDERER_PACKAGES = ("mujoco_warp", "mujoco-warp", "warp", "warp-lang", "nvdiffrast", "torch", "triton")


@pytest.mark.family("dmc")
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
    cls = _visual(
        "Measured",
        visual_fidelity=dict(
            VF, **{REFERENCE_VERSIONS_KEY: {"mujoco": "3.11.0"}, MEASURED_WITH_KEY: {"mujoco_warp": "3.11.0"}}
        ),
    )
    register(cls)
    assert reference_versions_of(cls) == {"mujoco": "3.11.0"}
    with pytest.raises(ValueError, match="measured_with"):
        register(_visual("Bad", visual_fidelity=dict(VF, **{MEASURED_WITH_KEY: {"mujoco_warp": 3}})))


def test_visual_fidelity_reference_versions_take_precedence_over_the_class_attribute():
    """``reference_versions_of`` is the one reading; ``describe()`` and the certifier both use it."""
    cls = _visual(
        "Both",
        reference_versions={"mujoco": "3.10.0", "gymnasium": "1.3.0"},
        visual_fidelity=dict(VF, **{REFERENCE_VERSIONS_KEY: {"mujoco": "3.11.0"}}),
    )
    register(cls)
    assert reference_versions_of(cls) == {"mujoco": "3.11.0", "gymnasium": "1.3.0"}
