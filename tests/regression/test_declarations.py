"""Fidelity declarations, registration refusals, and reference-version declarations.

An env that declares ``visual_fidelity`` must carry its measurements, ``measured_at`` and a
``visual_note``; ``register`` enforces this. Each refusal is checked against a synthetic class that
breaks one rule, on CPU; the registry-wide tests check that the real classes satisfy the rules.
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
from tests.helpers import pixel_tolerances as PT

#: A complete measurement; tests below remove one field at a time.
VF = {
    "mean_abs": 2.45,
    "within8_frac": 0.935,
    "ssim": 0.97,
    "measured_at": "a test",
}


@pytest.fixture(autouse=True)
def _registry_is_restored():
    """Restore the process-wide REGISTRY after each test registers synthetic classes."""
    before = dict(se.registry.REGISTRY)
    yield
    se.registry.REGISTRY.clear()
    se.registry.REGISTRY.update(before)


def _cls(name: str, doc: str, **attrs):
    """A minimal registerable env class that does not render."""
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
    """A well-formed class with a ``visual_fidelity`` record, with ``over`` applied."""
    attrs = {
        "visual_fidelity": dict(VF),
        "visual_note": "a different rasterizer draws the same scene",
    }
    attrs.update(over)
    return _cls(name, "A test env drawn by a different rasterizer.", **attrs)


# ------------------------------------------------------------------ the declaration


def test_a_well_formed_declaration_of_each_kind_registers():
    """Classes with and without ``visual_fidelity`` register (the control for the refusals below)."""
    register(_cls("Plain", "An ordinary env."))
    register(_visual("GoodVisual"))
    assert {"Plain", "GoodVisual"} <= set(se.registry.REGISTRY)


@pytest.mark.parametrize("missing", REQUIRED_VISUAL_FIDELITY)
def test_a_visual_env_missing_any_required_measurement_is_refused(missing):
    """An env missing any one required ``visual_fidelity`` field is refused.

    ``measured_at`` records where the measurements were taken.
    """
    vf = {k: v for k, v in VF.items() if k != missing}
    with pytest.raises(ValueError, match="declares visual_fidelity without"):
        register(_visual("Bad", visual_fidelity=vf))


def test_a_visual_env_with_no_visual_note_is_refused():
    """An env with ``visual_fidelity`` and an empty ``visual_note`` is refused."""
    with pytest.raises(ValueError, match="visual_fidelity with no visual_note"):
        register(_visual("Bad", visual_note=""))


# ------------------------------------------------------------------ registration itself


def test_two_classes_cannot_share_a_lookup_key():
    """Registering a different class under an existing name raises instead of overwriting."""
    a = register(_cls("Twin", "doc"))
    b = _cls("Twin", "a different class with the same name")
    with pytest.raises(ValueError, match="env name collision"):
        register(b)
    assert se.registry.REGISTRY["Twin"] is a


def test_registering_the_same_class_twice_is_not_a_collision():
    """Registering the same class twice is idempotent."""
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


def test_the_listing_names_each_class_and_its_reference():
    """``format_registry`` lists every registered class with its reference ID."""
    register(_cls("PlainOne", "An ordinary env."))
    txt = format_registry()
    assert txt.startswith(f"{len(registry_table())} environments")
    line = next(line for line in txt.splitlines() if line.strip().startswith("PlainOne"))
    assert "fake/Env-v0" in line


def test_registry_table_reports_no_state_dim_rather_than_a_wrong_one():
    """``registry_table`` reports ``None`` for a class whose ``state_dim`` is computed at construction (0)."""
    register(_cls("Deferred", "doc", state_dim=0))
    row = {r["name"]: r for r in registry_table()}["Deferred"]
    assert row["render_state_dim"] is None


def test_describe_reports_the_measured_agreement_only_when_declared():
    """``describe()`` includes ``visual_fidelity`` and ``visual_note`` only for a class that declares them."""
    plain = _cls("P2", "doc")(num_envs=1, device="cpu").describe()
    assert plain["reference_id"] == "fake/Env-v0"
    assert "visual_fidelity" not in plain and "visual_note" not in plain

    vis = _visual("V2")(num_envs=1, device="cpu").describe()
    assert vis["visual_fidelity"]["ssim"] == 0.97
    assert vis["visual_note"] == "a different rasterizer draws the same scene"


# ------------------------------------------------------------------ the real registry


def test_every_registered_class_still_satisfies_its_fidelity_declaration():
    """Every registered class passes ``check_fidelity_declaration``.

    Also catches classes added to ``REGISTRY`` without ``register``. Runs on CPU without building envs.
    """
    bad = []
    for name, cls in sorted(se.registry.REGISTRY.items()):
        try:
            check_fidelity_declaration(cls)
        except ValueError as e:
            bad.append(f"{name}: {e}")
    assert not bad, "registered classes with an inadmissible declaration:\n  " + "\n  ".join(bad)


def test_every_class_with_visual_fidelity_has_agreement_limits_and_no_other_does():
    """A registered class declares ``visual_fidelity`` if and only if ``pixel_tolerances.AGREEMENT`` names it.

    The source-comparison tests pick the exact or the agreement criterion from that table.
    """
    bad = [
        f"{name}: visual_fidelity={bool(cls.visual_fidelity)}, AGREEMENT entry={name in PT.AGREEMENT}"
        for name, cls in sorted(se.registry.REGISTRY.items())
        if bool(cls.visual_fidelity) != (name in PT.AGREEMENT)
    ]
    assert not bad, "visual_fidelity and tests/helpers/pixel_tolerances.AGREEMENT disagree:\n  " + "\n  ".join(bad)
    assert not set(PT.AGREEMENT) & set(PT.EXACT_OVERRIDES), "a class has both an exact and an agreement entry"


def test_the_import_error_report_is_a_readable_mapping_and_costs_nothing():
    """``import_errors()`` returns a fresh ``{module: "ExcType: reason"}`` dict of strings."""
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
    """No registered class comes from a module that failed to import, and the registry is non-empty."""
    errs = se.import_errors()
    assert se.registry.REGISTRY, f"no envs registered at all; import errors: {errs}"
    for mod in errs:
        offenders = [n for n, c in se.registry.REGISTRY.items() if c.__module__.endswith(f".{mod}")]
        assert not offenders, f"{mod} is recorded as having failed to import, yet {offenders} claim to come from it"


# ------------------------------------------------------------- base/subclass attribute collisions


def test_no_env_shadows_an_attribute_the_base_class_owns():
    """No env class assigns a ``self.<name>`` that the base ``StableEnv`` also assigns.

    Checked statically from source, so it runs where the envs cannot be built.
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
        f"read the base class wrong: it must assign self._render_state, self.device and self._live_rows, "
        f"and it appears to assign {sorted(owned)}"
    )

    def env_classes(tree: ast.Module) -> list[ast.ClassDef]:
        """Classes that inherit the base, transitively within the file; helper classes are excluded."""
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

    # Base attributes an env deliberately restates with the base meaning. Navigation's reset()
    # sets ``_live_rows = self.B`` after installing a full batch, exactly as ``Env.set_state`` does.
    restated = {"envs/navigation/envs.py": {"_live_rows"}}
    offenders: dict[str, set[str]] = {}
    for path in sorted((root / "stable_envs" / "envs").rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        clash = set().union(*(self_attrs(c) for c in env_classes(tree)), set()) & owned
        clash -= restated.get(path.relative_to(root / "stable_envs").as_posix(), set())
        # Kept strict: an env that must restate a base attribute is listed in ``restated``.
        if clash:
            offenders[path.relative_to(root / "stable_envs").as_posix()] = clash

    assert not offenders, (
        "these env modules assign attributes that stable_envs/env.py also assigns, so whichever "
        "runs last silently wins:\n  "
        + "\n  ".join(f"{f}: {sorted(c)}" for f, c in sorted(offenders.items()))
        + "\n\nRename the env's attribute, or the base class's if the base class is the newcomer."
    )


def test_the_live_context_cap_scales_with_the_registry():
    """The session live-context cap scales with registry size; the per-test rate cap stays at 8."""
    from tests._runner import resource_guard as RG

    lim = RG.limits()
    assert lim["session_live"] >= 192, "the session cap must not drop below 192"
    assert lim["session_live"] >= 2 * len(se.registry.REGISTRY), (
        f"session_live is {lim['session_live']} for a registry of {len(se.registry.REGISTRY)}; one context per "
        f"env is legitimate and never released, so a cap below that fails healthy runs"
    )
    assert lim["per_test"] == 8, (
        "per_test is the rate bound and must stay fixed; it catches a context created per state, "
        "per style value or per call"
    )


def test_a_test_that_creates_no_context_is_never_a_runaway():
    """A test with no context created or freed is not a violation, however many are live."""
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
        f"a test that created and released nothing was reported as {names}; live contexts are cached "
        f"for the life of the process, so a high live count alone is not a violation"
    )


# ------------------------------------------------------------------ reference versions

from stable_envs.env import MEASURED_WITH_KEY, REFERENCE_VERSIONS_KEY, reference_versions_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

RV = {"mujoco": "3.11.0", "mujoco_warp": "3.11.0"}


def test_a_visual_env_may_declare_the_reference_versions_it_was_measured_against():
    """An env's ``visual_fidelity`` may declare reference versions, published by ``describe()``.

    The certifier compares them with the measuring interpreter's ``reference_packages``.
    """
    cls = _visual("Versioned", visual_fidelity=dict(VF, **{REFERENCE_VERSIONS_KEY: dict(RV)}))
    register(cls)
    assert reference_versions_of(cls) == RV
    assert cls(num_envs=1, device="cpu").describe()[REFERENCE_VERSIONS_KEY] == RV


def test_an_env_carries_reference_versions_as_a_class_attribute():
    cls = _cls("VersionedParity", "An ordinary env.", reference_versions=dict(RV))
    register(cls)
    assert reference_versions_of(cls) == RV
    assert cls(num_envs=1, device="cpu").describe()[REFERENCE_VERSIONS_KEY] == RV


def test_an_undeclared_reference_version_is_absent_not_empty():
    """Absent means "no claim about the reference version", which the certifier must not confuse with
    "measured against nothing"; ``describe()`` leaves the key out rather than publishing ``{}``."""
    cls = _cls("Plain", "An ordinary env.")
    register(cls)
    assert reference_versions_of(cls) == {}
    assert REFERENCE_VERSIONS_KEY not in cls(num_envs=1, device="cpu").describe()


@pytest.mark.parametrize("bad", [[], "3.11.0", {}, {"mujoco": 3.11}, {"mujoco": ""}, {"": "3.11.0"}, {"mujoco": None}])
def test_a_malformed_reference_version_field_is_refused(bad):
    """A reference-version field the certifier cannot compare is refused."""
    with pytest.raises(ValueError, match="reference_versions"):
        register(_visual("Bad", visual_fidelity=dict(VF, **{REFERENCE_VERSIONS_KEY: bad})))
    with pytest.raises(ValueError, match="reference_versions"):
        register(_cls("BadParity", "doc", reference_versions=bad))


def _dmc_declaration_table() -> dict:
    """``stable_envs.envs.dmcontrol.VISUAL_FIDELITY``, read off the source rather than the module.

    Reading the source keeps the claim about the declaration table valid on hosts whose GL backend cannot
    register the cells. Only the module-level assignments the table is built from are executed: each
    module's ``VISUAL_FIDELITY`` and the ``_<DOMAIN>_AT*`` provenance names it reads. The tables are merged.
    """
    import ast

    pkg = ROOT / "stable_envs" / "envs" / "dmcontrol"
    want = {
        "_SWIMMER_AT",
        "_SWIMMER_AT_VERSIONS",
        "_SWIMMER_AT_MEASURED_WITH",
        "_QUADRUPED_AT",
        "_QUADRUPED_AT_MEASURED_WITH",
        "VISUAL_FIDELITY",
    }
    # ``__init__.py`` and ``_family.py`` are skipped: their ``VISUAL_FIDELITY`` is a VIEW of the
    # per-domain tables (``{n: globals()[n].visual_fidelity ...}``), not a declaration, and reading
    # it would either need the classes or hide the literals this check exists to read.
    skip = {"__init__.py", "_family.py"}
    paths = sorted(p for p in pkg.glob("*.py") if p.name not in skip)
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
            rows.append((name, rv, str((getattr(cls, "visual_fidelity", {}) or {}).get("measured_at", ""))))
    for name, vf in sorted(_dmc_declaration_table().items()):
        for key in (REFERENCE_VERSIONS_KEY, MEASURED_WITH_KEY):
            if key in vf:
                rows.append((f"dmcontrol.VISUAL_FIDELITY[{name}].{key}", vf[key], str(vf.get("measured_at", ""))))
    return rows


@pytest.mark.family("dmc")
def test_every_declared_reference_version_is_the_one_measured_at_names():
    """Every declared reference version (package and version) appears in the class's ``measured_at`` text."""
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
    """``reference_versions`` never names this repo's own renderer packages.

    The field gates certification; this repo's side goes in ``measured_with``, which is never compared.
    """
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
    """``reference_versions_of`` merges both sources, with ``visual_fidelity`` winning on conflicts."""
    cls = _visual(
        "Both",
        reference_versions={"mujoco": "3.10.0", "gymnasium": "1.3.0"},
        visual_fidelity=dict(VF, **{REFERENCE_VERSIONS_KEY: {"mujoco": "3.11.0"}}),
    )
    register(cls)
    assert reference_versions_of(cls) == {"mujoco": "3.11.0", "gymnasium": "1.3.0"}
