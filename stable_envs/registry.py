"""``make`` / ``spec`` / ``list_envs``, plus the registration decorator and its declaration checks.

THE CLASSES ARE CANONICAL AND THIS MODULE HOLDS NO SEMANTICS. ``snv.PushT`` is the environment;
``snv.make("swm/PushT-v1")`` is a string that finds it, for configs, serialisation and interop. The
registry knows three things: the class a name refers to, the SWM ids that are aliases for it, and
whether a class declared its fidelity tier completely before it was allowed in.

Auto-discovery of ``stable_envs/envs/*`` is kept (``discover()``), and so is the rule that a missing
optional dependency costs exactly the family that needs it: the failure is recorded in
``import_errors()``, never silent. Discovery is performed by ``importlib`` at CALL time rather than by
an import statement at module scope, deliberately — contract §8 requires the core modules to import
nothing from ``envs/``, and a static import here would both invert that arrow and make importing the
registry pay for every optional dependency in the repository.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from ._config import log as _log
from .env import (
    FIDELITY_TIERS,
    MEASURED_WITH_KEY,
    REFERENCE_VERSIONS_KEY,
    RENDER_DECODERS,
    REQUIRED_VISUAL_FIDELITY,
    Env,
    _check_reference_versions_shape,
)

#: class name -> class, populated by ``register``. One place, no parallel tables.
REGISTRY: dict[str, type[Env]] = {}


def check_fidelity_declaration(cls: type[Env]) -> None:
    """A class must declare exactly one tier, completely. Enforced at registration, not by review.

    The failure this exists to prevent is an env quietly adopting the weaker claim: setting
    ``fidelity_tier = "visual"`` is a way to make the parity assertion stop applying, so the cost of
    doing it has to be paid at the same moment — measured numbers, thresholds, a stated mechanism, and
    the words in the class docstring that a reader will actually see.

    The docstring check is mechanical on purpose. ``describe()`` and the registry listing mark the
    tier, but the thing a person reads first is ``help(cls)``, and a marker that depends on remembering
    to write it is the kind of rule this repo has learned to make impossible instead.
    """
    tier = getattr(cls, "fidelity_tier", "parity")
    if tier not in FIDELITY_TIERS:
        raise ValueError(f"{cls.__name__}: fidelity_tier must be one of {FIDELITY_TIERS}, got {tier!r}")
    vf = dict(getattr(cls, "visual_fidelity", {}) or {})
    # The reference-version field is optional on both tiers and malformed on neither: a version the
    # certifier cannot compare is worse than none, because it looks like provenance.
    if REFERENCE_VERSIONS_KEY in cls.__dict__ or getattr(cls, REFERENCE_VERSIONS_KEY, None):
        _check_reference_versions_shape(cls, REFERENCE_VERSIONS_KEY, getattr(cls, REFERENCE_VERSIONS_KEY))
    if REFERENCE_VERSIONS_KEY in vf:
        _check_reference_versions_shape(
            cls, f"visual_fidelity[{REFERENCE_VERSIONS_KEY!r}]", vf[REFERENCE_VERSIONS_KEY]
        )
    if MEASURED_WITH_KEY in vf:
        _check_reference_versions_shape(cls, f"visual_fidelity[{MEASURED_WITH_KEY!r}]", vf[MEASURED_WITH_KEY])
    if tier == "parity":
        if vf:
            raise ValueError(
                f"{cls.__name__} declares fidelity_tier='parity' and also carries visual_fidelity. The "
                f"two tiers make different claims and an env makes exactly one of them; drop the dict "
                f"or declare the visual tier."
            )
        return
    missing = [k for k in REQUIRED_VISUAL_FIDELITY if k not in vf]
    if missing:
        raise ValueError(
            f"{cls.__name__} declares the VISUAL tier without {missing}. The tier exists to let an env "
            f"state a measured, weaker claim; an unmeasured weaker claim is just a missing test."
        )
    if not getattr(cls, "visual_note", ""):
        raise ValueError(
            f"{cls.__name__} declares the VISUAL tier with no visual_note. The note is what tells a "
            f"reader which residual they are accepting and what it disqualifies the frames from."
        )
    if getattr(cls, "parity_interior_min", None) is not None:
        raise ValueError(
            f"{cls.__name__} declares the VISUAL tier but leaves parity_interior_min at "
            f"{cls.parity_interior_min!r}. Set it to None: the parity gate does not apply to this env, "
            f"and leaving a number there is how a visual env ends up being compared against the strong "
            f"claim, or worse, how the strong claim ends up being loosened to accommodate it."
        )
    doc = (cls.__doc__ or "").upper()
    if "VISUAL TIER" not in doc:
        raise ValueError(
            f"{cls.__name__} declares the VISUAL tier and does not say so in its class docstring. "
            f"Put the words 'VISUAL TIER' in it, with what the residual is; help({cls.__name__}) is "
            f"where a user looks first and it must not read like a parity env."
        )


def check_decoder_declaration(cls: type[Env]) -> None:
    """A ``"learned"`` decoder is declared completely, or the class does not register.

    Same shape as the tier check and for the same reason: a learned decoder is a way to make the
    frames better on every declared number at once, so the cost of adopting it has to be paid at the
    same moment — the tier it is allowed in, the words a reader will actually see, and a named
    checkpoint in ``visual_fidelity`` so the declaration cannot be read as describing the renderer.
    """
    dec = getattr(cls, "render_decoder", "renderer")
    if dec not in RENDER_DECODERS:
        raise ValueError(f"{cls.__name__}: render_decoder must be one of {RENDER_DECODERS}, got {dec!r}")
    if dec == "renderer":
        return
    if getattr(cls, "fidelity_tier", "parity") != "visual":
        raise ValueError(
            f"{cls.__name__} declares render_decoder='learned' outside the VISUAL tier. A frame a "
            f"network drew is not the official frame and cannot make the parity claim, whatever it "
            f"measures; the two must not be able to be declared together."
        )
    vf = dict(getattr(cls, "visual_fidelity", {}) or {})
    if not str(vf.get("decoder", "")).strip():
        raise ValueError(
            f"{cls.__name__} declares a LEARNED decoder and its visual_fidelity has no 'decoder' "
            f"entry naming the checkpoint the numbers were measured through. Those numbers describe "
            f"a (renderer, checkpoint) pair; without the second half they read as a claim about the "
            f"renderer, which is the one misreading this field exists to prevent."
        )
    missing = [
        k
        for k in ("mean_abs", "within8_frac", "ssim", "mean_abs_max", "within8_min", "ssim_min", "measured_at")
        if k not in (getattr(cls, "variation_fidelity", {}) or {})
    ]
    if missing:
        raise ValueError(
            f"{cls.__name__} declares a LEARNED decoder and its variation_fidelity is missing "
            f"{missing}. A network fitted at one appearance is measured at 9.06/255 under a declared "
            f"variation it never saw, against 8.91 for doing nothing and 7.93 for the render it "
            f"replaces -- so for this decoder the default-appearance declaration is not the claim, it "
            f"is the best case. The variation regime is measured or the class does not register."
        )
    doc = (cls.__doc__ or "").upper()
    if "LEARNED" not in doc:
        raise ValueError(
            f"{cls.__name__} draws its frames through a LEARNED decoder and does not say so in its "
            f"class docstring. Put the word 'LEARNED' in it, with what the network supplies; the "
            f"difference between 'MuJoCo's renderer approximated' and 'a learned approximation of "
            f"MuJoCo's renderer' is invisible in the picture and help({cls.__name__}) is where "
            f"someone finds it."
        )


def register(cls: type[Env]) -> type[Env]:
    """Decorator. Duplicate names are an error, not a silent overwrite; so is a half-declared tier."""
    key = cls.__name__
    if key in REGISTRY and REGISTRY[key] is not cls:
        raise ValueError(
            f"env name collision: {key} is already registered by "
            f"{REGISTRY[key].__module__}; two classes cannot share a lookup key"
        )
    check_fidelity_declaration(cls)
    check_decoder_declaration(cls)
    REGISTRY[key] = cls
    return cls


def registry_table() -> list[dict[str, Any]]:
    """One row per registered env, with its FIDELITY TIER. The listing, in a form that can be printed.

    ``sorted(REGISTRY)`` is a list of names and says nothing about what the frames are worth. Anything
    that shows a user the registry should show this instead, so that the one env in the package whose
    frames are not the official frames cannot be picked by name alone.
    """
    # ``state_dim`` is CLASS-level here and is None for the envs that compute it at construction
    # (Fetch from its MJCF, RoboCasa from its measured live-qpos set). Reporting the class default of
    # 0 for those would be a number that is wrong rather than absent, so it is not reported.
    return [
        {
            "name": n,
            "tier": getattr(c, "fidelity_tier", "parity"),
            "decoder": getattr(c, "render_decoder", "renderer"),
            "state_dim": (int(c.state_dim) or None) if isinstance(getattr(c, "state_dim", None), int) else None,
            "reference_id": getattr(c, "reference_id", None),
            "module": c.__module__,
        }
        for n, c in sorted(REGISTRY.items())
    ]


def format_registry(width: int = 34) -> str:
    """The registry as text, with visual-tier envs marked. ``print(stable_envs.format_registry())``."""
    rows = registry_table()
    n_visual = sum(r["tier"] == "visual" for r in rows)
    n_learned = sum(r["decoder"] == "learned" for r in rows)
    out = [
        f"{len(rows)} environments; {len(rows) - n_visual} parity, {n_visual} visual"
        + (f" ({n_learned} of them through a LEARNED decoder)" if n_learned else "")
    ]
    for r in rows:
        mark = (
            "   [VISUAL TIER — LEARNED decoder, not the official pixels]"
            if r["decoder"] == "learned"
            else "   [VISUAL TIER — not the official pixels]"
            if r["tier"] == "visual"
            else ""
        )
        out.append(f"  {r['name']:<{width}}{mark}")
    if n_visual:
        out.append(
            "  visual tier: fast re-render of the same scene by a different rasterizer. Good "
            "for live training data; NOT a substitute for a stored dataset when a policy will "
            "be evaluated on official pixels. See each env's visual_fidelity / visual_note."
        )
    if n_learned:
        out.append(
            "  learned decoder: that re-render was then passed through a fixed neural "
            "checkpoint fitted to predict the official frame, so it is closer to the official "
            "pixels on every declared number AND part of it is synthesised rather than drawn. "
            "The frame is still a pure function of (state, style). See render_decoder."
        )
    return "\n".join(out)


#: SWM id -> the stable-envs cell(s) that serve it, VERBATIM from the support matrix
#: (``tests/fixtures/regression/support_matrix.json``, 42 rows, 40 pixel-producing and 2 symbolic).
#:
#: These are aliases, not a second registry: every id resolves to a class that is already registered
#: under its canonical name, and ``tests/unit/test_registry.py`` asserts this table against
#: the frozen matrix row for row, so the two cannot drift. An id with SEVERAL cells (a DM-Control id
#: that covers two tasks, the two OGB maze layouts) does not pick one for the caller — the cells are
#: different environments and choosing silently is how a benchmark ends up reporting the wrong one.
SWM_IDS: dict[str, tuple[str, ...]] = {
    "swm/AcrobotControl-v1": ("Acrobot",),
    "swm/AcrobotDMControl-v0": ("DMCAcrobotSwingup",),
    "swm/BallInCupDMControl-v0": ("DMCBallInCupCatch",),
    "swm/CartPoleControl-v1": ("CartPole",),
    "swm/CartpoleDMControl-v0": ("DMCCartpoleBalance", "DMCCartpoleSwingup"),
    "swm/CheetahDMControl-v0": ("DMCCheetahRun",),
    "swm/CraftaxClassicPixels-v1": ("CraftaxClassic",),
    "swm/CraftaxClassicSymbolic-v1": (),
    "swm/CraftaxPixels-v1": ("CraftaxFull",),
    "swm/CraftaxSymbolic-v1": (),
    "swm/FetchPickAndPlace-v3": ("FetchPickAndPlace",),
    "swm/FetchPickAndPlaceDense-v3": ("FetchPickAndPlaceDense",),
    "swm/FetchPickAndPlaceDict-v3": ("FetchPickAndPlaceDict",),
    "swm/FetchPush-v3": ("FetchPush",),
    "swm/FetchPushDense-v3": ("FetchPushDense",),
    "swm/FetchPushDict-v3": ("FetchPushDict",),
    "swm/FetchReach-v3": ("FetchReach",),
    "swm/FetchReachDense-v3": ("FetchReachDense",),
    "swm/FetchReachDict-v3": ("FetchReachDict",),
    "swm/FetchSlide-v3": ("FetchSlide",),
    "swm/FetchSlideDense-v3": ("FetchSlideDense",),
    "swm/FetchSlideDict-v3": ("FetchSlideDict",),
    "swm/FingerDMControl-v0": ("DMCFingerTurnHard",),
    "swm/HopperDMControl-v0": ("DMCHopperHop",),
    "swm/HumanoidDMControl-v0": ("DMCHumanoidStand",),
    "swm/ManipulatorDMControl-v0": ("DMCManipulatorBringBall",),
    "swm/MountainCarContinuousControl-v0": ("MountainCarContinuous",),
    "swm/MountainCarControl-v0": ("MountainCar",),
    "swm/OGBCube-v0": ("OGBCube",),
    "swm/OGBMaze-v0": ("PointmazeLargeNavigate", "PointmazeLargeStitch"),
    "swm/OGBScene-v0": ("OGBScene",),
    "swm/PFRocketLanding-v0": ("PFRocketLanding",),
    "swm/PendulumControl-v1": ("Pendulum",),
    "swm/PendulumDMControl-v0": ("DMCPendulumSwingup",),
    "swm/Piecewise-v0": ("Piecewise", "PiecewiseZones"),
    "swm/PushT-Discrete-v1": ("PushT",),
    "swm/PushT-v1": ("PushT",),
    "swm/QuadrupedDMControl-v0": ("DMCQuadrupedWalk",),
    "swm/ReacherDMControl-v0": ("DMCReacherHard",),
    "swm/SimplePointMaze-v0": ("SimplePointMaze",),
    "swm/TwoRoom-v1": ("TwoRoom", "TwoRoomTarget"),
    "swm/WalkerDMControl-v0": ("DMCWalkerWalk",),
}


#: The two symbolic ids: ``stable_worldmodel`` serves them and this package deliberately does not,
#: because there is no frame to be a function of the state. Recorded so ``make`` can say that rather
#: than "unknown env".
SYMBOLIC_IDS = tuple(sorted(k for k, v in SWM_IDS.items() if not v))


def canonical_names(name: str) -> tuple[str, ...]:
    """The class name(s) ``name`` refers to: itself if it is one, else its SWM id's cells."""
    if name in REGISTRY:
        return (name,)
    return SWM_IDS.get(name, ())


def _resolve(name: str, what: str) -> type[Env]:
    if name in REGISTRY:
        return REGISTRY[name]
    cells = SWM_IDS.get(name)
    if cells is None:
        raise KeyError(
            f"unknown env {name!r}; known: {sorted(REGISTRY)}"
            + (f"; known SWM ids: {sorted(SWM_IDS)}" if name.startswith("swm/") else "")
        )
    if not cells:
        raise KeyError(
            f"{name} is one of the two SYMBOLIC stable_worldmodel ids ({', '.join(SYMBOLIC_IDS)}): "
            f"its observation is a symbol array, not a frame, so this package has nothing to render "
            f"for it and does not pretend to. Use stable_worldmodel's own env."
        )
    live = [c for c in cells if c in REGISTRY]
    if len(cells) > 1:
        raise KeyError(
            f"{name} maps onto {len(cells)} stable-envs cells ({', '.join(cells)}), which are "
            f"different environments — the id names the family and the cell names the task. Ask for "
            f"one by its class name, e.g. {what}({cells[0]!r})."
        )
    if not live:
        errs = import_errors()
        raise KeyError(
            f"{name} maps onto {cells[0]}, which is not registered in this interpreter. Import "
            f"errors recorded during discovery: {errs or 'none'} — most often an optional dependency "
            f"the family needs."
        )
    return REGISTRY[live[0]]


def make(name: str, **kwargs) -> Env:
    """``make("PushT", num_envs=64)`` or ``make("swm/PushT-v1", device="cuda")``."""
    return _resolve(name, "make")(**kwargs)


def spec(name: str) -> dict[str, Any]:
    """Everything about an env that can be known WITHOUT constructing one.

    Construction needs a device, and for several families a GPU, an EGL context or a compiled scene —
    so a ``spec`` that instantiated would be unanswerable on the machine where the question is usually
    asked (a laptop choosing what to run). Every field here is a class attribute or a registry fact.
    """
    cls = _resolve(name, "spec")
    ids = tuple(sorted(i for i, cells in SWM_IDS.items() if cls.__name__ in cells))
    dim = getattr(cls, "state_dim", None)
    return {
        "name": cls.__name__,
        "class": cls,
        "module": cls.__module__,
        "family": cls.__module__.rsplit(".", 1)[-1],
        "state_dim": int(dim) if isinstance(dim, int) and dim else None,
        "state_names": list(getattr(cls, "state_names", ()) or ()),
        "default_size": getattr(cls, "default_size", None),
        "reference_id": getattr(cls, "reference_id", None),
        "fidelity_tier": getattr(cls, "fidelity_tier", "parity"),
        "render_decoder": getattr(cls, "render_decoder", "renderer"),
        "action_dim": getattr(cls, "action_dim", None),
        "has_dynamics": any(getattr(cls, m, None) is not getattr(Env, m, None) for m in ("step", "transition")),
        "swm_ids": ids,
    }


def list_envs(family: str | None = None) -> list[str]:
    """Registered class names, sorted. ``family`` filters by the module they live in ("pusht")."""
    names = sorted(REGISTRY)
    if family is None:
        return names
    return [n for n in names if REGISTRY[n].__module__.rsplit(".", 1)[-1] == family]


def list_families() -> list[str]:
    """The env modules that contributed at least one registered class."""
    return sorted({c.__module__.rsplit(".", 1)[-1] for c in REGISTRY.values()})


# ------------------------------------------------------------------ auto-discovery

#: module name -> the exception that stopped it importing
_IMPORT_ERRORS: dict[str, str] = {}
_DISCOVERED = False


#: Private modules that still register a public class. ``pkgutil`` skips ``_``-prefixed names, so
#: these would otherwise never load. The error key is the historical module name a missing extra was
#: recorded under (``craftax_full``), not the private filename.
_EXPLICIT_ENV_MODULES: tuple[tuple[str, str], ...] = (("_craftax_full", "craftax_full"),)


def _import_env_module(mod_name: str, error_key: str) -> None:
    """Import one env module. A missing extra costs that family; anything else is recorded as unexpected.

    Never raised to the caller: discovery must not fail the package import.
    """
    try:
        importlib.import_module(f"stable_envs.envs.{mod_name}")
    except (ImportError, ModuleNotFoundError) as e:
        _IMPORT_ERRORS[error_key] = f"{type(e).__name__}: {e}"
        _log("DEBUG", f"stable_envs.envs.{mod_name} did not import (missing extra): {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001 — recorded, never fatal
        _IMPORT_ERRORS[error_key] = f"{type(e).__name__}: {e} (unexpected; this is not an optional-dependency miss)"
        _log(
            "DEBUG",
            f"stable_envs.envs.{mod_name} did not import (unexpected, not an "
            f"optional-dep miss): {type(e).__name__}: {e}",
        )


def discover(force: bool = False) -> dict[str, str]:
    """Import every module under ``stable_envs/envs`` so its registered classes exist.

    Failures are RECORDED, never silent: most env modules need an optional dependency (``pymunk`` for
    PushT, ``minigrid`` for the grid family, ``mujoco``/``ogbench`` for the mazes), and a machine
    without one should lose exactly that family rather than the package. ``ImportError`` /
    ``ModuleNotFoundError`` are missing extras; any other exception is still recorded, with a message
    that says it is not an optional-dependency miss.
    """
    global _DISCOVERED
    if _DISCOVERED and not force:
        return dict(_IMPORT_ERRORS)
    from ._mujoco_gl import configure_mujoco_gl

    configure_mujoco_gl()
    pkg = importlib.import_module("stable_envs.envs")
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):
            continue
        _import_env_module(mod.name, mod.name)
    for mod_name, error_key in _EXPLICIT_ENV_MODULES:
        _import_env_module(mod_name, error_key)
    _DISCOVERED = True
    return dict(_IMPORT_ERRORS)


def import_errors() -> dict[str, str]:
    """``module -> error`` for env modules that could not be imported."""
    return dict(_IMPORT_ERRORS)


__all__ = [
    "REGISTRY",
    "SWM_IDS",
    "SYMBOLIC_IDS",
    "canonical_names",
    "check_decoder_declaration",
    "check_fidelity_declaration",
    "discover",
    "format_registry",
    "import_errors",
    "list_envs",
    "list_families",
    "make",
    "register",
    "registry_table",
    "spec",
]
