"""Count EGL contexts and MuJoCo renderers as they are created and released.

The rule in ``conformance.py`` — *create few, never release* — was, until now, a comment. A comment
cannot fail a build, and every one of the four incidents this module exists to prevent was a violation
of it that nothing noticed until the symptom appeared somewhere else entirely:

  1. a reference-env pool that CLOSED the oldest entry, so unrelated envs came back 44-90/255 apart;
  2. ``test_state_is_sufficient`` building two reference envs per env, 50 live contexts over 25 cells;
  3. a bake path constructing and closing a ``mujoco.Renderer`` per plate, ~175 releases;
  4. ``check_per_env_variation`` spreading a B-wide ``linspace`` of DISTINCT style values, so a family
     that bakes one static plate per distinct value needed B contexts for one knob.

Every one of those presents as *pixel differences in an environment that did nothing wrong*, minutes
later, in a different test. The delay is what made them expensive: three of the four were first
diagnosed as renderer bugs or as state-sufficiency violations, which is the one conclusion this repo
must never reach by accident.

What this module does is turn each of them into an immediate, named failure attributed to the test that
caused it. It patches the two constructors that matter and counts:

    mujoco.egl.GLContext.__init__ / .free      the EGL context itself
    mujoco.Renderer.__init__ / .close          which owns one, and whose per-plate churn was incident 3

``mujoco.GLContext`` and ``mujoco.gl_context.GLContext`` are the same class object as
``mujoco.egl.GLContext`` under ``MUJOCO_GL=egl`` (verified on the H200 node: all three ``is`` the same
type), so patching the class covers every alias, including the ones ogbench and gymnasium reach it
through. ``GLContext.__del__`` calls ``free``, so a context dropped by garbage collection is counted as
a release exactly like an explicit one — which is what incident 1 and incident 2 actually were.

WHAT THE NUMBERS MEAN, and why the release count is the strict one. Live contexts are not a scarce
resource: 96 simultaneous contexts were held on this hardware with zero pixel difference, and closing
one of 96 changed nothing in the survivors. So a cap on how many exist is a COST bound (each distinct
static appearance re-bakes a plate through MuJoCo) and is set generously. A RELEASE, by contrast, is
never legitimate inside a run and is checked at zero tolerance: one is a defect.

Usable outside pytest. ``install()`` is idempotent and safe to call when mujoco is absent, so a script
or a benchmark can wrap a suspicious section in ``take()`` / ``delta()`` without importing pytest.

ONE PIECE OF NOISE, and it is not this module's — checked rather than assumed, because the wrappers
appear in the traceback and would otherwise be blamed for it. At interpreter shutdown mujoco's own
``__del__`` chain (``Renderer.__del__`` -> ``close`` -> ``GLContext.free``) runs after the EGL bindings
have been torn down and raises. The same script, same node, with and without ``install()``::

    without:  Exception ignored in <function Renderer.__del__>
              mujoco/egl/__init__.py:131 free -> OpenGL.raw.EGL._errors.EGLError
    with:     Exception ignored in <function Renderer.__del__>
              mujoco/egl/__init__.py:129 free -> TypeError: 'NoneType' object is not callable

Both are the same teardown, both are printed and ignored by CPython, and neither is suppressed here:
everything this module does at that moment is inside a ``try``, but swallowing an exception raised by
mujoco's own ``free`` would be an observability tool changing the behaviour it observes.

THE MECHANISM BEHIND THAT ``EGLError``, now that it has been traced, because it has twice been read as
"the suite cannot run this family". mujoco and dm_control each call
``eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device)`` and get back the SAME ``EGLDisplay``
handle, and each registers ``atexit.register(eglTerminate, EGL_DISPLAY)`` on it — dm_control at import
of ``_render.pyopengl.egl_renderer``, mujoco at its first context. So at shutdown the display is
terminated while contexts are still alive, and any ``eglDestroyContext`` running afterwards raises. It
is exit-time noise after the test report is written, not a run-time fault: measured on the DM Control
conformance run (16 envs, venvs/dmc) the suite finished 208 passed / 48 skipped, rc 0, 16 contexts
created and 0 released. A DM Control family with no parity coverage is a MISSING IMPORT, not this.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from dataclasses import dataclass, field

#: Recorded creation/release sites, most recent last. Bounded so a long run cannot grow without limit;
#: the cap is well above any legitimate count and any diagnosis needs the FIRST few sites, not all of
#: them, so the tail is what gets dropped.
MAX_SITES = 512

#: Frames of Python stack kept per event. Enough to name the caller and its caller — which is what
#: identifies "a bake path" or "a pool eviction" — without turning the log into a transcript.
SITE_DEPTH = 12


@dataclass
class Counts:
    """Cumulative creation/release counts, per kind."""

    context_created: int = 0
    context_freed: int = 0
    renderer_created: int = 0
    renderer_closed: int = 0
    sites: list = field(default_factory=list)

    @property
    def context_live(self) -> int:
        return self.context_created - self.context_freed


_LOCK = threading.Lock()
_COUNTS = Counts()
_PEAK_LIVE = 0
_INSTALLED = False
_PATCHED: list[str] = []


def _record(kind: str, action: str) -> None:
    """Count one event and remember where it came from."""
    global _PEAK_LIVE
    with _LOCK:
        setattr(_COUNTS, f"{kind}_{action}", getattr(_COUNTS, f"{kind}_{action}") + 1)
        if _COUNTS.context_live > _PEAK_LIVE:
            _PEAK_LIVE = _COUNTS.context_live
        if len(_COUNTS.sites) < MAX_SITES:
            # The two innermost frames are this function and the patched wrapper; drop them so the
            # site names the CALLER, which is the thing a failure message has to point at.
            stack = traceback.extract_stack(limit=SITE_DEPTH + 2)[:-2]
            _COUNTS.sites.append((f"{kind}.{action}", [f"{f.filename}:{f.lineno} {f.name}" for f in stack[-6:]]))


def _count_one(obj, kind: str, action: str) -> None:
    if action in ("freed", "closed"):
        # Count the RELEASE, not the call. ``GLContext.__del__`` calls ``free()`` again on an object
        # that was freed explicitly, so one context was reported as two releases and left the live
        # count at -1 (observed on the real driver, mujoco/egl/__init__.py:138). The verdict does not
        # change — one release is already a failure — but a wrong number in a failure message is a
        # wrong number, and this one is free to get right.
        try:
            if getattr(obj, "_se_released", False):
                return
            obj._se_released = True
        except Exception:  # noqa: BLE001
            pass  # __slots__ or a C type: fall back to counting
    _record(kind, action)


def _wrap(cls, name: str, kind: str, action: str) -> None:
    """Count calls to ``cls.name`` as ``kind.action``. Idempotent per method."""
    fn = getattr(cls, name, None)
    if fn is None or getattr(fn, "_se_guarded", False):
        return

    def wrapper(self, *a, __fn=fn, __kind=kind, __action=action, **k):
        out = __fn(self, *a, **k)
        try:
            _count_one(self, __kind, __action)
        except Exception:  # noqa: BLE001
            # Never let accounting change what the program does. These wrappers run from ``__del__``
            # during interpreter shutdown, when module globals have already been set to None, and an
            # exception raised there is printed as "Exception ignored in" noise that looks like a
            # defect in the thing being measured.
            pass
        return out

    wrapper._se_guarded = True  # type: ignore[attr-defined]
    wrapper.__name__ = name
    wrapper.__qualname__ = f"{cls.__name__}.{name}"
    setattr(cls, name, wrapper)
    _PATCHED.append(f"{cls.__module__}.{cls.__name__}.{name}")


def _wrap_dm_control() -> None:
    """Count dm_control's OWN EGL contexts, which are not ``mujoco.egl.GLContext``.

    This was a hole straight through the rule the module exists to enforce. dm_control does not use
    mujoco's GL backend: it ships ``dm_control._render.pyopengl.egl_renderer.EGLContext``, built on
    the SAME ``EGLDisplay``, and every ``Physics`` that renders owns one. Measured on the DM Control
    conformance run (16 envs, venvs/dmc): the guard reported "16 EGL contexts created, 0 released"
    and all 16 of those were OUR plate-bake renderers — dm_control's 16 reference contexts were
    invisible, and a reference-env cache that dropped one (incident 1, in this exact family) would
    have gone uncounted while producing precisely the silent cross-context corruption the rule
    forbids.

    Patched on the shared base rather than on ``EGLContext``, because ``ContextBase.__init__`` and
    ``ContextBase._free_unconditionally`` are the single points every backend goes through —
    ``free()``, ``__del__`` and dm_control's own ``atexit`` handler all funnel into the latter, so one
    patch covers explicit closes, garbage collection and shutdown alike.

    Never imports dm_control. If the module is not already loaded, this session was not going to
    create one of these contexts before something else imported it, and importing it here would cost
    every unrelated session an EGL display and a couple of seconds.
    """
    mod = sys.modules.get("dm_control._render.base")
    cls = getattr(mod, "ContextBase", None) if mod is not None else None
    if cls is None:
        return
    _wrap(cls, "__init__", "context", "created")
    _wrap(cls, "_free_unconditionally", "context", "freed")


def install() -> list[str]:
    """Patch the constructors. Idempotent, and CHEAP TO CALL AGAIN; returns what is patched so far.

    A no-op when mujoco is not importable, because the non-MuJoCo half of the registry (the 2-D
    rasterizer families) has no EGL context to leak and must not be made to depend on one being
    installable. Callers get an empty list and can say so.

    Re-callable rather than once-only, because the thing to patch is not all importable at the moment
    the plugin starts. ``pytest_configure`` runs before any test module is collected, so
    ``stable_envs`` — and therefore ``dm_control`` — has not been imported yet, and a one-shot install
    would permanently miss every context class that arrives with an env family. Each pass only patches
    what is ALREADY in ``sys.modules`` (except mujoco, which it will import), so calling it per test
    costs a few dict lookups and never drags a 2 s dependency or an EGL display into a session that
    was not going to use one.
    """
    global _INSTALLED
    _INSTALLED = True
    _wrap_dm_control()
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return list(_PATCHED)

    wrap = _wrap

    # The GL context. Under MUJOCO_GL=egl, mujoco.GLContext IS mujoco.egl.GLContext, so one patch
    # covers ogbench, gymnasium and any direct user. The osmesa/glfw backends are patched too when
    # present, so that a node configured differently is instrumented rather than silently unguarded.
    seen = set()
    for modname in ("egl", "osmesa", "glfw"):
        try:
            mod = __import__(f"mujoco.{modname}", fromlist=["GLContext"])
        except Exception:  # noqa: BLE001
            continue
        cls = getattr(mod, "GLContext", None)
        if cls is None or id(cls) in seen:
            continue
        seen.add(id(cls))
        wrap(cls, "__init__", "context", "created")
        wrap(cls, "free", "context", "freed")
    cls = getattr(mujoco, "GLContext", None)
    if cls is not None and id(cls) not in seen:
        wrap(cls, "__init__", "context", "created")
        wrap(cls, "free", "context", "freed")

    # The renderer. It owns a context, so its creations are already counted above; it is instrumented
    # separately because incident 3 was specifically a renderer built and closed PER BAKED PLATE, and
    # naming that in the failure message is the difference between a five-minute fix and an afternoon.
    rcls = getattr(mujoco, "Renderer", None)
    if rcls is not None:
        wrap(rcls, "__init__", "renderer", "created")
        wrap(rcls, "close", "renderer", "closed")
    return list(_PATCHED)


def take() -> Counts:
    """A snapshot of the counters, safe to hold across arbitrary work."""
    with _LOCK:
        return Counts(
            _COUNTS.context_created,
            _COUNTS.context_freed,
            _COUNTS.renderer_created,
            _COUNTS.renderer_closed,
            list(_COUNTS.sites),
        )


def delta(before: Counts, after: Counts | None = None) -> dict:
    """What happened between two snapshots, with the sites that were new in that window."""
    after = take() if after is None else after
    return {
        "context_created": after.context_created - before.context_created,
        "context_freed": after.context_freed - before.context_freed,
        "renderer_created": after.renderer_created - before.renderer_created,
        "renderer_closed": after.renderer_closed - before.renderer_closed,
        "context_live": after.context_live,
        "sites": after.sites[len(before.sites) :],
    }


def peak_live() -> int:
    return _PEAK_LIVE


def totals() -> dict:
    c = take()
    return {
        "context_created": c.context_created,
        "context_freed": c.context_freed,
        "context_live": c.context_live,
        "context_peak_live": _PEAK_LIVE,
        "renderer_created": c.renderer_created,
        "renderer_closed": c.renderer_closed,
        "patched": list(_PATCHED),
    }


def format_sites(sites, limit: int = 3) -> str:
    """The first few recorded sites, as a readable block for a failure message."""
    out = []
    for tag, frames in sites[:limit]:
        out.append(f"    {tag}")
        out.extend(f"      {f}" for f in frames[-4:])
    if len(sites) > limit:
        out.append(f"    ... and {len(sites) - limit} more")
    return "\n".join(out)


# ------------------------------------------------------------------ policy


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def limits() -> dict:
    """The thresholds, overridable from the environment so a bisect can loosen them without an edit.

    ``per_test`` is a COST bound, not a safety one, and it is set from what the suite measurably needs:
    the most any healthy test creates is the first test on an OGBench cell, which builds the cached
    ogbench env, its two plate renderers and the reference env. The incidents overshot it by an order
    of magnitude (B contexts for one knob at B=8 or 16; two per env across 25 cells), so a bound a
    little above the honest maximum separates them cleanly.

    ``session_live`` is far above the 96 that were measured to be harmless, because live contexts are
    not the failure mode — it exists only to catch something genuinely runaway before the driver does.
    It SCALES WITH THE REGISTRY, and that is the whole point rather than a convenience.

    A context here is created once per env that owns one and then deliberately never released, so the
    legitimate live count is a function of how many envs the interpreter holds — not a constant. As a
    fixed 192 it was comfortable at a registry of 64 and became a false positive the moment the whole
    library moved into one interpreter: a 191-env run reached **193 live contexts** doing exactly what
    the design requires, and then every subsequent test failed `EGL_CONTEXTS_RUNAWAY` having created
    nothing itself. 1,790 violations, 1,776 of them tests that touched no context at all.

    Scaling it keeps the tripwire pointed at what it was aimed at. Runaway creation is a RATE — a
    context per state, per style value or per call — and that is caught by ``per_test`` and
    ``EGL_CONTEXTS_SCALE_WITH_BATCH``, both of which are per-test and unaffected by registry size.
    The floor of 192 is kept so no existing run gets a LOOSER bound than it had.

    ``freed`` has no knob. Zero is the rule.
    """
    try:  # import here: this module must not need the registry
        from stable_envs import REGISTRY

        n_envs = len(REGISTRY)
    except Exception:  # noqa: BLE001  — a guard must never be the thing that fails
        n_envs = 0
    # One per env, plus reference envs and plate renderers for the families that bake them, plus slack.
    scaled = max(192, 2 * n_envs + 64)
    return {
        "per_test": _int_env("SE_MAX_CONTEXTS_PER_TEST", 8),
        "session_live": _int_env("SE_MAX_LIVE_CONTEXTS", scaled),
    }


def violations(d: dict, batch: int | None = None) -> list[tuple[str, str]]:
    """``(name, message)`` for every rule the delta ``d`` breaks. Empty when the test behaved.

    Names are stable and greppable on purpose — ``EGL_CONTEXT_RELEASED`` in a CI log should be enough
    to know which of the four incidents has come back without reading the rest of the output.
    """
    lim = limits()
    out = []
    if d["context_freed"]:
        out.append(
            (
                "EGL_CONTEXT_RELEASED",
                (
                    f"this test RELEASED {d['context_freed']} EGL context(s). Releasing one corrupts rendering "
                    f"in every other live context in the process, silently, and the damage surfaces later as "
                    f"pixel differences in an unrelated env — which is how a reference-env pool, a two-envs-"
                    f"per-test parity check and a per-plate mujoco.Renderer each cost this repo a day. Cache "
                    f"the object and keep it for the life of the process; do not close it.\n"
                    f"  released at:\n{format_sites([s for s in d['sites'] if s[0].endswith('freed')])}"
                ),
            )
        )
    if d["renderer_closed"]:
        out.append(
            (
                "MUJOCO_RENDERER_CLOSED",
                (
                    f"this test closed {d['renderer_closed']} mujoco.Renderer(s). A Renderer owns an EGL "
                    f"context, so closing one is the same defect as EGL_CONTEXT_RELEASED — it is reported "
                    f"separately because the shape it took here before was a renderer constructed and closed "
                    f"per baked plate, ~175 releases across 25 envs.\n"
                    f"  closed at:\n{format_sites([s for s in d['sites'] if s[0].endswith('closed')])}"
                ),
            )
        )
    n = d["context_created"]
    if batch and batch >= 4 and n >= batch:
        out.append(
            (
                "EGL_CONTEXTS_SCALE_WITH_BATCH",
                (
                    f"this test created {n} EGL contexts with a batch of {batch}. Context creation that "
                    f"tracks the batch size means a per-environment value is being turned into a per-"
                    f"environment SCENE — a B-wide ramp of distinct style values re-bakes a MuJoCo plate per "
                    f"row. Two distinct values falsify a collapse to environment 0 just as well and cost two "
                    f"bakes; see check_per_env_variation and uniform_style.\n"
                    f"  created at:\n{format_sites([s for s in d['sites'] if s[0].endswith('created')])}"
                ),
            )
        )
    elif n > lim["per_test"]:
        out.append(
            (
                "EGL_CONTEXTS_PER_TEST",
                (
                    f"this test created {n} EGL contexts, over the limit of {lim['per_test']} "
                    f"(SE_MAX_CONTEXTS_PER_TEST). Nothing in a healthy run needs that many: the most expensive "
                    f"legitimate test builds one cached ogbench env, its plate renderers and one reference "
                    f"env. Creating them per state, per style value or per call is the bug.\n"
                    f"  created at:\n{format_sites([s for s in d['sites'] if s[0].endswith('created')])}"
                ),
            )
        )
    if d["context_live"] > lim["session_live"]:
        out.append(
            (
                "EGL_CONTEXTS_RUNAWAY",
                (
                    f"{d['context_live']} EGL contexts are live, over {lim['session_live']} "
                    f"(SE_MAX_LIVE_CONTEXTS). 96 simultaneous contexts were measured to be harmless on this "
                    f"hardware, so this bound is not a safety margin — it is a tripwire for something creating "
                    f"contexts in a loop."
                ),
            )
        )
    return out
