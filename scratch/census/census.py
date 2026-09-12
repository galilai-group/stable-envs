"""What does THIS interpreter register, and what does each env DECLARE? One JSON line, for merging.

The registry is not a single number, because no interpreter can hold every family: robocasa pins
mujoco 3.3.1 while mujoco_warp needs >= 3.11, JAX and torch CUDA builds disagree, and several families
ship their own fork of a reference package. The honest count is the union over interpreters, with the
per-interpreter counts kept so a reader can see which venv contributes what.

``declarations`` was added for ``scratch/swm_matrix.py``, which needs each env's ``reference_id``,
fidelity tier and step behaviour to join the registry against ``stable_worldmodel``'s. Everything in
it is read off the CLASS -- no env is constructed, nothing touches a device -- so this stays as cheap
as the name census it grew out of and runs on an interpreter with no GPU. The cost of that choice is
that attributes an env only sets in ``__init__`` (Fetch computes ``state_dim`` from its MuJoCo model)
come back null, which is reported as null rather than filled in from somewhere else.

``merge_census.py`` reads ``registry`` and ignores the rest, so it is unaffected.
"""
from __future__ import annotations

import inspect
import json
import sys

#: Class attributes worth carrying across interpreters. Each is a DECLARATION -- something the env
#: states about itself and the conformance suite then holds it to -- which is exactly the kind of
#: fact a coverage matrix may quote without re-measuring it.
DECLARED = ("name", "reference_id", "fidelity_tier", "render_decoder", "visual_fidelity",
            "visual_note", "parity_interior_min", "parity_note", "state_dim", "default_size",
            "state_names", "step_reference_exact", "step_reference_control", "step_reference_gate")


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def _backends(cls, base) -> list[str]:
    """Which ``stable_envs.render`` backends draw this env, read off the module that defines its draw.

    Deliberately not a class-name heuristic. An env either overrides ``render`` (and then its own
    module is what draws) or leaves the base ``render``, which builds a ``Scene2D`` from ``scene`` and
    hands it to the fused rasterizer. Either way the defining module's imports name the backends it
    actually reaches, which is a fact about the code rather than about what the class is called.
    """
    try:
        fn = cls.render if cls.render is not base.render else cls.scene
        mod = inspect.getmodule(fn)
        hits = set()
        for v in vars(mod).values():
            m = v if inspect.ismodule(v) else inspect.getmodule(v)
            n = getattr(m, "__name__", "") or ""
            if n.startswith("stable_envs.render."):
                hits.add(n.split(".")[-1])
        return sorted(hits)
    except Exception:
        return []


out = {"python": sys.executable}
try:
    import stable_envs as se
    from stable_envs.core.env import StableEnv

    out["registry"] = sorted(se.REGISTRY)
    out["n"] = len(se.REGISTRY)
    out["import_errors"] = se.envs.import_errors()
    out["file"] = se.__file__
    decls = {}
    for name, cls in se.REGISTRY.items():
        d = {k: _jsonable(getattr(cls, k, None)) for k in DECLARED}
        d["class"] = cls.__name__
        d["module"] = cls.__module__
        d["doc"] = (inspect.getdoc(cls) or "").strip()
        # ``StableEnv`` declares no ``step``: an env that has one is offering a batched GPU solver,
        # and ``step_reference_exact`` is where it says whether that solver is bit-exact against the
        # official one. The two are separate facts and the matrix keeps them in separate columns.
        d["has_step"] = hasattr(cls, "step")
        d["render_overridden"] = cls.render is not StableEnv.render
        d["backends"] = _backends(cls, StableEnv)
        decls[name] = d
    out["declarations"] = decls
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"
print("CENSUS " + json.dumps(out))
