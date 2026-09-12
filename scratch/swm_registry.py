"""Enumerate stable_worldmodel's registered gymnasium ids LIVE, and probe each for constructibility.

The registry is built dynamically at import time (stable_worldmodel/envs/__init__.py calls
gymnasium.envs.registration.register), and an optional-dependency block can silently skip a whole
family (the ale import is inside a try/except ImportError), so the only trustworthy list is the one
this process observes.
"""
import datetime
import json
import os

# Agg BEFORE stable_worldmodel imports matplotlib. SimplePointMaze draws through a matplotlib canvas
# and then reshapes the buffer to (H, W, 4); on a HiDPI display the default macOS backend hands back a
# 2x buffer and the reshape raises, which reads as "this env does not render" when it renders fine.
# A headless probe wants the headless backend anyway.
os.environ.setdefault("MPLBACKEND", "Agg")
import platform
import sys
import traceback

import gymnasium
import stable_worldmodel as swm
from gymnasium.envs import registration

out = {}
out["probe_env"] = {
    "python": sys.executable,
    "platform": platform.platform(),
    "stable_worldmodel": getattr(swm, "__version__", "unknown"),
    "gymnasium": gymnasium.__version__,
    "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
    "probed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}
ids = sorted(i for i in registration.registry if i.startswith("swm/"))
out["ids"] = ids
out["WORLDS"] = sorted(swm.envs.WORLDS)
out["DISCRETE_WORLDS"] = sorted(swm.envs.DISCRETE_WORLDS)
out["entry_points"] = {i: str(registration.registry[i].entry_point) for i in ids}
out["kwargs"] = {i: {k: str(v) for k, v in (registration.registry[i].kwargs or {}).items()} for i in ids}


def _render_model_signature(env):
    """A constructed env's RENDER MODEL as a comparable digest, taken after ``reset()``.

    Same technique the matrix uses to decide whether two dm_control tasks draw the same picture,
    pointed at the stable_worldmodel wrappers instead. Taken after the reset because that is a task's
    last chance to change the picture: ``reacher`` and ``finger`` size their target in
    ``initialize_episode`` and nowhere else, and a digest of the freshly compiled model would call
    ``easy`` and ``hard`` identical.
    """
    import hashlib

    import numpy as _np
    u = getattr(env, "unwrapped", env)
    d = getattr(u, "dmc_env", None)
    if d is None:
        return None
    m = getattr(d, "physics", None)
    m = getattr(m, "model", None) if m is not None else None
    if m is None:
        inner = getattr(d, "_env", d)
        m = getattr(getattr(inner, "physics", None), "model", None)
    if m is None:
        return None
    parts = []
    # SITES ARE INCLUDED DELIBERATELY. `reacher` draws its target as a geom and `finger` draws its
    # as a site, so a geom-only digest calls finger's turn_easy and turn_hard identical -- a false
    # "one render config" on exactly the axis this probe exists to measure.
    for f in ("geom_type", "geom_size", "geom_rgba", "geom_matid", "mat_rgba", "body_pos",
              "geom_pos", "geom_quat", "cam_pos", "cam_quat", "cam_fovy", "light_pos",
              "site_type", "site_size", "site_rgba", "site_pos", "site_quat", "site_matid"):
        a = getattr(m, f, None)
        if a is not None:
            parts.append(f.encode() + _np.ascontiguousarray(a).tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()[:12]


def _swm_task_space(env_id, u):
    """The tasks THIS swm wrapper accepts, and which of them draw the same picture.

    Two things the upstream dm_control suite cannot answer and that were wrong when taken from it:

    * **which tasks exist.** ``AcrobotDMControlWrapper`` takes no ``task`` argument at all -- it is
      one fixed sparse task -- while the suite lists two acrobot tasks, and ``WalkerDMControlWrapper``
      accepts eight where the suite lists three. Reading the suite over-reports one and under-reports
      the other. The accepted values are read off the wrapper: a ``task`` parameter in ``__init__``,
      and ``_TASKS`` in the module that defines the class.
    * **which of them are RENDERER configurations.** A task that only changes a reward, a desired
      speed or a success condition needs no second renderer; one that changes a drawn target does.
      Each accepted task is constructed and its render model digested, so the equivalence classes are
      measured rather than argued -- and the method demonstrably discriminates, because it splits
      reacher's ``easy`` from ``hard`` on target size while collapsing walker's eight speeds to one.
    """
    import inspect

    import gymnasium as _gym
    cls = type(u)
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return None
    if "task" not in params:
        return {"accepts_task": False,
                "why": f"{cls.__name__}.__init__ takes no `task` argument; it is one fixed task"}
    default = params["task"].default
    mod = inspect.getmodule(cls)
    tasks = sorted(getattr(mod, "_TASKS", ()) or ())
    out = {"accepts_task": True, "default_task": default, "tasks": tasks,
           "source": f"{getattr(mod, '__name__', '?')}._TASKS and {cls.__name__}.__init__ default"}
    groups = {}
    for t in tasks:
        try:
            e = _gym.make(env_id, task=t, render_mode="rgb_array")
            e.reset(seed=0)
            groups[t] = _render_model_signature(e)
            e.close()
        except Exception as ex:
            groups[t] = f"ERR {type(ex).__name__}: {ex}"[:120]
    out["render_signature_by_task"] = groups
    classes = {}
    for t, g in sorted(groups.items()):
        classes.setdefault(str(g), []).append(t)
    out["render_equivalence_classes"] = [sorted(v) for _, v in sorted(classes.items())]
    out["distinct_render_configs"] = len(classes)
    return out


def _dmc_signature(task, drop=()):
    """A dm_control task's identity as a comparable value: class name plus its scalar attributes.

    ``_visualize_reward`` and ``_random`` are bookkeeping that the constructor sets from arguments
    unrelated to which task this is, so they are excluded; everything else a task stores about itself
    (``_move_speed``, ``_sparse``, ``_target_size``, ...) is exactly what distinguishes
    ``reacher-easy`` from ``reacher-hard``.
    """
    skip = {"_visualize_reward", "_random"} | set(drop)
    return (type(task).__name__.removeprefix("Custom"),
            tuple(sorted((k, v) for k, v in vars(task).items()
                         if k not in skip and isinstance(v, (int, float, bool, str)))))


def _dmc_reference_index():
    """``signature -> ["domain-task", ...]`` over every task dm_control's own suite can build.

    Built by CONSTRUCTING each task rather than by reading stable_worldmodel's source, because the
    id says nothing about the task: ``swm/ReacherDMControl-v0`` is ``reacher-hard`` and
    ``swm/CartpoleDMControl-v0`` is ``cartpole-swingup_sparse``, and neither is the name's obvious
    reading. Returns ``None`` if dm_control is not importable, and the caller records that.
    """
    try:
        from dm_control import suite
    except Exception:
        return None
    index = {}
    for dom, tsk in suite.ALL_TASKS:
        try:
            index.setdefault(_dmc_signature(suite.load(dom, tsk).task), []).append(f"{dom}-{tsk}")
        except Exception:
            continue
    return index


def _resolve_dmc(u, index):
    """Which ``dm_control/<domain>-<task>`` a constructed swm DM Control wrapper actually is.

    Two stages, and the stage is reported so a consumer can tell a match from a deduction:

    ``exact``    the wrapper's task object has the same class and the same attributes as one the
                 suite builds. Nine of the twelve ids resolve here.
    ``by_class`` stable_worldmodel subclasses the suite's task (``CustomPlanarWalker``) and adds its
                 own ``_goal`` string, so the signatures cannot be equal. Compared again against only
                 the attributes the REFERENCE task carries, and accepted only if exactly one suite
                 task survives -- an ambiguous result is reported as ambiguous, not guessed.
    """
    d = getattr(u, "dmc_env", None)
    if d is None or index is None:
        return None
    inner = getattr(d, "_env", d)
    task = getattr(inner, "_task", getattr(inner, "task", None))
    if task is None:
        return None
    sig = _dmc_signature(task)
    if sig in index and len(index[sig]) == 1:
        return {"reference_id": "dm_control/" + index[sig][0], "how": "exact", "signature": str(sig)}
    cls, attrs = sig
    ours = dict(attrs)
    hits = sorted({name for (rcls, rattrs), names in index.items() if rcls == cls
                   for name in names if all(ours.get(k) == v for k, v in rattrs)})
    if len(hits) == 1:
        return {"reference_id": "dm_control/" + hits[0], "how": "by_class", "signature": str(sig)}
    return {"reference_id": None, "how": "ambiguous" if hits else "no_match",
            "candidates": hits, "signature": str(sig)}


_DMC_INDEX = _dmc_reference_index()
out["dmc_index_available"] = _DMC_INDEX is not None
probe = {}
for i in ids:
    # Pessimistic defaults. Every probed property starts as "we did not find out"; the probe
    # OVERWRITES a default only when it actually observed something. An id that raises on
    # gymnasium.make therefore keeps unknown_probe_failed everywhere instead of inheriting a
    # confident-looking None from a key that was never reached.
    rec = {"variation_space_status": "unknown_probe_failed",
           "variation_names": None,
           "frame_status_probe": "unknown_probe_failed"}
    try:
        e = gymnasium.make(i, render_mode="rgb_array")
        rec["make"] = "ok"
        u = e.unwrapped
        rec["class"] = type(u).__name__
        rec["obs_space"] = str(e.observation_space)[:300]
        rec["act_space"] = str(e.action_space)[:300]
        # TRI-STATE, not a bare None. "this env has no variation_space" and "we could not find out"
        # are different facts and only one of them is a gap; collapsing them is how a coverage table
        # ends up asserting that Fetch exposes no variation space when the truth is that Fetch did not
        # construct. The status is written for EVERY id, including the ones that failed to construct,
        # which is why the default is set before the try.
        # LOOK ALONG THE WHOLE WRAPPER CHAIN, not at `unwrapped`. stable_worldmodel puts the
        # variation space on ITS OWN wrapper, and for the classic-control ids that wrapper sits above
        # a stock gymnasium env -- so `unwrapped` is `CartPoleEnv`, which has no variation space, and
        # reading it there reports "absent" for four ids that in fact declare 5-11 leaves apiece.
        # Measured: CartPole 10, Acrobot 11, Pendulum 9, MountainCar 5.
        #
        # `absent` is still a real answer and is kept distinct from a failure: the Craftax wrappers
        # DEFINE `variation_space` and set it to None, which is the env saying it has none.
        vs, holder = None, None
        try:
            cur = e
            while cur is not None and vs is None:
                cand = getattr(cur, "variation_space", None)
                if cand is not None:
                    vs, holder = cand, type(cur).__name__
                cur = getattr(cur, "env", None)
            if vs is None:
                rec["variation_space_status"] = "absent"
                rec["variation_names"] = None
                rec["variation_space_holder"] = None
            else:
                rec["variation_names"] = vs.names()
                rec["variation_space_status"] = "present"
                rec["variation_space_holder"] = holder
        except Exception as ex:
            rec["variation_space_status"] = "unknown_probe_failed"
            rec["variation_space_error"] = f"{type(ex).__name__}: {ex}"[:300]
            rec["variation_names"] = None
        # WHICH TASK/DOMAIN THE ID ACTUALLY MEANS. The DM Control ids carry no task in their
        # registration kwargs -- `swm/ReacherDMControl-v0` is `reacher`/`hard`, not `reacher`/`easy`
        # -- so the only place the join to a stable-envs env can be read off is the constructed
        # object. Recorded for every id, not just the DMC ones, because the same is true of any
        # wrapper that picks a variant in its own `__init__` default.
        # The MRO is what says which ROBOT an OGBench maze id is: `swm/OGBMaze-v0` registers
        # `MazeEnv` with no kwargs, and only `PointEnv` in its bases distinguishes the point maze
        # from the ant and humanoid ones the same class can build.
        rec["mro"] = [c.__name__ for c in type(u).__mro__]
        for attr, key in (("env_name", "env_name"), ("_task_name", "task_name"),
                          ("camera_id", "camera_id"), ("action_repeat", "action_repeat"),
                          ("_maze_type", "maze_type"), ("_env_type", "env_type"),
                          ("_ob_type", "ob_type")):
            if hasattr(u, attr):
                try:
                    rec[key] = str(getattr(u, attr))
                except Exception as ex:
                    rec[key] = f"ERR {ex}"
        # THE ENV UNDERNEATH THE WRAPPER, and what it calls itself. stable_worldmodel wraps a
        # third-party env and the wrapper's own name says nothing about which one; the inner object's
        # class and its declared `name` do. This is what turns the Craftax join from a spelling
        # coincidence into an identity: the inner env reports
        # `Craftax-Classic-Pixels-NoAutoReset-v1` against our declared `Craftax-Classic-Pixels-v1`.
        for attr in ("craftax_env", "dmc_env", "_env", "env"):
            inner = getattr(u, attr, None)
            if inner is None or hasattr(inner, "step_wait"):
                continue
            mod = type(inner).__module__ or ""
            if mod.startswith("stable_worldmodel") or mod.startswith("gymnasium"):
                continue
            rec["inner_env"] = {"attr": attr, "class": type(inner).__name__, "module": mod}
            for k in ("name", "env_name"):
                v = getattr(inner, k, None)
                if isinstance(v, str):
                    rec["inner_env"][k] = v
            break
        dmc = _resolve_dmc(u, _DMC_INDEX)
        if dmc is not None:
            rec["dmc_reference"] = dmc
        ts = _swm_task_space(i, u)
        if ts is not None:
            rec["swm_task_space"] = ts
        try:
            o, info = e.reset(seed=0)
            rec["reset"] = "ok"
            rec["info_keys"] = sorted(info.keys()) if isinstance(info, dict) else str(type(info))
            f = e.render()
            rec["frame_shape"] = None if f is None else list(getattr(f, "shape", []))
            rec["frame_status_probe"] = "pixels" if rec["frame_shape"] else "no_frame_returned"
        except Exception as ex:
            rec["reset"] = f"ERR {type(ex).__name__}: {ex}"[:400]
            # an env that says IT HAS NO FRAME has answered the question; anything else has not
            rec["frame_status_probe"] = ("symbolic_no_frame"
                                         if "do not produce a frame" in str(ex)
                                         else "unknown_probe_failed")
        e.close()
    except Exception as ex:
        rec["make"] = f"ERR {type(ex).__name__}: {ex}"[:400]
    probe[i] = rec
    print(i, rec.get("make"), rec.get("reset", ""), flush=True)

out["probe"] = probe
with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=1)
print("WROTE", sys.argv[1])
