"""Reproduce, and localise, the EGL failure that stops DM Control parity from being asserted at all.

`test_render_parity_with_official_renderer` cannot run for the 16 DM Control envs: the reference env's
EGL teardown raises `OpenGL.raw.EGL._errors.EGLError` and the process does not survive it. The repo's
rule is that an EGL context is created and never released in a process, so *something* is releasing one
-- and `stable_envs.testing.resource_guard` cannot see it, because it patches `mujoco.egl.GLContext`
and dm_control does not use that class. dm_control ships its OWN EGL context
(`dm_control._render.pyopengl.egl_renderer.EGLContext`) on the SAME EGLDisplay, plus two `atexit`
handlers that call `eglTerminate` on it.

So this instruments both class families and walks the parity flow one step at a time, printing where
each context is created and freed, and printing a marker at the very end so an exit-time failure can be
told apart from a mid-run one.

    python scratch/audit/dmc_egl_repro.py --envs DMCBallInCupCatch,DMCCheetahRun
"""
from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np

import stable_envs as se
from stable_envs.testing import conformance as C
from stable_envs.testing import resource_guard as RG

EVENTS: list[str] = []


def patch_dm_control() -> str:
    """Count dm_control's own EGL context, which resource_guard does not patch."""
    try:
        from dm_control._render.pyopengl import egl_renderer
    except Exception as e:                                               # noqa: BLE001
        return f"dm_control EGL not importable: {type(e).__name__}: {e}"
    cls = egl_renderer.EGLContext
    for meth, tag in (("_platform_init", "dmc_ctx_created"), ("_platform_free", "dmc_ctx_freed")):
        fn = getattr(cls, meth)
        if getattr(fn, "_se_probe", False):
            continue

        def wrapper(self, *a, __fn=fn, __tag=tag, **k):
            where = "".join(traceback.format_stack(limit=8)[:-1]).strip().splitlines()[-4:]
            EVENTS.append(f"{__tag}\n      " + "\n      ".join(w.strip() for w in where))
            print(f"  [{__tag}]", flush=True)
            return __fn(self, *a, **k)

        wrapper._se_probe = True                                         # type: ignore[attr-defined]
        setattr(cls, meth, wrapper)
    return f"patched {cls.__module__}.{cls.__name__}"


def one(name: str, states_n: int = 8) -> None:
    cls = se.REGISTRY[name]
    print(f"\n=== {name} ({cls.__mro__[1].__name__}, tier={cls.fidelity_tier})", flush=True)
    env = se.make(name, batch=16, device="cuda")
    print(f"  our env built; guard {RG.totals()}", flush=True)
    ref = C.reference_env(cls, env.size)
    print(f"  reference env built; guard {RG.totals()}", flush=True)
    states = env.sample_states(states_n)
    print(f"  sampled {states_n} states; guard {RG.totals()}", flush=True)
    official = cls.reference_render(ref, states)
    print(f"  reference rendered {official.shape}; guard {RG.totals()}", flush=True)
    ours = env.render(states).cpu().numpy()
    r = C.pixel_agreement(ours, official)
    print(f"  parity: interior {r['interior_exact_frac']} all {r['exact_frac']:.5f} "
          f"mean {r['mean_abs']:.3f} max {r['max_abs']}", flush=True)
    # the second half of the flow: test_state_is_sufficient drives the SAME env twice
    fa = cls.reference_render(ref, states)
    fb = cls.reference_render(ref, states[::-1])[::-1]
    print(f"  state-sufficiency max {int(np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max())}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="DMCBallInCupCatch,DMCCartpoleBalance,DMCCheetahRun")
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    print("resource_guard:", RG.install(), flush=True)
    print("dm_control:", patch_dm_control(), flush=True)
    for _ in range(a.repeat):
        for n in a.envs.split(","):
            if n not in se.REGISTRY:
                print(f"{n}: not registered", flush=True)
                continue
            try:
                one(n)
            except Exception:                                            # noqa: BLE001
                traceback.print_exc()
    print("\nGUARD TOTALS:", RG.totals(), flush=True)
    print("EVENTS:", flush=True)
    for e in EVENTS:
        print("  " + e, flush=True)
    print("\n== REACHED END OF MAIN ==", flush=True)
    sys.stdout.flush()
