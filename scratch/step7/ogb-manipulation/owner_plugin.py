"""A pytest plugin that answers "which test BUILT the renderer this test just closed?".

The resource guard reports the close, and by then the stack names the garbage collector -- every
remaining OGB failure has the identical `__del__`-inside-a-warp-allocation trace, which is the
collector waking up, not the owner. My standalone probe could not reproduce it (8 created, 0 died),
because the thing that matters is something the SUITE does and my imitation of it did not.

So: record the creation stack AND the test that was running at creation, then print both again at
close. Installed with `-p owner_plugin`; it changes nothing but the accounting.
"""
from __future__ import annotations

import traceback

_CURRENT = ["<session>"]
_BORN: dict[int, tuple[int, str, list[str]]] = {}
_SEQ = [0]


def pytest_configure(config):                                                # noqa: ARG001
    import mujoco

    real_init = mujoco.Renderer.__init__
    real_close = getattr(mujoco.Renderer, "close", None)
    real_del = getattr(mujoco.Renderer, "__del__", None)
    if getattr(mujoco.Renderer, "_owner_patched", False):
        return

    real_new = mujoco.Renderer.__new__

    def new(cls_, *a, **k):
        # __new__ AS WELL AS __init__: the last run reported four deaths whose birth this plugin
        # never saw ("born during <not recorded>"). Whatever builds them does not reach the __init__
        # this patch wraps, so the allocation itself is where the record has to be taken.
        obj = real_new(cls_) if real_new is object.__new__ else real_new(cls_, *a, **k)
        _SEQ[0] += 1
        where = [ln.strip().replace("\n", " | ") for ln in traceback.format_stack()[-10:-1]]
        _BORN[id(obj)] = (_SEQ[0], _CURRENT[0], where)
        print(f"\n[RENDERER NEW {_SEQ[0]}] during {_CURRENT[0]}\n    " + "\n    ".join(where[-5:]),
              flush=True)
        return obj

    def init(self, *a, **k):
        out = real_init(self, *a, **k)
        _SEQ[0] += 1
        where = [ln.strip().replace("\n", " | ") for ln in traceback.format_stack()[-10:-1]]
        _BORN.setdefault(id(self), (_SEQ[0], _CURRENT[0], where))
        print(f"\n[RENDERER BORN {_SEQ[0]}] during {_CURRENT[0]}\n    "
              + "\n    ".join(where[-4:]), flush=True)
        return out

    def report(self, how):
        n, born_in, where = _BORN.get(id(self), (-1, "<not recorded>", []))
        print(f"\n[RENDERER {how} {n}] born during {born_in}, {how} during {_CURRENT[0]}\n    "
              + "\n    ".join(where[-4:] or ["<unrecorded>"]), flush=True)

    def close(self, *a, **k):
        report(self, "CLOSED")
        return real_close(self, *a, **k) if real_close else None

    def dele(self):
        report(self, "FINALISED")
        if real_del is not None:
            try:
                real_del(self)
            except Exception:                                                # noqa: BLE001
                pass

    mujoco.Renderer.__new__ = new
    mujoco.Renderer.__init__ = init
    if real_close is not None:
        mujoco.Renderer.close = close
    mujoco.Renderer.__del__ = dele
    mujoco.Renderer._owner_patched = True


def pytest_runtest_setup(item):
    _CURRENT[0] = item.nodeid


def pytest_sessionfinish(session, exitstatus):                               # noqa: ARG001
    _CURRENT[0] = "<sessionfinish>"
    print(f"\n[RENDERER SUMMARY] {_SEQ[0]} created this session", flush=True)
    for rid, (n, born_in, _w) in _BORN.items():                              # noqa: B007
        print(f"    #{n} born during {born_in}", flush=True)
