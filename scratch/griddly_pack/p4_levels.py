"""P4: two ways to put a chosen scene into griddly, priced against the Vulkan device budget.

p2 proved the decoder, and then the run died with ``vkGetDeviceQueue: Invalid device`` — because
``GymWrapper.load_state`` RETURNS A NEW WRAPPER, each of which owns a Vulkan device, and p2 kept every
one alive to avoid ``GymWrapper.__del__ -> game.release()``. So the question is which of these holds:

  A. wrappers may be DROPPED mid-run (release is only fatal at interpreter exit), or
  B. ``reset(level_string=...)`` mutates in place and can express the scene, in which case no wrapper
     is created at all.

B is the better answer if the level-string grammar can express a cell holding two objects, so the
grammar is probed rather than assumed.
"""
import copy
import gc
import os
import sys

import numpy as np
import griddly
from griddly import GymWrapper, gd


def frame(env):
    return np.array(env.game.observe(), copy=True).swapaxes(0, 2)


def codes_of(state):
    import collections
    m = collections.defaultdict(list)
    for o in state["Objects"]:
        if o["Name"].startswith("_"):
            continue
        m[tuple(int(v) for v in o["Location"])].append(o["Name"])
    return {k: tuple(sorted(v)) for k, v in m.items()}


env = GymWrapper(yaml_file="Single-Player/GVGAI/sokoban.yaml", level=0,
                 global_observer_type=gd.ObserverType.SPRITE_2D,
                 player_observer_type=gd.ObserverType.SPRITE_2D)
env.seed(0)
env.reset(level_id=0)
base = frame(env)
print("base shape", base.shape)

# ---------------------------------------------------------------- B. level strings
LEVELS = {
    "plain": "wwwww\nw.b.w\nw.A.w\nw.h.w\nwwwww",
    "spaced": "w  w  w  w  w\nw  .  b  .  w\nw  .  A  .  w\nw  .  h  .  w\nw  w  w  w  w",
    "slash": "w  w  w  w  w\nw  .  b  .  w\nw  .  A/h  .  w\nw  .  h  .  w\nw  w  w  w  w",
    "slash_tight": "wwwww\nw.b.w\nw.A/h.w\nw.h.w\nwwwww",
    "player_id": "w  w  w  w  w\nw  .  b  .  w\nw  .  A1  .  w\nw  .  h  .  w\nw  w  w  w  w",
}
for tag, lvl in LEVELS.items():
    try:
        env.reset(level_string=lvl)
        fr = frame(env)
        print(f"LEVEL[{tag}] ok shape={fr.shape} cells={codes_of(env.get_state())}")
    except Exception as e:                                          # noqa: BLE001
        print(f"LEVEL[{tag}] FAILED: {type(e).__name__}: {e}")

# back to the shipped level; is the frame the SAME as before the excursion?
env.reset(level_id=0)
print("reset(level_id=0) after level_string, max_abs vs base:",
      int(np.abs(frame(env).astype(int) - base.astype(int)).max()))

# ---------------------------------------------------------------- A. dropping wrappers
env.seed(0)
env.reset(level_id=0)
st = copy.deepcopy(env.get_state())
ok, first, bad = 0, None, None
for i in range(600):
    try:
        e = env.load_state(copy.deepcopy(st))
        f = frame(e)
        if first is None:
            first = f
        elif not np.array_equal(f, first):
            bad = i
            break
        del e
        if i % 50 == 0:
            gc.collect()
        ok += 1
    except Exception as ex:                                         # noqa: BLE001
        print(f"DROP-CYCLE failed at {i}: {type(ex).__name__}: {ex}")
        break
print(f"DROP-CYCLE: {ok} load_state wrappers created and dropped, first divergence at {bad}")

# how many can be held at once?
KEEP = []
held = 0
try:
    for i in range(400):
        e = env.load_state(copy.deepcopy(st))
        KEEP.append(e)
        f = frame(e)
        if not np.array_equal(f, first):
            print("HELD divergence at", i)
            break
        held = i + 1
except Exception as ex:                                             # noqa: BLE001
    print(f"HOLD failed at {held}: {type(ex).__name__}: {ex}")
print("HELD wrappers:", held)

sys.stdout.flush()
os._exit(0)
