"""Reproduce ``check_batch_size_invariance``'s 2/255 on H1Slide and localise it.

Three renders of the SAME state, differing only in what else was in the launch:
  A  alone            (chunk of 1, world 0)
  B  in its own chunk (rows 640..703 of the corpus, world 42)
  C  the full 1024-state corpus, same chunk boundaries as B
If A == B == C the earlier reading was a stale-context artefact; if A != B the difference is the
world index or the chunk contents, both of which probe12/probe13 already measured at 0.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

import stable_envs as se

task = sys.argv[1] if len(sys.argv) > 1 else "slide"
robot = sys.argv[2] if len(sys.argv) > 2 else "h1"
name = (f"HumanoidBench{'H1Hand' if robot == 'h1hand' else 'H1'}"
        f"{''.join(p.capitalize() for p in task.split('_'))}")
env = se.make(name, batch=8, device="cuda")
states = env.sample_states(1024, rng=np.random.default_rng(11))
i = 682
lo = (i // 64) * 64

if len(sys.argv) > 3 and sys.argv[3] == "after512":
    # ``check_resolution_consistency`` renders at 2S first, which builds a SECOND MJWarp render
    # context at 512 px in the same process. The 2/255 only ever appeared in a script that did this
    # first, so it is the variable under test rather than an incidental warm-up.
    env.render(states[:8], size=2 * env.size)

A = env.render(states[i:i + 1]).cpu().numpy()[0]
B = env.render(states[lo:lo + 64]).cpu().numpy()[i - lo]
C = env.render(states).cpu().numpy()[i]
D = env.render(states[i:i + 1]).cpu().numpy()[0]

for tag, x, y in (("alone vs its 64-chunk", A, B), ("alone vs full 1024", A, C),
                  ("64-chunk vs full 1024", B, C), ("alone vs alone again", A, D)):
    d = np.abs(x.astype(np.int16) - y.astype(np.int16))
    print(f"{name} {tag:24s}: max {int(d.max())}/255 over {int((d.max(-1) > 0).sum())} px")
