"""Does world 0's frame depend on what the OTHER worlds in the same launch contain?

``H1Slide`` is the one shipped cell where ``check_batch_size_invariance`` is non-zero (2/255) even
with the world count pinned, so the batch size cannot be what changes -- the only thing left that
differs between "state 682 alone" and "state 682 inside a 1024-state corpus" is the CONTENTS of the
other 63 worlds. ``mujoco_warp.refit_bvh`` refits one acceleration structure for the whole launch, so
a scene with coincident surfaces (slide's ramp lies on the floor) can have a depth tie broken by the
traversal order, and the traversal order is a function of every world.

Decisive form: hold world 0's ``qpos`` fixed and vary only worlds 1..63.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

LOGS = "/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench"


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "slide"
    robot = sys.argv[2] if len(sys.argv) > 2 else "h1"
    import stable_envs as se
    from stable_envs.envs import humanoidbench as HB

    name = (f"HumanoidBench{'H1Hand' if robot == 'h1hand' else 'H1'}"
            f"{''.join(p.capitalize() for p in task.split('_'))}")
    env = se.make(name, batch=1, device="cuda")
    pool = env.sample_states(64)
    c = HB._context(robot, task, env.size, HB.MJW_CHUNK)
    sty = env.resolve_style(None, None, batch=1)

    frames = []
    for tag, filler in (("copies of row 0", np.tile(pool[0], (HB.MJW_CHUNK, 1))),
                        ("63 other states", pool[:HB.MJW_CHUNK].copy()),
                        ("zeros", np.zeros((HB.MJW_CHUNK, env.state_dim), np.float32))):
        q = np.asarray(filler, np.float32).copy()
        q[0] = pool[0]
        import torch
        qt = c.wp.to_torch(c.wd.qpos)
        qt[:] = torch.as_tensor(q, device=qt.device, dtype=qt.dtype)
        env._upload_appearance = None                       # unused; appearance is the default
        env._upload(c, torch.as_tensor(q, device=env.device, dtype=torch.float32), sty_pad(sty, q.shape[0]), 0, q.shape[0])
        qt[:] = torch.as_tensor(q, device=qt.device, dtype=qt.dtype)
        torch.cuda.synchronize()
        c.wp.capture_launch(c.graph)
        c.wp.synchronize()
        frames.append((tag, c.bgra[0][..., [2, 1, 0]].contiguous().cpu().numpy()))

    base = frames[0][1]
    for tag, f in frames:
        d = np.abs(f.astype(np.int16) - base.astype(np.int16))
        print(f"world 0 with {tag:20s}: max {int(d.max())}/255 over {int((d.max(-1) > 0).sum())} px")


def sty_pad(sty, n):
    """The default style broadcast to ``n`` rows; the probe varies geometry, not appearance."""
    return sty.for_batch(n, owner="probe12")


if __name__ == "__main__":
    main()
