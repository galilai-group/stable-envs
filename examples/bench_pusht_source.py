"""Paired PushT render timing: ``env.render(state)`` against the pinned stable-worldmodel source.

Both sides render the same saved states on one CPU thread. The source side sets each state with
``_set_state`` and calls ``render()`` on one source env. ``first`` is the first ``env.render`` of
each state batch on a fresh env (it builds one pymunk scene per lane); ``warm`` repeats them.

    PYTHONPATH=/path/to/stable-worldmodel python examples/bench_pusht_source.py --batches 1,64,1024
"""

from __future__ import annotations

import argparse
import json
import platform
import time


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batches", default="1,16,64,256,1024")
    p.add_argument("--sizes", default="224,96")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--source-frames", type=int, default=512)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    import numpy as np
    import torch
    from stable_worldmodel.envs.pusht.env import PushT as SourcePushT

    import stable_envs as snv

    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    rows = []
    for size in map(int, args.sizes.split(",")):
        source = SourcePushT(resolution=size)
        source.reset(seed=0)
        for batch in map(int, args.batches.split(",")):
            env = snv.PushT(num_envs=batch, size=size)
            saved = []
            for k in range(args.episodes):
                env.reset(seed=k)
                for _ in range(3 if k else 0):
                    env.step(torch.as_tensor(rng.uniform(-1, 1, (batch, 2)), dtype=torch.float32), frames=None)
                saved.append(env.get_state())

            fresh = snv.PushT(num_envs=batch, size=size)
            t = time.perf_counter()
            for state in saved:
                fresh.render(state)
            first = (time.perf_counter() - t) / (args.episodes * batch)
            fresh.close()

            t = time.perf_counter()
            for _ in range(args.repeats):
                for state in saved:
                    env.render(state)
            warm = (time.perf_counter() - t) / (args.repeats * args.episodes * batch)

            poses = []
            for state in saved:
                agent = state["current", "body", "agent", "position"].numpy()
                block = state["current", "body", "block", "position"].numpy()
                angle = state["current", "body", "block", "angle"].numpy()
                poses += [np.array([*agent[i], *block[i], angle[i], 0.0, 0.0]) for i in range(batch)]
            poses = poses[: args.source_frames]
            t = time.perf_counter()
            for pose in poses:
                source._set_state(pose)
                source.render()
            reference = (time.perf_counter() - t) / len(poses)
            env.close()

            row = dict(size=size, batch=batch, first_ms=first * 1e3, warm_ms=warm * 1e3, source_ms=reference * 1e3)
            rows.append(row)
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        source.close()
    if args.output:
        meta = dict(host=platform.node(), processor=platform.processor(), python=platform.python_version(), threads=1)
        with open(args.output, "w") as f:
            json.dump(dict(meta=meta, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
