"""S_cpu as a PAIRED measurement, because the certification node is a shared machine.

``examples/bench_step3.py`` times the official renderer once, after the whole batch sweep. On a quiet
node that is fine. node2338 carried a load average of 47 while the gfxdraw sweep ran, and the same
official Acrobot renderer measured 125 frames/s in one arm of the job and 55 in the other, minutes
apart -- a 2.3x swing in the DENOMINATOR, which is larger than several of the tier boundaries it is
used to decide.

So this alternates the two measurements instead of separating them: in one process, for one env, it
times a burst of official frames and then a burst of our own, round after round, and reports the
median of the per-ROUND ratio. Node drift that is slower than a round cancels; drift faster than a
round shows up as spread, which is reported rather than averaged away.

Same timing methodology as the frozen benchmark -- ``perf_counter``, one thread, the reference
reached through the env's own ``reference_render`` bridge, states from the frozen Step 4 corpus --
and no threshold is applied here.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="Acrobot,CartPole,MountainCar,MountainCarContinuous,Pendulum")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--ref-frames", type=int, default=24)
    ap.add_argument("--inner", type=int, default=3,
                    help="back-to-back renders inside a round; the frozen benchmark times a burst, "
                         "and a single cold-cache render after a reference burst is a different "
                         "measurement -- 1,121 f/s against 1,636 on the same laptop")
    ap.add_argument("--arm", required=True, help="a label recorded in the output, e.g. before/after")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_num_threads(1)
    import stable_envs as se
    from stable_envs.testing import conformance as C

    doc = {"arm": a.arm, "host": socket.gethostname(), "cpu": platform.processor() or platform.machine(),
           "threads": torch.get_num_threads(), "batch": a.batch, "rounds": a.rounds,
           "torch": torch.__version__, "tree": str(ROOT),
           # the tree actually imported, not the one the job script meant to import
           "stable_envs_file": se.__file__, "pythonpath": __import__("os").environ.get("PYTHONPATH"),
           "envs": {}}
    print(f"arm {a.arm}: stable_envs from {se.__file__}", flush=True)
    for name in a.envs.split(","):
        pool = np.load(ROOT / f"examples/results/step4/states/{name}.npz")["states"]
        env = se.make(name, batch=a.batch, device="cpu")
        cls = type(env)
        st = np.asarray(pool[np.arange(a.batch) % len(pool)], np.float32)
        ref_states = np.asarray(pool[np.arange(a.ref_frames) % len(pool)], np.float32)
        ref = C.reference_env(cls, env.size)
        out = torch.empty((a.batch, *env.render(st[:1]).shape[1:]), dtype=torch.uint8)

        cls.reference_render(ref, ref_states[:4])                       # warm surfaces
        env.render(st, out=out)
        ours, theirs, ratios = [], [], []
        for _ in range(a.rounds):
            t0 = time.perf_counter()
            cls.reference_render(ref, ref_states)
            r = a.ref_frames / (time.perf_counter() - t0)
            inner = []
            for _ in range(a.inner):
                t0 = time.perf_counter()
                env.render(st, out=out)
                inner.append(time.perf_counter() - t0)
            s = a.batch / statistics.median(inner)
            theirs.append(r)
            ours.append(s)
            ratios.append(s / r)
        rec = {"official_fps_median": statistics.median(theirs),
               "official_fps_min": min(theirs), "official_fps_max": max(theirs),
               "se_fps_median": statistics.median(ours),
               "se_fps_min": min(ours), "se_fps_max": max(ours),
               "S_cpu_paired_median": statistics.median(ratios),
               "S_cpu_paired_min": min(ratios), "S_cpu_paired_max": max(ratios),
               "loadavg": __import__("os").getloadavg()[0],
               "digest": C.frames_digest(env.render(st).numpy())}
        doc["envs"][name] = rec
        print(f"[{name:22s}] official {rec['official_fps_median']:8.1f} "
              f"({rec['official_fps_min']:.0f}-{rec['official_fps_max']:.0f})  "
              f"ours {rec['se_fps_median']:9.1f} ({rec['se_fps_min']:.0f}-{rec['se_fps_max']:.0f})  "
              f"S_cpu {rec['S_cpu_paired_median']:6.2f} "
              f"({rec['S_cpu_paired_min']:.2f}-{rec['S_cpu_paired_max']:.2f})  "
              f"load {rec['loadavg']:.0f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
