"""WHO creates the `mujoco.Renderer` objects that are later finalised? Recorded, not inferred.

After the `_PlateRig` fix the gate run still reports `renderers 6 created / 4 closed` -- but with
`0 EGL contexts released`, and every remaining stack is identical: `__del__` firing from inside a
warp allocation, i.e. deferred garbage collection of an object that became garbage in some EARLIER
test. The close stack therefore names the collector, never the owner, and the four failing rows are
the tests unlucky enough to be running when the collector woke up.

So this records the CREATION stack of every Renderer at `__init__`, keeps it on the instance, and
prints it again when that instance is closed or finalised. The pairing is what identifies the owner:
whoever built the four that die is the code that has to hold them.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import socket
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", default=["OGBCube", "OGBScene"])
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    import mujoco
    import numpy as np

    born: dict[int, dict] = {}
    died: list[dict] = []
    seq = [0]

    real_init = mujoco.Renderer.__init__
    real_close = getattr(mujoco.Renderer, "close", None)
    real_del = getattr(mujoco.Renderer, "__del__", None)

    def init(self, *args, **kw):
        out = real_init(self, *args, **kw)
        seq[0] += 1
        rec = {"n": seq[0], "id": id(self),
               "where": [l.strip() for l in traceback.format_stack()[-12:-1]]}
        born[id(self)] = rec
        print(f"[BORN {rec['n']}] id={rec['id']}\n    " + "\n    ".join(rec["where"][-4:]), flush=True)
        return out

    def note_death(self, how):
        rec = born.get(id(self))
        died.append({"how": how, "born": rec,
                     "collected_at": [l.strip() for l in traceback.format_stack()[-8:-1]]})
        print(f"[{how.upper()} ] id={id(self)} born at:\n    "
              + "\n    ".join((rec or {}).get("where", ["<not recorded>"])[-5:]), flush=True)

    def close(self, *args, **kw):
        note_death(self, "closed")
        return real_close(self, *args, **kw) if real_close else None

    def dele(self):
        note_death(self, "finalised")
        if real_del is not None:
            try:
                real_del(self)
            except Exception:                                                # noqa: BLE001
                pass

    mujoco.Renderer.__init__ = init
    if real_close is not None:
        mujoco.Renderer.close = close
    mujoco.Renderer.__del__ = dele

    import stable_envs as se
    import stable_envs.testing.conformance as C

    for cname in a.cells:
        print(f"\n===== {cname} =====", flush=True)
        cls = se.REGISTRY[cname]
        env = se.make(cname, batch=a.n, device="cuda")
        st = env.sample_states(a.n, np.random.default_rng(0))

        print("--- render() ---", flush=True)
        env.render(st)
        print("--- reference_render() ---", flush=True)
        cls.reference_render(cls.reference_env(env.size), st[:2])
        print("--- a shared style, broadcast ---", flush=True)
        try:
            C.check_style_broadcast(env, st)
        except AttributeError:
            print("  (no check_style_broadcast in this build)", flush=True)
        except Exception as e:                                               # noqa: BLE001
            print(f"  raised {type(e).__name__}: {e}", flush=True)
        print("--- a second env at a second size (what a dataset path does) ---", flush=True)
        env2 = se.make(cname, batch=a.n, device="cuda", size=112)
        env2.render(env2.sample_states(a.n, np.random.default_rng(1)))
        del env2
        gc.collect()
        print("--- after gc.collect() ---", flush=True)

    gc.collect()
    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "renderer_owners.json").write_text(json.dumps(
        {"job": f"job {os.environ.get('SLURM_JOB_ID')} on {socket.gethostname()}",
         "created": list(born.values()), "died": died}, indent=1))
    print(f"\nCREATED {len(born)}  DIED {len(died)}", flush=True)
    for d in died:
        w = (d.get("born") or {}).get("where", ["<unrecorded>"])
        print(f"  {d['how']}: born at {w[-1] if w else '?'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
