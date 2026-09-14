"""Does CACHE PRESSURE still change the frame?  The scenario that made Step 4 observe 1.

``check_style_is_state`` renders a batch of sampled appearances twice -- once from the env under
test, once from a fresh one -- and asserts the two agree. In isolation it always agrees: the second
render finds the 64 plates the first one baked still in ``_SCENE_CACHE`` and never re-bakes. It
disagreed in Step 4 only for the two configurations whose 64 inserts happened to cross
``_SCENE_CACHE_MAX`` in a long session, because ``_scene`` did not evict an entry there, it did
``_SCENE_CACHE.clear()`` on the whole dict -- throwing away the keys the check was actively using.
A re-bake is not free: ``bake_probe.py`` measures the same key baking to two different plates.

So this reproduces the pressure instead of waiting for it: fill the cache to just under the cap with
keys nobody will ask for, then run the frozen check verbatim. Under the wholesale clear the check's
own inserts trip it mid-check and the first plates it baked are gone; under LRU the keys it just
inserted are the most recently used and the synthetic filler is what goes.

Run it against ``$SCR/se6/baseline`` and against the branch IN ONE JOB to get the before and after.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# $PYTHONPATH decides which stable_envs is measured; see replay_step4.py.
sys.path.append(str(ROOT))


def _root_tag(p: str) -> str:
    return re.sub(r"^/oscar/scratch/([^/]+)", r"$SCR", p)


class _Filler:
    """A cache value that is not ``None``, so ``_scene`` would treat it as a hit if anything asked."""


def probe(cfg, device, batch, headroom):
    import numpy as np
    import stable_envs as se
    import stable_envs.envs.ogbench as OGB

    env = se.make(cfg, batch=batch, size=None, device=device)
    st = env.sample_states(env.B)
    sty = env.sample_style(env.B)

    OGB._SCENE_CACHE.clear()
    fill = max(0, OGB._SCENE_CACHE_MAX - headroom)
    for i in range(fill):
        OGB._SCENE_CACHE[("__pressure__", cfg, i)] = _Filler()
    rec = {
        "config": cfg,
        "stable_envs_from": _root_tag(se.__file__),
        "cache_class": type(OGB._SCENE_CACHE).__name__,
        "cache_max": int(OGB._SCENE_CACHE_MAX),
        "filler_inserted": fill,
        "batch": int(env.B),
    }

    a = env.render(st, style=sty).cpu().numpy()
    rec["cache_len_after_first_render"] = len(OGB._SCENE_CACHE)
    rec["filler_survivors_after_first_render"] = sum(
        1 for k in OGB._SCENE_CACHE if isinstance(k, tuple) and k and k[0] == "__pressure__")

    fresh = type(env)(batch=env.B, size=env.size, supersample=env.ss, device=str(env.device))
    fresh.set_state(st, style=sty)
    b = fresh.render().cpu().numpy()

    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    m = d.max(-1)
    rec["check_style_is_state_under_pressure"] = {
        "max_abs_diff": int(d.max()),
        "frames_differing": int((m.reshape(len(a), -1).max(1) > 0).sum()),
        "px_differing": int((m > 0).sum()),
    }
    rec["PASS"] = rec["check_style_is_state_under_pressure"]["max_abs_diff"] == 0
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--headroom", type=int, default=8,
                    help="cache slots left free before the check runs; the check inserts `batch` "
                         "keys, so anything below `batch` guarantees the cap is crossed mid-check")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    rep = {"host": socket.gethostname(), "device": args.device, "batch": args.batch,
           "headroom": args.headroom, "slurm_job": os.environ.get("SLURM_JOB_ID"), "configs": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for cfg in args.configs:
        try:
            rep["configs"][cfg] = probe(cfg, args.device, args.batch, args.headroom)
        except Exception as exc:                                            # noqa: BLE001
            import traceback
            rep["configs"][cfg] = {"config": cfg, "error": repr(exc)[:300],
                                   "tb": traceback.format_exc()[-1200:]}
        r = rep["configs"][cfg]
        print(f"{cfg:30s} cache={r.get('cache_class')} "
              f"observed={((r.get('check_style_is_state_under_pressure') or {}).get('max_abs_diff'))} "
              f"PASS={r.get('PASS')} filler_left={r.get('filler_survivors_after_first_render')} "
              f"{r.get('error', '')}", flush=True)
        args.out.write_text(json.dumps(rep, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
