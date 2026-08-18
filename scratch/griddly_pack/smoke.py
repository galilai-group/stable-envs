"""Fast smoke over one env module: build, render, parity, and the conformance checks that bite first.

    python scratch/griddly_pack/smoke.py griddly
    python scratch/griddly_pack/smoke.py overcooked
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import torch                                                            # noqa: E402
import stable_envs as se                                                # noqa: E402
from stable_envs.testing import conformance as C                        # noqa: E402

mod = sys.argv[1] if len(sys.argv) > 1 else "griddly"
print("import_errors:", se.envs.import_errors(), flush=True)
names = [n for n, c in sorted(se.REGISTRY.items()) if c.__module__.endswith("." + mod)]
print(f"{len(names)} envs from {mod}: {names}", flush=True)
bad = 0
for name in names:
    t0 = time.time()
    try:
        env = se.make(name, batch=8, device="cuda")
        st = env.sample_states(16)
        fr = env.render(st).cpu().numpy()
        ref = type(env).reference_render(C.reference_env(type(env), env.size), st)
        agree = C.pixel_agreement(fr, ref)
        dims = C.check_state_dims_matter(env)
        var = C.check_variations_do_something(env)
        per = C.check_per_env_variation(env)
        col = C.check_colors_apply_exactly(env)
        sty = C.check_style_is_state(env)
        det = C.check_determinism(env)
        binv = C.check_batch_invariance(env)
        rt = C.check_state_roundtrip(env)
        ok = (agree["exact_frac"] == 1.0 and not dims["dead_dims"] and not var["inert_variations"]
              and not per.get("collapsed_variations") and not col["colors_not_applied"]
              and sty.get("max_abs_diff", 0) == 0 and det["identical"]
              and binv["max_abs_diff"] == 0 and rt["max_abs_err"] == 0)
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {name:26s} shape={list(fr.shape)} "
              f"state_dim={env.state_dim} exact={agree['exact_frac']:.6f} "
              f"max_abs={agree['max_abs']} interior={agree['interior_exact_frac']} "
              f"dead={dims['dead_dims'][:3]} inert={var['inert_variations']} "
              f"collapsed={per.get('collapsed_variations')} colors={col['colors_not_applied']} "
              f"style={sty.get('max_abs_diff')} det={det['identical']} "
              f"batch={binv['max_abs_diff']} rt={rt['max_abs_err']} {time.time()-t0:.1f}s",
              flush=True)
    except Exception as e:                                              # noqa: BLE001
        import traceback
        traceback.print_exc()
        bad += 1
        print(f"ERR {name}: {type(e).__name__}: {e}", flush=True)
print("FAILURES:", bad, flush=True)
sys.stdout.flush()
