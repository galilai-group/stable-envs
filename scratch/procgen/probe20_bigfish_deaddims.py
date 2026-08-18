import numpy as np, torch, json
import stable_envs as se
from stable_envs.testing import conformance as C
from stable_envs.envs.procgen import BIGFISH_SLOTS
for B in (8, 32):
    e = se.make("ProcgenBigfish", batch=B, device="cuda")
    r = C.check_state_dims_matter(e)
    dead = r["dead_dims"]
    slots = sorted({int(d.split("_")[0][4:]) for d in dead if d.startswith("fish")})
    print(f"B={B}: {len(dead)} dead dims; dead fish slots {slots}")
    pool = e.sample_states(512)
    cnt = pool[:, 6+6::7][:, :BIGFISH_SLOTS].sum(1)
    print(f"   pool fish counts: min {cnt.min():.0f} max {cnt.max():.0f} mean {cnt.mean():.1f} "
          f"p90 {np.percentile(cnt,90):.0f}")
    print("   non-fish dead:", [d for d in dead if not d.startswith("fish")])
