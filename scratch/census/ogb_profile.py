"""Where does an OGBench frame go? It already tiles the primary cast, so the cost is elsewhere."""
import cProfile, io, pstats, time
import numpy as np, torch, stable_envs as se

for name in ["PointmazeMediumNavigate", "AntmazeMediumNavigate", "HumanoidmazeMediumNavigate",
             "PointMazeMedium"]:
    if name not in se.REGISTRY:
        print(f"{name}: not registered"); continue
    print(f"[{name}] constructing...", flush=True)
    try:
        env = se.make(name, batch=256, size=256, device="cuda")
    except Exception as e:
        print(f"[{name}] make failed: {type(e).__name__}: {e}", flush=True); continue
    print(f"[{name}] built, sampling...", flush=True)
    st = env.sample_states(256, rng=np.random.default_rng(0))
    f = lambda: env.render(st)
    for _ in range(3): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5): f()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 5
    print(f"\n{'='*76}\n{name}  256px b256: {dt*1e3:.1f} ms = {256/dt:,.0f} f/s  "
          f"TILE={getattr(env,'TILE',None)}  prims={len(env._scene().prims) if hasattr(env,'_scene') else '?'}")
    pr = cProfile.Profile(); pr.enable()
    for _ in range(3): f()
    torch.cuda.synchronize(); pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(10)
    for line in s.getvalue().splitlines():
        t = line.strip()
        if t and not t.startswith(("Ordered by","ncalls","Random listing")) and "function calls" not in t:
            print("   " + line.rstrip()[:140])
    del env; torch.cuda.empty_cache()
