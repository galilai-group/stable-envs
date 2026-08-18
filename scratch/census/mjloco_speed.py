import time, numpy as np, torch, stable_envs as se
for name in ["MujocoPusher","MujocoHalfCheetah","MujocoReacher","MujocoWalker2d","MujocoInvertedPendulum"]:
    if name not in se.REGISTRY: continue
    for size in (240, 256):
        try:
            env = se.make(name, batch=256, size=size, device="cuda")
        except Exception as e:
            print(f"{name}@{size}: {type(e).__name__}"); continue
        st = env.sample_states(256, rng=np.random.default_rng(0))
        f = lambda: env.render(st)
        for _ in range(3): f()
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(5): f()
        torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/5
        on = env._mover_idx(env._scene(), env._fk(torch.as_tensor(st,device=env.device,dtype=torch.float32), env._scene())[0], size, env.ss) is not None
        print(f"{name:26s} @{size} b256: {256/dt:9,.0f} f/s   window={'FITS' if on else 'none'}", flush=True)
        del env; torch.cuda.empty_cache()
