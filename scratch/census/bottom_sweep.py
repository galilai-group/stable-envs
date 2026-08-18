import sys, time, numpy as np, torch, stable_envs as se
for name in ["DMCBallInCupCatch","DMCFingerSpin","MujocoPusher","CarRacing"]:
    if name not in se.REGISTRY: print(name,"not registered"); continue
    for size in (84, 256):
        try:
            p = se.make(name, batch=1, size=size, device="cuda")
        except Exception as e:
            print(f"{name} @{size}: {type(e).__name__}"); continue
        pool = p.sample_states(4096); hw = p.render(pool[:1]).shape[1:3]; del p
        torch.cuda.empty_cache()
        out=[]
        for B in (256, 1024, 4096):
            try:
                env = se.make(name, batch=B, size=size, device="cuda")
                st = torch.as_tensor(pool[:B], device="cuda")
                buf = torch.empty((B,*hw,3), dtype=torch.uint8, device="cuda")
                for _ in range(3): env.render(st, out=buf)
                torch.cuda.synchronize(); t0=time.perf_counter()
                for _ in range(5): env.render(st, out=buf)
                torch.cuda.synchronize()
                out.append(f"b{B}={B*5/(time.perf_counter()-t0):,.0f}")
                del env, st, buf; torch.cuda.empty_cache()
            except Exception as e:
                out.append(f"b{B}=ERR({type(e).__name__})"); torch.cuda.empty_cache()
        print(f"{name:22s} @{size:4d} {hw[0]}x{hw[1]}  " + "  ".join(out), flush=True)
