import os,sys,json,time
import numpy as np, torch
sys.path.insert(0,"/mnt/vast/home/randall.balestriero/code/stable-envs")
from stable_envs.render.tilemap import blit
OUT="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/overcooked_jax"
z=np.load(os.path.join(OUT,"blit_input_cramped_room.npz"))
atlas=torch.from_numpy(z["atlas"]).cuda().contiguous(); TS=int(z["tile_size"])
out={}
for name,(H,W) in {"v2_cramped_room_128x160":(4,5),"v1_cramped_room_192x224":(6,7)}.items():
    rows=[]
    for B in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]:
        idx=torch.randint(0,atlas.shape[0],(B,H,W),device="cuda",dtype=torch.int32)
        o=torch.empty(B,H*TS,W*TS,3,dtype=torch.uint8,device="cuda")
        for _ in range(5): blit(atlas,idx,TS,out=o)
        torch.cuda.synchronize()
        it=max(5,min(400,int(3e9/(o.numel()+1))))
        t0=time.perf_counter()
        for _ in range(it): blit(atlas,idx,TS,out=o)
        torch.cuda.synchronize()
        dt=(time.perf_counter()-t0)/it
        rows.append(dict(batch=B,fps=B/dt,gbps=o.numel()*2/dt/1e9))
        del o,idx; torch.cuda.empty_cache()
    best=max(rows,key=lambda r:r["fps"])
    out[name]=dict(frame=[H*TS,W*TS,3],rows=rows,best_fps=best["fps"],best_batch=best["batch"],
                   fps_b1=rows[0]["fps"],best_gbps=best["gbps"])
json.dump(out,open(os.path.join(OUT,"a5_blit_shapes.json"),"w"),indent=2)
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!="rows"} for k,v in out.items()},indent=2))
