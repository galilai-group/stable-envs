"""Is the swm REGISTERED DEFAULT reproducible through the mapping?"""
import json,sys,numpy as np,torch
from stable_envs.core.env import make
from stable_envs.compat.worldmodel import variation_map, style_from_variation, variation_from_style
from stable_envs.compat.variation_maps.dmc_control import MAPS, swm_defaults
out={}
cache={}
for cell in MAPS:
    env=make(cell,batch=1,device="cpu"); m=variation_map(env)
    if m.swm_id not in cache: cache[m.swm_id]=swm_defaults(m.swm_id)
    d=cache[m.swm_id]
    base=np.asarray(env.default_style(),np.float32).reshape(-1)
    mapped=style_from_variation(env,d)
    lay=env.variation_space.flat_layout()
    diff={p:{"default":base[o:o+w].tolist(),"mapped":mapped[o:o+w].tolist()}
          for p,o,w in lay if not np.array_equal(base[o:o+w],mapped[o:o+w])}
    rec={"swm_id":m.swm_id,"style_slots_differing":diff}
    # pixels at the registered size over 8 states
    sts=env.sample_states(8,rng=np.random.default_rng(3))
    st=torch.as_tensor(sts)
    a=env.render(st,style=torch.as_tensor(np.repeat(base[None],8,0))).cpu().numpy()
    b=env.render(st,style=torch.as_tensor(np.repeat(mapped[None],8,0))).cpu().numpy()
    dd=np.abs(a.astype(np.int32)-b.astype(np.int32))
    rec["px_default_vs_mapped"]=int((dd.max(-1)>0).sum()); rec["px_total"]=int(a[...,0].size)
    rec["max_abs"]=int(dd.max()); rec["size"]=int(a.shape[1])
    # round trip
    back=variation_from_style(env,mapped)
    rt={}
    for k,v in back.items():
        if k in d:
            rt[k]=float(np.max(np.abs(np.asarray(v,np.float64).reshape(-1)-np.asarray(d[k],np.float64).reshape(-1))))
    rec["roundtrip_max_abs"]=rt
    out[cell]=rec
    print(f"{cell:26s} px={rec['px_default_vs_mapped']:7d}/{rec['px_total']:7d} max={rec['max_abs']:3d} "
          f"slots={sorted(diff)} rt_worst={max(rt.values()) if rt else 0:.3e}")
json.dump(out,open(sys.argv[1],"w"),indent=1,default=float)
