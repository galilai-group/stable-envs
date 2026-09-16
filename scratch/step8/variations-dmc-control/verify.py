import json,sys,collections,numpy as np
from stable_envs.core.env import make
from stable_envs.compat.worldmodel import variation_map, check_mapping
from stable_envs.compat.variation_maps.dmc_control import MAPS, EXPECTED_COUNTS
tot=collections.Counter(); bad=[]
rows={}
seen_ids=set()
for cell in MAPS:
    env=make(cell,batch=1,device="cpu")
    r=check_mapping(env)
    m=variation_map(env)
    if r["undocumented"]: bad.append((cell,"undocumented",r["undocumented"]))
    if r["mapped_but_absent_from_swm"]: bad.append((cell,"absent",r["mapped_but_absent_from_swm"]))
    if set(r["ours_only"])!=set(r["ours_only_documented"]):
        bad.append((cell,"ours_only mismatch",sorted(set(r["ours_only"])^set(r["ours_only_documented"]))))
    c=collections.Counter(r["classification"].values())
    exp=EXPECTED_COUNTS[m.swm_id]
    got={"total":len(r["classification"]),**{k:c.get(k,0) for k in ("mapped","unsupported","dead_upstream","not_applicable")}}
    if got!=exp: bad.append((cell,"counts",got,exp))
    if m.swm_id not in seen_ids:
        seen_ids.add(m.swm_id)
        for k,v in got.items():
            if k!="total": tot[k]+=v
        tot["total"]+=got["total"]
    rows[cell]={"swm_id":m.swm_id,**got}
    print(f"{cell:26s} {m.swm_id:38s} {got}")
print("TOTALS over 17 ids:",dict(tot))
print("PROBLEMS:",bad if bad else "none")
