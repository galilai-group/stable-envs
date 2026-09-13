import hashlib, json, sys, numpy as np, mujoco, importlib.metadata as md
from dm_control import suite
res = {"mujoco": mujoco.__version__, "dm_control": md.version("dm_control"), "domains": {}}
for dom,task in (("cheetah","run"),("hopper","hop"),("walker","walk"),("humanoid","stand"),("reacher","hard")):
    env=suite.load(dom,task,task_kwargs={"random":0}); env.reset()
    p=env.physics; m=p.model.ptr
    D={"ntex":int(m.ntex),"nmat":int(m.nmat),"textures":[], "materials":[]}
    for t in range(m.ntex):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t)
        adr=int(m.tex_adr[t]); w=int(m.tex_width[t]); h=int(m.tex_height[t])
        nch=int(m.tex_nchannel[t]) if hasattr(m,'tex_nchannel') else 3
        n=w*h*nch
        data=np.asarray(m.tex_data[adr:adr+n])
        D["textures"].append(dict(id=t,name=name,type=int(m.tex_type[t]),w=w,h=h,nch=nch,adr=adr,
            sha=hashlib.sha256(data.tobytes()).hexdigest()[:16],mean=round(float(data.mean()),3),
            distinct=int(len(np.unique(data.reshape(-1,nch),axis=0))), min=int(data.min()), max=int(data.max())))
    for i in range(m.nmat):
        nm=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i)
        D["materials"].append(dict(id=i,name=nm,texid=np.asarray(m.mat_texid[i]).reshape(-1).tolist(),
            texrepeat=m.mat_texrepeat[i].tolist(),texuniform=int(m.mat_texuniform[i]),
            rgba=np.round(m.mat_rgba[i],4).tolist(),reflectance=float(m.mat_reflectance[i]),
            shininess=float(m.mat_shininess[i]),specular=float(m.mat_specular[i]),
            emission=float(m.mat_emission[i])))
    D["vis"]={"offsamples":int(m.vis.quality.offsamples),"shadowsize":int(m.vis.quality.shadowsize),
              "offwidth":int(m.vis.global_.offwidth),"offheight":int(m.vis.global_.offheight)}
    D["ngeom"]=int(m.ngeom); D["nlight"]=int(m.nlight)
    D["geom_matid"]=np.asarray(m.geom_matid).reshape(-1).tolist()
    D["light"]={"castshadow":np.asarray(m.light_castshadow).reshape(-1).tolist(),"diffuse":np.round(m.light_diffuse,4).tolist(),"specular":np.round(m.light_specular,4).tolist(),"ambient":np.round(m.light_ambient,4).tolist()}
    D["global_headlight"]={"ambient":np.round(np.asarray(m.vis.headlight.ambient),4).tolist(),"diffuse":np.round(np.asarray(m.vis.headlight.diffuse),4).tolist(),"specular":np.round(np.asarray(m.vis.headlight.specular),4).tolist(),"active":int(m.vis.headlight.active)}
    res["domains"][f"{dom}-{task}"]=D
out=sys.argv[1]
json.dump(res, open(out,"w"), indent=1)
for k,D in res["domains"].items():
    print(k, "ntex",D["ntex"], [(t["name"],t["type"],t["w"],t["h"],t["nch"],t["sha"],t["distinct"]) for t in D["textures"]])
    print("   mats", [(mm["name"],mm["texid"],mm["texrepeat"],mm["texuniform"]) for mm in D["materials"]])
