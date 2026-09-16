import numpy as np, gymnasium, stable_worldmodel, mujoco  # noqa
def holder(e):
    n=e
    while n is not None:
        if getattr(n,"variation_space",None) is not None: return n
        n=getattr(n,"env",None)
for i in ["swm/AcrobotDMControl-v0","swm/BallInCupDMControl-v0","swm/FingerDMControl-v0",
          "swm/ReacherDMControl-v0","swm/ManipulatorDMControl-v0"]:
    e=gymnasium.make(i); h=holder(e); h.reset(seed=1)
    m=h.env.physics.model; m=m.ptr if hasattr(m,"ptr") else m
    print("==",i)
    for s in range(m.nsite):
        nm=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,s)
        if nm and "target" in nm:
            mid=int(m.site_matid[s])
            print(f"   site {nm}: group={int(m.site_group[s])} rgba={np.asarray(m.site_rgba[s]).tolist()} matid={mid}"
                  + (f" mat_rgba={np.asarray(m.mat_rgba[mid]).tolist()}" if mid>=0 else ""))
    for g in range(m.ngeom):
        nm=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g)
        if nm and "target" in nm:
            mid=int(m.geom_matid[g])
            print(f"   geom {nm}: group={int(m.geom_group[g])} rgba={np.asarray(m.geom_rgba[g]).tolist()} matid={mid}"
                  + (f" mat_rgba={np.asarray(m.mat_rgba[mid]).tolist()}" if mid>=0 else ""))
    # also: geoms carrying explicit mass (why density is ignored)
    e.close()
for i in ["swm/AcrobotDMControl-v0","swm/CartpoleDMControl-v0","swm/PendulumDMControl-v0"]:
    e=gymnasium.make(i); h=holder(e)
    src=h._mjcf_model.to_xml_string()
    import re
    print("==",i,"geoms with explicit mass:",re.findall(r'<geom[^>]*name="([^"]+)"[^>]*mass="([^"]+)"',src) or
          re.findall(r'<geom[^>]*mass="([^"]+)"[^>]*name="([^"]+)"',src))
    e.close()
