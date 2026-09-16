import numpy as np, gymnasium, stable_worldmodel, hashlib, mujoco  # noqa
def holder(e):
    n=e
    while n is not None:
        if getattr(n,"variation_space",None) is not None: return n
        n=getattr(n,"env",None)
# camera modes for each domain
for i in ["swm/AcrobotDMControl-v0","swm/BallInCupDMControl-v0","swm/CartpoleDMControl-v0",
          "swm/CheetahDMControl-v0","swm/FingerDMControl-v0","swm/HopperDMControl-v0",
          "swm/HumanoidDMControl-v0","swm/ManipulatorDMControl-v0","swm/PendulumDMControl-v0",
          "swm/QuadrupedDMControl-v0","swm/ReacherDMControl-v0","swm/WalkerDMControl-v0"]:
    e=gymnasium.make(i); h=holder(e); m=h.env.physics.model
    m=m.ptr if hasattr(m,"ptr") else m
    modes={0:"fixed",1:"track",2:"trackcom",3:"targetbody",4:"targetbodycom"}
    cams=[(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_CAMERA,c), modes.get(int(m.cam_mode[c]),int(m.cam_mode[c])))
          for c in range(m.ncam)]
    print(f"{i:36s} cam_id={h.camera_id} ncam={m.ncam} {cams} ntendon={m.ntendon} nlight={m.nlight} light_diffuse={np.asarray(m.light_diffuse).tolist()}")
    e.close()
