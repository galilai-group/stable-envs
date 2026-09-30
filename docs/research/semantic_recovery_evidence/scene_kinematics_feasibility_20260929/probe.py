import json, time
import mujoco, numpy as np
from stable_envs._adapters.ogbench_manipulation_compat import CompatibleScene

e=CompatibleScene(seed=0)
e._mjcf_model=e.modify_mjcf_model(e.build_mjcf_model())
e.compile_model_and_data()
m=e.model
ref=mujoco.MjData(m)
candidate=mujoco.MjData(m)
opt=mujoco.MjvOption()
cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FIXED; cam.fixedcamid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'front_pixels')
scene=[mujoco.MjvScene(m,maxgeom=10000) for _ in range(2)]
geom_names=['type','dataid','objtype','objid','category','texcoord','segid','matid','size','pos','mat','rgba','emission','specular','shininess','reflectance','label']
def fields(obj):
 out={}
 for name in dir(obj):
  if name.startswith('_'):continue
  value=getattr(obj,name)
  if isinstance(value,(np.ndarray,float,int,str,bool,np.number)):out[name]=np.asarray(value).copy()
 return out

def snapshot(s):
 out={'ngeom':s.ngeom,'nlight':s.nlight}
 for i,g in enumerate(s.geoms[:s.ngeom]):
  for k in geom_names:
   if hasattr(g,k):out[f'geom/{i}/{k}']=np.asarray(getattr(g,k)).copy()
 for label,items in [('light',s.lights[:s.nlight]),('camera',s.camera)]:
  for i,obj in enumerate(items):
   for k,v in fields(obj).items():out[f'{label}/{i}/{k}']=v
 return out

rng=np.random.default_rng(17)
poses=[]; mismatches=[]; maximum=0
for i in range(128):
 q=m.qpos0.copy()+rng.normal(0,.15,m.nq)
 mujoco.mj_normalizeQuat(m,q)
 mocap_pos=rng.uniform([.2,-.3,.02],[.6,.3,.4],(m.nmocap,3))
 mocap_quat=rng.normal(size=(m.nmocap,4));mocap_quat/=np.linalg.norm(mocap_quat,axis=-1,keepdims=True)
 poses.append((q,mocap_pos,mocap_quat))
 for d in [ref,candidate]:
  d.qpos[:]=q;d.mocap_pos[:]=mocap_pos;d.mocap_quat[:]=mocap_quat
 mujoco.mj_forward(m,ref)
 mujoco.mj_kinematics(m,candidate);mujoco.mj_camlight(m,candidate)
 for d,s in zip([ref,candidate],scene):mujoco.mjv_updateScene(m,d,opt,None,cam,mujoco.mjtCatBit.mjCAT_ALL,s)
 a,b=map(snapshot,scene)
 for k,v in a.items():
  if not np.array_equal(v,b[k]):mismatches.append((i,k))
 assert np.array_equal(candidate.qpos,q)

samples={}
for name,fn in [('forward',lambda:mujoco.mj_forward(m,candidate)),('kinematics_camlight',lambda:(mujoco.mj_kinematics(m,candidate),mujoco.mj_camlight(m,candidate)))]:
 times=[]
 for repeat in range(10):
  start=time.perf_counter_ns()
  for q,p,r in poses:
   candidate.qpos[:]=q;candidate.mocap_pos[:]=p;candidate.mocap_quat[:]=r;fn()
  times.append((time.perf_counter_ns()-start)/1e6/len(poses))
 samples[name]={'ms_per_pose_mean':float(np.mean(times)),'ms_per_pose_sd':float(np.std(times,ddof=1))}
print(json.dumps(dict(mujoco=mujoco.__version__,poses=len(poses),scene_attributes=len(a),mismatches=mismatches,model={n:np.asarray(getattr(m,n)).tolist() for n in ['nq','nv','ntendon','nflex','nskin','nplugin','nlight','ncam','cam_mode','light_mode','tendon_num','wrap_type']},timings=samples),indent=2))
