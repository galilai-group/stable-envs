"""Official frames + segmentation on the FROZEN Step-4 corpus, one mujoco version per run (Apple GL)."""
import sys, os, json, numpy as np, mujoco, importlib.metadata as md
from dm_control import suite
states_dir, out = sys.argv[1], sys.argv[2]
DOM = {"DMCCheetahRun": ("cheetah","run"), "DMCHopperHop": ("hopper","hop"),
       "DMCWalkerWalk": ("walker","walk"), "DMCHumanoidStand": ("humanoid","stand")}
saved = {"__meta__": np.array(json.dumps({"mujoco": mujoco.__version__, "dm_control": md.version("dm_control"), "gl": os.environ.get("MUJOCO_GL")}))}
for cfg, (dom, task) in DOM.items():
    st = np.load(f"{states_dir}/{cfg}.npz", allow_pickle=False)["states"].astype(np.float64)
    env = suite.load(dom, task, task_kwargs={"random": 0}); env.reset()
    p = env.physics; m = p.model.ptr
    assert st.shape[1] == m.nq, (cfg, st.shape, m.nq)
    rgb, seg = [], []
    for row in st:
        with p.reset_context():
            p.data.qpos[:] = row; p.data.qvel[:] = 0
        rgb.append(p.render(height=84, width=84, camera_id=0).copy())
        seg.append(p.render(height=84, width=84, camera_id=0, segmentation=True).copy())
    saved[cfg + "_rgb"] = np.stack(rgb); saved[cfg + "_seg"] = np.stack(seg).astype(np.int16)
    saved[cfg + "_floor_ids"] = np.array([g for g in range(m.ngeom) if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)])
    print(cfg, "done", mujoco.__version__, flush=True)
np.savez_compressed(out, **saved)
