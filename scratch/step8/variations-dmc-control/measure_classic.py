"""Classic control: is SWM's exact-match post-render recolor the same picture as our draw-in-colour?"""
import json, sys, numpy as np, torch, gymnasium, stable_worldmodel  # noqa
from stable_envs.core.env import make

PAIRS = {  # swm id -> (our cell, {swm leaf: [our leaves it recolours]})
 "swm/AcrobotControl-v1": ("Acrobot", {"visuals.bg":["background.color"],"visuals.link":["link.color"],
      "visuals.joint":["joint.color"],"visuals.line":["ground.color"]}),
 "swm/CartPoleControl-v1": ("CartPole", {"visuals.bg":["background.color"],
      "visuals.cart":["cart.color","track.color"],"visuals.pole":["pole.color"],"visuals.axle":["axle.color"]}),
 "swm/PendulumControl-v1": ("Pendulum", {"visuals.bg":["background.color"],"visuals.rod":["rod.color"],
      "visuals.hub":["axle.color"]}),
 "swm/MountainCarControl-v0": ("MountainCar", {"visuals.bg":["background.color"],
      "visuals.fg":["car.color","terrain.color","flag.pole_color"]}),
 "swm/MountainCarContinuousControl-v0": ("MountainCarContinuous", {"visuals.bg":["background.color"],
      "visuals.fg":["car.color","terrain.color","flag.pole_color"]}),
}
def holder(e):
    n=e
    while n is not None:
        if getattr(n,"variation_space",None) is not None: return n
        n=getattr(n,"env",None)

out={}
rng=np.random.default_rng(0)
for sid,(cell,m) in PAIRS.items():
    env = make(cell, batch=1, device="cpu")
    cls = type(env)
    states = env.sample_states(8, rng=np.random.default_rng(7))
    e = gymnasium.make(sid, render_mode="rgb_array"); h = holder(e); vs = h.variation_space
    ref = h.env.unwrapped
    h.reset(seed=0)   # build the pygame surface etc.
    layout = {p:(o,w) for p,o,w in env.variation_space.flat_layout()}

    def swm_frames(vals):
        for k,v in vals.items():
            node=vs
            for p in k.split("."): node=node[p]
            node.set_value(np.asarray(v, node.dtype))
        fr=[]
        for st in states:
            cls.reference_set_state(ref, st)
            fr.append(h.render().copy())
        return np.stack(fr)

    def se_frames(vals):
        row = np.array(env.default_style(), np.float32).reshape(-1).copy()
        for k,v in vals.items():
            u8 = np.rint(np.clip(np.asarray(v,np.float64),0,1)*255.0)  # swm's own quantisation
            for ours in m[k]:
                o,w = layout[ours]; row[o:o+w] = u8[:w]
        sty = np.repeat(row[None], len(states), 0)
        return env.render(torch.as_tensor(states), style=torch.as_tensor(sty)).cpu().numpy()

    rec = {"cell": cell, "cases": {}}
    # case 0: the registered default
    defaults = {k: np.asarray(vs[k.split('.')[0]][k.split('.')[1]].init_value) for k in m}
    cases = {"default": defaults}
    # case i: each leaf alone at a probe value
    for k in m:
        node = vs[k.split('.')[0]][k.split('.')[1]]
        lo,hi,iv = np.asarray(node.low,np.float64),np.asarray(node.high,np.float64),np.asarray(node.init_value,np.float64)
        far = np.where(np.abs(iv-lo)>np.abs(iv-hi), lo, hi)
        v = iv + 0.9*(far-iv)
        cases[k] = {**defaults, k: v}
    cases["all_random"] = {k: np.asarray(vs[k.split('.')[0]][k.split('.')[1]].sample(), np.float64) for k in m}
    for name,vals in cases.items():
        A = swm_frames(vals); B = se_frames(vals)
        d = np.abs(A.astype(np.int32)-B.astype(np.int32))
        rec["cases"][name] = {"values": {k: np.asarray(v).tolist() for k,v in vals.items()},
            "px_diff": int((d.max(-1)>0).sum()), "px_total": int(A.shape[0]*A.shape[1]*A.shape[2]),
            "max_abs": int(d.max()), "exact_frac": float((d.max(-1)==0).mean())}
        print(sid, name, rec["cases"][name]["px_diff"], rec["cases"][name]["max_abs"], file=sys.stderr)
    out[sid]=rec
    e.close()
json.dump(out, open(sys.argv[1],"w"), indent=1)
