import os, sys
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pygame, gymnasium as gym
import ref_envs as E, ref_gfx as G

pygame.init()
print("== pygame.draw.line with float endpoints: which row? ==")
for yf in (363.6363636363636, 20.2, 20.5, 20.8, 21.0):
    s = pygame.Surface((64,64)); s.fill((255,255,255))
    pygame.draw.line(s,(0,0,0),(0.0,yf),(60.0,yf))
    a=np.transpose(np.array(pygame.surfarray.pixels3d(s)),(1,0,2))
    ys,xs=np.where((a!=255).any(-1))
    rng_ = f"x[{xs.min()},{xs.max()}]" if xs.size else "EMPTY"
    print(f"  y={yf:.6f} -> rows {sorted(set(ys.tolist()))}  trunc={int(yf)} round={int(round(yf))}  {rng_}")
s = pygame.Surface((500,500)); s.fill((255,255,255))
sc=500/4.4; off=250.0
pygame.draw.line(s,(0,0,0),(-2.2*sc+off, 1*sc+off),(2.2*sc+off, 1*sc+off))
a=np.transpose(np.array(pygame.surfarray.pixels3d(s)),(1,0,2))
ys,xs=np.where((a!=255).any(-1))
print(f"  acrobot ground: rows {sorted(set(ys.tolist()))} x[{xs.min()},{xs.max()}] n={len(xs)}  (1*sc+off={1*sc+off:.4f})")

print("\n== acrobot residual map (aa=True) ==")
rng=np.random.default_rng(0)
env=gym.make("Acrobot-v1",render_mode="rgb_array").unwrapped; env.reset(seed=0)
tot=None
for i in range(8):
    env.state=np.array([rng.uniform(-np.pi,np.pi),rng.uniform(-np.pi,np.pi),0.,0.])
    f=env.render().copy()
    g=E.render_acrobot(float(env.state[0]),float(env.state[1]),aa=True)
    d=(np.abs(f.astype(int)-g.astype(int)).max(-1)>0)
    tot = d if tot is None else (tot|d)
    ys,xs=np.where(d)
    rows={}
    for y in ys: rows[y]=rows.get(y,0)+1
    top=sorted(rows.items(),key=lambda kv:-kv[1])[:4]
    print(f"  frame {i}: {d.sum():5d} diff px; busiest rows {top}")
ys,xs=np.where(tot)
print(f"  union rows present: {len(set(ys.tolist()))}; row histogram top: ", sorted({y:int((ys==y).sum()) for y in set(ys.tolist())}.items(), key=lambda kv:-kv[1])[:5])
env.close()

print("\n== mountaincar residual (aa=True) ==")
env=gym.make("MountainCar-v0",render_mode="rgb_array").unwrapped; env.reset(seed=0)
P=E.MOUNTAINCAR
surf=pygame.Surface((P["W"],P["H"])); surf.fill((255,255,255))
xs_=np.linspace(P["minp"],P["maxp"],100); ys_=E.mc_height(xs_)
pygame.draw.aalines(surf,points=list(zip((xs_-P["minp"])*P["scale"],ys_*P["scale"])),closed=False,color=(0,0,0))
base=np.transpose(np.array(pygame.surfarray.pixels3d(surf)),(1,0,2)).astype(np.int64)
flagx=int((0.5-P["minp"])*P["scale"]); flagy1=int(float(E.mc_height(0.5))*P["scale"]); flagy2=flagy1+50
def render_mc(pos,aa=True):
    cv=base.copy(); ang=np.cos(3*pos)
    cxp=(pos-P["minp"])*P["scale"]; cyp=P["clearance"]+float(E.mc_height(pos))*P["scale"]
    car=[(E._rot(c,ang)[0]+cxp,E._rot(c,ang)[1]+cyp) for c in [(-20.,0.),(-20.,20.),(20.,20.),(20.,0.)]]
    E._poly(cv,car,P["car_c"],aa)
    for c in [(10.,0.),(-10.,0.)]:
        c=E._rot(c,ang); E._circle(cv,int(c[0]+cxp),int(c[1]+cyp),8,P["wheel_c"],aa)
    cv[flagy1:flagy2+1,flagx]=np.array((0,0,0),np.int64)
    E._poly(cv,[(flagx,flagy2),(flagx,flagy2-10),(flagx+25,flagy2-5)],(204,204,0),aa)
    return np.flipud(cv).astype(np.uint8)
nb=0
for i in range(40):
    env.state=np.array([rng.uniform(-1.2,0.6),0.])
    f=env.render().copy(); g=render_mc(float(env.state[0]))
    d=np.abs(f.astype(int)-g.astype(int)).max(-1)
    if d.max()>0:
        ys,xs=np.where(d>0); nb+=1
        print(f"  frame {i} pos={env.state[0]:+.3f}: {len(xs)} px at {list(zip((399-ys).tolist(),xs.tolist()))[:6]}  ref={f[ys[0],xs[0]].tolist()} got={g[ys[0],xs[0]].tolist()}")
print(f"  frames with any diff: {nb}/40")
env.close()

print("\n== aacircle bake residuals (fit error per radius) ==")
for r in (5,8,11):
    st=G.bake_aacircle(r)
    errs=[e for (_,_,_,e) in st]
    print(f"  r={r}: {len(st)} fringe px, fit err max {max(errs) if errs else 0}, nonzero {sum(1 for e in errs if e)}")
