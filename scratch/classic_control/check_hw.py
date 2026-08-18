import os, sys
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pygame
from pygame import gfxdraw
import gpu_render as R, ref_gfx as G
pygame.init()
bad=[]
for r in range(0, 41):
    S=4*r+9; c=S//2
    s=pygame.Surface((S,S)); s.fill((255,255,255))
    gfxdraw.filled_circle(s,c,c,r,(0,0,0))
    a=np.transpose(np.array(pygame.surfarray.pixels3d(s)),(1,0,2))
    m=(a==0).all(-1)
    meas=[]
    for dy in range(r+1):
        xs=np.where(m[c+dy])[0]
        meas.append(int(xs.max()-c) if xs.size else -1)
    got=R.circle_hw(r); gotg=G._midpoint_hw(r)
    if got!=meas or gotg!=meas:
        bad.append(r); print(f"  r={r} MISMATCH measured={meas}\n        gpu={got}\n        ref={gotg}")
print(f"circle_hw exact for r=0..40: {len(bad)==0}  (mismatches: {bad})")
