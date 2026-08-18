import os, sys
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_gfx as G
for r in (1,2,5):
    a = sorted(G.bake_aacircle(r,(0,0,0)))
    b = sorted(G.bake_aacircle(r,(204,77,77)))
    print(f"r={r}: black {len(a)} entries, red {len(b)} entries")
    sa = {(x,y) for (x,y,_,_) in a}; sb = {(x,y) for (x,y,_,_) in b}
    print(f"   positions equal: {sa==sb}   only-black {sorted(sa-sb)[:6]}   only-red {sorted(sb-sa)[:6]}")
    da = {(x,y,p):w for (x,y,w,p) in a}; db = {(x,y,p):w for (x,y,w,p) in b}
    diff = [(k, da.get(k), db.get(k)) for k in sorted(set(da)|set(db)) if da.get(k)!=db.get(k)]
    print(f"   weight mismatches: {len(diff)}  e.g. {diff[:6]}")
