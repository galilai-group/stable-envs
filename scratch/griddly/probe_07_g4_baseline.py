"""Probe 7: the G4 comparison, done like-for-like.

  A. decode baseline on THIS node, reproducing scratch/robocasa/bench_decode.py so the numbers
     are comparable to docs/research/when_does_this_pay_off.md, plus a decode baseline on REAL
     griddly frames at their real sizes.
  B. is TileSize configurable? sweep it by rewriting the GDY yaml text, measure frame size,
     exactness of the W*T x H*T layout, and the official renderer's f/s at each size.
  C. what a gather would cost: the repo's own stable_envs.render.tilemap.blit at griddly's
     frame shape, on this GPU. That is the achievable regeneration rate, measured with the
     repo's existing kernel rather than a new one.
"""
import io, os, sys, time, json, copy
import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

import griddly
from griddly import GymWrapper, gd
from PIL import Image

GRIDDLY_DIR = os.path.dirname(griddly.__file__)
GAMES_DIR = os.path.join(GRIDDLY_DIR, "resources", "games")


def gframe(env):
    return np.array(env.game.observe(), copy=True).swapaxes(0, 2)


def decode_fps(img, quality=90, n=None):
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()
    n = n or max(200, int(4e7 / img.size * 3))
    for _ in range(20):
        np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    t0 = time.perf_counter()
    for _ in range(n):
        np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    dt = (time.perf_counter() - t0) / n
    return 1.0 / dt, len(b)


print("### A1. reproduce bench_decode.py calibration on this node ###")
rng = np.random.default_rng(0)
for res in (64, 96, 128, 224, 256):
    img = np.zeros((res, res, 3), np.uint8)
    img[:] = rng.integers(30, 80, size=(1, 1, 3))
    for _ in range(12):
        y, x = rng.integers(0, res, 2); h, w = rng.integers(res // 8, res // 2, 2)
        img[y:y + h, x:x + w] = rng.integers(0, 255, size=3)
    f, nb = decode_fps(img)
    print(f"  {res:4d}^2 jpeg {nb:7d} B  decode {f:10.0f} f/s/core")

print("\n### A2. decode baseline on REAL griddly frames ###")
GAMES = [("sokoban", "Single-Player/GVGAI/sokoban.yaml"),
         ("zelda", "Single-Player/GVGAI/zelda.yaml"),
         ("butterflies", "Single-Player/GVGAI/butterflies.yaml"),
         ("GriddlyRTS", "RTS/GriddlyRTS.yaml")]
real = {}
for name, yf in GAMES:
    env = GymWrapper(yaml_file=yf, level=0,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     player_observer_type=gd.ObserverType.SPRITE_2D)
    env.seed(0); env.reset(level_id=0)
    fr = gframe(env)
    f, nb = decode_fps(fr, n=300)
    real[name] = (fr.shape, f, nb)
    print(f"  {name:12s} {fr.shape[0]}x{fr.shape[1]} jpeg {nb:7d} B  decode {f:8.0f} f/s/core "
          f"({fr.size/1e6*f:.0f} MB/s raw)")
    del env

print("\n### B. TileSize configurability ###")
import re
base = open(os.path.join(GAMES_DIR, "Single-Player/GVGAI/sokoban.yaml")).read()
for T in (4, 8, 12, 16, 24, 32, 48, 64):
    ys = re.sub(r"(Sprite2D:\s*\n\s*TileSize:\s*)\d+", lambda m: m.group(1) + str(T), base, count=1)
    if f"TileSize: {T}" not in ys:
        print(f"  T={T}: substitution failed"); continue
    try:
        env = GymWrapper(yaml_string=ys, level=0,
                         global_observer_type=gd.ObserverType.SPRITE_2D,
                         player_observer_type=gd.ObserverType.SPRITE_2D)
        env.seed(0); env.reset(level_id=0)
        fr = gframe(env)
        ok = (fr.shape[0] == env.grid_height * T and fr.shape[1] == env.grid_width * T)
        sink = np.empty_like(fr)

        def r():
            sink[:] = np.array(env.game.observe(), copy=False).swapaxes(0, 2)
        for _ in range(20):
            r()
        t = time.perf_counter(); n = 300
        for _ in range(n):
            r()
        dt = (time.perf_counter() - t) / n
        print(f"  TileSize={T:3d} -> frame {fr.shape[0]}x{fr.shape[1]} exact_grid={ok} "
              f"reported_tile={env.get_tile_size(0)} render {1/dt:8.0f} f/s")
        del env
    except Exception as ex:
        print(f"  TileSize={T}: FAILED {ex}")

print("\n### C. what a pure gather costs on this GPU (repo's tilemap.blit) ###")
try:
    import torch
    from stable_envs.render import tilemap as tm
    print("  triton:", tm.HAVE_TRITON, "| cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    dev = "cuda"
    for (name, (shape, dec_fps, _)) in real.items():
        H, W = shape[0], shape[1]
        T = 24 if name != "GriddlyRTS" else 16
        gh, gw = H // T, W // T
        n_tiles = 64
        atlas = torch.randint(0, 255, (n_tiles, T, T, 3), dtype=torch.uint8, device=dev)
        for B in (256, 4096):
            idx = torch.randint(0, n_tiles, (B, gh, gw), dtype=torch.int32, device=dev)
            out = torch.empty(B, gh * T, gw * T, 3, dtype=torch.uint8, device=dev)
            for _ in range(5):
                tm.blit(atlas, idx, T, out=out)
            torch.cuda.synchronize()
            t = time.perf_counter(); n = 30
            for _ in range(n):
                tm.blit(atlas, idx, T, out=out)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t) / n
            fps = B / dt
            print(f"  {name:12s} {H}x{W} T={T} grid={gh}x{gw} B={B:5d}: "
                  f"{fps:12.0f} f/s ({out.numel()/dt/1e9:6.1f} GB/s) "
                  f"= {fps/dec_fps:8.0f} cores of decode")
except Exception as ex:
    import traceback; traceback.print_exc()
    print("  gather bench failed:", ex)
