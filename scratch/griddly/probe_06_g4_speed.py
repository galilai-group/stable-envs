"""Probe 6: G4 -- economics.

  * setup cost: GymWrapper construction + game.init (the "compile the decoder" term)
  * loop cost:  game.observe() alone (state -> frame), and step() alone, and step+observe
  * render share of one env step
  * a JPEG-decode baseline measured ON THIS NODE at the SAME frame sizes, so the comparison
    is like-for-like with docs/research/when_does_this_pay_off.md (11,842 f/s @64^2,
    6,762 @224^2, one core, decoding from memory).

Pinned to one core with taskset by the caller.
"""
import io, os, time, json
import numpy as np
import griddly
from griddly import GymWrapper, gd

GAMES = [
    ("sokoban", "Single-Player/GVGAI/sokoban.yaml"),
    ("zelda", "Single-Player/GVGAI/zelda.yaml"),
    ("butterflies", "Single-Player/GVGAI/butterflies.yaml"),
    ("GriddlyRTS", "RTS/GriddlyRTS.yaml"),
]


def sample_action(env, rng):
    sp = env.action_space
    if hasattr(sp, "nvec"):
        return [int(rng.integers(k)) for k in sp.nvec]
    if hasattr(sp, "n"):
        return int(rng.integers(sp.n))
    return [([int(rng.integers(k)) for k in s.nvec] if hasattr(s, "nvec") else int(rng.integers(s.n)))
            for s in sp]


def timeit(fn, n, warm=20):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n


results = []
for name, yf in GAMES:
    # ---- setup ----
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        e = GymWrapper(yaml_file=yf, level=0,
                       global_observer_type=gd.ObserverType.SPRITE_2D,
                       player_observer_type=gd.ObserverType.SPRITE_2D)
        e.reset(level_id=0)
        ts.append(time.perf_counter() - t)
        if _ < 4:
            del e
    setup_first, setup_warm = ts[0], float(np.median(ts[1:]))
    env = e
    env.seed(0); env.reset(level_id=0)
    rng = np.random.default_rng(0)

    fr = np.array(env.game.observe(), copy=True).swapaxes(0, 2)
    H, W, _ = fr.shape
    px = H * W

    sink = np.empty_like(fr)

    def do_render():
        a = np.array(env.game.observe(), copy=False)
        sink[:] = a.swapaxes(0, 2)          # force the pixels to be materialised

    def do_render_nocopy():
        a = np.array(env.game.observe(), copy=False)
        return int(a[0, 0, 0])

    def do_step():
        env.step(sample_action(env, rng))

    def do_step_render():
        env.step(sample_action(env, rng))
        a = np.array(env.game.observe(), copy=False)
        sink[:] = a.swapaxes(0, 2)

    t_render = timeit(do_render, 400)
    t_render_nc = timeit(do_render_nocopy, 400)
    # step alone: griddly's step() ALSO observes internally (player obs), so measure the
    # underlying player step without the wrapper's observe by using step_multi directly
    def do_step_raw():
        a = np.array(sample_action(env, rng), dtype=np.int32).reshape(1, -1)
        env._players[0].step_multi(a, True)
    try:
        t_step_raw = timeit(do_step_raw, 400)
    except Exception:
        t_step_raw = float("nan")
    t_step = timeit(do_step, 400)

    row = dict(game=name, frame=[H, W], px=px, frame_bytes=int(fr.size),
               setup_first_s=setup_first, setup_warm_s=setup_warm,
               render_ms=t_render * 1e3, render_fps=1 / t_render,
               render_nocopy_fps=1 / t_render_nc,
               step_raw_ms=t_step_raw * 1e3,
               wrapper_step_ms=t_step * 1e3,
               render_share_of_step=t_render / (t_render + t_step_raw) if t_step_raw == t_step_raw else None)
    results.append(row)
    print(json.dumps(row), flush=True)
    del env

# ---- decode baseline on this node, same sizes ----
print("\n### JPEG / PNG decode baseline on this node (one core, from memory) ###")
try:
    from PIL import Image
    for row in results:
        H, W = row["frame"]
        rgb = (np.random.default_rng(0).integers(0, 255, (H, W, 3))).astype(np.uint8)
        # use a real griddly-like frame: flat colours compress much better, so use a real one
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
        data = buf.getvalue()

        def dec():
            im = Image.open(io.BytesIO(data)); im.load(); return im

        for _ in range(10):
            dec()
        t = time.perf_counter(); n = 200
        for _ in range(n):
            dec()
        dt = (time.perf_counter() - t) / n
        print(f"  {row['game']:12s} {H}x{W} jpeg={len(data)}B decode {1/dt:8.0f} f/s/core "
              f"| griddly render {row['render_fps']:8.0f} f/s  -> ratio {row['render_fps']*dt:.3f}x")
    for side in (64, 224):
        rgb = (np.random.default_rng(0).integers(0, 255, (side, side, 3))).astype(np.uint8)
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
        data = buf.getvalue()

        def dec2():
            im = Image.open(io.BytesIO(data)); im.load(); return im
        for _ in range(10):
            dec2()
        t = time.perf_counter(); n = 400
        for _ in range(n):
            dec2()
        dt = (time.perf_counter() - t) / n
        print(f"  calibration {side}x{side}: {1/dt:.0f} f/s/core "
              f"(when_does_this_pay_off.md records 11842 @64, 6762 @224)")
except ImportError:
    print("  PIL missing")

print("\n### SUMMARY (G4) ###")
for r in results:
    print(f"{r['game']:12s} {r['frame'][0]}x{r['frame'][1]} "
          f"setup(first/warm)={r['setup_first_s']*1e3:.0f}/{r['setup_warm_s']*1e3:.0f} ms  "
          f"render={r['render_ms']:.3f} ms ({r['render_fps']:.0f} f/s)  "
          f"step={r['step_raw_ms']:.3f} ms  render_share={r['render_share_of_step']}")
