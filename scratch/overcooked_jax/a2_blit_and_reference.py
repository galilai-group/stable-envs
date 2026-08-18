"""A2: two things the JAX venv cannot do, done in the venv that has torch and overcooked_ai_py.

  1. PARITY + THROUGHPUT. Take the atlas and the per-frame index that a1 harvested from JaxMARL's own
     renderer and push them through ``stable_envs.render.tilemap.blit`` -- the repo's existing kernel,
     unmodified. Diff against JaxMARL's frames before timing anything, because a throughput number
     from a decoder that draws something else is the RoboCasa mistake. Then sweep the batch and state
     the crossover rather than one ratio.

  2. IS IT THE SAME PICTURE? Render `overcooked_ai_py`'s `StateVisualizer` for the same layout and
     compare it to JaxMARL's frame, at JaxMARL's tile size and at the reference's. This repo does not
     silently substitute a reference, so the answer has to be a measurement and not a shrug.
"""
import os, io, sys, json, time
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import torch
from PIL import Image
from stable_envs.render.tilemap import blit, HAVE_TRITON

OUT = os.path.dirname(os.path.abspath(__file__))
res = {"have_triton": HAVE_TRITON,
       "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def decode_fps(img, quality=90, n=None):
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()
    n = n or max(200, int(4e7 / img.size * 3))
    for _ in range(20):
        np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    t0 = time.perf_counter()
    for _ in range(n):
        np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    return n / (time.perf_counter() - t0), len(b)


# ------------------------------------------------------------------ 1. blit parity + throughput
z = np.load(os.path.join(OUT, "blit_input_cramped_room.npz"))
atlas, idx, frames, TS = z["atlas"], z["idx"], z["frames"], int(z["tile_size"])
assert len(np.unique(frames[0])) > 1, "reference frame is uniform; parity would be meaningless"
a_t = torch.from_numpy(atlas).cuda().contiguous()
i_t = torch.from_numpy(idx).cuda().to(torch.int32)
got = blit(a_t, i_t, TS).cpu().numpy()
d = np.abs(got.astype(np.int32) - frames.astype(np.int32))
res["blit_parity"] = dict(
    frames=int(frames.shape[0]), frame_shape=list(frames.shape[1:]),
    atlas_tiles=int(atlas.shape[0]), atlas_bytes=int(atlas.nbytes),
    max_abs=int(d.max()), exact_frames=int((d.reshape(d.shape[0], -1).max(1) == 0).sum()),
    pixel_exact_frac=float((d == 0).mean()),
    reference_distinct_colors=int(len(np.unique(frames[0].reshape(-1, 3), axis=0))))

H, W = idx.shape[1], idx.shape[2]
rows = []
for B in BATCHES:
    rep = i_t[torch.randint(0, i_t.shape[0], (B,), device="cuda")].contiguous()
    try:
        out = torch.empty(B, H * TS, W * TS, 3, dtype=torch.uint8, device="cuda")
        for _ in range(5):
            blit(a_t, rep, TS, out=out)
        torch.cuda.synchronize()
        it = max(5, min(400, int(3e9 / (out.numel() + 1))))
        t0 = time.perf_counter()
        for _ in range(it):
            blit(a_t, rep, TS, out=out)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / it
        rows.append(dict(batch=B, fps=B / dt, gbps=out.numel() * 2 / dt / 1e9))
        del out
    except torch.cuda.OutOfMemoryError:
        break
    torch.cuda.empty_cache()
dec, jb = decode_fps(frames[0])
best = max(rows, key=lambda r: r["fps"])
res["blit_throughput"] = dict(
    rows=rows, best_fps=best["fps"], best_batch=best["batch"], best_gbps=best["gbps"],
    decode_fps_per_core=dec, jpeg_bytes=jb,
    cores_of_decode_replaced=best["fps"] / dec,
    crossover_batch_vs_decode=next((r["batch"] for r in rows if r["fps"] > dec), None))

# ------------------------------------------------------------------ 2. same picture?
ref = {}
try:
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
    from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
    import overcooked_ai_py
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    st = mdp.get_standard_start_state()
    vis = StateVisualizer()
    surf = vis.render_state(st, mdp.terrain_mtx)
    import pygame
    r75 = pygame.surfarray.array3d(surf).swapaxes(0, 1)
    vis15 = StateVisualizer(tile_size=StateVisualizer.UNSCALED_TILE_SIZE)
    r15 = pygame.surfarray.array3d(vis15.render_state(st, mdp.terrain_mtx)).swapaxes(0, 1)
    gfx = os.path.join(os.path.dirname(overcooked_ai_py.__file__), "data", "graphics")
    assets = sorted(os.listdir(gfx)) if os.path.isdir(gfx) else []
    j15 = np.load(os.path.join(OUT, "jaxmarl_v1_cramped_room_t15.npy"))
    j75 = np.load(os.path.join(OUT, "jaxmarl_v1_cramped_room_t75.npy"))

    def cmp(a, b):
        if a.shape != b.shape:
            return dict(shape_a=list(a.shape), shape_b=list(b.shape), same_shape=False)
        dd = np.abs(a.astype(int) - b.astype(int))
        return dict(shape_a=list(a.shape), shape_b=list(b.shape), same_shape=True,
                    max_abs=int(dd.max()), pixel_exact_frac=float((dd == 0).all(-1).mean()))
    ref = dict(
        ok=True,
        overcooked_ai_version=getattr(overcooked_ai_py, "__version__", "1.1.0"),
        reference_tile_size=StateVisualizer.DEFAULT_VALUES["tile_size"],
        reference_unscaled_tile_size=StateVisualizer.UNSCALED_TILE_SIZE,
        reference_font_size=StateVisualizer.DEFAULT_VALUES.get("cooking_timer_font_size"),
        reference_font_color=str(StateVisualizer.DEFAULT_VALUES.get("cooking_timer_font_color")),
        reference_shape_t75=list(r75.shape), reference_shape_t15=list(r15.shape),
        reference_distinct_colors_t75=int(len(np.unique(r75.reshape(-1, 3), axis=0))),
        jaxmarl_shape_t15=list(j15.shape), jaxmarl_shape_t75=list(j75.shape),
        jaxmarl_distinct_colors_t75=int(len(np.unique(j75.reshape(-1, 3), axis=0))),
        graphics_assets=assets, n_graphics_assets=len(assets),
        graphics_bytes=int(sum(os.path.getsize(os.path.join(gfx, a)) for a in assets)),
        compare_t75=cmp(r75, j75), compare_t15=cmp(r15, j15),
    )
    np.save(os.path.join(OUT, "overcooked_ai_cramped_room_t75.npy"), r75)
    np.save(os.path.join(OUT, "overcooked_ai_cramped_room_t15.npy"), r15)
except Exception as e:                                                  # noqa: BLE001
    import traceback; traceback.print_exc()
    ref = dict(ok=False, error=f"{type(e).__name__}: {e}")
res["same_picture"] = ref

with open(os.path.join(OUT, "a2_blit_and_reference.json"), "w") as f:
    json.dump(res, f, indent=2, default=str)
print(json.dumps(res, indent=2, default=str)[:7000])
print("WROTE", os.path.join(OUT, "a2_blit_and_reference.json"))
