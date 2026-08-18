"""Everything a corrector has to survive before it can be recommended, in one pass.

Six things, and five of them are attempts to make it fail:

1. **The four declared metrics** on a held-out state set, against the shipped 256 px render, the
   bilinear and Lanczos baselines at the same render resolution, and JPEG q95 of the official frame.
   Nothing here is scored against the corrector's own input.
2. **Throughput end to end**, render plus network, at batch 1024, with the network replayed from a
   CUDA graph the way it would ship. A corrector that doubles fidelity and halves speed is reported
   as exactly that.
3. **Dynamic versus static pixels.** Most of a fixed-camera scene never moves, so a network that
   memorised the room scores well on the frame mean and learns nothing. The mean over the quarter of
   pixels whose official value varies most across the evaluation states is the number that cannot be
   won by memorisation.
4. **The error TAILS**: max |diff| and the 99.9th percentile. A hallucinated object is a small number
   of pixels that are completely wrong, which a mean cannot see.
5. **Out of distribution**: the same checkpoint on a camera it never saw, and on a DIFFERENT KITCHEN.
   If it only works where it was trained, that is a limitation and it has to be stated in the same
   table as the win.
6. **Two hallucination controls.**
   * *Blank input.* Feed the network a constant grey frame at its input resolution and look at what
     it draws. A corrector that reads its input outputs nearly nothing; a corrector that memorised
     the room paints the room, and the residual's structure says which.
   * *Foreign input.* Feed it a render of a different kitchen and measure how much of the residual it
     produces is the SAME residual it produces on its own kitchen -- a correlation, per pixel, over
     states. High correlation is a screen-space overlay wearing a network's clothes.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

import aux as AX
import data as D
import nets
from probe_budget_frontier import (ctx_cache, env_module, family_of, jpeg_roundtrip,
                                   metrics, patched, upsample)
from probe_speed import FXAA, BilinearOnly, Lanczos
from train import _forward_u8

OUT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet"


def load_model(path: str, dtype=torch.float16, table=None):
    """Rebuild the corrector from the checkpoint's OWN aux spec, never from a flag.

    ``table`` overrides the descriptor table, which is how a corrector is pointed at a scene it was
    not trained on: the descriptor columns are defined for any compiled model, so swapping the table
    is the transfer experiment. A LEARNED embedding refuses the swap (``GeomTable.set_table``),
    because two scenes' embedding rows have no correspondence.
    """
    ck = torch.load(path, map_location="cuda", weights_only=False)
    spec = ck.get("aux_spec")
    if spec is not None and spec.get("geom") == "descriptor":
        spec = dict(spec, table=D.geom_table(ck["env"]) if table is None else table)
    m = nets.build(ck["net"], aux=spec).cuda()
    m.load_state_dict(ck["state_dict"])
    if table is not None and m.geom is not None:
        m.geom.set_table(torch.as_tensor(np.asarray(table, np.float32)).cuda())
    return m.to(dtype).eval().to(memory_format=torch.channels_last), ck


def tails(pred: np.ndarray, official: np.ndarray, dyn: np.ndarray) -> dict:
    d = np.abs(pred.astype(np.int16) - official.astype(np.int16)).max(-1)
    flat = d.reshape(-1)
    return {"max_abs": int(d.max()),
            "p99": float(np.percentile(flat, 99)),
            "p999": float(np.percentile(flat, 99.9)),
            "p9999": float(np.percentile(flat, 99.99)),
            "mean_abs_dynamic": float(d[:, dyn].mean()),
            "mean_abs_static": float(d[:, ~dyn].mean()),
            "frac_over_64": float((flat > 64).mean())}


def full_metrics(pred: np.ndarray, official: np.ndarray, dyn: np.ndarray, block: int = 64) -> dict:
    """The repo's four metrics plus the tails, computed in BLOCKS OF 64 FRAMES and averaged.

    Blocking is not an optimisation, it is what makes the number comparable to a declaration.
    ``testing.conformance.ssim`` runs ``scipy.ndimage.uniform_filter(x, size=7)`` over the whole
    ``(N, H, W)`` stack, and a 3-D uniform filter smooths along the **frame axis** as well as the two
    spatial ones -- so SSIM measured on 512 frames is a different quantity from SSIM measured on the
    64 that every ``visual_fidelity`` entry in this repository was measured on. The other three
    metrics are per-pixel means and are unaffected by blocking; this makes all four agree with how
    they are declared.
    """
    n = max(1, len(pred) // block) * block
    rows = [dict(metrics(pred[i:i + block], official[i:i + block]),
                 **tails(pred[i:i + block], official[i:i + block], dyn))
            for i in range(0, n, block)]
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if r[k] is not None]
        if not vals:
            out[k] = None
        elif k == "max_abs":
            out[k] = int(max(vals))
        else:
            out[k] = float(np.mean(vals))
    return out


@torch.no_grad()
def apply_model(model, lr_u8: np.ndarray, shrink: float, batch: int = 256,
                depth=None, ids=None) -> np.ndarray:
    if model.aux_dim() and depth is None:
        raise ValueError("this corrector needs aux planes and the split has none")
    outs = []
    for i in range(0, len(lr_u8), batch):
        sl = slice(i, i + batch)
        x = torch.as_tensor(lr_u8[sl], device="cuda")
        d = None if depth is None else torch.as_tensor(depth[sl], device="cuda")
        g = None if ids is None else torch.as_tensor(ids[sl], device="cuda")
        outs.append(_forward_u8(model, x, shrink, d, g).cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def apply_module(mod, lr_u8: np.ndarray, batch: int = 256) -> np.ndarray:
    outs = []
    for i in range(0, len(lr_u8), batch):
        outs.append(mod(torch.as_tensor(lr_u8[i:i + batch], device="cuda")).cpu().numpy())
    return np.concatenate(outs)


def shipped_frames(env_name, states, size=256, chunk=512, style=None) -> np.ndarray:
    """The 256 px render the tier ships today, for the same states AND the same appearance.

    ``style`` is not optional in spirit. On a split generated with the env's declared camera and
    light variation, rendering the shipped configuration at the DEFAULT camera compares a frame of
    one scene against an official frame of another, and it reports mean 41.6 / SSIM 0.22 -- a number
    that says nothing about the renderer and everything about the harness.
    """
    import stable_envs as se
    mod = env_module(env_name)
    ctx_cache(mod).clear()
    torch.cuda.empty_cache()
    env = se.make(env_name, batch=1, size=size, device="cuda", seed=0)
    out = np.empty((len(states), size, size, 3), np.uint8)
    for i in range(0, len(states), chunk):
        j = min(i + chunk, len(states))
        out[i:j] = env.render(states[i:j], size=size,
                              style=None if style is None else style[i:j]).cpu().numpy()
    del env
    ctx_cache(mod).clear()
    torch.cuda.empty_cache()
    return out


def throughput(env_name, model, render_size, tex_lod, batch=1024, reps=5, shrink=0.0,
               aux=False, depth_scale=8.0):
    """End-to-end f/s: MJWarp at ``render_size`` into a buffer, then the network from a CUDA graph.

    The graph is how this would ship -- the network's shape is fixed, so its launches are captured
    once -- and it is also the only way the corrector's cost is measured rather than Python's. The
    render is NOT in the graph (it is a warp graph replay of its own); the two are timed together.

    With ``aux`` the timed region additionally contains **the two extraction kernels**
    (``get_depth``, ``get_segmentation``) and the uint8/int32 conversions, because a consumer of an
    aux corrector pays for those on every frame. Charging only the megakernel's own 0.4-2.3 % would
    understate the cost of the thing being recommended.
    """
    import stable_envs as se
    mod = env_module(env_name)
    if aux:
        orig = AX.patch_context(mod, tex_lod)
    else:
        orig = patched(mod, dict(enable_backface_culling=False), tex_lod)
    try:
        env = se.make(env_name, batch=1, size=256, device="cuda", seed=0)
        st = torch.as_tensor(env.sample_states(64), device="cuda", dtype=torch.float32)
        idx = torch.as_tensor(np.random.default_rng(0).integers(0, 64, batch), device="cuda")
        st = st[idx]
        low = torch.empty((batch, render_size, render_size, 3), dtype=torch.uint8, device="cuda")
        env.render(st, size=render_size, out=low)
        torch.cuda.synchronize()

        chunk = int(mod.MJW_CHUNK)
        dep = seg = None
        cam = None
        if aux:
            cls = se.REGISTRY[env_name]
            fam = family_of(env_name)
            dep = torch.empty((batch, render_size, render_size), dtype=torch.uint8, device="cuda")
            seg = torch.empty((batch, render_size, render_size), dtype=torch.int32, device="cuda")

            def render_and_read():
                for i in range(0, batch, chunk):
                    j = min(i + chunk, batch)
                    env.render(st[i:j], size=render_size, out=low[i:j])
                    n = j - i
                    nw = 1 if n <= 1 else min(chunk, 1 << (n - 1).bit_length())
                    c = aux_context(mod, fam, cls, render_size, nw)
                    d, g = AX.read_aux(c, n, render_size, 0, depth_scale)
                    dep[i:j] = d
                    seg[i:j] = g
            render_and_read()
        else:
            def render_and_read():
                env.render(st, size=render_size, out=low)
        torch.cuda.synchronize()

        graph = None
        if model is not None:
            with torch.no_grad():
                for _ in range(3):
                    _forward_u8(model, low, shrink, dep, seg)
                torch.cuda.synchronize()
                sstream = torch.cuda.Stream()
                sstream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(sstream):
                    for _ in range(3):
                        _forward_u8(model, low, shrink, dep, seg)
                torch.cuda.current_stream().wait_stream(sstream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _forward_u8(model, low, shrink, dep, seg)

        def step():
            render_and_read()
            if graph is not None:
                graph.replay()
            else:
                upsample(low, 256, "bilinear")

        step()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        dt = float(np.median(ts))
        del low, st, graph, env, dep, seg
    finally:
        mod.MW.create_render_context = orig
        ctx_cache(mod).clear()
        torch.cuda.empty_cache()
    return {"fps": batch / dt, "seconds_per_batch": dt, "batch": batch, "aux": aux}


def aux_context(mod, family, cls, res, nworld):
    """The cached MJWarp context the env just rendered through. Single-camera, so index 0.

    The render profile has to be the class's OWN, not this module's, because it is part of the cache
    key: asking for a different profile builds a SECOND context and reads depth out of a buffer no
    frame was ever rendered into.
    """
    if family == "RoboCasa":
        return mod._context(cls.KITCHEN, cls.CAMERA, res, nworld,
                            int(getattr(cls, "TEX_LOD", 1)), getattr(cls, "BACKFACE_CULL", None))
    return mod._context(cls.ROBOT, cls.TASK, res, mod.MJW_CHUNK)


def shipped_throughput(env_name, batch=1024, reps=5):
    import stable_envs as se
    mod = env_module(env_name)
    ctx_cache(mod).clear()
    env = se.make(env_name, batch=1, size=256, device="cuda", seed=0)
    st = torch.as_tensor(env.sample_states(64), device="cuda", dtype=torch.float32)
    idx = torch.as_tensor(np.random.default_rng(0).integers(0, 64, batch), device="cuda")
    st = st[idx]
    out = torch.empty((batch, 256, 256, 3), dtype=torch.uint8, device="cuda")
    env.render(st, size=256, out=out)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        env.render(st, size=256, out=out)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    del out, st, env
    ctx_cache(mod).clear()
    torch.cuda.empty_cache()
    return batch / float(np.median(ts))


# ------------------------------------------------------------------------- hallucination controls

@torch.no_grad()
def blank_probe(model, render_size: int, shrink: float = 0.0) -> dict:
    """What does the network draw when its input carries no scene at all?

    Three constant frames (grey, black, white). The residual is the whole output minus the bilinear
    upsample of that constant, which for a constant input is the constant itself -- so any structure
    in the difference was produced from the weights and not from the input. Reported as the residual's
    standard deviation over pixels, and as the fraction of pixels it moves by more than 8/255.
    """
    out = {}
    # An aux corrector gets a blank scene as well as a blank picture: zero depth and geom id 0,
    # which is what the segmentation writes for a ray that hit nothing. Handing it the real scene's
    # id map would be asking a different question -- "what do you draw from geometry alone" -- and
    # that question is answered by the foreign-input control instead.
    zd = torch.zeros((1, render_size, render_size), dtype=torch.uint8, device="cuda")
    zi = torch.zeros((1, render_size, render_size), dtype=torch.int32, device="cuda")
    for name, val in (("grey", 128), ("black", 0), ("white", 255)):
        x = torch.full((1, render_size, render_size, 3), val, dtype=torch.uint8, device="cuda")
        y = _forward_u8(model, x, shrink, zd if model.use_depth else None,
                        zi if model.geom is not None else None).float()
        res = (y - float(val))[0]
        out[name] = {"residual_mean_abs": float(res.abs().mean()),
                     "residual_std": float(res.std()),
                     "residual_max_abs": float(res.abs().max()),
                     "frac_moved_over_8": float((res.abs().amax(-1) > 8).float().mean())}
        out[name + "_map"] = res.abs().amax(-1).cpu().numpy()
    return out


def overlay_correlation(res_home: np.ndarray, res_foreign: np.ndarray) -> dict:
    """How much of the residual is the same picture regardless of what the network was shown?

    ``res_home`` and ``res_foreign`` are ``(N, H, W)`` residual magnitudes on two different scenes.
    A corrector that reads its input produces uncorrelated residuals on unrelated scenes; a
    screen-space overlay produces the same map twice. Pearson correlation between the two MEAN maps,
    which is the quantity an overlay would drive to 1.
    """
    a = res_home.mean(0).ravel().astype(np.float64)
    b = res_foreign.mean(0).ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return {"mean_map_pearson": float((a * b).sum() / den) if den > 0 else None,
            "home_mean_residual": float(res_home.mean()),
            "foreign_mean_residual": float(res_foreign.mean())}


def residual_maps(model, lr: np.ndarray, shrink: float, batch: int = 256,
                  depth=None, ids=None) -> np.ndarray:
    """``(N, 256, 256)`` |output - bilinear(input)|, the thing the corrector actually added."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(lr), batch):
            sl = slice(i, i + batch)
            x = torch.as_tensor(lr[sl], device="cuda")
            d = None if depth is None else torch.as_tensor(depth[sl], device="cuda")
            g = None if ids is None else torch.as_tensor(ids[sl], device="cuda")
            y = _forward_u8(model, x, shrink, d, g).float()
            b = upsample(x, 256, "bilinear").float()
            outs.append((y - b).abs().amax(-1).cpu().numpy())
    return np.concatenate(outs)


# ------------------------------------------------------------------------------------ side by side

def label(img: np.ndarray, text: str, pad: int = 18) -> np.ndarray:
    from PIL import Image, ImageDraw
    h, w = img.shape[:2]
    out = np.zeros((h + pad, w, 3), np.uint8)
    out[pad:] = img
    im = Image.fromarray(out)
    ImageDraw.Draw(im).text((3, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


def sbs(path: str, official, cols, amplify: int = 8):
    from PIL import Image
    top, bot = [], []
    for name, img in [("official (mujoco.Renderer)", official)] + list(cols):
        d = np.abs(img.astype(np.int16) - official.astype(np.int16)).max(-1)
        top.append(label(img, name))
        bot.append(label(np.repeat(np.clip(d * amplify, 0, 255).astype(np.uint8)[..., None], 3, 2),
                         f"|diff| x{amplify}   mean {d.mean():.2f}/255   max {d.max()}"))
    grid = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
    Image.fromarray(grid).save(path)
    print("wrote", path, grid.shape, flush=True)


# ------------------------------------------------------------------------------------------ driver

def eval_split(env_name, model, ck, split, tag, shrink, want_shipped=True, want_jpeg=True,
               dyn_frac=0.25, sbs_state=None, sbs_path=None, limit=256):
    # ``limit`` caps the frames the EXPENSIVE metrics see: ``interior_mask`` is a per-frame scipy
    # pass and SSIM is five separable 3-D filters, so the battery is minutes per variant at 2,048
    # frames. 256 is four blocks of 64, i.e. exactly the sample a ``visual_fidelity`` declaration is
    # measured on, four times over.
    lr, hr, states, sty, meta, dep, seg = D.load_split(env_name, split, tag, limit)
    dyn = D.dynamic_mask(hr, dyn_frac)
    t0 = time.perf_counter()
    row = {"env": env_name, "split": split, "tag": tag, "n": len(lr),
           "dynamic_frac": float(dyn.mean()), "shrink": shrink, "variants": {}}

    ours = apply_model(model, lr, shrink, depth=dep, ids=seg)
    row["variants"]["corrector"] = full_metrics(ours, hr, dyn)
    row["has_aux"] = dep is not None
    print(f"    [{split}] corrector metrics in {time.perf_counter() - t0:.0f}s", flush=True)

    bil = apply_module(BilinearOnly(256).cuda(), lr)
    row["variants"]["bilinear"] = full_metrics(bil, hr, dyn)
    lan = apply_module(Lanczos(lr.shape[1], 256).cuda(), lr)
    row["variants"]["lanczos3"] = full_metrics(lan, hr, dyn)
    del lan
    fx = apply_module(FXAA(256).cuda(), lr)
    row["variants"]["fxaa"] = full_metrics(fx, hr, dyn)
    del fx

    ship = None
    if want_shipped:
        ship = shipped_frames(env_name, states, style=sty)
        row["variants"]["shipped_256"] = full_metrics(ship, hr, dyn)
    if want_jpeg:
        jo, nb = jpeg_roundtrip(hr, 95)
        row["variants"]["jpeg_q95_of_official"] = full_metrics(jo, hr, dyn)
        row["variants"]["jpeg_q95_of_official"]["bytes_per_frame"] = nb
        del jo
    print(f"    [{split}] all variants in {time.perf_counter() - t0:.0f}s", flush=True)

    if sbs_path is not None:
        k = 0 if sbs_state is None else sbs_state
        cols = []
        if ship is not None:
            cols.append((f"shipped 256px MJWarp", ship[k]))
        cols += [(f"bilinear {lr.shape[1]}->256", bil[k]),
                 (f"CORRECTOR {ck['net']}", ours[k])]
        jo1, _ = jpeg_roundtrip(hr[k:k + 1], 95)
        cols.append(("JPEG q95 of official", jo1[0]))
        sbs(sbs_path, hr[k], cols)
    return row, ours, bil, ship


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--env", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--splits", default="heldout,decl")
    ap.add_argument("--ood", default="",
                    help="comma-separated ``Env`` or ``Env=tag`` to evaluate zero-shot")
    ap.add_argument("--shrink", type=float, default=0.0)
    ap.add_argument("--shrink-sweep", default="0,0.5,1.0,2.0")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--encoders", action="store_true")
    ap.add_argument("--enc-states", type=int, default=64)
    ap.add_argument("--out", default="")
    ap.add_argument("--outdir", default=OUT, help="where the side-by-side and blank images go")
    # A checkpoint fitted on SEVERAL cells is evaluated once per cell, and every one of those runs
    # has the same checkpoint stem and the same tag -- so the pictures overwrite each other and the
    # last cell's frames end up filed under a name that says nothing about which cell they are. The
    # stem is overridable for exactly that case.
    ap.add_argument("--stem", default="", help="image-name stem; default is the checkpoint's")
    a = ap.parse_args()
    OUT = a.outdir
    os.makedirs(OUT, exist_ok=True)

    model, ck = load_model(a.ckpt)
    use_aux = model.aux_dim() > 0
    env_name = a.env or ck["env"]
    tag = a.tag or ck["tag"]
    rsz = nets.render_size(ck["net"])
    tex_lod = next((int(p[3:]) for p in tag.split("_") if p.startswith("tex")), 1)
    res = {"ckpt": a.ckpt, "net": ck["net"], "aux": ck.get("aux", "none"),
           "aux_spec": ck.get("aux_spec"), "env": env_name, "tag": tag, "render_size": rsz,
           "tex_lod": tex_lod, "params": nets.n_params(model),
           "gflops_per_frame": nets.gflops_per_frame(ck["net"]),
           "gpu": torch.cuda.get_device_name(0),
           "train_meta": ck.get("train_meta"), "heldout_meta": ck.get("heldout_meta"),
           "splits": [], "ood": [], "throughput": {}, "hallucination": {}}

    # ---- throughput, first, before any big tensor is allocated -------------------------------
    res["throughput"]["shipped_256_fps"] = shipped_throughput(env_name, a.batch)
    res["throughput"]["render_only_fps"] = throughput(env_name, None, rsz, tex_lod, a.batch)["fps"]
    if use_aux:
        res["throughput"]["render_plus_aux_fps"] = throughput(
            env_name, None, rsz, tex_lod, a.batch, aux=True)["fps"]
    res["throughput"]["render_plus_corrector"] = throughput(env_name, model, rsz, tex_lod, a.batch,
                                                            shrink=a.shrink, aux=use_aux)
    print(json.dumps(res["throughput"]), flush=True)

    # ---- fidelity on each split --------------------------------------------------------------
    home_res = None
    # The image name carries the CHECKPOINT stem and the data tag, not the architecture. Two
    # checkpoints of one architecture -- trained at the default appearance and trained under the
    # env's declared variation -- are the comparison this study rests on, and naming their pictures
    # after the architecture made the second overwrite the first.
    stem = a.stem or os.path.basename(a.ckpt)[:-3]
    for split in a.splits.split(","):
        p = os.path.join(OUT, f"sbs_{stem}__{tag}_{split}.png")
        row, ours, bil, ship = eval_split(env_name, model, ck, split, tag, a.shrink,
                                          sbs_path=p)
        res["splits"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "variants"}), flush=True)
        for k, v in row["variants"].items():
            print(f"  {k:26s} mean {v['mean_abs']:7.3f}  dyn {v['mean_abs_dynamic']:7.3f}  "
                  f"stat {v['mean_abs_static']:7.3f}  w8 {v['within8_frac']:.3f}  "
                  f"ssim {v['ssim']:.4f}  iexact {str(v['interior_exact_frac'])[:6]}  "
                  f"max {v['max_abs']}  p99.9 {v['p999']:.1f}", flush=True)
        if split == "decl":
            # The ``decl`` split IS the draw a ``visual_fidelity`` declaration is measured on: 64
            # states x 4 independent seeds, in blocks of 64. The repo declares the WORST of the four,
            # so the four are reported separately here rather than averaged into one number that no
            # conformance test would ever compute.
            lr, hr, dstates, dsty, _, ddep, dseg = D.load_split(env_name, split, tag)
            dship = shipped_frames(env_name, dstates, style=dsty)
            per = []
            for k in range(len(lr) // 64):
                sl = slice(64 * k, 64 * (k + 1))
                dk = D.dynamic_mask(hr[sl])
                per.append({"draw": k,
                            "corrector": full_metrics(
                                apply_model(model, lr[sl], a.shrink,
                                            depth=None if ddep is None else ddep[sl],
                                            ids=None if dseg is None else dseg[sl]), hr[sl], dk),
                            "shipped_256": full_metrics(dship[sl], hr[sl], dk),
                            "jpeg_q95": full_metrics(jpeg_roundtrip(hr[sl], 95)[0], hr[sl], dk)})
            row["per_draw"] = per
            row["worst_of_draws"] = {
                w: {"mean_abs": round(max(d[w]["mean_abs"] for d in per), 2),
                    "within8_frac": round(min(d[w]["within8_frac"] for d in per), 3),
                    "ssim": round(min(d[w]["ssim"] for d in per), 3),
                    "interior_mean_abs": round(max(d[w]["interior_mean_abs"] for d in per), 2),
                    "interior_exact_frac": round(min(d[w]["interior_exact_frac"] for d in per), 3)}
                for w in ("corrector", "shipped_256", "jpeg_q95")}
            print("worst of 4 draws: " + json.dumps(row["worst_of_draws"]), flush=True)
        if split == "heldout":
            lr, hr, _, _, _, dp, sg = D.load_split(env_name, split, tag, 256)
            home_res = residual_maps(model, lr, a.shrink, depth=dp, ids=sg)
        del ours, bil, ship
        torch.cuda.empty_cache()

    # ---- shrink sweep: what interior-exactness costs -----------------------------------------
    lr, hr, _, _, _, dp, sg = D.load_split(env_name, "heldout", tag, 256)
    dyn = D.dynamic_mask(hr)
    res["shrink_sweep"] = []
    for s in [float(x) for x in a.shrink_sweep.split(",")]:
        m = full_metrics(apply_model(model, lr, s, depth=dp, ids=sg), hr, dyn)
        m["shrink"] = s
        res["shrink_sweep"].append(m)
        print(f"shrink {s:4.1f}  mean {m['mean_abs']:7.3f}  iexact {m['interior_exact_frac']}  "
              f"ssim {m['ssim']:.4f}", flush=True)

    # ---- out of distribution ------------------------------------------------------------------
    for spec in [x for x in a.ood.split(",") if x]:
        other, _, otag = spec.partition("=")
        otag = otag or tag
        try:
            swapped = False
            if model.geom is not None and not model.geom.learned:
                t = D.geom_table(other)
                if t is None:
                    raise FileNotFoundError(f"no geom_features.npy for {other}; run gen_aux.py")
                model.geom.set_table(torch.as_tensor(t).cuda())
                swapped = True
            row, ours, bil, ship = eval_split(
                other, model, ck, "heldout", otag, a.shrink,
                sbs_path=os.path.join(OUT, f"sbs_{stem}__OOD_{other}.png"))
            row["ood_of"] = env_name
            row["geom_table_swapped_to_target_scene"] = swapped
            res["ood"].append(row)
            print(f"OOD {other}", flush=True)
            for k, v in row["variants"].items():
                print(f"  {k:26s} mean {v['mean_abs']:7.3f}  dyn {v['mean_abs_dynamic']:7.3f}  "
                      f"w8 {v['within8_frac']:.3f}  ssim {v['ssim']:.4f}  max {v['max_abs']}",
                      flush=True)
            if home_res is not None:
                ol, _, _, _, _, od, og = D.load_split(other, "heldout", otag, 256)
                fr = residual_maps(model, ol, a.shrink, depth=od, ids=og)
                res["hallucination"][f"overlay_vs_{other}"] = overlay_correlation(home_res, fr)
                print(json.dumps(res["hallucination"][f"overlay_vs_{other}"]), flush=True)
            del ours, bil, ship
        except Exception as e:                                    # noqa: BLE001
            res["ood"].append({"env": other, "error": f"{type(e).__name__}: {e}"})
            print(json.dumps(res["ood"][-1]), flush=True)
        finally:
            if model.geom is not None and not model.geom.learned:
                model.geom.set_table(torch.as_tensor(D.geom_table(env_name)).cuda())
        torch.cuda.empty_cache()

    # ---- blank-input control -----------------------------------------------------------------
    bp = blank_probe(model, rsz, a.shrink)
    maps = {k: bp.pop(k) for k in list(bp) if k.endswith("_map")}
    res["hallucination"]["blank_input"] = bp
    print("blank input:", json.dumps(bp), flush=True)
    from PIL import Image
    g = np.concatenate([np.clip(maps[k] * 8, 0, 255).astype(np.uint8) for k in sorted(maps)], 1)
    Image.fromarray(np.repeat(g[..., None], 3, 2)).save(
        os.path.join(OUT, f"blank_{stem}.png"))

    # ---- frozen-encoder distance, the repo's own proxy ---------------------------------------
    if a.encoders:
        res["encoders"] = encoder_distances(env_name, model, tag, a.shrink, a.enc_states)
        print(json.dumps(res["encoders"], indent=1), flush=True)

    p = a.out or os.path.join(OUT, f"eval_{env_name}_{ck['net']}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("wrote", p, flush=True)


def encoder_distances(env_name, model, tag, shrink, n):
    """``probe_repr``'s test with the corrector added as a variant. Same controls, same scale."""
    import timm

    from probe_repr import ENCODERS, embed
    lr, hr, states, sty, _, dp, sg = D.load_split(env_name, "heldout", tag, n)
    ship = shipped_frames(env_name, states, style=sty)
    variants = {"jpeg_q95": jpeg_roundtrip(hr, 95)[0],
                "blank": np.full_like(hr, 128),
                "shipped_256": ship,
                "bilinear_lr": apply_module(BilinearOnly(256).cuda(), lr),
                "corrector": apply_model(model, lr, shrink, depth=dp, ids=sg)}
    out = {}
    for enc in ENCODERS:
        try:
            m = timm.create_model(enc, pretrained=True, num_classes=0).eval().cuda()
            cfg = timm.data.resolve_data_config({}, model=m)
            z0 = embed(m, cfg, hr)
            perm = torch.randperm(len(z0))
            scale = float((1.0 - (z0 * z0[perm]).sum(-1)).mean())
            row = {"state_to_state_cosine_dist": scale}
            for k, v in variants.items():
                d = float((1.0 - (z0 * embed(m, cfg, v)).sum(-1)).mean())
                row[k] = {"cosine_dist": d, "frac_of_state_change": d / scale if scale else None}
            out[enc] = row
            del m
        except Exception as e:                                    # noqa: BLE001
            out[enc] = {"error": f"{type(e).__name__}: {e}"}
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    main()
