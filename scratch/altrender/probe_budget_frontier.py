"""The JPEG budget, and the speed frontier inside it.

The comparison this repo actually makes is not "our frame against MuJoCo's". It is "our frame against
what a stored dataset hands a training loop", and a stored dataset is **JPEG q95 in a Lance table** --
which is not MuJoCo's frame either. So the first thing measured here is the error the storage baseline
introduces ON ITS OWN, in exactly the four units a visual-tier class declares, against the same
official frames our renderer is scored against. That number is the budget. Everything after it is
spending the budget on speed.

Both halves are measured through the ENV's own surface, so the chunking, the graph capture, the
rounding patch and the channel swap are the shipped ones and the only thing that varies is the
configuration under test:

* the JPEG path is the repo's own -- ``PIL.Image.save(format="JPEG", quality=95)`` from
  ``examples/bench_storage.py`` and ``torchvision.io.decode_jpeg`` from ``examples/bench_lance.py``,
  so what is scored is the byte stream a worker process really decodes, chroma subsampling and all;
* the render path is ``env.render(states, size=r)`` followed by an upsample to 256, and **the upsample
  is inside the timed region**, because a consumer asked for 256 px and must be given 256 px.

Four rows exist for each configuration and they answer different questions. ``ours`` is what the class
would declare. ``ours+jpeg`` is what a dataset BUILT from this renderer would deliver, which is the
honest comparison against ``jpeg_official`` -- storing our frames does not make the JPEG error go away,
it adds it.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
import torch.nn.functional as F


def jpeg_roundtrip(frames: np.ndarray, quality: int = 95) -> np.ndarray:
    """``PIL`` encode / ``torchvision`` decode -- the repo's own write and read paths, unchanged."""
    from PIL import Image
    from torchvision.io import ImageReadMode, decode_jpeg
    blobs = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=quality)
        blobs.append(buf.getvalue())
    ts = [torch.frombuffer(b, dtype=torch.uint8) for b in blobs]
    out = torch.stack(decode_jpeg(ts, mode=ImageReadMode.RGB))
    return out.permute(0, 2, 3, 1).contiguous().numpy(), sum(len(b) for b in blobs) / len(blobs)


_TEX_KEEP: list = []


def context_with_tex_lod(m, lod: int, half: bool, **kw):
    """``create_render_context`` with every texture reduced by ``lod`` before it reaches the GPU.

    A RoboCasa kitchen ships **356 MB of texture data** and ``mujoco_warp`` uploads all of it at full
    resolution into float32 ``(h, w, 4)`` arrays -- 16 bytes a texel, so ~475 MB resident and sampled
    by a megakernel whose rays hit it in whatever order the BVH hands them. The texture ABLATION says
    sampling is only 5.4% of the frame, but that removes the arithmetic and leaves the working set;
    this removes the working set. Box-filtered by ``area``, which for an integer factor is the mip
    level a GL pipeline would have picked anyway -- and MJWarp's known minification gap
    (``render/mjwarp.py``, "the second gap") means the base level was the wrong level to sample at
    grazing incidence in the first place.
    """
    import warp as wp

    import mujoco_warp as mjw
    from mujoco_warp._src import render_util as ru

    original = ru.create_warp_texture
    blob: dict = {}

    def lod_texture(model, tex_id: int):
        if "data" not in blob:
            blob["data"] = wp.array(model.tex_data, dtype=wp.uint8)
        w, h = int(model.tex_width[tex_id]), int(model.tex_height[tex_id])
        out = wp.zeros((h, w, 4), dtype=float)
        wp.launch(ru._convert_texture_data, dim=(w, h),
                  inputs=[w, int(model.tex_adr[tex_id]), int(model.tex_nchannel[tex_id]),
                          blob["data"]], outputs=[out])
        if lod > 1 or half:
            t = wp.to_torch(out)
            if lod > 1:
                hh, ww = max(1, h // lod), max(1, w // lod)
                t = F.interpolate(t.permute(2, 0, 1)[None], size=(hh, ww),
                                  mode="area")[0].permute(1, 2, 0)
            t = (t.to(torch.float16) if half else t).contiguous()
            _TEX_KEEP.append(t)                       # wp.from_torch does not take ownership
            out = wp.from_torch(t)
        return wp.Texture2D(out, filter_mode=wp.TextureFilterMode.LINEAR)

    ru.create_warp_texture = lod_texture
    try:
        return mjw.create_render_context(m, **kw)
    finally:
        ru.create_warp_texture = original


def upsample(x: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    """``(B,r,r,3)`` uint8 -> ``(B,size,size,3)`` uint8. A no-op when already at ``size``."""
    if x.shape[1] == size:
        return x
    y = x.permute(0, 3, 1, 2).to(torch.float32)
    kw = {} if mode == "nearest" else {"align_corners": False}
    y = F.interpolate(y, size=(size, size), mode=mode, **kw)
    return y.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def metrics(ours: np.ndarray, official: np.ndarray) -> dict:
    from stable_envs.testing.conformance import pixel_agreement, ssim
    out = pixel_agreement(ours, official)
    out["ssim"] = ssim(ours, official)
    return {k: out[k] for k in ("mean_abs", "within8_frac", "ssim", "interior_exact_frac",
                                "exact_frac", "interior_mean_abs", "max_abs")}


#: label -> (internal render size, upsample kernel, create_render_context overrides, texture lod)
def config_list(family: str) -> dict:
    FREE = dict(enable_backface_culling=False)        # 0 pixels changed, measured on all four cells
    cfg = {"shipped": (256, "bilinear", {}, 1),
           "nobfc": (256, "bilinear", FREE, 1)}
    for r in (224, 192, 176, 160, 144, 128, 112, 96):
        cfg[f"r{r}"] = (r, "bilinear", FREE, 1)
    for m in ("nearest", "bicubic"):
        for r in (192, 160, 128):
            cfg[f"r{r}_{m}"] = (r, m, FREE, 1)
    for lod in (2, 4):
        cfg[f"tex{lod}"] = (256, "bilinear", FREE, lod)
        for r in (192, 176, 160, 144, 128, 112):
            cfg[f"r{r}_tex{lod}"] = (r, "bilinear", FREE, lod)
    cfg["nospec"] = (256, "bilinear", dict(FREE, enable_specular=False), 1)
    if family == "DMControl":
        # These five already clear their stored-frame bar by 1.39-5.39x, so the resolution ladder is
        # swept to LOCATE them on the frontier, not to recommend. The lever that matters here is
        # TEXTURE LOD, and it is aimed at a mechanism the classes already declare: the residual on
        # every one of the five is dominated by the skybox (skybox_share 0.83-0.89 on cheetah and
        # hopper) or the checker floor (floor_share 0.96-0.98 on humanoid and swimmer), and
        # ``VISUAL_NOTE`` names them as "the unfiltered speckled skybox and the minified checker
        # floor". That IS mujoco_warp's num_mip_levels=1 minification gap, so a box-filtered texture
        # should move these cells TOWARD the reference rather than away from it.
        for lod in (8, 16):
            cfg[f"tex{lod}"] = (256, "bilinear", FREE, lod)
        for r in (256, 224, 192, 160):
            cfg[f"r{r}_noshadow"] = (r, "bilinear", dict(FREE, use_shadows=False), 1)
        cfg["noskybox"] = (256, "bilinear", dict(FREE, render_skybox=False), 1)
        # ``render_size = 0`` means "the env's own size", so a declaration can be measured at the
        # size the declaration is asserted at (84 px, see dmcontrol._AT) rather than at 256 and
        # downsampled -- which would be measuring a different picture from the one that ships.
        cfg["nospec_native"] = (0, "bilinear", dict(enable_specular=False), 1)
        cfg["nobfc_native"] = (0, "bilinear", FREE, 1)
        cfg["shipped_native"] = (0, "bilinear", {}, 1)
    elif family == "HumanoidBench":
        for r in (256, 224, 192, 176, 160, 144, 128):
            cfg[f"r{r}_noshadow"] = (r, "bilinear", dict(FREE, use_shadows=False), 1)
    else:
        cfg["r128_tex4_nospec"] = (128, "bilinear", dict(FREE, enable_specular=False), 4)
        cfg["r160_tex2_nospec"] = (160, "bilinear", dict(FREE, enable_specular=False), 2)
        cfg["r112_tex2"] = (112, "bilinear", FREE, 2)
    return cfg


def env_module(name):
    import importlib
    if name.startswith("RoboCasa"):
        return importlib.import_module("stable_envs.envs.robocasa")
    if name.startswith("HumanoidBench"):
        return importlib.import_module("stable_envs.envs.humanoidbench")
    if name.startswith("DMC"):
        return importlib.import_module("stable_envs.envs.dmcontrol")
    raise SystemExit(f"unhandled {name}")


def family_of(name: str) -> str:
    return ("RoboCasa" if name.startswith("RoboCasa")
            else "DMControl" if name.startswith("DMC") else "HumanoidBench")


def ctx_cache(mod):
    """The env module's MJWarp-context cache, whatever it calls it.

    ``robocasa`` and ``humanoidbench`` name it ``_CTX_CACHE``; ``dmcontrol`` names it ``_TC_CACHE``
    and keys it by (domain, task, resolution, worlds). Both have to be emptied whenever
    ``create_render_context`` is swapped, or the next render silently reuses a context built under
    the previous configuration — which reads as "the lever did nothing".
    """
    for attr in ("_CTX_CACHE", "_TC_CACHE"):
        c = getattr(mod, attr, None)
        if c is not None:
            return c
    raise SystemExit(f"{mod.__name__} has no MJWarp context cache to clear")


def patched(mod, overrides, lod=1):
    """Force a render profile on the next context this module builds.

    ``stable_envs.render.mjwarp.create_render_context`` now takes ``tex_lod`` itself and the env
    classes pass their own ``TEX_LOD`` through, so the LOD is HANDED ON rather than reimplemented
    here: the box filter this function used to install is now the one in the library, and the two
    produced the same texels. What is still monkeypatched is only the override dict, because a
    caller sweeping ``enable_backface_culling`` has no class attribute to set.

    ``max`` and not assignment: a class that already asks for LOD 2 must not be silently reset to 1
    by a probe that did not think about textures.
    """
    import stable_envs.render.mjwarp as MW
    original = MW.create_render_context

    def wrap(m, **kw):
        kw.update(overrides)
        want = max(int(kw.pop("tex_lod", 1)), int(lod))
        if want > 1:
            kw["tex_lod"] = want
        return original(m, **kw)

    mod.MW.create_render_context = wrap
    ctx_cache(mod).clear()
    return original


def throughput(env, states, batch, r, size, mode, reps=3):
    st = torch.as_tensor(states, device="cuda", dtype=torch.float32)
    idx = np.random.default_rng(0).integers(0, len(states), batch)
    st = st[torch.as_tensor(idx, device="cuda")]
    low = torch.empty((batch, r, r, 3), dtype=torch.uint8, device="cuda")

    def step():
        env.render(st, size=r, out=low)
        return upsample(low, size, mode)

    step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    dt = float(np.median(ts))
    del low, st
    torch.cuda.empty_cache()
    return {"fps": batch / dt, "seconds_per_batch": dt, "batch": batch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--configs", default="")
    ap.add_argument("--budget-only", action="store_true")
    # The DM Control TrackCom cells need dm_control AND mujoco_warp, and the only interpreter that
    # has both has no torchvision -- which is the repo's JPEG READ path and the whole reason the
    # budget number means anything. So the two halves are split across interpreters rather than the
    # decoder being quietly swapped: this flag runs the render frontier without touching JPEG, and
    # ``--dump-official`` hands the official frames to ``jpeg_budget_from_npy.py`` in the
    # interpreter that does have torchvision. Nothing is measured with a substitute codec.
    ap.add_argument("--no-jpeg", action="store_true")
    ap.add_argument("--dump-official", default="")
    a = ap.parse_args()

    import stable_envs as se
    cls = se.REGISTRY[a.env]
    mod = env_module(a.env)
    family = family_of(a.env)

    ref = cls.reference_env(a.size)
    probe = se.make(a.env, batch=1, size=a.size, device="cuda")
    states = probe.sample_states(a.states)
    official = cls.reference_render(ref, states)
    del probe
    ctx_cache(mod).clear()
    torch.cuda.empty_cache()

    res = {"env": a.env, "size": a.size, "batch": a.batch, "n_states": a.states,
           "jpeg_quality": a.quality, "gpu": torch.cuda.get_device_name(0),
           "declared": dict(getattr(cls, "visual_fidelity", {}) or {}), "rows": []}

    if a.dump_official:
        np.save(a.dump_official, official)
        print(f"wrote {a.dump_official} {official.shape}", flush=True)

    # ---- TASK 1: what does the storage baseline itself cost? --------------------------------
    for q in ([] if a.no_jpeg else sorted({a.quality, 90, 100})):
        jo, nbytes = jpeg_roundtrip(official, q)
        row = {"config": f"jpeg_q{q}_of_official", "kind": "storage_baseline",
               "jpeg_bytes_per_frame": nbytes}
        row.update(metrics(jo, official))
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
        del jo
    if a.budget_only:
        _write(res, a)
        return

    # ---- TASK 2: spend it -------------------------------------------------------------------
    cfgs = config_list(family)
    want = a.configs.split(",") if a.configs else list(cfgs)
    for name in want:
        r, mode, ov, lod = cfgs[name]
        orig = patched(mod, ov, lod)
        try:
            env = se.make(a.env, batch=1, size=a.size, device="cuda")
            low = env.render(states, size=r)
            ours = upsample(low, a.size, mode).cpu().numpy()
            del low
            torch.cuda.empty_cache()
            row = {"config": name, "kind": "render", "render_size": r, "upsample": mode,
                   "tex_lod": lod, "overrides": {k: str(v) for k, v in ov.items()}}
            row.update(metrics(ours, official))
            if not a.no_jpeg:
                oj, nbytes = jpeg_roundtrip(ours, a.quality)
                row["with_jpeg"] = metrics(oj, official)
                row["with_jpeg"]["jpeg_bytes_per_frame"] = nbytes
                del oj
            del ours
            torch.cuda.empty_cache()
            row.update(throughput(env, states, a.batch, r, a.size, mode))
            del env
        except Exception as e:                                        # noqa: BLE001
            row = {"config": name, "error": f"{type(e).__name__}: {e}"}
        finally:
            mod.MW.create_render_context = orig
            ctx_cache(mod).clear()
            torch.cuda.empty_cache()
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
    _write(res, a)


def _write(res, a):
    p = (f"/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/"
         f"budget_{a.env}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
