"""Phase attribution for the Craftax tile/atlas path, in the vocabulary section 8H asks for.

``profile_note.py`` gives an operator trace, which says ``aten::mul`` and not "compositing". This
adds the phase timers the brief names — atlas indexing, gather, recolouring, layout, compositing,
allocation, output conversion, and the static/dynamic split — by re-executing the renderer's own
stages one at a time, out of the same texture tables the renderer uses, and timing each.

It is a measurement harness, not a second renderer: the reconstructed pipeline is asserted
pixel-identical to ``env.render`` on the states it profiles before any timing is reported, and the
run aborts if it is not. That assertion is what makes a per-stage number attributable to the real
path rather than to a lookalike.

Both pipelines are timed when both exist:

* **blend** — gather the block texture into a full frame, composite the player into the centre tile,
  composite a full-frame mob layer over everything, apply the channel gain, round. This is the
  pre-Step-6 path and is reconstructible from ``_Assets`` alone, so it can be timed against either
  checkout.
* **atlas** — build the gather index and ``index_select`` the composited-tile table straight into the
  uint8 output. Timed only where the checkout has ``_Composite``.

It also reports the arithmetic that explains the batch curve: the bytes of image-sized float32
temporary each stage touches, per frame and per batch, against this host's last-level cache.

    python scratch/step6/tilemap/attribute.py --device cpu --threads 1 --batch 1 16 64 256 \
        --out examples/results/step6/tilemap/profile_before/
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sync(device: str) -> None:
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def bench(fn, device: str, repeat: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    sync(device)
    xs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        sync(device)
        xs.append((time.perf_counter() - t0) * 1e3)
    return {"ms_median": statistics.median(xs), "ms_min": min(xs), "ms_max": max(xs),
            "ms_stdev": statistics.pstdev(xs) if len(xs) > 1 else 0.0}


def lscpu_cache() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=20).stdout
        return "; ".join(l.strip() for l in out.splitlines() if "cache" in l.lower())
    except Exception:                                              # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--states", type=Path,
                    default=ROOT / "examples" / "results" / "step4" / "states"
                    / "CraftaxClassic.npz")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    import stable_envs  # noqa: F401  (recorded below: which checkout is being measured)
    import stable_envs as se
    from stable_envs.envs.craftax import (N_BLOCK, N_MOB, N_PLAYER, OBS_H, OBS_W, CraftaxClassic,
                                          _Assets)

    torch.set_num_threads(int(a.threads))
    dev = torch.device(a.device)
    t = int(a.size)
    has_atlas = hasattr(CraftaxClassic, "_render_atlas")
    if has_atlas:
        from stable_envs.envs.craftax import _Composite

    note = {
        "what": "craftax tile/atlas phase attribution",
        "stable_envs_file": str(Path(stable_envs.__file__).resolve()),
        "label": a.label, "device": a.device, "size": t, "threads": torch.get_num_threads(),
        "host": socket.gethostname(), "cpu": platform.processor() or platform.machine(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "caches": lscpu_cache(), "torch": torch.__version__,
        "pipeline": "atlas" if has_atlas else "blend",
        "states": str(a.states), "batches": {},
    }

    pool = np.load(a.states)["states"]
    block, ptex, palpha, mtex, malpha = _Assets.get(dev, t)
    f = CraftaxClassic._tiles_to_frame
    frame_px = OBS_H * t * OBS_W * t * 3

    for B in a.batch:
        env = se.make("CraftaxClassic", batch=B, size=t, device=a.device)
        states = np.asarray(pool[np.arange(B) % len(pool)], np.float32)
        s = torch.as_tensor(states, device=dev)
        sty = env.resolve_style(None, None, batch=B)
        gain = sty["channel_gain"].view(B, 1, 1, 3)
        pv = (sty["player_visible"] > 0.5).float().view(B, 1, 1, 1)
        mv = (sty["mobs_visible"] > 0.5).float().view(B, 1, 1, 1)
        nb = OBS_H * OBS_W
        cr, cc = OBS_H // 2, OBS_W // 2

        # ---- decoded class ids, the only genuinely dynamic quantity in the frame
        def decode():
            blk = s[:, :nb].round().remainder(N_BLOCK).long().view(B, OBS_H, OBS_W)
            mob = s[:, nb:2 * nb].round().remainder(N_MOB).long().view(B, OBS_H, OBS_W)
            pid = s[:, 2 * nb].round().remainder(N_PLAYER).long()
            return blk, mob, pid

        blk, mob, pid = decode()

        # ---- the blend pipeline, stage by stage
        def st_gather():
            return block[blk]

        def st_layout():
            return f(block[blk])

        def st_player(px):
            sl = px[:, cr * t:(cr + 1) * t, cc * t:(cc + 1) * t, :]
            al = palpha[pid] * pv
            px[:, cr * t:(cr + 1) * t, cc * t:(cc + 1) * t, :] = sl * (1 - al) + ptex[pid] * al
            return px

        def st_mob_alpha():
            return f(malpha[mob]) * mv

        def st_mob_blend(px, al):
            return px * (1 - al) + f(mtex[mob]) * al

        def st_gain(px):
            return px * gain

        def st_out(px):
            return px.round().clamp_(0, 255).to(torch.uint8)

        def blend_full():
            px = st_player(st_layout())
            al = st_mob_alpha()
            px = st_mob_blend(px, al)
            return st_out(st_gain(px))

        # the reconstruction must BE the renderer, or none of the numbers below attribute anything
        ref = env.render(states)
        got = blend_full()
        assert torch.equal(ref, got), (
            "the reconstructed blend pipeline does not reproduce env.render; every phase number "
            "below would be attributing time to a lookalike")

        rec: dict = {"batch": B, "stages": {}}
        rec["render_official"] = bench(lambda: env.render(states), a.device, a.repeat, a.warmup)
        rec["fps_official"] = 1000.0 * B / rec["render_official"]["ms_median"]
        rec["set_state"] = bench(lambda: env.set_state(states), a.device, a.repeat, a.warmup)

        S = rec["stages"]
        S["decode (atlas indexing)"] = bench(decode, a.device, a.repeat, a.warmup)
        S["blend: tile gather"] = bench(st_gather, a.device, a.repeat, a.warmup)
        S["blend: gather + layout"] = bench(st_layout, a.device, a.repeat, a.warmup)
        base_px = st_layout()
        S["blend: player composite"] = bench(lambda: st_player(base_px), a.device,
                                             a.repeat, a.warmup)
        S["blend: mob alpha layer"] = bench(st_mob_alpha, a.device, a.repeat, a.warmup)
        alpha = st_mob_alpha()
        S["blend: mob composite"] = bench(lambda: st_mob_blend(base_px, alpha), a.device,
                                          a.repeat, a.warmup)
        blended = st_mob_blend(base_px, alpha)
        S["blend: channel gain (recolour)"] = bench(lambda: st_gain(blended), a.device,
                                                    a.repeat, a.warmup)
        gained = st_gain(blended)
        S["blend: output conversion"] = bench(lambda: st_out(gained), a.device, a.repeat, a.warmup)
        S["blend: whole pipeline"] = bench(blend_full, a.device, a.repeat, a.warmup)
        S["allocation: empty uint8 frame"] = bench(
            lambda: torch.empty(B, OBS_H * t, OBS_W * t, 3, dtype=torch.uint8, device=dev),
            a.device, a.repeat, a.warmup)
        S["allocation: empty float frame"] = bench(
            lambda: torch.empty(B, OBS_H * t, OBS_W * t, 3, dtype=torch.float32, device=dev),
            a.device, a.repeat, a.warmup)

        if has_atlas:
            _Composite.get(dev, t, want_float=True)          # bake once, outside the timed region

            def idx_build():
                vis_m = (sty["mobs_visible"] > 0.5).view(B, 1, 1)
                vis_p = (sty["player_visible"] > 0.5).view(B)
                m = torch.where(vis_m, mob, torch.zeros_like(mob))
                slot = torch.full((B, OBS_H, OBS_W), N_PLAYER, dtype=torch.long, device=dev)
                slot[:, cr, cc] = torch.where(vis_p, pid, torch.full_like(pid, N_PLAYER))
                tile = ((slot * N_BLOCK + blk) * N_MOB + m).to(torch.int32)
                return (tile.unsqueeze(2) * t
                        + torch.arange(t, dtype=torch.int32, device=dev).view(1, 1, t, 1)
                        ).reshape(-1)

            rows = idx_build()
            atlas_u8 = _Composite.get(dev, t, want_float=False)[1]
            dst = torch.empty(B, OBS_H * t, OBS_W * t, 3, dtype=torch.uint8, device=dev)
            S["atlas: index build"] = bench(idx_build, a.device, a.repeat, a.warmup)
            S["atlas: gather into uint8 output"] = bench(
                lambda: torch.index_select(atlas_u8, 0, rows, out=dst.view(-1, t * 3)),
                a.device, a.repeat, a.warmup)
            S["atlas: bake (once per device+size)"] = bench(
                lambda: _Composite._build(dev, t), a.device, 3, 1)
            gvals = sty.values.clone()
            off, w = sty.layout["channel_gain"]
            gvals[:, off:off + w] = torch.tensor([0.5, 1.0, 1.5], device=dev)
            S["atlas: gain path (whole render)"] = bench(
                lambda: env.render(states, style=gvals), a.device, a.repeat, a.warmup)

        # what the batch curve is actually made of
        rec["bytes"] = {
            "frame_uint8": frame_px,
            "frame_float32": frame_px * 4,
            "blend_float_temporaries_per_frame": frame_px * 4 * 10,
            "blend_float_temporaries_per_batch": frame_px * 4 * 10 * B,
            "atlas_table_uint8": (N_PLAYER + 1) * N_BLOCK * N_MOB * t * t * 3 if has_atlas else None,
        }
        note["batches"][str(B)] = rec
        print(f"B={B:<5} render {rec['render_official']['ms_median']:9.3f} ms "
              f"({rec['fps_official']:10.1f} fps)  tree={note['pipeline']}")

    a.out.mkdir(parents=True, exist_ok=True)
    host = note["host"].split(".")[0]
    tag = f"attribution_{a.label or note['pipeline']}_{a.device.replace(':', '')}_{host}"
    (a.out / f"{tag}.json").write_text(json.dumps(note, indent=2))
    (a.out / f"{tag}.md").write_text(md(note))
    print("wrote", a.out / f"{tag}.json")


def md(n: dict) -> str:
    L = [f"# Craftax tile/atlas phase attribution — {n['label'] or n['pipeline']}", "",
         f"- tree measured: `{n['stable_envs_file']}`",
         f"- pipeline present in that tree: **{n['pipeline']}**",
         f"- host: {n['host']}  ·  cpu: {n['cpu']}  ·  gpu: {n['gpu'] or 'none'}  ·  "
         f"device: {n['device']}  ·  threads: {n['threads']}",
         f"- slurm job: {n['slurm_job']}  ·  load average: "
         + (f"{n['load_average'][0]:.1f}" if n["load_average"] else "n/a"),
         f"- caches: {n['caches']}", f"- torch {n['torch']}  ·  tile size {n['size']} px", ""]
    for B, rec in n["batches"].items():
        L += [f"## batch {B}", "",
              f"`env.render` **{rec['render_official']['ms_median']:.3f} ms** = "
              f"**{rec['fps_official']:.0f} fps**  ·  set_state "
              f"{rec['set_state']['ms_median']:.3f} ms", "",
              "| phase | ms median | % of render | frames/s if this were all |",
              "|---|---|---|---|"]
        tot = rec["render_official"]["ms_median"]
        for k, v in rec["stages"].items():
            ms = v["ms_median"]
            L.append(f"| {k} | {ms:.3f} | {100 * ms / tot:.1f} | "
                     f"{1000.0 * int(B) / ms if ms else float('inf'):.0f} |")
        b = rec["bytes"]
        L += ["", f"image-sized float32 temporaries in the blend pipeline: "
                  f"{b['blend_float_temporaries_per_frame'] / 1e6:.2f} MB per frame, "
                  f"{b['blend_float_temporaries_per_batch'] / 1e6:.1f} MB per batch, against a "
                  f"{b['frame_uint8'] / 1e3:.1f} KB frame"
              + (f"; composited atlas {b['atlas_table_uint8'] / 1e3:.0f} KB"
                 if b["atlas_table_uint8"] else ""), ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
