"""How much of the Fetch per-frame GPU cost is reachable, and what does removing it cost in pixels.

Read ``findings.md`` first. The stage profile shows the Fetch path is COMPUTE bound from B=64 up:
the plateau at ~5.2k f/s is per-frame device work that scales with the batch, not a fixed cost that
fails to amortise. ``BATCH_SCALES = fps(4096)/fps(1)`` can therefore only be moved by making the
per-frame device work smaller, and the arithmetic in findings.md says by how much. This measures how
much is actually available, and what each unit of it costs in pixels.

METHOD. Every ablation is a FLAG on ``mujoco_warp``'s render context or a context manager this repo
already ships -- nothing here edits ``stable_envs`` and nothing touches the raytracer kernels. The
flag is applied by wrapping ``stable_envs.render.mjwarp.create_render_context`` (or
``stable_envs.envs.fetch._context``) from the outside for the duration of one cell, the context cache
is cleared between cells so the flag actually reaches a fresh context, and then the ORDINARY PUBLIC
``env.render(states)`` is timed and digested. So each row is a caller-visible fps AND a sha256 of the
frames over the frozen Step 4 corpus, against the unmodified control.

A row is only interesting if its digest MATCHES the control: Fetch declares the PARITY tier at
``parity_interior_min = 0.99`` and ``docs/research/fetch.md`` measures 0.9919-0.9948, so an ablation
that moves pixels is a fidelity proposal and not a performance one.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import socket
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def digest(t) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy())).hexdigest()


def timed(fn, repeat: int, warmup: int) -> float:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(repeat):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        s.append(a.elapsed_time(b))
    return statistics.median(s)


@contextlib.contextmanager
def ctx_flags(MW, **over):
    """Force flags onto every ``create_render_context`` call made inside the block."""
    orig = MW.create_render_context

    def patched(m, **kw):
        kw.update(over)
        return orig(m, **kw)
    MW.create_render_context = patched
    try:
        yield
    finally:
        MW.create_render_context = orig


@contextlib.contextmanager
def lightspec(fetch_mod, enabled: bool):
    """Wrap ``_context`` in the repo's compile-time light specialisation."""
    if not enabled:
        yield
        return
    from stable_envs.render import mjwarp_lightspec as LS
    orig = fetch_mod._context

    def patched(*a, **k):
        with LS.specialised_lights(True):
            return orig(*a, **k)
    fetch_mod._context = patched
    try:
        yield
    finally:
        fetch_mod._context = orig


@contextlib.contextmanager
def chunk(fetch_mod, n):
    if n is None:
        yield
        return
    old = fetch_mod.MJW_CHUNK
    fetch_mod.MJW_CHUNK = int(n)
    try:
        yield
    finally:
        fetch_mod.MJW_CHUNK = old


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="FetchReach")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 256, 4096])
    ap.add_argument("--digest-batch", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--states-dir", type=Path,
                    default=ROOT / "examples" / "results" / "step4" / "states")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import torch

    import stable_envs as se
    from stable_envs.envs import fetch as fetch_mod
    from stable_envs.render import mjwarp as MW

    pool = np.load(args.states_dir.expanduser().resolve() / f"{args.env}.npz")["states"]

    CELLS = [
        ("control", {}, False, None),
        ("skybox_off", {"render_skybox": False}, False, None),
        ("specular_off", {"enable_specular": False}, False, None),
        ("textures_off", {"use_textures": False}, False, None),
        # ``render_seg`` is load-bearing: the composite decides mover pixels from the segmentation
        # buffer, so a context without one raises in ``_context``. Kept out of the table rather than
        # reported as a saving.
        ("lightspec", {}, True, None),
        ("chunk_512", {}, False, 512),
        ("chunk_1024", {}, False, 1024),
        # The two cells that came back pixel-identical to the control, together and with the chunk
        # knob, because neither is a saving worth proposing on its own.
        ("skybox+textures_off", {"render_skybox": False, "use_textures": False}, False, None),
        ("skybox+textures_off+chunk512", {"render_skybox": False, "use_textures": False},
         False, 512),
    ]

    note = {"workstream": "step6/fetch-gpu", "env": args.env,
            "what": "context-flag ablation: how much per-frame device cost is reachable, "
                    "and which of it is free of pixel change",
            "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
            "cpu": platform.processor() or platform.machine(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "states": str(args.states_dir / f"{args.env}.npz"), "cells": {}}

    for name, flags, ls, ch in CELLS:
        print(f"[ablate] {name} flags={flags} lightspec={ls} chunk={ch}", flush=True)
        fetch_mod._CTX_CACHE.clear()
        torch.cuda.empty_cache()
        rec: dict = {"flags": flags, "lightspec": ls, "chunk": ch, "fps": {}}
        try:
            with ctx_flags(MW, **flags), lightspec(fetch_mod, ls), chunk(fetch_mod, ch):
                # digest first, on the frozen corpus, at the batch Step 4 used
                n = args.digest_batch
                e = se.make(args.env, batch=n, device="cuda")
                st = torch.as_tensor(np.asarray(pool[np.arange(n) % len(pool)], np.float32),
                                     device="cuda")
                rec["digest"] = digest(e.render(st))
                del e, st
                for B in args.batch:
                    e = se.make(args.env, batch=B, device="cuda")
                    st = torch.as_tensor(np.asarray(pool[np.arange(B) % len(pool)], np.float32),
                                         device="cuda")
                    ms = timed(lambda: e.render(st), args.repeat, args.warmup)
                    rec["fps"][str(B)] = B / (ms / 1e3)
                    rec.setdefault("us_per_frame", {})[str(B)] = ms * 1e3 / B
                    del e, st
                    torch.cuda.empty_cache()
        except Exception as exc:                                        # pragma: no cover
            rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
        note["cells"][name] = rec
        fetch_mod._CTX_CACHE.clear()
        torch.cuda.empty_cache()

    ctrl = note["cells"].get("control", {})
    for name, rec in note["cells"].items():
        if "fps" not in rec or not ctrl.get("fps"):
            continue
        rec["pixels_identical_to_control"] = rec.get("digest") == ctrl.get("digest")
        rec["speedup_vs_control"] = {b: rec["fps"][b] / ctrl["fps"][b]
                                     for b in rec["fps"] if b in ctrl["fps"]}
        if "1" in rec["fps"] and str(max(args.batch)) in rec["fps"]:
            rec["batch_scales"] = rec["fps"][str(max(args.batch))] / rec["fps"]["1"]

    args.out.mkdir(parents=True, exist_ok=True)
    tag = socket.gethostname().split(".")[0]
    p = args.out / f"{args.env}_ablation_{tag}.json"
    p.write_text(json.dumps(note, indent=1))
    print("wrote", p)

    L = [f"# Context-flag ablation — {args.env} on {note['gpu']}", "",
         f"host {note['host']}  ·  slurm {note['slurm_job']}  ·  frozen states `{note['states']}`",
         "", "| cell | " + " | ".join(f"fps({b})" for b in args.batch)
         + " | us/frame(max B) | vs control | BATCH_SCALES | pixels identical |",
         "|---" * (len(args.batch) + 5) + "|"]
    mb = str(max(args.batch))
    for name, r in note["cells"].items():
        if "fps" not in r or not r["fps"]:
            L.append(f"| {name} | " + " | ".join("-" for _ in args.batch)
                     + f" | - | - | - | {r.get('error', '-')[:60]} |")
            continue
        L.append(f"| {name} | "
                 + " | ".join(f"{r['fps'].get(str(b), float('nan')):.0f}" for b in args.batch)
                 + f" | {r.get('us_per_frame', {}).get(mb, float('nan')):.1f}"
                 + f" | {r.get('speedup_vs_control', {}).get(mb, float('nan')):.3f}x"
                 + f" | {r.get('batch_scales', float('nan')):.1f}x"
                 + f" | {'YES' if r.get('pixels_identical_to_control') else 'NO'} |")
    (args.out / f"{args.env}_ablation_{tag}.md").write_text("\n".join(L) + "\n")
    print("wrote", args.out / f"{args.env}_ablation_{tag}.md")


if __name__ == "__main__":
    main()
