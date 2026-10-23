"""Tabulate the GPU-vs-EGL and CPU-vs-software-GL offline sweeps and the online ladder as Markdown."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from benchmarks.summarize_sweep import BATCHES, _best, _cell, _load, _num, family


def _src(d: dict, n: str):
    v = d.get(n, {})
    return v.get("fps"), v.get("status", "-")


def offline(r: Path) -> dict:
    gpu, cpu = _load(r / "ours-cuda"), _load(r / "ours-cpu-swgl")
    e1, e8 = _load(r / "source-egl-1p"), _load(r / "source-egl-8p")
    w1, w8 = _load(r / "source-swgl-1p"), _load(r / "source-swgl-8p")
    names = sorted(set(gpu) | set(cpu))
    out = {"gpu": {}, "cpu": {}}
    for device, ours, s1, s8, label in (
        ("gpu", gpu, e1, e8, "GPU: stable-envs on CUDA vs the source with EGL (GPU OpenGL)"),
        ("cpu", cpu, w1, w8, "CPU: stable-envs on CPU vs the source with software OpenGL (llvmpipe), both CPU only"),
    ):
        print(f"\n## Offline, {label}\n")
        head = ["env", "frame"] + [f"b{b}" for b in BATCHES] + ["src 1p", "src 8p", "best / src 8p", "best / src 1p"]
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for n in names:
            o = ours.get(n, {})
            b = _best(o)
            f1, st1 = _src(s1, n)
            f8, st8 = _src(s8, n)
            r8 = b["fps"] / f8 if b and f8 else None
            r1 = b["fps"] / f1 if b and f1 else None
            out[device][n] = (b, f1, f8, r8)
            shape = (b or {}).get("frame_shape") or (_best(gpu.get(n)) or {}).get("frame_shape")
            row = [n, "x".join(map(str, shape[:2])) if shape else "-"]
            row += [_cell(o.get("points"), x) for x in BATCHES]
            row += [_num(f1) if f1 else st1, _num(f8) if f8 else st8, _num(r8, "{:.2f}x"), _num(r1, "{:.2f}x")]
            print("| " + " | ".join(row) + " |")
    print("\n## Family summary: best stable-envs batch / source in 8 processes\n")
    print("| family | classes | GPU min | GPU median | GPU max | CPU min | CPU median | CPU max |")
    print("|---|---|---|---|---|---|---|---|")
    fams = sorted({family(n) for n in names})
    for fam in fams:
        cells = []
        for device in ("gpu", "cpu"):
            v = sorted(x[3] for n, x in out[device].items() if family(n) == fam and x[3])
            cells += [f"{v[0]:.2f}x", f"{statistics.median(v):.2f}x", f"{v[-1]:.2f}x"] if v else ["-"] * 3
        count = sum(1 for n in names if family(n) == fam)
        print(f"| {fam} | {count} | " + " | ".join(cells) + " |")
    print("\n## Slower than the source in 8 processes\n")
    print("| env | device | best fps (batch) | src 8p | ratio |\n|---|---|---|---|---|")
    for device in ("gpu", "cpu"):
        for n, (b, _, f8, r8) in sorted(out[device].items()):
            if r8 is not None and r8 < 1:
                print(f"| {n} | {device.upper()} | {b['fps']:.0f} ({b['batch']}) | {f8:.0f} | {r8:.2f}x |")
    return out


def online(r: Path) -> None:
    rows = {}
    for f in sorted((r / "online").glob("*.json")) if (r / "online").exists() else []:
        env, tag = f.stem.split("_", 1)
        rung, v = next(iter(json.loads(f.read_text())["rungs"].items()))
        rows[(env, tag)] = v
    if not rows:
        return
    print("\n## Online ladder: 1,024 episodes x 196 steps\n")
    print("L0-swgl and L3-swgl render on CPU under software OpenGL; L0-egl renders with GPU OpenGL.\n")
    print("| env | rung | lanes | status | steps/s | median s | repeats | note |\n|---|---|---|---|---|---|---|---|")
    order = ["L0-egl", "L0-swgl", "L1", "L2", "L3-swgl", "L4"]
    for (env, tag), v in sorted(
        rows.items(), key=lambda x: (x[0][0], order.index(x[0][1]) if x[0][1] in order else 9)
    ):
        lanes = v.get("lanes") or (f"{v.get('procs')}x{v.get('lanes_per_proc')}" if v.get("procs") else "-")
        ok = v.get("status") == "ok"
        note = "" if ok else (v.get("error") or "")[:110].replace("|", "/")
        sps = _num(v.get("steps_per_s")) if ok else "-"
        med = _num(v.get("median_s"), "{:.1f}") if ok else "-"
        print(f"| {env} | {tag} | {lanes} | {v.get('status')} | {sps} | {med} | {v.get('repeats', '-')} | {note} |")
    print("\n### Source render rate from the ladder\n")
    print("Render rate = 1 / (1/L0 - 1/L1) steps/s: the source's frames/s for rendering alone, 8 processes.\n")
    print("| env | GL | L0 steps/s | L1 steps/s | source render frames/s |\n|---|---|---|---|---|")
    for env in sorted({e for e, _ in rows}):
        l1 = rows.get((env, "L1"), {}).get("steps_per_s")
        for tag in ("L0-egl", "L0-swgl"):
            l0 = rows.get((env, tag), {}).get("steps_per_s")
            if l0 and l1 and 1 / l0 > 1 / l1:
                print(f"| {env} | {tag[3:]} | {l0:.0f} | {l1:.0f} | {1 / (1 / l0 - 1 / l1):.0f} |")


def main(root: str) -> None:
    r = Path(root)
    offline(r)
    online(r)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
