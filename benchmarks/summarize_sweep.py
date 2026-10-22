"""Tabulate offline_sweep and online_ladder results as Markdown."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

BATCHES = [1, 256, 512, 1024, 2048, 4096]
FAMILY = [
    ("classic", ("CartPole", "Pendulum", "Acrobot", "MountainCar")),
    ("pusht", ("PushT",)),
    ("navigation", ("TwoRoom", "Piecewise", "SimplePointMaze")),
    ("dmc", ("DMC",)),
    ("fetch", ("Fetch",)),
    ("ogbench-maze", ("PointMaze", "AntMaze", "HumanoidMaze")),
    ("ogbench-manip", ("OGB",)),
    ("rocket", ("PyFlyt",)),
]


def family(name: str) -> str:
    for fam, prefixes in FAMILY:
        if name.startswith(prefixes):
            return fam
    return "other"


def _load(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        out.update(json.loads(f.read_text()).get("envs", {}))
    return out


def _cell(points, b):
    for p in points or []:
        if p.get("batch") == b:
            return f"{p['fps']:.0f}" if p.get("status") == "ok" else p.get("status", "?")
    return "-"


def _best(entry):
    pts = [p for p in (entry or {}).get("points", []) if p.get("status") == "ok"]
    return max(pts, key=lambda p: p["fps"]) if pts else None


def _num(x, fmt="{:.0f}"):
    return fmt.format(x) if isinstance(x, (int, float)) else "-"


def offline(r: Path) -> None:
    gpu, cpu, cpu_sw = _load(r / "ours-cuda"), _load(r / "ours-cpu"), _load(r / "ours-cpu-swgl")
    s1, s8, codec = _load(r / "source-1p"), _load(r / "source-8p"), _load(r / "codec")
    for n, v in cpu_sw.items():
        v["software_gl"] = True
        cpu[n] = v
    names = sorted(set(gpu) | set(cpu) | set(s1) | set(s8))
    print("## Offline rendering: frames/s over 10,000 stored states\n")
    print("GPU and CPU columns are stable-envs at each batch size. `src` columns are the source renderer.")
    print("CPU cells marked † ran under software OpenGL (xvfb + llvmpipe).\n")
    head = ["env", "frame"] + [f"GPU {b}" for b in BATCHES] + [f"CPU {b}" for b in BATCHES]
    head += ["src 1p", "src 8p", "GPU best/src 8p", "CPU best/src 8p"]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    ratios, slower = {}, []
    for n in names:
        g, c = gpu.get(n, {}), cpu.get(n, {})
        gb, cb = _best(g), _best(c)
        src8, src1 = s8.get(n, {}).get("fps"), s1.get(n, {}).get("fps")
        rg = gb["fps"] / src8 if gb and src8 else None
        rc = cb["fps"] / src8 if cb and src8 else None
        if rg:
            ratios.setdefault(family(n), []).append((rg, rc, n))
        if (rg and rg < 1) or (rc and rc < 1):
            slower.append((n, gb, cb, src1, src8, rg, rc))
        shape = (gb or cb or {}).get("frame_shape")
        mark = "†" if c.get("software_gl") else ""
        row = [n, "x".join(map(str, shape[:2])) if shape else "-"]
        row += [_cell(g.get("points"), b) for b in BATCHES]
        cells = [_cell(c.get("points"), b) for b in BATCHES]
        row += [x + (mark if x[:1].isdigit() else "") for x in cells]
        row += [
            _num(src1) if src1 else s1.get(n, {}).get("status", "-"),
            _num(src8) if src8 else s8.get(n, {}).get("status", "-"),
            _num(rg, "{:.1f}x"),
            _num(rc, "{:.2f}x"),
        ]
        print("| " + " | ".join(row) + " |")
    print("\n## Family summary: best stable-envs batch / source with 8 processes\n")
    print("| family | classes | GPU min | GPU median | GPU max (class) | CPU min | CPU median | CPU max |")
    print("|---|---|---|---|---|---|---|---|")
    for fam, vals in sorted(ratios.items()):
        g = sorted(v[0] for v in vals)
        c = sorted(v[1] for v in vals if v[1])
        top = max(vals)[2]
        cs = (
            [_num(min(c), "{:.2f}x"), _num(statistics.median(c), "{:.2f}x"), _num(max(c), "{:.2f}x")]
            if c
            else ["-"] * 3
        )
        gs = f"{g[0]:.1f}x | {statistics.median(g):.1f}x | {g[-1]:.1f}x ({top})"
        print(f"| {fam} | {len(vals)} | {gs} | " + " | ".join(cs) + " |")
    print("\n## Classes slower than the source (8 processes) on at least one device\n")
    print("| env | GPU best fps (batch) | CPU best fps (batch) | src 1p | src 8p | GPU/src 8p | CPU/src 8p |")
    print("|---|---|---|---|---|---|---|")
    for n, gb, cb, src1, src8, rg, rc in slower:
        gs = f"{gb['fps']:.0f} ({gb['batch']})" if gb else "-"
        cs = f"{cb['fps']:.0f} ({cb['batch']})" if cb else "-"
        print(f"| {n} | {gs} | {cs} | {_num(src1)} | {_num(src8)} | {_num(rg, '{:.2f}x')} | {_num(rc, '{:.2f}x')} |")
    print("\n## Codec baseline\n")
    print("Bytes per frame for the stored state, raw uint8, JPEG (quality 95) and PNG, from stable-envs frames.")
    print("Decode throughput uses a torch DataLoader with 8 workers over the same 10,000 frames.\n")
    print("| env | state B | raw B | JPEG B | PNG B | JPEG decode fps | PNG decode fps | GPU best fps |")
    print("|---|---|---|---|---|---|---|---|")
    for n in sorted(codec):
        k = codec[n]
        if k.get("status") != "ok":
            continue
        gb = _best(gpu.get(n))
        print(
            f"| {n} | {k.get('state_bytes', '-')} | {k.get('raw_bytes', '-')} | {_num(k.get('jpeg_bytes'))} | "
            f"{_num(k.get('png_bytes'))} | {_num(k.get('jpeg_decode', {}).get('fps'))} | "
            f"{_num(k.get('png_decode', {}).get('fps'))} | {_num(gb['fps'] if gb else None)} |"
        )


def online(r: Path) -> None:
    rows = []
    for sub in ("online", "online2"):
        for f in sorted((r / sub).glob("*.json")) if (r / sub).exists() else []:
            for rung, v in json.loads(f.read_text())["rungs"].items():
                rows.append((f.stem.split("_")[0], rung, sub, v))
    if not rows:
        return
    print("\n## Online ladder: 1,024 episodes x 196 steps\n")
    print("| env | rung | run | lanes | status | steps/s | median s | repeats | note |")
    print("|---|---|---|---|---|---|---|---|---|")
    for env, rung, sub, v in sorted(rows, key=lambda x: (x[0], x[1], x[2])):
        lanes = v.get("lanes") or (f"{v.get('procs')}x{v.get('lanes_per_proc')}" if v.get("procs") else "-")
        ok = v.get("status") == "ok"
        note = "" if ok else (v.get("error") or "")[:110].replace("|", "/")
        sps = _num(v.get("steps_per_s")) if ok else "-"
        med = _num(v.get("median_s"), "{:.1f}") if ok else "-"
        print(
            f"| {env} | {rung} | {sub} | {lanes} | {v.get('status')} | {sps} | {med} | {v.get('repeats', '-')} | {note} |"
        )


def main(root: str) -> None:
    r = Path(root)
    offline(r)
    online(r)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
