"""Tabulate offline_sweep and online_ladder results as Markdown."""

from __future__ import annotations

import json
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


def _fps(points, b):
    for p in points or []:
        if p.get("batch") == b:
            return f"{p['fps']:.0f}" if p.get("status") == "ok" else p.get("status", "?")
    return "-"


def main(root: str) -> None:
    r = Path(root)
    gpu, cpu = _load(r / "ours-cuda"), _load(r / "ours-cpu")
    s1, s8, codec = _load(r / "source-1p"), _load(r / "source-8p"), _load(r / "codec")
    names = sorted(set(gpu) | set(cpu) | set(s1) | set(s8))
    head = ["env", "frame"]
    head += [f"GPU b{b}" for b in BATCHES] + [f"CPU b{b}" for b in BATCHES]
    head += ["src 1p", "src 8p", "JPEG dec", "PNG dec", "B/state", "B/JPEG", "B/PNG", "GPU best / src 8p"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    best = {}
    for n in names:
        g, c = gpu.get(n, {}), cpu.get(n, {})
        pts = [p for p in g.get("points", []) if p.get("status") == "ok"]
        shape = pts[0]["frame_shape"] if pts else None
        gbest = max((p["fps"] for p in pts), default=None)
        src8 = s8.get(n, {}).get("fps")
        src1 = s1.get(n, {}).get("fps")
        k = codec.get(n, {})
        ratio = gbest / src8 if gbest and src8 else None
        if ratio:
            best.setdefault(family(n), []).append((ratio, n))
        row = [n, "x".join(map(str, shape[:2])) if shape else "-"]
        row += [_fps(g.get("points"), b) for b in BATCHES] + [_fps(c.get("points"), b) for b in BATCHES]
        row += [
            f"{src1:.0f}" if src1 else s1.get(n, {}).get("status", "-"),
            f"{src8:.0f}" if src8 else s8.get(n, {}).get("status", "-"),
            f"{k['jpeg_decode']['fps']:.0f}" if "jpeg_decode" in k else "-",
            f"{k['png_decode']['fps']:.0f}" if "png_decode" in k else "-",
            str(k.get("state_bytes", "-")),
            f"{k['jpeg_bytes']:.0f}" if "jpeg_bytes" in k else "-",
            f"{k['png_bytes']:.0f}" if "png_bytes" in k else "-",
            f"{ratio:.1f}x" if ratio else "-",
        ]
        print("| " + " | ".join(row) + " |")
    print("\n| family | classes | min GPU best / src 8p | median | max (class) |\n|---|---|---|---|---|")
    for fam, vals in sorted(best.items()):
        vals.sort()
        print(
            f"| {fam} | {len(vals)} | {vals[0][0]:.1f}x | {vals[len(vals) // 2][0]:.1f}x | {vals[-1][0]:.1f}x ({vals[-1][1]}) |"
        )
    online = r / "online"
    if online.exists():
        print("\n| env | rung | status | steps/s | median s | repeats | note |\n|---|---|---|---|---|---|---|")
        for f in sorted(online.glob("*.json")):
            for rung, v in json.loads(f.read_text())["rungs"].items():
                sps = f"{v['steps_per_s']:.0f}" if v.get("status") == "ok" else "-"
                med = f"{v['median_s']:.1f}" if v.get("status") == "ok" else "-"
                note = v.get("error", "")[:120] if v.get("status") != "ok" else ""
                env = f.stem.rsplit("_", 1)[0]
                print(f"| {env} | {rung} | {v.get('status')} | {sps} | {med} | {v.get('repeats', '-')} | {note} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
