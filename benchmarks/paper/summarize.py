"""Turn one run.sh results directory into Markdown tables: offline and online speed, MSE, and Craftax."""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Families whose source renders with MuJoCo OpenGL, so EGL puts the source on the GPU.
SOURCE_GPU = ("DMC", "AntMaze", "HumanoidMaze", "PointMaze", "OGBCube", "OGBScene", "Fetch")


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _fmt(x, digits=0):
    if x is None:
        return "–"
    if x >= 100:
        return f"{x:,.0f}"
    return f"{x:.{max(digits, 1)}f}" if x < 10 else f"{x:.0f}"


def _ratio(a, b):
    return f"{a / b:.2f}×" if a and b else "–"


def _best(points):
    ok = [p for p in points if p.get("status") == "ok" and p.get("fps")]
    return max(ok, key=lambda p: p["fps"]) if ok else None


def _b1(points):
    return next((p for p in points if p.get("status") == "ok" and p.get("batch") == 1), None)


def _src(path, name):
    d = _load(path)
    r = (d or {}).get("envs", {}).get(name, {})
    return r.get("fps") if r.get("status") == "ok" else None


def offline(root: Path, names):
    rows = []
    for n in names:
        o = root / "offline" / n
        if not o.exists():
            continue
        gpu = ((_load(o / "snv-gpu.json") or {}).get("envs", {}).get(n) or {}).get("points", [])
        cpu = ((_load(o / "snv-cpu.json") or {}).get("envs", {}).get(n) or {}).get("points", [])
        cores = (_load(root / "run.json") or {}).get("ncores", 8)
        sg1, sgp = _src(o / "src-gpu-1p.json", n), _src(o / f"src-gpu-{cores}p.json", n)
        sc1, scp = _src(o / "src-cpu-1p.json", n), _src(o / f"src-cpu-{cores}p.json", n)
        mg = (_load(o / "mse-gpu.json") or {}).get("envs", {}).get(n) or {}
        mc = (_load(o / "mse-cpu.json") or {}).get("envs", {}).get(n) or {}
        bg, bc = _best(gpu), _best(cpu)
        b1g, b1c = _b1(gpu), _b1(cpu)
        gpu_source = n.startswith(SOURCE_GPU)
        rows.append(
            {
                "env": n,
                "snv_gpu_b1": b1g and b1g["fps"],
                "snv_gpu_best": bg and bg["fps"],
                "snv_gpu_best_b": bg and bg["batch"],
                "src_gpu_1p": sg1 if gpu_source else None,
                "src_gpu_np": sgp if gpu_source else None,
                "snv_cpu_b1": b1c and b1c["fps"],
                "snv_cpu_best": bc and bc["fps"],
                "snv_cpu_best_b": bc and bc["batch"],
                "src_cpu_1p": sc1,
                "src_cpu_np": scp,
                "mse_gpu": mg.get("mse_mean"),
                "mse_gpu_max": mg.get("mse_max"),
                "mse_cpu": mc.get("mse_mean"),
                "mse_cpu_max": mc.get("mse_max"),
                "gpu_source": gpu_source,
            }
        )
    return rows


def _arm(d, prefix):
    """Best steps/s over T of the arms whose key starts with ``prefix``, with its T."""
    best = None
    for k, v in (d or {}).get("arms", {}).items():
        if k.startswith(prefix) and v.get("status") == "ok":
            if best is None or v["steps_per_s"] > best[0]:
                best = (v["steps_per_s"], v.get("actions_per_call"), v)
    return best


def online(root: Path, names):
    rows = []
    for n in names:
        o = root / "online" / n
        if not o.exists():
            continue
        step, sg, sc = _load(o / "step.json"), _load(o / "src-render-gpu.json"), _load(o / "src-render-cpu.json")
        ng, nc = _load(o / "snv-gpu.json"), _load(o / "snv-render-cpu.json")
        src_step = _arm(step, "src-pipelined-step")
        src_rg, src_rc = _arm(sg, "src-pipelined-render"), _arm(sc, "src-pipelined-render")
        snv_step, snv_step_gpu = _arm(step, "snv-step-T"), _arm(ng, "snv-step-gpu")
        snv_rg, snv_rc = _arm(ng, "snv-render-gpu"), _arm(nc, "snv-render-cpu")
        gpu_source = n.startswith(SOURCE_GPU)

        def v(x):
            return x and x[0]

        def t(x):
            return x and x[1]

        def check(x):
            return (x[2].get("frame_check") or {}).get("mismatched") if x else None

        def rollout(x):
            return ((x[2].get("rollout_mse") or {}).get("mse_mean")) if x else None

        rows.append(
            {
                "env": n,
                "src_step": v(src_step),
                "snv_step": v(snv_step),
                "snv_step_t": t(snv_step),
                "snv_step_gpu": v(snv_step_gpu),
                "snv_step_gpu_mode": snv_step_gpu and snv_step_gpu[2].get("execution_mode"),
                "src_render_gpu": v(src_rg) if gpu_source else None,
                "snv_render_gpu": v(snv_rg),
                "snv_render_gpu_t": t(snv_rg),
                "src_render_cpu": v(src_rc),
                "snv_render_cpu": v(snv_rc),
                "snv_render_cpu_t": t(snv_rc),
                "src_frames_bad": [check(src_rg), check(src_rc)],
                "rollout_mse_gpu": rollout(src_rg),
                "rollout_mse_cpu": rollout(src_rc),
                "gpu_source": gpu_source,
            }
        )
    return rows


def craftax(root: Path):
    out = []
    for dev in ("gpu", "cpu"):
        d = _load(root / "craftax" / f"{dev}.json")
        for name, rows in (d or {}).get("envs", {}).items():
            by = {}
            for r in rows:
                if r.get("status") == "ok":
                    by.setdefault(r["batch"], {})[r["arm"]] = r["steps_per_s"]
            for b, arms in sorted(by.items()):
                out.append({"device": dev, "env": name, "batch": b, **arms})
    return out


def markdown(roots: list[Path]) -> str:
    off, on, cx, prov = [], [], [], []
    for root in roots:
        run = _load(root / "run.json") or {}
        names = sorted({p.name for p in (root / "offline").glob("*")} | {p.name for p in (root / "online").glob("*")})
        off += offline(root, names)
        on += online(root, names)
        cx += craftax(root)
        prov.append(
            f"| {root.name} | `{str(run.get('commit'))[:8]}` | `{str(run.get('main'))[:8]}` | {run.get('host')} | "
            f"{run.get('job')} | {run.get('gpu')} | {run.get('cores')} | {run.get('start')} | "
            f"{run.get('n_states')} | {run.get('batches')} | {run.get('episodes')} x {run.get('length')} "
            f"({run.get('software_gl_episodes', '–')} sw GL) | {run.get('repeats')} |"
        )
    lines = ["# stable-envs vs source: speed and frame agreement", ""]
    lines += [
        "Speeds are medians over repeats. Every arm of a job runs pinned to the same CPU cores, one GPU visible,",
        "on an exclusive node. MSE is per frame on the 0-255 scale.",
        "",
        "| Job | Harness | Main | Host | Slurm job | GPU, driver | Cores | Start | States | Batches | Episodes x steps | Repeats |",
        "|---|" + "---|" * 11,
        *prov,
        "",
    ]
    lines += [
        "## Offline rendering (frames/s, the same stored States)",
        "",
        "GPU: SNV on CUDA vs the source's MuJoCo OpenGL on the GPU (EGL); 'no src GPU' where the source has no GPU renderer.",
        "CPU: SNV on CPU vs the source on software OpenGL (llvmpipe) or its own CPU renderer, on the same cores.",
        "The source renders one State per call; 1p is one process, Np is one process per core.",
        "OGBCube/OGBScene offline source is the exact-OpenGL reference path on the upstream model, not the upstream env.",
        "",
        "| Env | SNV GPU B=1 | SNV GPU best (B) | Src GPU 1p | Src GPU Np | GPU speedup (best vs Np) "
        "| SNV CPU B=1 | SNV CPU best (B) | Src CPU 1p | Src CPU Np | CPU speedup (best vs Np) | MSE GPU (worst) | MSE CPU (worst) |",
        "|---|" + "---|" * 12,
    ]
    for r in off:
        lines.append(
            f"| {r['env']} | {_fmt(r['snv_gpu_b1'])} | {_fmt(r['snv_gpu_best'])} ({r['snv_gpu_best_b'] or '–'}) "
            f"| {_fmt(r['src_gpu_1p'])} | {_fmt(r['src_gpu_np'])} | {_ratio(r['snv_gpu_best'], r['src_gpu_np']) if r['gpu_source'] else 'no src GPU'} "
            f"| {_fmt(r['snv_cpu_b1'])} | {_fmt(r['snv_cpu_best'])} ({r['snv_cpu_best_b'] or '–'}) "
            f"| {_fmt(r['src_cpu_1p'])} | {_fmt(r['src_cpu_np'])} | {_ratio(r['snv_cpu_best'], r['src_cpu_np'])} "
            f"| {_fmt(r['mse_gpu'], 2)} ({_fmt(r['mse_gpu_max'], 2)}) | {_fmt(r['mse_cpu'], 2)} ({_fmt(r['mse_cpu_max'], 2)}) |"
        )
    lines += [
        "",
        "## Online stepping and rendering (steps/s, the same seeded episodes and actions)",
        "",
        "Source: its envs in ProcessAdapter workers (one process per core), one action per call, the next step",
        "submitted before the current one is taken. SNV: best over T actions per call (T shown).",
        "Step: physics only. Render: physics plus a frame every step. GPU render for the source is MuJoCo on EGL.",
        "",
        "| Env | Src step | SNV step CPU (T) | SNV step GPU (mode) | Step speedup "
        "| Src render GPU | SNV render GPU (T) | GPU speedup | Src render CPU | SNV render CPU (T) | CPU speedup "
        "| Rollout MSE GPU | Rollout MSE CPU | Src frame check |",
        "|---|" + "---|" * 13,
    ]
    for r in on:
        best_step = max(x for x in (r["snv_step"] or 0, r["snv_step_gpu"] or 0))
        bad = [b for b in r["src_frames_bad"] if b is not None]
        lines.append(
            f"| {r['env']} | {_fmt(r['src_step'])} | {_fmt(r['snv_step'])} ({r['snv_step_t'] or '–'}) "
            f"| {_fmt(r['snv_step_gpu'])} ({r['snv_step_gpu_mode'] or '–'}) | {_ratio(best_step, r['src_step'])} "
            f"| {_fmt(r['src_render_gpu'])} | {_fmt(r['snv_render_gpu'])} ({r['snv_render_gpu_t'] or '–'}) "
            f"| {_ratio(r['snv_render_gpu'], r['src_render_gpu']) if r['gpu_source'] else 'no src GPU'} "
            f"| {_fmt(r['src_render_cpu'])} | {_fmt(r['snv_render_cpu'])} ({r['snv_render_cpu_t'] or '–'}) "
            f"| {_ratio(r['snv_render_cpu'], r['src_render_cpu'])} | {_fmt(r['rollout_mse_gpu'], 2)} | {_fmt(r['rollout_mse_cpu'], 2)} "
            f"| {'ok' if bad and not any(bad) else ('MISMATCH' if any(bad) else '–')} |"
        )
    if cx:
        lines += [
            "",
            "## Craftax (steps/s): upstream jit(vmap(step)) vs SNV",
            "",
            "| Device | Env | B | Upstream | SNV step | SNV step + frames | Step / upstream |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in cx:
            lines.append(
                f"| {r['device']} | {r['env']} | {r['batch']} | {_fmt(r.get('upstream'))} | {_fmt(r.get('snv-step'))} "
                f"| {_fmt(r.get('snv-step-render'))} | {_ratio(r.get('snv-step'), r.get('upstream'))} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    roots = [Path(a) for a in sys.argv[1:]]
    md = markdown(roots)
    (roots[0].parent / "summary.md").write_text(md)
    print(md)
