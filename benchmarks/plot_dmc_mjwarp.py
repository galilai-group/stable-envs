"""Regenerate published DMC MJWarp benchmark curves from recorded measurements.

Run from the repository root with the optional plotting dependency:
    uv run --no-project --with matplotlib==3.11.2 python benchmarks/plot_dmc_mjwarp.py

The default inputs are the committed 16-environment evidence files. No environment,
CUDA runtime or benchmark is invoked. Figures use medians from all 80 matched pairs;
source coverage, pairing and summary ratios are validated before plotting.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

METRICS = {
    "raw_physics": ("raw_physics_two_control_rounds", "Raw physics", "#2467A5"),
    "public_state_and_tasks": ("public_step_no_pixels", "Public State/tasks", "#168577"),
    "public_with_pixels": ("public_step_with_pixels", "Public + frame", "#C66A20"),
}
BATCHES = [1, 16, 64, 256, 1024]
NATIVE = "#72509D"
WARP = "#168577"


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def simple_tick(x, _):
    return f"{x:g}"


def setup(ax, ylim):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(0.8, 1280)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(FixedLocator(BATCHES))
    ax.xaxis.set_major_formatter(FuncFormatter(simple_tick))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(simple_tick))
    ax.grid(axis="y", which="major", color="#D8DEE4", lw=0.65)
    ax.grid(axis="x", which="major", color="#E8ECEF", lw=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A9B0B7")
    ax.tick_params(labelsize=9, length=3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    assets = Path(__file__).resolve().parents[1] / "docs" / "assets"
    evidence = assets / "dmc-mjwarp"
    ap.add_argument("--input", type=Path, default=evidence / "benchmark-16.json.gz")
    ap.add_argument("--summary", type=Path, default=evidence / "benchmark-summary-16.json")
    ap.add_argument("--output", type=Path, default=assets)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    raw = load(a.input)
    summary = load(a.summary)
    records = raw["records"]
    envs = raw["config"]["envs"]
    assert len(records) == 160 and len(envs) == 16 and "DMCReacher" not in envs
    assert raw["config"]["batches"] == BATCHES and all(r["status"] == "ok" for r in records)
    by = {(r["env"], r["batch"], r["simulation_backend"]): r for r in records}
    assert len(by) == 160
    assert len(raw["pairs"]) == 80 and all(p["same_segment"] and p["speedup_eligible"] for p in raw["pairs"])
    ratios = {}
    values = {}
    for env in envs:
        for batch in BATCHES:
            n = by[env, batch, "native"]
            w = by[env, batch, "mjwarp"]
            ns = n["continuation_provenance"]["segment_sha256"]
            ws = w["continuation_provenance"]["segment_sha256"]
            assert ns == ws
            for key, (field, label, color) in METRICS.items():
                nm = n["metrics"][field]["median_ms"]
                wm = w["metrics"][field]["median_ms"]
                assert nm > 0 and wm > 0
                ratios[env, batch, key] = nm / wm
                for backend, r in [("native", n), ("mjwarp", w)]:
                    m = r["metrics"][field]
                    values[env, batch, key, backend] = m["median_ms"]
    for r in summary["batch_1024_by_environment"]:
        for key in METRICS:
            actual = ratios[r["env"], 1024, key]
            expected = r["native_over_mjwarp"][key]
            assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    for row in summary["speedups_by_batch"]:
        for key in METRICS:
            actual = math.exp(sum(math.log(ratios[env, row["batch"], key]) for env in envs) / 16)
            assert math.isclose(actual, row["ratios"][key]["geometric_mean_native_over_mjwarp"], rel_tol=1e-12)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "semibold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    rv = list(ratios.values())
    ratio_lim = (10 ** math.floor(math.log10(min(rv))), 10 ** math.ceil(math.log10(max(rv))))
    pv = [v for (env, batch, key, backend), v in values.items() if key == "public_state_and_tasks"]
    latency_lim = (10 ** math.floor(math.log10(min(pv))), 10 ** math.ceil(math.log10(max(pv))))
    workload = (
        "RTX A6000 · AMD EPYC 7443 · 1 Torch thread/process · seed 0, zero actions · 5 warmups + 10 timed samples"
    )
    footer = "Native: CPU physics with CUDA outputs. Warp: CUDA physics. Fresh process per record; default rendering. Performance is not numerical acceptance."
    fig, axes = plt.subplots(4, 4, figsize=(18, 13), sharex=True, sharey=True)
    for ax, env in zip(axes.flat, envs, strict=True):
        setup(ax, ratio_lim)
        ax.axhline(1, color="#555D65", ls=(0, (3, 3)), lw=1.1, zorder=2)
        for key, (_, label, color) in METRICS.items():
            ax.plot(
                BATCHES, [ratios[env, b, key] for b in BATCHES], color=color, marker="o", ms=4, lw=1.9, label=label
            )
        ax.set_title(env, fontsize=10, pad=7)
        ax.tick_params(labelbottom=True, labelleft=True)
    fig.suptitle("DMC stepping: per-environment latency ratios", fontsize=21, x=0.06, ha="left", y=0.985)
    fig.text(0.06, 0.950, workload, fontsize=10.5, color="#3B4550")
    handles = [Line2D([0], [0], color=v[2], lw=2, marker="o", label=v[1]) for v in METRICS.values()]
    handles.append(Line2D([0], [0], color="#555D65", lw=1.1, ls="--", label="Parity (1×)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.056, 0.934), ncol=4, frameon=False, fontsize=11)
    fig.text(0.94, 0.919, "Above 1× favors Warp", ha="right", fontsize=11, weight="semibold", color="#168577")
    fig.supxlabel("Batch size (log scale)", y=0.039, fontsize=12)
    fig.supylabel("Native median latency / Warp median latency (×, log scale)", x=0.012, fontsize=12)
    fig.text(0.06, 0.012, footer, fontsize=9, color="#4B5560")
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.093, top=0.882, hspace=0.38, wspace=0.25)
    fig.savefig(a.output / "dmc-mjwarp-speedup.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(18, 13), sharex=True, sharey=True)
    for ax, env in zip(axes.flat, envs, strict=True):
        setup(ax, latency_lim)
        for backend, label, color in [
            ("native", "Native (CPU physics)", NATIVE),
            ("mjwarp", "Warp (CUDA physics)", WARP),
        ]:
            ax.plot(
                BATCHES,
                [values[env, b, "public_state_and_tasks", backend] for b in BATCHES],
                color=color,
                marker="o",
                ms=4,
                lw=1.9,
                label=label,
            )
        ax.set_title(env, fontsize=10, pad=7)
        ax.tick_params(labelbottom=True, labelleft=True)
    fig.suptitle("DMC stepping: absolute public-step latency", fontsize=21, x=0.06, ha="left", y=0.985)
    fig.text(0.06, 0.950, workload, fontsize=10.5, color="#3B4550")
    fig.legend(
        handles=[
            Line2D([0], [0], color=NATIVE, lw=2, marker="o", label="Native (CPU physics)"),
            Line2D([0], [0], color=WARP, lw=2, marker="o", label="Warp (CUDA physics)"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.056, 0.934),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.text(
        0.94,
        0.919,
        "Lower is faster · final State + tasks, no frame/history",
        ha="right",
        fontsize=10.5,
        color="#3B4550",
    )
    fig.supxlabel("Batch size (log scale)", y=0.039, fontsize=12)
    fig.supylabel("Median public-step latency (ms, log scale)", x=0.012, fontsize=12)
    fig.text(0.06, 0.012, footer, fontsize=9, color="#4B5560")
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.093, top=0.882, hspace=0.38, wspace=0.25)
    fig.savefig(a.output / "dmc-mjwarp-latency.png", dpi=170)
    plt.close(fig)
    print(f"Plotted {len(envs)} environments and {len(raw['pairs'])} matched pairs in {a.output}")


if __name__ == "__main__":
    main()
