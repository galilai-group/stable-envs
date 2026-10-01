"""Regenerate the standalone plot from the completed, same-node batch sweep."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
report = json.loads((root / "profile-v3/result.json").read_text())
assert report["status"] == "PASS"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
for kind, color, label in [("threads", "#d97706", "Threads"), ("async_shared", "#2563eb", "Processes (Gymnasium)")]:
    for workers, style in [(4, "--"), (8, "-")]:
        rows = sorted(
            [r for r in report["rows"] if r["kind"] == kind and r["workers"] == workers], key=lambda r: r["batch"]
        )
        x = [r["batch"] for r in rows]
        axes[0].errorbar(
            x,
            [r["steps_s_mean"] for r in rows],
            yerr=[r["steps_s_sd"] for r in rows],
            color=color,
            linestyle=style,
            marker="o",
            capsize=3,
            label=f"{label}, {workers} workers",
        )
        axes[1].plot(
            x,
            [statistics.mean(s["worker_cpu_cores"] for s in r["samples"]) for r in rows],
            color=color,
            linestyle=style,
            marker="o",
        )
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512, 1024, 2048], labels=["64", "128", "256", "512", "1024", "2048"])
    ax.set_xlabel("Independent environments in batch")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Task transitions / second")
axes[0].set_ylim(0, 5000)
axes[0].legend(fontsize=8, loc="lower right")
axes[0].set_title("Processes remove the thread throughput ceiling")
axes[1].set_ylabel("Active worker CPU equivalents")
axes[1].set_ylim(0, 8.7)
axes[1].axhline(8, color="#555555", linestyle=":", linewidth=1)
axes[1].text(70, 8.12, "8 allocated CPU cores", fontsize=9, color="#555555")
axes[1].set_title("Eight threads use only about 2.5 cores")
fig.suptitle("FetchPush CPU stepping — same node, 8-action chunks, 3 repeats", fontsize=13)
fig.savefig(root / "scaling.png", dpi=170)
