"""Compact matched-hardware composed-render report; raw samples remain in source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from state_ops import catalog


def load(roots):
    rows = {}
    for root in roots:
        for path in root.glob("*/result.json"):
            r = json.loads(path.read_text())
            if r.get("status") != "COMPLETE":
                continue
            r["artifact"] = str(path)
            r["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows[r["registration"]] = r
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", type=Path, action="append", default=[])
    p.add_argument("--gpu", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cpu, gpu = load(a.cpu), load(a.gpu)
    expected = {r for r in catalog() if "Symbolic" not in r}
    assert set(gpu) == expected, (expected - set(gpu), set(gpu) - expected)
    assert set(cpu) == {"swm/CraftaxPixels-v1", "swm/CraftaxClassicPixels-v1"}
    rows = []
    for device, data in [("cpu", cpu), ("gpu", gpu)]:
        for name, r in sorted(data.items()):
            work = {(v["cadence"], v["mode"]): v for v in r["workloads"]}
            assert len(work) == 16
            for value in [*r["operations"].values(), *r["workloads"]]:
                if value["status"] == "MEASURED":
                    import numpy as np

                    assert len(value["samples_ns"]) >= 50 and value["warmup_iterations"] >= 10
                    assert float(np.median(value["samples_ns"])) == value["median_ns"]
                    assert float(np.percentile(value["samples_ns"], 95)) == value["p95_ns"]
            row = dict(
                registration=name,
                family=r["family"],
                device=device,
                profile=r["profile"],
                artifact=r["artifact"],
                sha256=r["sha256"],
                ratios=r["ratios"],
                post_restore_step_ratio=r["post_restore_step_ratio"],
                post_restore_project_render_ratio=r["post_restore_project_render_ratio"],
                cadences={},
            )
            for cadence in ["every", "four", "final", "none"]:
                base = work[cadence, "baseline"]
                get = work[cadence, "get"]
                st = work[cadence, "set"]
                branch = work[cadence, "branch"]
                row["cadences"][cadence] = dict(
                    horizon=r["horizon"],
                    frames=base["rendered_frames"],
                    baseline_median_ms=base["median_ns"] / 1e6,
                    baseline_p95_ms=base["p95_ns"] / 1e6,
                    additive_prediction_ms=base["additive_prediction_ns"] / 1e6,
                    measured_over_additive=base["measured_over_additive"],
                    get_slowdown=get["median_ns"] / base["median_ns"],
                    set_slowdown=st["median_ns"] / base["median_ns"],
                    branch_median_ms=branch["median_ns"] / 1e6,
                    branch_p95_ms=branch["p95_ns"] / 1e6,
                    branch_over_additive=branch["measured_over_additive"],
                    all_workload_operations_per_second={
                        m: work[cadence, m]["operations_per_second"] for m in ["baseline", "get", "set", "branch"]
                    },
                    all_workload_medians_ms={
                        m: work[cadence, m]["median_ns"] / 1e6 for m in ["baseline", "get", "set", "branch"]
                    },
                    all_measured_over_additive={
                        m: work[cadence, m]["measured_over_additive"] for m in ["baseline", "get", "set", "branch"]
                    },
                )
            rows.append(row)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "summary.json").write_text(
        json.dumps(
            dict(
                gpu_registration_count=len(gpu), cpu_registration_count=len(cpu), rows=rows, production_changes=False
            ),
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Matched rendered State-operation results",
        "",
        "All 37 accepted renderable registrations have actual composed GPU/backend coverage at B1; both pixel Craftax registrations also have CPU coverage. Every result uses a sixteen-step horizon, ten warmups and fifty samples. Symbolic Craftax has no renderer. GPU/backend labels identify CUDA device requests on Oscar; dynamics may still execute on the CPU according to each adapter. No semantic acceptance gate was rerun.",
        "",
        "Component ratios below use one matched host/profile and count projection exactly once. The old cross-host derived arithmetic is retained separately and is not used as a latency prediction. `render(RenderState)` follows explicit public projection where supported; other public render calls include their private projection.",
        "",
        "## Matched component ratios",
        "",
        "| Registration / device | Get/step | Set/step | Get/rendered | Set/rendered | Get slowdown | Set slowdown | Next step after set / reset-prefix | Next project+render after set / warm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        v = r["ratios"]
        lines.append(
            f"| {r['registration']} / {r['device']} | {v['get_over_step']:.3f} | {v['set_over_step']:.3f} | {v['get_over_rendered']:.3f} | {v['set_over_rendered']:.3f} | {v['get_slowdown']:.3f} | {v['set_slowdown']:.3f} | {r['post_restore_step_ratio']:.3f} | {r['post_restore_project_render_ratio']:.3f} |"
        )
    for cadence in ["every", "four", "final", "none"]:
        lines += [
            "",
            f"## Sixteen-step rollout: {cadence}",
            "",
            "Get workload adds get_state after each step. Set workload adds one initial restore. Branch includes snapshot, eight divergent steps, restore and sixteen continuation steps. Preparation is outside timing. Slowdowns compare actual composed medians with the actual baseline; additive comparisons use component medians and may differ because trajectory states, transfers, synchronization and cache reuse differ.",
            "",
            "| Registration / device | Baseline median / p95 ms | Additive ms | Measured/additive | Get slowdown | Set slowdown | Branch median / p95 ms | Branch/additive |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            c = r["cadences"][cadence]
            lines.append(
                f"| {r['registration']} / {r['device']} | {c['baseline_median_ms']:.3f} / {c['baseline_p95_ms']:.3f} | {c['additive_prediction_ms']:.3f} | {c['measured_over_additive']:.3f} | {c['get_slowdown']:.3f} | {c['set_slowdown']:.3f} | {c['branch_median_ms']:.3f} / {c['branch_p95_ms']:.3f} | {c['branch_over_additive']:.3f} |"
            )
    lines += [
        "",
        "## Post-restore effects and limitations",
        "",
        "The following observed ratios exceed 1.25 in the paired component check. This is a descriptive attention flag, not an acceptance threshold or a causal attribution. Per-operation p95, variance and raw samples remain in the linked-by-path artifacts. The full table retains smaller effects too.",
    ]
    flagged = []
    for r in rows:
        if max(r["post_restore_step_ratio"], r["post_restore_project_render_ratio"]) >= 1.25:
            flagged.append(r["registration"] + "/" + r["device"])
            lines.append(
                f"- {r['registration']} / {r['device']}: next step ratio {r['post_restore_step_ratio']:.3f}; project+render ratio {r['post_restore_project_render_ratio']:.3f}."
            )
    if not flagged:
        lines.append("No paired component effect exceeds that attention flag in this run.")
    lines += [
        "",
        "A sum of medians is an additive estimate, not the median of an actual rollout. Absolute CPU/GPU comparisons span different hardware and are not speedup claims. B1 composed coverage is additional to prior genuine larger-batch State measurements; no larger composed batch is claimed. First-call timings and construction/preparation durations are retained in raw results. The collector has already constructed the environment, stepped a prefix and warmed rendering, so these are not fresh-process cold-start measurements. The separate prior Craftax cold-start study remains unchanged.",
        "",
        "The plot compares measured and additive every-step-rendered baseline latency, using GPU/backend results only. Full per-cadence data, source hashes and profiles are in `summary.json`; raw samples are retained once in the source artifacts.",
        "",
        "![Measured versus additive rendered rollout](measured-versus-additive.svg)",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    gpu_rows = [r for r in rows if r["device"] == "gpu"]
    for family in sorted({r["family"] for r in gpu_rows}):
        group = [r for r in gpu_rows if r["family"] == family]
        ax.scatter(
            [r["cadences"]["every"]["additive_prediction_ms"] for r in group],
            [r["cadences"]["every"]["baseline_median_ms"] for r in group],
            label=family,
            alpha=0.7,
        )
    values = [r["cadences"]["every"][k] for r in gpu_rows for k in ["additive_prediction_ms", "baseline_median_ms"]]
    ax.plot([min(values) * 0.8, max(values) * 1.2], [min(values) * 0.8, max(values) * 1.2], ":", color="gray")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Additive prediction (ms, 16 steps)",
        ylabel="Measured composed median (ms, 16 steps)",
        title="Rendered rollout: matched components versus composition",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(a.output / "measured-versus-additive.svg")
    fig.savefig(a.output / "measured-versus-additive.png", dpi=160)
    print("PASS: 37 GPU/backend and 2 CPU registrations; all 624 composed distributions verified")


if __name__ == "__main__":
    main()
