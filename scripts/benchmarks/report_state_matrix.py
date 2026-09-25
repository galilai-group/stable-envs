"""Build a descriptive per-registration report from sealed benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from state_ops import catalog


def metric(operation, key="median_ns", scale=1):
    return operation.get(key, 0) / scale if operation.get("status") == "MEASURED" else None


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def plot_costs(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    for family, items in families.items():
        step = [r["cpu"]["batches"]["1"]["operations"]["step"]["median_ns"] / 1e6 for r in items]
        restore = [r["cpu"]["batches"]["1"]["operations"]["set_state_same"]["median_ns"] / 1e6 for r in items]
        axis.scatter(step, restore, label=family, alpha=0.8)
        for row, x, y in zip(items, step, restore, strict=True):
            if "OGBCube" in row["registration"] or "OGBScene" in row["registration"]:
                axis.annotate(
                    row["registration"].split("/")[1],
                    (x, y),
                    xytext=(5, -14 if "OGBCube" in row["registration"] else 8),
                    textcoords="offset points",
                    fontsize=8,
                )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Step median (ms, CPU B1)",
        ylabel="Same-instance restore median (ms, CPU B1)",
    )
    axis.set_title("Measured State restoration cost across 39 registrations")
    low, high = axis.get_xlim()
    axis.plot([low, high], [low, high], linestyle=":", color="grey", label="Equal latency")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(output / "state-operation-costs.svg", metadata={"Date": None})
    figure.savefig(output / "state-operation-costs.png", dpi=160)
    plt.close(figure)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", type=Path, action="append", required=True)
    p.add_argument(
        "--gpu",
        type=Path,
        action="append",
        required=True,
        help="Ordered runs; later complete cases supersede earlier profiles.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cold", type=Path, action="append", required=True)
    p.add_argument("--dmc-native", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    expected = catalog()
    cpu, gpu = {}, {}
    for root, target in [(r, cpu) for r in a.cpu] + [(r, gpu) for r in a.gpu]:
        for path in sorted(root.glob("*/result.json")):
            data = json.loads(path.read_text())
            if data["status"] == "COMPLETE":
                data["artifact"] = str(path)
                data["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                target[data["registration"]] = data
    assert set(cpu) == set(expected), ("CPU coverage", sorted(set(expected) - set(cpu)))
    assert set(gpu) == set(expected), ("GPU coverage", sorted(set(expected) - set(gpu)))
    dmc_supplement = json.loads(a.dmc_native.read_text())
    assert dmc_supplement["status"] == "COMPLETE"
    dmc_native = {r["registration"]: r["native"] for r in dmc_supplement["rows"]}
    assert set(dmc_native) == {rid for rid, row in expected.items() if row["family"] == "dmcontrol"}
    rows = []
    for rid in expected:
        c, g = cpu[rid], gpu[rid]
        assert all(x.get("status") == "MEASURED" for x in c["batches"].values())
        assert "1" in c["batches"] and len(c["batches"]) > 1
        assert set(c["batches"]) == set(g["batches"])
        for run in [c, g]:
            operations = [op for cell in run["batches"].values() for op in cell["operations"].values()]
            if "oracle" in run:
                operations.extend([run["oracle"]["reset"], run["oracle"]["step"]])
                operations.extend(run["oracle"]["native"][k] for k in ["get", "set"] if k in run["oracle"]["native"])
            for op in operations:
                if op["status"] != "MEASURED":
                    continue
                assert op["warmup_iterations"] >= 10 and op["measured_iterations"] >= 50
                assert len(op["samples_ns"]) == op["measured_iterations"]
                assert op["median_ns"] == float(np.median(op["samples_ns"]))
                assert op["p95_ns"] == float(np.percentile(op["samples_ns"], 95))
                assert op["operations_per_second"] == 1e9 / op["median_ns"]
        if "Symbolic" not in rid:
            assert all(cell["operations"]["render"]["status"] == "MEASURED" for cell in g["batches"].values()), rid
        rows.append(
            dict(
                registration=rid,
                family=c["family"],
                implementation=c["stable_envs_class"],
                cpu=c,
                gpu=g,
                native_classification=c["oracle"]["native"]["classification"],
                closest_native=dmc_native.get(rid, c["oracle"]["native"]),
            )
        )
        closest = rows[-1]["closest_native"]
        operations = c["batches"]["1"]["operations"]
        rows[-1]["closest_native_ratios_not_like_for_like"] = (
            {
                "get_over_native_get": operations["get_state"]["median_ns"] / closest["get"]["median_ns"],
                "set_over_native_set": operations["set_state_same"]["median_ns"] / closest["set"]["median_ns"],
            }
            if "get" in closest
            else {}
        )
    report = dict(
        schema=1,
        coverage=len(rows),
        rows=rows,
        all_native_comparisons_like_for_like=False,
        status="DESCRIPTIVE_ONLY",
        batch64="DEFERRED_RUNTIME",
        production_changes=False,
        dmc_integration_supplement=dmc_supplement,
        dmc_integration_artifact=str(a.dmc_native),
        dmc_integration_sha256=hashlib.sha256(a.dmc_native.read_bytes()).hexdigest(),
    )
    cold = []
    for root in a.cold:
        for path in sorted(root.glob("*/result.json")):
            item = json.loads(path.read_text())
            assert item["status"] == "COMPLETE"
            item["artifact"] = str(path)
            item["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            cold.append(item)
    assert {(v["registration"], v["device"], v["kind"]) for v in cold} == {
        (r, device, kind)
        for r in expected
        if "Craftax" in r
        for device in ["cpu", "cuda"]
        for kind in ["public", "oracle"]
    }
    assert len(cold) == 16
    report["cold_craftax"] = cold
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_costs(rows, a.output)
    lines = [
        "# State-operation performance results",
        "",
        "All 39 accepted registrations are measured separately on CPU at B1 and a genuine larger public batch. GPU rendering uses Oscar; Craftax also has GPU dynamics/State measurements. No production optimization or promotion follows from these descriptive timings.",
        "",
        "Times below are warm medians. Raw result JSON includes p95, calls/s, every sample, variance, first-call and construction times, exact commands, versions and hardware. Calls/s count whole batch calls; they are not rendered FPS. NPZ sizes are uncompressed named numeric serialization, not Python heap size.",
        "",
        "![CPU B1 step and restoration costs](state-operation-costs.svg)",
        "",
        "CPU: Apple M5, family-specific accepted environments. GPU: RTX 3090, Linux x86_64, profile details in each artifact. Original GPU job 6693308 leaked normal-profile site-packages into child profiles; its Craftax/Fetch/Cube/Scene results are superseded by the isolated-profile retry. Both runs are retained. No duplicate jobs were submitted.",
        "",
        "## CPU per-registration State costs",
        "",
        "| Registration | B1 payload / NPZ bytes | Step µs | Get µs | Same set µs | Independent set µs | Get/step | Set/step | Branch ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        c = row["cpu"]["batches"]["1"]
        o = c["operations"]
        s = c["state_size"]
        rat = c["ratios"]
        lines.append(
            f"| {row['registration']} | {s['numeric_payload_bytes']} / {s['serialized_npz_uncompressed_bytes']} | "
            + " | ".join(
                fmt(v)
                for v in [
                    metric(o["step"], scale=1000),
                    metric(o["get_state"], scale=1000),
                    metric(o["set_state_same"], scale=1000),
                    metric(o["set_state_independent"], scale=1000),
                    rat["get_state_over_step"],
                    rat["set_state_same_over_step"],
                    metric(o["branch_cycle"], scale=1e6),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Larger public CPU batches",
        "",
        "B16 is used except B4 for reconstruction-heavy Cube/Scene. B64 is explicitly deferred for runtime; it was not emulated or claimed unsupported. These times include the whole batch.",
        "",
        "| Registration | Batch | Payload / NPZ bytes | Step ms | Get ms | Set ms | Branch ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        c = next(v for k, v in row["cpu"]["batches"].items() if k != "1")
        o = c["operations"]
        s = c["state_size"]
        lines.append(
            f"| {row['registration']} | {c['batch_size']} | {s['numeric_payload_bytes']} / {s['serialized_npz_uncompressed_bytes']} | "
            + " | ".join(fmt(metric(o[k], scale=1e6)) for k in ["step", "get_state", "set_state_same", "branch_cycle"])
            + " |"
        )
    lines += [
        "",
        "## Oracle and closest native baseline (CPU B1)",
        "",
        "Every native comparison is a strict subset and is **not like-for-like exhaustive State**. NONE means no comparable native interface. DMC uses Physics get/set with the explicit integration signature; its original default-signature lower bound is also retained in raw results. Fetch/OGBench use MuJoCo integration state; Craftax copies/installs its engine pytree. Pixel Craftax intrinsically generates observations in both upstream and delegated public steps; the public API then exports State rather than returning the Gym observation. No extra standalone render call is timed inside step.",
        "",
        "| Registration | Oracle reset µs | Oracle step µs | Native class | Native get / set µs | Public/native get / set |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        d = row["cpu"]
        o = d["oracle"]
        n = row["closest_native"]
        rat = row["closest_native_ratios_not_like_for_like"]
        lines.append(
            f"| {row['registration']} | {fmt(metric(o['reset'], scale=1000))} | {fmt(metric(o['step'], scale=1000))} | {n['classification']} | {fmt(metric(n.get('get', {}), scale=1000))} / {fmt(metric(n.get('set', {}), scale=1000))} | {fmt(rat.get('get_over_native_get'))} / {fmt(rat.get('set_over_native_set'))} |"
        )
    lines += [
        "",
        "## GPU operations (B1)",
        "",
        "Dashes mean the operation is unsupported or was intentionally covered by CPU dynamics. Exact refusal reasons are retained in JSON. Explicit public State → RenderState is timed where exposed. Rendering is a separate call on a full snapshot.",
        "",
        "| Registration | Projection ms | Render ms | Get ms | Set ms | Branch ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        o = row["gpu"]["batches"]["1"]["operations"]
        lines.append(
            f"| {row['registration']} | "
            + " | ".join(
                fmt(metric(o.get(k, {}), scale=1e6))
                for k in ["to_render_state", "render", "get_state", "set_state_same", "branch_cycle"]
            )
            + " |"
        )
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    lines += [
        "",
        "## Fresh-process Craftax initialization",
        "",
        "One fresh process per registration/device/API kind, with a separate compilation-cache path. Seconds below include compilation when first triggered; interpreter startup and hardware probing are excluded. Call order is constructor → reset → step → public get/set → second step. Constructor/reset can internally invoke other operations. The second step can still incur initialization work after restoration (visible for Classic CPU); it is not labelled steady-state. Oracle cold steps use action 0; public cold steps use the recorded script's fixed action seed. Warm benchmark oracle/public actions are matched separately. These are individual cold observations, not p95 estimates; the repeated warm measurements above provide distributions.",
        "",
        "| Registration | Device | API | Constructor s | First reset s | First step s | Second step s | First set s |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in cold:
        values = item["measurements"]
        lines.append(
            f"| {item['registration']} | {item['device']} | {item['kind']} | "
            + " | ".join(
                fmt(values[k] / 1e9) if k in values else "—"
                for k in [
                    "construction",
                    "first_explicit_reset",
                    "first_explicit_step",
                    "second_step_after_compilation",
                    "first_explicit_set_state",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Family ranges (CPU B1)",
        "",
        "Ranges span registrations, including shared adapters. They are not independent backend counts.",
        "",
        "| Family | Registrations | Payload bytes range | Get/step range | Same set/step range | Native class |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for family, items in groups.items():
        cells = [r["cpu"]["batches"]["1"] for r in items]

        def extent(values):
            return f"{min(values):.3f}–{max(values):.3f}"

        lines.append(
            f"| {family} | {len(items)} | {extent([v['state_size']['numeric_payload_bytes'] for v in cells])} | {extent([v['ratios']['get_state_over_step'] for v in cells])} | {extent([v['ratios']['set_state_same_over_step'] for v in cells])} | {', '.join(sorted({r['native_classification'] for r in items}))} |"
        )
    ranked = sorted(rows, key=lambda r: r["cpu"]["batches"]["1"]["ratios"]["set_state_same_over_step"], reverse=True)
    lines += ["", "## Outliers and limits", ""]
    for row in ranked[:5]:
        c = row["cpu"]["batches"]["1"]
        lines.append(
            f"- {row['registration']}: same-instance restore is {c['ratios']['set_state_same_over_step']:.2f}× step; branch median {c['branch']['median_ns'] / 1e6:.3f} ms."
        )
    variances = [
        (r["registration"], b, k, v["coefficient_of_variation"])
        for r in rows
        for b, c in r["cpu"]["batches"].items()
        for k, v in c["operations"].items()
        if v.get("status") == "MEASURED" and v["coefficient_of_variation"] > 0.25
    ]
    lines += [
        "",
        f"{len(variances)} CPU operation cells have coefficient of variation above 0.25. Medians, p95 and raw samples remain available; no accuracy ranking or noise-based deletion is claimed.",
        "",
        "Cube/Scene restoration includes upstream reconstruction; Craftax costs include exhaustive tensor/JAX conversion and synchronization. These are candidates for the post-42/42 optimization phase. Hardware/profile differences preclude treating CPU/GPU ratios as engine speedups. Branch timing includes eight divergent and two continuation steps, snapshot and restore; preparation is outside timing.",
        "",
        "Rocket remains INCOMPLETE; its opaque native snapshots are OPAQUE_SAME_PROCESS_NON_EQUIVALENT. PushT remains INCOMPLETE; its research typed-body/contact-order accessor is NATIVE_PARTIAL. Their diagnostic timing and exact continuation failures are documented separately. No NATIVE_EQUIVALENT baseline was established.",
        "",
        "See `summary.json` for source artifact hashes and full nested results. Reproduce individual runs with each stored command and profile; the harness documentation explains adding accepted registrations. Accepted evidence was preserved and never overwritten.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"PASS: {len(rows)} CPU and GPU registration records")


if __name__ == "__main__":
    main()
