"""Composed State branch timing on one matched host/profile; no adapter changes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from state_ops import catalog, measure, profile, synchronize


def scheduled(step, horizon, cadence):
    return (
        cadence == "every" or cadence == "four" and (step + 1) % 4 == 0 or cadence == "final" and step + 1 == horizon
    )


def ratios(step, get, restore, projection, render):
    base = step + projection + render
    return dict(
        get_over_step=get / step,
        set_over_step=restore / step,
        get_over_rendered=get / base,
        set_over_rendered=restore / base,
        get_slowdown=1 + get / base,
        set_slowdown=1 + restore / base,
    )


def derive(source, output):
    """Retain all 37 rows but never silently mix CPU and GPU hardware."""
    old = json.loads(source.read_text())
    rows = []
    for r in old["rows"]:
        if "Symbolic" in r["registration"]:
            continue
        for device in ["cpu", "gpu"]:
            c = r[device]["batches"]["1"]["operations"]
            if c["render"]["status"] != "MEASURED":
                continue
            matched = all(k in c and c[k]["status"] == "MEASURED" for k in ["step", "get_state", "set_state_same"])
            dynamic = c if matched else r["cpu"]["batches"]["1"]["operations"]
            t = {k: dynamic[k]["median_ns"] for k in ["step", "get_state", "set_state_same"]}
            projection = c["to_render_state"].get("median_ns", 0)
            render = c["render"]["median_ns"]
            rows.append(
                dict(
                    registration=r["registration"],
                    device=device,
                    matched_hardware=matched,
                    classification="MATCHED_EXISTING"
                    if matched
                    else "CROSS_HOST_ARITHMETIC_ONLY_NOT_A_VALID_LATENCY_PREDICTION",
                    projection_accounting="Old render(full State) includes projection; literal requested sum counts it twice when explicit projection exists.",
                    explicit_projection_ns=projection,
                    render_full_state_inclusive_ns=render,
                    literal_requested=ratios(t["step"], t["get_state"], t["set_state_same"], projection, render),
                    without_double_counting=ratios(t["step"], t["get_state"], t["set_state_same"], 0, render),
                )
            )
    output.mkdir(parents=True, exist_ok=False)
    (output / "derived.json").write_text(json.dumps(dict(source=str(source), rows=rows), indent=2) + "\n")


def run(a):
    import stable_envs as se

    row = catalog()[a.registration]
    name = (row.get("accepted_classes") or row["stable_envs_classes"])[0]
    a.output.mkdir(parents=True, exist_ok=False)
    data = dict(
        registration=a.registration,
        family=row["family"],
        device=a.device,
        batch_size=1,
        horizon=a.horizon,
        profile=profile(),
        command=sys.argv,
        source_commit=os.environ.get("STATE_BENCH_SOURCE_COMMIT"),
        operations={},
        workloads=[],
        status="RUNNING",
        projection_accounting="Explicit public projection followed by render(RenderState), once only; otherwise render(State) includes private projection.",
    )

    def save():
        (a.output / "result.json").write_text(json.dumps(data, indent=2) + "\n")

    save()
    started = time.perf_counter()
    e = se.make(name, num_envs=1, device=a.device)
    e.reset(seed=17)
    rng = np.random.default_rng(12345)
    actions = [e.action_spec.sample(1, rng=rng) for _ in range(max(a.horizon, 10) + 2)]
    for x in actions[:2]:
        e.step(x)
    snapshot = e.get_state()
    synchronize()
    data["construction_and_preparation_seconds"] = time.perf_counter() - started
    data["actions"] = [x.detach().cpu().tolist() for x in actions]
    explicit = callable(getattr(e, "to_render_state", None))
    data["explicit_public_projection"] = explicit

    def project(s):
        return e.to_render_state(s) if explicit else s

    def frame(s):
        return e.render(project(s))

    projected = project(snapshot)
    synchronize(e.render(projected))

    def reset_prefix():
        e.reset(seed=17)
        for x in actions[:2]:
            e.step(x)

    def restore():
        e.set_state(snapshot)

    def timeit(fn, prep=None):
        return measure(fn, prepare=prep, warmup=a.warmup, iterations=a.iterations, minimum_seconds=0)

    try:
        ops = data["operations"]
        for label, fn, prep in [
            ("step_after_restore", lambda: e.step(actions[2]), restore),
            ("step_after_reset_prefix", lambda: e.step(actions[2]), reset_prefix),
            ("get_state", e.get_state, restore),
            ("set_state", restore, lambda: [e.step(x) for x in actions[:8]]),
            ("projection", lambda: project(snapshot), None),
            ("render_projected", lambda: e.render(projected), None),
            ("project_render_warm", lambda: frame(snapshot), None),
            ("project_render_after_restore", lambda: frame(snapshot), restore),
        ]:
            if label == "projection" and not explicit:
                ops[label] = dict(status="INCLUDED_IN_RENDER", median_ns=0)
                continue
            ops[label] = timeit(fn, prep)
            save()
        for cadence in ["every", "four", "final", "none"]:
            frames = sum(scheduled(i, a.horizon, cadence) for i in range(a.horizon))
            for mode in ["baseline", "get", "set", "branch"]:

                def workload():
                    if mode == "branch":
                        checkpoint = e.get_state()
                        for x in actions[:8]:
                            e.step(x)
                        e.set_state(checkpoint)
                    elif mode == "set":
                        e.set_state(snapshot)
                    value = None
                    for i, x in enumerate(actions[2 : 2 + a.horizon]):
                        value = e.step(x)["state"]
                        if mode == "get":
                            value = e.get_state()
                        if scheduled(i, a.horizon, cadence):
                            value = frame(value)
                    return value

                timing = timeit(workload, restore)
                step = ops["step_after_restore"]["median_ns"]
                get = ops["get_state"]["median_ns"]
                st = ops["set_state"]["median_ns"]
                prediction = a.horizon * step + frames * (
                    ops["projection"]["median_ns"] + ops["render_projected"]["median_ns"]
                )
                if mode == "get":
                    prediction += a.horizon * get
                if mode == "set":
                    prediction += st
                if mode == "branch":
                    prediction += get + 8 * step + st
                data["workloads"].append(
                    dict(
                        cadence=cadence,
                        mode=mode,
                        rendered_frames=frames,
                        divergent_steps=8 if mode == "branch" else 0,
                        additive_prediction_ns=prediction,
                        measured_over_additive=timing["median_ns"] / prediction,
                        **timing,
                    )
                )
                save()
                print(a.registration, cadence, mode, timing["median_ns"] / 1e6, flush=True)
        data["ratios"] = ratios(
            ops["step_after_restore"]["median_ns"],
            ops["get_state"]["median_ns"],
            ops["set_state"]["median_ns"],
            ops["projection"]["median_ns"],
            ops["render_projected"]["median_ns"],
        )
        data["post_restore_step_ratio"] = (
            ops["step_after_restore"]["median_ns"] / ops["step_after_reset_prefix"]["median_ns"]
        )
        data["post_restore_project_render_ratio"] = (
            ops["project_render_after_restore"]["median_ns"] / ops["project_render_warm"]["median_ns"]
        )
        data["status"] = "COMPLETE"
        save()
    finally:
        e.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registration")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--derive", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    a = p.parse_args()
    if a.derive:
        derive(a.derive, a.output)
    else:
        run(a)


if __name__ == "__main__":
    main()
