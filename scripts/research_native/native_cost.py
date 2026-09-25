"""Diagnostic subset costs, never exhaustive-State-equivalent benchmarks."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path


def measure(fn):
    samples = []
    sink = None
    for i in range(210):
        start = time.perf_counter_ns()
        sink = fn()
        dt = time.perf_counter_ns() - start
        if i >= 10:
            samples.append(dt)
    return dict(
        median_ns=statistics.median(samples),
        p95_ns=sorted(samples)[189],
        operations_per_second=1e9 / statistics.median(samples),
        samples_ns=samples,
        warmup=10,
        iterations=200,
        result_type=type(sink).__name__,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["rocket", "pusht"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--library", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        engine=a.engine,
        python=sys.version,
        platform=platform.platform(),
        cpu="Apple M5",
        batch_size=1,
        compilation_included=False,
        operations={},
    )
    if a.engine == "rocket":
        import pybullet
        import rocket_native_probe as probe

        env = probe.probe.RocketLandingEnv()
        env.reset(seed=17)
        result["module"] = pybullet.__file__
        result["version"] = metadata.version("pybullet")
        filename = a.output.resolve() / "transfer.txt"
        os.environ["SE_ROCKET_CACHE_FILE"] = str(filename)
        os.environ["SE_ROCKET_CACHE_MASK"] = "31"

        def transfer(mode):
            os.environ["SE_ROCKET_CACHE_OPERATION"] = mode
            try:
                return env.env.getPhysicsEngineParameters()
            finally:
                del os.environ["SE_ROCKET_CACHE_OPERATION"]

        transfer("export")
        result["operations"]["numeric_cache_export_with_text_transport"] = measure(lambda: transfer("export"))
        result["operations"]["numeric_cache_import_with_text_transport"] = measure(lambda: transfer("import"))
        result["numeric_text_bytes"] = filename.stat().st_size
        # Keep one opaque handle live and remove each save timing result afterwards.
        saved = []
        result["operations"]["opaque_save"] = measure(lambda: saved.append(env.env.saveState()))
        sid = saved[0]
        result["operations"]["opaque_restore"] = measure(lambda: env.env.restoreState(sid))
        for sid in saved:
            env.env.removeState(sid)
        result["native_classification"] = "OPAQUE_SAME_PROCESS_NON_EQUIVALENT"
        result["scope"] = (
            "Opaque snapshots fail continuation; numeric cache subset lacks complete contacts/solver state. Text IO included."
        )
        env.close()
    else:
        import pusht_native_probe as probe

        native = probe.Native(a.library)
        env, _ = probe.naturally_reached("preserved_multiple_contact")
        values = [native.body(env.agent), native.body(env.block)]
        order = native.active(env)
        result["operations"]["numeric_body_export"] = measure(lambda: [native.body(env.agent), native.body(env.block)])
        result["operations"]["numeric_body_import"] = measure(
            lambda: [native.body(env.agent, values[0]), native.body(env.block, values[1])]
        )
        result["operations"]["active_order_export"] = measure(lambda: native.active(env))
        result["operations"]["active_order_import"] = measure(lambda: native.active(env, order))
        result["version"] = metadata.version("pymunk")
        result["numeric_subset_bytes"] = sum(v.nbytes for v in values) + order.nbytes
        result["native_classification"] = "NATIVE_PARTIAL"
        result["scope"] = (
            "Research body cache and active-order subset only; not full numeric reconstruction or exhaustive State."
        )
        env.close()
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
