"""Fresh-process Craftax initialization costs omitted by warm checkpoint preparation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from state_ops import authoritative, catalog, profile, synchronize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registration", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--kind", choices=["public", "oracle"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert "Craftax" in a.registration and a.registration in catalog()
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        registration=a.registration,
        device=a.device,
        kind=a.kind,
        batch_size=1,
        profile=profile(),
        measurements={},
        call_order=[],
        note="Fresh process. First public calls in this order; constructor/reset may internally call other operations. Compilation is included when first triggered. Warm benchmark checkpoint preparation already exercised step/set, so its first-call field alone is not a JIT cold-start measurement.",
    )

    def timed(name, fn):
        synchronize()
        start = time.perf_counter_ns()
        value = synchronize(fn())
        result["measurements"][name] = time.perf_counter_ns() - start
        result["call_order"].append(name)
        return value

    if a.kind == "public":
        import stable_envs as se

        row = catalog()[a.registration]
        env = timed("construction", lambda: se.make(row["accepted_classes"][0], num_envs=1, device=a.device))
        try:
            timed("first_explicit_reset", lambda: env.reset(seed=17))
            action = env.action_spec.sample(1, rng=np.random.default_rng(12345))
            timed("first_explicit_step", lambda: env.step(action, frames=None))
            state = timed("first_explicit_get_state", env.get_state)
            timed("first_explicit_set_state", lambda: env.set_state(state))
            timed("second_step_after_compilation", lambda: env.step(action, frames=None))
        finally:
            env.close()
    else:
        env = timed("construction", lambda: authoritative(a.registration))
        try:
            timed("first_explicit_reset", lambda: env.reset(seed=17))
            timed("first_explicit_step", lambda: env.step(0))
            timed("second_step_after_compilation", lambda: env.step(0))
        finally:
            env.close()
    result["status"] = "COMPLETE"
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["measurements"]), flush=True)


if __name__ == "__main__":
    main()
