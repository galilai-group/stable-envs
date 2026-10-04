"""CPU-only prerequisite for DMC direct-source/shared-memory comparisons.

This does not render, measure performance, select a production process default,
or satisfy a CPU/CUDA renderer acceptance gate. It verifies the benchmark's
direct source calls, complete State restoration and numeric output transport.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from contextlib import closing
from pathlib import Path

import numpy as np
import torch

from scripts.benchmarks.online_stepping.dmc_parallel_source import (
    NAMES,
    ParallelDMCSource,
    import_source,
    source,
    source_step,
)
from scripts.benchmarks.online_stepping.dmc_profile import make as make_adapter
from scripts.benchmarks.renderer_table.common import PIN, equal_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domains", nargs="+", choices=NAMES, default=list(NAMES))
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest["commit"] == PIN
    for f, h in manifest["files"].items():
        assert hashlib.sha256((args.source_root / f).read_bytes()).hexdigest() == h, f
    import_source(args.source_root)
    torch.set_num_threads(1)
    files = [Path(__file__).relative_to(Path.cwd()),
             Path("scripts/benchmarks/online_stepping/dmc_parallel_source.py"),
             Path("scripts/benchmarks/online_stepping/classic_parallel_source.py"),
             Path("scripts/benchmarks/renderer_table/common.py"),
             Path("scripts/benchmarks/online_stepping/dmc_profile.py"),
             *Path("stable_envs/_adapters").glob("dmcontrol*.py"),
             *Path("tests/helpers/reference").glob("dmc*semantics.py"),
             Path("tests/helpers/reference/dmc_semantics.py")]
    report = dict(status="RUNNING", scope="CPU B2/P2 source-transport versus local adapters; no public Env, rendering or timing",
        host=platform.node(), domains=args.domains, source_pin=PIN, source_files=manifest["files"],
        opengl_not_exercised=True, configured_gl=os.environ.get("MUJOCO_GL"),
        candidate_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "dm-control", "mujoco", "gymnasium", "tensordict")},
        source_runtime_files={}, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for domain in args.domains:
            ref = source(domain, 2)
            try:
                with closing(make_adapter(domain, 2, "cpu")) as env:
                    initial = env.reset(seed=17)
                    equal_state(initial, ref.get_state())
                    action_dim = ref.envs[0].action_space.shape[0]
                    actions = np.random.default_rng(31).uniform(-0.5, 0.5, (2, 8, action_dim))
                    for horizon in (1, 8):
                        with ParallelDMCSource(domain, args.source_root, 2, 2, horizon, ()) as parallel:
                            equal_state(initial, parallel.snapshot())
                            report["source_runtime_files"].update(parallel.runtime_files)
                            for boundary in (False, True):
                                state = initial.clone()
                                if boundary:
                                    limit = int(ref.envs[0].env._env._step_limit)
                                    state["lifecycle", "step_count"][:, 0] = torch.tensor([limit - 1, limit])
                                    state["lifecycle", "reset_next_step"][1] = True
                                    state["rng", "has_gauss"].fill_(True)
                                    state["rng", "cached_gaussian"].fill_(0.125)
                                ref.set_state(state)
                                parallel.restore(state)
                                env.set_state(state)
                                action = actions[:, :horizon]
                                expected = source_step(ref, action, (), "cpu")
                                batched = parallel.step(action, "cpu")
                                actual = torch.stack([env.step(action[:, t]) for t in range(horizon)], dim=1)
                                assert not bool(actual["terminated"].any() or actual["truncated"].any())
                                equal_state(ref.get_state(), parallel.snapshot())
                                equal_state(actual["state"][:, -1], ref.get_state())
                                public_keys = {".".join(k) if isinstance(k, tuple) else k
                                               for k in actual.exclude("state").keys(True, True)}
                                assert public_keys == set(expected) == set(batched)
                                for key in expected:
                                    torch.testing.assert_close(batched[key], expected[key], rtol=0, atol=0, equal_nan=True)
                                    path = tuple(key.split(".")) if "." in key else key
                                    value = actual[path]
                                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0, equal_nan=True)
                                held = {k: v.clone() for k, v in batched.items()}
                                next_output = parallel.step(-action, "cpu")
                                assert not torch.equal(next_output["observation"], held["observation"])
                                for key in held:
                                    torch.testing.assert_close(batched[key], held[key], rtol=0, atol=0, equal_nan=True)
                                report["cases"].append(dict(domain=domain, horizon=horizon, reset_boundary=boundary,
                                    batch=2, workers=2, state_task="EXACT", output_ownership="PASS",
                                    task_fields=sorted(public_keys), valid_steps=2 * horizon))
                                save()
                                print(domain, horizon, "boundary" if boundary else "reset", "PASS", flush=True)
            finally:
                ref.close()
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
