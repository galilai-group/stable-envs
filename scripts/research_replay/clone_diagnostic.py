"""Isolated whole-environment clone probes; never a numeric State implementation."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/recovery"))


def modules(engine):
    if engine == "rocket":
        import rocket_feasibility as old

        return old
    path = ROOT / "docs/research/semantic_recovery_evidence/pusht_phase_a/probe.py"
    spec = importlib.util.spec_from_file_location("old_pusht", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make(engine, case):
    old = modules(engine)
    if engine == "rocket":
        config = dict(render_mode="rgb_array", render_resolution=(64, 64))
        if case == "terminal":
            config["ceiling"] = 3.2
        env = old.RocketLandingEnv(**config)
        env.reset(seed=17, options={"accelerate_drop": case != "terminal"})
        burn = np.array([0.15, -0.1, 0.2, 1.0, 0.5, 0.1, -0.1])
        zero = np.zeros(7)
        if case == "free":
            for _ in range(3):
                env.step(burn)
        else:
            for _ in range(1300):
                env.step(zero)
                if env.termination or env.truncation:
                    break
                if case == "first_contact" and env.landing_pad_contact:
                    break
            if case == "combined":
                for _ in range(10):
                    env.step(burn)
        return env, [burn, zero] * 4
    if case == "combined":
        env, _ = old.prepare("multiple_contact")
    else:
        env = old.PushT(render_mode="rgb_array", resolution=64)
        options = copy.deepcopy(old.SAFE)
        if case in ["first_contact", "terminal"]:
            options["state"] = np.array([250, 100, 250, 250, 0, 0, 0], np.float64)
        if case == "terminal":
            options["goal_state"] = options["state"].copy()
        env.reset(seed=17, options=options)
        if case == "first_contact":
            for _ in range(50):
                out = env.step(np.clip((np.asarray(env.block.position) - np.asarray(env.agent.position)) / 100, -1, 1))
                if out[4]["n_contacts"]:
                    break
        else:
            env.step(np.zeros(2))
    return env, old.ACTIONS * 2


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def capture(e, engine):
    old = modules(engine)
    return plain(old.capture(e) if engine == "rocket" else old.candidate(e))


def identity(e, engine):
    if engine == "rocket":
        return dict(native_client=e.env._client, owner_pid=e.env._pid, python_engine=id(e.env))
    from pymunk._chipmunk_cffi import ffi

    return dict(
        native_space=int(ffi.cast("uintptr_t", e.space._space)),
        agent=int(ffi.cast("uintptr_t", e.agent._body)),
        block=int(ffi.cast("uintptr_t", e.block._body)),
    )


def output(e, a):
    out = e.step(a.copy())
    return plain(out)


def child(a):
    source, actions = make(a.engine, a.case)
    clone = None
    result = dict(engine=a.engine, case=a.case, method=a.method, source_identity=identity(source, a.engine))
    before = capture(source, a.engine)
    try:
        if a.method == "copy":
            clone = copy.copy(source)
        elif a.method == "deepcopy":
            clone = copy.deepcopy(source)
        elif a.method in ["pickle", "cloudpickle"]:
            serializer = pickle if a.method == "pickle" else __import__("cloudpickle")
            payload = serializer.dumps(source)
            result["serialized_bytes"] = len(payload)
            clone = serializer.loads(payload)
        elif a.method == "engine_copy":
            if a.engine == "rocket":
                result.update(
                    classification="COPY_REJECTED",
                    reason="Pinned Bullet/PyFlyt exposes no whole-environment independent clone operation; opaque snapshots were already proved non-equivalent.",
                )
                return result
            # Binding-supported Space.copy traverses callback ownership; do not edit callbacks to make it serialize.
            native = source.space.copy()
            result["native_copy_succeeded"] = True
            result.update(
                classification="IMMEDIATE_MISMATCH",
                reason="Native Space.copy returns only an engine subset, not a complete environment clone.",
            )
            del native
            return result
        result["clone_identity"] = identity(clone, a.engine)
        key = "native_client" if a.engine == "rocket" else "native_space"
        aliased = result["clone_identity"][key] == result["source_identity"][key]
        result["native_aliased"] = aliased
        result["immediate_numeric_exact"] = capture(clone, a.engine) == before
        result["immediate_pixels_exact"] = bool(np.array_equal(source.render(), clone.render()))
        clone.step(actions[0].copy())
        result["source_unchanged_after_clone_step"] = capture(source, a.engine) == before
        if aliased or not result["source_unchanged_after_clone_step"]:
            result["classification"] = "ALIASED_NATIVE_ENGINE"
        elif not result["immediate_numeric_exact"] or not result["immediate_pixels_exact"]:
            result["classification"] = "IMMEDIATE_MISMATCH"
        else:
            # Recreate an unstepped branch before testing identical futures.
            if a.method == "copy":
                clone = copy.copy(source)
            elif a.method == "deepcopy":
                clone = copy.deepcopy(source)
            else:
                clone = serializer.loads(payload)
            result["continuation_exact"] = all(
                output(source, x) == output(clone, x) and capture(source, a.engine) == capture(clone, a.engine)
                for x in actions
            )
            result["classification"] = (
                "EXACT_INDEPENDENT_CLONE" if result["continuation_exact"] else "CONTINUATION_DIVERGENCE"
            )
            if result["classification"] == "EXACT_INDEPENDENT_CLONE":
                # Fresh-process serialization and reverse ownership are mandatory before this label can stand.
                result["classification"] = "RESOURCE_OWNERSHIP_FAILURE"
                result["reason"] = (
                    "Independent continuation passed; additional fresh-process/ownership proof is required before any positive clone classification."
                )
        rng_before = copy.deepcopy(source.np_random.bit_generator.state)
        clone.np_random.random()
        result["source_rng_unchanged_after_clone_draw"] = source.np_random.bit_generator.state == rng_before
        try:
            clone.close()
            source.step(actions[-1].copy())
            result["source_usable_after_clone_close"] = True
        except Exception as exc:
            result["source_usable_after_clone_close"] = False
            result["close_error"] = repr(exc)
    except Exception as exc:
        result.update(
            classification="SERIALIZATION_REJECTED" if a.method in ["pickle", "cloudpickle"] else "COPY_REJECTED",
            error=repr(exc),
            exception_type=type(exc).__name__,
        )
    finally:
        try:
            source.close()
        except Exception:
            pass
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["rocket", "pusht"], required=True)
    p.add_argument("--case")
    p.add_argument("--method")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.case:
        print("CLONE_RESULT " + json.dumps(child(a), sort_keys=True))
        return
    a.output.mkdir(parents=True, exist_ok=False)
    records = []
    versions = {}
    for n in ["numpy", "PyFlyt", "pybullet", "pymunk", "gymnasium", "cloudpickle"]:
        try:
            versions[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            pass
    for case in ["free", "first_contact", "combined", "terminal"]:
        for method in ["copy", "deepcopy", "pickle", "cloudpickle", "engine_copy"]:
            command = [sys.executable, __file__, "--engine", a.engine, "--case", case, "--method", method]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=180)
            lines = [
                x.removeprefix("CLONE_RESULT ") for x in proc.stdout.splitlines() if x.startswith("CLONE_RESULT ")
            ]
            record = (
                json.loads(lines[-1])
                if lines
                else dict(
                    engine=a.engine,
                    case=case,
                    method=method,
                    classification="RESOURCE_OWNERSHIP_FAILURE",
                    error=proc.stderr[-2500:],
                )
            )
            record["process_exit_code"] = proc.returncode
            records.append(record)
            print(a.engine, case, method, record["classification"], flush=True)
            (a.output / "result.json").write_text(
                json.dumps(
                    dict(profile=versions, command=sys.argv, records=records, production_changes=False), indent=2
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
