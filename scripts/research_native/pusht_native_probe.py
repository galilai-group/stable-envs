"""Controlled native-field additions to preserved PushT research; no production adapter."""

from __future__ import annotations

import argparse
import copy
import ctypes
import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np
from pymunk._chipmunk_cffi import ffi

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "docs/research/semantic_recovery_evidence/pusht_phase_a/probe.py"
spec = importlib.util.spec_from_file_location("preserved_pusht_probe", OLD)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
FIELDS = [
    "cog_x",
    "cog_y",
    "position_x",
    "position_y",
    "velocity_x",
    "velocity_y",
    "force_x",
    "force_y",
    "angle",
    "angular_velocity",
    "torque",
    "transform_a",
    "transform_b",
    "transform_c",
    "transform_d",
    "transform_tx",
    "transform_ty",
    "sleeping_idle_time",
    "bias_velocity_x",
    "bias_velocity_y",
    "bias_angular_velocity",
]


def pointer(value):
    return int(ffi.cast("uintptr_t", value))


class Native:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.se_body_fields.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.se_active_arbiters.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
        self.lib.se_restore_active_order.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]

    def body(self, body, values=None, bias=False):
        result = np.zeros(len(FIELDS), np.float64) if values is None else np.asarray(values, np.float64).copy()
        count = self.lib.se_body_fields(
            pointer(body._body),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(values is not None),
            int(bias),
        )
        assert count == len(FIELDS)
        return result

    def active(self, env, values=None):
        if values is None:
            result = np.zeros((64, 2), np.uint64)
            count = self.lib.se_active_arbiters(
                pointer(env.space._space), result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), 64
            )
            assert count >= 0
            return result[:count].copy()
        result = np.asarray(values, np.uint64)
        return self.lib.se_restore_active_order(
            pointer(env.space._space), result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), len(result)
        )


def naturally_reached(case):
    if case.startswith("preserved_"):
        return old.prepare(case.removeprefix("preserved_"))
    env = old.PushT()
    options = copy.deepcopy(old.SAFE)
    options["state"] = np.array([250, 100, 250, 250, 0, 0, 0], np.float64)
    if case == "success_boundary":
        options["goal_state"] = options["state"].copy()
    env.reset(seed=17, options=options)
    history = []

    def toward(target, steps, until_contact=False):
        for _ in range(steps):
            action = np.clip((np.asarray(target) - np.asarray(env.agent.position)) / 100, -1, 1)
            out = env.step(action)
            history.append(
                {
                    "action": action.tolist(),
                    "contacts": out[4]["n_contacts"],
                    "terminated": bool(out[2]),
                    "reward": float(out[1]),
                }
            )
            if until_contact and out[4]["n_contacts"]:
                return

    if case == "free":
        toward([180, 100], 2)
    elif case == "success_boundary":
        toward([250, 100], 1)
    else:
        if case == "corner_contact":
            toward([300, 130], 4)
        elif case == "edge_contact":
            toward([220, 130], 4)
        toward(np.asarray(env.block.position), 30, until_contact=True)
        assert any(h["contacts"] for h in history), (case, "no naturally reached contact")
        if case in ["sustained", "separation_recontact", "resting"]:
            toward(np.asarray(env.block.position) + [0, 30], 8)
        if case == "separation_recontact":
            toward([100, 100], 8)
            toward(np.asarray(env.block.position), 30, until_contact=True)
        if case == "resting":
            toward(np.asarray(env.agent.position), 10)
    return env, history


def output_values(env, action):
    return old.vector(env.step(action.copy()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    native = Native(args.library)
    result = {
        "body_fields": FIELDS,
        "scope": "research additions to private numeric cache reconstruction; not a complete State API",
        "cases": [],
    }
    for case in [
        "free",
        "first_contact",
        "sustained",
        "separation_recontact",
        "corner_contact",
        "edge_contact",
        "resting",
        "success_boundary",
        "preserved_wall",
        "preserved_agent_block",
        "preserved_multiple_contact",
    ]:
        for mode in [
            "clean_independent",
            "existing_numeric_cache",
            "plus_bias",
            "plus_body_cache",
            "plus_active_order",
        ]:
            source, history = naturally_reached(case)
            target, _ = naturally_reached(case if mode == "clean_independent" else "free")
            try:
                for action in (
                    [] if mode == "clean_independent" else [np.array([-0.8, 0.7]), np.array([0.6, -0.8])] * 8
                ):
                    target.step(action)
                record = {"case": case, "mode": mode, "history": history}
                result["cases"].append(record)
                try:
                    if mode != "clean_independent":
                        old.restore(source, target, "detect_then_numeric_cache")
                    if mode not in {"existing_numeric_cache", "clean_independent"}:
                        for sb, tb in [(source.agent, target.agent), (source.block, target.block)]:
                            native.body(tb, native.body(sb), bias=mode == "plus_bias")
                    if mode == "plus_active_order":
                        record["active_order_restore_code"] = native.active(target, native.active(source))
                    record["contact_cache_source"] = old.cache_fields(source)
                    record["contact_cache_target"] = old.cache_fields(target)
                    record["contact_cache_exact"] = record["contact_cache_source"] == record["contact_cache_target"]
                    record["native_source_fields"] = [
                        native.body(source.agent).tolist(),
                        native.body(source.block).tolist(),
                    ]
                    record["native_target_fields"] = [
                        native.body(target.agent).tolist(),
                        native.body(target.block).tolist(),
                    ]
                    record["candidate_immediate_differences"] = old.differences(
                        old.candidate(source), old.candidate(target)
                    )
                    record["native_immediate_exact"] = all(
                        np.array_equal(native.body(a), native.body(b))
                        for a, b in [(source.agent, target.agent), (source.block, target.block)]
                    )
                    record["active_source"] = native.active(source).tolist()
                    record["active_target"] = native.active(target).tolist()
                    frame_a, frame_b = source.render(), target.render()
                    record["immediate_pixels_exact"] = bool(np.array_equal(frame_a, frame_b))
                    outputs_a, outputs_b = [], []
                    future_differences = []
                    native_a, native_b = [], []
                    for action in old.ACTIONS * 8:
                        outputs_a.append(output_values(source, action))
                        outputs_b.append(output_values(target, action))
                        future_differences.append(old.differences(old.candidate(source), old.candidate(target)))
                        native_a.append(np.stack([native.body(source.agent), native.body(source.block)]))
                        native_b.append(np.stack([native.body(target.agent), native.body(target.block)]))
                    a, b = np.stack(outputs_a), np.stack(outputs_b)
                    record["continuation_exact"] = bool(np.array_equal(a, b))
                    record["future_candidate_differences"] = future_differences
                    record["future_native_exact"] = bool(np.array_equal(native_a, native_b))
                    record["max_output_delta"] = float(np.abs(a - b).max())
                    record["reward_delta"] = float(np.abs(a[:, 7] - b[:, 7]).max())
                    record["done_exact"] = bool(np.array_equal(a[:, 8:10], b[:, 8:10]))
                    np.savez_compressed(
                        args.output / (case + "-" + mode + ".npz"),
                        source=a,
                        target=b,
                        source_native=np.stack(native_a),
                        target_native=np.stack(native_b),
                        source_frame=frame_a,
                        target_frame=frame_b,
                    )
                except Exception as exc:
                    record["failure"] = repr(exc)
                    record["traceback"] = traceback.format_exc()
                print(case, mode, record.get("continuation_exact"), record.get("failure"), flush=True)
            finally:
                source.close()
                target.close()
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    best = [r for r in result["cases"] if r["mode"] == "plus_active_order"]
    result["classification"] = (
        "EXACT_RESTORE_NOT_YET_FEASIBLE"
        if any(
            r.get("failure")
            or not r.get("continuation_exact")
            or not r.get("future_native_exact")
            or any(r.get("future_candidate_differences", []))
            for r in best
        )
        else "NARROW_NATIVE_EXTENSION_FEASIBLE"
    )
    result["classification_scope"] = (
        "Bounded prototype only; target-only detection and full numeric export/import remain to be reviewed; no production permission inferred."
    )
    result["production_dependency_changed"] = False
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
