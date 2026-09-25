"""Bounded exact replay diagnostics; preserve compact decisive lifecycle/reset witnesses."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path

import numpy as np
from clone_diagnostic import modules


def same(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def different(a, b):
    return [k for k in a if not np.array_equal(a[k], b[k])]


def exact_output(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_output(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact_output(x, y) for x, y in zip(a, b))
    return bool(np.array_equal(a, b))


def rocket(output):
    from rocket_replay import Replay, capture, rng_numeric, walk

    zero = np.zeros(7)
    burn = np.array([0.15, -0.1, 0.2, 1.0, 0.5, 0.1, -0.1])
    records = []
    witness = {}
    report = dict(
        registration="swm/PFRocketLanding-v0", records=records, official_engine_only=True, production_changes=False
    )

    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    cases = []
    for seed in [0, 17, 42]:
        cases.extend([(f"reset_{seed}", seed, {}, {}, 0), (f"thrust_{seed}", seed, {}, {}, 12)])
    cases.extend(
        [
            ("near_contact", 17, {}, {}, "near"),
            ("crash", 17, {}, {}, "terminal"),
            ("post_terminal_contact", 17, {}, {}, "post"),
            ("landing", 17, {"ceiling": 3.2}, {"accelerate_drop": False}, "terminal"),
            ("time_limit", 17, {"max_duration_seconds": 0.25}, {"accelerate_drop": False}, "terminal"),
            (
                "render_variation",
                42,
                {"render_mode": "rgb_array", "render_resolution": (64, 64)},
                {"variation": ["all"], "randomize_drop": True},
                12,
            ),
        ]
    )
    for name, seed, config, options, length in cases:
        src = Replay(config, seed, options)
        dst = Replay(config, 42, {})
        try:
            if isinstance(length, int):
                for _ in range(length):
                    src.step(burn)
            else:
                for _ in range(src.capacity):
                    if src.env.termination or src.env.truncation:
                        break
                    if length == "near" and src.env.lin_pos[2] <= 6:
                        break
                    src.step(zero)
                if length == "post":
                    for i in range(32):
                        src.step(burn if i % 2 else zero)
            for _ in range(16):
                dst.step(np.array([-0.3, 0.4, -0.1, 1.0, 0.8, -0.2, 0.2]))
            state = src.get_state()
            dst.set_state(state)
            actual = dst.get_state()
            record = dict(
                case=name,
                seed=seed,
                config=config,
                options=options,
                history_length=src.length,
                history_capacity=src.capacity,
                post_terminal_steps=src.post,
                terminal=bool(src.env.termination),
                truncated=bool(src.env.truncation),
                contacts=int(src.env.env.contact_array.sum()),
                immediate_exact=same(state, actual),
                independent_clients=src.env.env._client != dst.env.env._client,
                numeric_bytes=sum(v.nbytes for v in state.values()),
            )
            buf = io.BytesIO()
            np.savez(buf, **state)
            record["serialized_npz_bytes"] = buf.tell()
            record["schema"] = (
                [dict(key=k, shape=list(v.shape), dtype=str(v.dtype)) for k, v in state.items()]
                if not records
                else "same keys; shapes depend on compatible history capacity and render resolution"
            )
            before = src.get_state()
            record["independent_storage"] = not any(np.shares_memory(v, src.history) for v in state.values())
            record["render_between_steps_nonmutating"] = True
            trajectory = []
            for i, a in enumerate([burn, zero] * 4):
                if src.env.render_mode is not None:
                    current = src.get_state()
                    frame = src.render()
                    after = src.get_state()
                    record["render_between_steps_nonmutating"] &= same(current, after)
                    if not np.array_equal(frame, dst.render()):
                        record["pixels_exact"] = False
                x, y = src.step(a), dst.step(a)
                trajectory.append(exact_output(x, y) and same(src.get_state(), dst.get_state()))
            record["suffix_exact"] = all(trajectory)
            record.setdefault("pixels_exact", True if src.env.render_mode is not None else None)
            # External main-stream draws must remain independently captured/restorable.
            x = src.env.np_random.random(5)
            y = dst.env.np_random.random(5)
            record["future_rng_draw_exact"] = bool(np.array_equal(x, y))
            dst_before = dst.get_state()
            src.env.np_random.random()
            record["independent_rng"] = same(dst_before, dst.get_state())
            for next_seed in [0, 42]:
                a = src.env.reset(seed=next_seed, options=options)
                b = dst.env.reset(seed=next_seed, options=options)
                record.setdefault("future_seeded_resets_exact", []).append(
                    exact_output(a, b) and same(capture(src.env), capture(dst.env))
                )
            records.append(record)
            save()
            print(name, record["immediate_exact"], record["suffix_exact"], flush=True)
            if not record["immediate_exact"] or not record["suffix_exact"] or record["pixels_exact"] is False:
                report["classification"] = "REPLAY_DIVERGES"
                report["first_failure"] = name
                save()
                return
        finally:
            src.close()
            dst.close()
    # Invalid candidate validation must leave destination numeric state unchanged.
    src = Replay(seed=17)
    dst = Replay(seed=42)
    try:
        src.step(burn)
        saved = src.get_state()
        before = dst.get_state()
        invalid = []
        for change in ["length", "mask", "nan", "dtype", "config", "inconsistent_current"]:
            bad = {k: v.copy() for k, v in saved.items()}
            if change == "length":
                bad["history/length"][...] = src.capacity + 1
            if change == "mask":
                bad["history/mask"][0] = False
            if change == "nan":
                bad["history/actions"][0, 0] = np.nan
            if change == "dtype":
                bad["history/actions"] = bad["history/actions"].astype(np.float32)
            if change == "config":
                bad["recipe/config"][0] += 1
            if change == "inconsistent_current":
                bad["current/wrapper/reward"][...] += 1
            try:
                dst.set_state(bad)
                rejected = False
            except ValueError:
                rejected = True
            invalid.append(dict(change=change, rejected=rejected, destination_unchanged=same(before, dst.get_state())))
        report["atomic_invalid_candidates"] = invalid
        # A source-supported unseeded future reset reseeds Variation from fresh OS entropy.
        # Compare two restored, otherwise exact engines without replacing that public reset behavior.
        dst.set_state(saved)
        reset_options = {"variation": ["environment"], "randomize_drop": True}
        a = src.env.reset(seed=None, options=reset_options)
        b = dst.env.reset(seed=None, options=reset_options)
        sa, sb = capture(src.env), capture(dst.env)
        variation_a = {p: rng_numeric(getattr(n, "_np_random", None)) for p, n in walk(src.env.variation_space)}
        variation_b = {p: rng_numeric(getattr(n, "_np_random", None)) for p, n in walk(dst.env.variation_space)}
        mismatches = different(sa, sb)
        var_mismatch = different(variation_a, variation_b)
        report["future_unseeded_reset"] = dict(
            options=reset_options,
            outputs_exact=exact_output(a, b),
            numeric_mismatch_keys=mismatches,
            variation_rng_mismatch_paths=var_mismatch,
            interpretation="The actual reset(None) path seeds each Variation stream from new external entropy. An exact prior ReplayState cannot determine two independent future entropy draws; a reset-entropy-input contract would be additional semantics.",
        )
        for k in mismatches:
            witness["source/" + k] = sa[k]
            witness["destination/" + k] = sb[k]
        for k in var_mismatch:
            witness["source/variation" + k + "/rng"] = variation_a[k]
            witness["destination/variation" + k + "/rng"] = variation_b[k]
        witness["prior/history_actions"] = saved["history/actions"][: int(saved["history/length"])]
        np.savez_compressed(output / "unseeded-reset-counterexample.npz", **witness)
        report["seeded_replay_classification"] = "EXACT_REPLAY_FEASIBLE_IN_TESTED_SEEDED_SCOPE"
        report["classification"] = (
            "RESET_ANCHOR_NOT_REPRODUCIBLE" if mismatches or var_mismatch else "EXACT_REPLAY_FEASIBLE"
        )
        report["stop_reason"] = (
            "Unseeded future reset exactness failed; stop before broader replay performance or production design claims."
            if mismatches or var_mismatch
            else None
        )
        save()
    finally:
        src.close()
        dst.close()


def pusht(output):
    import gymnasium as gym

    old = modules("pusht")
    records = []
    witness = {}
    for registration in ["swm/PushT-v1", "swm/PushT-Discrete-v1"]:
        for seed in [0, 17, 42]:
            env = gym.make(registration)
            target = gym.make(registration)
            source = env.unwrapped
            options = copy.deepcopy(old.SAFE)
            options["goal_state"] = options["state"].copy()
            env.reset(seed=seed, options=options)
            target.reset(seed=seed, options=options)
            discrete = "Discrete" in registration
            actions = [
                np.array([0, 0], np.int64) if discrete else np.zeros(2),
                np.array([15, 8], np.int64) if discrete else np.array([0.2, -0.1]),
            ]
            outputs = []
            native = []
            term = []
            for i in range(1025):
                a = actions[0] if i == 0 else actions[1]
                out = env.step(a)
                term.append(bool(out[2]))
                outputs.append(old.vector(out))
                native.append(old.candidate(source)["space"]["stamp"])
            # A short reset/public-action replay has exact continuation but no public maximum history.
            target.reset(seed=seed, options=options)
            replay = []
            for i in range(1025):
                replay.append(old.vector(target.step(actions[0] if i == 0 else actions[1])))
            record = dict(
                registration=registration,
                seed=seed,
                spec_max_episode_steps=env.spec.max_episode_steps,
                wrappers=[type(env).__name__, type(env.env).__name__, type(source).__name__],
                first_success=term[0],
                physics_steps_after_success=int(native[-1] - native[0]),
                step_calls=1025,
                finite_prefix_replay_exact=bool(np.array_equal(outputs, replay)),
                truncated_ever=bool(np.asarray(outputs)[:, 9].any()),
                final_render_exact=bool(np.array_equal(env.render(), target.render())),
                classification="UNBOUNDED_PUBLIC_HISTORY",
                bound_reason="No registered TimeLimit; step always executes native substeps, returns truncated=False and does not guard previous success. Any proposed fixed prefix capacity can be exceeded.",
            )
            records.append(record)
            if seed == 17:
                witness[registration + "/source_tail"] = np.asarray(outputs)[-8:]
                witness[registration + "/replay_tail"] = np.asarray(replay)[-8:]
                witness[registration + "/physics_stamps"] = np.asarray([native[0], native[-1]], np.int64)
            env.close()
            target.close()
            print(registration, seed, record["finite_prefix_replay_exact"], flush=True)
    np.savez_compressed(output / "unbounded-history-witness.npz", **witness)
    (output / "result.json").write_text(
        json.dumps(
            dict(
                records=records,
                proposed_history_bound=None,
                candidate_fixed_shape=False,
                production_changes=False,
                stop_reason="Registered lifecycle has no finite physics horizon; no fixed-shape ReplayState or horizon performance claimed.",
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, choices=["rocket", "pusht"])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    (rocket if a.engine == "rocket" else pusht)(a.output)


if __name__ == "__main__":
    main()
