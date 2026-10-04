"""Close public PushT boundary, quantizer, Variation and projection evidence.

The independent pinned source advances normally. No source checkpoint is restored.
Task predicate boundaries are also compared directly to avoid physics moving an
ULP boundary before it can be evaluated. Reports retain numeric comparisons.
"""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pusht_replay_gate import diagnostics, equal, import_source

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import stable_envs as se


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    profile = json.loads(args.profile.read_text())
    for path, digest in profile["source_files"].items():
        assert sha(args.source / path) == digest, path
    for path, digest in profile["production_files"].items():
        assert sha(ROOT / path) == digest, path
    args.output.mkdir(parents=True, exist_ok=False)
    classes = import_source(args.source)
    records = []
    arrays = {}

    def witness(label, source, actual):
        source, actual = np.asarray(source), np.asarray(actual)
        np.testing.assert_array_equal(source, actual, err_msg=label)
        arrays[label + "/source"] = source
        arrays[label + "/actual"] = actual

    def check_step(label, oracle, env, action):
        saved = action.copy()
        expected = oracle.step(action.copy())
        got = env.step(action[None], states=None, frames=None)
        equal(saved, action)
        witness(label + "/task", expected[1:4], [got[k][0].item() for k in ("reward", "terminated", "truncated")])
        witness(label + "/observation", expected[0]["state"], got["state"]["current", "observation"][0].cpu().numpy())
        equal(diagnostics(oracle), diagnostics(env._simulator.envs[0].env))
        witness(label + "/frame", oracle.render(), env.render()[0].cpu().numpy())

    for discrete, cls in enumerate(classes):
        name = "PushTDiscrete" if discrete else "PushT"
        oracle = cls(resolution=96)
        with se.make(name, device="cpu", size=96) as env:
            try:
                oracle.reset(seed=17)
                env.reset(seed=17)
                # Cartesian product includes exact thresholds and adjacent doubles.
                distances = [0.0, np.nextafter(20.0, -np.inf), 20.0, np.nextafter(20.0, np.inf)]
                angles = [
                    0.0,
                    np.nextafter(np.pi / 9, -np.inf),
                    np.pi / 9,
                    np.nextafter(np.pi / 9, np.inf),
                    np.nextafter(2 * np.pi, 0.0),
                    2 * np.pi,
                ]
                for i, (distance, angle, velocity) in enumerate(itertools.product(distances, angles, (0.0, 3.0))):
                    goal = np.array([distance, 0, 0, 0, angle, velocity, -velocity], np.float64)
                    cur = np.zeros(7, np.float64)
                    witness(
                        name + f"/boundary{i}",
                        oracle.eval_state(goal, cur),
                        env._simulator.envs[0].env.eval_state(goal, cur),
                    )
                records.append(dict(registration=env.reference_id, gate="task_boundaries", cases=i + 1))
                controls = (
                    [
                        np.array([i, j], dtype=dtype)
                        for dtype in (np.int32, np.int64)
                        for i, j in ((0, 0), (0, 15), (15, 0), (15, 15))
                    ]
                    if discrete
                    else [
                        np.array([x, y], dtype=dtype)
                        for dtype in (np.float32, np.float64)
                        for x, y in itertools.product((-1.0, 0.0, 1.0), repeat=2)
                    ]
                )
                for i, control in enumerate(controls):
                    check_step(name + f"/action{i}", oracle, env, control)
                records.append(dict(registration=env.reference_id, gate="action_boundaries", cases=len(controls)))
                # Compact render projection must be refused by set_state atomically.
                state = env.get_state()
                projection = env.to_render_state(state)
                witness(name + "/projection", env.render(state).numpy(), env.render(projection).numpy())
                try:
                    env.set_state(projection)
                except (ValueError, KeyError):
                    pass
                else:
                    raise AssertionError("RenderState accepted as exhaustive State")
                for key, value in state.items(True, True):
                    assert torch.equal(value, env.get_state()[key]), key
                # All public persistent Variation leaves, with typed nondefaults.
                for i, key in enumerate(env.variation_names):
                    node = env.variation_space._walk(key)
                    if hasattr(node, "n"):
                        value = int(node.start + ((int(node.init_value) - node.start + 1) % node.n))
                    else:
                        value = np.asarray(node.low + 0.37 * (node.high.astype(float) - node.low), dtype=node.dtype)
                    overrides = {key: value}
                    before = env.get_state()
                    env.set_variation(overrides)
                    # Configuration may change future resets, not current body state.
                    for k, v in before["current"].items(True, True):
                        assert torch.equal(v, env.get_state()["current"][k]), key
                    oracle.variation_space.set_init_value(overrides)
                    options = {"variation": []}
                    oracle.reset(seed=42, options=options)
                    env.reset(seed=42, options=options)
                    equal(diagnostics(oracle), diagnostics(env._simulator.envs[0].env))
                    control = np.array([7, 12], np.int64) if discrete else np.array([0.2, -0.3], np.float64)
                    check_step(name + f"/variation{i}", oracle, env, control)
                    state = env.get_state()
                    env.step(control[None], frames=None)
                    env.set_state(state)
                    equal(diagnostics(oracle), diagnostics(env._simulator.envs[0].env))
                records.append(
                    dict(
                        registration=env.reference_id,
                        gate="persistent_variation",
                        paths=env.variation_names,
                        cases=i + 1,
                    )
                )
            finally:
                oracle.close()
        # Rendering configuration modes have separate public execution witnesses.
        for render_action, with_target in itertools.product((False, True), repeat=2):
            oracle = cls(resolution=96, render_action=render_action, with_target=with_target)
            try:
                with se.make(name, device="cpu", size=96, render_action=render_action, with_target=with_target) as env:
                    oracle.reset(seed=42)
                    env.reset(seed=42)
                    control = np.array([15, 15], np.int64) if discrete else np.ones(2, np.float32)
                    check_step(f"{name}/config{render_action}-{with_target}", oracle, env, control)
            finally:
                oracle.close()
        records.append(
            dict(
                registration="swm/PushT-Discrete-v1" if discrete else "swm/PushT-v1",
                gate="render_configuration",
                cases=4,
            )
        )
        print(name, "boundary/variation/configuration PASS", flush=True)

    # Exhaustive polar indices: source discrete vs public discrete AND public
    # continuous with source-dequantized actions on independently advancing lanes.
    oracle = classes[1](resolution=32)
    try:
        with (
            se.make("PushTDiscrete", size=32, device="cpu") as discrete,
            se.make("PushT", size=32, device="cpu") as continuous,
        ):
            oracle.reset(seed=17)
            discrete.reset(seed=17)
            continuous.reset(seed=17)
            for i, (r, a) in enumerate(itertools.product(range(16), repeat=2)):
                indices = np.array([r, a], np.int32 if i % 2 else np.int64)
                control = oracle.quantizer.dequantize(indices)
                witness(f"quantizer/{i}", control, discrete._simulator.envs[0].env.quantizer.dequantize(indices))
                check_step(f"discrete_equivalence/{i}", oracle, discrete, indices)
                result = continuous.step(control[None], frames=None)
                witness(
                    f"continuous_equivalence/{i}/obs",
                    oracle._get_obs(),
                    result["state"]["current", "observation"][0].numpy(),
                )
                equal(diagnostics(oracle), diagnostics(continuous._simulator.envs[0].env))
                witness(f"continuous_equivalence/{i}/frame", oracle.render(), continuous.render()[0].numpy())
                task = oracle.eval_state(oracle.goal_state, oracle._get_obs())
                witness(
                    f"continuous_equivalence/{i}/task",
                    [-task[1], task[0], False],
                    [result[k][0].item() for k in ("reward", "terminated", "truncated")],
                )
            records.append(
                dict(registration="swm/PushT-Discrete-v1", gate="all_polar_bins_and_continuous_equivalence", cases=256)
            )
    finally:
        oracle.close()
    np.savez_compressed(args.output / "witnesses.npz", **arrays)
    report = dict(
        status="PASS_PUBLIC_ACCEPTANCE_EDGES",
        device="cpu",
        source_never_restored=True,
        records=records,
        raw_comparisons=len(arrays) // 2,
        artifact="witnesses.npz",
        artifact_sha256=sha(args.output / "witnesses.npz"),
        source_files=profile["source_files"],
        production_files=profile["production_files"],
        script_sha256=sha(Path(__file__)),
        limitations=[
            "Public actions validate the declared action domain; out-of-domain source behavior is not claimed.",
            "Nondefault source-only constructor damping/block_cog/relative=False are not exposed.",
            "No accelerated GPU rendering claim.",
        ],
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["raw_comparisons"], flush=True)


if __name__ == "__main__":
    main()
