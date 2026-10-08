"""Private PushT replay: independent-source shape, reset and scalar-type cases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path

import numpy as np
from pusht_adapter_gate import ROOT, sha
from pusht_replay_gate import PIN, action, diagnostics, equal, import_source

from stable_envs._adapters.pusht.replay_batch import PushTReplayAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip() == PIN
    classes = import_source(args.source.resolve())
    source_files = {}
    for file in (
        "spaces.py",
        "utils.py",
        "envs/utils.py",
        "envs/quantizer.py",
        "envs/pusht/env.py",
        "envs/pusht/env_discrete.py",
    ):
        relative = "stable_worldmodel/" + file
        assert (args.source / relative).read_bytes() == subprocess.check_output(
            ["git", "-C", str(args.source), "show", PIN + ":" + relative]
        )
        source_files[relative] = sha(args.source / relative)
    report = dict(
        status="RUNNING",
        source_pin=PIN,
        source_files=source_files,
        source_never_restored=True,
        public_acceptance=False,
        script_sha256=sha(Path(__file__)),
        production_files={
            str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "stable_envs/_adapters").glob("pusht_*.py"))
        },
        cases=[],
    )

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for discrete, cls in enumerate(classes):
            variant = "discrete" if discrete else "continuous"
            for agent_shape in range(8):
                for block_shape in range(1, 8):
                    label = f"{variant}-agent{agent_shape}-block{block_shape}"
                    config = dict(resolution=96, render_action=True)
                    oracle = cls(**config)
                    candidate = PushTReplayAdapter(1, seed=17, discrete=bool(discrete), config=config)
                    try:
                        oracle.reset(seed=17)
                        for step in range(10):
                            control = action(oracle, variant, step)
                            oracle.step(control.copy())
                            candidate.step(control[None].copy())
                        # Persist changed initial values; then an ordinary source reset
                        # retains the previous action marker in the goal cache.
                        initial = {"agent.scale": np.float32(35), "block.scale": np.float64(45)}
                        oracle.variation_space.set_init_value(initial)
                        candidate.envs[0].env.variation_space.set_init_value(initial)
                        options = dict(
                            variation=["all"],
                            variation_values={"agent.shape": agent_shape, "block.shape": np.int64(block_shape)},
                        )
                        oracle.reset(seed=42, options=options)
                        candidate.reset(42, options)
                        equal(diagnostics(oracle), diagnostics(candidate.envs[0].env))
                        checkpoint = candidate.get_state()
                        # Force restoration to replace a divergent engine.
                        candidate.step(np.array([[15, 15]], np.int64) if discrete else np.ones((1, 2), np.float64))
                        candidate.set_state(checkpoint)
                        equal(diagnostics(oracle), diagnostics(candidate.envs[0].env))
                        source_frames, replay_frames, source_obs, replay_obs, controls, tasks = [], [], [], [], [], []
                        for step in range(100):
                            control = action(oracle, variant, step)
                            controls.append(control.astype(np.float64))
                            expected = oracle.step(control.copy())
                            actual = candidate.step(control[None].copy())
                            equal(diagnostics(oracle), diagnostics(candidate.envs[0].env))
                            assert float(actual["reward"][0]) == expected[1]
                            assert bool(actual["terminated"][0]) == expected[2]
                            assert bool(actual["truncated"][0]) == expected[3]
                            source_obs.append(expected[0]["state"])
                            replay_obs.append(candidate.envs[0].env._get_obs())
                            tasks.append([expected[1], expected[2], expected[3]])
                            if step % 10 == 0 or step == 99:
                                source_frames.append(oracle.render())
                                replay_frames.append(candidate.envs[0].env.render())
                                equal(source_frames[-1], replay_frames[-1])
                        file = args.output / (label + ".npz")
                        np.savez_compressed(
                            file,
                            controls=controls,
                            source_observations=source_obs,
                            replay_observations=replay_obs,
                            source_frames=source_frames,
                            replay_frames=replay_frames,
                            source_tasks=tasks,
                        )
                        report["cases"].append(
                            dict(
                                variant=variant,
                                agent_shape=agent_shape,
                                block_shape=block_shape,
                                future_steps=100,
                                exact=True,
                                artifact=file.name,
                                artifact_sha256=sha(file),
                            )
                        )
                        save()
                        print(label, "PASS", flush=True)
                    finally:
                        oracle.close()
                        candidate.close()
        report["status"] = "PASS_PRIVATE_REPLAY_SHAPE_AND_RESET_GATE"
    except BaseException:
        report["status"] = "FAIL"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    main()
