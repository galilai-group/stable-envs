"""Scoped entropy-only diagnostic patches; never mock task or physics outputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from gates import compare, digest, plain, run_reset
from ownership import BOX_SAMPLE, CubeEnv, OwnedCube, OwnedScene, SceneEnv, capture, restore

OUT = Path(__file__).resolve().parent


def semantic_view(state):
    state = copy.deepcopy(state)
    for key in ["ownership", "action_box", "last_box"]:
        state.pop(key, None)
    return state


def check(raw_cls, owned_cls, seed):
    candidate, raw = owned_cls(entropy_seed=seed), raw_cls()
    try:
        candidate.reset(seed=seed, options={"variation": [], "task_id": 2})
        raw.reset(seed=seed, options={"variation": [], "task_id": 4})
        anchor = capture(candidate)
        # Record actual authoritative Box draws from the owned candidate.
        natural = run_reset(candidate, {"variation": ["cube.start_position"]})
        tape = natural["settling_actions"]
        selected_task = natural["state"]["python"]["cur_task_id"]
        variation_seed = natural["state"]["ownership"]["variation_seeds"][-1]
        restore(candidate, anchor)
        raw_anchor = copy.deepcopy(anchor)
        raw_anchor["ownership"] = {}
        raw_anchor["action_box"] = None
        raw_anchor["last_box"] = None
        restore(raw, raw_anchor)
        original_var_seed = raw.variation_space.seed
        records = {}
        for label, env in [("candidate", candidate), ("authoritative", raw)]:
            sampled = []
            box_specs = []
            original_sample = gym.spaces.Box.sample

            def fixed_sample(box, *args, **kwargs):
                if type(box) is gym.spaces.Box and box.shape == (5,):
                    action = tape[len(sampled)].copy()
                    sampled.append(action)
                    box_specs.append(
                        {"low": box.low.copy(), "high": box.high.copy(), "shape": box.shape, "dtype": str(box.dtype)}
                    )
                    return action
                return original_sample(box, *args, **kwargs)

            options = {"variation": ["cube.start_position"]}
            if label == "authoritative":
                options["task_id"] = selected_task
            with patch.object(gym.spaces.Box, "sample", fixed_sample):
                if label == "authoritative":
                    # Inject at the raw source's root seeding boundary only.
                    with patch.object(
                        raw.variation_space, "seed", lambda seed=None: original_var_seed(variation_seed)
                    ):
                        output = env.reset(options=options)
                else:
                    output = env.reset(options=options)
            assert len(sampled) == 2
            records[label] = {
                "state": semantic_view(capture(env)),
                "output": copy.deepcopy(output),
                "actions": sampled,
                "box_specs": box_specs,
            }
        assert gym.spaces.Box.sample is BOX_SAMPLE
        diff = compare(records["candidate"], records["authoritative"])
        assert not diff, diff
        # The forced candidate also reproduces its naturally sampled physical result.
        natural_view = {"state": semantic_view(natural["state"]), "output": natural["output"]}
        candidate_view = {k: records["candidate"][k] for k in ["state", "output"]}
        assert not compare(natural_view, candidate_view)
        return {
            "seed": seed,
            "task_id": selected_task,
            "variation_seed": variation_seed,
            "settling_actions": tape,
            "records": records,
            "diff": diff,
            "natural_semantics_sha256": digest(natural_view),
            "forced_semantics_sha256": digest(candidate_view),
            "sample_implementation": "unmodified gymnasium.spaces.Box.sample outside scoped fixed-tape diagnostic",
        }
    finally:
        candidate.close()
        raw.close()


def main():
    results = {}
    for name, raw, candidate in [("Cube", CubeEnv, OwnedCube), ("Scene", SceneEnv, OwnedScene)]:
        results[name] = []
        for seed in [0, 17, 42]:
            results[name].append(check(raw, candidate, seed))
            print(name, seed, "FIXED_TAPE_SEMANTIC_PRESERVATION_PASS", flush=True)
    assert gym.spaces.Box.sample is BOX_SAMPLE
    (OUT / "semantic_preservation.json").write_text(json.dumps(plain(results), indent=2) + "\n")
    print("OGB_FIXED_TAPE_SEMANTIC_PRESERVATION_PASS_PATCHES_REMOVED", flush=True)


if __name__ == "__main__":
    main()
