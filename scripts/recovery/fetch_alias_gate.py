"""Backend-only dense and Dict equivalence; SWM observation integration deferred."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch

import stable_envs.envs.fetch as fetch
from scripts.recovery.fetch_public_gate import compare_output, equal_state, full_compare


def main(task, output):
    results = []
    for seed in (0, 17, 42):
        for batch in (1, 3):
            for dtype in (np.float32, np.float64):
                envs = [getattr(fetch, f"Fetch{task}{suffix}")(batch, seed=seed) for suffix in ("", "Dense", "Dict")]
                refs = [
                    [gym.make(f"swm/Fetch{task}{suffix}-v3") for _ in range(batch)] for suffix in ("", "Dense", "Dict")
                ]
                try:
                    for group in refs:
                        for lane, ref in enumerate(group):
                            ref.reset(seed=seed + lane, options={"variation": []})
                    for env, group in zip(envs, refs):
                        full_compare(env.reset(seed=seed), group)
                        equal_state(env.get_state(), envs[0].get_state())
                        assert torch.equal(
                            env._pack_render_state(env.get_state()), envs[0]._pack_render_state(envs[0].get_state())
                        )
                    actions = np.random.default_rng(seed + 845).uniform(-2, 2, (40, batch, 4)).astype(dtype)
                    reward_differences = 0
                    for step, action in enumerate(actions):
                        if step == 32:
                            snap = envs[0].get_state()
                            for env in envs[1:]:
                                for other in actions[:11]:
                                    env.step(other, capture="states")
                                env.set_state(snap)
                                equal_state(env.get_state(), snap)
                        outs = []
                        for env, group in zip(envs, refs):
                            expected = [ref.step(action[lane]) for lane, ref in enumerate(group)]
                            out = env.step(action, capture="states")
                            compare_output(out, expected)
                            full_compare(out["state"], group)
                            outs.append(out)
                        for out in outs[1:]:
                            equal_state(out["state"], outs[0]["state"])
                            for key in ("success", "terminated", "truncated"):
                                assert torch.equal(out[key], outs[0][key])
                        assert torch.equal(outs[2]["reward"], outs[0]["reward"])
                        for lane, ref in enumerate(refs[1]):
                            u = ref.unwrapped
                            distance = np.linalg.norm(u._get_obs()["achieved_goal"] - u.goal)
                            assert outs[1]["reward"][lane].item() == -distance
                        reward_differences += int(not torch.equal(outs[0]["reward"], outs[1]["reward"]))
                    assert reward_differences
                    for env, group in zip(envs, refs):
                        for ref in group:
                            ref.reset(options={"variation": []})
                        full_compare(env.reset(), group)
                        equal_state(env.get_state(), envs[0].get_state())
                    results.append(
                        dict(
                            seed=seed,
                            batch=batch,
                            dtype=np.dtype(dtype).name,
                            steps=32,
                            different_history=11,
                            restored_steps=8,
                            reset_rng_exact=True,
                            state_success_done_exact=True,
                            sparse_dict_reward_exact=True,
                            dense_reward_exact=True,
                            render_projection_exact=True,
                        )
                    )
                finally:
                    for env in envs:
                        env.close()
                    for group in refs:
                        for ref in group:
                            ref.close()
    output.write_text(
        json.dumps(
            dict(
                status="PASS",
                task=task,
                cases=results,
                observation_integration="DEFERRED",
                observation_container_tested=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print("FETCH_ALIAS_CPU_GATE_PASS", task, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.task, args.output)
