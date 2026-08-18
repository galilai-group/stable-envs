"""Run an UNMODIFIED ``stable_worldmodel`` workflow against the stable-envs renderer.

The claim being tested is not "the adapter has the right methods" — that is a compile-time property
and it proves nothing. It is that a workflow written against ``stable_worldmodel``, using its
``World``, its wrapper stack and a policy out of its own ``policy.py``, runs against our renderer
with the caller unchanged, and that what it sees is the same shape and the same key set it would have
seen from the official env.

Four arms, all driving the same ``swm.World`` / ``swm.policy.RandomPolicy`` / ``world.evaluate``
sequence and differing only in what backs the environments:

    A  official        swm/TwoRoom-v1                  the baseline
    B  facade          stable_envs/TwoRoom-v0          one string changes; N render calls
    C  batched pool    B + attach_batched_pool(world)  one extra line; ONE render call per step
    D  renderer swap   swm/PushT-v1, render() only     official dynamics, our pixels, pixel-compared

Arm D is the one that can be pixel-compared. Arms A and B do not visit the same states (different
simulators, different reset draws), so comparing their frames would be comparing two different
episodes; arm D holds everything but the renderer fixed, which is the only way a pixel number means
anything.

Run on a GPU node:

    python examples/swm_compat/run_swm_workflow.py --out examples/swm_compat/results
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import stable_envs as se
from stable_envs.compat import worldmodel as wm

STEPS = 20
NUM_ENVS = 8


def _keys(infos) -> list[str]:
    return sorted(str(k) for k in infos)


def _shapes(infos) -> dict:
    out = {}
    for k, v in infos.items():
        out[str(k)] = list(np.asarray(v).shape) if hasattr(v, "shape") else type(v).__name__
    return out


def drive(world, policy, steps: int, seed: int = 0) -> dict:
    """The caller's loop. Identical in every arm; only what backs ``world`` differs.

    Deliberately written against the public surface — ``set_policy``, ``reset``, and the pool's
    ``step`` — rather than calling ``World._run_iter``, so that "unchanged caller" means a loop
    somebody would actually write.
    """
    world.set_policy(policy)
    world.reset(seed=seed)
    infos = world.infos
    first = {"keys": _keys(infos), "shapes": _shapes(infos)}
    t0 = time.perf_counter()
    for _ in range(steps):
        actions = policy.get_action(world.infos)
        _, _, _, _, world.infos = world.envs.step(np.asarray(actions))
    dt = time.perf_counter() - t0
    return {"first_info": first, "last_keys": _keys(world.infos), "steps": steps,
            "seconds": dt, "env_steps_per_s": steps * world.num_envs / dt,
            "pixels_shape": list(np.asarray(world.infos["pixels"]).shape)}


def arm_official(swm, num_envs: int, steps: int) -> dict:
    world = swm.World("swm/TwoRoom-v1", num_envs=num_envs, max_episode_steps=1000,
                      image_shape=(224, 224))
    try:
        return drive(world, swm.policy.RandomPolicy(), steps)
    finally:
        world.close()


def arm_facade(swm, num_envs: int, steps: int) -> dict:
    """Arm B. The ONLY difference from arm A is the env id string."""
    world = swm.World("stable_envs/TwoRoom-v0", num_envs=num_envs, max_episode_steps=1000,
                      image_shape=(224, 224))
    try:
        return drive(world, swm.policy.RandomPolicy(), steps)
    finally:
        world.close()


def arm_batched(swm, num_envs: int, steps: int) -> dict:
    """Arm C. Arm B plus one line, and the render becomes a single launch for the whole world."""
    world = swm.World("stable_envs/TwoRoom-v0", num_envs=num_envs, max_episode_steps=1000,
                      image_shape=(224, 224))
    try:
        wm.attach_batched_pool(world)
        return drive(world, swm.policy.RandomPolicy(), steps)
    finally:
        world.close()


def _render_at(env, state):
    """Drive the OFFICIAL env to a state and call its (swapped) render, as a live workflow would."""
    u = env.unwrapped
    u.agent.position = [float(state[0]), float(state[1])]
    u.block.angle = float(state[4])
    u.block.position = [float(state[2]), float(state[3])]
    u.space.reindex_shapes_for_body(u.block)
    u.space.reindex_shapes_for_body(u.agent)
    return u.render()


def arm_renderer_swap(steps: int = 40, size: int = 224, seed: int = 0) -> dict:
    """Arm D: the official ``swm/PushT-v1``, stepped by pymunk, rendered by us.

    Both arms are the SAME env object driven through the same action sequence, so the states are
    identical by construction and the frames are the only variable. That is what makes the pixel
    numbers here comparable to the repo's other parity tables.
    """
    import gymnasium
    import stable_worldmodel  # noqa: F401
    from stable_envs.testing import conformance as C

    env = gymnasium.make("swm/PushT-v1", resolution=size)
    u = env.unwrapped
    env.reset(seed=seed, options={"variation": []})
    rng = np.random.default_rng(seed)

    official, states = [], []
    for _ in range(steps):
        env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        states.append(wm.pusht_state(u))
        official.append(np.ascontiguousarray(u.render()))
    official = np.stack(official)
    states = np.stack(states)

    # Replay the recorded states through the swapped renderer rather than re-stepping: the official
    # env has already moved on, and stepping it again would be a second, different rollout. The swap
    # itself is still exercised — it is what a live workflow calls — and the frames it produces on
    # those states are the ones compared.
    wm.swap_renderer(env, "PushT", wm.pusht_state, size=size)
    swapped = np.stack([np.ascontiguousarray(_render_at(env, st)) for st in states[:4]])
    r = se.make("PushT", batch=len(states), size=size, device="cuda")
    ours = r.render(states.astype(np.float32)).cpu().numpy()
    swap_matches = int(np.abs(swapped.astype(np.int16) - ours[:4].astype(np.int16)).max())
    env.close()

    agree = C.pixel_agreement(ours, official)
    return {"steps": steps, "size": size, "agreement": agree,
            "swap_hook_max_abs_diff": swap_matches,
            "state_dim": int(states.shape[1]),
            "bytes_per_step_states": int(states.shape[1] * 4),
            "bytes_per_step_frame": int(np.prod(official.shape[1:]))}


def arm_scaling(swm, steps: int = 10) -> dict:
    """Where the batched pool actually pays: env-steps/s against the world's size.

    The facade and the official envs are both O(N) Python calls into a CPU renderer, so their
    throughput per environment is flat and their total is linear. The batched pool issues ONE render
    launch whatever N is, so its total climbs until the launch stops being the cost. Three sizes are
    enough to show which curve each one is on; more would only cost official-env construction time.
    """
    out = {}
    for n in (8, 32, 128):
        row = {}
        for arm, fn in (("official", arm_official), ("facade", arm_facade),
                        ("batched", arm_batched)):
            try:
                row[arm] = fn(swm, n, steps)["env_steps_per_s"]
            except Exception as e:                                          # noqa: BLE001
                row[arm] = f"ERR {type(e).__name__}: {e}"
        if all(isinstance(v, float) for v in row.values()):
            row["batched_vs_official"] = row["batched"] / row["official"]
        out[f"num_envs={n}"] = row
        print(" ", n, row, flush=True)
    return out


def arm_variation_mapping() -> dict:
    out = {}
    for name in ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones", "PushT"):
        if name not in se.REGISTRY:
            continue
        env = se.make(name, batch=1, device="cuda")
        out[name] = wm.check_mapping(env)
    return out


def arm_config_drives_our_renderer() -> dict:
    """An existing swm variation config, unchanged, driving our renderer.

    The end-to-end statement the mapping exists for: take the variation values off a LIVE official
    env, convert them once, render with them, and check that the official env and ours agree on the
    frame they describe.
    """
    import gymnasium
    import stable_worldmodel  # noqa: F401
    from stable_envs.testing import conformance as C

    env = se.make("TwoRoom", batch=8, device="cuda")
    ref = gymnasium.make("swm/TwoRoom-v1")
    u = ref.unwrapped
    # Randomise exactly the leaves the map CLAIMS to cover, all at once. Not ``['all']``: that also
    # resamples ``rendering.render_target``, which selects a different one of our classes
    # (TwoRoomTarget), so a disagreement would be measuring the wrong claim rather than the mapping.
    covered = wm.check_mapping(env)["covered"]
    ref.reset(seed=3, options={"variation": covered})
    style = wm.style_from_swm_env(env, ref)

    states = env.sample_states(8, rng=np.random.default_rng(7))
    official = []
    import torch
    for st in states:
        u.agent_position = torch.as_tensor(st[:2], dtype=torch.float32)
        official.append(np.ascontiguousarray(u.render()))
    official = np.stack(official)
    ours = env.render(states, style=np.tile(style, (8, 1))).cpu().numpy()
    default = env.render(states).cpu().numpy()
    ref.close()
    return {"randomised_leaves": covered,
            "agreement_with_swm_config": C.pixel_agreement(ours, official),
            "agreement_if_config_ignored": C.pixel_agreement(default, official),
            "swm_variation_values": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                     for k, v in wm._flatten(u.variation_space.value).items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/swm_compat/results")
    ap.add_argument("--num-envs", type=int, default=NUM_ENVS)
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()

    import stable_worldmodel as swm

    wm.register_gym_ids()
    report = {}
    arms = {
        "A_official": lambda: arm_official(swm, args.num_envs, args.steps),
        "B_facade": lambda: arm_facade(swm, args.num_envs, args.steps),
        "C_batched_pool": lambda: arm_batched(swm, args.num_envs, args.steps),
        "D_renderer_swap": lambda: arm_renderer_swap(),
        "E_variation_mapping": arm_variation_mapping,
        "F_swm_config_drives_us": arm_config_drives_our_renderer,
        "G_scaling": lambda: arm_scaling(swm),
    }
    for name, fn in arms.items():
        print(f"\n===== {name}", flush=True)
        try:
            report[name] = fn()
            print(json.dumps(report[name], indent=1, default=str)[:4000], flush=True)
        except Exception as e:                                              # noqa: BLE001
            import traceback
            report[name] = {"error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()}
            print(report[name]["traceback"], flush=True)

    ok = [a for a in ("A_official", "B_facade", "C_batched_pool") if "error" not in report[a]]
    if len(ok) == 3:
        a, b, c = (report[k] for k in ("A_official", "B_facade", "C_batched_pool"))
        report["key_parity"] = {
            "official_keys": a["first_info"]["keys"],
            "facade_keys": b["first_info"]["keys"],
            "batched_keys": c["first_info"]["keys"],
            "facade_missing": sorted(set(a["first_info"]["keys"]) - set(b["first_info"]["keys"])),
            "batched_missing": sorted(set(a["first_info"]["keys"]) - set(c["first_info"]["keys"])),
        }
        report["throughput"] = {k: report[k]["env_steps_per_s"] for k in ok}
        print("\n", json.dumps(report["key_parity"], indent=1), flush=True)
        print(json.dumps(report["throughput"], indent=1), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "swm_workflow.json", "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("\nWROTE", out / "swm_workflow.json")


if __name__ == "__main__":
    main()
