"""Dump hashes of every public output under forced process counts, to compare two checkouts.

Run the same command in each checkout, with the same interpreter and on the same node, then compare:

    python -m scripts.testing.process_ab run --family dmcontrol --out base.jsonl
    python -m scripts.testing.process_ab run --family dmcontrol --out branch.jsonl
    python -m scripts.testing.process_ab compare base.jsonl branch.jsonl

``--processes 0`` runs the local adapter. A family without process execution in a checkout runs
its local adapter at every process count; the record notes the adapter type.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import traceback

import numpy as np
import torch

EXECUTION = {
    "classic_control": "stable_envs.envs.classic_control.simulation.execution",
    "dmcontrol": "stable_envs.envs.dmcontrol.simulation.execution",
    "fetch": "stable_envs.envs.fetch.simulation.execution",
    "ogbench": "stable_envs.envs.ogbench.simulation.execution",
    "pusht": "stable_envs.envs.pusht.simulation.execution",
}
BATCHES = (1, 3, 8, 17)
CLASSIC_LIMITS = dict(cartpole=500, acrobot=500, pendulum=200, mountain_car=200, mountain_car_continuous=999)
PROCESSES = (0, 1, 2, 8)


def force_processes(count):
    """Make every family's routing return ``count`` workers (capped by lanes) for eligible targets."""
    for name in EXECUTION.values():
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue

        def chosen(num_envs, cpus, *, eligible=True, path=None):
            return min(count, num_envs) if eligible else 0

        module.process_count = chosen
        module.cpu_affinity = lambda: None


def digest(value):
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(np.asarray(value))
    return f"{array.dtype}{list(array.shape)}:{hashlib.sha256(array.tobytes()).hexdigest()[:24]}"


class Record:
    def __init__(self):
        self.items = []

    def add(self, label, value):
        if hasattr(value, "items") and hasattr(value, "batch_size"):
            for key in sorted(value.keys(True, True), key=str):
                self.items.append((label, "/".join(key if isinstance(key, tuple) else (key,)), digest(value[key])))
        else:
            self.items.append((label, "", digest(value)))


def random_actions(env, rng, length):
    spec = env.action_spec
    dtype = torch.empty((), dtype=spec.dtype).numpy().dtype
    shape = (env.num_envs, length, *spec.shape)
    if spec.dtype.is_floating_point:
        low = np.broadcast_to(np.asarray(spec.low, np.float64), spec.shape)
        high = np.broadcast_to(np.asarray(spec.high, np.float64), spec.shape)
        return rng.uniform(low, high, shape).astype(dtype)
    if getattr(spec, "discrete_n", None):
        return rng.integers(0, int(spec.discrete_n), shape).astype(dtype)
    return rng.integers(int(np.min(spec.low)), int(np.max(spec.high)) + 1, shape).astype(dtype)


def finish_some_lanes(env, state):
    """Edit State so some lanes end inside the next chunks, where the family allows it."""
    keys = set(state.keys(True, True))
    state = state.clone()
    lanes = torch.arange(env.num_envs)
    if ("lifecycle", "elapsed_steps") in keys and env.family == "classic_control":
        limit = CLASSIC_LIMITS[env.CLASSIC_DOMAIN]
        state["lifecycle", "elapsed_steps"][:, 0] = limit - 1 - (lanes % 7) * 3
    if ("root", "position") in keys and ("task", "goal_pos") in keys:
        even = lanes[lanes % 2 == 0]
        state["root", "position"][even] = state["task", "goal_pos"][even, :2]
    return state


def masked_steps(env, rng, record, label):
    simulator = env._simulator
    if env.family == "dmcontrol":
        return
    for t in range(4):
        actions = random_actions(env, rng, 1)[:, 0]
        mask = torch.from_numpy(rng.random(env.num_envs) < 0.6)
        mask[0] = True
        out = simulator.step(actions, active=mask)
        record.add(f"{label}/masked{t}", out)


def scenario(name, num_envs, count, seed=17):
    import stable_envs as se

    record = Record()
    rng = np.random.default_rng(seed)
    env = se.make(name, num_envs=num_envs, device="cpu", seed=seed)
    other = None
    try:
        mode = f"{type(env._simulator).__module__}.{type(env._simulator).__name__}"
        record.add("initial", env.get_state())
        record.add("reset", env.reset(seed=seed + 100))
        if env.family == "fetch":
            env._simulator._set_episode_limits(np.arange(num_envs) % 7 + 2)
        else:
            env.set_state(finish_some_lanes(env, env.get_state()))
        actions = [random_actions(env, rng, 10) for _ in range(5)]
        saved = [env.get_state()]
        frames = getattr(env, "produces_rgb", True)
        for chunk, block in enumerate(actions):
            selection = slice(-1, None) if frames and chunk in (0, 4) else None
            out = env.step(block, states=slice(None, None, 3), frames=selection)
            record.add(f"chunk{chunk}", out)
            saved.append(env.get_state())
            record.add(f"get{chunk}", saved[-1])
        masked_steps(env, rng, record, "first")
        env.set_state(saved[2])
        record.add("restored", env.get_state())
        for chunk, block in enumerate(actions[2:], 2):
            record.add(f"again{chunk}", env.step(block, states=slice(None), frames=None))
        other = se.make(name, num_envs=num_envs, device="cpu", seed=seed + 1)
        other.reset(seed=seed + 7)
        other.step(random_actions(other, np.random.default_rng(seed + 2), 5), frames=None)
        foreign = other.get_state()
        env.set_state(foreign)
        record.add("foreign", env.get_state())
        record.add("foreign_step", env.step(random_actions(env, rng, 6), states=slice(None), frames=None))
        masked_steps(env, rng, record, "second")
        if frames:
            record.add("render_live", env.render())
            record.add("render_saved", env.render(saved[1]))
            record.add("render_foreign", env.render(foreign))
        error = None
    except Exception:
        error = traceback.format_exc(limit=4).strip().splitlines()[-1]
        mode = locals().get("mode", "construction")
    finally:
        for owner in (env, other):
            if owner is not None:
                owner.close()
    return mode, error, record.items


def run(args):
    import stable_envs as se

    names = args.classes or se.list_envs(family=args.family)
    with open(args.out, "a") as out:
        for name in names:
            for num_envs in args.batches:
                for count in args.processes:
                    if count > num_envs:
                        continue
                    force_processes(count)
                    start = time.perf_counter()
                    mode, error, items = scenario(name, num_envs, count)
                    row = dict(
                        family=args.family,
                        name=name,
                        num_envs=num_envs,
                        processes=count,
                        mode=mode,
                        error=error,
                        items=items,
                        seconds=round(time.perf_counter() - start, 2),
                    )
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(name, num_envs, count, mode, error or f"{len(items)} items", row["seconds"], flush=True)


def compare(args):
    def load(path):
        rows = {}
        for line in open(path):
            row = json.loads(line)
            rows[(row["name"], row["num_envs"], row["processes"])] = row
        return rows

    base, branch = load(args.base), load(args.branch)
    same = differ = errors = 0
    for key in sorted(set(base) | set(branch), key=str):
        a, b = base.get(key), branch.get(key)
        if a is None or b is None:
            print("MISSING", key, "base" if a is None else "branch")
            differ += 1
            continue
        if a["error"] or b["error"]:
            errors += 1
        if a["items"] == b["items"] and a["error"] == b["error"]:
            same += 1
            continue
        differ += 1
        print("DIFF", key, a["mode"], b["mode"], a["error"], b["error"])
        for x, y in zip(a["items"], b["items"]):
            if x != y:
                print("   ", x, "!=", y)
                break
        if len(a["items"]) != len(b["items"]):
            print("    item counts", len(a["items"]), len(b["items"]))
    print(f"identical={same} different={differ} with_errors={errors} total={same + differ}")
    # Within one checkout, every process count must match the local adapter.
    for label, rows in (("base", base), ("branch", branch)):
        mismatched = [key for key, row in rows.items() if row["items"] != rows.get((key[0], key[1], 0), row)["items"]]
        print(f"{label}: configs differing from processes=0: {len(mismatched)}", mismatched[:10])
    return 1 if differ else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--family", required=True, choices=sorted(EXECUTION))
    runner.add_argument("--classes", nargs="*")
    runner.add_argument("--batches", nargs="*", type=int, default=list(BATCHES))
    runner.add_argument("--processes", nargs="*", type=int, default=list(PROCESSES))
    runner.add_argument("--out", required=True)
    checker = sub.add_parser("compare")
    checker.add_argument("base")
    checker.add_argument("branch")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
