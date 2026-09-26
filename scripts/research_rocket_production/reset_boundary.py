"""Pinned source reset-entropy/goal reconstruction evidence; not production acceptance."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from stable_worldmodel.envs.rocket_landing.pyflyt_rocketlanding import RocketLandingEnv


def walk(space, path=""):
    if hasattr(space, "spaces"):
        for name, child in space.spaces.items():
            yield from walk(child, path + "." + name if path else name)
    else:
        yield path, space


def capture(env):
    leaves = dict(walk(env.variation_space))
    return dict(
        obs=env.state.copy(),
        goal=env.current_goal.copy(),
        values={k: np.asarray(v.value).copy() for k, v in leaves.items()},
        rng={k: json.dumps(v.np_random.bit_generator.state, sort_keys=True) for k, v in leaves.items()},
        main=json.dumps(env.np_random.bit_generator.state, sort_keys=True),
    )


def same(a, b):
    return (
        np.array_equal(a["obs"], b["obs"])
        and np.array_equal(a["goal"], b["goal"])
        and a["rng"] == b["rng"]
        and a["main"] == b["main"]
        and all(np.array_equal(v, b["values"][k]) for k, v in a["values"].items())
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(inspect.getfile(RocketLandingEnv))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha == "5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23"
    records = []
    witness = {}
    for mode in [None, "rgb_array"]:
        e = RocketLandingEnv(render_mode=mode)
        try:
            for seed in [0, 17, 42]:
                e.reset(seed=seed, options={"variation": ["all"]})
                a = capture(e)
                e.step(np.array([0.1, -0.2, 0.3, 1, 0.5, 0.1, -0.1]))
                e.reset(seed=seed, options={"variation": ["all"]})
                b = capture(e)
                assert same(a, b)
                rng = e.np_random
                assert all(o.np_random is rng for o in [e.env, e.env.drones[0], e.env.drones[0].boosters])
                e.reset(seed=None, options={"variation": ["all"]})
                c = capture(e)
                assert e.np_random is rng
                e.reset(seed=None, options={"variation": ["all"]})
                d = capture(e)
                assert e.np_random is rng
                changed = [k for k in c["values"] if not np.array_equal(c["values"][k], d["values"][k])]
                fresh = [k for k in c["rng"] if c["rng"][k] != d["rng"][k]]
                assert len(changed) == len(fresh) == 9
                for label, s in [("unseeded_a", c), ("unseeded_b", d)]:
                    for name, val in s["values"].items():
                        witness[f"{mode}/{seed}/{label}/{name}"] = val
                records.append(
                    dict(
                        mode=mode,
                        seed=seed,
                        seeded_reset_exact=True,
                        goal_reset_reconstructed_exact=True,
                        goal_shape=list(a["goal"].shape),
                        goal_bytes=a["goal"].nbytes,
                        goal_sha256=hashlib.sha256(a["goal"].tobytes()).hexdigest(),
                        owned_main_rng_identity_preserved=True,
                        fresh_unseeded_value_paths=changed,
                        fresh_unseeded_rng_paths=fresh,
                    )
                )
        finally:
            e.close()
            if e.modified_pad_urdf:
                Path(e.modified_pad_urdf).unlink(missing_ok=True)
    np.savez_compressed(args.output / "unseeded-values.npz", **witness)
    result = dict(
        status="SOURCE_RESET_BOUNDARY_CONFIRMED",
        classification="SEEDED_REPLAYABLE_FRESH_UNSEEDED_ENTROPY",
        source_sha256=sha,
        records=records,
        scope="Source-only; all-selected nine leaves at registered resolution. No production State or complete renderer/Variation acceptance. Does not justify dropping goal cache in unaudited cases.",
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n" for p in sorted(args.output.iterdir())
        )
    )
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
