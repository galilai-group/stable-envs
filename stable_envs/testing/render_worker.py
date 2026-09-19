"""Render the registry in a FRESH process and report digests. The other half of the repo's claim.

Everything else in the test suite runs inside one interpreter, and a renderer can be perfectly
self-consistent there and still not reproduce anything. The user the repo is written for does the
opposite: they delete 46 GB of frames today, and regenerate them next month, on another machine, from
a different process, out of a loader that hands the states back in a different order and a different
batch size. "Deterministic" has to mean *that*, and nothing inside a single process can test it.

So this module is a small program rather than a function. It builds every registered env, draws states
and style from the SEEDED API, renders, and writes:

    <out_npz>    the exact float32 states and style it used, per env
    <out_json>   sha256 of those arrays, and sha256 of the frames they produced

The parent process then does two independent things with that, and keeping them apart is the point:

  * it re-renders the arrays out of the npz and compares frame digests. This is unconditional — it
    holds whether or not the seeded sampler is reproducible, because the states came over the wire.
  * it re-draws states from the same seeds and compares array digests. This is the seeded API's own
    claim, and it is stated separately because a seeded API that does not reproduce is worse than an
    unseeded one: the caller believes they have a handle they do not have.

Run:  python -m stable_envs.testing.render_worker --out-npz S.npz --out-json S.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

#: Seeds are constants of the protocol, not arguments, so the parent and the worker cannot drift apart
#: by one of them being invoked with a default the other does not share.
STATE_SEED = 20260809
STYLE_SEED = 20260810
ROLLOUT_SEED = 7
ROLLOUT_STEPS = 4


def _digest(a) -> str:
    from .conformance import frames_digest

    return frames_digest(a)


def sample_inputs(env, n: int):
    """The states and style both sides of the process boundary must agree on. One definition, here.

    Written once and imported by the parent test rather than duplicated there: two copies of "how the
    inputs are drawn" is how a cross-process check quietly becomes a comparison of two different
    experiments that both pass.
    """
    states = np.asarray(env.sample_states(n, rng=np.random.default_rng(STATE_SEED)), np.float32)
    style = (
        np.asarray(env.sample_style(n, rng=np.random.default_rng(STYLE_SEED)), np.float32)
        if env.style_dim
        else np.zeros((n, 0), np.float32)
    )
    return states, style


def rollout_digests(env, n: int) -> dict:
    """``env.sample(seed=)`` digests, for the envs that ship a batched simulator.

    A rollout is the one API here that consumes randomness at every step rather than once, so it is
    where a seed leaks: an unseeded global generator, a policy that reseeds on ``set_env`` but not on
    ``reset``, a CUDA generator whose default device changed. Four steps is enough — a seed that is
    going to differ differs at step one.
    """
    from stable_envs.dynamics import has_dynamics

    if not has_dynamics(env):
        return {}
    tr = env.sample(steps=ROLLOUT_STEPS, batch=n, seed=ROLLOUT_SEED)
    out = {"rollout_states_sha": _digest(tr["states"].detach().cpu().numpy())}
    if tr.get("actions") is not None:
        out["rollout_actions_sha"] = _digest(tr["actions"].detach().cpu().numpy())
    return out


def run(names, n: int, device: str = "cuda") -> tuple[dict, dict]:
    import stable_envs as se

    report, arrays = {}, {}
    for name in names:
        try:
            env = se.make(name, num_envs=n, device=device)
            states, style = sample_inputs(env, n)
            sty = style if env.style_dim else None
            frames = env.render(states)
            frames_styled = env.render(states, style=sty)
            rec = {
                "states_sha": _digest(states),
                "style_sha": _digest(style),
                "frames_sha": _digest(frames),
                "frames_styled_sha": _digest(frames_styled),
                "shape": list(frames.shape),
                "size": int(env.size),
                "state_dim": int(env.state_dim),
                "style_dim": int(env.style_dim),
            }
            rec.update(rollout_digests(env, n))
            report[name] = rec
            arrays[f"{name}::states"] = states
            arrays[f"{name}::style"] = style
        except Exception:  # noqa: BLE001
            # One env that cannot be built must not cost the other 43 their cross-process check, and a
            # silent omission would read to the parent as "this env has no result" rather than as a
            # failure. The traceback travels back so the skip message names a cause.
            report[name] = {"error": traceback.format_exc(limit=6)}
        finally:
            # ONE LINE PER ENV, FLUSHED. The parent streams these and turns each into an event, which
            # is what keeps the supervisor's stall watchdog satisfied while this worker does
            # legitimate work. Without it a large registry LIVELOCKS: the watchdog fires at 900s of
            # no test starting or finishing, the segment is killed, the resumed segment restarts this
            # worker from the beginning and is killed again. Measured at 12 segments and 4h56m with
            # zero progress on 191 envs, where the same worker finished comfortably at 64.
            print(f"render_worker: done {name}", flush=True)
    return report, arrays


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--envs", default="", help="comma-separated; default every registered env")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    import stable_envs as se

    names = [s for s in a.envs.split(",") if s] or sorted(se.registry.REGISTRY)
    report, arrays = run(names, a.n, a.device)
    np.savez(a.out_npz, **arrays)
    with open(a.out_json, "w") as f:
        json.dump(
            {"envs": report, "n": a.n, "device": a.device, "state_seed": STATE_SEED, "style_seed": STYLE_SEED}, f
        )
    n_err = sum("error" in v for v in report.values())
    print(f"render_worker: {len(report) - n_err}/{len(report)} envs rendered, {n_err} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
