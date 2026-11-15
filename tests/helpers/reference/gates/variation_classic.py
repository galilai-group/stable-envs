"""Classic-control physics Variation leaves against pinned stable_worldmodel."""

from __future__ import annotations

import numpy as np
import torch

import stable_envs as se
from stable_envs.envs.classic_control.simulation.local import _PHYSICS_SPACES
from tests.helpers.reference.native_acceptance import CASES as CLASSIC_CASES
from tests.helpers.reference.native_acceptance import _actions as _classic_actions
from tests.helpers.reference.native_acceptance import compare_state as _classic_compare
from tests.helpers.reference.native_classic_semantics import NativeClassicReference

#: Start velocity that sits above every ``max_speed`` the leaf can take, so the clip decides the rollout.
_FAST_FIELD = {"pendulum": "theta_dot", "mountain_car": "velocity", "mountain_car_continuous": "velocity"}
_FAST_SPEED = {"pendulum": 14.0, "mountain_car": 0.2, "mountain_car_continuous": 0.2}


def classic_physics_gate(domain: str, seed: int = 13, steps: int = 24) -> dict:
    """Every physics leaf of a classic-control cell, explicit and sampled, against stable-worldmodel.

    Lanes sit at 10 %, 50 % and 90 % of each leaf's range; State and the whole step output
    (reward, termination) must equal stable-worldmodel's exactly. Where the cell's frame draws physics leaves
    (CartPole's pole length, Acrobot's link lengths), every reset and step frame must equal stable-worldmodel's
    too, as worst-frame normalized MSE 0. A plain reset then restores the defaults.
    """
    batch = 3
    evidence = {"domain": domain, "seed": seed, "steps": steps, "leaves": {}}
    actions = _classic_actions(domain, seed, steps, batch)
    leaves = {f"physics.{name}": (lo, hi) for name, (lo, hi, _) in _PHYSICS_SPACES[domain].items()}

    def frame_mse(env, ref):
        ours = env._render().cpu().numpy().astype(np.float64) / 255
        want = np.stack([wrapper.render() for wrapper in ref.envs]).astype(np.float64) / 255
        return float(((ours - want) ** 2).mean(axis=(1, 2, 3)).max())

    def run(env, options, plain=None, fast=False, frames=False):
        ref = NativeClassicReference(domain, batch, options=options, render_mode="rgb_array" if frames else None)
        worst = [0.0]

        def framed(got, want):
            _classic_compare(got, want)
            if frames:
                worst.append(frame_mse(env, ref))
            return got

        try:
            framed(env.reset(seed=seed, options=plain), ref.reset(seed))
            if fast:
                start = ref.get_state()
                start[_FAST_FIELD[domain]].fill_(_FAST_SPEED[domain])
                env.set_state(start)
                ref.set_state(start)
            states = [
                framed(got, want)["state"]
                for got, want in zip(
                    (env.step(a, states=None, frames=None) for a in actions),
                    (ref.step(a) for a in actions),
                    strict=True,
                )
            ]
            assert max(worst) == 0.0, f"{domain} frames differ from SWM: worst-frame MSE {max(worst)}"
            return states, max(worst)
        finally:
            ref.close()

    with se.make(CLASSIC_CASES[domain]["stable_env"], num_envs=batch, device="cpu") as env:
        assert set(leaves) == set(env.dynamics_variation_leaves)
        drawn_leaves = set(env.geometry_leaves)
        draws = bool(drawn_leaves)
        baselines = {fast: run(env, None, fast=fast, frames=draws)[0][-1] for fast in (False, domain in _FAST_FIELD)}
        for leaf, (lo, hi) in leaves.items():
            values = np.asarray([[lo + f * (hi - lo)] for f in (0.1, 0.5, 0.9)], np.float32)
            per_lane = [{"variation_values": {leaf: v}} for v in values]
            fast = leaf.endswith("max_speed") and domain in _FAST_FIELD
            states, worst = run(
                env, per_lane, {"variation_values": {leaf: values}}, fast=fast, frames=leaf in drawn_leaves
            )
            last, baseline = states[-1], baselines[fast]
            moved = [
                any(
                    not torch.equal(last[k][lane], baseline[k][lane])
                    for k in last.keys(True, True)
                    if k[0] not in ("rng", "physics")
                )
                for lane in (0, 2)
            ]
            evidence["leaves"][leaf] = {"values": values.reshape(-1).tolist(), "moved_vs_default": moved}
            if leaf in drawn_leaves:
                evidence["leaves"][leaf]["frame_worst_mse"] = worst
        drawn = env.reset(seed=seed + 1, options={"variation": list(leaves)})
        sampled = {leaf: env._style[leaf].cpu().numpy().reshape(batch, 1) for leaf in leaves}
        for leaf, (lo, hi) in leaves.items():
            assert len(set(sampled[leaf].reshape(-1).tolist())) == batch, leaf
            assert ((sampled[leaf] >= lo) & (sampled[leaf] <= hi)).all(), leaf
        per_lane = [{"variation_values": {leaf: sampled[leaf][lane] for leaf in leaves}} for lane in range(batch)]
        ref = NativeClassicReference(domain, batch, options=per_lane, render_mode="rgb_array" if draws else None)
        try:
            _classic_compare(drawn, ref.reset(seed + 1))
            worst = [frame_mse(env, ref)] if draws else [0.0]
            for a in actions:
                _classic_compare(env.step(a, states=None, frames=None), ref.step(a))
                if draws:
                    worst.append(frame_mse(env, ref))
            assert max(worst) == 0.0, f"{domain} sampled frames differ from SWM: worst-frame MSE {max(worst)}"
        finally:
            ref.close()
        evidence["sampled"] = {leaf: sampled[leaf].reshape(-1).tolist() for leaf in leaves}
        if draws:
            evidence["sampled_frame_worst_mse"] = max(worst)
        run(env, None, None, frames=draws)
        evidence["plain_reset_restores_defaults"] = True
    return evidence
