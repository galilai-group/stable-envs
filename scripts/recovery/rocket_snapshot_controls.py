"""Source-only controls isolating Rocket engine snapshot failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rocket_feasibility import RocketLandingEnv, capture, differences, probe, restore, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    burn = np.array([0.15, -0.1, 0.2, 1.0, 0.5, 0.1, -0.1])
    zero = np.zeros(7)
    result = {}
    a, b = RocketLandingEnv(), RocketLandingEnv()
    try:
        a.reset(seed=17)
        b.reset(seed=17)
        first, second = trace(a, [burn] * 12), trace(b, [burn] * 12)
        result["clean_independent_same_seed_actions"] = differences(first, second)
        assert not result["clean_independent_same_seed_actions"]
        state = capture(a)
        filename = args.output / "thrust.bullet"
        a.env.saveBullet(str(filename))
        reference = trace(a, [burn, zero] * 4)
        restore(b, state, "file", filename=filename)
        restored = capture(b)
        continuation = trace(b, [burn, zero] * 4)
        result["independent_official_file_restore"] = {
            "immediate_differences": differences(state, restored),
            "trajectory_differences": differences(reference, continuation),
        }
        np.savez_compressed(args.output / "independent-reference.npz", **reference)
        np.savez_compressed(args.output / "independent-restored.npz", **continuation)
        # At the exact source checkpoint, restore immediately without any intervening history.
        a.reset(seed=17)
        trace(a, [burn] * 12)
        state = capture(a)
        sid = a.env.saveState()
        before = state["diagnostic/cached_links"].copy()
        restore(a, state, "memory", state_id=sid)
        after = capture(a)
        result["immediate_official_restore"] = differences(state, after)
        np.savez_compressed(
            args.output / "immediate-cache-change.npz", before=before, after=after["diagnostic/cached_links"]
        )
        a.env.removeState(sid)
    finally:
        a.close()
        b.close()
    # A bounded constructor-supported low-altitude fixture, reached by actual gravity steps.
    # This does not claim registered-default successful-landing policy coverage.
    e = RocketLandingEnv(ceiling=3.2)
    try:
        e.reset(seed=17, options={"accelerate_drop": False})
        steps = 0
        while not (e.termination or e.truncation) and steps < 200:
            e.step(zero)
            steps += 1
        assert e.info["env_complete"] and e.truncation and not e.termination
        result["successful_landing"] = {
            "constructor_ceiling": 3.2,
            "reset_options": {"accelerate_drop": False},
            "steps": steps,
            "info": {k: v for k, v in e.info.items() if k != "goal"},
            "height": float(e.lin_pos[2]),
            "fixture_scope": "source-supported alternate ceiling, actual dynamics; not registered-default acceptance",
        }
        result["landing_contact_replay"] = probe(e, "landing_contact", args.output, [zero] * 8, raw=True)
    finally:
        e.close()
    e = RocketLandingEnv(render_mode="rgb_array")
    try:
        e.reset(seed=17)
        trace(e, [burn] * 12)
        state = capture(e)
        sid = e.env.saveState()
        frame = e.render().copy()
        assert not differences(state, capture(e))
        restore(e, state, "memory", state_id=sid)
        restored = e.render().copy()
        np.savez_compressed(args.output / "source-frame-restore.npz", before=frame, after=restored)
        result["official_restore_immediate_pixels"] = {
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "changed_channels": int(np.count_nonzero(frame != restored)),
            "max_abs": int(np.max(np.abs(frame.astype(int) - restored.astype(int)))),
        }
        e.env.removeState(sid)
    finally:
        e.close()
    result["status"] = "BLOCKED_EXACT_ENGINE_STATE"
    result["public_state_or_adapter_implemented"] = False
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
