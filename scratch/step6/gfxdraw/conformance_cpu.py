"""The repository's own conformance checks, run on CPU for the five gfxdraw configurations.

``tests/gpu/test_conformance.py`` requires CUDA by design, so on the laptop the invariants a canvas
rewrite is most likely to break -- batch invariance, batch-SIZE invariance, chunking, order
independence, render purity, style-as-state, per-environment variation, backend agreement -- have no
runner. This is that runner, calling the same functions with the same arguments the GPU suite uses.
It is not a substitute for the GPU suite; it is the loop that catches a mistake in seconds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CHECKS_NO_STATES = ("check_batch_invariance", "check_variations_do_something",
                    "check_per_env_variation", "check_style_is_state",
                    "check_default_style_is_official", "check_colors_apply_exactly",
                    "check_determinism", "check_state_roundtrip", "check_state_dims_matter",
                    "check_set_state_copies", "check_backend_agreement",
                    "check_visual_fidelity", "check_visual_fidelity_under_variation")
CHECKS_STATES = ("check_batch_size_invariance", "check_chunked_render_matches",
                 "check_order_independence", "check_render_is_pure",
                 "check_input_form_invariance")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs",
                    default="Acrobot,CartPole,MountainCar,MountainCarContinuous,Pendulum")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    import stable_envs as se
    from stable_envs.testing import conformance as C

    print(f"stable_envs from {se.__file__}", flush=True)
    bad = 0
    for name in a.configs.split(","):
        states = np.load(ROOT / f"examples/results/step4/states/{name}.npz")["states"][:a.batch]
        line = []
        for fn in CHECKS_NO_STATES + CHECKS_STATES:
            env = se.make(name, batch=a.batch, device=a.device)
            env.rng = np.random.default_rng(0)
            try:
                r = (getattr(C, fn)(env) if fn in CHECKS_NO_STATES
                     else getattr(C, fn)(env, states))
                ok = r.get("status", "measured") not in ("error", "fail") and r.get("ok", True)
                # the checks report their own verdict under a variety of keys; treat any explicit
                # False as a failure rather than guessing which key a given check uses
                for k, v in r.items():
                    if isinstance(v, bool) and v is False and k not in ("blocked", "state_tensor_rebound"):
                        ok = False
                        line.append(f"{fn}.{k}=False")
                if not ok and not line:
                    line.append(f"{fn}={r}")
            except NotImplementedError:
                continue
            except Exception as e:                                        # noqa: BLE001
                line.append(f"{fn}: {type(e).__name__}: {e}")
        bad += bool(line)
        print(f"[{name:22s}] {'OK' if not line else 'FAIL'}")
        for m in line:
            print(f"     {m[:300]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
