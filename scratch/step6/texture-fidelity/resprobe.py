"""check_resolution_consistency under the BOUND'S OWN protocol, not the frozen corpus.

DOWNSAMPLE_EXACT_FRAC (tests/gpu/test_reproducibility.py) was measured on "8 states per env from
sample_states(rng=default_rng(render_worker.STATE_SEED)), default appearance" -- i.e. the
tests' own ``inputs`` fixture -- on an H200 under mujoco 3.9. Step 4 applied it to the 64 frozen
corpus states on a 3090 under 3.12. The check never touches the reference renderer, so a failure is
either our renderer or the bound's provenance. This reproduces the fixture's draw exactly
(``render_worker.sample_inputs(env, N)``, N = N = 8 as in the test module) so that the two
state sets can be told apart from the two MuJoCo versions.

    python scratch/step6/texture-fidelity/resprobe.py --configs A,B --states-dir ... --out file.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
N = N = 8                      # tests/gpu/test_reproducibility.py::N, the fixture's batch


def _bounds():
    """The bound constants read out of the test module's TEXT: importing it needs pytest, which the
    3.11 arm's interpreter does not carry, and the numbers are literals."""
    import re
    src = (ROOT / "tests" / "gpu" / "test_reproducibility.py").read_text()
    table = {m.group(1): float(m.group(2))
             for m in re.finditer(r'^\s*"(\w+)":\s*([0-9.]+),', src, re.M)}
    margin = float(re.search(r"^DOWNSAMPLE_MARGIN\s*=\s*([0-9.]+)", src, re.M).group(1))
    on = re.search(r'MEASURED_ON = \(("[^"]*"\s*)+\)', src, re.S).group(0)
    on = " ".join(re.findall(r'"([^"]*)"', on))
    return table, margin, on


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--states-dir", default=str(ROOT / "examples" / "results" / "step4" / "states"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mujoco
    import stable_envs as se
    from stable_envs.testing import conformance as C
    from stable_envs.testing import render_worker as W
    DOWNSAMPLE_EXACT_FRAC, DOWNSAMPLE_MARGIN, MEASURED_ON = _bounds()
    doc = {"mujoco": mujoco.__version__, "N": N, "state_seed": W.STATE_SEED,
           "bound_measured_on": MEASURED_ON, "margin": DOWNSAMPLE_MARGIN, "configs": {}}
    for cfg in a.configs.split(","):
        env = se.make(cfg, batch=N, device="cuda")
        if env.style_dim:
            env.set_style(None)
        states, _ = W.sample_inputs(env, N)
        r_fixture = C.check_resolution_consistency(env, states)
        z = np.load(Path(a.states_dir) / f"{cfg}.npz", allow_pickle=False)
        frozen = np.ascontiguousarray(z["states"], np.float32)
        r_frozen = C.check_resolution_consistency(env, frozen)
        # and the frozen corpus in batches of N, to separate "which states" from "how many"
        r_frozen_first = C.check_resolution_consistency(env, frozen[:N])
        b = DOWNSAMPLE_EXACT_FRAC[cfg]
        doc["configs"][cfg] = {
            "bound": b, "floor": b - DOWNSAMPLE_MARGIN,
            "fixture_protocol": r_fixture, "fixture_pass": bool(r_fixture["exact_frac"] >= b - DOWNSAMPLE_MARGIN),
            "frozen_corpus_64": r_frozen, "frozen_pass": bool(r_frozen["exact_frac"] >= b - DOWNSAMPLE_MARGIN),
            "frozen_corpus_first_N": r_frozen_first,
            "fixture_states_sha256": C.frames_digest(np.ascontiguousarray(states, np.float32))}
        print(f"[{cfg}] bound {b:.4f}  fixture-protocol exact={r_fixture['exact_frac']:.4f}  "
              f"frozen-64 exact={r_frozen['exact_frac']:.4f}  frozen-first-{N} exact={r_frozen_first['exact_frac']:.4f}",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
