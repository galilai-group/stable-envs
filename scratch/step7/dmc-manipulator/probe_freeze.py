"""The candidate FROZEN CORPUS for ``DMCManipulatorBringBall``, under examples/results/step7/.

WHAT REVIEW CORRECTED (R5/C3). The first version of this script wrote the 256-row POOL into
``states/<cell>.npz`` with a ``meta`` that used ``config_id`` and carried no ``selection``,
``indices`` or ``sha256``. ``fidelity_step4.corpus_for`` does not resample an existing file -- it
returns ``z["states"]`` verbatim -- so at the Step-8 freeze that file would have handed the harness
**256 states**, silently quadrupling every conformance run against a corpus that had never been
through the selection rule, and the missing ``sha256``/``indices`` would have left it unverifiable.

So the selection is now done by ``fidelity_step4.select`` ITSELF, imported rather than reimplemented,
at ``fidelity_step4.STATE_SEED``, and the file written is the SELECTED 64 in the exact Step-4
``states``/``meta`` schema (``env``, ``seed``, ``selection``, ``state_dim``, ``state_names``,
``sha256``, ``pool``, ``kept``, ``n_typical``, ``n_extreme``, ``n_spread``, ``indices``). The 256-row
pool is kept BESIDE it as ``pool/<cell>_pool.npz`` -- useful, and out of the path ``corpus_for`` reads.

The states come from the domain's OWN visited distribution (``sample_states`` = the task's
``initialize_episode`` plus random actions), because a parity number measured on states the env never
reaches proves very little. Digests are re-emitted on the SELECTED 64, at the certified size and at 84.

NOTHING under examples/results/step4|step5|step6 is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

CELL = "DMCManipulatorBringBall"


def _fidelity_step4():
    """``examples/fidelity_step4.py`` as a module, so ``select`` and ``STATE_SEED`` are the SHIPPED
    ones. Reimplementing the selection here would be a second definition of the corpus, and the two
    would drift exactly once."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    sys.path.insert(0, os.path.join(root, "examples"))
    import importlib
    return importlib.import_module("fidelity_step4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--also-size", type=int, default=84)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--n", type=int, default=64, help="the SELECTED corpus size, as in Step 4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="examples/results/step7/dmc-manipulator/corpus")
    a = ap.parse_args()

    import mujoco
    from importlib.metadata import version as _v
    import stable_envs as se
    from stable_envs.testing import conformance as C

    F = _fidelity_step4()
    cls = se.REGISTRY[CELL]
    env = cls(batch=min(a.pool, 64), size=a.size, device=a.device, seed=0)

    raw = env.sample_states(int(a.pool), rng=np.random.default_rng(F.STATE_SEED))
    states, how = F.select(np.asarray(raw, np.float32), int(a.n))
    states = np.ascontiguousarray(states, np.float32)

    # EXACTLY corpus_for's schema, key for key, so the Step-8 freeze can adopt this file unchanged.
    meta = {"seed": F.STATE_SEED,
            "selection": "typical + per-dim extremes + farthest-point spread",
            "state_dim": int(states.shape[1]), "state_names": list(env.state_names or []),
            "env": CELL, "sha256": C.frames_digest(states), **how}
    # Step-7 provenance, additive: corpus_for ignores keys it does not read, and a reader should be
    # able to see which interpreter and pin drew this without going to the findings.
    meta.update(step=7, workstream="dmc-manipulator", reference_id=cls.reference_id,
                default_size=int(cls.default_size), sizes=[a.size, a.also_size],
                mujoco=mujoco.__version__, dm_control=_v("dm_control"), node=platform.node(),
                status=("CANDIDATE for the Step-8 freeze; not frozen in Step 7 and not a replacement "
                        "for anything under examples/results/step4"))

    os.makedirs(os.path.join(a.out, "states"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "pool"), exist_ok=True)
    npz = os.path.join(a.out, "states", f"{CELL}.npz")
    np.savez(npz, states=states, meta=np.array(json.dumps(meta, sort_keys=True)))
    print("wrote", npz, states.shape, "sha256", meta["sha256"])

    pool_npz = os.path.join(a.out, "pool", f"{CELL}_pool.npz")
    pool_states = np.ascontiguousarray(raw, np.float32)
    np.savez_compressed(pool_npz, states=pool_states, meta=np.array(json.dumps(
        {"env": CELL, "seed": F.STATE_SEED, "pool": int(a.pool),
         "what_this_is": ("the unselected pool sample_states drew; the CORPUS is states/<env>.npz, "
                          "which is select()'s 64 rows of it"),
         "sha256": C.frames_digest(pool_states)}, sort_keys=True)))
    print("wrote", pool_npz, pool_states.shape)

    digests = {}
    for S in (a.size, a.also_size):
        e = cls(batch=int(a.n), size=S, device=a.device, seed=0)
        ours = e.render(torch.as_tensor(states, device=e.device)).cpu().numpy()
        ref = C.reference_env(cls, S)
        official = cls.reference_render(ref, states)
        pa = C.pixel_agreement(ours, official)
        pa["ssim"] = float(C.ssim(ours, official))
        digests[str(S)] = dict(n=int(a.n), states_sha256=meta["sha256"],
                               digest_ours=C.frames_digest(ours),
                               digest_official=C.frames_digest(official),
                               supersample=int(e.ss), parity=pa)
        print(S, json.dumps(digests[str(S)], default=float)[:420])

    p = os.path.join(a.out, "digests.json")
    json.dump(dict(meta=meta, digests=digests), open(p, "w"), indent=1, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()
