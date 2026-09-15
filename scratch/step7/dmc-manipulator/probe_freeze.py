"""The candidate FROZEN CORPUS for ``DMCManipulatorBringBall``, written under examples/results/step7/.

Same shape as ``examples/results/step4/states/<Config>.npz``: a ``states`` array and a ``meta`` string,
so the Step-8 freeze can adopt it without a format conversion and ``fidelity_step4.py --states-dir``
can read it. NOTHING under examples/results/step4|step5|step6 is touched.

The states come from the domain's OWN visited distribution (``sample_states`` = the task's
``initialize_episode`` plus random actions), because a parity number measured on states the env never
reaches proves very little -- the argument ``sample_states`` records. The seed is fixed and recorded
in ``meta`` so the corpus is reproducible from the seed rather than only from the file.

Also writes the digests a certification document needs: our frames and the official frames, at the
certified size and at 84.
"""
from __future__ import annotations

import argparse
import json
import os
import platform

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

CELL = "DMCManipulatorBringBall"
SEED = 20260915


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--also-size", type=int, default=84)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--corpus", type=int, default=64,
                    help="how many of --n the conformance run uses, as in Step 4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="examples/results/step7/dmc-manipulator/corpus")
    a = ap.parse_args()

    import mujoco
    from importlib.metadata import version as _v
    import stable_envs as se
    from stable_envs.testing import conformance as C

    cls = se.REGISTRY[CELL]
    env = cls(batch=min(a.n, 64), size=a.size, device=a.device, seed=0)
    states = env.sample_states(a.n, np.random.default_rng(SEED))

    meta = dict(
        config_id=CELL, reference_id=cls.reference_id, seed=SEED, n=a.n, corpus=a.corpus,
        state_dim=int(env.state_dim), state_names=list(env.state_names),
        default_size=int(cls.default_size), sizes=[a.size, a.also_size],
        mujoco=mujoco.__version__, dm_control=_v("dm_control"),
        node=platform.node(), step=7, workstream="dmc-manipulator",
        sampler=("sample_states: the domain's own initialize_episode plus random actions, seeded; "
                 "the episode half of each state is read out of the model the episode wrote it into"),
        note=("CANDIDATE corpus for the Step-8 freeze. Not frozen in Step 7 and not a replacement for "
              "anything under examples/results/step4."))

    os.makedirs(os.path.join(a.out, "states"), exist_ok=True)
    npz = os.path.join(a.out, "states", f"{CELL}.npz")
    np.savez_compressed(npz, states=states.astype(np.float32), meta=json.dumps(meta))
    print("wrote", npz, states.shape)

    digests = {}
    for S in (a.size, a.also_size):
        e = cls(batch=a.corpus, size=S, device=a.device, seed=0)
        st = states[:a.corpus]
        ours = e.render(torch.as_tensor(st, device=e.device)).cpu().numpy()
        ref = C.reference_env(cls, S)
        official = cls.reference_render(ref, st)
        pa = C.pixel_agreement(ours, official)
        pa["ssim"] = float(C.ssim(ours, official))
        digests[str(S)] = dict(digest_ours=C.frames_digest(ours),
                               digest_official=C.frames_digest(official),
                               supersample=int(e.ss), parity=pa)
        print(S, json.dumps(digests[str(S)], default=float)[:400])

    p = os.path.join(a.out, "digests.json")
    json.dump(dict(meta=meta, digests=digests), open(p, "w"), indent=1, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()
