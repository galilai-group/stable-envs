"""Phase 0 (2) — does the torch threefry2x32 reproduce jax.random.uniform bit for bit?

Sweeps many keys and every shape the Classic and full renderers actually draw, at both tile sizes,
and also checks the keys the real rollout produced. Reports exact agreement, and separately reports
what the pre-0.4.36 non-partitionable path gives, so the jax-version dependence is measured.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import threefry_torch as tf3  # noqa: E402

OUT = "examples/results/step7/craftax-classic-full/phase0"

SHAPES = {
    "classic_7px": (7 * 7, 9 * 7),        # (49, 63)
    "classic_16px": (7 * 16, 9 * 16),     # (112, 144)
    "full_7px": (9 * 7, 11 * 7),          # (63, 77) — W2's window at agent size
    "full_10px": (9 * 10, 11 * 10),
    "odd_1": (3, 5),                      # odd element count, exercises the padding branch
    "tiny": (1, 1),
}


def jax_uniform(key_u32, shape):
    k = jnp.asarray(key_u32, dtype=jnp.uint32)
    return np.asarray(jax.device_get(jax.random.uniform(k, shape)))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    rng = np.random.default_rng(7)

    keys = [np.array([0, 0], np.uint32), np.array([0, 1], np.uint32),
            np.array([0xFFFFFFFF, 0xFFFFFFFF], np.uint32)]
    keys += [np.asarray(jax.device_get(jax.random.PRNGKey(int(s))), np.uint32)
             for s in range(8)]
    keys += [rng.integers(0, 2 ** 32, size=2, dtype=np.uint64).astype(np.uint32)
             for _ in range(53)]

    # the state_rngs the real rollout actually produced
    real_keys = []
    p = "scratch/step7/craftax-classic-full/states.pkl"
    if os.path.isfile(p):
        with open(p, "rb") as f:
            for st in pickle.load(f):
                real_keys.append(np.asarray(jax.device_get(st.state_rng), np.uint32))

    def compare(tag, keylist, shape):
        n_exact = n_tot = 0
        worst = 0.0
        n_exact_orig = 0
        for k in keylist:
            ref = jax_uniform(k, shape)
            got = tf3.uniform(torch.tensor(k.astype(np.int64)), shape).numpy()
            n_tot += 1
            same = np.array_equal(ref.view(np.uint32), got.view(np.uint32))
            n_exact += int(same)
            worst = max(worst, float(np.abs(ref - got).max()))
            got_o = tf3.uniform(torch.tensor(k.astype(np.int64)), shape,
                                partitionable=False).numpy()
            n_exact_orig += int(np.array_equal(ref.view(np.uint32), got_o.view(np.uint32)))
        return dict(tag=tag, shape=list(shape), n_keys=n_tot,
                    bit_exact_partitionable=n_exact, bit_exact_original=n_exact_orig,
                    max_abs_diff=worst)

    for name, shape in SHAPES.items():
        rows.append(compare(name, keys, shape))
    if real_keys:
        rows.append(compare("classic_7px_real_rollout_keys", real_keys, SHAPES["classic_7px"]))
        rows.append(compare("classic_16px_real_rollout_keys", real_keys, SHAPES["classic_16px"]))

    # batched path must equal the per-key path
    kb = np.stack(keys[:32]).astype(np.int64)
    ref = np.stack([jax_uniform(k.astype(np.uint32), SHAPES["classic_7px"]) for k in kb])
    got = tf3.uniform_batch(torch.tensor(kb), SHAPES["classic_7px"]).numpy()
    batched = dict(tag="batched_vs_jax", n_keys=int(kb.shape[0]),
                   bit_exact=bool(np.array_equal(ref.view(np.uint32), got.view(np.uint32))),
                   max_abs_diff=float(np.abs(ref - got).max()))

    res = dict(
        jax_version=jax.__version__,
        jax_threefry_partitionable_default=bool(jax.config.jax_threefry_partitionable),
        jax_default_prng_impl=str(jax.config.jax_default_prng_impl),
        torch_version=torch.__version__,
        per_shape=rows, batched=batched,
        note=("bit_exact_original is the pre-jax-0.4.36 code path; where it is 0 and "
              "bit_exact_partitionable is the full count, the night noise field is a function of "
              "the jax CONFIG as well as of the state."))
    with open(os.path.join(OUT, "prng_exactness.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
