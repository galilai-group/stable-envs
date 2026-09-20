"""Freeze golden vectors for ``stable_envs/render/prng.py`` out of a real jax.

Run in the reference venv (`/Users/samib/se7-craftax-venv`, jax 0.11.1). Writes
``tests/fixtures/regression/fidelity/prng_threefry_golden.json``, which ``tests/unit/test_craftax_prng.py`` asserts against
so the test needs no jax at all — the repo ships no jax and the Oscar interpreters have none, so a
test that imported jax would simply skip everywhere and assert nothing.

Small shapes are frozen in full; the render-sized ones as a sha256 over the uint32 bytes plus their
first and last eight values, because a digest catches any bit change and a 112x144 array does not
belong in a source tree.
"""

from __future__ import annotations

import hashlib
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

OUT = "tests/fixtures/regression/fidelity/prng_threefry_golden.json"

FULL_SHAPES = [(1, 1), (2, 2), (3, 5), (8,), (7, 9)]
DIGEST_SHAPES = [(49, 63), (112, 144), (63, 77), (90, 110)]


def bits_of(key, shape):
    """The uint32 bit patterns behind ``jax.random.uniform(key, shape)``.

    Taken from the floats rather than from a private bits API, so what is frozen is exactly what a
    renderer consumes: ``uniform`` is ``bitcast((bits >> 9) | 0x3f800000) - 1``, which is invertible
    on the mantissa, and the mantissa is all a float32 uniform carries.
    """
    u = np.asarray(jax.device_get(jax.random.uniform(jnp.asarray(key, jnp.uint32), shape)))
    return (u + 1.0).astype(np.float32).view(np.uint32) & 0x7FFFFF


def main():
    rng = np.random.default_rng(20260915)
    keys = [[0, 0], [0, 1], [1, 0], [0xFFFFFFFF, 0xFFFFFFFF], [0x80000000, 0x7FFFFFFF]]
    keys += [np.asarray(jax.device_get(jax.random.PRNGKey(s)), np.uint32).tolist() for s in range(4)]
    keys += [rng.integers(0, 2**32, size=2, dtype=np.uint64).astype(np.uint32).tolist() for _ in range(7)]

    full, digest = [], []
    for k in keys:
        for shape in FULL_SHAPES:
            u = np.asarray(jax.device_get(jax.random.uniform(jnp.asarray(k, jnp.uint32), shape)))
            full.append(
                {
                    "key": [int(x) for x in k],
                    "shape": list(shape),
                    "uniform_bits": [int(v) for v in u.view(np.uint32).ravel()],
                }
            )
        for shape in DIGEST_SHAPES:
            u = np.asarray(jax.device_get(jax.random.uniform(jnp.asarray(k, jnp.uint32), shape)))
            b = np.ascontiguousarray(u).view(np.uint32).ravel()
            digest.append(
                {
                    "key": [int(x) for x in k],
                    "shape": list(shape),
                    "sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                    "first8": [int(v) for v in b[:8]],
                    "last8": [int(v) for v in b[-8:]],
                }
            )

    # the scaled field the craftax renderers actually draw, end to end
    scaled = []
    for k in keys[:6]:
        for shape in [(49, 63), (112, 144)]:
            u = np.asarray(jax.device_get(jax.random.uniform(jnp.asarray(k, jnp.uint32), shape))) * 95 + 32
            b = np.ascontiguousarray(u.astype(np.float32)).view(np.uint32).ravel()
            scaled.append(
                {
                    "key": [int(x) for x in k],
                    "shape": list(shape),
                    "expr": "uniform(key, shape) * 95 + 32",
                    "sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                }
            )

    out = {
        "what": (
            "golden vectors for stable_envs/render/prng.py, frozen from a real jax so the test "
            "needs none. 'uniform_bits' are the float32 bit patterns of "
            "jax.random.uniform(key, shape); digests are sha256 over those bytes."
        ),
        "jax": jax.__version__,
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
        "generated_by": "scratch/step7/craftax-classic-full/gen_prng_golden.py",
        "full": full,
        "digest": digest,
        "scaled": scaled,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(
        f"wrote {OUT}: {len(full)} full, {len(digest)} digest, {len(scaled)} scaled, "
        f"jax {jax.__version__}, partitionable={out['jax_threefry_partitionable']}"
    )


if __name__ == "__main__":
    main()
