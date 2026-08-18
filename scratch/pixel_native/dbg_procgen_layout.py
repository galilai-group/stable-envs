"""Dump the head of a Procgen state blob so the field layout can be read off instead of guessed."""
from __future__ import annotations

import struct
import sys

import numpy as np


def main():
    from procgen import ProcgenGym3Env

    name = sys.argv[1] if len(sys.argv) > 1 else "bigfish"
    env = ProcgenGym3Env(num=1, env_name=name, rand_seed=0, distribution_mode="hard")
    rng = np.random.RandomState(0)
    for _ in range(20):
        env.act(rng.randint(0, env.ac_space.eltype.n, size=(1,)).astype(np.int32))
    blob = env.callmethod("get_state")[0]
    print(f"{name}: {len(blob)} bytes")
    print("first 48 ints:", struct.unpack_from("<48i", blob, 0))
    print("first 96 bytes repr:", blob[:96])
    # every ASCII run of length >= 8
    runs = []
    start = None
    for i, c in enumerate(blob):
        printable = 32 <= c < 127
        if printable and start is None:
            start = i
        elif not printable and start is not None:
            if i - start >= 8:
                runs.append((start, i, blob[start:min(i, start + 40)]))
            start = None
    print(f"ascii runs >=8 bytes: {len(runs)}")
    for s, e, sample in runs[:6]:
        prefix = struct.unpack_from("<i", blob, s - 4)[0] if s >= 4 else None
        print(f"  at {s}..{e} (len {e-s}) len-prefix-at-s-4={prefix}  {sample!r}")
    for s, e, sample in runs[-3:]:
        prefix = struct.unpack_from("<i", blob, s - 4)[0] if s >= 4 else None
        print(f"  at {s}..{e} (len {e-s}) len-prefix-at-s-4={prefix}  {sample!r}")


if __name__ == "__main__":
    main()
