"""One stone, one horizontal slice through its centre: ours against the reference, level by level.

Aggregates say "the rim disagrees". They cannot say WHY. A slice through a single isolated stone
shows the radius, the stroke band and the anti-aliasing ramp as numbers, so a half-pixel offset, a
wrong radius and a coverage-model difference look completely different instead of all looking like
"the rim disagrees".
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

import stable_envs as se


def slice_through(name: str, cell: int, value: float):
    cls = se.REGISTRY[name]
    env = se.make(name, batch=1, device="cuda")
    spec, b = env._spec, env._b
    st = np.full((1, env.state_dim), float(spec["empty"]), np.float32)
    st[0, cell] = value
    off = cls.reference_render(cls.reference_env(size=env.size), st)[0]
    ours = env.render(st).cpu().numpy()[0]

    row_i = int(round((cell // spec["ncols"] + 1) * b.grid)) - 1
    cx = (cell % spec["ncols"] + 1) * b.grid
    print(f"--- {name} cell={cell} value={value} r={spec['radius']:.4f} "
          f"centre_x={cx} row={row_i} ---")
    lo, hi = int(cx - spec["radius"] - 4), int(cx + spec["radius"] + 5)
    print("     x  |  ref(R)  ours(R) | diff")
    for x in range(lo, hi):
        rr, oo = int(off[row_i, x, 0]), int(ours[row_i, x, 0])
        d = abs(rr - oo)
        flag = "" if d == 0 else ("  <<<" if d > 8 else "  <")
        print(f"  {x:4d}  |  {rr:5d}   {oo:5d}  | {rr-oo:+5d}{flag}")
    # where is the reference's own transition? locate the 50% crossing on each side
    prof = off[row_i, :, 0].astype(np.float64)
    print(f"  reference profile min={prof.min()} max={prof.max()}")


if __name__ == "__main__":
    # a BLACK stone (fill == stroke, one edge) and a WHITE stone (fill + a 1 px black ring)
    slice_through("PgxGo9x9", 40, 1.0)
    print()
    slice_through("PgxGo9x9", 40, -1.0)
    print()
    slice_through("PgxOthello", 27, 1.0)
