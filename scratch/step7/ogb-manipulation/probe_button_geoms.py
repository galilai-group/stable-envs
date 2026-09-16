"""Are the button-coloured geoms MOVERS, or do they live in the plate?

This decides whether pinning the plate at the all-zero button bits is correct or harmful. The mover
pass draws each world at its own button colours; the plate is one shared bitmap. So any geom whose
colour depends on a button bit and which is NOT in the mover set has its colour frozen in the plate,
and the bit stops reaching those pixels for every world at once. If that set is empty the pinning is
exact; if it is not, the fix is to widen the mover set, NOT to widen the fidelity ceiling.

No GPU: a model compile and swm's own `_apply_button_states`.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket

import numpy as np


def movers(m) -> set[int]:
    out = set()
    for i in range(m.ngeom):
        b = int(m.geom_bodyid[i])
        while b > 0:
            if int(m.body_dofnum[b]) > 0 or int(m.body_mocapid[b]) >= 0:
                out.add(i)
                break
            b = int(m.body_parentid[b])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-id", default="swm/OGBScene-v0")
    a = ap.parse_args()

    import gymnasium
    import stable_worldmodel  # noqa: F401

    env = gymnasium.make(a.env_id)
    env.reset(seed=0)
    u = env.unwrapped
    m = u.model
    mv = movers(m)
    n = int(getattr(u, "_num_buttons", 0) or 0)
    saved = list(getattr(u, "_cur_button_states", []))

    u._cur_button_states = [0] * n
    u._apply_button_states()
    g0, t0 = np.array(m.geom_rgba, copy=True), np.array(m.mat_rgba, copy=True)
    u._cur_button_states = [1] * n
    u._apply_button_states()
    g1, t1 = np.array(m.geom_rgba, copy=True), np.array(m.mat_rgba, copy=True)
    u._cur_button_states = saved
    u._apply_button_states()

    gd = [i for i in range(m.ngeom) if not np.array_equal(g0[i], g1[i])]
    td = [i for i in range(m.nmat) if not np.array_equal(t0[i], t1[i])]
    mat_users = {i: [g for g in range(m.ngeom) if int(m.geom_matid[g]) == i] for i in td}
    stranded = sorted(set([i for i in gd if i not in mv])
                      | {g for gs in mat_users.values() for g in gs if g not in mv})

    rep = {
        "env_id": a.env_id, "nbuttons": n, "ngeom": int(m.ngeom), "n_movers": len(mv),
        "geoms_coloured_by_buttons": gd,
        "materials_coloured_by_buttons": td,
        "material_users": {str(k): v for k, v in mat_users.items()},
        "STRANDED_IN_PLATE": stranded,
        "verdict": ("pinning is EXACT: every button-coloured geom is a mover and is drawn per world"
                    if not stranded else
                    "pinning FREEZES these geoms at the zero-bit colour for every world; widen the "
                    "mover set to include them"),
        "job": f"job {os.environ.get('SLURM_JOB_ID')} on {socket.gethostname()}",
    }
    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "button_geoms.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
