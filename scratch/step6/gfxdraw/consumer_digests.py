"""Bit-exactness snapshot for the OTHER consumers of ``render/gfxdraw``.

``LunarLander`` and ``CarRacing`` draw through the same primitive layer as classic control but are
not in the frozen Step 4 corpus, so a gfxdraw edit has no oracle for them. This makes one: the same
seeded states rendered before and after, digested with the repository's own ``frames_digest``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="LunarLander,CarRacing")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--snapshot-out", default="")
    ap.add_argument("--snapshot-in", default="")
    a = ap.parse_args()

    import stable_envs as se
    from stable_envs.testing import conformance as C

    prev = json.loads(Path(a.snapshot_in).read_text()) if a.snapshot_in else None
    out, bad = {}, 0
    for name in a.envs.split(","):
        if name not in se.REGISTRY:
            out[name] = {"skipped": "not registered in this interpreter"}
            print(f"[{name:14s}] not registered")
            continue
        try:
            env = se.make(name, batch=a.batch, device=a.device)
            st = env.sample_states(a.batch, np.random.default_rng(a.seed))
            d = {"digest": C.frames_digest(env.render(st).cpu().numpy()),
                 "states_digest": C.frames_digest(np.ascontiguousarray(st, np.float32))}
            style = C.uniform_style(env, a.batch, groups=2, seed=0)
            if style is not None:
                d["digest_style"] = C.frames_digest(env.render(st, style=style).cpu().numpy())
        except Exception as e:                                            # noqa: BLE001
            d = {"error": f"{type(e).__name__}: {e}"[:300]}
        out[name] = d
        if prev is not None and "digest" in d:
            same = all(prev.get(name, {}).get(k) == v for k, v in d.items())
            bad += not same
            print(f"[{name:14s}] {'OK' if same else 'MISMATCH'}  {d}")
            if not same:
                print(f"                 was {prev.get(name)}")
        else:
            print(f"[{name:14s}] {d}")
    if a.snapshot_out:
        Path(a.snapshot_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.snapshot_out).write_text(json.dumps(out, indent=1))
        print(f"wrote {a.snapshot_out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
