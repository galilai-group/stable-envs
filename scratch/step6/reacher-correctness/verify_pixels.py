"""Does the BASELINE and does THIS BRANCH draw the same picture at the default appearance?

Asked directly, on the GPU, in one job, against the frozen Step-4 corpora -- the same-card rule from
AGENT_COMMON applied to pixels instead of to a throughput number. Which tree is measured is decided
by ``PYTHONPATH``, so the caller runs this twice in one SLURM job and diffs the two JSON files.

    PYTHONPATH=$SCR/se6/baseline           python verify_pixels.py --out before.json --states ...
    PYTHONPATH=$SCR/se6/reacher-correctness python verify_pixels.py --out after.json  --states ...

Only ``digest_ours`` (no style argument -- the env's own default) and ``digest_default`` (the packed
``default_style()`` passed explicitly) are comparable across the two trees. A digest taken under
``uniform_style`` is NOT: it draws from the variation space, and this patch adds a leaf to it, so the
two trees would be asked for different appearances and a difference would say nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True, help="comma-separated env names")
    ap.add_argument("--states", required=True, help="directory of frozen <Env>.npz corpora")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import stable_envs as se
    from stable_envs.testing import conformance as C

    out = {"tree": se.__file__, "device": a.device, "envs": {}}
    for name in a.envs.split(","):
        p = Path(a.states) / f"{name}.npz"
        if not p.exists():
            out["envs"][name] = {"status": "no corpus", "path": str(p)}
            continue
        try:
            st = np.load(p)["states"]
            e = se.make(name, batch=len(st), device=a.device)
            f = e.render(st).cpu().numpy()
            ds = np.tile(np.asarray(e.default_style(), np.float32), (len(st), 1))
            g = e.render(st, style=ds).cpu().numpy() if e.style_dim else f
            out["envs"][name] = {
                "status": "measured",
                "digest_ours": C.frames_digest(f),
                "digest_default": C.frames_digest(g),
                "mean_pixel": float(f.mean()), "n": int(len(st)),
                "style_dim": int(e.style_dim), "style_names": list(e.style_names),
                "state_names": list(e.state_names),
            }
        except BaseException as ex:                                         # noqa: BLE001
            out["envs"][name] = {"status": "error", "why": f"{type(ex).__name__}: {str(ex)[:400]}"}
        print(name, out["envs"][name].get("digest_ours", out["envs"][name].get("why")), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
