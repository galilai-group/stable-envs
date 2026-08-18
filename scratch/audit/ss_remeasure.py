"""Re-measure the rows that were published at ``--ss 1`` while their env declares ``default_supersample = 2``.

44 of the 196 registered classes declare 2. ``bench_vs_decode.py`` forced 1 on all of them, so the
"ours f/s" and "cores replaced" columns describe a picture ``se.make(name)`` does not produce
(DMCBallInCupCatch: 1,149 f/s at ss=1 against 356 at the env's own default, one process, one node).
``bench_storage.py`` never passed a supersample, so the two published tables disagree by that factor
on exactly these envs.

This does not re-run the whole sweep. It removes the GPU block for the affected envs from a tag file
so ``bench_vs_decode.py`` re-measures them, and keeps everything that is not being corrected — the
decode baseline and the official-renderer column, which is single-threaded CPU and does not travel
across allocations, so re-taking it would replace a good number with a differently-loaded one.

    python scratch/audit/ss_remeasure.py strip   --tag dmc
    python examples/bench_vs_decode.py --tag dmc --envs ... --ss native --out examples/results/decode
    python scratch/audit/ss_remeasure.py restore --tag dmc
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import stable_envs as se

RES = Path("/mnt/vast/home/randall.balestriero/code/stable-envs/examples/results/decode")
BACKUP = RES / "pre_ss_fix"


def affected(names) -> list[str]:
    """Registered envs whose class declares a supersample other than 1."""
    return sorted(n for n in names
                  if n in se.REGISTRY and int(getattr(se.REGISTRY[n], "default_supersample", 1)) != 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["strip", "restore", "list"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    path = RES / f"bench_vs_decode_{a.tag}.json"
    # Working files live beside the backups, not in the published results directory.
    BACKUP.mkdir(parents=True, exist_ok=True)
    side = BACKUP / f"_ss_carry_{a.tag}.json"
    d = json.loads(path.read_text())
    names = affected(d["envs"])

    if a.mode == "list":
        print(" ".join(names))
        return

    if a.mode == "strip":
        BACKUP.mkdir(parents=True, exist_ok=True)
        if not (BACKUP / path.name).exists():
            shutil.copy2(path, BACKUP / path.name)
        carry = {}
        for n in names:
            e = d["envs"][n]
            carry[n] = {k: e[k] for k in ("cpu", "decode", "state_bytes", "size") if k in e}
            e.pop("gpu", None)
            e.pop("gpu_best", None)
        side.write_text(json.dumps(carry, indent=2))
        path.write_text(json.dumps(d, indent=2))
        print(f"stripped {len(names)} envs from {path.name}; carried cpu/decode into {side.name}")
        print(" ".join(names))
        return

    carry = json.loads(side.read_text())
    n_put = 0
    for n, keep in carry.items():
        e = d["envs"].get(n)
        if not isinstance(e, dict):
            continue
        for k, v in keep.items():
            if k == "cpu" and not isinstance(e.get("cpu"), dict):
                e["cpu"] = v
                n_put += 1
    path.write_text(json.dumps(d, indent=2))
    print(f"restored the official-renderer column on {n_put} envs in {path.name}")


if __name__ == "__main__":
    main()
