"""Find the fits whose TRUNK DIED, and emit the same fit at a lower learning rate.

README.md §6 records this failure and its fix -- ReLU units that cannot come back, cured by SiLU plus
gradient clipping -- and it came back on HumanoidBench at ``lr 5e-4`` with the wider ``w96_d6``
trunk. The signature is unchanged and it is unmistakable: the residual collapses to zero, so the
model reverts to its zero-initialised fallback and the TRAINING loss returns to the value it had at
epoch 0, which is the bilinear baseline to three digits. The run does not look broken -- the loss
curve reads as a plateau -- and it still saves a checkpoint, because ``train.py`` keeps the best
epoch, so what a collapsed run costs is the last two thirds of its schedule rather than the whole of
it.

**The detector reads the TRAINING loss and nothing else.** That matters: choosing a learning rate
from the held-out number would make the held-out number a selection statistic and it would no longer
mean what this study says it means. A run is re-fitted when its final training loss is no better than
its first, which is a statement about the optimiser and not about generalisation.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

CKPT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/ckpt"


def collapsed(js: str, tol: float = 0.98) -> tuple[bool, float]:
    d = json.load(open(js))
    h = d.get("history") or []
    if not h or "train_l1_255" not in h[0]:
        return False, float("nan")
    ratio = h[-1]["train_l1_255"] / max(h[0]["train_l1_255"], 1e-9)
    return ratio > tol, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="?", default="*_e300.json")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--suffix", default="_lr2e-4")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--tol", type=float, default=0.98)
    a = ap.parse_args()
    lines = []
    for p in sorted(glob.glob(os.path.join(CKPT, a.pattern))):
        bad, ratio = collapsed(p, a.tol)
        if not bad:
            continue
        d = json.load(open(p))
        ar = d["args"]
        stem = os.path.basename(p)[:-5]
        out = stem + a.suffix
        if os.path.exists(os.path.join(CKPT, out + ".pt")):
            continue
        print(f"collapsed  {stem:62s} train_l1 ratio {ratio:.3f}")
        if "envs" in ar:                                          # a train_joint.py fit
            lines.append(f"scratch/altrender/rendernet/train_joint.py --envs {ar['envs']} "
                         f"--net {ar['net']} --tag {ar['tag']} --epochs {ar['epochs']} "
                         f"--lr {a.lr} --out {out}")
        else:
            lines.append(f"scratch/altrender/rendernet/train.py {ar['env']} --net {ar['net']} "
                         f"--tag {ar['tag']} --epochs {ar['epochs']} --batch {ar['batch']} "
                         f"--lr {a.lr} --out {out}")
    if a.tasks:
        with open(a.tasks, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"\n{a.tasks}  {len(lines)} re-fits at lr {a.lr}")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
