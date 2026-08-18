"""Emit the evaluation command for every checkpoint this campaign produced.

Three shapes of command, and the differences are not cosmetic:

* **A per-cell corrector** is evaluated on its own cell's held-out states and on ``decl``. RoboCasa
  correctors are fitted under the env's declared camera and light variation and are therefore scored
  TWICE -- once at the default appearance, which is where a ``visual_fidelity`` declaration is
  measured, and once under the variation they were trained on, which is where a consumer who turns
  the knobs will meet them. Reporting only the first would hide the failure mode README.md §3 found.
* **A joint corrector** (one checkpoint, several cells) is evaluated once per cell with ``--env``,
  and gets its own image stem per cell so three runs do not overwrite each other's pictures.
* **A descriptor corrector** additionally carries the zero-shot block: the other camera, the wrist
  camera and the other kitchen, each with the target scene's own descriptor table swapped in. That is
  the block the row-randomisation repair has to move.
"""
from __future__ import annotations

import argparse
import os

CKPT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/ckpt"
E = "scratch/altrender/rendernet/eval_corrector.py"

HB_TASKS = ["Walk", "Run", "Stand", "Stair", "Slide", "SitSimple", "SitHard", "BalanceSimple"]
HB = ([f"HumanoidBenchH1{t}" for t in HB_TASKS]
      + [f"HumanoidBenchH1Hand{t}" for t in HB_TASKS + ["Spoon"]])
RC = ["RoboCasaCounterToSink", "RoboCasaCounterToSinkRight", "RoboCasaCounterToSinkWrist",
      "RoboCasaCounterToSinkL45S33"]
L52 = RC[:3]
RC_NETS = ("unet_112t128_w48_d2_u2", "espcn_112t128_w64_d6_u2")
HB_NETS = ("espcn_112t128_w96_d6_u2", "espcn_112t128_w64_d6_u2")
#: The architecture the HumanoidBench VARIATION campaign uses, one per cell. ``w96_d6`` is 19.1 us a
#: frame against ``w64_d6``'s 14.3 and it destabilised at lr 5e-4 on five cells (the trunk returns to
#: the bilinear fallback and the training loss goes back to its epoch-0 value); ``w64_d6`` did not
#: destabilise anywhere and is the faster of the two, so it carries the second, larger campaign.
HB_STY_NET = "espcn_112t128_w64_d6_u2"
OOD_RC = "RoboCasaCounterToSinkRight,RoboCasaCounterToSinkL45S33,RoboCasaCounterToSinkWrist"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--skip-done", action="store_true",
                    help="drop commands whose --out JSON already exists, so a second pass covers "
                         "only the checkpoints that appeared since the first")
    a = ap.parse_args()
    O, lines = a.outdir, []

    def have(stem):
        return os.path.exists(os.path.join(CKPT, stem + ".pt"))

    for e in RC:
        for net in RC_NETS:
            s = f"{e}_{net}_sty_e300"
            if not have(s):
                continue
            lines.append(f"{E} {CKPT}/{s}.pt --splits heldout,decl --tag r112_tex2 "
                         f"--outdir {O} --out {O}/eval_{s}_on_default.json")
            lines.append(f"{E} {CKPT}/{s}.pt --splits heldout "
                         f"--outdir {O} --stem {s}_ONSTY --out {O}/eval_{s}_on_sty.json")
    for e in HB:
        for net in HB_NETS:
            s = f"{e}_{net}_e300"
            if not have(s):
                continue
            lines.append(f"{E} {CKPT}/{s}.pt --splits heldout,decl "
                         f"--outdir {O} --out {O}/eval_{s}.json")
            if net == HB_STY_NET:
                # The same DEFAULT-TRAINED checkpoint under the env's own declared variation. On
                # RoboCasa this is the measurement that says a corrector trained at one camera pose
                # is worse than not correcting at all (README §3); HumanoidBench declares +-40 deg of
                # azimuth against RoboCasa's +-8, so it is the measurement that decides whether these
                # cells can ship at all.
                lines.append(f"{E} {CKPT}/{s}.pt --splits heldout --tag r112_tex1_sty "
                             f"--outdir {O} --stem {s}_ONSTY --out {O}/eval_{s}_on_sty.json")
        s = f"{e}_{HB_STY_NET}_sty_e300"
        if have(s):
            lines.append(f"{E} {CKPT}/{s}.pt --splits heldout,decl --tag r112_tex1 "
                         f"--outdir {O} --out {O}/eval_{s}_on_default.json")
            lines.append(f"{E} {CKPT}/{s}.pt --splits heldout "
                         f"--outdir {O} --stem {s}_ONSTY --out {O}/eval_{s}_on_sty.json")
    # lead (a): one checkpoint, three cameras of one kitchen
    for s in (f"RoboCasaL52S03x3_{n}_sty_e300" for n in RC_NETS):
        if not have(s):
            continue
        for cell in L52:
            lines.append(f"{E} {CKPT}/{s}.pt --env {cell} --splits heldout,decl --tag r112_tex2 "
                         f"--outdir {O} --stem {s}__{cell} --out {O}/eval_{s}_on_{cell}.json")
            lines.append(f"{E} {CKPT}/{s}.pt --env {cell} --splits heldout --tag r112_tex2_sty "
                         f"--outdir {O} --stem {s}__{cell}_ONSTY "
                         f"--out {O}/eval_{s}_on_{cell}_sty.json")
    s = "RoboCasaL52S03x3budget_unet_112t128_w48_d2_u2_sty_e300"
    if have(s):
        for cell in L52:
            lines.append(f"{E} {CKPT}/{s}.pt --env {cell} --splits heldout,decl --tag r112_tex2 "
                         f"--outdir {O} --stem {s}__{cell} --out {O}/eval_{s}_on_{cell}.json")
    for s in ("RoboCasaCounterToSink_unet_112t128_w48_d2_u2_sty_jointloop_e300",):
        if have(s):
            lines.append(f"{E} {CKPT}/{s}.pt --splits heldout,decl --tag r112_tex2 "
                         f"--outdir {O} --out {O}/eval_{s}_on_default.json")
    # lead (b): the descriptor table's rows randomised the way the style is
    for v in ("ctl", "rowdrop", "rowjit", "rowres", "rowall", "rowheavy"):
        s = f"RoboCasaCounterToSink_unet_112t128_w48_d2_u2_sty_aux-descriptor-{v}_e300"
        if not have(s):
            continue
        lines.append(f"{E} {CKPT}/{s}.pt --splits heldout,decl --tag r112_tex2 --ood {OOD_RC} "
                     f"--outdir {O} --out {O}/eval_{s}_on_default.json")

    os.makedirs(O, exist_ok=True)
    if a.skip_done:
        lines = [ln for ln in lines if not os.path.exists(ln.rsplit("--out ", 1)[1].strip())]
    with open(a.tasks, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(a.tasks, len(lines))


if __name__ == "__main__":
    main()
