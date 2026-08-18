"""How far does "the Atari frame is a function of the 128-byte RAM" actually hold?

probe_ale.py found, on 10 ROMs, that two emulator states forced to identical RAM produce an identical
next frame.  That is a surprising positive and it is the claim most worth trying to break, so this
widens it:

  * 30 ROMs instead of 10, 8 random state pairs each, so a single lucky pair cannot carry the result;
  * a harsher second branch -- RAM written into a *freshly reset* console, where the program counter,
    the stack, the bank register and the TIA are wherever power-on left them;
  * multi-step divergence: after forcing RAM equal, how many frames stay identical?  If the answer is
    "one", then RAM pins the picture but not the dynamics, and a state dataset still cannot be rolled
    forward without the emulator.

Run under SLURM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_ram(ale, ram):
    for i, v in enumerate(ram):
        ale.setRAM(int(i), int(v))


def probe(rom_path, n_pairs=8, horizon=8, seed=0):
    from ale_py import Action, ALEInterface, LoggerMode

    ale = ALEInterface()
    ALEInterface.setLoggerMode(LoggerMode.Error)
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.loadROM(rom_path)
    acts = ale.getMinimalActionSet()
    rng = np.random.RandomState(seed)

    ale.reset_game()
    reset_state = ale.cloneSystemState()

    same_1step, same_1step_fresh = [], []
    same_ram_1step = []
    divergence = []
    ram_diff_bytes = []

    for p in range(n_pairs):
        ale.restoreSystemState(reset_state)
        for _ in range(100 + 60 * p):
            ale.act(acts[rng.randint(len(acts))])
        S_t, ram_t = ale.cloneSystemState(), ale.getRAM().copy()
        for _ in range(37 + 11 * p):
            ale.act(acts[rng.randint(len(acts))])
        S_u, ram_u = ale.cloneSystemState(), ale.getRAM().copy()
        ram_diff_bytes.append(int((ram_t != ram_u).sum()))

        plan = [acts[rng.randint(len(acts))] for _ in range(horizon)]

        def roll(S):
            ale.restoreSystemState(S)
            write_ram(ale, ram_t)
            fr, rm = [], []
            for a in plan:
                ale.act(a)
                fr.append(ale.getScreen().copy())
                rm.append(ale.getRAM().copy())
            return fr, rm

        f_t, r_t = roll(S_t)
        f_u, r_u = roll(S_u)
        f_f, _ = roll(reset_state)

        same_1step.append(bool(np.array_equal(f_t[0], f_u[0])))
        same_1step_fresh.append(bool(np.array_equal(f_t[0], f_f[0])))
        same_ram_1step.append(bool(np.array_equal(r_t[0], r_u[0])))
        k = 0
        while k < horizon and np.array_equal(f_t[k], f_u[k]):
            k += 1
        divergence.append(k)

    rom_bytes = Path(rom_path).stat().st_size
    return {
        "rom": Path(rom_path).stem,
        "rom_bytes": rom_bytes,
        "bankswitched": rom_bytes > 4096,
        "n_pairs": n_pairs,
        "ram_diff_bytes_mean": float(np.mean(ram_diff_bytes)),
        "frac_same_frame_1step": float(np.mean(same_1step)),
        "frac_same_frame_1step_from_fresh_console": float(np.mean(same_1step_fresh)),
        "frac_same_ram_1step": float(np.mean(same_ram_1step)),
        "frames_identical_before_divergence_mean": float(np.mean(divergence)),
        "frames_identical_before_divergence_max": int(np.max(divergence)),
        "horizon": horizon,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=30)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/ale_ext.json")
    args = ap.parse_args()

    import ale_py.roms as roms
    romdir = Path(roms.__file__).parent
    games = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "montezuma_revenge",
             "freeway", "asterix", "boxing", "qbert", "alien", "amidar", "assault", "asteroids",
             "atlantis", "bank_heist", "battle_zone", "beam_rider", "berzerk", "bowling",
             "centipede", "chopper_command", "crazy_climber", "demon_attack", "enduro",
             "fishing_derby", "frostbite", "gopher", "gravitar", "hero", "ice_hockey",
             "jamesbond", "kangaroo", "krull", "kung_fu_master", "pitfall", "private_eye",
             "riverraid", "road_runner", "robotank", "skiing", "solaris", "star_gunner",
             "tennis", "time_pilot", "tutankham", "up_n_down", "venture", "video_pinball",
             "wizard_of_wor", "yars_revenge", "zaxxon"][:args.n_games]

    rows = []
    for g in games:
        p = romdir / f"{g}.bin"
        if not p.exists():
            print(f"skip {g}")
            continue
        r = probe(str(p), n_pairs=args.n_pairs, horizon=args.horizon)
        rows.append(r)
        print(f"{r['rom']:<20} rom {r['rom_bytes']:6d}B bank={r['bankswitched']!s:5s} "
              f"| same-frame after 1 step: {100*r['frac_same_frame_1step']:5.1f}% "
              f"(from fresh console {100*r['frac_same_frame_1step_from_fresh_console']:5.1f}%)  "
              f"same-RAM after 1 step: {100*r['frac_same_ram_1step']:5.1f}%  "
              f"| identical frames before divergence: "
              f"{r['frames_identical_before_divergence_mean']:.2f}/{args.horizon}", flush=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    agg = {
        "n_roms": len(rows),
        "mean_frac_same_frame_1step": float(np.mean([r["frac_same_frame_1step"] for r in rows])),
        "mean_frac_same_frame_1step_fresh": float(np.mean([r["frac_same_frame_1step_from_fresh_console"] for r in rows])),
        "mean_frac_same_ram_1step": float(np.mean([r["frac_same_ram_1step"] for r in rows])),
        "mean_divergence": float(np.mean([r["frames_identical_before_divergence_mean"] for r in rows])),
    }
    print("\nAGG", json.dumps(agg, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
