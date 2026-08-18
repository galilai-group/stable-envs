"""Is an Atari frame a function of the 128-byte console RAM?  Measured on the official emulator.

The claim under test is the one that makes "store states, not images" work: frame = f(state) for some
compact state.  For ALE the tempting candidate is the 128-byte RIOT RAM, which ale-py exposes directly.
The test that settles it does not need any argument about TIA internals -- it only needs two emulator
states that agree on all 128 RAM bytes and disagree on the next frame:

  branch 1: restore full system state S_t, write RAM = ram_t (a no-op), emulate one frame, read screen
  branch 2: restore a DIFFERENT full system state S_u, write RAM = ram_t, emulate one frame, read screen

Both branches enter the frame with byte-identical RAM.  If the two screens differ, RAM is not the state
and no RAM -> pixels renderer can exist.

Also measured: the size of what the emulator itself considers a complete state (cloneState /
cloneSystemState), the screen size, whether restoring reproduces a frame bit for bit, and the
single-core emulation throughput (which is what "regenerate frames from actions" would actually cost).

Run under SLURM.  Writes JSON to --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def h(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def state_bytes(st):
    """Serialized length of an ALEState, however this ale-py version exposes it."""
    for meth in ("serialize", "serializeAsString", "getBytes"):
        if hasattr(st, meth):
            try:
                return len(getattr(st, meth)())
            except Exception:
                pass
    import pickle
    return len(pickle.dumps(st))


def write_ram(ale, ram):
    for i, v in enumerate(ram):
        ale.setRAM(int(i), int(v))


def probe(rom_path, n_burn=300, n_collide=5000, seed=0):
    from ale_py import Action, ALEInterface, LoggerMode

    ale = ALEInterface()
    ALEInterface.setLoggerMode(LoggerMode.Error)
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.loadROM(rom_path)
    acts = ale.getMinimalActionSet()
    rng = np.random.RandomState(seed)

    hh, ww = ale.getScreenDims()
    ram_size = ale.getRAMSize()

    ale.reset_game()
    for _ in range(n_burn):
        ale.act(acts[rng.randint(len(acts))])
    S_t = ale.cloneSystemState()
    ram_t = ale.getRAM().copy()
    scr_t = ale.getScreen().copy()
    sz_sys = state_bytes(S_t)
    sz_lite = state_bytes(ale.cloneState())

    # sanity: does restoring the full system state reproduce the frame bit for bit?
    for _ in range(50):
        ale.act(acts[rng.randint(len(acts))])
    ale.restoreSystemState(S_t)
    restore_exact = h(ale.getScreen()) == h(scr_t)
    restore_ram_exact = h(ale.getRAM()) == h(ram_t)

    # a DIFFERENT full state, reached later in the same episode
    ale.restoreSystemState(S_t)
    for _ in range(200):
        ale.act(acts[rng.randint(len(acts))])
    S_u = ale.cloneSystemState()
    ram_u = ale.getRAM().copy()

    NOOP = Action.NOOP

    # branch 1
    ale.restoreSystemState(S_t)
    write_ram(ale, ram_t)                       # no-op, but goes through the same code path
    pre1 = ale.getRAM().copy()
    ale.act(NOOP)
    f1, r1 = ale.getScreen().copy(), ale.getRAM().copy()

    # branch 2
    ale.restoreSystemState(S_u)
    write_ram(ale, ram_t)
    pre2 = ale.getRAM().copy()
    ale.act(NOOP)
    f2, r2 = ale.getScreen().copy(), ale.getRAM().copy()

    ram_agree_before = bool(np.array_equal(pre1, pre2))
    frame_agree_after = bool(np.array_equal(f1, f2))
    ram_agree_after = bool(np.array_equal(r1, r2))
    frac_px_differ = float((f1 != f2).mean())

    # collision search: over a long rollout, does the same 128-byte RAM ever coexist with a
    # different screen?
    ale.reset_game()
    seen = {}
    collisions = 0
    n_repeat_ram = 0
    for _ in range(n_collide):
        ale.act(acts[rng.randint(len(acts))])
        rk, sk = h(ale.getRAM()), h(ale.getScreen())
        if rk in seen:
            n_repeat_ram += 1
            if seen[rk] != sk:
                collisions += 1
        else:
            seen[rk] = sk

    # single-core throughput of the emulator (the cost of "regenerate from actions")
    ale.reset_game()
    t0 = time.perf_counter()
    N = 3000
    for _ in range(N):
        ale.act(acts[rng.randint(len(acts))])
    fps = N / (time.perf_counter() - t0)

    return {
        "rom": Path(rom_path).stem,
        "screen_hw": [int(hh), int(ww)],
        "screen_bytes_indexed": int(hh * ww),
        "screen_bytes_rgb": int(hh * ww * 3),
        "ram_bytes": int(ram_size),
        "clone_state_bytes": int(sz_lite),
        "clone_system_state_bytes": int(sz_sys),
        "restore_reproduces_frame": bool(restore_exact),
        "restore_reproduces_ram": bool(restore_ram_exact),
        "ram_identical_before_step": ram_agree_before,
        "frames_identical_after_step": frame_agree_after,
        "ram_identical_after_step": ram_agree_after,
        "frac_pixels_differing": frac_px_differ,
        "ram_t_vs_ram_u_differ_bytes": int((ram_t != ram_u).sum()),
        "collision_steps": n_collide,
        "ram_repeats": n_repeat_ram,
        "ram_repeats_with_different_screen": collisions,
        "emu_fps_single_core": float(fps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=["pong", "breakout", "space_invaders", "seaquest",
                                                   "ms_pacman", "montezuma_revenge", "freeway",
                                                   "asterix", "boxing", "qbert"])
    ap.add_argument("--n-collide", type=int, default=5000)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/ale_state.json")
    args = ap.parse_args()

    import ale_py.roms as roms
    romdir = Path(roms.__file__).parent

    rows = []
    for g in args.games:
        p = romdir / f"{g}.bin"
        if not p.exists():
            print(f"skip {g}: no rom")
            continue
        r = probe(str(p), n_collide=args.n_collide)
        rows.append(r)
        print(f"{r['rom']:<20} ram {r['ram_bytes']}B  cloneState {r['clone_state_bytes']}B  "
              f"cloneSystemState {r['clone_system_state_bytes']}B  screen {r['screen_bytes_indexed']}B  "
              f"| same-RAM->same-frame: {r['frames_identical_after_step']} "
              f"(differ on {100*r['frac_pixels_differing']:.1f}% px)  "
              f"| ram-repeats {r['ram_repeats']} of which different-screen {r['ram_repeats_with_different_screen']}  "
              f"| {r['emu_fps_single_core']:.0f} fps", flush=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
