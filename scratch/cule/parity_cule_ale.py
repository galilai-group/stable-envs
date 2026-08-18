"""Bit-parity: CuLE frames vs ale-py frames for the same (ROM, seed, actions).

The claim under test is the replay route: a rollout is (ROM, seed, action
sequence) = 1 byte/step, and replaying the emulator regenerates the frames.
That only replaces stored frames if the replayed frames ARE the frames.  CuLE
is a re-implementation of Stella's 6507 + TIA, so "same frames" is a
measurement, not an assumption.

Protocol, per ROM:
  * CuLE (GPU backend and CPU backend): Env(rom, num_envs=1, rgb, frameskip=1,
    repeat_prob=0, max_noop_steps=1), reset(seeds=0) -- eval mode, so reset
    performs no random noops.
  * ale-py: ALEInterface, frame_skip=1, repeat_action_probability=0, loadROM,
    reset_game().
  * Drive all three with the SAME ALE action codes (drawn from CuLE's minimal
    action set; the two action sets are compared and recorded).
  * After every single emulated frame, diff the 210x160x3 observation and the
    128-byte RAM.

Two action streams per ROM:
  * "rand": uniform over the minimal action set.
  * "noop": all NOOP.  With no action semantics in play, a shift search over
    +-30 frames separates "different emulation" from "same emulation, different
    boot phase" -- which a random stream cannot distinguish, because a phase
    offset also desynchronises the actions.

Three comparisons: CuLE-GPU vs ALE, CuLE-CPU vs ALE, CuLE-GPU vs CuLE-CPU.

Usage: python parity_cule_ale.py <out.json> [--steps N] [--roms a,b,c]
"""
import argparse
import json
import os

import numpy as np
import torch

ROMS = [
    "pong", "breakout", "space_invaders", "seaquest", "asterix",
    "qbert", "ms_pacman", "freeway", "boxing", "road_runner",
    "chopper_command", "alien",
]


def cule_rollout(rom, actions_idx, device, steps):
    from torchcule.atari import Env

    env = Env(rom, 1, color_mode="rgb", device=device, rescale=False,
              frameskip=1, repeat_prob=0.0, episodic_life=False,
              max_noop_steps=1)
    action_codes = [int(a) for a in env.action_set.cpu().numpy()]
    seeds = torch.zeros(1, dtype=torch.int32, device=device)
    env.reset(seeds=seeds)
    frames = np.empty((steps, env.height, env.width, 3), dtype=np.uint8)
    rams = np.empty((steps, 128), dtype=np.uint8)
    rewards = np.empty(steps, dtype=np.float32)
    for t in range(steps):
        a = torch.tensor([actions_idx[t]], dtype=torch.uint8, device=device)
        obs, rew, done, info = env.step(a)
        frames[t] = obs[0].cpu().numpy()
        rams[t] = env.ram[0].cpu().numpy()
        rewards[t] = float(rew[0])
    del env
    return frames, rams, rewards, action_codes


def _as_int(a):
    """ale-py returns Action enum members, which pybind11 does not convert with int()."""
    return int(a.value) if hasattr(a, "value") else int(a)


# CuLE encodes joystick actions as a BITMASK (cule/atari/flags.hpp:
# RIGHT=1, LEFT=2, DOWN=4, UP=8, FIRE=16); ALE numbers them 0..17 in a fixed
# order.  The raw integers therefore mean different things in the two
# emulators, and feeding one's code to the other silently plays a different
# game.  Measured, not read off: pong's CuLE set [0,16,1,2,17,18] maps to
# ALE's [0,1,3,4,11,12], which is exactly ale-py's minimal set for pong.
CULE_BIT_RIGHT, CULE_BIT_LEFT, CULE_BIT_DOWN, CULE_BIT_UP, CULE_BIT_FIRE = 1, 2, 4, 8, 16
_ALE_ORDER = {
    (0, 0, 0, 0, 0): 0,   # NOOP
    (1, 0, 0, 0, 0): 1,   # FIRE
    (0, 1, 0, 0, 0): 2,   # UP
    (0, 0, 0, 0, 1): 3,   # RIGHT
    (0, 0, 0, 1, 0): 4,   # LEFT
    (0, 0, 1, 0, 0): 5,   # DOWN
    (0, 1, 0, 0, 1): 6,   # UPRIGHT
    (0, 1, 0, 1, 0): 7,   # UPLEFT
    (0, 0, 1, 0, 1): 8,   # DOWNRIGHT
    (0, 0, 1, 1, 0): 9,   # DOWNLEFT
    (1, 1, 0, 0, 0): 10,  # UPFIRE
    (1, 0, 0, 0, 1): 11,  # RIGHTFIRE
    (1, 0, 0, 1, 0): 12,  # LEFTFIRE
    (1, 0, 1, 0, 0): 13,  # DOWNFIRE
    (1, 1, 0, 0, 1): 14,  # UPRIGHTFIRE
    (1, 1, 0, 1, 0): 15,  # UPLEFTFIRE
    (1, 0, 1, 0, 1): 16,  # DOWNRIGHTFIRE
    (1, 0, 1, 1, 0): 17,  # DOWNLEFTFIRE
}


def cule_code_to_ale(c):
    key = (int(bool(c & CULE_BIT_FIRE)), int(bool(c & CULE_BIT_UP)),
           int(bool(c & CULE_BIT_DOWN)), int(bool(c & CULE_BIT_LEFT)),
           int(bool(c & CULE_BIT_RIGHT)))
    return _ALE_ORDER[key]


def ale_rollout(rom, action_codes_seq, steps):
    import ale_py
    from ale_py import roms

    ale = ale_py.ALEInterface()
    ale.setLoggerMode(ale_py.LoggerMode.Error)
    ale.setInt("frame_skip", 1)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setInt("random_seed", 0)
    ale.loadROM(roms.get_rom_path(rom))
    minimal = [_as_int(a) for a in ale.getMinimalActionSet()]
    ale.reset_game()
    h, w = ale.getScreenDims()
    frames = np.empty((steps, h, w, 3), dtype=np.uint8)
    rams = np.empty((steps, 128), dtype=np.uint8)
    rewards = np.empty(steps, dtype=np.float32)
    for t in range(steps):
        rewards[t] = ale.act(action_codes_seq[t])
        frames[t] = ale.getScreenRGB()
        rams[t] = ale.getRAM()
    return frames, rams, rewards, minimal


def eq_frac(a, b):
    n = min(len(a), len(b))
    return float(np.mean([np.array_equal(a[i], b[i]) for i in range(n)]))


def best_shift(a, b, max_shift=30):
    """Match fraction at the best alignment shift (catches a boot-phase offset)."""
    best_s, best_f = 0, -1.0
    n = len(a)
    for s in range(-max_shift, max_shift + 1):
        x, y = (a[s:], b[: n - s]) if s >= 0 else (a[: n + s], b[-s:])
        if len(x) < 10:
            continue
        f = eq_frac(x, y)
        if f > best_f:
            best_s, best_f = s, f
    return best_s, best_f


def compare(name, fa, ra, fb, rb, with_shift):
    if fa.shape[1:] != fb.shape[1:]:
        return {name: {"shape_mismatch": [list(fa.shape[1:]), list(fb.shape[1:])],
                       "frames_exact_frac": 0.0, "frac_pixels_exact": 0.0,
                       "max_abs_diff": None, "ram_exact_frac": None,
                       "first_frame_divergence": 0, "n_frames": int(len(fa))}}
    n = min(len(fa), len(fb))
    eq = np.array([np.array_equal(fa[t], fb[t]) for t in range(n)])
    d = np.abs(fa[:n].astype(np.int16) - fb[:n].astype(np.int16))
    rameq = np.array([np.array_equal(ra[t], rb[t]) for t in range(n)])
    out = {
        "frames_exact_frac": float(eq.mean()),
        "frames_exact_n": int(eq.sum()),
        "n_frames": int(n),
        "first_frame_divergence": int(np.argmin(eq)) if not eq.all() else None,
        "max_abs_diff": int(d.max()),
        "mean_abs_diff": float(d.mean()),
        "frac_pixels_exact": float((d == 0).mean()),
        "ram_exact_frac": float(rameq.mean()),
        "first_ram_divergence": int(np.argmin(rameq)) if not rameq.all() else None,
    }
    if with_shift:
        s, f = best_shift(fa, fb)
        out["best_shift"] = int(s)
        out["frames_exact_frac_at_best_shift"] = float(f)
    return {name: out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--roms", default=",".join(ROMS))
    ap.add_argument("--skip-cpu-backend", action="store_true")
    args = ap.parse_args()

    roms = args.roms.split(",")
    rng = np.random.default_rng(12345)
    import ale_py
    res = {"steps": args.steps, "node": os.uname().nodename,
           "ale_py_version": ale_py.__version__, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "roms": {}}

    for rom in roms:
        entry = {}
        try:
            from torchcule.atari import Env
            probe = Env(rom, 1, color_mode="gray", device="cpu", rescale=True,
                        frameskip=1, repeat_prob=0.0, max_noop_steps=1)
            action_codes = [int(a) for a in probe.action_set.cpu().numpy()]
            del probe
            import ale_py as _ap
            from ale_py import roms as _roms
            _ale = _ap.ALEInterface()
            _ale.setLoggerMode(_ap.LoggerMode.Error)
            _ale.loadROM(_roms.get_rom_path(rom))
            ale_set = [_as_int(a) for a in _ale.getMinimalActionSet()]
            del _ale
            mapped = [cule_code_to_ale(c) for c in action_codes]
            entry["cule_action_set"] = action_codes
            entry["cule_action_set_as_ale_codes"] = mapped
            entry["ale_minimal_action_set"] = ale_set
            entry["action_sets_match"] = mapped == ale_set
            entry["action_sets_match_as_sets"] = sorted(mapped) == sorted(ale_set)

            noop_code = 0
            noop_idx = [action_codes.index(noop_code)] * args.steps \
                if noop_code in action_codes else [0] * args.steps

            for stream in ("rand", "noop"):
                if stream == "rand":
                    idx = list(rng.integers(len(action_codes), size=args.steps))
                else:
                    idx = noop_idx
                codes = [mapped[i] for i in idx]

                gf, gram, grew, _ = cule_rollout(rom, idx, "cuda", args.steps)
                af, aram, arew, _ = ale_rollout(rom, codes, args.steps)
                sub = {}
                sub.update(compare("cule_gpu_vs_ale", gf, gram, af, aram,
                                   with_shift=(stream == "noop")))
                if not args.skip_cpu_backend:
                    cf, cram, crew, _ = cule_rollout(rom, idx, "cpu", args.steps)
                    sub.update(compare("cule_cpu_vs_ale", cf, cram, af, aram,
                                       with_shift=(stream == "noop")))
                    sub.update(compare("cule_gpu_vs_cule_cpu", gf, gram, cf, cram,
                                       with_shift=False))
                    sub["cule_cpu_total_reward"] = float(crew.sum())
                sub["cule_gpu_total_reward"] = float(grew.sum())
                sub["ale_total_reward"] = float(arew.sum())
                entry[stream] = sub
        except Exception as e:  # noqa: BLE001
            entry["error"] = repr(e)
        res["roms"][rom] = entry
        printable = {k: v for k, v in entry.items()
                     if k not in ("cule_action_set", "cule_action_set_as_ale_codes", "ale_minimal_action_set")}
        print(rom, json.dumps(printable)[:1200], flush=True)
        json.dump(res, open(args.out, "w"), indent=1)

    ok = [v for v in res["roms"].values() if "rand" in v]
    if ok:
        def agg(stream, cmp, field):
            vals = [v[stream][cmp][field] for v in ok if cmp in v[stream]]
            return float(np.mean(vals)) if vals else None
        res["summary"] = {
            "n_roms": len(ok),
            "rand_gpu_vs_ale_mean_frames_exact_frac": agg("rand", "cule_gpu_vs_ale", "frames_exact_frac"),
            "rand_gpu_vs_ale_mean_frac_pixels_exact": agg("rand", "cule_gpu_vs_ale", "frac_pixels_exact"),
            "rand_gpu_vs_ale_mean_ram_exact_frac": agg("rand", "cule_gpu_vs_ale", "ram_exact_frac"),
            "noop_gpu_vs_ale_mean_frames_exact_frac": agg("noop", "cule_gpu_vs_ale", "frames_exact_frac"),
            "noop_gpu_vs_ale_mean_frames_exact_frac_at_best_shift":
                agg("noop", "cule_gpu_vs_ale", "frames_exact_frac_at_best_shift"),
            "rand_gpu_vs_cpu_mean_frames_exact_frac": agg("rand", "cule_gpu_vs_cule_cpu", "frames_exact_frac"),
            "n_roms_action_sets_match": int(sum(
                bool(res["roms"][r].get("action_sets_match")) for r in res["roms"])),
            "max_abs_diff_overall": int(max(
                v["rand"]["cule_gpu_vs_ale"]["max_abs_diff"] for v in ok)),
            "n_roms_fully_exact_rand": int(sum(
                v["rand"]["cule_gpu_vs_ale"]["frames_exact_frac"] == 1.0 for v in ok)),
        }
    json.dump(res, open(args.out, "w"), indent=1)
    print("WROTE", args.out)
    print(json.dumps(res.get("summary", {}), indent=1))


if __name__ == "__main__":
    main()
