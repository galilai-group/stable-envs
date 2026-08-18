"""Atari storage economics on one node: bytes per step and decode cost per frame.

The replay claim is that (ROM, seed, actions) = 1 byte/step regenerates the
frames, so nobody has to store them.  The comparison that decides whether that
matters is regeneration cost against DECODE cost -- the same G4 comparison the
repo runs for every other family.  Atari frames are palette art, so PNG (or
the raw 1-byte-per-pixel indexed screen) is the honest stored form; JPEG would
be lossy and break the exactness the whole argument is about.

Reports, on one core: PNG encode/decode f/s, raw indexed-screen memcpy f/s,
stored bytes per frame in each format, and the ale-py single-core replay rate.

Usage: python bench_atari_storage.py <out.json>
"""
import json
import os
import sys
import time

import cv2
import numpy as np

GAMES = ["pong", "breakout", "space_invaders"]
N = 500


def collect(rom, n):
    import ale_py
    from ale_py import roms

    ale = ale_py.ALEInterface()
    ale.setLoggerMode(ale_py.LoggerMode.Error)
    ale.setInt("frame_skip", 1)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setInt("random_seed", 0)
    ale.loadROM(roms.get_rom_path(rom))
    acts = ale.getMinimalActionSet()
    rng = np.random.default_rng(0)
    ale.reset_game()
    rgb, idx = [], []
    t0 = time.perf_counter()
    for _ in range(n):
        ale.act(acts[rng.integers(len(acts))])
        rgb.append(ale.getScreenRGB())
        idx.append(ale.getScreen())
        if ale.game_over():
            ale.reset_game()
    dt = time.perf_counter() - t0
    return np.stack(rgb), np.stack(idx), n / dt


def main():
    out = sys.argv[1]
    res = {"node": os.uname().nodename, "n_frames": N,
           "cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())),
           "cv2": cv2.__version__, "games": {}}
    cv2.setNumThreads(1)
    for g in GAMES:
        rgb, idx, collect_fps = collect(g, N)
        png = [cv2.imencode(".png", f[:, :, ::-1])[1] for f in rgb]
        png_idx = [cv2.imencode(".png", f)[1] for f in idx]

        t0 = time.perf_counter()
        for b in png:
            cv2.imdecode(b, cv2.IMREAD_COLOR)
        dec_rgb = N / (time.perf_counter() - t0)

        t0 = time.perf_counter()
        for b in png_idx:
            cv2.imdecode(b, cv2.IMREAD_GRAYSCALE)
        dec_idx = N / (time.perf_counter() - t0)

        t0 = time.perf_counter()
        for b in png[:200]:
            cv2.imencode(".png", cv2.imdecode(b, cv2.IMREAD_COLOR))
        enc_rgb = 200 / (time.perf_counter() - t0)

        res["games"][g] = {
            "ale_replay_frames_per_s_1core_with_getScreen": collect_fps,
            "png_rgb_decode_frames_per_s_1core": dec_rgb,
            "png_indexed_decode_frames_per_s_1core": dec_idx,
            "png_rgb_encode_frames_per_s_1core": enc_rgb,
            "bytes_raw_rgb": int(rgb[0].nbytes),
            "bytes_raw_indexed": int(idx[0].nbytes),
            "bytes_png_rgb_mean": float(np.mean([len(b) for b in png])),
            "bytes_png_indexed_mean": float(np.mean([len(b) for b in png_idx])),
            "bytes_replay_per_step": 1,
        }
        print(g, json.dumps(res["games"][g], indent=1), flush=True)
        json.dump(res, open(out, "w"), indent=1)
    json.dump(res, open(out, "w"), indent=1)
    print("WROTE", out)


if __name__ == "__main__":
    main()
