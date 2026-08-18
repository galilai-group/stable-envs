"""Full parity sweep of the numpy reference renderer against the shipped procgen renderer."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refrender as R  # noqa: E402
import state_io as S  # noqa: E402


def main():
    from procgen import ProcgenGym3Env
    games = [a for a in sys.argv[1:] if not a.startswith("-")] or ["maze", "heist", "miner"]
    levels = int(os.environ.get("LEVELS", "20"))
    steps = int(os.environ.get("STEPS", "12"))
    rng = np.random.default_rng(0)
    for game in games:
        tot = bad = worst = 0
        nf = 0
        badframes = []
        for level in range(levels):
            env = ProcgenGym3Env(num=1, env_name=game, distribution_mode="hard", rand_seed=0,
                                 start_level=level, num_levels=1, use_backgrounds=True)
            for step in range(steps):
                env.act(rng.integers(0, 15, size=1).astype(np.int32))
                st = S.parse(env.get_state()[0])
                _, obs, _ = env.observe()
                got = np.asarray(obs["rgb"][0])
                pred = R.render(game, st["grid"], st["entities"], st["background_index"],
                                st["bg_pct_x"], st.get("has_keys", ()))
                d = np.abs(pred.astype(np.int16) - got.astype(np.int16))
                nb = int((d.max(-1) > 0).sum())
                bad += nb
                tot += 64 * 64
                worst = max(worst, int(d.max()))
                nf += 1
                if nb:
                    badframes.append((level, step, nb, int(d.max())))
        print(f"{game:6s}: {nf} frames  exact_px {100*(1-bad/tot):.4f}%  "
              f"exact_frames {nf - len({(l, s) for l, s, _, _ in badframes})}/{nf}  "
              f"max|diff| {worst}")
        for b in badframes[:8]:
            print(f"    level {b[0]} step {b[1]}: {b[2]} bad px, max {b[3]}")


if __name__ == "__main__":
    main()
