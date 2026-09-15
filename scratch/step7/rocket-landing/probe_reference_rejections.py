"""M4 probe: which ``sample_states`` draws does the PyBullet reference bridge REJECT, and why?

Job 6406061's ``test_visual_tier_matches_its_declared_fidelity`` failed inside ``_reference_frame``
("did not reach the requested pose (pos 2.662e-01 m)") on pytest's own 64-state draw, while the four
measure_tier draws (seeds 1000..1003) and the under-variation draw had 0 rejections. CPU only
(PyBullet DIRECT); runs locally in /Users/samib/se7-rocket-venv. Records, per rejected state, the
height, tilt, |v|, |w| and the error text, so the mechanism can be read off the geometry.
"""
import json, os, sys, time, platform
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from stable_envs.envs import rocket_landing as RL
OUT = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "tier")


class _Rng:
    def __init__(self, seed): self.rng = np.random.default_rng(seed)


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000, 1001, 1002, 1003, 20260809]
    ref = RL.PFRocketLanding.reference_env(480)
    sty = RL._default_style_row()
    res = {"host": platform.node(), "seeds": seeds, "n_per_seed": 64, "rejections": [], "per_seed": {}}
    t0 = time.time()
    for seed in seeds:
        st = RL.PFRocketLanding.sample_states(_Rng(seed), 64)          # the env's own draw at this seed
        rej = 0
        for i, s in enumerate(st):
            try:
                RL._reference_frame(ref, np.asarray(s, np.float64), sty)
            except AssertionError as e:
                rej += 1
                q = s[3:7] / np.linalg.norm(s[3:7])
                Rm = RL.quat_to_mat(q[None])[0]
                tilt = float(np.degrees(np.arccos(np.clip(Rm[2, 2], -1, 1))))   # body z vs world z
                res["rejections"].append({"seed": seed, "i": int(i), "height": float(s[12]), "tilt_deg": tilt,
                                          "speed": float(np.linalg.norm(s[7:10])), "omega": float(np.linalg.norm(s[0:3])),
                                          "error": str(e)[:160]})
        res["per_seed"][str(seed)] = rej
        print("seed", seed, "rejections", rej, flush=True)
    hs = np.array([r["height"] for r in res["rejections"]]) if res["rejections"] else np.zeros(0)
    res["summary"] = {"total_states": 64 * len(seeds), "total_rejections": len(res["rejections"]),
                      "rejected_height_max": float(hs.max()) if len(hs) else None,
                      "rejected_height_min": float(hs.min()) if len(hs) else None,
                      "seconds": round(time.time() - t0, 1)}
    print(json.dumps(res["summary"]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "reference_rejections_local.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
