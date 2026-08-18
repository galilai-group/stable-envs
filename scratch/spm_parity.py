"""Parity for SimplePointMaze, plus the assertion that our default wall layout IS the official one."""
import json, sys, time
import numpy as np, torch
import stable_envs as se
from stable_envs.testing import conformance as C

name = "SimplePointMaze"
rec = {}
env = se.make(name, batch=8, device="cuda")
cls = type(env)
ref = C.reference_env(cls, env.size)
u = ref.unwrapped
ref.reset(seed=0, options={"variation": []})
live_shape = np.asarray(u.variation_space["walls"]["shape"].value, np.float32)
live_pos = np.asarray(u.variation_space["walls"]["positions"].value, np.float32)
from stable_envs.envs.simple_point_maze import DEFAULT_WALL_SHAPE, DEFAULT_WALL_POS
rec["default_wall_shape_err"] = float(np.abs(live_shape - DEFAULT_WALL_SHAPE).max())
rec["default_wall_pos_err"] = float(np.abs(live_pos - DEFAULT_WALL_POS).max())
rec["live_number"] = int(u.variation_space["walls"]["number"].value)
rec["live_radius"] = float(np.asarray(u.variation_space["agent"]["radius"].value).reshape(-1)[0])
print("wall layout err", rec["default_wall_shape_err"], rec["default_wall_pos_err"], flush=True)

for fn in (C.check_state_roundtrip, C.check_state_dims_matter, C.check_batch_invariance,
           C.check_variations_do_something, C.check_per_env_variation, C.check_style_is_state,
           C.check_default_style_is_official, C.check_colors_apply_exactly, C.check_determinism,
           C.check_backend_agreement):
    rec[fn.__name__] = fn(env)
    print(" ", fn.__name__, rec[fn.__name__], flush=True)

states = env.sample_states(48, rng=np.random.default_rng(3))
official = cls.reference_render(ref, states)
ours = env.render(states).cpu().numpy()
rec["parity"] = C.pixel_agreement(ours, official)
print(" parity", rec["parity"], flush=True)
d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
if d.max():
    bad = np.argwhere(d > 0)
    rec["n_bad"] = int((d > 0).sum())
    rec["examples"] = [{"idx": [int(v) for v in b], "ours": [int(v) for v in ours[tuple(b)]],
                        "official": [int(v) for v in official[tuple(b)]]}
                       for b in bad[::max(1, len(bad)//15)][:15]]
    # where are they: border, or interior?
    r, c = bad[:, 1], bad[:, 2]
    rec["bad_on_border"] = int(((r < 3) | (r > 496) | (c < 3) | (c > 496)).sum())
    for x in rec["examples"]: print("   ", x, flush=True)
    print("   n_bad", rec["n_bad"], "on border", rec["bad_on_border"], flush=True)

rec["step_vs_reference"] = C.check_step_matches_reference(env, steps=60, episodes=8)
print(" step", {k: v for k, v in rec["step_vs_reference"].items() if "err" in k or "exact" in k}, flush=True)

big = se.make(name, batch=256, device="cuda")
st = torch.as_tensor(big.sample_states(256, rng=np.random.default_rng(0)), device="cuda")
buf = torch.empty((256, 500, 500, 3), dtype=torch.uint8, device="cuda")
for _ in range(3): big.render(st, out=buf)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(10): big.render(st, out=buf)
torch.cuda.synchronize()
rec["fps_gpu"] = 256*10/(time.perf_counter()-t0)
t0 = time.perf_counter(); cls.reference_render(ref, states[:20]); rec["fps_reference"] = 20/(time.perf_counter()-t0)
rec["speedup"] = rec["fps_gpu"]/rec["fps_reference"]
print(" fps", rec["fps_gpu"], rec["fps_reference"], rec["speedup"], flush=True)
json.dump(rec, open("/mnt/vast/home/randall.balestriero/stable_envs_logs/agentB/spm_parity.json","w"), indent=1, default=str)
