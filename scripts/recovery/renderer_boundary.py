"""Digest identical frozen rows on baseline/current in separate processes, no new corpus."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import stable_envs as se

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--device", default="cuda")
a = p.parse_args()
results = {}
for name in ("CartPole", "Pendulum", "PushT", "TwoRoom", "DMCReacherHard", "DMCCartpoleBalance", "DMCHopperHop"):
    path = Path("tests/fixtures/regression/join/step4/states") / f"{name}.npz"
    corpus = np.load(path)
    rows = corpus["states"][:4]
    with se.make(name, num_envs=len(rows), device=a.device) as env:
        render = getattr(env, "_render", env.render)
        frame = render(rows).cpu().numpy()
        results[name] = dict(
            rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
            frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
            shape=list(frame.shape),
        )
    print(name, results[name], flush=True)
Path(a.output).write_text(json.dumps(results, indent=2) + "\n")
