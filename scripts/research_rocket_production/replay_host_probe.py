"""Read-only foreign-host replay checks under explicitly recorded math profiles."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import stable_envs as se
from stable_envs._adapters.rocket.replay import RocketReplay

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
raw = np.load(a.input, allow_pickle=False)
report = {
    "job": os.environ.get("SLURM_JOB_ID"),
    "profile": {k: os.environ.get(k) for k in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES")},
    "cases": [],
}
from threadpoolctl import threadpool_info

report["threadpools"] = threadpool_info()
original = RocketReplay.get_state
captured = []


def capture(self):
    data = original(self)
    captured.append(data)
    return data


RocketReplay.get_state = capture
try:
    with se.make(
        "PyFlytRocketLanding",
        num_envs=2,
        device="cpu",
        size=96,
        simulator_config=dict(render_mode="rgb_array", render_resolution=(96, 96)),
    ) as env:
        for t in range(8):
            prefix = f"state/{t}/"
            state = TensorDict(
                {
                    tuple(k[len(prefix) :].split("/")): torch.from_numpy(raw[k])
                    for k in raw.files
                    if k.startswith(prefix)
                },
                [2],
            )
            try:
                env.set_state(state)
                case = {"t": t, "status": "PASS_EXACT"}
            except ValueError as e:
                case = {"t": t, "status": "REJECTED", "error": str(e)}
            report["cases"].append(case)
finally:
    RocketReplay.get_state = original
report["input_sha256"] = hashlib.sha256(a.input.read_bytes()).hexdigest()
report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(a.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
