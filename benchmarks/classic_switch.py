"""Time the classic-control backends at batches 1, 2, 4, ..., 4096, as stable-envs tune times them.

usage: python -m benchmarks.classic_switch [DOMAIN,...]

On the CPU device it times Gymnasium and NumPy; on CUDA it times Gymnasium and NumPy on the host and Torch on the GPU.
The values that ``classic_control/simulation/batched.py`` ships come from this output.
"""

import json
import sys

import torch

from stable_envs._execution import backends
from stable_envs.envs.classic_control.simulation.batched import CPU_MIN_ENVS, _measure


def main():
    domains = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CPU_MIN_ENVS)
    out = {}
    for domain in domains:
        for dev, names in (("cpu", ("gym", "numpy")), ("cuda", ("gym", "numpy", "torch"))):
            device = torch.device(dev)
            measure = _measure(domain, device)
            rows = []
            for b in [1 << k for k in range(13)]:
                times = {name: backends.time_backend(measure, name, b, device) * 1e6 for name in names}
                rows.append((b, times))
                print(domain, dev, b, {k: round(v) for k, v in times.items()}, flush=True)
            out[f"{domain}/{dev}"] = rows
    print(json.dumps(out))


# Acrobot batches step in worker processes, which import this module again.
if __name__ == "__main__":
    main()
