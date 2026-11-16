"""Time the classic-control backends at batches 1, 2, 4, ..., 4096, as stable-envs tune times them.

usage: python -m benchmarks.classic_switch [DOMAIN,...]

On the CPU device it times Gymnasium and NumPy; on CUDA it times Gymnasium and NumPy on the host and Torch on the GPU.
The values that ``classic_control/simulation/batched.py`` ships come from this output.
"""

import json
import sys

import torch

import stable_envs as snv
from stable_envs import tuning

#: Class of each classic-control task, and whether the task has the compiled ``"numba"`` step.
CLASSES = {
    "cartpole": ("CartPole", True),
    "pendulum": ("Pendulum", False),
    "acrobot": ("Acrobot", False),
    "mountain_car": ("MountainCar", True),
    "mountain_car_continuous": ("MountainCarContinuous", False),
}


def _measure(name: str, path: str, dev: str):
    """The ``measure`` that building ``name`` on ``dev`` registers for ``path``."""
    with snv.make(name, num_envs=2, device=dev):
        pass
    return next(measure for p, _, measure in tuning.simulation_paths(dev) if p == path)


def main():
    domains = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CLASSES)
    out = {}
    for domain in domains:
        name, compiled = CLASSES[domain]
        extra = ("numba",) if compiled else ()
        for dev, names, path in (
            ("cpu", ("gym", "numpy", *extra), f"classic-cpu/{domain}"),
            ("cuda", ("gym", "numpy", *extra, "torch"), f"classic-sim/{domain}"),
        ):
            device = torch.device(dev)
            measure = _measure(name, path, dev)
            rows = []
            for b in [1 << k for k in range(13)]:
                times = {backend: tuning.time_backend(measure, backend, b, device) * 1e6 for backend in names}
                rows.append((b, times))
                print(domain, dev, b, {k: round(v) for k, v in times.items()}, flush=True)
            out[f"{domain}/{dev}"] = rows
    print(json.dumps(out))


# Acrobot batches step in worker processes, which import this module again.
if __name__ == "__main__":
    main()
