"""Isolate CPU/software-GL and native-GPU measurements on one allocated node."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .bridges import CLASSIC, DMC, NAV
from .common import catalog, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cohort", choices=["native", "navigation", "dmc", "fetch", "craftax", "manipulation", "rocket"])
    p.add_argument("output", type=Path)
    p.add_argument("--only", nargs="+")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    chosen = []
    for row in catalog().values():
        name = row["name"]
        if args.only and name not in args.only:
            continue
        if (
            (args.cohort == "native" and name in {*CLASSIC, *NAV})
            or (args.cohort == "navigation" and name in NAV)
            or (args.cohort == "dmc" and name in {*DMC, "PointMazeLargeNavigate"})
            or (args.cohort == "fetch" and name.startswith("Fetch"))
            or (args.cohort == "craftax" and name.startswith("Craftax"))
            or (args.cohort == "manipulation" and name in {"OGBCube", "OGBScene"})
            or (args.cohort == "rocket" and name == "PyFlytRocketLanding")
        ):
            chosen.append(row)
    results = []
    for row in chosen:
        directory = args.output / row["name"]
        directory.mkdir()
        status = dict(registration=row["registration"], phases={})
        for phase in ["cpu", "gpu"]:
            env = os.environ.copy()
            if phase == "cpu":
                env.update(
                    MUJOCO_GL="glfw", PYOPENGL_PLATFORM="glx", LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe"
                )
            else:
                env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
                for key in ["LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER"]:
                    env.pop(key, None)
            if args.cohort == "craftax":
                env.update(
                    JAX_PLATFORMS="cpu" if phase == "cpu" else "cuda,cpu",
                    CRAFTAX_JAX_BACKEND="cpu" if phase == "cpu" else "gpu",
                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                )
            if args.cohort == "fetch" and phase == "cpu":
                env["PYTHONPATH"] = os.environ["FETCH_CPU_OVERLAY"] + ":" + env["PYTHONPATH"]
            command = [
                sys.executable,
                "-m",
                "scripts.benchmarks.renderer_table.collect",
                row["registration"],
                phase,
                str(directory / phase),
            ]
            if phase == "gpu":
                command += ["--corpus", str(directory / "cpu")]
            with (directory / (phase + ".log")).open("w") as log:
                code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            status["phases"][phase] = dict(returncode=code, command=command)
            write(directory / "status.json", status)
            print(json.dumps(status), flush=True)
            if code and phase == "cpu":
                break
        results.append(status)
        write(args.output / "status.json", results)


if __name__ == "__main__":
    main()
