"""Run paired DMC checks with isolated CPU/software-GL and GPU/EGL processes."""

import argparse
import json
import os
import select
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", required=True)
    p.add_argument("--reference-adapters", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for device in ("cpu", "cuda"):
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", LP_NUM_THREADS="1")
        xvfb = None
        try:
            if device == "cpu":
                env.update(MUJOCO_GL="glfw", PYOPENGL_PLATFORM="glx", LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe", CUDA_VISIBLE_DEVICES="")
                read_fd, write_fd = os.pipe()
                try:
                    with (args.output / "xvfb.log").open("w") as log:
                        xvfb = subprocess.Popen(["Xvfb", "-displayfd", str(write_fd), "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
                                                pass_fds=(write_fd,), stdout=log, stderr=subprocess.STDOUT)
                    os.close(write_fd)
                    write_fd = None
                    if not select.select([read_fd], [], [], 15)[0]:
                        raise RuntimeError("Xvfb did not report a display")
                    display = os.read(read_fd, 64).decode().strip()
                    if not display.isdigit() or xvfb.poll() is not None:
                        raise RuntimeError("Xvfb failed; inspect xvfb.log")
                    env["DISPLAY"] = ":" + display
                finally:
                    os.close(read_fd)
                    if write_fd is not None:
                        os.close(write_fd)
            else:
                env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
                for name in ("LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER"):
                    env.pop(name, None)
            command = [sys.executable, "scripts/benchmarks/online_stepping/dmc_state_export.py",
                       "--domain", args.domain, "--reference-adapters", str(args.reference_adapters),
                       "--devices", device, "--verify-gl", "--output", str(args.output / f"{device}.json")]
            with (args.output / f"{device}.log").open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{device} failed; inspect {args.output / (device + '.log')}")
            report = json.loads((args.output / f"{device}.json").read_text())
            assert report["status"] == "PASS"
            print(args.domain, device, "PASS", flush=True)
        finally:
            if xvfb is not None:
                xvfb.terminate()
                try:
                    xvfb.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    xvfb.kill()
                    xvfb.wait()


if __name__ == "__main__":
    main()
