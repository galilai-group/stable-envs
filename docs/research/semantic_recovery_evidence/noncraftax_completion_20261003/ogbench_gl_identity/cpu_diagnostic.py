"""Run the frozen OGBench CPU gate without waiting for a GPU allocation."""

import os
import select
import subprocess
import sys
from pathlib import Path

base, family, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
output.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, MUJOCO_GL="glfw", PYOPENGL_PLATFORM="glx", LIBGL_ALWAYS_SOFTWARE="1",
           GALLIUM_DRIVER="llvmpipe", CUDA_VISIBLE_DEVICES="", LP_NUM_THREADS="1")
read_fd, write_fd = os.pipe()
server = None
try:
    with (output / "xvfb.log").open("w") as log:
        server = subprocess.Popen(
            ["Xvfb", "-displayfd", str(write_fd), "-noreset", "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
            pass_fds=(write_fd,), stdout=log, stderr=subprocess.STDOUT,
        )
    os.close(write_fd)
    write_fd = None
    if not select.select([read_fd], [], [], 15)[0]:
        raise RuntimeError("Xvfb did not report a display")
    display = os.read(read_fd, 64).decode().strip()
    if not display.isdigit() or server.poll() is not None:
        raise RuntimeError("Invalid Xvfb display")
    env["DISPLAY"] = ":" + display
    command = [sys.executable, "scripts/benchmarks/online_stepping/ogbench_source_frame_gate.py",
               "--domain", family, "--devices", "cpu", "--verify-gl",
               "--source-root", str(base / "source"), "--source-manifest", str(base / "source_manifest.json"),
               "--output", str(output / "cpu.json")]
    with (output / "cpu.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
finally:
    if server is not None:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    os.close(read_fd)
    if write_fd is not None:
        os.close(write_fd)
