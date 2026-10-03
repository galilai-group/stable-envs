"""Run paired export/process checks with isolated CPU/software-GL and GPU/EGL processes."""

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
    p.add_argument("--reference-adapters", type=Path)
    p.add_argument("--benchmark", choices=["dmc_state_export", "dmc_process_probe", "dmc_pipeline", "dmc_source_pipeline", "ogbench_state_export", "ogbench_maze_process", "ogbench_manipulation_process", "ogbench_render_defaults", "ogbench_source_frame_gate", "fetch_source_frame_gate", "fetch_source_pipeline"], default="dmc_state_export")
    p.add_argument("--source-root", type=Path)
    p.add_argument("--source-manifest", type=Path)
    p.add_argument("--image-gates", type=Path, help="Completed per-device Fetch image gate directory")
    p.add_argument("--source-mode", choices=["source_serial", "source_parallel", "ours"])
    p.add_argument("--batches", type=int, nargs="+", help="Override the selected benchmark's batch sweep")
    p.add_argument("--gates-only", action="store_true")
    p.add_argument("--production", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.benchmark == "dmc_state_export" and (args.reference_adapters is None or args.gates_only):
        p.error("Exporter comparison requires --reference-adapters and does not support --gates-only")
    if args.production and (args.benchmark not in ("dmc_process_probe", "ogbench_maze_process", "ogbench_manipulation_process") or not args.gates_only):
        p.error("Production route checks require a supported process probe with --gates-only")
    source_args = (args.source_root, args.source_manifest, args.source_mode)
    if args.benchmark in ("dmc_source_pipeline", "fetch_source_pipeline"):
        if any(value is None for value in source_args):
            p.error("Source comparison requires --source-root, --source-manifest and --source-mode")
    elif args.benchmark in ("ogbench_source_frame_gate", "fetch_source_frame_gate"):
        if args.source_root is None or args.source_manifest is None or args.source_mode is not None:
            p.error("Source frame gate requires source root/manifest without a strategy mode")
        if args.batches is not None or args.gates_only or args.reference_adapters is not None:
            p.error("Source frame gate uses fixed batches/workers and does not accept benchmark overrides")
    elif any(value is not None for value in source_args):
        p.error("Source options require a source benchmark or frame gate")
    if args.benchmark == "fetch_source_pipeline":
        needs_images = "PickAndPlace" not in args.domain
        if needs_images != (args.image_gates is not None):
            p.error("Fetch rendered comparisons require --image-gates; PickAndPlace remains no-frame only")
    elif args.image_gates is not None:
        p.error("--image-gates applies only to fetch_source_pipeline")
    args.output.mkdir(parents=True, exist_ok=False)
    for device in ("cpu", "cuda"):
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", LP_NUM_THREADS="1")
        xvfb = None
        read_fd = None
        try:
            if device == "cpu":
                env.update(MUJOCO_GL="glfw", PYOPENGL_PLATFORM="glx", LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe", CUDA_VISIBLE_DEVICES="")
                read_fd, write_fd = os.pipe()
                try:
                    with (args.output / "xvfb.log").open("w") as log:
                        # The GL proof closes its last client before public
                        # construction. A reset can invalidate the display and
                        # re-emit its number after the readiness pipe is closed.
                        xvfb = subprocess.Popen(["Xvfb", "-displayfd", str(write_fd), "-noreset", "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
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
                    if write_fd is not None:
                        os.close(write_fd)
            else:
                env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
                for name in ("LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER"):
                    env.pop(name, None)
            command = [sys.executable, f"scripts/benchmarks/online_stepping/{args.benchmark}.py",
                       "--domain", args.domain,
                       "--devices", device, "--verify-gl", "--output", str(args.output / f"{device}.json")]
            if args.benchmark in ("dmc_source_pipeline", "fetch_source_pipeline"):
                command += ["--source-root", str(args.source_root), "--source-manifest", str(args.source_manifest),
                            "--mode", args.source_mode]
            if args.benchmark in ("ogbench_source_frame_gate", "fetch_source_frame_gate"):
                command += ["--source-root", str(args.source_root), "--source-manifest", str(args.source_manifest)]
            if args.image_gates is not None:
                command += ["--image-gate", str(args.image_gates / f"{device}.json")]
            if args.reference_adapters is not None:
                command += ["--reference-adapters", str(args.reference_adapters)]
            if args.batches is not None:
                command += ["--batches", *map(str, args.batches)]
            if args.gates_only:
                command.append("--gates-only")
            if args.production:
                command.append("--production")
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
            # Some X servers announce again during driver initialization even
            # with -noreset. Keep the pipe alive until the server has stopped.
            if read_fd is not None:
                os.close(read_fd)


if __name__ == "__main__":
    main()
