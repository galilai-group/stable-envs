"""Run one planned registration/device in an existing Oscar allocation.

Stdlib-only driver. Backend variants share a node/device but use fresh Python
processes. It does not submit jobs or modify the accepted production checkout.
Upload this file, profile_all.py and plan.json together into a new directory.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def child(harness):
    # Resolve helper imports from the immutable accepted production checkout;
    # load only the new profiler from the separate campaign directory.
    spec = importlib.util.spec_from_file_location("scripts.benchmarks.renderer_table.profile_all", harness)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.argv = [str(harness), *sys.argv[3:]]
    module.main()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("job", type=int)
    a = p.parse_args()
    assert os.environ.get("SLURM_JOB_ID"), "Run inside an allocated Oscar job."
    plan = json.loads((a.root / "plan.json").read_text())
    job = plan["jobs"][a.job]
    source = Path(plan["production_source"])
    for path, expected in plan["production_sha256"].items():
        assert hashlib.sha256((source / path).read_bytes()).hexdigest() == expected, path
    oracle = Path(
        "/oscar/scratch/sboughan/se-semantic-recovery/render-rerun-df80c9c5-20260928T021357Z/run-6750648_2/oracle"
    )
    venvs = Path("/oscar/scratch/sboughan/venvs")
    py = venvs / "se6-verify-vis311/bin/python"
    paths = [source, oracle] + [
        venvs / f"{v}/lib/python3.12/site-packages" for v in ("se-vis311", "se-step6", "se-step4-swm")
    ]
    registration, device = job["registration"], job["device"]
    env = os.environ.copy()
    if "Fetch" in registration:
        py = venvs / "se6-verify-fetch311/bin/python"
        paths = [source, oracle] + [
            venvs / f"{v}/lib/python3.12/site-packages" for v in ("se-s3b-fetch311", "se-step6", "se-step4-swm")
        ]
    elif "OGBCube" in registration or "OGBScene" in registration:
        py = Path(
            "/oscar/scratch/sboughan/se-semantic-recovery/ogb-renderer-294675f0-complete/run-6669909/profile/bin/python"
        )
        paths = [source, oracle]
    elif "Craftax" in registration:
        py = Path(
            "/oscar/scratch/sboughan/se-semantic-recovery/stable-envs-craftax-b6a8242f-gpu/run-6679082/venv/bin/python"
        )
        paths = [source, oracle]
        env.update(
            JAX_PLATFORMS="cuda" if device == "cuda" else "cpu",
            CRAFTAX_JAX_BACKEND="gpu" if device == "cuda" else "cpu",
            SWM_ROOT=str(oracle),
        )
    elif "Rocket" in registration:
        paths.append(venvs / "se7-rocket/lib/python3.12/site-packages")
    env.update(
        PYTHONPATH=":".join(map(str, paths)),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        LP_NUM_THREADS="1",
    )
    run = a.root / f"job-{a.job}-{env['SLURM_JOB_ID']}"
    run.mkdir(exist_ok=False)
    (run / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    subprocess.run(["lscpu"], stdout=(run / "lscpu.txt").open("w"), check=True)
    if device == "cuda":
        subprocess.run(["nvidia-smi", "-q"], stdout=(run / "nvidia-smi.txt").open("w"), check=True)
    xvfb = None
    if device == "cpu":
        env.update(
            MUJOCO_GL="glfw",
            PYOPENGL_PLATFORM="glx",
            LIBGL_ALWAYS_SOFTWARE="1",
            GALLIUM_DRIVER="llvmpipe",
            CUDA_VISIBLE_DEVICES="",
        )
        env["DISPLAY"] = ":" + str(100 + int(env["SLURM_JOB_ID"]) % 5000)
        xvfb = subprocess.Popen(
            ["Xvfb", env["DISPLAY"], "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
            stdout=(run / "xvfb.log").open("w"),
            stderr=subprocess.STDOUT,
        )
        time.sleep(2)
        assert xvfb.poll() is None, "Xvfb failed; inspect log"
    else:
        env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
        env.pop("LIBGL_ALWAYS_SOFTWARE", None)
        env.pop("GALLIUM_DRIVER", None)
    codes = {}
    try:
        for index in job["tasks"]:
            task = plan["tasks"][index]
            output = a.root / "results" / str(index)
            assert not output.exists(), "Never overwrite evidence; retry into a new campaign directory."
            cache = run / f"cache-{index}"
            env.update(
                WARP_CACHE_PATH=str(cache / "warp"),
                TORCH_EXTENSIONS_DIR=str(cache / "torch"),
                XDG_CACHE_HOME=str(cache / "xdg"),
            )
            command = [
                str(py),
                str(Path(__file__).resolve()),
                "--child",
                str(a.root / "profile_all.py"),
                registration,
                device,
                str(output),
                "--backend",
                task["backend"],
            ]
            if task["corpus"]:
                command += ["--corpus", task["corpus"]]
            if False:
                command += ["--selected-result", "/oscar/scratch/sboughan/se-semantic-recovery/renderer-saturation-df80c9c5-20260929/results/" + str(index) + "/result.json"]
            if os.environ.get("PAIRED_PILOT"):
                command += ["--batches", "1", "64", "--poses", "4"]
            command += ["--batches", "256", "512", "1024", "2048", "4096", "8192", "16384"]
            with (run / f"task-{index}.log").open("w") as log:
                completed = subprocess.run(command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT)
            codes[index] = completed.returncode
            (run / "exit-codes.json").write_text(json.dumps(codes, indent=2) + "\n")
    finally:
        if xvfb:
            xvfb.terminate()
            xvfb.wait(timeout=10)
    if any(codes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(Path(sys.argv[2]))
    else:
        main()
