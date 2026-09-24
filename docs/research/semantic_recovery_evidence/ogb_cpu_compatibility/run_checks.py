"""Run this checkpoint's CPU-only checks with immutable authority dependencies first."""

import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = "/private/tmp/ogb-manip-authority-swm-lock/bin/python"
TEST_SITE = str(ROOT / ".venv/lib/python3.12/site-packages")
LOCAL = str(ROOT / ".venv/bin/python")
RUFF = str(ROOT / ".venv/bin/ruff")
CODE = [
    "stable_envs/_adapters/ogbench_manipulation.py",
    "stable_envs/_adapters/ogbench_manipulation_compat.py",
    "stable_envs/envs/ogbench/manipulation_semantics.py",
    "stable_envs/envs/ogbench/__init__.py",
    "tests/unit/test_ogbench_manipulation_cpu.py",
    "tests/unit/test_envs_ogbench.py",
    str(HERE.relative_to(ROOT) / "audit.py"),
    str(HERE.relative_to(ROOT) / "run_checks.py"),
]


def main():
    env = dict(os.environ, MUJOCO_GL="disable", SDL_VIDEODRIVER="dummy", PYTHONPATH="/Users/samib/stable-worldmodel:.")
    bootstrap = (
        f"import sys; sys.path.append({TEST_SITE!r}); import pytest; "
        "sys.exit(pytest.main(['-q', '-rA', 'tests/unit/test_ogbench_manipulation_cpu.py']))"
    )
    checks = [
        ("cpu-tests.log", [AUTHORITY, "-c", bootstrap]),
        (
            "shared-tests.log",
            [
                LOCAL,
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_full_state_contract.py",
                "tests/unit/test_semantic_harness.py",
                "tests/unit/test_replacement_readiness.py",
                "tests/unit/test_swm_matrix.py",
                "tests/unit/test_repo_contracts.py",
            ],
        ),
        (
            "family-structure-tests.log",
            [
                LOCAL,
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_envs_ogbench.py",
                "-k",
                "not lifecycle and not dataset and not private_projection and not renderer_only and not private_render",
            ],
        ),
        ("ruff.log", [RUFF, "check", *CODE]),
        ("format.log", [RUFF, "format", "--check", *CODE]),
        (
            "docs.log",
            [LOCAL, "-m", "mkdocs", "build", "--strict", "--site-dir", "/private/tmp/stable-envs-ogb-cpu-site"],
        ),
        ("whitespace.log", ["git", "diff", "--check"]),
    ]
    results = []
    for name, command in checks:
        start = time.monotonic()
        with (HERE / name).open("w") as output:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        results.append(
            {
                "log": name,
                "argv": command,
                "cwd": str(ROOT),
                "environment": {k: env[k] for k in ["MUJOCO_GL", "SDL_VIDEODRIVER", "PYTHONPATH"]},
                "returncode": completed.returncode,
                "seconds": round(time.monotonic() - start, 3),
            }
        )
        (HERE / "commands.json").write_text(json.dumps(results, indent=2) + "\n")
        print(name, completed.returncode, flush=True)
    if any(r["returncode"] for r in results):
        raise SystemExit(1)
    print("CPU_CHECKPOINT_CHECKS_PASS", flush=True)


if __name__ == "__main__":
    main()
