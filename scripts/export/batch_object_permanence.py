"""Plan or export audited source members with one isolated Blender process/job.

    python -m scripts.export.batch_object_permanence --source-root <checkout> \
        --output <dataset> --report <report.json> --tasks all --plan-only

An unregistered member is reported as unsupported, never inferred to be rigid.
Execution exits unsuccessfully if any requested job is unsupported or fails.
Existing clips are not overwritten. Reports are not renderer asset indexes.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

from .object_permanence import (
    ALPHA_CONTRACT,
    ALPHA_TASKS,
    AUDITED_TASKS,
    CONTRACT_ONLY_TASKS,
    DEFERRED_TASKS,
    ORTHOGRAPHIC_TASKS,
    RAY_CONTRACT_TASKS,
    SOURCE_COMMIT,
    capability_group,
    check_source,
    task_manifests,
)


def run(
    root, output, report, *, tasks="all", plan_only=False, blender_python=sys.executable, visibility_contract=None
):
    if visibility_contract not in (
        None,
        "stable-envs/opaque-pixel-rays-v1",
        "stable-envs/opaque-pixel-rays-v2",
        ALPHA_CONTRACT,
    ):
        raise ValueError("unknown visibility contract")
    check_source(root)
    selected = None if tasks == "all" else set(tasks.split(","))
    paths = task_manifests(root)
    known = {json.loads(path.read_text())["id"] for path in paths}
    if selected is not None and selected - known:
        raise ValueError(f"unknown source tasks: {sorted(selected - known)}")
    jobs = []
    for path in paths:
        task = json.loads(path.read_text())
        known.add(task["id"])
        if selected is not None and task["id"] not in selected:
            continue
        for member, name in enumerate(task["member_scripts"]):
            job = {"task": task["id"], "member": member, "script": str((path.parent / name).relative_to(root))}
            job["capability_group"] = capability_group(
                ast.parse((path.parent / name).read_text()), audited=member == 0 and task["id"] in AUDITED_TASKS
            )
            versioned = (
                (visibility_contract == "stable-envs/opaque-pixel-rays-v1" and task["id"] in RAY_CONTRACT_TASKS)
                or (visibility_contract == "stable-envs/opaque-pixel-rays-v2" and task["id"] in ORTHOGRAPHIC_TASKS)
                or (visibility_contract == ALPHA_CONTRACT and task["id"] in ALPHA_TASKS)
            )
            if member == 0 and visibility_contract == ALPHA_CONTRACT and task["id"] not in ALPHA_TASKS:
                job.update(status="unsupported", reason="entry is not audited for the alpha material contract")
            elif member == 0 and task["id"] in CONTRACT_ONLY_TASKS and not versioned:
                job.update(status="unsupported", reason="requires explicit versioned ray contract")
            elif member == 0 and task["id"] in DEFERRED_TASKS and not versioned:
                job.update(
                    status="unsupported", reason=DEFERRED_TASKS[task["id"]], capability_group="source-fidelity-review"
                )
            elif member != 0 or task["id"] not in AUDITED_TASKS:
                job.update(status="unsupported", reason="preparation entry point not yet audited")
            elif plan_only:
                job["status"] = "ready-for-runtime-check"
            else:
                command = [
                    str(blender_python),
                    "-m",
                    "scripts.export.object_permanence",
                    "export",
                    "--source-root",
                    str(root),
                    "--output",
                    str(output),
                    "--task",
                    task["id"],
                ]
                try:
                    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
                    if completed.returncode == 0:
                        # Validate independently reconstructed source triangles,
                        # not merely successful file writing. Pixel certification
                        # remains the per-task frozen-oracle test suite.
                        completed = subprocess.run(
                            [
                                str(blender_python),
                                "-m",
                                "scripts.testing.check_object_permanence_export",
                                "--source-root",
                                str(root),
                                "--dataset",
                                str(output),
                                "--task",
                                task["id"],
                            ],
                            capture_output=True,
                            text=True,
                            timeout=600,
                        )
                        job["geometry_check"] = "passed" if completed.returncode == 0 else "failed"
                    job["status"] = "exported" if completed.returncode == 0 else "failed"
                    job["detail"] = (completed.stdout if completed.returncode == 0 else completed.stderr)[-4000:]
                except (OSError, subprocess.TimeoutExpired) as error:
                    job.update(status="failed", detail=str(error))
            jobs.append(job)
    result = {
        "schema": "stable-envs/object-permanence-batch-v1",
        "source_commit": SOURCE_COMMIT,
        "visibility_contract": visibility_contract,
        "plan_only": plan_only,
        "jobs": jobs,
        "counts": {
            status: sum(job["status"] == status for job in jobs) for status in sorted({job["status"] for job in jobs})
        },
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--blender-python", default=sys.executable)
    parser.add_argument("--visibility-contract")
    args = parser.parse_args()
    result = run(
        args.source_root.resolve(),
        args.output.resolve(),
        args.report.resolve(),
        tasks=args.tasks,
        plan_only=args.plan_only,
        blender_python=args.blender_python,
        visibility_contract=args.visibility_contract,
    )
    print(json.dumps(result["counts"], sort_keys=True))
    if not args.plan_only and any(job["status"] != "exported" for job in result["jobs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
