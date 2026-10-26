"""Plan, import, freeze or validate explicitly audited source members.

    python -m scripts.export.batch_object_permanence --source-root <checkout> \
        --output <dataset> --report <report.json> --tasks all --plan-only

An unregistered member is reported as unsupported, never inferred to be rigid.
Execution exits unsuccessfully if any requested job is unsupported or fails.
Existing clips are not overwritten. Reports are not renderer asset indexes.
The default export stage retains the historical serialization-only behavior.
Oracle/all stages also refuse existing reference directories. Validation reuses
existing inputs and checks complete source fidelity and exact cache parity.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

from scripts.object_permanence_member import Member
from scripts.testing.freeze_object_permanence_member import supports

from .object_permanence import (
    ALPHA_CONTRACT,
    ALPHA_TASKS,
    AUDITED_MEMBERS,
    AUDITED_TASKS,
    CONTRACT_ONLY_TASKS,
    DEFERRED_TASKS,
    ORTHOGRAPHIC_MEMBERS,
    ORTHOGRAPHIC_TASKS,
    RAY_CONTRACT_TASKS,
    SOURCE_COMMIT,
    capability_group,
    check_source,
    task_manifests,
)
from .object_permanence_optical import MEMBERS as OPTICAL_MEMBERS


def run(
    root,
    output,
    report,
    *,
    tasks="all",
    members="all",
    profile="nominal",
    stage="export",
    plan_only=False,
    blender_python=sys.executable,
    render_python=sys.executable,
    oracle_root=None,
    visibility_contract=None,
):
    Member("G01", 0, profile)  # Reject unaudited diversity before any source execution.
    if stage not in ("export", "oracle", "validate", "all"):
        raise ValueError("unknown member batch stage")
    chosen = None if members == "all" else {int(m) for m in members.split(",")}
    if chosen is not None and (not chosen or min(chosen) < 0):
        raise ValueError("members must be nonnegative script indices")
    if stage != "export" and (oracle_root is None or visibility_contract is None):
        raise ValueError("oracle/validate/all stages require oracle_root and an explicit visibility contract")
    if visibility_contract not in (
        None,
        "stable-envs/opaque-pixel-rays-v1",
        "stable-envs/opaque-pixel-rays-v2",
        ALPHA_CONTRACT,
        "stable-envs/sampled-optical-rays-v1",
    ):
        raise ValueError("unknown visibility contract")
    check_source(root)
    selected = None if tasks == "all" else set(tasks.split(","))
    paths = task_manifests(root)
    known = {json.loads(path.read_text())["id"] for path in paths}
    if selected is not None and selected - known:
        raise ValueError(f"unknown source tasks: {sorted(selected - known)}")
    if chosen is not None:
        for path in paths:
            metadata = json.loads(path.read_text())
            if selected is None or metadata["id"] in selected:
                missing = chosen - set(range(len(metadata["member_scripts"])))
                if missing:
                    raise ValueError(f"unknown source members for {metadata['id']}: {sorted(missing)}")
    jobs = []
    for path in paths:
        task = json.loads(path.read_text())
        known.add(task["id"])
        if selected is not None and task["id"] not in selected:
            continue
        for member, name in enumerate(task["member_scripts"]):
            if chosen is not None and member not in chosen:
                continue
            entry = Member(task["id"], member, profile)
            extra = (task["id"], member) in AUDITED_MEMBERS
            extra_version = "v2" if (task["id"], member) in ORTHOGRAPHIC_MEMBERS else "v1"
            optical = (task["id"], member) in OPTICAL_MEMBERS
            audited = (member == 0 and task["id"] in AUDITED_TASKS) or extra or optical
            job = {"task": task["id"], "member": member, "script": str((path.parent / name).relative_to(root))}
            job.update(profile=profile, clip=entry.clip, stage=stage)
            job["capability_group"] = capability_group(ast.parse((path.parent / name).read_text()), audited=audited)
            versioned = (
                (visibility_contract == "stable-envs/opaque-pixel-rays-v1" and task["id"] in RAY_CONTRACT_TASKS)
                or (visibility_contract == "stable-envs/opaque-pixel-rays-v2" and task["id"] in ORTHOGRAPHIC_TASKS)
                or (visibility_contract == ALPHA_CONTRACT and task["id"] in ALPHA_TASKS)
                or (visibility_contract == "stable-envs/sampled-optical-rays-v1" and optical)
            )
            if optical and visibility_contract != "stable-envs/sampled-optical-rays-v1":
                job.update(status="unsupported", reason="optical member requires explicit sampled optical contract")
            elif extra and visibility_contract != f"stable-envs/opaque-pixel-rays-{extra_version}":
                job.update(
                    status="unsupported",
                    reason=f"additional opaque member requires explicit {extra_version} ray contract",
                )
            elif not optical and visibility_contract == "stable-envs/sampled-optical-rays-v1":
                job.update(status="unsupported", reason="member is not audited for the optical contract")
            elif member == 0 and visibility_contract == ALPHA_CONTRACT and task["id"] not in ALPHA_TASKS:
                job.update(status="unsupported", reason="entry is not audited for the alpha material contract")
            elif member == 0 and task["id"] in CONTRACT_ONLY_TASKS and not versioned:
                job.update(status="unsupported", reason="requires explicit versioned ray contract")
            elif member == 0 and task["id"] in DEFERRED_TASKS and not versioned:
                job.update(
                    status="unsupported", reason=DEFERRED_TASKS[task["id"]], capability_group="source-fidelity-review"
                )
            elif not audited:
                job.update(status="unsupported", reason="preparation entry point not yet audited")
            elif stage != "export" and not supports(entry, visibility_contract):
                job.update(status="unsupported", reason="no independently audited versioned oracle producer")
            elif plan_only:
                job["status"] = "ready-for-runtime-check"
            else:
                common = ["--task", task["id"], "--member", str(member), "--profile", profile]
                source = ["--source-root", str(root)]
                commands = []
                if stage in ("export", "all"):
                    commands.extend(
                        [
                            (
                                "export",
                                [
                                    str(blender_python),
                                    "-m",
                                    "scripts.export.object_permanence",
                                    "export",
                                    *source,
                                    "--output",
                                    str(output),
                                    *common,
                                ],
                            ),
                            (
                                "geometry_check",
                                [
                                    str(blender_python),
                                    "-m",
                                    "scripts.testing.check_object_permanence_export",
                                    *source,
                                    "--dataset",
                                    str(output),
                                    *common,
                                ],
                            ),
                        ]
                    )
                if stage in ("oracle", "all"):
                    commands.append(
                        (
                            "oracle",
                            [
                                str(blender_python),
                                "-m",
                                "scripts.testing.freeze_object_permanence_member",
                                *source,
                                "--output",
                                str(oracle_root),
                                *common,
                                "--visibility-contract",
                                visibility_contract,
                            ],
                        )
                    )
                if stage in ("validate", "all"):
                    member_report = report.parent / (report.stem + "-" + entry.clip + "-validation.json")
                    job["validation_report"] = str(member_report)
                    commands.append(
                        (
                            "validation",
                            [
                                str(render_python),
                                "-m",
                                "scripts.testing.check_object_permanence_member",
                                "--recordings",
                                str(output),
                                "--oracles",
                                str(oracle_root),
                                *common,
                                "--visibility-contract",
                                visibility_contract,
                                "--report",
                                str(member_report),
                            ],
                        )
                    )
                try:
                    for label, command in commands:
                        completed = subprocess.run(command, capture_output=True, text=True, timeout=3600)
                        job[label] = "passed" if completed.returncode == 0 else "failed"
                        job["detail"] = (completed.stdout if completed.returncode == 0 else completed.stderr)[-4000:]
                        if completed.returncode:
                            job["status"] = "failed"
                            break
                    else:
                        # Export alone certifies serialization, never pixel fidelity.
                        job["status"] = {
                            "export": "exported",
                            "oracle": "frozen",
                            "validate": "verified",
                            "all": "verified",
                        }[stage]
                except (OSError, subprocess.TimeoutExpired) as error:
                    job.update(status="failed", detail=str(error))
            jobs.append(job)
    if not jobs:
        raise ValueError("selection contains no source members")
    result = {
        "stage": stage,
        "profile": profile,
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
    parser.add_argument("--members", default="all", help="all or comma-separated zero-based script indices")
    parser.add_argument("--profile", default="nominal")
    parser.add_argument("--stage", choices=("export", "oracle", "validate", "all"), default="export")
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--render-python", default=sys.executable)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--blender-python", default=sys.executable)
    parser.add_argument("--visibility-contract")
    args = parser.parse_args()
    result = run(
        args.source_root.resolve(),
        args.output.resolve(),
        args.report.resolve(),
        tasks=args.tasks,
        members=args.members,
        profile=args.profile,
        stage=args.stage,
        oracle_root=args.oracle_root,
        render_python=args.render_python,
        plan_only=args.plan_only,
        blender_python=args.blender_python,
        visibility_contract=args.visibility_contract,
    )
    print(json.dumps(result["counts"], sort_keys=True))
    if not args.plan_only and any(job["status"] not in ("exported", "frozen", "verified") for job in result["jobs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
