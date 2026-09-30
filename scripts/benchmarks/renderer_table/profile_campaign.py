"""Create an explicit 42-registration profile plan or export completed profiles.

Planning never submits jobs. Export retains missing/error rows rather than
misrepresenting a partial run as complete.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .common import ROOT, catalog, write


def plan(parity_manifest):
    manifest = json.loads(parity_manifest.read_text())
    tasks = []
    for registration, row in sorted(catalog().items()):
        saved = manifest["registrations"].get(registration, {})
        for device in ("cpu", "cuda"):
            role = "source_gpu" if device == "cuda" and "Craftax" in registration else "source_cpu"
            corpus = saved.get(role)
            if "Symbolic" in registration or row["name"] == "PushT":
                corpus = None
            else:
                assert corpus, (registration, device)
            if device == "cpu":
                backends = (
                    ["opengl"]
                    if row["name"].startswith("Fetch") or row["name"] in {"OGBCube", "OGBScene"}
                    else ["auto"]
                )
                if "WalkerDMControl" in registration:
                    backends += ["opengl"]
            else:
                backends = ["auto"]
                if row["name"] == "OGBCube":
                    backends += ["mjwarp_fidelity", "opengl"]
                elif row["name"] == "OGBScene" or "WalkerDMControl" in registration or row["name"].startswith("Fetch"):
                    backends += ["opengl"]
            for backend in backends:
                tasks.append(
                    dict(
                        index=len(tasks),
                        registration=registration,
                        name=row["name"],
                        device=device,
                        backend=backend,
                        corpus=corpus,
                        scope=saved.get("scope", "Saved parity corpus"),
                        expected_status="NO_RGB" if "Symbolic" in registration else "COMPLETE",
                    )
                )
    jobs = []
    for task in tasks:
        key = (task["registration"], task["device"])
        if jobs and (jobs[-1]["registration"], jobs[-1]["device"]) == key:
            jobs[-1]["tasks"].append(task["index"])
        else:
            jobs.append(dict(index=len(jobs), registration=key[0], device=key[1], tasks=[task["index"]]))
    return dict(
        base_commit=manifest["base_commit"],
        production_source=manifest["campaign"] + "/source-v12",
        parity_manifest_sha256=hashlib.sha256(parity_manifest.read_bytes()).hexdigest(),
        registration_count=len(catalog()),
        tasks=tasks,
        jobs=jobs,
        production_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "stable_envs").rglob("*.py"))
        },
        limitations=[
            "Higher batches measure prepared-input rendering only, not full State projection.",
            "PushT registrations share one diagnostic renderer; no new acceptance claims.",
            "No new source renderer timing except Craftax actual JAX get_obs; cached render is not recomputation.",
            "CPU/GPU jobs can land on different hosts; hardware is recorded per result.",
            "Torch traces may not expose external OpenGL, Warp or JAX GPU kernels.",
        ],
    )


def export(plan_file, results, output):
    campaign = json.loads(plan_file.read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows, functions, operators, statuses, records = [], [], [], [], {}
    for task in campaign["tasks"]:
        directory = results / str(task["index"])
        path = directory / "result.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        if data:
            assert all(data[k] == task[k] for k in ("registration", "device", "backend")), path
            records[(task["registration"], task["device"], task["backend"])] = data
        status = data.get("status", "MISSING")
        common = dict(
            registration=task["registration"],
            device=task["device"],
            backend=task["backend"],
            status=status,
            host=data.get("profile", {}).get("host"),
            cpu=data.get("profile", {}).get("cpu"),
            gpu=data.get("profile", {}).get("gpu"),
            result=str(path),
        )
        statuses.append(common)
        for value in data.get("measurements", []):
            timing = value.get("timing", {})
            rows.append(
                dict(
                    **common,
                    stage=value["stage"],
                    batch=value["batch"],
                    count=timing.get("count"),
                    mean_ms=timing.get("mean_ms"),
                    sd_ms=timing.get("sd_ms"),
                    median_ms=timing.get("median_ms"),
                    p95_ms=timing.get("p95_ms"),
                    first_call_ms=timing.get("first_call_ms"),
                    ms_per_frame=timing["mean_ms"] / value["batch"] if timing else None,
                    scope=value.get("scope", data.get("scope")),
                )
            )
        for path in directory.glob("*.profile.json"):
            prof = json.loads(path.read_text())
            for value in prof.get("python_functions", []):
                functions.append(dict(**common, stage=path.name.removesuffix(".profile.json"), **value))
            for value in prof.get("torch_operators", []):
                operators.append(dict(**common, stage=path.name.removesuffix(".profile.json"), **value))
    for name, values in (
        ("timings.csv", rows),
        ("functions.csv", functions),
        ("torch_operators.csv", operators),
        ("coverage.csv", statuses),
    ):
        with (output / name).open("w") as stream:
            if values:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    complete = all(s["status"] == t["expected_status"] for s, t in zip(statuses, campaign["tasks"]))
    per_environment = []
    for registration in sorted({t["registration"] for t in campaign["tasks"]}):
        entry = dict(env=registration)
        for device, label in (("cpu", "cpu"), ("cuda", "gpu")):
            backend = "auto"
            if "OGBScene" in registration or "WalkerDMControl" in registration:
                backend = "opengl"
            elif "OGBCube" in registration:
                backend = "opengl" if device == "cpu" else "mjwarp_fidelity"
            elif device == "cpu" and "Fetch" in registration:
                backend = "opengl"
            data = records.get((registration, device, backend), {})
            entry[label + "_backend"] = backend
            entry[label + "_status"] = data.get("status", "MISSING")
            entry[label + "_host"] = data.get("profile", {}).get("host")
            entry[label + "_hardware"] = data.get("profile", {}).get(label)
            metrics = {(m["stage"], m["batch"]): m.get("timing", {}) for m in data.get("measurements", [])}
            for stage, batch in (
                ("public_full_state", 1),
                ("prepare_validate", 1),
                ("prepared_render", 1),
                ("prepared_batch_render", 256),
            ):
                timing = metrics.get((stage, batch), {})
                for statistic in ("mean_ms", "sd_ms", "median_ms"):
                    value = timing.get(statistic)
                    entry[f"{label}_{stage}_b{batch}_{statistic}_per_frame"] = (
                        value / batch if value is not None else None
                    )
                entry[f"{label}_{stage}_b{batch}_samples"] = timing.get("count")
        entry["scope"] = (
            "NO_RGB"
            if "Symbolic" in registration
            else "DIAGNOSTIC_ONLY shared candidate renderer"
            if "PushT" in registration
            else "Source corpus; reset-only"
            if "FetchPickAndPlace" in registration
            else "Source corpus"
        )
        per_environment.append(entry)
    with (output / "per_environment_42.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_environment[0]))
        writer.writeheader()
        writer.writerows(per_environment)
    write(
        output / "summary.json",
        dict(
            status="COMPLETE" if complete else "INCOMPLETE",
            expected_tasks=len(statuses),
            expected_registrations=42,
            statuses={s: sum(r["status"] == s for r in statuses) for s in sorted({r["status"] for r in statuses})},
            limitations=campaign["limitations"],
        ),
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    create = sub.add_parser("plan")
    create.add_argument("output", type=Path)
    create.add_argument(
        "--parity-manifest",
        type=Path,
        default=ROOT / "docs/research/semantic_recovery_evidence/renderer_parity_1000_20260928/manifest-v6.json",
    )
    collect = sub.add_parser("export")
    collect.add_argument("plan", type=Path)
    collect.add_argument("results", type=Path)
    collect.add_argument("output", type=Path)
    a = p.parse_args()
    if a.command == "plan":
        assert not a.output.exists(), a.output
        write(a.output, plan(a.parity_manifest))
    else:
        export(a.plan, a.results, a.output)


if __name__ == "__main__":
    main()
