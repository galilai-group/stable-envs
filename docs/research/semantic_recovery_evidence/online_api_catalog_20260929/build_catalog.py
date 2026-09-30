"""Rebuild the source-audit catalogue; no environment construction or benchmarks."""

import ast
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MAP = ROOT / "docs/research/semantic_recovery_implementation_map.json"
MATRIX = ROOT / "docs/research/semantic_recovery_matrix.json"


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(row):
    family, name = row["family"], row["historical_id"]
    if family == "classic_control":
        native = (
            "CartPoleVectorEnv exists; autoreset/RNG/Variation not equivalent"
            if "CartPole" in name
            else "No custom vector entry point found for this scalar registration in inspected Gymnasium"
        )
        return dict(
            sources=["stable_envs/envs/_classic_semantics.py"],
            step_symbol="ClassicControlAdapter.step",
            current_execution="SCALAR_CPU_LOOP",
            native_batch=native,
            upstream_api="Gymnasium scalar reset/step/render; SyncVectorEnv serial; AsyncVectorEnv processes",
            parallel_route="Existing vector executor if State/lifecycle facade fits; otherwise persistent worker shards",
            thread_constraint="Python task work; no thread throughput/safety proof; measure IPC overhead for cheap steps",
            state_requirement="Per-lane PCG state, TimeLimit/post-terminal fields, action-dependent dtype and physics Variation",
            lifecycle="Preserve scalar reset/options and post-terminal behavior; no new vector autoreset",
            renderer_boundary="Native renderer separately batches snapshots",
        )
    if family == "dmcontrol":
        source = row["state_schema"]["candidate_source_catalog"][0].removeprefix("candidate:")
        cls = {
            "dmcontrol_aligned.py": "DMCAlignedAdapter",
            "dmcontrol_continuation.py": "DMCContinuationAdapter",
            "dmcontrol_humanoid.py": "DMCHumanoidAdapter",
        }.get(Path(source).name, "ReacherAdapter" if "Reacher" in name else "FirstFamilyAdapter")
        return dict(
            sources=[source],
            step_symbol=cls + ".step",
            current_execution="SCALAR_CPU_LOOP",
            upstream_api="dm_control scalar Environment.reset/step -> TimeStep; MuJoCo physics underneath",
            native_batch="MuJoCo threaded engine rollout available; no drop-in complete task batch API used",
            parallel_route="Benchmark independent-owner threads versus persistent processes; retain task step",
            thread_constraint="MuJoCo C calls release GIL; Python task work remains; wrapper safety not proven",
            state_requirement="Integration/warmstart plus task model fields, lifecycle, task RNG and accumulated outputs",
            lifecycle="Preserve task-specific repeats/reward accumulation and upstream reset-next-step behavior",
            renderer_boundary="Independent renderer; GPU rendering does not change CPU dynamics",
        )
    if family == "fetch":
        return dict(
            sources=["stable_envs/_adapters/fetch_pickandplace.py", "stable_envs/envs/_fetch_semantics.py"],
            step_symbol="FetchAdapter.step",
            current_execution="SCALAR_CPU_LOOP",
            upstream_api="Gymnasium Robotics scalar task step plus TimeLimit(50)",
            native_batch="Generic AsyncVectorEnv available; raw MuJoCo rollout omits task semantics",
            parallel_route="Existing process vectorization with State facade or validated persistent thread/process shards",
            thread_constraint="Native physics may overlap; Python wrapper and resource ownership need tests",
            state_requirement="MuJoCo integration, goal, RNG, counters, gripper outputs and configured mass",
            lifecycle="Keep 50-step TimeLimit, reward variant, action dtype and reset/mass application order",
            renderer_boundary="GPU meshes or CPU OpenGL are separate from simulator ownership",
        )
    if family == "ogbench":
        maze = "Maze" in name
        return dict(
            sources=["stable_envs/_adapters/ogbench.py"]
            if maze
            else [
                "stable_envs/_adapters/ogbench_manipulation.py",
                "stable_envs/_adapters/ogbench_manipulation_compat.py",
            ],
            step_symbol="OGBMazeAdapter.step" if maze else "OGBManipulationAdapter.step",
            current_execution="SCALAR_CPU_LOOP",
            upstream_api="Scalar upstream-compatible OGBench task over MuJoCo",
            native_batch="Raw MuJoCo threaded rollout is engine-level, not complete OGBench task stepping",
            parallel_route="Persistent process shards"
            if maze
            else "Persistent owners; benchmark processes versus validated threads",
            thread_constraint="UNSAFE concurrent _in_legacy_rng: process-global NumPy state is swapped"
            if maze
            else "Separate mutable model/data/task/RNG owners; thread safety not established",
            state_requirement="Integration, goal, PCG and lane-owned legacy RNG"
            if maze
            else "Integration, task/button/controller state, multiple RNG streams, model/Variation; staged restoration",
            lifecycle="Preserve upstream task success/termination/reset and lane RNG streams",
            renderer_boundary="GPU renderer separately batches inputs"
            if maze
            else "Cube accepted batched compositor; Scene accepted fidelity renderer is serial OpenGL",
        )
    if family == "navigation":
        return dict(
            sources=["stable_envs/envs/navigation.py"],
            step_symbol="SimplePointMaze.step" if "SimplePointMaze" in name else "_HiddenTargetSemantics.step",
            current_execution="BATCHED_TORCH",
            upstream_api="Historical SWM scalar task; accepted candidate has native Torch batch dynamics",
            native_batch="Already implemented across tensor batch on configured CPU/CUDA device",
            parallel_route="Keep tensor batching; no process pool for ordinary step",
            thread_constraint="Reset/restore cost is distinct from batched step; no external thread safety claim",
            state_requirement="Agent/hidden target and family output/dtype state; independent owning snapshots",
            lifecycle="Keep historical target/reset semantics and exact output fields",
            renderer_boundary="Native batched renderer; SimplePointMaze reset/restore additionally rebuilds source goal cache with Matplotlib"
            if "SimplePointMaze" in name
            else "Native Torch renderer",
        )
    if family == "craftax":
        return dict(
            sources=[
                "stable_envs/_adapters/craftax.py",
                "stable_envs/_adapters/craftax_render.py",
                "stable_envs/envs/craftax_semantics.py",
            ],
            step_symbol="CraftaxAdapter.step",
            current_execution="SCALAR_JAX_KERNEL_LOOP",
            upstream_api="Functional key/State/action/params functions; scalar JIT currently selected per CPU/GPU backend",
            native_batch="Composable JAX vmap plus jit; not currently used for adapter step",
            parallel_route="DEFERRED: batch upstream functions with JAX; no multiprocessing for device lanes",
            thread_constraint="Do not confuse a Python loop of GPU kernels with native batched stepping",
            state_requirement="Engine pytree, wrapper key, RNG split order, lifecycle and retained output cache",
            lifecycle="auto_reset=False; explicit reset on terminal; preserve per-lane key splitting",
            renderer_boundary="Pixels computed during upstream step for pixel variants; symbolic registrations have no RGB renderer",
        )
    if family == "pusht":
        return dict(
            sources=["stable_envs/envs/pusht.py"],
            step_symbol="PushT.step",
            current_execution="UNACCEPTED_APPROXIMATE_NATIVE",
            upstream_api="Historical scalar Pymunk task; candidate approximation is not an accepted substitute",
            native_batch="Candidate tensor batch does not establish exact upstream semantics",
            parallel_route="BLOCKED_BY_SEMANTICS; document only, no implementation",
            thread_constraint="No accepted wrapper threading route or exact complete contact restore",
            state_requirement="Unresolved contact/broadphase history; physical fields alone insufficient",
            lifecycle="Continuous and discrete registrations require independent acceptance",
            renderer_boundary="No accepted matched-State renderer corpus in current paired table",
        )
    assert family == "rocket", family
    return dict(
        sources=[
            "stable_envs/_adapters/rocket.py",
            "stable_envs/_adapters/rocket_replay.py",
            "stable_envs/envs/rocket_semantics.py",
        ],
        step_symbol="RocketAdapter.step",
        current_execution="UNACCEPTED_SERIAL_CPU_REPLAY",
        upstream_api="Scalar PyFlyt task over Bullet client; multi-drone world is not independent task batching",
        native_batch="No compatible independent-task vector API established in accepted path",
        parallel_route="BLOCKED_BY_SEMANTICS; process-owned clients only a future candidate",
        thread_constraint="Native client ownership and wrapper globals require isolation audit",
        state_requirement="Replay/hidden contact state and future RNG/lifecycle guarantees remain incomplete",
        lifecycle="Preserve reset recipe, replay bounds and future reset behavior; no promotion",
        renderer_boundary="CPU TinyRenderer exists; GPU paired table currently ERROR",
    )


def main():
    mapping = json.loads(MAP.read_text())
    statuses = {r["historical_id"]: r["state_status"] for r in json.loads(MATRIX.read_text())["rows"]}
    records, hashes, symbols = [], {}, {}
    for row in mapping["rows"]:
        item = classify(row)
        paths = item.pop("sources")
        for path in paths:
            source = ROOT / path
            hashes[path] = fingerprint(source)
            tree = ast.parse(source.read_text())
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            symbols[node.name + "." + method.name] = f"{path}:{method.lineno}"
        symbol = item["step_symbol"]
        assert symbol in symbols, symbol
        record = dict(
            registration=row["historical_id"],
            family=row["family"],
            semantic_status=statuses[row["historical_id"]],
            **item,
            step_returns="Full State directly, with lifecycle fields"
            if row["family"] == "craftax"
            else "UNACCEPTED"
            if statuses[row["historical_id"]] != "VERIFIED"
            else "TensorDict with state plus family outputs",
            adapter_step_location=symbols[symbol],
            adapter_sources=";".join(paths),
            upstream_repository=row.get("upstream", {}).get("repository") or "Pinned stable-worldmodel",
            historical_profile_versions=json.dumps(
                row.get("upstream", {}).get("tested_profile_versions", {}), sort_keys=True
            ),
            swm_source_url=row.get("swm_implementation", {}).get("url", ""),
            audit_kind="STATIC_SOURCE; parallel-route recommendation unbenchmarked",
        )
        records.append(record)
    assert len(records) == len({r["registration"] for r in records}) == 42
    with (HERE / "catalog.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    package_root = Path("/Users/samib/se6-venv/lib/python3.12/site-packages")
    installed = {}
    for relative in [
        "gymnasium/vector/async_vector_env.py",
        "gymnasium/vector/sync_vector_env.py",
        "gymnasium/envs/classic_control/cartpole.py",
        "gymnasium/envs/__init__.py",
        "mujoco/rollout.py",
        "dm_control/rl/control.py",
    ]:
        path = package_root / relative
        installed[str(path)] = fingerprint(path)
    report = dict(
        kind="STATIC_SOURCE_AUDIT_NOT_ACCEPTANCE_OR_PERFORMANCE_GATE",
        date="2026-09-29",
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        counts=dict(Counter(r["current_execution"] for r in records)),
        semantic_counts=dict(Counter(r["semantic_status"] for r in records)),
        pinned_swm_commit=mapping["pinned_swm_commit"],
        input_sha256={str(p.relative_to(ROOT)): fingerprint(p) for p in [MAP, MATRIX]},
        current_adapter_sha256=hashes,
        installed_inspection_profile={"gymnasium": "1.3.0", "mujoco": "3.12.0", "dm-control": "1.0.45"},
        installed_source_sha256=installed,
        profile_note="Historical row metadata and installed inspection versions are distinct; no new profile acceptance claimed.",
        report_sha256=fingerprint(ROOT / "docs/research/online_execution_api_catalog.md"),
        catalog_sha256=fingerprint(HERE / "catalog.csv"),
        generator_sha256=fingerprint(Path(__file__)),
        rows=records,
    )
    (HERE / "catalog.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"registrations": len(records), "execution": report["counts"], "semantics": report["semantic_counts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
