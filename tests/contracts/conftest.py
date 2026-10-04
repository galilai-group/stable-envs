"""Required source profiles with explicit coverage and immutable run artifacts."""

import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from stable_envs.testing.evidence import EvidenceRun, sha256
from stable_envs.testing.provenance import pinned_source_identity
from tests.helpers.reference.cases import CaseKey, coverage
from tests.helpers.reference.profiles import CONTRACTS, PINS, PROFILES
from tests.helpers.reference.saved import SWM_PIN


def pytest_addoption(parser):
    parser.addoption("--contract-profile", choices=tuple(PROFILES), default=None)
    parser.addoption("--contract-device", choices=("cpu", "cuda"), default="cpu")
    parser.addoption("--contract-candidate-manifest", default=None, help="Verified file manifest for exported code")
    parser.addoption("--contract-source", default=None, help="Clean pinned stable-worldmodel checkout")
    parser.addoption("--contract-output", default=None, help="New directory for sealed results")


def parameters(profile):
    return [(case, seed, dtype) for case in PROFILES[profile] for seed in (17, 42) for dtype in case.dtypes]


def label(value):
    case, seed, dtype = value
    return f"{case.name}-seed{seed}-{dtype}"


def case_map(profile, device="cpu"):
    return {
        f"tests/contracts/{node}[{label(value)}]": CaseKey(
            value[0].registration, axis, f"seed{value[1]}/{value[2]}", device
        )
        for node, axis in CONTRACTS.items()
        for value in parameters(profile)
    }


def pytest_configure(config):
    profile = config.getoption("--contract-profile")
    config._contract_results = {}
    config._contract_duplicates = []
    config._contract_identity = None
    config._contract_run = None
    if profile:
        if getattr(config.option, "numprocesses", None):
            raise pytest.UsageError("Run required source profiles in one supervisor process, without pytest-xdist")
        output = config.getoption("--contract-output")
        if output:
            config._contract_run = EvidenceRun(output)
            config._contract_run.json(
                "plan.json",
                dict(
                    profile=profile,
                    scope="source regression subset; no promotion",
                    required=[asdict(k) for k in case_map(profile, config.getoption("--contract-device")).values()],
                ),
            )
            root = Path(__file__).resolve().parents[2]
            paths = [
                path
                for directory in ("stable_envs", "tests/contracts", "tests/helpers/reference", "scripts/testing")
                for path in (root / directory).rglob("*.py")
            ]
            paths += [root / "tests/conftest.py", root / "pyproject.toml"]
            paths += [
                root / name
                for name in (
                    "scripts/recovery/rocket_feasibility.py",
                    "scripts/research_replay/pusht_replay_gate.py",
                    "docs/research/semantic_recovery_evidence/ogb_rng_feasibility/ownership.py",
                    "docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/probe.py",
                    "scripts/benchmarks/online_stepping/fetch_source_pipeline.py",
                    "scripts/benchmarks/online_stepping/fetch_source_render_lifecycle.py",
                )
            ]
            manifest_path = config.getoption("--contract-candidate-manifest")
            if manifest_path:
                import json

                manifest = json.loads(Path(manifest_path).read_text())
                commit, dirty = manifest["commit"], manifest.get("tracked_dirty", False)
                for path in paths:
                    name = str(path.relative_to(root))
                    if manifest["files"].get(name) != sha256(path):
                        raise pytest.UsageError(f"Candidate manifest mismatch: {name}")
            else:
                commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
                dirty = subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
                )
            config._contract_run.json(
                "candidate.json",
                dict(
                    commit=commit,
                    tracked_dirty=bool(dirty),
                    files={str(p.relative_to(root)): sha256(p) for p in paths},
                ),
            )


def pytest_generate_tests(metafunc):
    if "source_case" in metafunc.fixturenames:
        # Unrequested live profiles skip explicitly; frozen corpus tests still run.
        profile = metafunc.config.getoption("--contract-profile") or "native"
        values = parameters(profile)
        metafunc.parametrize("source_case", values, ids=[label(value) for value in values])


@pytest.fixture(scope="session")
def source_identity(request):
    profile = request.config.getoption("--contract-profile")
    if profile is None:
        pytest.skip("Select --contract-profile and --contract-source for independent live source checks")
    root = request.config.getoption("--contract-source")
    try:
        if not root:
            raise ValueError("--contract-source is required")
        import torch

        device = request.config.getoption("--contract-device")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Required CUDA output device unavailable")
        identity = pinned_source_identity(root, SWM_PIN, PINS[profile])
        from tests.helpers.reference.bridges import load_source

        load_source(root, profile)
        identity["candidate_state_device"] = device
        identity["gpu"] = torch.cuda.get_device_name() if device == "cuda" else None
        identity["import_scope"] = "Unmodified source envs/utils/spaces; learning-stack package initializers omitted"
        # Import required source modules now: transitive dependency failures are
        # BLOCKED prerequisites, not simulator discrepancies repeated per case.
        import importlib

        modules = {
            "native": "stable_worldmodel.envs.gymnasium_control.cartpole",
            "fetch": "stable_worldmodel.envs.gymnasium_robotics",
            "rocket": "stable_worldmodel.envs.rocket_landing.pyflyt_rocketlanding",
            "dmc": "stable_worldmodel.envs.dmcontrol.reacher",
            "navigation": "stable_worldmodel.envs.two_room.env",
            "ogbench": "stable_worldmodel.envs.ogbench.maze_env",
            "pusht": "stable_worldmodel.envs.pusht.env",
        }
        importlib.import_module(modules[profile])
    except (ValueError, OSError, ImportError) as error:
        request.config._contract_identity = {"status": "BLOCKED", "reason": str(error)}
        pytest.skip(f"BLOCKED required source: {error}")
    request.config._contract_identity = identity
    return identity


@pytest.fixture
def worlds(source_case, source_identity, request):
    import stable_envs as se
    from tests.helpers.reference.bridges import BRIDGES

    case, seed, dtype = source_case
    kwargs = dict(num_envs=2, device=request.config.getoption("--contract-device"), size=64)
    if case.family == "rocket":
        kwargs["simulator_config"] = dict(render_mode="rgb_array", render_resolution=(64, 64))
    with se.make(case.name, **kwargs) as first, se.make(case.name, **kwargs) as second:
        reference = BRIDGES[case.family](case, 2)
        try:
            first.reset(seed=seed)
            second.reset(seed=seed + 100)
            reference.reset(seed)
            rng = np.random.default_rng(seed + 1000)
            shape = (24, 2, case.width)
            actions = (
                rng.integers(
                    0, 16 if case.family == "pusht" else 2 if case.domain == "cartpole" else 3, shape, dtype=np.int64
                )
                if dtype == "int64"
                else rng.uniform(-0.2, 0.2, shape).astype(dtype)
            )
            if case.family == "navigation":
                actions *= 0.05
            if case.family == "rocket":
                actions[..., 3:5] = 0
            yield first, second, reference, actions
        finally:
            reference.close()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    profile = item.config.getoption("--contract-profile")
    if not profile or report.nodeid not in case_map(profile):
        return
    previous = item.config._contract_results.get(report.nodeid)
    if report.failed:
        status = "FAIL"
    elif report.skipped:
        status = "BLOCKED"
    elif report.when == "call":
        status = "PASS"
    else:
        return
    if previous and previous["status"] == "FAIL":
        return
    if previous and report.when == "call":
        item.config._contract_duplicates.append(report.nodeid)
    item.config._contract_results[report.nodeid] = dict(
        status=status, phase=report.when, detail=str(report.longrepr) if not report.passed else None
    )
    run = item.config._contract_run
    if run:
        progress = "progress/" + report.nodeid.split("/")[-1].replace("::", "-") + "-" + report.when
        if report.nodeid in item.config._contract_duplicates:
            progress += "-duplicate-" + str(len(item.config._contract_duplicates))
        run.json(progress + ".json", dict(node=report.nodeid, **item.config._contract_results[report.nodeid]))
    if report.failed and report.when == "call" and run and "worlds" in item.funcargs:
        candidate, _, _, actions = item.funcargs["worlds"]
        # Keep the deterministic action plan and current numeric State even when
        # an assertion interrupts a trajectory. The node ID replays that case.
        prefix = "failures/" + report.nodeid.split("/")[-1].replace("::", "-")
        arrays = {"actions": actions}
        try:
            arrays.update(
                {
                    "candidate/" + (key if isinstance(key, str) else "/".join(key)): value.detach().cpu().numpy()
                    for key, value in candidate.get_state().items(True, True)
                }
            )
        except Exception as error:
            run.json(prefix + "-capture-error.json", dict(error=str(error)))
        run.arrays(prefix + ".npz", **arrays)
        run.json(
            prefix + ".json",
            dict(
                node=report.nodeid,
                reason=str(report.longrepr),
                replay=[
                    "python",
                    "-m",
                    "pytest",
                    report.nodeid,
                    "--contract-profile",
                    profile,
                    "--contract-source",
                    item.config.getoption("--contract-source"),
                ],
            ),
        )


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    profile = config.getoption("--contract-profile")
    if not profile or config.option.collectonly:
        return
    mapping = case_map(profile, config.getoption("--contract-device"))
    results = config._contract_results
    observed = [(mapping[node], row["status"]) for node, row in results.items()]
    observed += [(mapping[node], results[node]["status"]) for node in config._contract_duplicates]
    accounting = coverage(mapping.values(), observed)
    status = accounting["status"]
    if exitstatus not in (0, 5) and status == "PASS":
        status = "FAIL"
    if status != "PASS":
        session.exitstatus = 1
    report = dict(
        status=status,
        profile=profile,
        scope="source regression subset; no promotion",
        required=accounting["required"],
        passed=accounting["passed"],
        missing=[asdict(k) for k in accounting["missing"]],
        cases=results,
        duplicates=config._contract_duplicates,
        source=config._contract_identity,
    )
    run = config._contract_run
    if run:
        run.json("result.json", report)
        run.seal()
    terminal = config.pluginmanager.getplugin("terminalreporter")
    if terminal:
        terminal.write_line(
            f"Required {profile} source profile: {status}, {accounting['passed']}/{accounting['required']} passed"
        )
