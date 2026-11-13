"""Checks for the ``examples/`` scripts.

Every script compiles, does no work at import time, prints ``--help`` if it uses argparse, and runs
end to end on CPU with small arguments in a subprocess. Scripts needing an optional dependency or
env this interpreter lacks skip with the reason.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"

#: Directories under ``examples/`` that are outputs or vendored, not scripts to check.
SKIP_DIRS = {"results", "media", "logs", "tables", "figures", "config", "__pycache__", ".hydra", ".venv"}


def _scripts() -> list[Path]:
    out = []
    for p in sorted(EXAMPLES.rglob("*.py")):
        if SKIP_DIRS & set(p.relative_to(EXAMPLES).parts):
            continue
        out.append(p)
    return out


SCRIPTS = _scripts()
IDS = [str(p.relative_to(EXAMPLES)) for p in SCRIPTS]


def test_there_are_scripts_to_check():
    """The script glob finds scripts, so the parametrized tests below are not empty."""
    assert len(SCRIPTS) >= 7, f"only {len(SCRIPTS)} scripts found under {EXAMPLES}"


@pytest.mark.parametrize("path", SCRIPTS, ids=IDS)
def test_every_example_script_parses(path: Path):
    """Each script compiles under the running interpreter."""
    src = path.read_text()
    compile(src, str(path), "exec")


@pytest.mark.parametrize("path", SCRIPTS, ids=IDS)
def test_a_runnable_script_does_its_work_behind_a_main_guard(path: Path):
    """No module-level ``main()``/``run()``/``sweep()`` call, so importing a script does no work."""
    tree = ast.parse(path.read_text(), str(path))
    calls = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    named = [c for c in calls if isinstance(c.value.func, ast.Name) and c.value.func.id in {"main", "run", "sweep"}]
    assert not named, (
        f"{path.relative_to(ROOT)} calls {named[0].value.func.id}() at import time; put it behind "
        f'`if __name__ == "__main__":`'
    )


ARGPARSE = [p for p in SCRIPTS if "argparse" in p.read_text()]
ARGPARSE_IDS = [str(p.relative_to(EXAMPLES)) for p in ARGPARSE]


def test_the_argparse_sweep_below_is_not_empty():
    assert len(ARGPARSE) >= 6, f"only {len(ARGPARSE)} scripts under examples/ use argparse"


def _run(path: Path, args: list[str]) -> subprocess.CompletedProcess:
    """Run a script on CPU; skip when it stops on a package this interpreter does not have."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(EXAMPLES), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    p = subprocess.run(
        [sys.executable, str(path), *args], env=env, capture_output=True, text=True, timeout=600, cwd=str(ROOT)
    )
    if p.returncode != 0:
        tail = (p.stderr or p.stdout)[-2500:]
        last = [ln for ln in (p.stderr or "").strip().splitlines() if ln.strip()]
        if last and last[-1].startswith(("ModuleNotFoundError", "ImportError")):
            pytest.skip(f"{path.relative_to(ROOT)} needs a package this interpreter does not have: {last[-1]}")
        pytest.fail(f"{path.relative_to(ROOT)} {' '.join(args)} exited {p.returncode}:\n{tail}")
    return p


@pytest.mark.parametrize("path", ARGPARSE, ids=ARGPARSE_IDS)
def test_every_argparse_script_can_print_its_own_help(path: Path):
    """``--help`` imports the module, builds the parser, prints usage and exits 0."""
    p = _run(path, ["--help"])
    assert "usage" in (p.stdout + p.stderr).lower(), (
        f"{path.relative_to(ROOT)} --help exited 0 without printing a usage line"
    )


#: Small CPU arguments per script, and the env each needs registered.
SMOKE = {
    "action_sequences.py": ("CartPole", []),
    "batched_env.py": ("CartPole", ["--env", "CartPole", "--num-envs", "3", "--short", "2", "--device", "cpu"]),
    "bench_gpu_vs_cpu.py": ("CartPole", ["--envs", "CartPole", "--batches", "1,4", "--pool", "2", "--reps", "1"]),
    "bench_pusht_source.py": (
        "PushT",
        ["--batches", "1", "--sizes", "96", "--episodes", "1", "--repeats", "1", "--source-frames", "1"],
    ),
    "domain_randomization.py": ("CartPole", ["--num-envs", "4", "--device", "cpu", "--reps", "1"]),
    "gallery.py": ("CartPole", ["--envs", "CartPole", "--num-envs", "2", "--steps", "2", "--device", "cpu"]),
    "render.py": ("CartPole", ["--env", "CartPole", "--num-envs", "2", "--device", "cpu"]),
    "lewm/state_render.py": ("PushT", []),
    "lewm/make_dataset.py": ("PushT", ["{tmp}/states.h5", "{tmp}/pixels.h5", "--size", "96", "--batch", "16"]),
    "lewm/train.py": (
        "PushT",
        [
            "--data",
            "{tmp}/states.h5",
            "--out",
            "{tmp}/runs",
            "--half",
            "--precision",
            "32",
            "--accelerator",
            "cpu",
            "--max-steps",
            "2",
            "loader.batch_size=4",
            "loader.num_workers=0",
            "img_size=56",
        ],
    ),
    "lewm/eval.py": (
        "PushT",
        [
            "--data",
            "{tmp}/states.h5",
            "--policy",
            "random",
            "--device",
            "cpu",
            "--num-eval",
            "2",
            "eval.eval_budget=4",
            "eval.goal_offset_steps=3",
            "img_size=96",
        ],
    ),
}


def _write_lewm_states(path: Path) -> None:
    """Write two short PushT episodes in the stable-worldmodel HDF5 layout: states, proprio and actions."""
    import numpy as np

    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    n, episodes = 40, 2
    state = np.concatenate(
        [
            rng.uniform(150, 350, (n * episodes, 4)),
            rng.uniform(0, 2 * np.pi, (n * episodes, 1)),
            np.zeros((n * episodes, 2)),
        ],
        axis=1,
    ).astype(np.float32)
    with h5py.File(path, "w") as f:
        f["state"] = state
        f["proprio"] = state[:, [0, 1, 5, 6]]
        f["action"] = rng.uniform(-1, 1, (n * episodes, 2)).astype(np.float32)
        f["ep_idx"] = np.repeat(np.arange(episodes), n)
        f["step_idx"] = np.tile(np.arange(n), episodes)
        f["ep_len"] = np.full(episodes, n, np.int32)
        f["ep_offset"] = np.arange(episodes, dtype=np.int64) * n


def test_every_script_has_smoke_arguments():
    assert sorted(SMOKE) == sorted(IDS), "add small CPU arguments for each script to SMOKE"


@pytest.mark.parametrize("path", SCRIPTS, ids=IDS)
def test_every_script_runs_end_to_end(path: Path, tmp_path: Path):
    """Each script runs to completion on CPU with small arguments."""
    import stable_envs as snv

    name, args = SMOKE[str(path.relative_to(EXAMPLES))]
    if name not in snv.list_envs():
        pytest.skip(f"{name} is not registered in this interpreter")
    if path.name == "gallery.py":
        args = [*args, "--out", str(tmp_path)]
    if path.parent.name == "lewm":
        _write_lewm_states(tmp_path / "states.h5")
        args = [arg.replace("{tmp}", str(tmp_path)) for arg in args]
    _run(path, args)
    if path.name == "gallery.py":
        written = sorted(p.name for p in (tmp_path / "CartPole").iterdir())
        assert written == ["reset.png", "rollout.png", "variation.png"], written
        assert all((tmp_path / "CartPole" / n).read_bytes().startswith(b"\x89PNG") for n in written)


def test_the_readme_lists_the_scripts_that_exist():
    """``examples/README.md`` mentions every top-level script and points only at existing ones."""
    readme = (EXAMPLES / "README.md").read_text()
    top = sorted(p.name for p in EXAMPLES.glob("*.py"))
    missing = [n for n in top if n not in readme]
    assert not missing, f"examples/README.md does not mention {missing}"
    for line in readme.splitlines():
        for tok in line.replace("`", " ").replace("(", " ").replace(")", " ").split():
            if tok.startswith("examples/") and tok.endswith(".py"):
                assert (ROOT / tok).exists(), f"examples/README.md points at {tok}, which is not there"
