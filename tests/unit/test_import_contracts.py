from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "stable_envs"
CORE = [
    "env.py",
    "_core/dynamics.py",
    "specs.py",
    "registry.py",
    "_core/config.py",
    "_core/style.py",
    "_core/mujoco_gl.py",
]


def _imports(path: Path, top_level_only: bool) -> list[str]:
    """Every module an import statement in ``path`` names, resolved to a dotted stable_envs path."""
    tree = ast.parse(path.read_text())
    rel = path.relative_to(ROOT).with_suffix("")
    pkg_parts = list(rel.parts[:-1])
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    out: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_parts[: len(pkg_parts) - node.level + 1]
                out.append(".".join([*base, node.module] if node.module else base))
            elif node.module:
                out.append(node.module)
    return out


@pytest.mark.parametrize("name", CORE)
def test_a_core_module_imports_no_backend_and_no_environment(name):
    bad = [
        m
        for m in _imports(PKG / name, top_level_only=True)
        if m.startswith(
            (
                "stable_envs.render",
                "stable_envs.envs",
                "stable_envs.interop",
                "stable_envs._interop",
                "stable_envs.compat",
                "stable_envs.testing",
                "stable_envs.data",
                "stable_envs.testing._rollout",
                "stable_worldmodel",
            )
        )
    ]
    assert bad == [], (
        f"stable_envs/{name} imports {bad} at module scope. The core vocabulary of the package must "
        f"not depend on a backend, an environment or an interop target — that arrow points the other "
        f"way, and a cycle here is a review BLOCK (contract §8)."
    )


@pytest.mark.parametrize("path", sorted((PKG / "render").glob("*.py")), ids=lambda p: p.name)
def test_a_renderer_knows_nothing_about_environments_or_stable_worldmodel(path):
    bad = [
        m
        for m in _imports(path, top_level_only=False)
        if m.startswith(
            (
                "stable_envs.envs",
                "stable_worldmodel",
                "stable_envs.interop",
                "stable_envs._interop",
                "stable_envs.compat",
            )
        )
    ]
    assert bad == [], f"stable_envs/render/{path.name} imports {bad}"


def test_the_installed_package_never_imports_stable_worldmodel():
    offenders = {}
    for path in PKG.rglob("*.py"):
        hits = [m for m in _imports(path, top_level_only=False) if m.startswith("stable_worldmodel")]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits
    assert offenders == {}, f"{offenders} import stable_worldmodel; the installed package must not"


def test_runtime_identifiers_do_not_embed_source_adapter_names():
    # Attribution and reference IDs are data; copied source adapter types are executable names.
    offenders = []
    for path in PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            name = (
                node.id
                if isinstance(node, ast.Name)
                else node.attr
                if isinstance(node, ast.Attribute)
                else node.arg
                if isinstance(node, ast.arg)
                else node.name
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                else ""
            )
            if any(token in name for token in ("_SWM", "SWM_", "swm_ref", "stable_worldmodel")):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{name}")
    assert offenders == [], offenders


def test_the_core_modules_have_no_cycle_among_themselves():
    """``env`` may import ``_core.dynamics``; ``_core.dynamics`` may not import ``env``."""
    edges = {
        name: {m for m in _imports(PKG / name, top_level_only=True) if m.startswith("stable_envs.")} for name in CORE
    }
    assert "stable_envs.env" not in edges["_core/dynamics.py"], (
        "dynamics must not import env: env imports dynamics, and the reverse arrow is how the pair "
        "becomes unimportable in the wrong order. Env sets `_se_dynamics_base` on itself instead."
    )
    assert not (edges["specs.py"] | edges["_core/config.py"]) & {"stable_envs.env", "stable_envs.registry"}


def test_the_package_imports_with_no_environment_family_available(tmp_path):
    """Importing ``stable_envs`` must not require any optional dependency."""
    import subprocess
    import sys

    code = "import sys; sys.modules['gym_pusht'] = None;import stable_envs as snv; print(snv.__version__)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.mark.parametrize("mujoco_gl", [None, "egl", "glfw"])
def test_package_import_returns_before_a_gl_hang(mujoco_gl):
    """``import stable_envs`` must not block inside GLFW.

    On macOS, ``MUJOCO_GL=glfw`` (or unset, which is glfw) hangs in ``dm_control._render`` at
    package import; ``egl`` is what cluster scripts export and is not a mujoco backend here.
    """
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    if mujoco_gl is None:
        env.pop("MUJOCO_GL", None)
    else:
        env["MUJOCO_GL"] = mujoco_gl
    code = "import stable_envs as snv; print('ok', snv.__version__, len(snv.list_envs()))"
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=20,
        env=env,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.startswith("ok "), r.stdout


def test_dm_control_import_does_not_hang_after_gl_configure():
    """Discovery imports dm_control. With GLFW that never returns; configure() must make it return."""
    import subprocess
    import sys

    pytest.importorskip("dm_control")
    code = (
        "from stable_envs._core.mujoco_gl import configure_mujoco_gl; "
        "configure_mujoco_gl(); "
        "from dm_control import mujoco as dm_mujoco; "
        "dm_mujoco.Physics.from_xml_string('<mujoco/>'); "
        "print('ok')"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=15,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "ok"


GONE_SHIMS = (
    "stable_envs/core",
    "stable_envs/compat",
    "stable_envs/interop.py",
    "stable_envs/_interop",
    "stable_envs/data.py",
    "stable_envs/envs/swm_nav.py",
    "stable_envs/envs/simple_point_maze.py",
    "stable_envs/envs/ogb_manip.py",
    "stable_envs/envs/craftax_full.py",
)
BANNED_IMPORTS = (
    "stable_envs.core",
    "stable_envs.compat",
    "stable_envs.interop",
    "stable_envs._interop",
    "stable_envs.data",
    "stable_envs.envs.swm_nav",
    "stable_envs.envs.simple_point_maze",
    "stable_envs.envs.ogb_manip",
    "stable_envs.envs.craftax_full",
)


def test_unpublished_step8_shims_are_gone():
    remaining = []
    for rel in GONE_SHIMS:
        path = ROOT / rel
        files = path.rglob("*") if path.is_dir() else [path]
        if any(
            file.is_file() and file.suffix in {".py", ".pyc", ".so", ".pyd"} and "__pycache__" not in file.parts
            for file in files
        ):
            remaining.append(rel)
    assert remaining == [], remaining


@pytest.mark.parametrize("error", [ImportError("missing extra"), RuntimeError("broken family")])
def test_craftax_import_failures_are_isolated(monkeypatch, error):
    import importlib

    import stable_envs.envs.craftax as craftax
    import stable_envs.registry as registry

    monkeypatch.setattr(registry, "_IMPORT_ERRORS", {})
    attempted = []

    def fail_import(name):
        attempted.append(name)
        raise error

    monkeypatch.setattr(registry.importlib, "import_module", fail_import)
    importlib.reload(craftax)
    assert attempted == [f"stable_envs.envs.craftax.{name}" for name in ("classic", "full", "semantics")]
    errors = registry.import_errors()
    assert set(errors) == {"craftax", "craftax_full", "craftax_semantics"}
    for message in errors.values():
        assert f"{type(error).__name__}: {error}" in message
        assert ("unexpected" in message) == isinstance(error, RuntimeError)


def _is_banned(mod: str) -> bool:
    return any(mod == b or mod.startswith(b + ".") for b in BANNED_IMPORTS)


def test_maintained_tests_and_examples_do_not_import_deleted_modules():
    offenders = {}
    for folder in ("tests", "examples", "stable_envs"):
        for path in (ROOT / folder).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            hits = [m for m in _imports(path, top_level_only=False) if _is_banned(m)]
            if hits:
                offenders[rel] = hits
    assert offenders == {}, offenders
