import ast
import hashlib
import importlib.metadata as md
import json
import subprocess
from pathlib import Path

root = Path("/private/tmp/stable-envs-semantic-recovery")
h = root / "docs/research/semantic_recovery_evidence/craftax_source_audit"
files = {}
modules = {}
roots = {
    "swm": Path("/Users/samib/stable-worldmodel"),
    "installed": Path("/private/tmp/craftax-recovery-312/lib/python3.12/site-packages"),
    "oracle_installed": Path("/Users/samib/stable-worldmodel/.venv/lib/python3.10/site-packages"),
}
for label, p in roots.items():
    selected = (
        [
            p / "stable_worldmodel/envs/__init__.py",
            p / "stable_worldmodel/envs/craftax/craftax.py",
            p / "pyproject.toml",
            p / "uv.lock",
        ]
        if label == "swm"
        else sorted((p / "craftax").rglob("*.py"))
    )
    for f in selected:
        rel = str(f.relative_to(p))
        key = label + ":" + rel
        files[key] = {"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size}
        if f.suffix == ".py":
            tree = ast.parse(f.read_text())
            methods = []
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                    methods.append({"name": n.name, "kind": type(n).__name__, "line": n.lineno, "end": n.end_lineno})
            modules[key] = methods
pairs = []
for k, v in files.items():
    if k.startswith("installed:"):
        other = files[k.replace("installed:", "oracle_installed:", 1)]
        assert other["sha256"] == v["sha256"], k
        pairs.append(k)
result = {
    "swm_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=roots["swm"], text=True).strip(),
    "official_repository": "https://github.com/MichaelTMatthews/Craftax",
    "official_release": "https://github.com/MichaelTMatthews/Craftax/releases/tag/v1.5.0",
    "official_tag": "v1.5.0",
    "official_commit": "47fff7a0b44b2842006661a2caab9ee6505fa210",
    "pypi": "https://pypi.org/project/craftax/1.5.0/",
    "wheel_sha256": "96826abc7d1fcf5ffbdd022100a6815b38c2b9bd22eaf43932a2252d2d7304d8",
    "sdist_sha256": "a8deddea60167934c23723a83df882e07a683c0e3ec7fb3a3e0a6364543401a9",
    "installed_craftax": md.version("craftax"),
    "isolated_vs_oracle_source_files_exact": len(pairs),
    "files": files,
    "definitions": modules,
    "production_written": False,
    "swm_game_fork": False,
    "swm_wrapper_lines": len((roots["swm"] / "stable_worldmodel/envs/craftax/craftax.py").read_text().splitlines()),
}
(h / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
print("SOURCE_HASH_AUDIT_PASS", len(pairs), "upstream Python files match oracle-installed Craftax")
