"""Audit the release archives using only the Python standard library."""

from __future__ import annotations

import argparse
import email
import re
import tarfile
import zipfile
from pathlib import Path

CORE_REQUIREMENTS = {"gymnasium", "loguru", "numpy", "tensordict", "torch"}
REQUIRED_EXTRAS = {
    "dev",
    "docs",
    "humanoidbench",
    "interop",
    "metaworld",
    "mjwarp",
    "rocket",
}
EXPECTED_AUTHORS = ("Randall Balestriero", "Sami BuGhanem")


def _package_sources(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in (root / "stable_envs").rglob("*.py")
        if "__pycache__" not in path.parts
    }


def _audit_metadata(text: str) -> None:
    metadata = email.message_from_string(text)
    assert metadata["Name"] == "stable-envs"
    assert metadata["Version"]
    assert metadata["Requires-Python"] == ">=3.10"
    assert metadata["License-Expression"] == "MIT"

    requirements = metadata.get_all("Requires-Dist", [])
    assert not any(" @ " in requirement for requirement in requirements), (
        "PyPI rejects direct-URL Requires-Dist metadata"
    )
    core = {
        re.match(r"[A-Za-z0-9_.-]+", requirement).group(0).lower().replace("_", "-")
        for requirement in requirements
        if "extra ==" not in requirement
    }
    assert core == CORE_REQUIREMENTS, (core, CORE_REQUIREMENTS)

    extras = set(metadata.get_all("Provides-Extra", []))
    assert REQUIRED_EXTRAS <= extras
    assert "all" not in extras, "the dependency stacks are not compatible in one environment"

    author_fields = [metadata.get("Author"), metadata.get("Author-email"), metadata.get("Author-Email")]
    author_fields.extend(metadata.get_all("Author") or [])
    author_blob = " ".join(field for field in author_fields if field)
    assert "Bou Ghanem" not in author_blob, author_blob
    positions = [author_blob.find(name) for name in EXPECTED_AUTHORS]
    assert all(position >= 0 for position in positions), author_blob
    assert positions == sorted(positions), author_blob


def audit_wheel(path: Path, expected_sources: set[str]) -> None:
    assert path.name.endswith("-py3-none-any.whl"), path.name
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_files) == 1, metadata_files
        _audit_metadata(archive.read(metadata_files[0]).decode())
        assert expected_sources <= names, sorted(expected_sources - names)
        unexpected = [
            name
            for name in names
            if not (name.startswith("stable_envs/") or ".dist-info/" in name)
        ]
        assert unexpected == [], unexpected


def audit_sdist(path: Path, expected_sources: set[str]) -> None:
    with tarfile.open(path, "r:gz") as archive:
        raw_names = archive.getnames()
        roots = {name.split("/", 1)[0] for name in raw_names}
        assert len(roots) == 1, roots
        root = roots.pop()
        names = {
            name.removeprefix(f"{root}/")
            for name in raw_names
            if name != root
        }
        required = {"LICENSE", "README.md", "pyproject.toml", *expected_sources}
        assert required <= names, sorted(required - names)
        pkg_info = archive.extractfile(f"{root}/PKG-INFO")
        assert pkg_info is not None
        _audit_metadata(pkg_info.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    wheels = sorted(args.dist.glob("*.whl"))
    sdists = sorted(args.dist.glob("*.tar.gz"))
    assert len(wheels) == 1, wheels
    assert len(sdists) == 1, sdists

    expected_sources = _package_sources(root)
    audit_wheel(wheels[0], expected_sources)
    audit_sdist(sdists[0], expected_sources)
    print(
        f"audited {wheels[0].name} and {sdists[0].name}: "
        f"{len(expected_sources)} package modules"
    )


if __name__ == "__main__":
    main()
