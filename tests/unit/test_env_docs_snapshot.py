"""The environment page snapshots in docs/environments/_data match the registered classes.

``scripts/gen_env_docs.py`` writes the snapshots. Each class is checked in the profile that owns its
module, against an env built on CPU.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

import stable_envs as snv
from tests.helpers.exemptions import source_owners

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "environments" / "_data"
PAGES = ROOT / "docs" / "environments"
MEDIA = ROOT / "docs" / "assets" / "media" / "envs"
SNAPSHOTS = sorted(DATA.glob("*.json"))


def _generator():
    spec = importlib.util.spec_from_file_location("gen_env_docs", ROOT / "scripts" / "gen_env_docs.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _generator()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _profile(path: Path) -> str:
    module = _load(path)["module"].replace(".", "/")
    source = f"{module}.py" if (ROOT / f"{module}.py").is_file() else f"{module}/__init__.py"
    found = source_owners(source)
    return found[0] if len(found) == 1 else "core"


def test_there_are_snapshots():
    assert len(SNAPSHOTS) >= 74, f"only {len(SNAPSHOTS)} snapshots under {DATA}"


def test_every_registered_class_has_a_snapshot():
    missing = sorted(set(snv.list_envs()) - {p.stem for p in SNAPSHOTS})
    assert not missing, f"no snapshot for {missing}; run scripts/gen_env_docs.py --meta --devices --media"


def test_every_snapshot_has_a_page_section():
    sections = set()
    for page in PAGES.glob("*.md"):
        sections |= set(re.findall(r"^<!-- env: (\w+) -->$", page.read_text(), re.M))
    assert sections == {p.stem for p in SNAPSHOTS}


@pytest.mark.parametrize("path", SNAPSHOTS, ids=[p.stem for p in SNAPSHOTS])
def test_snapshot_media_exists(path):
    media = _load(path).get("media") or {}
    for clip in ("rollout", "variation"):
        if media.get(clip):
            assert (MEDIA / path.stem / media[clip]["file"]).is_file(), f"{path.stem}: missing {clip} clip"


@pytest.mark.parametrize(
    "path", [pytest.param(p, id=p.stem, marks=pytest.mark.family(_profile(p))) for p in SNAPSHOTS]
)
def test_snapshot_matches_the_live_class(path):
    name = path.stem
    if name not in snv.list_envs():
        pytest.skip(f"{name} is not registered in this interpreter")
    assert GEN.metadata(name) == GEN.comparable(_load(path)), (
        f"{name}: the snapshot is stale; run scripts/gen_env_docs.py {name} --meta"
    )
