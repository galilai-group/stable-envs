"""Audit optional extras against a live import of the core package."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILY_EXTRAS = {
    "box2d": ("box2d-py", "pygame"),
    "craftax": ("craftax", "jax"),
    "crafter": ("crafter",),
    "dmcontrol": ("dm-control", "mujoco"),
    "griddly": ("griddly",),
    "highway": ("highway-env",),
    "humanoidbench": ("jax", "mujoco"),
    "jumanji": ("jumanji", "jax"),
    "metaworld": ("metaworld", "mujoco"),
    "minigrid": ("minigrid",),
    "minihack": ("minihack", "nle"),
    "mjwarp": ("mujoco", "mujoco-warp", "warp-lang"),
    "mujoco-locomotion": ("mujoco",),
    "ogbench": ("ogbench", "mujoco"),
    "overcooked": ("jaxmarl", "jax", "matplotlib"),
    "pgx": ("pgx",),
    "procgen": ("procgen2",),
    "pusht": ("gym-pusht", "pygame", "pymunk"),
    "robotics": ("gymnasium-robotics", "mujoco"),
    "rocket": ("pybullet", "pillow", "scipy"),
    "sokoban": ("gym-sokoban", "imageio"),
    "swm": ("stable-worldmodel",),
}


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    assert match is not None, requirement
    return match.group(0).lower().replace("_", "-")


def main() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    core = [_requirement_name(req) for req in pyproject["project"]["dependencies"]]

    assert "all" not in extras
    assert "nvdiffrast" not in {name for reqs in extras.values() for name in map(_requirement_name, reqs)}
    assert not any("@" in req for reqs in extras.values() for req in reqs)

    missing_extras = sorted(set(FAMILY_EXTRAS) - set(extras))
    assert missing_extras == [], missing_extras

    import stable_envs as snv

    errors = dict(snv.import_errors())
    registered = list(snv.list_envs())
    assert "PushT" in registered
    # A missing optional family must not prevent importing the package or PushT.
    assert "pusht" not in {name.lower() for name in errors}

    payload = {
        "core": core,
        "extras": sorted(extras),
        "registered_envs": len(registered),
        "optional_import_errors": {name: str(err)[:200] for name, err in sorted(errors.items())},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
