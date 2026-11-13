"""Install-profile smoke: extras import, registry discovery, tiny CPU render where practical.

    python scripts/ci/smoke_extra.py --profile cheap

Profiles follow real extra compatibility, not an ``[all]`` extra. Fetch has no CPU renderer;
those cells are import / ``info()`` only.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

import stable_envs as snv

PROFILES: dict[str, dict] = {
    "cheap": {
        "expect": ("PushT", "CartPole"),
        "render": ("PushT", "CartPole"),
    },
    "mujoco": {
        "expect": ("DMCReacher", "PointMazeMediumNavigate", "OGBCube", "MSHumanLocomotion"),
        "render": ("DMCReacher", "PointMazeMediumNavigate", "OGBCube", "MSHumanLocomotion"),
    },
    "robotics": {
        "expect": ("FetchReach",),
        "render": (),
    },
    "special-import": {
        "expect": (),
        "render": (),
    },
}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    args = ap.parse_args(argv)
    spec = PROFILES[args.profile]
    registered = set(snv.list_envs())
    missing = [name for name in spec["expect"] if name not in registered]
    if missing:
        raise SystemExit(f"profile {args.profile!r} did not register {missing}. import_errors={snv.import_errors()!r}")

    rendered = {}
    for name in spec["render"]:
        cls = getattr(snv, name)
        with cls(num_envs=2, device="cpu", size=getattr(cls, "default_size", 32) or 32, seed=0) as env:
            state = env.reset(seed=0)
            pixels = env.render(state)
            assert pixels.shape[0] == 2 and pixels.shape[-1] == 3
            rendered[name] = list(pixels.shape)

    print(
        json.dumps(
            {
                "profile": args.profile,
                "device_auto": _device(),
                "registered": len(registered),
                "families": snv.list_families(),
                "import_errors": sorted(snv.import_errors()),
                "checked": list(spec["expect"]),
                "rendered": rendered,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
