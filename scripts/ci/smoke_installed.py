"""Smoke-test an installed distribution without importing from the repository root."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import stable_envs as snv


def main() -> None:
    assert importlib.metadata.version("stable-envs") == snv.__version__
    assert Path(snv.__file__).name == "__init__.py"
    assert "CartPole" in snv.list_envs()

    # These modules must import without their optional CUDA/MuJoCo implementations.
    from stable_envs.render import mjwarp, nvdiff

    assert mjwarp is not None and nvdiff is not None

    with snv.CartPole(num_envs=2, device="cpu", seed=7) as env:
        state = env.reset(seed=7)
        pixels = env.render(state)
        assert pixels.shape[0] == 2 and pixels.shape[-1] == 3
        assert str(pixels.dtype).endswith("uint8")

    print(
        json.dumps(
            {
                "distribution_version": snv.__version__,
                "package": str(Path(snv.__file__).resolve()),
                "registered_envs": len(snv.list_envs()),
                "optional_import_errors": sorted(snv.import_errors()),
                "pusht_shape": list(pixels.shape),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
