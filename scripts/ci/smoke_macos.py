"""macOS / ordinary-CPU smoke: ``device='auto'`` and a tiny PushT render."""

from __future__ import annotations

import json
import sys

import torch

import stable_envs as snv


def main() -> None:
    assert "PushT" in snv.list_envs()
    auto = "cuda" if torch.cuda.is_available() else "cpu"
    with snv.PushT(num_envs=2, device="auto", size=32, seed=3) as env:
        assert str(env.device).startswith(auto), (env.device, auto)
        state = env.reset(seed=3)
        pixels = env.render(state)
        assert tuple(pixels.shape) == (2, 32, 32, 3)
    print(
        json.dumps(
            {
                "device_auto": auto,
                "env_device": str(env.device),
                "registered": len(snv.list_envs()),
                "pusht_shape": list(pixels.shape),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
