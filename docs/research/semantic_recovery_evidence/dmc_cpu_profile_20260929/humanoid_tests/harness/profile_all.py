"""Run the current registered Humanoid boundary tests on the candidate source."""
import json
import sys
from pathlib import Path


def main():
    import pytest
    import torch
    from stable_envs.render import cpu3d

    torch.set_num_threads(1)
    output = Path(sys.argv[3])
    output.mkdir(parents=True, exist_ok=False)
    # Invalid work budgets must fail at the boundary, before accessing buffers.
    for value in (0, -1, 1.5, True):
        try:
            cpu3d.render(*([None] * 8), chunk_size=value)
        except ValueError as error:
            assert "positive integer" in str(error)
        else:
            raise AssertionError(value)
    code = pytest.main([
        "tests/unit/test_envs_dmcontrol.py",
        "-k", "HumanoidWalk and (constructor or projection or private_render)",
        "-q", f"--junitxml={output / 'pytest.xml'}",
    ])
    (output / "test_status.json").write_text(json.dumps({"exit_code": int(code), "chunk_validation_passed": True}) + "\n")
    raise SystemExit(code)
