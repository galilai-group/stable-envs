"""Run the DMC MJWarp CUDA integration contracts; missing CUDA or any skip fails."""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class RequireEveryCase:
    def __init__(self):
        self.incomplete = False

    def pytest_runtest_logreport(self, report):
        self.incomplete |= report.skipped

    def pytest_collectreport(self, report):
        self.incomplete |= report.skipped

    def pytest_deselected(self, items):
        self.incomplete |= bool(items)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.incomplete:
            session.exitstatus = 1


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("DMC MJWarp integration requires an actual CUDA device")
    from stable_envs.registry import REGISTRY

    names = {name for name, cls in REGISTRY.items() if cls.family == "dmcontrol"}
    if len(names) != 17:
        raise RuntimeError(f"Expected all 17 DMC MJWarp environments, found {len(names)}")
    os.environ["SE_REQUIRE_FAMILIES"] = "dmc"
    return pytest.main(
        [
            str(ROOT / "tests/gpu/test_dmc_mjwarp.py"),
            str(ROOT / "tests/gpu/test_dmc_mjwarp_reacher.py"),
            str(ROOT / "tests/gpu/test_dmc_mjwarp_streams.py"),
            str(ROOT / "tests/unit/test_dmc_mjwarp_contacts.py")
            + "::test_constraint_order_preserves_contact_impulses[cuda]",
            str(ROOT / "tests/unit/test_dmc_mjwarp_contacts.py")
            + "::test_elliptic_contact_reduction_preserves_impulses[cuda]",
            str(ROOT / "tests/unit/test_dmc_mjwarp_contacts.py")
            + "::test_near_parallel_capsules_preserve_contacts_and_impulses[cuda]",
            "-ra",
        ],
        plugins=[RequireEveryCase()],
    )


if __name__ == "__main__":
    raise SystemExit(main())
