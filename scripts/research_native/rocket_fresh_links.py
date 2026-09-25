"""Diagnostic semantic migration: make PyFlyt component getters request current link frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PyFlyt.core.abstractions.boring_bodies import BoringBodies
from PyFlyt.core.abstractions.lifting_surfaces import LiftingSurfaces

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/recovery"))
import rocket_feasibility as probe  # noqa: E402


def patch(cls):
    original = cls.state_update

    def update(self, rotation_matrix):
        prior = self.p.__dict__.get("getLinkStates")
        getter = self.p.getLinkStates

        def fresh(*args, **kwargs):
            return getter(*args, **dict(kwargs, computeForwardKinematics=True))

        self.p.getLinkStates = fresh
        try:
            return original(self, rotation_matrix)
        finally:
            if prior is None:
                del self.p.getLinkStates
            else:
                self.p.getLinkStates = prior

    cls.state_update = update


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    patch(BoringBodies)
    patch(LiftingSurfaces)
    sys.argv = ["rocket_feasibility", "--output", str(args.output)]
    probe.main()
    reference = ROOT / "docs/research/semantic_recovery_evidence/rocket_source_audit/feasibility-final"
    differences = {}
    for path in args.output.glob("*-reference.npz"):
        with np.load(path, allow_pickle=False) as fresh, np.load(reference / path.name, allow_pickle=False) as pinned:
            differences[path.stem] = probe.differences(dict(pinned), dict(fresh))
    (args.output / "ordinary-trajectory-changes.json").write_text(
        json.dumps(
            {
                "semantics": "Possible migration only, not faithful restoration",
                "getter_change": "computeForwardKinematics=True during body/fin component state_update",
                "differences_from_pinned_oracle": differences,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
