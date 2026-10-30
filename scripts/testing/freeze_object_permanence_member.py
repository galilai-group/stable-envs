"""Member-level independent oracle entry; existing producers remain callable.

The output is an oracle *root*, with task/contract/member subdirectories. Never
overwrite a frozen directory. Preparation and reference geometry remain in the
independent producers, separate from candidate export code.
"""

import argparse
from pathlib import Path

from scripts.object_permanence_member import Member


def directory(root, entry, contract):
    versions = {
        "stable-envs/opaque-pixel-rays-v1": "opaque_pixel_rays_v1",
        "stable-envs/opaque-pixel-rays-v2": "opaque_pixel_rays_v2",
        "stable-envs/transparent-pixel-rays-v1": "transparent_pixel_rays_v1",
        "stable-envs/sampled-optical-rays-v1": "sampled_optical_rays_v1",
    }
    if contract not in versions:
        raise ValueError("member oracle entry requires an explicit versioned visibility contract")
    return root / entry.task.lower() / versions[contract] / entry.leaf


def supports(entry, contract):
    """Independent producer availability; never infer support from script text."""
    if contract == "stable-envs/sampled-optical-rays-v1":
        return (entry.task, entry.member) in {("G01", 0), ("G01", 1), ("G03", 1)}
    if contract == "stable-envs/transparent-pixel-rays-v1":
        from .freeze_object_permanence_alpha import CASES

        return entry.member == 0 and entry.task in CASES
    from .freeze_object_permanence import REFERENCE_CASES, REFERENCE_MEMBERS, REFERENCE_ORTHOGRAPHIC_MEMBERS

    if entry.member:
        version = "v2" if (entry.task, entry.member) in REFERENCE_ORTHOGRAPHIC_MEMBERS else "v1"
        return (
            entry.task,
            entry.member,
        ) in REFERENCE_MEMBERS and contract == f"stable-envs/opaque-pixel-rays-{version}"
    return contract in {"stable-envs/opaque-pixel-rays-v1", "stable-envs/opaque-pixel-rays-v2"} and (
        entry.task in REFERENCE_CASES
    )


def freeze(source, output, entry, contract):
    if not supports(entry, contract):
        raise ValueError("no independently audited oracle producer for this member and contract")
    destination = directory(output, entry, contract)
    if contract == "stable-envs/sampled-optical-rays-v1":
        from .freeze_object_permanence_optical import freeze as produce_optical

        return produce_optical(source, destination, task=entry.task, member=entry.member)
    if contract == "stable-envs/transparent-pixel-rays-v1":
        from .freeze_object_permanence_alpha import freeze as produce_alpha

        if entry.member:
            raise ValueError("no independently audited alpha members beyond member zero")
        return produce_alpha(source, destination, entry.task)
    from .freeze_object_permanence_rays import freeze as produce_rays

    return produce_rays(
        source, destination, entry.task, member=entry.member, profile=entry.profile, visibility_contract=contract
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="oracle root, not a member directory")
    parser.add_argument("--task", required=True)
    parser.add_argument("--member", type=int, default=0)
    parser.add_argument("--profile", default="nominal")
    parser.add_argument("--visibility-contract", required=True)
    args = parser.parse_args()
    freeze(args.source_root, args.output, Member(args.task, args.member, args.profile), args.visibility_contract)


if __name__ == "__main__":
    main()
