"""Certify one member's complete oracle and cached/uncached render parity.

Run in the CUDA interpreter. Blender is used in separate import/oracle jobs.
Reports do not mutate either recordings or independent frozen references.
"""

import argparse
import json
from pathlib import Path

from scripts.object_permanence_member import Member
from scripts.testing.freeze_object_permanence_member import directory


def cache_check(recordings, entry, contract):
    import numpy as np
    import torch

    from stable_envs.envs.object_permanence import RigidTrajectory
    from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer

    results = []
    with RigidTrajectory(recordings, entry.clip) as clip:
        entry.check_source(clip.manifest["source"])
        frames = np.arange(len(clip))
        modes: tuple[str, ...] = ("rgb", "depth", "ids")
        if contract == "stable-envs/sampled-optical-rays-v1":
            modes += ("masks", "path_depth")
        elif contract == "stable-envs/transparent-pixel-rays-v1":
            modes += ("masks",)
        for size in (90, 180):
            with (
                RigidSceneRenderer(clip, size=size, visibility_contract=contract, cache_static=False) as full,
                RigidSceneRenderer(clip, size=size, visibility_contract=contract) as cached,
            ):
                expected = full.render_many(frames, modes=modes)
                actual = cached.render_many(frames[::-1].copy(), modes=modes)
                equal = all(torch.equal(actual[k], expected[k].flip(0)) for k in modes)
                probes = np.array([len(clip) // 2, 0, len(clip) - 1, len(clip) // 2, len(clip) // 3])
                for chunk in (1, 7):
                    cached.chunk_size = chunk
                    output = cached.render_many(probes, modes=modes)
                    equal &= all(torch.equal(output[k], expected[k][probes]) for k in modes)
                    # Caller mutation must not damage retained static seeds.
                    output["ids"].zero_()
                    equal &= torch.equal(cached.render(probes, mode="ids"), expected["ids"][probes])
                coverage = getattr(cached.renderer, "_ray_coverage", None)
                builds = None if coverage is None else coverage.builds
                fallbacks = None if coverage is None else coverage.fallbacks
                # Empty static partitions legitimately build no static seed.
                stable = coverage is None or (coverage.builds <= 1 and coverage.fallbacks == 0)
                results.append(
                    {
                        "height": size,
                        "frames": len(frames),
                        "bit_identical": bool(equal),
                        "chunks": [4, 1, 7],
                        "static_builds": builds,
                        "fallbacks": fallbacks,
                        "passed": bool(equal and stable),
                    }
                )
    return results


def check(recordings, oracles, entry, contract):
    oracle = directory(oracles, entry, contract)
    metadata = json.loads((oracle / "manifest.json").read_text())
    entry.check_source(
        {
            "task": metadata["task"],
            "member": metadata.get("member", 0),
            "entry_profile": metadata.get("entry_profile", "nominal"),
        }
    )
    if metadata.get("visibility_contract") != contract:
        raise ValueError("member oracle visibility contract differs")
    if contract == "stable-envs/sampled-optical-rays-v1":
        from .check_object_permanence_optical import check as optical

        results = [optical(recordings, oracle, cache_static=cache) for cache in (True, False)]
    elif contract == "stable-envs/transparent-pixel-rays-v1":
        from .check_object_permanence_alpha import check as alpha

        if entry.member:
            raise ValueError("no audited additional alpha members")
        results = [alpha(recordings, oracle, entry.task, cache_static=cache) for cache in (True, False)]
    else:
        from .check_object_permanence_rendering import check as opaque

        results = [
            opaque(
                recordings,
                oracle,
                entry.task,
                member=entry.member,
                profile=entry.profile,
                require_all_frames=True,
                visibility_contract=contract,
                cache_static=cache,
            )
            for cache in (True, False)
        ]
    caches = cache_check(recordings, entry, contract)
    return {
        "schema": "stable-envs/object-permanence-member-validation-v1",
        **entry.as_dict(),
        "clip": entry.clip,
        "visibility_contract": contract,
        "fidelity": results,
        "cache": caches,
        "passed": all(r["passed"] for r in results + caches),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--oracles", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--member", type=int, default=0)
    parser.add_argument("--profile", default="nominal")
    parser.add_argument("--visibility-contract", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = check(
        args.recordings, args.oracles, Member(args.task, args.member, args.profile), args.visibility_contract
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"clip": result["clip"], "passed": result["passed"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
