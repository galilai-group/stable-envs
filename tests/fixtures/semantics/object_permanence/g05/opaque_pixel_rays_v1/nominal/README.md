# G05 independent geometric oracle v2

Contract: `stable-envs/opaque-pixel-rays-v1`. All source frames, 320x180.
Generated from evaluated Blender source geometry without candidate imports or BVH queries.
Depth and IDs share pixel-center rays; exactly equal depths choose lowest semantic ID.
Reproduce in a new directory with `python -m scripts.testing.freeze_object_permanence_rays --source-root <source> --output <new> --task G05`.
Legacy BVH references are retained separately; no thresholds are changed.
CC-BY-NC-4.0 source-derived data; see SOURCE_LICENSE.txt.
