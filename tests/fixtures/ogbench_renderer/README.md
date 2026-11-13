# Cube/Scene renderer baseline

Each `.npz` holds RenderState leaves and 257 uint8 RGB frames at 64x64 captured through `env.render`;
`manifest.json` pins the dependency versions, GPU and archive hashes.
`tests/gpu/test_ogbench_golden.py` requires exact pixel equality on the pinned GPU profile.
Capture with `scripts/testing/capture_ogbench_golden.py`, which writes to a new directory only.
