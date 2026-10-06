# Cube/Scene renderer baseline

Captured through the public `env.render` API from commit
`28f7c8049b93b96d40e5c752922f1527b6eebb40`, before editing the manipulation
renderers. `manifest.json` pins the dependency versions, GPU, and archive hashes.

Each compressed archive contains numeric RenderState leaves and 257 uint8 RGB
frames at 64 × 64. Cases include default and varied cube/floor colors, all four
Scene button combinations, and nonidentity target orientations. The last row has
a distinct orientation and crosses the renderer's 256-row internal chunk boundary.

Run `pytest tests/gpu/test_ogbench_golden.py` on the pinned GPU profile. The test
requires exact equality and verifies that a changed pixel fails the comparison.
The initial GPU gate passed all three render cases and eight existing CPU boundary
tests. This guards cleanup against pixel changes; it is not a new source-parity claim.

`scripts/testing/capture_ogbench_golden.py` captures to a new directory only.
Do not regenerate these references to make a failing cleanup pass.
