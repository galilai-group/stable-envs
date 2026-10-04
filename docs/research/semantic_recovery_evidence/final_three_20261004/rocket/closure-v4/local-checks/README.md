# Local shared closure checks

The first run passed 55 cases and failed two MiniGrid cases before construction
because the optional MiniGrid package was absent. After installing MiniGrid
2.5.0 with --no-deps, the same selection passed all 57 cases. The unrelated
Craftax renderer-only case was explicitly deselected; zero selected cases skipped.
These are CPU/shared contracts, not CUDA evidence or a full-registry source rerun.

The artifact reverification resolves the existing 37 accepted non-Craftax rows'
183 distinct implementation-map artifact hashes. It is a preservation check.
