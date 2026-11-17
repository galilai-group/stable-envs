"""Performance benchmarks for stable-envs; correctness and fidelity are checked by the test suite."""

import sys as _sys

# Ours imports stable_envs before torch, as a user's script that starts with `import stable_envs` does; its import-time
# defaults (OMP_WAIT_POLICY) then apply. Source processes keep their own import order.
if any(a.startswith("snv-") for a in _sys.argv) or _sys.argv[1:2] in (["ours"], ["mse"]):
    import stable_envs  # noqa: F401
