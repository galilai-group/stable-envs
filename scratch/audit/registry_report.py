"""What does THIS interpreter actually register, and what did it fail to import?

The DM Control parity gap has two possible shapes and they need completely different fixes: either the
envs are registered and the parity test dies on EGL, or they are not registered in the interpreter the
suite runs and the test never collects for them at all. Print both, per interpreter.
"""
from __future__ import annotations

import sys

import stable_envs as se

print("python:", sys.executable)
print(f"registry: {len(se.REGISTRY)} envs")
dmc = sorted(n for n in se.REGISTRY if n.startswith("DMC"))
print(f"DMC envs registered: {len(dmc)} {dmc}")
errs = se.envs.import_errors()
print(f"import errors: {len(errs)}")
for k, v in sorted(errs.items()):
    print(f"  {k}: {v}")
