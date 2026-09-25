# State performance / native feasibility starting checkpoint

Start: `13e51037cdb4afb9ce31aba3a4837ccf5c95adaf`, clean `repair/environment-semantics`, 39 VERIFIED / 3 INCOMPLETE.

`start.json` resolves all 39 registrations and 208 distinct indexed artifacts. `preservation.json` additionally resolves Craftax/Cube/Scene manifests, terminal evidence and retained Rocket counterexamples. `production-closure.json` verifies no production or frozen fixture changed since the final accepted checkpoint.

`rocket-commands-corrected.json` supersedes only the two incorrect Ruff result strings in the original Rocket commands metadata. The original tool output had exit code 0, "All checks passed!" and "3 files already formatted". Fresh read-only repetitions retain stdout/stderr and exit codes. The collector used a substring test for `verify_rocket_audit`, accidentally matching the filename in Ruff arguments. The sealed original commands and raw evidence remain unchanged; no accepted evidence is rewritten.

No push.
