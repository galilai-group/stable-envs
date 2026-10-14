# Rocket public CPU render report: coverage fixture

`result.json` keeps only the fields `verify_rocket_render` reads before loading frames: status, oracle
source hash, package versions, code hashes, device, the 204 case keys and the 41 group sizes.
It has no pixels or metrics, so it cannot pass verification. `tests/regression/test_saved_evidence_verifiers.py`
uses it to check that coverage and code-identity defects are rejected first.
