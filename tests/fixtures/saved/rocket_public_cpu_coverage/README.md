# Rocket public CPU render report: coverage fixture

`result.json` keeps only the fields `verify_rocket_render` reads before loading frames: status, oracle
source hash, package versions, code hashes, device, the 204 case keys and the 41 group sizes.
It has no pixels or metrics, so it cannot pass verification. `tests/regression/test_saved_evidence_verifiers.py`
uses it to check that coverage and code-identity defects are rejected first.

`recorded_code.tar.gz` stores the files named in `production_hashes`, at their recorded paths, as they were at the
revision in `recorded_code.json`. The verifier checks their hashes against these stored copies instead of reading Git
history.

`../rocket_render_mse.json` records the per-frame MSE of the full CPU and CUDA archives. `verify_rocket_render`
checks a new measurement against it with the same band as `tests/fixtures/source_mse.json`.
