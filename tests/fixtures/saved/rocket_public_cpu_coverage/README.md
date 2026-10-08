# Rocket public CPU render report: coverage fixture

`result.json` keeps only the fields that `verify_rocket_render` reads before it
loads any frames: report status, oracle source hash, package versions, recorded
code hashes, device, the 204 case keys and the 41 group sizes. It contains no
pixels or metrics, so it cannot pass verification; tests use it to check that
coverage and code-identity defects are rejected first.

Derived from the accepted report
`docs/research/semantic_recovery_evidence/final_three_20261004/rocket/closure-v4/final/public-cpu/result.json`
(sha256 `8edc4c32ddad91e94d5185108c479fc42b8fcace467428d15ac7f53417a9a6a6`),
whose production hashes were recorded at commit
`ebf82e6eeed8b2b9c7560e4dff6f9a6b5c95e95b`. The full report and frames remain in
Git history at tag `pre-repository-cleanup`.
