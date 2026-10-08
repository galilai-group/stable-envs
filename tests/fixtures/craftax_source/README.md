# Craftax source trajectories

Per-wrapper, per-scenario arrays recorded from the pinned stable-worldmodel
Craftax wrappers (`<Wrapper>-<case>.npz`). `tests/helpers/reference/craftax_source.py`
loads them as the source side of the Craftax rendering and replay checks.

These 32 files are byte-identical to the arrays first committed under
`docs/research/semantic_recovery_evidence/craftax_source_audit/`; `SHA256SUMS`
is that audit's hash list restricted to them. The audit's probes, logs and
provenance records remain in Git history at tag `pre-repository-cleanup`.
