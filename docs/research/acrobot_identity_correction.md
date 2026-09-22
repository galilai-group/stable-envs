# Acrobot sparse-Balance identity correction

`swm/AcrobotDMControl-v0` selects `acrobot.Balance(sparse=True)`. Its canonical
public class is now `DMCAcrobotBalance`, with native ID `dmc-acrobot-balance` and
reference task `dm_control/acrobot-swingup_sparse`. The former
`DMCAcrobotSwingup` is not a public alias. The renderer still uses the same
Acrobot asset and unchanged compact layout.

Private reference/crosswalk machinery translates historical fixture names.
The original renderer corpus, Phase-0 audit, performance results, semantic
corpora and previous gate provenance are untouched. Additive
`tests/fixtures/semantics/dmc_first_family/acrobot_identity.json` binds the
canonical identity to the original oracle manifest hash.

The exact three-default DMC gate is rerun after the rename, including sparse
Balance reset, full-State restoration, reward, lifecycle/RNG and Variation
checks. CPU registry/DMC/oracle/mapping tests passed: 198 tests and three expected
CPU-only-profile skips for an unrelated CUDA-only cell. Three-cell renderer
preservation uses the same frozen rows, translating only the historical name.
Coordinated CPU/Oscar results are recorded in `semantic_recovery_evidence/`.

Subsequent family recovery integrates only into `repair/environment-semantics`.
`v0-draft` remains at the previously delivered `9471a61` checkpoint. OGBench
Scene is explicitly blocked by its default constructor; its failure is not a
permanent skip or a completed-family claim.
