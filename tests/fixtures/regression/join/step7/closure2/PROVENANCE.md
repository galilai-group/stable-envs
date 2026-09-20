# The Step-7 closure RE-SWEEP: what was measured, where, and by which job

| | |
|---|---|
| integration SHA verified | `fec4f6eea606e390c22e2b206b3d0738d11468c8` (all seven workstreams merged) |
| previous sweep this is compared against | `275970cc2d9e0d92eebf5297ee175b08944b7d04`, `examples/results/step7/closure/` |
| branch | `step7/closure2` |
| allocation | SLURM `6443480` (`se7-interactive-3090-b`), **gpu2264**, NVIDIA GeForce RTX 3090, driver 575.57.08, AMD EPYC 7532 |
| how | one stage at a time through `srun --jobid=6443480 --overlap` from the login node. Per Sami's rule of 2026-09-16, **nothing computes on a login node**: `srun` is a client that forwards I/O, and every render, every check and every pytest ran on gpu2264 |

| stage | what | log |
|---|---|---|
| `stage1` | the 49 frozen Step-4 configurations replayed on cuda, each shard in its own frozen interpreter, plus the 3.11 `dmcvis` arm, `scratch/step4/certify.py`, the field-by-field diff, and CraftaxClassic's world-row obligation | `closure2_stage1` |
| `stage2_gatea` | Gate A on the seven Step-7 cells on cuda and on cpu, the by-design cpu refusals asked ON THE CARD, and the per-interpreter declaration dumps | `closure2_stage2_gatea` |
| `stage3_gputests` | `tests/gpu` (`test_conformance`, `test_reproducibility`, `test_fast_paths`) in the `se6-verify` interpreter | `closure2_stage3_gputests` |
| `stage4_refreeze` | the six Mujoco locomotion nvdiff cells against the Step-7 re-freeze artifact, their Gate A, and the contiguous CPU width sweep | `closure2_stage4_refreeze` |

Logs live under `$SCR/logs/` and are not committed: they carry absolute interpreter paths, and
`tests/unit/test_repo_contracts.py` reads every `.txt` under `examples/`.

**Not re-run here, on the coordinator's instruction**: the OGB-interpreter `tests/gpu` suite (W5's
gate on this same tree is committed under `examples/results/step7/ogb-manipulation/m6/`), and the
`se7-rocket` / `se7-craftax` arms. `PFRocketLanding`'s two known `tests/gpu` failures therefore do
not appear in this sweep's files at all — it is not registered in `se6-verify` — and are neither
confirmed nor contradicted here.

The six `refreeze/branch_Mujoco*.npz` frame dumps the render phase writes are **not** committed:
they are ~8 MB of regenerable frames, and the sha256 of each is in `refreeze/branch.json`, which is
what the comparison actually reads.
