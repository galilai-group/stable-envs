# The four `cpu_1thread_*.json` and two `gpu_*.json` shards here predate the per-leg SHA fix

`git_commit` in all six reads `d862e2ac1427`, the AFTER tree's commit, because
`STABLE_ENVS_GIT_COMMIT` was exported once for the job rather than once per leg. The field is wrong
on the three `*_before_*` shards and on `gpu_before.json`.

**What is right, and is what the reader should use:** `provenance.stable_envs_file`. The before legs
carry `/oscar/scratch/sboughan/se6/raster2d-preB/stable_envs/__init__.py` and the after legs
`/oscar/scratch/sboughan/se6/raster2d/stable_envs/__init__.py`. That field is written from the
import that actually resolved, so it cannot be wrong in the way the SHA was: it is the tree the
measurement ran out of. `raster2d-preB` was built by overwriting exactly two files —
`stable_envs/render/raster2d.py` and `stable_envs/envs/swm_nav.py` — with their `1103326` contents,
and its own `GIT_COMMIT` file reads `1103326`.

Fixed in `cpu_sweep.sbatch` (commit `62997b4`) and `gpu_check.sbatch` (this round). These shards are
annotated rather than rewritten: an artifact is a record of a run, and editing the record is worse
than recording that it was wrong.

`sweep_r3/` is the re-run with the fix in place; read that one in preference for the GPU numbers,
which this round's change also alters.
