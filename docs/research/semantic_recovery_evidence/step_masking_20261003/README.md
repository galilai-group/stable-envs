# Activity-aware output padding

Parent `c39880f92f436adda94050535e06c4ef240cdccf`. The frozen reference dispatcher
is a byte-for-byte copy of that commit's `stable_envs/_stepping.py`; provenance
is recorded in `reference-provenance.json`. Only execution-mask reuse and
skipping all-active padding change. Output cloning and mixed/inactive padding
remain required. See [the report](../../step_masking_checkpoint.md).

`local/` contains independently completed CPU public gates and paired timings
for TwoRoom, Piecewise and SimplePointMaze, plus the shared contract test JUnit
report (226 passed, 76 CUDA skips). The 54-row CSV requires each native
registration's 16 exact State/task/frame cases and all 36 timing cells, six
samples per cell. Local runs used the Mac CPU; they are not cluster or upstream
speedups. Paired controls alternate order; restore and warmup are excluded.

Oscar bundle `step-masking-v1-20261003` is source-hashed under `oscar/`. Rebuild
from committed paths unless `source_overrides.json` supplies a path. The archive
includes the reference-helper fixture; all 302 native/shared tests collect from
the isolated archive. Array `6964504` runs independent CPU/CUDA native gates,
paired measurements, and all 302 shared/native tests. Array `6964505` runs all
twelve public Fetch pipeline regression gates on CPU/CUDA and the Fetch adapter
unit suite, using its accepted separate interpreter. Both request RTX 3090.
These GPU results are pending, not inferred from CPU results.

Reproduce the local summary into a new output path:

```sh
python3 scripts/benchmarks/online_stepping/summarize_step_masking.py \
  docs/research/semantic_recovery_evidence/step_masking_20261003/local \
  /tmp/masking-cpu.csv --source . --devices cpu
```

The default summary requires both devices, all three registrations, exact gates,
complete cells, matching source hashes, and positive finite timings. Each
T8/final-frame call renders one image per lane; steps/s and frames/s remain
separate. No source payload, State field, renderer setting or public API changes.
