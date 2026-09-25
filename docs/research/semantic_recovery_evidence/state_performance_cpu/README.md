# CPU State-operation measurements

Each of the 39 accepted registrations is a separate child process. B1 + B16,
except B1 + B4 for Cube/Scene; B64 deferred for runtime. No scalar batch emulation.
Raw child commands and exit codes are in `runs.json`, operations in each
`*/result.json`, stdout/stderr in the matching `.log`.

Exact dispatcher invocation from `/private/tmp/stable-envs-semantic-recovery`:

```sh
PYTHONPATH=.:/Users/samib/stable-worldmodel SWM_ROOT=/Users/samib/stable-worldmodel SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=disable JAX_PLATFORMS=cpu CRAFTAX_JAX_BACKEND=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/benchmarks/run_state_matrix.py --output docs/research/semantic_recovery_evidence/state_performance_cpu --profile normal=/private/tmp/stable-envs-semantic-recovery/.venv/bin/python --profile fetch=/private/tmp/fetch-authority-swm-lock/bin/python --profile manipulation=/private/tmp/ogb-render-unified-312/bin/python --profile craftax=/private/tmp/craftax-recovery-312/bin/python
```

Use a **new output path** for reproduction. Do not rerun over this evidence.
SWM pin: `176d54532b3c0a251a4f509b74f2fdacc143a63d`. Benchmark source initially
`8d153702`; later child records name subsequent documentation/research commits,
with the same production and benchmark code. The checkpoint preservation record
verifies production immutability.

The dispatcher was stopped between Craftax and Fetch while native prototypes
were compiled/measured. Full CraftaxSymbolic finished at 09:44:56 local file time;
the first native build log was created at 09:45:57. It resumed after native
measurements. Dispatcher elapsed time can include this pause; individual timed
operation samples do not. File timestamps are operational context, not a semantic
gate. This is descriptive timing on the development Mac, not a claim of an
otherwise idle operating system. Raw variance and outliers remain visible.

CPU rendering refusals are recorded; required GPU rendering is in the separate
Oscar run. Pixel Craftax oracle step includes intrinsic pixel observations;
there is no extra render call inside the dynamics timer. Native comparisons are
partial and never asserted exhaustive-equivalent.
