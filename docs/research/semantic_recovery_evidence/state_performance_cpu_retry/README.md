# Bounded CPU profile retries

The initial normal profile lacked OGBench and the lazy Matplotlib dependency
needed for SimplePointMaze's exact cached `info.goal`. Both original errors remain
in `state_performance_cpu`. No production package or code was changed.

OGBMaze uses the already accepted MuJoCo 3.12 / OGBench 1.2.1 environment.
SimplePointMaze uses normal Python plus the existing `native-nav-py312-deps`
directory, including Matplotlib 3.10.9, matching its CPU acceptance profile.
`navigation-python` records that small launcher. The dispatcher runs with a clean
base PYTHONPATH so the navigation dependencies do not override OGBench's profile.

Exact invocation (use a fresh output path):

```sh
PYTHONPATH=.:/Users/samib/stable-worldmodel SWM_ROOT=/Users/samib/stable-worldmodel SDL_VIDEODRIVER=dummy MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu CRAFTAX_JAX_BACKEND=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/benchmarks/run_state_matrix.py --only 'OGBMaze|SimplePointMaze' --output docs/research/semantic_recovery_evidence/state_performance_cpu_retry --profile normal=/private/tmp/state-bench-navigation-python --profile ogbench=/private/tmp/ogb-render-unified-312/bin/python
```

Both registrations complete at B1 and B16. Child commands, exit codes, samples,
versions, State sizes and unsupported CPU render reasons are retained separately.
GPU rendering is covered by the original correct normal-profile GPU run.
