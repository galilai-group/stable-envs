# Historical research artifacts

These files are preserved historical/research source snapshots. Relative paths and import
assumptions may reflect the repository layout at the time they were produced. They are not
maintained entry points; use the corresponding historical git revision for exact reproduction.

The public `examples/` tree is for using stable-envs as stable-envs.

| path | what it is |
|---|---|
| [`bench_step3.py`](bench_step3.py) / [`bench_step3_pipeline.py`](bench_step3_pipeline.py) | frozen throughput / pipeline harness |
| [`fidelity_step4.py`](fidelity_step4.py) | frozen-corpus fidelity/conformance measurement |
| [`paper/make_tables.py`](paper/make_tables.py) | paper tables and figure generators |
| [`results/bench_gpu_vs_cpu.md`](results/bench_gpu_vs_cpu.md) | historical result dumps from those harnesses |
| [`step5_lewm_pusht/`](step5_lewm_pusht/README.md) | PushT + LeWM: official vs stable-envs pixels under the same planner |
| [`step5_downstream/`](step5_downstream/README.md) | Step-5 ingest / paired-gate / Reacher arm on top of that evidence |

Do not delete them. They are certification history. Do not edit them to make them runnable from
this location. New user examples belong under `examples/`.

Replacement-readiness of the 42 historical cells: [step9_replacement_readiness.md](../step9_replacement_readiness.md).
