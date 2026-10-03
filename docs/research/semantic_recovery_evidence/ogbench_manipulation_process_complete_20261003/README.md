# Cube/Scene process correctness and bounded throughput complete

Both tasks in array 6966557 completed on RTX 3090 hosts with eight allocated
physical CPUs. Four independent CPU/software-GL and CUDA/EGL gates pass 22
cases each, including exact State/task/frame/RNG/lifecycle outputs, variations,
owned snapshots, observed rendering overlap and rejected-restore recovery.
The strict summarizer verifies complete source hashes and all 32 paired cells:
B16/64, T1/8, P4/P8 against local execution, six samples per strategy.

B64/T8, all States retained and frames disabled:

| Registration | Output device | Local steps/s ± SD | P8 steps/s ± SD | Ratio |
| --- | --- | ---: | ---: | ---: |
| OGBCube | cpu | 320 ± 3 | 1825 ± 35 | 5.70x |
| OGBCube | cuda | 318 ± 1 | 1676 ± 32 | 5.27x |
| OGBScene | cpu | 320 ± 1 | 1964 ± 54 | 6.14x |
| OGBScene | cuda | 314 ± 1 | 1823 ± 49 | 5.80x |

Physics is CPU-side for both output devices. These are public adapter-step
comparisons, not render FPS or upstream speedups. Construction, warmup and
restoration are excluded from timing; repeated source-world rebuilds make the
whole gate much longer than its measured transition samples.

The frozen source is recorded under ../ogbench_manipulation_process_retry_20261003/v2/.
This is the unchanged V1 production candidate with only the benchmark selecting
explicit OpenGL, after V1 failed by requesting a CUDA mesh renderer on CPU.
Failed V1 evidence and its passing process tests remain preserved in the retry
folder. The CPU automatic-renderer fix subsequently passed its own independent
gates under ../ogbench_auto_render_20261003/; it is not retroactively included in
this earlier process sweep. Default process selection is NOT enabled here.
It requires unpatched constructor gates against the integrated current source,
including the actual default renderer. Larger batches, component/source baselines
and full pipeline throughput also remain required.

Reproduce comparison.csv with summarize_ogbench_manipulation_process.py results
NEW_OUTPUT --source FROZEN_CHECKOUT. Accepted semantic status is unchanged.
