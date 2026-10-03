# First completed same-device DMC online source comparison

Registration `swm/CartpoleDMControl-v0`. Task 6976162_2 (raw job 6976162)
completed on gpu2114 with exit 0. All six fresh strategy/device reports pass the
strict collector and match the same host, eight-core allocation, versions,
source hashes, candidate hashes, initial State and action policy. Both archived
CSVs reproduce byte-for-byte from the frozen bundle. Production is unchanged.

At B64, eight actions, all frames, 64px output:

| Device | Source serial FPS | Source P8 FPS | Ours FPS | Ours / best source | MSE mean ± SD | Maximum frame MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 49.25 ± 0.30 | 312.20 ± 12.46 | 1971.68 ± 6.09 | 6.315× | 0.00010073 ± 0.00003005 | 0.00018867 |
| RTX 3090 | 414.40 ± 2.14 | 2847.44 ± 294.55 | 6175.45 ± 147.61 | 2.169× | 0.00008026 ± 0.00003854 | 0.00022355 |

FPS measures complete online stepping and selected rendering, not rendering
alone. Values are means and sample SDs across six timing samples; image errors
are normalized RGB MSE across 512 matched frames for each displayed cell. Source
GPU rendering is verified NVIDIA OpenGL on the same allocation. Both paths
retain CPU physics. Source timings include observation construction, selected
rendering, readback and output delivery. Our public path additionally returns
mandatory full final State; intermediate State histories are disabled.
Source P8 is a persistent eight-process source baseline, not a native vector API.

[comparison.csv](tables/comparison.csv) retains all 12 cells (B64, T1/8,
no/final/all frames, CPU/CUDA), including slower ratios. Separate component
measurements are in [components.csv](tables/components.csv). These bounded
results do not prove sustained large-batch performance or universal acceptance.
No semantic matrix status changed.

The [updated 42-registration coverage CSV](coverage/coverage.csv) adds this
result to the preceding online-resume checkpoint without altering its accepted
artifacts. Eight registrations now have bounded CPU/CUDA rendered comparisons;
three PickAndPlace registrations have no-frame comparisons only. Seven rendered
comparisons use source-CPU-plus-CUDA-delivery fallback because those sources lack
GPU renderers; this DMC registration has an actual same-device source-GPU row.
Other statuses are the preceding checkpoint's snapshot and require fresh
collection. Reproduce with `reconcile.py --evidence <this directory> --previous
<online_resume_20261003/coverage/coverage.csv> --output <new directory>`.

Strict collection: run `scripts/benchmarks/online_stepping/summarize_dmc_source.py`
with `results/`, a new output directory, `--batches 64`, the frozen candidate tree
from `dmc-source-comparison-v1-20261003/code`, and its pinned source root/manifest.
The prepared source/candidate manifests and launcher resolve in the earlier
`dmc_source_pipeline_20261003` checkpoint. The unchanged upstream pin is
`176d54532b3c0a251a4f509b74f2fdacc143a63d`.
