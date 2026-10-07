# Cube default-renderer candidate — historical pending checkpoint

The follow-up [acceptance record](../testing.md#cube-default-cuda-renderer-acceptance)
closes the gates below. This note and its pending evidence describe the earlier
checkpoint; the original JSON, archive and fixtures remain unchanged.

This candidate starts from `1b1422da9fe74b9dece405d278a7b867ac8d4c79`.
At that checkpoint it was **not an accepted backend promotion**. The three
immutable A40 golden-image checks and terminal scheduler verification were
still outstanding. Oscar rejected
fresh SSH authentication before their submission; the control socket was absent.

Cube's CUDA default selects the existing two-pass fidelity renderer and defaults
to 2× supersampling. Explicit sampling settings, including those in supplied
RenderStates, remain authoritative. CPU rendering remains software OpenGL.
Configuration grouping now operates on columns, dependency versions are checked
once, and the fidelity pass avoids an unused background composition. The shared
raster operation still supplies both the fast and fidelity paths.

On 32 independently reached States at 64×64, normalized MSE changed from
**0.002059 ± 0.000129** to **0.0006046 ± 0.0000364**. Worst-frame error changed
from **0.002323** to **0.000691**. This meets 0.001 on that corpus, not 0.0005.

| Batch | Candidate GPU FPS | Source GPU FPS | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 144 | 1,948 | 0.074× |
| 64 | 4,105 | 1,791 | 2.29× |
| 256 | 5,812 | 1,787 | 3.25× |

These paired RTX 3090 measurements include public rendering and synchronization,
with three warmups and six timed samples. Disk loading, rollout, State projection,
initial transfer, and indexing are excluded. Larger batches repeat the 32 States.
Source OpenGL renders independently replayed, ready source instances serially and
returns NumPy images, including its readback; ours returns CUDA images. This is
neither an online benchmark nor the requested full 38-registration advisor table.
The CSV preserves latency and FPS dispersion. At B256, the candidate is 1.60×
faster than the previous 2× fidelity implementation, but 22% slower than the old,
less accurate default. B1 remains slower than the source.

Validation completed so far:

- OGBench source profiles: 72/72 CPU and 72/72 CUDA, including all eight Cube
  source-pixel cases per device. Cube worst CPU MSE is 0.00007766.
- The CUDA bundle then failed to collect a separate regression stage because its
  exported helper was missing. Standalone source job `7093278` was submitted for
  a clean terminal record; collect its results before acceptance.
- A corrected RTX 3090 regression bundle passed the Cube batch/chunk/style/
  nonmutation check, including 257 rows at default resolution. Three A40-specific
  golden tests skipped. **Those skips are not passes.**
- The paired 3090 benchmark preserved exact images for explicit Cube `mjwarp`,
  `mjwarp_fidelity` at 1×, and `mjwarp_fidelity` at 2× on its 32-State corpus.
  This does not replace the broader A40 fixtures.
- 28 local unit checks, Ruff, strict documentation build, distribution audit and
  an installed-wheel PushT smoke outside the checkout passed. The smoke reused
  installed dependencies; it is not a complete clean-install family gate.

Scene's isolated diagnostic does not change production code. Worst MSE was
0.00763 for its default, 0.00163 at fidelity 2×, 0.00125 at 3× and 0.00116 at 4×.
B256 throughput fell from 4,136 FPS at 2× to 1,656 at 4×. At 4×, the rightmost
14 columns, containing the window, contribute 49% of total squared error across
the corpus. Investigate that composition before paying for more sampling.

Evidence lives under `tests/fixtures/testing/cube-default-candidate-*`.
The archive contains source snapshots, raw numeric frames/States, test reports,
timing samples, diagnostic failures and reproduction scripts. Its validation
report explicitly records pending acceptance. Historical accepted reports and
golden fixtures are unchanged.
