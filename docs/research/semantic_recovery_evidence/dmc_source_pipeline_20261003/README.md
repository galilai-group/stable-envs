# DMC same-device source pipeline campaign — prepared, not submitted

This freezes the first bounded full-render comparison for Acrobot, Pendulum and
Cartpole DMC. No GPU or full-render result is claimed here. Oscar SSH authentication
was unavailable when this bundle was prepared. Production and acceptance statuses
are unchanged; the previous [48-case CPU source-transport prerequisite](../dmc_source_transport_20261003/)
remains a separate, narrower result.

The candidate is based on `6baf0fbde34de43acfaf844012adb078b0453aec`; exact file
hashes are in `candidate_manifest.json`. The upstream is stable-worldmodel
`176d54532b3c0a251a4f509b74f2fdacc143a63d`; `source_manifest.json` records its
20 copied files. Source code and production code are unmodified. New code is
benchmark orchestration, measurement and report validation; an existing benchmark
import is made package-qualified so it also works from pytest.

## Measurement design

- One RTX 3090 and eight physical CPU cores per registration, B64, T1/T8,
  no/final/all frames, 64x64 images, six timed samples after warmup.
- Serial upstream wrappers, eight persistent upstream worker processes with
  shared observation buffers, and the ordinary library public API are each run
  in fresh interpreters on the same allocation. Their order rotates by domain.
- CPU uses verified software GL; GPU uses verified NVIDIA GL. Source physics
  remains CPU in both cases. Source rendering genuinely uses the GPU for the GPU
  comparison; source image readback and delivery of outputs to CUDA remain timed.
- The independent source oracle is closed before the measured population is
  constructed. No other strategy's full renderer population is deliberately
  retained during timing. Actual GPU memory behavior still needs cluster evidence.
- Each sample starts from the same seeded complete State. Restoration, oracle
  State export and output comparison are outside timing. Library timing includes
  its required final State; intermediate State history is disabled (`states=None`).
  Actions use seed 31 and resets use seed 17 plus lane. This initial bounded
  workload is not a diverse-state, 1,000-frame renderer certification.
- Full endpoint State, accepted numeric task outputs, validity and ownership are
  checked. Parallel source images must exactly match serial source. Library image
  MSE is normalized by 255 squared and every checked frame must be <=0.001.
- Separate library components measure task stepping with IPC/output transfer but
  without State capture, full `get_state`, and rendering an already prepared State.
  These components are not assumed additive or equivalent to bare physics timing.

The strict join requires all six strategy/device reports, matching host/job/core
allocation/dependency versions/file hashes, distinct process IDs, real source GL
identities, all expected cases and six finite positive timing samples. It checks
source trajectory/image hashes across strategies on the same device. Output
`comparison.csv` includes mean/SD steps/s and emitted frames/s for each strategy,
normalized MSE mean/SD/max, and ratios to serial, parallel and best measured source.
`components.csv` separately reports the component means/SDs and explicit units.
No-frame cells have no FPS. These are online rates, not renderer-only FPS.

## Local validation and reproduction

`local-summary-tests.xml` contains **15 passing tests** from the frozen bundle.
These tests exercise synthetic valid and invalid reports, task normalization,
NaN equality, image mismatch rejection and coordinator argument validation. They
do not run MuJoCo rendering or establish performance. Ruff, launcher shell syntax,
frozen-source hashes and CLI imports also pass; see `local-review.json`.

From a checkout matching the candidate manifest, with the pinned upstream source:

```bash
python3 docs/research/semantic_recovery_evidence/dmc_source_pipeline_20261003/prepare.py \
  --checkout . --source /path/to/stable-worldmodel --output /tmp/new-dmc-source-bundle
```

The preparer verifies every input and refuses to overwrite the output. It creates
`code/`, `source/`, both manifests, `run.sh`, and `bundle.tar.gz`. Archive metadata
can vary on rebuild; per-file hashes define the candidate. `deployment.json`
records the original prepared archive hash. The launcher verifies both manifests
and the existing isolated renderer extras without modifying the accepted venv.
Its fixed remote path is `/oscar/scratch/sboughan/se-semantic-recovery/dmc-source-comparison-v1-20261003`.
Upload into a new directory before submitting `sbatch --parsable run.sh`; retain
the returned job ID, raw reports/logs, generated CSVs and terminal accounting.

## Remaining acceptance

This campaign must actually run on Oscar and each registration must independently
pass. Larger batches, the remaining DMC registrations, sustained workload coverage
and resource utilization follow the bounded check; a prepared launcher proves
none of them. Quadruped's renderer-history issue remains open and its production
process default remains disabled. No accepted evidence has been regenerated or
overwritten, and no optimization is promoted by this checkpoint.
