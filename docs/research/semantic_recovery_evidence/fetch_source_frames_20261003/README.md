# Fetch selected source frames — CPU prerequisites complete, GL gate pending

Base `c8f42ed408ced3536a2cb5ce8e6e40c89cde7c4c`. Changes are benchmark helpers,
checks and documentation. Production adapters, renderers, State contract,
routing and slice-only public API are unchanged. Semantic status remains
39 VERIFIED / 3 INCOMPLETE. This checkpoint establishes neither GPU acceptance
nor a source speedup.

## Executed evidence

- `no-frame-final.json`: 96 passing real CPU B4/P2 cases, all 12 registrations,
  T1/T8, float32/64 and ordinary/mixed time limits. This reruns the existing
  independent serial-source / shared-source / accepted adapter-dispatcher gate
  against the final helper code.
- `empty-frame-final.json`: the same 96 cases with zero-length shared image
  columns and `frames=slice(20,30)`. Full State/tasks, restored replay, unseeded
  reset RNG, output ownership, active-lane validity, terminal State holds and
  continuing without autoreset all pass. No actual image is rendered.
- Both runs close 24 pools and 48 workers each, with exit code zero. Every
  recorded candidate input hash matches this checkpoint. Source pin and hashes
  are recorded separately. The local profile is Robotics 1.4.2 with isolated
  MuJoCo 3.11.0; prior dependency provenance is in `fetch_source_prerequisite_20261003`.
- `synthetic-final` and `frozen`: 36 tests pass, including frame selection order,
  terminal image holds, owned outputs, empty/None distinction, invalid inputs,
  PickAndPlace refusal before mutation/worker construction, image shape/dtype,
  exact source-parallel comparison, per-frame MSE rejection even when aggregate
  MSE is low, T1 task-axis normalization and invalid launcher options.
  Synthetic frames are not source-image fidelity evidence.

Only projected task/frame outputs use shared buffers. Independent full source
snapshots/restoration remain untimed pickled preparation. The source still
executes its original observation/info work. `copy=True` ensures owned parent
outputs. No source implementation has been patched.

## Prepared, not executed: nine independent image gates

`fetch_source_frame_gate.py` is Oscar-only and runs each Reach/Push/Slide plain,
Dense and Dict registration independently. The launcher uses nine array tasks,
three concurrent, eight physical CPU cores and one RTX 3090 per task, with fresh
CPU/software-GL and CUDA/NVIDIA-GL interpreters. Its historical Fetch environment
is `se6-verify-fetch311`; the gate requires MuJoCo 3.11.0 at runtime and records
installed dependency versions/source hashes. Source package files are the actual
five pinned registration/factory/wrapper dependencies, loaded without unrelated
policy/data top-level imports.

Each device runs B4, source P1/P2, T1/T8, None/empty/final/all/reverse-sparse frame
selections, and ordinary/mixed episode limits: 20 cases per device/registration.
Source owners render in their own processes; a separate parent source supplies
an independent no-render physics witness. The public library renderer is in the
parent. Source GL teardown therefore cannot invalidate a live library GL context.
This isolation also means the P1 source is a correctness witness over IPC, **not
a measured native serial baseline**. No timing is reported by this gate.

Checks require exact source P1/P2 image/task/full continuation State agreement,
source render nonmutation, library task/full State agreement with the independent
physics witness, held invalid State/frames, owned outputs across a following
call, clean worker exits and per-frame normalized RGB MSE <= 0.001. MSE is mean
squared `(uint8_candidate - uint8_source) / 255`; mean, sample SD and maximum are
recorded, but the limit applies to every checked frame. Source rendering retains
480x480 internal rasterization followed by the source wrapper's own resize to
64x64. No silent source width/height override or substitute resize is used.

All three PickAndPlace registrations refuse nonempty frame selections before
stepping or constructing source workers. Earlier source-only evidence shows
that their render callback changes continuation State and subsequent behavior.
Their no-frame and empty-frame paths remain supported. This is an unresolved
rendered benchmark protocol, not a semantic demotion or permission to repair the
source silently. Actual GL nonmutation for the other nine is still unproven
until these jobs pass.

## Reproduction and remaining work

`candidate_manifest.json` freezes 258 candidate files; `source_manifest.json`
freezes five upstream files at
`176d54532b3c0a251a4f509b74f2fdacc143a63d`. `prepare.py` validates every hash and
creates a new archive without modifying the checkout or source:

```bash
python3 docs/research/semantic_recovery_evidence/fetch_source_frames_20261003/prepare.py \
  --checkout . --source /path/to/pinned/stable-worldmodel --output /tmp/new-fetch-frame-bundle
```

The frozen local bundle is `/private/tmp/fetch-source-frames-bundle-v1-20261003`.
`bundle-sha256.json` identifies the prepared archive; archive rebuilding can vary
tar timestamps while retaining identical file manifests. `deployment.json`
explicitly records NOT_UPLOADED_NOT_SUBMITTED. Oscar SSH rejects keyboard-interactive
authentication; no job is claimed live and no previous job was restarted.

After authentication returns, inspect the existing Oscar workflow, upload the
frozen bundle to its new scratch directory, verify the archive/file hashes and
submit `run.sh`. Record the job IDs, terminal accounting, logs, device identities
and each registration's reports before accepting anything. Passing this B4 gate
will still leave actual production-process B64+ checks and resource-matched
source P1/P8 component/online throughput and memory sweeps. PickAndPlace rendered
lifecycle accounting remains a separate open item. Prepared DMC/Navigation/OGBench
campaigns and previously submitted jobs also remain to be executed or collected;
this checkpoint does not close the 35-registration online goal.
