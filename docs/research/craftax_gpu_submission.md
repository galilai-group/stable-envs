# Craftax GPU submission checkpoint

Craftax GPU job **6679082** was submitted successfully to Oscar's `gpu` partition
with `norm-gpu` QoS and last observed **RUNNING on gpu2110**. Fresh SSH authentication
then failed with `Permission denied (keyboard-interactive)`. No terminal result has
been retrieved. All four registrations remain independently **INCOMPLETE**; the matrix
remains **35 VERIFIED / 7 INCOMPLETE**. No duplicate job or push occurred.

This report supersedes the submission-status statements in the sealed
[CPU checkpoint](craftax_recovery_checkpoint.md). Those historical artifacts and the
prepared bundle metadata remain unchanged.

## Checkpoint and submitted source

- Starting HEAD: `28e23907f887c74edb56471f4011bbb819cdda5e`, clean before submission work.
- Branch: `repair/environment-semantics`.
- Worktree: `/private/tmp/stable-envs-semantic-recovery`.
- Cube/Scene promotion/report commit: `bcf8699d9983f95e97b3c71607521a61c354247b`.
- Submitted source commit: `b6a8242f5ff5a9cae5e9e883a15611d779c2c99c`.
- Source archive SHA256: `49235ccf34ee8ba947addbf866225f1fd87ddbefa4bec68b60bc7eeb9a206834`.
- Oracle archive SHA256: `c2253051f8ab584b9702e12c9fcc919f6cb8372cd0dbd1c85e270a05fc9d6b9a`.
- The containing commit and complete changed-file inventory are available through
  this report's Git history and the execution evidence inventory.

The six current Oscar skills were read before constructing cluster commands.
Preflight confirmed uv 0.6.11 supports the runner's `--torch-backend cu126` and
`--require-hashes` options. The Python base interpreter is 3.12.8. All four uploaded
bundle files passed remote SHA256 verification before submission and again at job
startup. The existing unrelated SimCLR jobs were untouched.

## Independent registration gates

| Registration | Existing CPU tests / exact array cases | GPU acceptance |
|---|---:|---|
| `swm/CraftaxClassicPixels-v1` | 28 / 11 | Pending terminal evidence |
| `swm/CraftaxClassicSymbolic-v1` | 17 / 11 | Pending terminal evidence |
| `swm/CraftaxPixels-v1` | 28 / 11 | Pending terminal evidence |
| `swm/CraftaxSymbolic-v1` | 19 / 11 | Pending terminal evidence |

The runner requires every registration's exact test count, zero failures/errors/skips,
its own source/candidate array gate and its own terminal marker. No sibling's result
substitutes for those gates. CPU/GPU equality remains unclaimed. Performance results
must come from the actual per-registration GPU profiles and result files.

The new `scripts/recovery/verify_craftax_gpu.py` verifies downloaded manifests, artifact
inventory, exact arrays, schemas, pinned profiles, pytest counts and terminal evidence.
It refuses Python's optimized mode, which disables assertions. Local checks exercise
valid hashes and rejection of corrupt payloads, path traversal and missing GPU markers.
These checks do not constitute GPU acceptance; the verifier has not yet inspected
actual GPU results.

## Evidence and preservation

[Execution evidence](semantic_recovery_evidence/craftax_gpu_execution/submission.json)
records the submission, uploaded hashes, initial scheduler state and authentication
failure. The directory also contains the retrieved skills, preflight/install-tool
capability checks, local validation logs, changed-file inventory and a SHA256 manifest.

[Preservation revalidation](semantic_recovery_evidence/craftax_gpu_execution/preservation.json)
passed: all 35 accepted rows, all 38 non-Craftax rows, 2,455 original files,
170 prior artifacts, 11 terminal logs, both Cube/Scene acceptance chains and the
recorded upstream source files remain intact. The sealed Craftax CPU manifest is
rechecked with mutable matrix/map entries resolved at the starting HEAD.

Only Craftax submission bookkeeping, an offline verifier and new evidence are changed.
No production implementation, shared contract, upstream game logic, PushT or Rocket
code changed. Production still does not import stable-worldmodel. No shared-contract
change is requested. No push occurred.

## Resume the existing job

Restore authenticated SSH access and leave the control connection open:

```sh
ssh -M -S /private/tmp/stable-envs-oscar-control sboughan@sshcampus.ccv.brown.edu
```

Remote bundle directory:
`/oscar/scratch/sboughan/se-semantic-recovery/stable-envs-craftax-b6a8242f-gpu`.
Query job **6679082** before considering another submission. Retrieve its
`slurm-6679082.log`, terminal `sacct` record and
`craftax-evidence-6679082.tar.gz`; verify remote/local archive hashes and safely
extract the evidence. If it failed before packaging, collect the failure/install
logs from `run-6679082/evidence` first. Preserve this submitted bundle unchanged.

Run the offline verifier against the actual download:

```sh
/private/tmp/craftax-recovery-312/bin/python scripts/recovery/verify_craftax_gpu.py \
  --evidence /path/to/download/evidence \
  --slurm-log /path/to/download/slurm-6679082.log \
  --sacct /path/to/download/sacct.txt --job 6679082 \
  --output /path/to/new-verification.json
```

Use `sacct --format=JobIDRaw,State,ExitCode,Elapsed,NodeList --parsable2` for its input.
Review every registration's evidence before promotion, then revalidate preservation,
shared regression, documentation and the complete diff. The intended all-pass result
is 39 VERIFIED / 3 INCOMPLETE; until then the incomplete set is all four Craftax
registrations plus `swm/PushT-v1`, `swm/PushT-Discrete-v1` and
`swm/PFRocketLanding-v0`.
