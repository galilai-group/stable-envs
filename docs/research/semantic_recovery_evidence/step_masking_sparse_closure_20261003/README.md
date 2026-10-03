# Sparse padding: native CPU/CUDA and Fetch regression closure

Arrays 6966161 and 6966162 completed on RTX 3090. All six independent native
registration/device gates pass their 16 exact public State/task/frame cases.
The source-hashed 108-row paired CSV covers B1/16/64, T1/8, all/no State history
and final-frame pipeline, six samples each. Actual valid transitions are counted.
302 shared/native tests and 38 Fetch adapter tests pass with no skips or failures.

All twelve Fetch registrations independently pass CPU and CUDA State/task/replay/
ownership checks, 13 cases per device. CUDA frames exactly match serial execution.
CPU Fetch frames were not exercised by this regression harness; their MSE cells
remain empty. fetch-gates.csv preserves all 24 gates and fetch-validation.json
records explicit scope and verified script/dispatcher/adapter hashes.

At B64, all States retained and frames disabled, SimplePointMaze T8 improves
1,164 to 2,537 valid steps/s on CPU (2.18x), and 23,307 to 24,867 on CUDA (1.07x).
T1 improves 1,447 to 4,457 CPU (3.08x), 19,320 to 20,160 CUDA (1.04x). Full means/SDs
and every workload are in comparison.csv. The earlier Mac improvement is not
extrapolated to Oscar. Small-task differences must be read from their own rows.

Source is the immutable step-masking-v2-20261003 bundle recorded under
../step_masking_sparse_20261003/oscar/. Earlier V1 and local reports are unchanged.
MSE is against the frozen library dispatcher/serial pipeline, not a fresh source
renderer survey. DMC and OGBench default/integration gates remain separate.
